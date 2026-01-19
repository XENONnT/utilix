from __future__ import annotations

import os
import sqlite3
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import traceback
import logging
import pymongo
from bson import BSON

OFFLINE_DEBUG = os.environ.get("OFFLINE_DEBUG", "0") not in ("0", "", "false", "False")


def _env_bool(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return v not in ("0", "", "false", "False", "no", "No", "NO")


def _dbg(msg):
    if OFFLINE_DEBUG:
        logging.debug(f"[offline-debug] {msg}")


def _dbg_stack(tag, n=6):
    if OFFLINE_DEBUG:
        logging.debug(f"[offline-debug] --- stack ({tag}) ---")
        logging.debug("".join(traceback.format_stack(limit=n)))
        logging.debug(f"[offline-debug] --- end stack ({tag}) ---")


def block(msg: str, cfg: SQLiteConfig) -> None:
    if cfg.hard:
        raise RuntimeError(f"[offline-hard] blocked: {msg}")
    _dbg(f"WARNING: {msg}")
    _dbg_stack("blocked")


@dataclass(frozen=True)
class SQLiteConfig:
    rundb_sqlite_path: Optional[Path]
    xedocs_sqlite_path: Optional[Path]
    offline_root: Optional[Path]
    compression: str
    debug: bool
    hard: bool
    stack: bool
    spy: bool

    def rundb_active(self) -> bool:
        return self.rundb_sqlite_path is not None and self.rundb_sqlite_path.exists()

    def xedocs_active(self) -> bool:
        return self.xedocs_sqlite_path is not None and self.xedocs_sqlite_path.exists()

    def sqlite_active(self) -> bool:
        return self.rundb_active() and self.xedocs_active()


def _load_sqlite_config() -> SQLiteConfig:
    sqp = os.environ.get("RUNDB_SQLITE_PATH", "").strip()
    rundb_sqlite_path = Path(sqp).expanduser().resolve() if sqp else None

    xsp = os.environ.get("XEDOCS_SQLITE_PATH", "").strip()
    xedocs_sqlite_path = Path(xsp).expanduser().resolve() if xsp else None

    offline_root = (
        rundb_sqlite_path.parent if (rundb_sqlite_path and rundb_sqlite_path.exists()) else None
    )

    debug = _env_bool("OFFLINE_DEBUG")
    hard = _env_bool("OFFLINE_HARD")
    stack = _env_bool("OFFLINE_STACK")
    spy = _env_bool("PYMONGO_SPY")

    return SQLiteConfig(
        rundb_sqlite_path=rundb_sqlite_path,
        xedocs_sqlite_path=xedocs_sqlite_path,
        offline_root=offline_root,
        compression="zstd",
        debug=debug,
        hard=hard,
        stack=stack,
        spy=spy,
    )


@dataclass(frozen=True)
class GridFSRow:
    db_name: str
    file_id: str
    config_name: str
    md5: str
    length: int
    uploadDate: int
    blob_path: str


class OfflineGridFS:
    """Minimal offline replacement for utilix.mongo_storage.MongoDownloader / APIDownloader
    behavior:

    - query SQLite table gridfs_files by config_name
    - pick the latest by uploadDate
    - stage/copy the blob into a local cache folder named by md5
    - return the staged path

    """

    def __init__(
        self,
        sqlite_path: str | Path,
        offline_root: str | Path,
        cache_dirs: Tuple[str | Path, ...] = ("./resource_cache", "/tmp/straxen_resource_cache"),
        gridfs_db_name: str = "files",
    ):
        self.sqlite_path = Path(sqlite_path).resolve()
        self.offline_root = Path(offline_root).resolve()
        self.cache_dirs = tuple(Path(p) for p in cache_dirs)
        self.gridfs_db_name = gridfs_db_name

        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.row_factory = sqlite3.Row

    # -----------------
    # cache dir helpers
    # -----------------
    def _pick_cache_dir(self) -> Path:
        for d in self.cache_dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            if os.access(d, os.W_OK):
                return d
        raise PermissionError(f"Cannot write to any cache dir: {self.cache_dirs}")

    # -----------------
    # sqlite queries
    # -----------------
    def latest_by_config_name(self, config_name: str) -> Optional[GridFSRow]:
        row = self.conn.execute(
            """
            SELECT db_name, file_id, config_name, md5, length, uploadDate, blob_path
            FROM gridfs_files
            WHERE db_name = ? AND config_name = ?
            ORDER BY uploadDate DESC
            LIMIT 1""", (self.gridfs_db_name, config_name),
        ).fetchone()

        if row is None:
            return None

        # Some older entries might have NULL md5; that's not usable for caching-by-md5.
        md5 = row["md5"]
        if md5 is None:
            raise RuntimeError(
                f"Found GridFS entry for {config_name} but md5 is NULL in sqlite index"
            )

        return GridFSRow(
            db_name=row["db_name"],
            file_id=row["file_id"],
            config_name=row["config_name"],
            md5=str(md5),
            length=int(row["length"] or 0),
            uploadDate=int(row["uploadDate"] or 0),
            blob_path=str(row["blob_path"]),
        )

    # -----------------
    # public API
    # -----------------
    def download_single(
        self,
        config_name: str,
        human_readable_file_name: bool = False,
        write_to: Optional[str | Path] = None,
    ) -> str:
        """Return absolute path to a staged file.

        Default behavior matches utilix: store under md5 in a cache dir.

        """
        _dbg(f"OfflineGridFS.download_single('{config_name}') [SQLITE]")

        entry = self.latest_by_config_name(config_name)
        if entry is None:
            raise KeyError(f"Config '{config_name}' not found in offline gridfs_files index")

        blob_abs = (self.offline_root / entry.blob_path).resolve()
        if not blob_abs.exists():
            raise FileNotFoundError(f"Blob missing on disk: {blob_abs} (from sqlite blob_path)")

        target_dir = Path(write_to).resolve() if write_to else self._pick_cache_dir()
        target_dir.mkdir(parents=True, exist_ok=True)

        target_name = config_name if human_readable_file_name else entry.md5
        target_abs = (target_dir / target_name).resolve()

        # If already staged, trust it (fast path)
        if target_abs.exists():
            return str(target_abs)

        # Copy in a safe-ish way (atomic replace)
        tmp = target_abs.with_suffix(target_abs.suffix + ".tmp")
        shutil.copyfile(blob_abs, tmp)
        tmp.replace(target_abs)

        return str(target_abs)

    def list_files(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT config_name FROM gridfs_files WHERE db_name=? ORDER BY config_name",
            (self.gridfs_db_name,),
        ).fetchall()
        return [r["config_name"] for r in rows if r["config_name"] is not None]

    def close(self) -> None:
        self.conn.close()


def smoke_test(
    sqlite_path: str | Path,
    offline_root: str | Path,
    config_name: str,
) -> None:
    g = OfflineGridFS(sqlite_path=sqlite_path, offline_root=offline_root)
    p = g.download_single(config_name)
    print("[OK] staged:", p)
    g.close()


# ---- OFFLINE RUNDB COLLECTION (SQLite-backed) ----


def _decompressor(algo: str):
    if algo == "zstd":
        import zstandard as zstd  # type: ignore

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress
    elif algo == "zlib":
        import zlib

        return zlib.decompress
    else:
        raise ValueError(f"Unknown compression algo: {algo}")


class OfflineMongoClient:
    """Dummy client to satisfy: collection.database.client."""

    def close(self):
        return


@dataclass
class OfflineMongoDatabase:
    name: str
    client: OfflineMongoClient


class OfflineSQLiteCollection:
    """Minimal pymongo.collection.Collection-like wrapper backed by our sqlite cache.

    Provides the attribute chain expected by straxen.storage.rundb.RunDB:
        collection.database.client
    And a few commonly-used methods: find_one, find, count_documents.

    """

    def __init__(
        self,
        sqlite_path: str | Path,
        db_name: str,
        coll_name: str,
        compression: str = "zstd",
    ):
        self.sqlite_path = Path(sqlite_path).resolve()
        self.db_name = str(db_name)
        self.name = str(coll_name)  # pymongo Collection has .name
        self._coll_name = str(coll_name)

        self._conn = sqlite3.connect(str(self.sqlite_path))
        self._conn.row_factory = sqlite3.Row
        self._decompress = _decompressor(compression)

        # mimic pymongo: collection.database.client
        self.database = OfflineMongoDatabase(name=self.db_name, client=OfflineMongoClient())

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # --- internal helpers ---

    def _decode_row(self, row) -> dict:
        raw = self._decompress(row["doc_bson_z"])
        return BSON(raw).decode()

    def _get_by_id(self, doc_id: str) -> dict:
        row = self._conn.execute(
            "SELECT doc_bson_z FROM kv_collections WHERE db_name=? AND coll_name=? AND doc_id=?",
            (self.db_name, self._coll_name, str(doc_id)),
        ).fetchone()
        if row is None:
            raise KeyError(f"Not found: {self.db_name}.{self._coll_name} _id={doc_id}")
        return self._decode_row(row)

    # --- pymongo-ish public API ---

    def find_one(self, filter: dict | None = None, *args, **kwargs):
        """
        Minimal behavior:
          - if filter contains _id, return that doc
          - else return first doc (used as connectivity test)
        """
        filter = filter or {}

        # _id special case
        if "_id" in filter:
            ...

        if self._coll_name == "runs" and "number" in filter:
            number = int(filter["number"])
            row = self._conn.execute(
                "SELECT doc_id FROM runs_index WHERE db_name=? AND number=? LIMIT 1",
                (self.db_name, number),
            ).fetchone()
            if row is None:
                return None
            return self._get_by_id(row["doc_id"])

        if row is None:
            return None
        return self._decode_row(row)

    def find(self, filter: dict | None = None, *args, **kwargs):
        filter = filter or {}

        # Special-case _id
        if "_id" in filter:
            try:
                doc = self._get_by_id(str(filter["_id"]))
                return _OfflineCursor([doc])  # small list OK
            except KeyError:
                return _OfflineCursor([])

        # Special-case xenonnt.runs by number
        if self._coll_name == "runs" and "number" in filter:
            number = int(filter["number"])
            row = self._conn.execute(
                "SELECT doc_id FROM runs_index WHERE db_name=? AND number=? LIMIT 1",
                (self.db_name, number),
            ).fetchone()
            if row is None:
                return _OfflineCursor([])
            doc = self._get_by_id(row["doc_id"])
            return _OfflineCursor([doc])

        # Default: streaming cursor over all docs
        return _OfflineStreamingCursor(self.iter_all())

    def count_documents(self, filter: dict | None = None, *args, **kwargs) -> int:
        filter = filter or {}

        if "_id" in filter:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM kv_collections \
                    WHERE db_name=? AND coll_name=? AND doc_id=?",
                (self.db_name, self._coll_name, str(filter["_id"])),
            ).fetchone()
            return int(row["n"]) if row else 0

        if self._coll_name == "runs" and "number" in filter:
            number = int(filter["number"])
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM runs_index WHERE db_name=? AND number=?",
                (self.db_name, number),
            ).fetchone()
            return int(row["n"]) if row else 0

        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM kv_collections WHERE db_name=? AND coll_name=?",
            (self.db_name, self._coll_name),
        ).fetchone()
        return int(row["n"]) if row else 0

    def iter_all(self):
        cur = self._conn.execute(
            "SELECT doc_bson_z FROM kv_collections WHERE db_name=? AND coll_name=?",
            (self.db_name, self._coll_name),
        )
        for row in cur:
            yield self._decode_row(row)

    def as_list(self, limit: int | None = None):
        out = []
        for i, d in enumerate(self.iter_all()):
            out.append(d)
            if limit is not None and i + 1 >= limit:
                break
        return out


class _OfflineCursor:
    """Small in-memory cursor (safe only for tiny result sets)."""

    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction=1):
        rev = direction == -1
        self._docs.sort(key=lambda d: d.get(key), reverse=rev)
        return self

    def skip(self, n):
        self._docs = self._docs[int(n):]
        return self

    def limit(self, n):
        self._docs = self._docs[:int(n)]
        return self

    def __iter__(self):
        return iter(self._docs)


class _OfflineStreamingCursor:
    """Streaming cursor: does NOT materialize docs."""

    def __init__(self, iterator):
        self._it = iterator
        self._skip = 0
        self._limit = None
        self._sort_key = None
        self._sort_dir = 1

    def sort(self, key, direction=1):
        # WARNING: true sort requires materialization.
        # Keep it conservative: only allow sort if limit is set (small-ish),
        # otherwise do nothing or raise.
        self._sort_key = key
        self._sort_dir = direction
        return self

    def skip(self, n):
        self._skip = int(n)
        return self

    def limit(self, n):
        self._limit = int(n)
        return self

    def __iter__(self):
        it = self._it

        # apply skip
        for _ in range(self._skip):
            try:
                next(it)
            except StopIteration:
                return iter(())

        # If no sort requested, stream directly
        if self._sort_key is None:
            if self._limit is None:
                return it
            else:
                # stream with limit
                def gen():
                    for i, d in enumerate(it):
                        if i >= self._limit:
                            break
                        yield d

                return gen()

        # If sort requested, we must materialize.
        # We materialize only up to limit if provided, else this is dangerous.
        if self._limit is None:
            raise RuntimeError(
                "Offline streaming cursor cannot sort without limit (would load everything)."
            )

        docs = []
        for i, d in enumerate(it):
            if i >= self._limit:
                break
            docs.append(d)

        rev = self._sort_dir == -1
        docs.sort(key=lambda d: d.get(self._sort_key), reverse=rev)
        return iter(docs)


# Add pymongo spy
_orig_mc = pymongo.MongoClient


class MongoClientSpy(_orig_mc):
    def __init__(self, *args, **kwargs):
        cfg = _load_sqlite_config()
        if cfg.spy:
            block(f"pymongo.MongoClient CREATED args={args} kwargs_keys={list(kwargs.keys())}", cfg)
        super().__init__(*args, **kwargs)


pymongo.MongoClient = MongoClientSpy
