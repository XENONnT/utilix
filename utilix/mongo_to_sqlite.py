#!/usr/bin/env python3
"""Dump selected MongoDB collections + GridFS into local SQLite(s).

NEW:
- xedocs:* is dumped into a separate SQLite file (xedocs.sqlite) with
  one table per xedocs collection and useful indexes.
- everything else stays as before (rundb.sqlite with kv_collections + runs_index + gridfs_files).

Spec file examples:
    xenonnt:runs
    files:GRIDFS
    xedocs:ALL
    corrections:ALL

"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import pymongo
from bson import BSON
from bson.objectid import ObjectId


# -------------------------
# Compression helpers
# -------------------------


def _compressor():
    try:
        import zstandard as zstd  # type: ignore

        cctx = zstd.ZstdCompressor(level=10)
        dctx = zstd.ZstdDecompressor()

        def compress(b: bytes) -> bytes:
            return cctx.compress(b)

        def decompress(b: bytes) -> bytes:
            return dctx.decompress(b)

        return "zstd", compress, decompress
    except Exception:
        import zlib

        def compress(b: bytes) -> bytes:
            return zlib.compress(b, level=6)

        def decompress(b: bytes) -> bytes:
            return zlib.decompress(b)

        return "zlib", compress, decompress


COMP_ALGO, compress_bytes, _ = _compressor()


# -------------------------
# Spec parsing
# -------------------------


@dataclass(frozen=True)
class SpecItem:
    db: str
    what: str  # collection name, "ALL", or "GRIDFS"


def parse_spec_lines(lines: Iterable[str]) -> List[SpecItem]:
    out: List[SpecItem] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            raise ValueError(f"Bad spec line (expected db:thing): {s}")
        db, what = s.split(":", 1)
        db, what = db.strip(), what.strip()
        if not db or not what:
            raise ValueError(f"Bad spec line (empty db/thing): {s}")
        out.append(SpecItem(db=db, what=what))
    return out


# -------------------------
# Mongo connection (utilix-friendly)
# -------------------------


def get_utilix_mongo_uri(experiment: str) -> str:
    """Mirrors utilix._collection style:

    mongodb://{user}:{password}@{url}

    """
    from utilix import uconfig  # type: ignore

    if experiment not in ("xent", "xe1t"):
        raise ValueError("experiment must be 'xent' or 'xe1t'")

    url = uconfig.get("RunDB", f"{experiment}_url")
    user = uconfig.get("RunDB", f"{experiment}_user")
    password = uconfig.get("RunDB", f"{experiment}_password")

    force_single_server = uconfig.get("RunDB", "force_single_server", fallback=True)
    if force_single_server:
        url = url.split(",")[-1]

    return f"mongodb://{user}:{password}@{url}"


def get_mongo_client(experiment: str, uri_override: Optional[str] = None) -> pymongo.MongoClient:
    uri = uri_override or get_utilix_mongo_uri(experiment)

    kwargs: Dict[str, object] = {
        "serverSelectionTimeoutMS": 30_000,
        "connectTimeoutMS": 30_000,
        "socketTimeoutMS": 60_000,
        "retryWrites": False,
        "readPreference": "secondaryPreferred",
    }
    if int(pymongo.__version__.split(".")[0]) >= 4:
        kwargs["directConnection"] = True

    return pymongo.MongoClient(uri, **kwargs)


# -------------------------
# SQLite schema (rundb.sqlite)
# -------------------------

SCHEMA_SQL_RUNDB = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS kv_collections (
  db_name       TEXT NOT NULL,
  coll_name     TEXT NOT NULL,
  doc_id        TEXT NOT NULL,
  doc_bson_z    BLOB NOT NULL,
  PRIMARY KEY (db_name, coll_name, doc_id)
);

CREATE TABLE IF NOT EXISTS runs_index (
  db_name       TEXT NOT NULL,
  doc_id        TEXT NOT NULL,
  number        INTEGER,
  name          TEXT,
  start         INTEGER,
  end           INTEGER,
  tags_json     TEXT,
  PRIMARY KEY (db_name, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_number ON runs_index(db_name, number);
CREATE INDEX IF NOT EXISTS idx_runs_name   ON runs_index(db_name, name);
CREATE INDEX IF NOT EXISTS idx_runs_start  ON runs_index(db_name, start);

CREATE TABLE IF NOT EXISTS gridfs_files (
  db_name       TEXT NOT NULL,
  file_id       TEXT NOT NULL,
  filename      TEXT,
  config_name   TEXT,
  length        INTEGER,
  chunkSize     INTEGER,
  uploadDate    INTEGER,
  md5           TEXT,
  metadata_json TEXT,
  logical_name  TEXT,
  blob_path     TEXT NOT NULL,
  PRIMARY KEY (db_name, file_id)
);

CREATE INDEX IF NOT EXISTS idx_gridfs_filename ON gridfs_files(db_name, filename);
CREATE INDEX IF NOT EXISTS idx_gridfs_configname ON gridfs_files(db_name, config_name);
"""


# -------------------------
# SQLite schema (xedocs.sqlite)
# -------------------------


def _schema_sql_xedocs_table(table: str, extra_label_cols: List[str]) -> str:
    """Create one table per xedocs collection.

    We keep a stable set of "core" columns (id/version/time/value/full doc), and
    *also* create additional TEXT columns for any label fields we discover from
    sampling documents in that collection.

    Note: extra label columns are quoted to tolerate odd names.

    """

    def q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # Core columns
    cols = [
        f"{q('_id')} TEXT PRIMARY KEY",
        f"{q('version')} TEXT",
        f"{q('time_ns')} INTEGER",
        f"{q('time_left_ns')} INTEGER",
        f"{q('time_right_ns')} INTEGER",
        f"{q('created_date_ns')} INTEGER",
        f"{q('value_num')} REAL",
        f"{q('value_json')} TEXT",
    ]

    # Discovered label columns (TEXT)
    for c in extra_label_cols:
        if c in {
            "_id",
            "version",
            "time_ns",
            "time_left_ns",
            "time_right_ns",
            "created_date_ns",
            "value_num",
            "value_json",
            "doc_bson_z",
        }:
            continue
        cols.append(f"{q(c)} TEXT")

    # Full original BSON (compressed)
    cols.append(f"{q('doc_bson_z')} BLOB NOT NULL")

    # Always-create indexes:
    # - time sampled lookup:   version + time
    # - time interval lookup:  version + interval
    # - common labels (if present)
    index_sql = [
        f"CREATE INDEX IF NOT EXISTS \
            {q('idx_' + table + '_version_time')} \
            ON {q(table)}({q('version')}, {q('time_ns')});",
        f"CREATE INDEX IF NOT EXISTS \
            {q('idx_' + table + '_version_interval')} \
            ON {q(table)}({q('version')}, {q('time_left_ns')}, {q('time_right_ns')});",
    ]

    # Optional label indexes (keep this small to avoid DB bloat)
    preferred = [
        "algorithm",
        "config_name",
        "detector",
        "source",
        "pmt",
        "gain_model",
    ]

    present = set(extra_label_cols)
    n_extra = 0
    for lab in preferred:
        if lab in present:
            index_sql.append(
                f"CREATE INDEX IF NOT EXISTS \
                    {q('idx_' + table + '_version_' + lab)} \
                        ON {q(table)}({q('version')}, {q(lab)});"
            )
            n_extra += 1
            if n_extra >= 6:
                break

    cols_sql = ",\n  ".join(cols)
    idx_sql = "\n\n".join(index_sql)

    return f"""
CREATE TABLE IF NOT EXISTS {q(table)} (
  {cols_sql}
);

{idx_sql}
"""


# -------------------------
# Utilities
# -------------------------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_s() -> float:
    return time.time()


def oid_to_str(x) -> str:
    if isinstance(x, ObjectId):
        return str(x)
    return str(x)


def to_unix_seconds(dtobj) -> Optional[int]:
    try:
        if dtobj is None:
            return None
        return int(dtobj.timestamp())
    except Exception:
        return None


def to_utc_ns(dtobj) -> Optional[int]:
    try:
        if dtobj is None:
            return None
        # bson datetime is usually naive but UTC
        # treat naive as UTC
        if getattr(dtobj, "tzinfo", None) is None:
            import datetime as dt

            dtobj = dtobj.replace(tzinfo=dt.timezone.utc)
        return int(dtobj.timestamp() * 1_000_000_000)
    except Exception:
        return None


def bson_pack(doc: dict) -> bytes:
    return BSON.encode(doc)


def pack_and_compress(doc: dict) -> bytes:
    return compress_bytes(bson_pack(doc))


def list_collection_names_safe(db: pymongo.database.Database) -> List[str]:
    try:
        return db.list_collection_names()
    except pymongo.errors.OperationFailure as e:
        raise RuntimeError(
            f"Not authorized to list collections in DB '{db.name}'. "
            f"Use explicit spec lines (db:collection) instead of db:ALL. "
            f"Mongo error: {e}"
        ) from e


# -------------------------
# Dump logic (generic -> rundb.sqlite kv_collections)
# -------------------------


def dump_generic_collection(
    mongo_db: pymongo.database.Database,
    coll_name: str,
    sql: sqlite3.Connection,
    out_db_name: str,
    batch_size: int,
    logger: logging.Logger,
    query: Optional[dict] = None,
    projection: Optional[dict] = None,
) -> int:
    query = query or {}
    coll = mongo_db[coll_name]

    logger.info(f"[mongo] dumping {mongo_db.name}.{coll_name} -> rundb.sqlite kv_collections")
    t0 = now_s()

    cur = coll.find(query, projection=projection, no_cursor_timeout=True, batch_size=batch_size)
    n = 0
    buf: List[Tuple[str, str, str, bytes]] = []

    insert_sql =\
                 "INSERT OR REPLACE INTO kv_collections(db_name, coll_name, doc_id, doc_bson_z) \
        VALUES (?,?,?,?)"

    for doc in cur:
        _id = doc.get("_id")
        doc_id = oid_to_str(_id) if _id is not None else f"noid:{n}"
        blob = pack_and_compress(doc)
        buf.append((out_db_name, coll_name, doc_id, blob))
        n += 1

        if len(buf) >= batch_size:
            sql.executemany(insert_sql, buf)
            sql.commit()
            buf.clear()

    if buf:
        sql.executemany(insert_sql, buf)
        sql.commit()

    dt = now_s() - t0
    logger.info(f"[mongo] done {mongo_db.name}.{coll_name}: {n} docs in {dt:.1f}s")
    return n


def dump_xenonnt_runs_index(
    mongo_db: pymongo.database.Database,
    runs_coll_name: str,
    sql: sqlite3.Connection,
    out_db_name: str,
    batch_size: int,
    logger: logging.Logger,
    drop_fields: Optional[List[str]] = None,
) -> int:
    drop_fields = drop_fields or []
    coll = mongo_db[runs_coll_name]

    logger.info(f"[mongo] dumping runs {mongo_db.name}.{runs_coll_name} with index + compression")
    t0 = now_s()

    cur = coll.find({}, no_cursor_timeout=True, batch_size=batch_size)
    n = 0
    buf_kv: List[Tuple[str, str, str, bytes]] = []
    buf_idx: List[
        Tuple[str, str, Optional[int], Optional[str], Optional[int], Optional[int], Optional[str]]
    ] = []

    ins_kv =\
             "INSERT OR REPLACE INTO kv_collections(db_name, coll_name, doc_id, doc_bson_z)\
        VALUES (?,?,?,?)"
    ins_idx = """
      INSERT OR REPLACE INTO runs_index(db_name, doc_id, number, name, start, end, tags_json)
      VALUES (?,?,?,?,?,?,?)
    """

    for doc in cur:
        _id = doc.get("_id")
        doc_id = oid_to_str(_id) if _id is not None else f"noid:{n}"

        number = doc.get("number") or doc.get("run_number") or doc.get("runNumber")
        try:
            number_i = int(number) if number is not None else None
        except Exception:
            number_i = None

        name = doc.get("name") or doc.get("run_name") or doc.get("runName")

        start = (
            doc.get("start")
            or doc.get("start_time")
            or doc.get("startTime")
            or doc.get("starttime")
        )
        end = doc.get("end") or doc.get("end_time") or doc.get("endTime") or doc.get("endtime")

        start_u = to_unix_seconds(start)
        end_u = to_unix_seconds(end)

        tags = doc.get("tags")
        tags_json = None
        try:
            if tags is not None:
                tags_json = json.dumps(tags, default=str)
        except Exception:
            tags_json = None

        if drop_fields:
            doc = dict(doc)
            for k in drop_fields:
                doc.pop(k, None)

        blob = pack_and_compress(doc)

        buf_kv.append((out_db_name, runs_coll_name, doc_id, blob))
        buf_idx.append(
            (
                out_db_name,
                doc_id,
                number_i,
                str(name) if name is not None else None,
                start_u,
                end_u,
                tags_json,
            )
        )
        n += 1

        if len(buf_kv) >= batch_size:
            sql.executemany(ins_kv, buf_kv)
            sql.executemany(ins_idx, buf_idx)
            sql.commit()
            buf_kv.clear()
            buf_idx.clear()

    if buf_kv:
        sql.executemany(ins_kv, buf_kv)
        sql.executemany(ins_idx, buf_idx)
        sql.commit()

    dt = now_s() - t0
    logger.info(f"[mongo] done runs {mongo_db.name}.{runs_coll_name}: {n} docs in {dt:.1f}s")
    return n


def dump_gridfs_db(
    mongo_db: pymongo.database.Database,
    sql: sqlite3.Connection,
    out_root: Path,
    logger: logging.Logger,
    batch_size: int,
    only_configs: Optional[List[str]] = None,
) -> int:
    import json as _json

    files_coll = mongo_db["fs.files"]
    chunks_coll = mongo_db["fs.chunks"]

    out_dir = out_root / "gridfs" / mongo_db.name / "blobs"
    ensure_dir(out_dir)

    query = {}
    if only_configs:
        query = {"config_name": {"$in": only_configs}}

    logger.info(f"[gridfs] dumping GridFS from DB '{mongo_db.name}' to {out_dir}")
    t0 = now_s()

    cursor = files_coll.find(query, no_cursor_timeout=True).sort("uploadDate", 1)

    n = 0
    buf: List[Tuple] = []

    ins = """
      INSERT OR REPLACE INTO gridfs_files(
        db_name, file_id, filename, config_name, length, chunkSize, uploadDate, md5,
        metadata_json, logical_name, blob_path
      )
      VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """

    for fdoc in cursor:
        file_id = fdoc["_id"]
        file_id_s = oid_to_str(file_id)

        filename = fdoc.get("filename")
        config_name = fdoc.get("config_name") or fdoc.get("name") or fdoc.get("config")

        length = int(fdoc.get("length", 0))
        chunk_size = int(fdoc.get("chunkSize", 255 * 1024))
        upload_u = to_unix_seconds(fdoc.get("uploadDate"))
        md5 = fdoc.get("md5")

        meta = fdoc.get("metadata")
        metadata_json = _json.dumps(meta, default=str) if meta is not None else None

        logical_name = (
            config_name
            or filename
            or (meta.get("filename") if isinstance(meta, dict) else None)
            or (meta.get("name") if isinstance(meta, dict) else None)
            or None
        )

        display = (logical_name or "NO_NAME").replace("/", "_")
        blob_name = f"{file_id_s}__{display}"
        blob_rel = str(Path("gridfs") / mongo_db.name / "blobs" / blob_name)
        blob_abs = out_root / blob_rel

        if not blob_abs.exists() or blob_abs.stat().st_size != length:
            tmp_path = blob_abs.with_suffix(blob_abs.suffix + ".tmp")
            ensure_dir(tmp_path.parent)

            with tmp_path.open("wb") as out_f:
                expected_n = 0
                ch_cur = chunks_coll.find({"files_id": file_id}, no_cursor_timeout=True).sort(
                    "n", 1
                )
                wrote = 0
                for ch in ch_cur:
                    n_chunk = int(ch["n"])
                    if n_chunk != expected_n:
                        raise RuntimeError(
                            f"[gridfs] Missing chunk for file_id={file_id_s}: "
                            f"expected n={expected_n}, got n={n_chunk}"
                        )
                    out_f.write(bytes(ch["data"]))
                    wrote += len(ch["data"])
                    expected_n += 1

                if wrote > length:
                    out_f.flush()
                    out_f.seek(length)
                    out_f.truncate()

            tmp_path.replace(blob_abs)

        buf.append(
            (
                mongo_db.name,
                file_id_s,
                filename,
                config_name,
                length,
                chunk_size,
                upload_u,
                md5,
                metadata_json,
                logical_name,
                blob_rel,
            )
        )
        n += 1

        if len(buf) >= batch_size:
            sql.executemany(ins, buf)
            sql.commit()
            buf.clear()

    if buf:
        sql.executemany(ins, buf)
        sql.commit()

    dt = now_s() - t0
    logger.info(f"[gridfs] done '{mongo_db.name}': {n} files in {dt:.1f}s")
    return n


# -------------------------
# Dump logic (xedocs -> xedocs.sqlite tables)
# -------------------------


def _xedocs_extract(doc: dict, label_cols: List[str]) -> Dict[str, Any]:
    """Extract core xedocs fields + discovered label columns."""
    out: Dict[str, Any] = {}

    out["_id"] = oid_to_str(doc.get("_id"))
    out["version"] = doc.get("version")

    created_date = doc.get("created_date") or doc.get("createdDate")
    out["created_date_ns"] = to_utc_ns(created_date)

    # time handling
    out["time_ns"] = None
    out["time_left_ns"] = None
    out["time_right_ns"] = None

    t = doc.get("time")
    if t is not None:
        if isinstance(t, dict) and ("left" in t or "right" in t):
            out["time_left_ns"] = to_utc_ns(t.get("left"))
            out["time_right_ns"] = to_utc_ns(t.get("right"))
        else:
            out["time_ns"] = to_utc_ns(t)

    # value columns
    v = doc.get("value", None)
    out["value_num"] = None
    try:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out["value_num"] = float(v)
    except Exception:
        pass

    try:
        out["value_json"] = json.dumps(v, default=str)
    except Exception:
        out["value_json"] = None

    # discovered labels (TEXT)
    for k in label_cols:
        if k in (
            "_id",
            "version",
            "time",
            "created_date",
            "createdDate",
            "value",
            "comments",
            "reviews",
        ):
            continue
        val = doc.get(k, None)
        if val is None:
            out[k] = None
            continue
        # Keep labels reasonably queryable: store simple types as strings,
        # otherwise JSON-encode.
        if isinstance(val, (str, int, float, bool)):
            out[k] = str(val) if not isinstance(val, str) else val
        else:
            try:
                out[k] = json.dumps(val, default=str)
            except Exception:
                out[k] = str(val)

    out["doc_bson_z"] = pack_and_compress(doc)
    return out


def dump_xedocs_collection_to_tables(
    mongo_db: pymongo.database.Database,
    coll_name: str,
    sql_x: sqlite3.Connection,
    batch_size: int,
    logger: logging.Logger,
    sample_n: int = 1000,
) -> int:
    """Dump xedocs.<coll> into xedocs.sqlite table <coll> with auto-discovered label columns."""
    coll = mongo_db[coll_name]
    table = coll_name

    logger.info(
        f"[mongo] dumping xedocs.{coll_name} -> xedocs.sqlite table '{table}' (auto-discover)"
    )

    # ---------
    # 1) Discover label columns from a sample of docs
    # ---------
    skip_keys = {
        "_id",
        "time",
        "value",
        "created_date",
        "createdDate",
        "comments",
        "reviews",
    }

    label_cols_set = set()
    try:
        sample_cursor = coll.find(
            {}, no_cursor_timeout=True, batch_size=min(batch_size, 500)
        ).limit(sample_n)
        for d in sample_cursor:
            for k in d.keys():
                if k in skip_keys:
                    continue
                # We keep 'version' as a core column, but allow it in schema generation
                # (it will be ignored if duplicated)
                label_cols_set.add(k)
    except Exception as e:
        logger.warning(
            f"[mongo] xedocs label discovery failed for {coll_name}: {type(e).__name__}: {e}"
        )

    # Deterministic order
    label_cols = sorted(label_cols_set)

    # ---------
    # 2) Create table schema (core + discovered labels)
    # ---------
    sql_x.executescript(_schema_sql_xedocs_table(table, extra_label_cols=label_cols))
    sql_x.commit()

    # ---------
    # 3) Dump all docs
    # ---------
    t0 = now_s()

    # Build INSERT dynamically
    # Core columns (must match schema)
    core_cols = [
        "_id",
        "version",
        "time_ns",
        "time_left_ns",
        "time_right_ns",
        "created_date_ns",
        "value_num",
        "value_json",
    ]

    # Only keep label columns that are not core columns and are valid SQL identifiers when quoted
    # (we always quote, so any name is okay)
    extra_cols = [
        c
        for c in label_cols
        if c
        not in {
            "_id",
            "version",
            "time_ns",
            "time_left_ns",
            "time_right_ns",
            "created_date_ns",
            "value_num",
            "value_json",
            "doc_bson_z",
        }
    ]

    all_cols = core_cols + extra_cols + ["doc_bson_z"]

    def q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    placeholders = ",".join(["?"] * len(all_cols))
    ins = f"INSERT OR REPLACE INTO \
        {q(table)}({','.join(q(c) for c in all_cols)}) \
        VALUES ({placeholders})"

    cur = coll.find({}, no_cursor_timeout=True, batch_size=batch_size)

    n = 0
    buf: List[Tuple[Any, ...]] = []

    for doc in cur:
        e = _xedocs_extract(doc, label_cols=extra_cols)
        row = tuple(e.get(c) for c in all_cols)
        buf.append(row)
        n += 1

        if len(buf) >= batch_size:
            sql_x.executemany(ins, buf)
            sql_x.commit()
            buf.clear()

    if buf:
        sql_x.executemany(ins, buf)
        sql_x.commit()

    dt = now_s() - t0
    logger.info(f"[mongo] done xedocs.{coll_name}: {n} docs in {dt:.1f}s")
    return n


# -------------------------
# Main
# -------------------------


def setup_logger(verbosity: int) -> logging.Logger:
    lvl = logging.INFO if verbosity == 0 else (logging.DEBUG if verbosity >= 1 else logging.INFO)
    logger = logging.getLogger("dump_mongo_offline")
    logger.setLevel(lvl)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(lvl)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for offline cache")
    ap.add_argument(
        "--experiment", default="xent", choices=["xent", "xe1t"], help="utilix experiment"
    )
    ap.add_argument(
        "--mongo-uri", default=None, help="Override Mongo URI (otherwise uses utilix uconfig)"
    )
    ap.add_argument(
        "--spec",
        required=True,
        help="Spec file with lines like 'xenonnt:runs', 'xedocs:ALL', 'files:GRIDFS'",
    )
    ap.add_argument(
        "--sqlite-name",
        default="rundb.sqlite",
        help="SQLite filename under --out for runs/gridfs/kv",
    )
    ap.add_argument(
        "--xedocs-sqlite-name",
        default="xedocs.sqlite",
        help="SQLite filename under --out for xedocs tables",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch size for Mongo cursor and SQLite inserts",
    )
    ap.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v/-vv)"
    )

    ap.add_argument(
        "--runs-drop-field",
        action="append",
        default=[],
        help="Drop a field from xenonnt:runs docs before storing (repeatable).",
    )

    ap.add_argument(
        "--gridfs-only-filenames",
        default=None,
        help="Text file with one filename per line to dump from GridFS",
    )
    args = ap.parse_args()

    logger = setup_logger(args.verbose)

    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    spec_path = Path(args.spec).resolve()
    spec_items = parse_spec_lines(spec_path.read_text().splitlines())

    logger.info(
        f"Connecting to Mongo (experiment={args.experiment}, uri_override={bool(args.mongo_uri)})"
    )
    client = get_mongo_client(args.experiment, uri_override=args.mongo_uri)

    # rundb.sqlite
    sqlite_path = out_root / args.sqlite_name
    logger.info(f"Opening rundb SQLite at {sqlite_path}")
    sql = sqlite3.connect(str(sqlite_path))
    sql.executescript(SCHEMA_SQL_RUNDB)
    sql.commit()

    # xedocs.sqlite (only opened if needed)
    xedocs_sqlite_path = out_root / args.xedocs_sqlite_name
    sql_x: Optional[sqlite3.Connection] = None

    gridfs_only = None
    if args.gridfs_only_filenames:
        gridfs_only = [
            ln.strip()
            for ln in Path(args.gridfs_only_filenames).read_text().splitlines()
            if ln.strip()
        ]

    manifest = {
        "format": "offline-mongo-sqlite-v2",
        "created_at_unix": int(time.time()),
        "compression": COMP_ALGO,
        "experiment": args.experiment,
        "spec_file": str(spec_path),
        "spec": [{"db": x.db, "what": x.what} for x in spec_items],
        "sqlite_rundb": str(sqlite_path.name),
        "sqlite_xedocs": str(xedocs_sqlite_path.name),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote manifest.json (compression={COMP_ALGO})")

    def _get_sql_x() -> sqlite3.Connection:
        nonlocal sql_x
        if sql_x is None:
            logger.info(f"Opening xedocs SQLite at {xedocs_sqlite_path}")
            sql_x = sqlite3.connect(str(xedocs_sqlite_path))
            # some pragmas for speed
            sql_x.execute("PRAGMA journal_mode = WAL;")
            sql_x.execute("PRAGMA synchronous = NORMAL;")
            sql_x.execute("PRAGMA temp_store = MEMORY;")
            sql_x.commit()
        return sql_x

    for item in spec_items:
        dbname = item.db
        what = item.what
        mongo_db = client[dbname]

        if what.upper() == "GRIDFS":
            dump_gridfs_db(
                mongo_db=mongo_db,
                sql=sql,
                out_root=out_root,
                logger=logger,
                batch_size=max(200, args.batch_size // 5),
                only_configs=gridfs_only,
            )
            continue

        if what.upper() == "ALL":
            names = list_collection_names_safe(mongo_db)
            logger.info(f"[mongo] {dbname}:ALL expanded to {len(names)} collections")

            for cname in names:
                if cname in ("fs.files", "fs.chunks"):
                    logger.info(f"[mongo] skipping {dbname}.{cname} (use {dbname}:GRIDFS instead)")
                    continue

                if dbname == "xedocs":
                    dump_xedocs_collection_to_tables(
                        mongo_db=mongo_db,
                        coll_name=cname,
                        sql_x=_get_sql_x(),
                        batch_size=args.batch_size,
                        logger=logger,
                        sample_n=1000,
                    )
                else:
                    dump_generic_collection(
                        mongo_db=mongo_db,
                        coll_name=cname,
                        sql=sql,
                        out_db_name=dbname,
                        batch_size=args.batch_size,
                        logger=logger,
                    )
            continue

        # Single collection
        cname = what

        if dbname == "xedocs":
            dump_xedocs_collection_to_tables(
                mongo_db=mongo_db,
                coll_name=cname,
                sql_x=_get_sql_x(),
                batch_size=args.batch_size,
                logger=logger,
                sample_n=1000,
            )
            continue

        if dbname == "xenonnt" and cname == "runs":
            dump_xenonnt_runs_index(
                mongo_db=mongo_db,
                runs_coll_name=cname,
                sql=sql,
                out_db_name=dbname,
                batch_size=args.batch_size,
                logger=logger,
                drop_fields=args.runs_drop_field,
            )
        else:
            dump_generic_collection(
                mongo_db=mongo_db,
                coll_name=cname,
                sql=sql,
                out_db_name=dbname,
                batch_size=args.batch_size,
                logger=logger,
            )

    logger.info("ANALYZE (optional)...")
    try:
        sql.execute("ANALYZE;")
        sql.commit()
    except Exception:
        logger.warning("ANALYZE failed for rundb.sqlite (continuing)")

    if sql_x is not None:
        try:
            sql_x.execute("ANALYZE;")
            sql_x.commit()
        except Exception:
            logger.warning("ANALYZE failed for xedocs.sqlite (continuing)")

    logger.info("All done.")
    logger.info(f"Offline cache written to: {out_root}")
    logger.info(f"rundb.sqlite : {sqlite_path}")
    if sql_x is not None:
        logger.info(f"xedocs.sqlite: {xedocs_sqlite_path}")


if __name__ == "__main__":
    main()
