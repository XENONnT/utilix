"""Tests for SQLite offline backend functionality."""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bson import BSON


class TestSQLiteConfig(unittest.TestCase):
    """Test SQLiteConfig dataclass and configuration loading."""

    def test_load_config_from_env(self):
        """Test loading SQLite config from environment variables."""
        from utilix.sqlite_backend import _load_sqlite_config

        with tempfile.TemporaryDirectory() as tmpdir:
            rundb_path = Path(tmpdir) / "rundb.sqlite"
            xedocs_path = Path(tmpdir) / "xedocs.sqlite"

            # Create empty files
            rundb_path.touch()
            xedocs_path.touch()

            with patch.dict(
                os.environ,
                {
                    "RUNDB_SQLITE_PATH": str(rundb_path),
                    "XEDOCS_SQLITE_PATH": str(xedocs_path),
                },
            ):
                cfg = _load_sqlite_config()

                self.assertIsNotNone(cfg.rundb_sqlite_path)
                self.assertIsNotNone(cfg.xedocs_sqlite_path)
                # Use resolve() on both sides to handle symlinks (e.g., /var -> /private/var on macOS)
                self.assertEqual(cfg.rundb_sqlite_path.resolve(), rundb_path.resolve())
                self.assertEqual(cfg.xedocs_sqlite_path.resolve(), xedocs_path.resolve())
                self.assertTrue(cfg.rundb_active())
                self.assertTrue(cfg.xedocs_active())
                self.assertTrue(cfg.sqlite_active())

    def test_sqlite_active_requires_both_files(self):
        """Test that sqlite_active() requires both files to exist."""
        from utilix.sqlite_backend import _load_sqlite_config

        with tempfile.TemporaryDirectory() as tmpdir:
            rundb_path = Path(tmpdir) / "rundb.sqlite"
            xedocs_path = Path(tmpdir) / "xedocs.sqlite"

            # Only create rundb file
            rundb_path.touch()

            with patch.dict(
                os.environ,
                {
                    "RUNDB_SQLITE_PATH": str(rundb_path),
                    "XEDOCS_SQLITE_PATH": str(xedocs_path),
                },
            ):
                cfg = _load_sqlite_config()

                self.assertTrue(cfg.rundb_active())
                self.assertFalse(cfg.xedocs_active())
                self.assertFalse(cfg.sqlite_active())  # Requires BOTH

    def test_sqlite_active_false_when_no_env_vars(self):
        """Test that sqlite_active() is False without environment variables."""
        from utilix.sqlite_backend import _load_sqlite_config

        with patch.dict(os.environ, {}, clear=True):
            # Remove RUNDB_SQLITE_PATH and XEDOCS_SQLITE_PATH if present
            os.environ.pop("RUNDB_SQLITE_PATH", None)
            os.environ.pop("XEDOCS_SQLITE_PATH", None)

            cfg = _load_sqlite_config()

            self.assertFalse(cfg.rundb_active())
            self.assertFalse(cfg.xedocs_active())
            self.assertFalse(cfg.sqlite_active())


class TestOfflineGridFS(unittest.TestCase):
    """Test OfflineGridFS for file operations."""

    def setUp(self):
        """Create temporary directory and mock SQLite database."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

        # Create mock SQLite database with gridfs_files table
        self.db_path = self.tmppath / "rundb.sqlite"
        self.blob_path = self.tmppath / "test_blob.txt"

        # Write test blob
        self.blob_path.write_text("test content")

        # Create database with gridfs_files table
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """CREATE TABLE gridfs_files ( db_name TEXT, file_id TEXT, config_name TEXT, md5 TEXT,
            length INTEGER, uploadDate INTEGER,

            blob_path TEXT )

            """
        )
        conn.execute(
            """INSERT INTO gridfs_files (db_name, file_id, config_name, md5, length, uploadDate,
            blob_path) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "files",
                "test_id",
                "test_config",
                "abc123",
                12,
                1234567890,
                "test_blob.txt",
            ),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    def test_offline_gridfs_list_files(self):
        """Test listing files from offline GridFS."""
        from utilix.sqlite_backend import OfflineGridFS

        gfs = OfflineGridFS(
            sqlite_path=self.db_path,
            offline_root=self.tmppath,
            cache_dirs=(self.tmppath / "cache",),
        )

        files = gfs.list_files()
        self.assertIn("test_config", files)
        gfs.close()

    def test_offline_gridfs_download_single(self):
        """Test downloading a single file from offline GridFS."""
        from utilix.sqlite_backend import OfflineGridFS

        cache_dir = self.tmppath / "cache"
        gfs = OfflineGridFS(
            sqlite_path=self.db_path,
            offline_root=self.tmppath,
            cache_dirs=(cache_dir,),
        )

        # Download file
        result_path = gfs.download_single("test_config")

        # Should be cached by md5
        self.assertTrue(Path(result_path).exists())
        self.assertIn("abc123", result_path)  # md5 in filename

        gfs.close()

    def test_offline_gridfs_missing_config_raises(self):
        """Test that missing config raises KeyError."""
        from utilix.sqlite_backend import OfflineGridFS

        gfs = OfflineGridFS(
            sqlite_path=self.db_path,
            offline_root=self.tmppath,
            cache_dirs=(self.tmppath / "cache",),
        )

        with self.assertRaises(KeyError):
            gfs.download_single("nonexistent_config")

        gfs.close()


class TestOfflineSQLiteCollection(unittest.TestCase):
    """Test OfflineSQLiteCollection for database queries."""

    def setUp(self):
        """Create temporary SQLite database with test data."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "rundb.sqlite"

        # Create database with kv_collections and runs_index tables
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""CREATE TABLE kv_collections (
                db_name TEXT,
                coll_name TEXT,
                doc_id TEXT,
                doc_bson_z BLOB
            )""")
        conn.execute("""CREATE TABLE runs_index (
                db_name TEXT,
                number INTEGER,
                doc_id TEXT
            )""")

        # Insert test document
        import zlib

        test_doc = {"_id": "test_id_123", "number": 12345, "name": "test_run"}
        bson_data = BSON.encode(test_doc)
        compressed = zlib.compress(bson_data, level=6)

        conn.execute(
            "INSERT INTO kv_collections (db_name, coll_name, doc_id, doc_bson_z) VALUES (?, ?, ?, ?)",
            ("xenonnt", "runs", "test_id_123", compressed),
        )

        conn.execute(
            "INSERT INTO runs_index (db_name, number, doc_id) VALUES (?, ?, ?)",
            ("xenonnt", 12345, "test_id_123"),
        )

        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up temporary directory."""
        self.tmpdir.cleanup()

    def test_find_one_by_id(self):
        """Test find_one with _id filter."""
        from utilix.sqlite_backend import OfflineSQLiteCollection

        coll = OfflineSQLiteCollection(
            sqlite_path=self.db_path,
            db_name="xenonnt",
            coll_name="runs",
            compression="zlib",
        )

        doc = coll.find_one({"_id": "test_id_123"})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["_id"], "test_id_123")
        self.assertEqual(doc["number"], 12345)

        coll.close()

    def test_find_one_by_number(self):
        """Test find_one with number filter for runs collection."""
        from utilix.sqlite_backend import OfflineSQLiteCollection

        coll = OfflineSQLiteCollection(
            sqlite_path=self.db_path,
            db_name="xenonnt",
            coll_name="runs",
            compression="zlib",
        )

        doc = coll.find_one({"number": 12345})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["number"], 12345)
        self.assertEqual(doc["_id"], "test_id_123")

        coll.close()

    def test_find_one_default_returns_first_doc(self):
        """Test find_one without filter returns first document."""
        from utilix.sqlite_backend import OfflineSQLiteCollection

        coll = OfflineSQLiteCollection(
            sqlite_path=self.db_path,
            db_name="xenonnt",
            coll_name="runs",
            compression="zlib",
        )

        doc = coll.find_one()
        self.assertIsNotNone(doc)
        self.assertEqual(doc["_id"], "test_id_123")

        coll.close()

    def test_count_documents(self):
        """Test count_documents method."""
        from utilix.sqlite_backend import OfflineSQLiteCollection

        coll = OfflineSQLiteCollection(
            sqlite_path=self.db_path,
            db_name="xenonnt",
            coll_name="runs",
            compression="zlib",
        )

        count = coll.count_documents({})
        self.assertEqual(count, 1)

        count = coll.count_documents({"number": 12345})
        self.assertEqual(count, 1)

        count = coll.count_documents({"number": 99999})
        self.assertEqual(count, 0)

        coll.close()

    def test_find_returns_cursor(self):
        """Test find method returns iterable cursor."""
        from utilix.sqlite_backend import OfflineSQLiteCollection

        coll = OfflineSQLiteCollection(
            sqlite_path=self.db_path,
            db_name="xenonnt",
            coll_name="runs",
            compression="zlib",
        )

        cursor = coll.find({"number": 12345})
        docs = list(cursor)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["number"], 12345)

        coll.close()


class TestXentCollectionOffline(unittest.TestCase):
    """Test xent_collection() function with offline mode."""

    def test_xent_collection_uses_sqlite_when_active(self):
        """Test that xent_collection uses SQLite when offline is active."""
        from utilix.sqlite_backend import SQLiteConfig, OfflineSQLiteCollection
        from utilix.rundb import xent_collection

        with tempfile.TemporaryDirectory() as tmpdir:
            rundb_path = Path(tmpdir) / "rundb.sqlite"
            xedocs_path = Path(tmpdir) / "xedocs.sqlite"
            rundb_path.touch()
            xedocs_path.touch()

            with patch.dict(
                os.environ,
                {
                    "RUNDB_SQLITE_PATH": str(rundb_path),
                    "XEDOCS_SQLITE_PATH": str(xedocs_path),
                },
            ):
                with patch("utilix.rundb.uconfig") as mock_config:
                    mock_config.get.return_value = "xenonnt"

                    coll = xent_collection("runs")

                    # Should return OfflineSQLiteCollection when offline is active
                    self.assertIsInstance(coll, OfflineSQLiteCollection)
                    coll.close()

    def test_xent_collection_uses_mongodb_when_offline_inactive(self):
        """Test that xent_collection uses MongoDB when offline is not active."""
        from utilix.rundb import xent_collection

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNDB_SQLITE_PATH", None)
            os.environ.pop("XEDOCS_SQLITE_PATH", None)

            with patch("utilix.rundb._collection") as mock_collection:
                mock_collection.return_value = MagicMock()

                coll = xent_collection("runs")

                # Should call _collection (MongoDB) when offline is not active
                mock_collection.assert_called_once()


if __name__ == "__main__":
    unittest.main()
