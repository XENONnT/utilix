import re
import time
import unittest

from utilix import *


class TestUpdate(unittest.TestCase):

    def test_update(self):

        creation_place = "test-" + str(time.time())

        db = rundb.DB()
        data = db.get_data(2000)

        db.update_data(
            2000,
            {
                "host": "Host",
                "checksum": "checksum",
                "location": "test_location",
                "creation_time": "2019-04-03T03:17:03.532308",
                "type": "test_plugin",
                "hash": "c793fa6223fifadsf",
                "status": "transferring",
                "lineage_hash": "c793fa6223f488e67646cf7ed06c14653ff04186",
                "protocol": "FileSytemBackend",
                "creation_place": creation_place,
            },
        )

        # pull back and make sure the data has been updated
        found = False
        data = db.get_data(2000)
        for entry in data:
            if "creation_place" in entry and entry["creation_place"] == creation_place:
                found = True
        self.assertTrue(found, msg="Unable to find added entry")

        # clean up test entries
        data = db.get_data(2000)
        for entry in data:
            if "creation_place" in entry and re.search("^test-", entry["creation_place"]):
                db.delete_data(2000, entry)

    def test_multiple_updates(self):

        creation_place = "test-" + str(time.time())

        db = rundb.DB()
        data = db.get_data(2000)

        db.update_data(
            2000,
            {
                "host": "Host",
                "checksum": "checksum",
                "location": "test_location",
                "creation_time": "2019-04-03T03:17:03.532308",
                "type": "test_plugin",
                "hash": "c793fa6223f",
                "status": "transferring",
                "lineage_hash": "c793fa6223f488e67646cf7ed06c14653ff04186",
                "protocol": "FileSytemBackend",
                "creation_place": creation_place,
            },
        )

        db.update_data(
            2000,
            {
                "host": "Host",
                "checksum": "checksumi23",
                "location": "test_location",
                "creation_time": "2019-04-09T03:17:03.532308",
                "type": "test_plugin",
                "hash": "c793fa6223f",
                "status": "transferring",
                "lineage_hash": "c793fa6223f488e67646cf7ed06c14653ff04186",
                "protocol": "FileSytemBackend",
                "creation_place": creation_place,
            },
        )

        # pull back and make sure the data has been updated
        found_count = 0
        data = db.get_data(2000)
        for entry in data:
            if "creation_place" in entry and entry["creation_place"] == creation_place:
                found_count += 1
        self.assertTrue(
            found_count == 1, msg="More than one entry after multiple updates: " + str(found_count)
        )

        # clean up test entries
        data = db.get_data(2000)
        for entry in data:
            if "creation_place" in entry and re.search("^test-", entry["creation_place"]):
                db.delete_data(2000, entry)

    def test_update_meta(self):

        creation_place = "test-" + str(time.time())

        db = rundb.DB()
        data = db.get_data(2000)

        db.update_data(
            2000,
            {
                "host": "Host",
                "checksum": "checksum",
                "location": "test_location",
                "creation_time": "2019-04-03T03:17:03.532308",
                "type": "test_plugin",
                "lineage_hash": "c793fa6223f488e67646cf7ed06c14653ff04186",
                "status": "transferring",
                "creation_place": creation_place,
                "meta": {
                    "chunks": [
                        {
                            "chunk_i": 0,
                            "filename": "000000",
                            "filesize": 10486921,
                            "first_endtime": 1505672371629233400,
                            "first_time": 1505672371629232340,
                            "last_endtime": 1505672379919902590,
                            "last_time": 1505672379919901570,
                            "n": 146594,
                            "nbytes": 36795094,
                        },
                        {
                            "chunk_i": 1,
                            "filename": "000001",
                            "filesize": 18922980,
                            "first_endtime": 1505672380120253340,
                            "first_time": 1505672380120252300,
                            "last_endtime": 1505672390381289830,
                            "last_time": 1505672390381288790,
                            "n": 350824,
                            "nbytes": 88056824,
                        },
                    ],
                    "compressor": "zstd",
                    "data_kind": "records",
                    "data_type": "records",
                },
            },
        )

        # pull back and make sure the data has been updated
        found = False
        data = db.get_data(2000)
        for entry in data:
            if "creation_place" in entry and entry["creation_place"] == creation_place:
                found = True
        self.assertTrue(found, msg="Unable to find added entry")

        # clean up test entries
        data = db.get_data(2000)
        for entry in data:
            if "creation_place" in entry and re.search("^test-", entry["creation_place"]):
                db.delete_data(2000, entry)


if __name__ == "__main__":
    unittest.main()
