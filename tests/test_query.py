import unittest

from utilix import *


class TestQuery(unittest.TestCase):

    def test_query(self):
        db = rundb.DB()
        data = db.query(page_num=1)
        # pprint(data)
        self.assertTrue(len(data) == 1000, "Number of records: " + str(len(data)))

    def test_query_paging(self):
        db = rundb.DB()
        data = db.query(page_num=1)
        self.assertTrue(len(data) == 1000, "Number of records: " + str(len(data)))
        data = db.query(page_num=2)
        self.assertTrue(len(data) == 1000, "Number of records: " + str(len(data)))
        data = db.query(page_num=3)
        self.assertTrue(len(data) == 1000, "Number of records: " + str(len(data)))

    def test_query_by_source(self):
        db = rundb.DB()
        data = db.query_by_source("calibration", page_num=1)
        # pprint(data)
        self.assertTrue(len(data) > 0)

    def test_query_by_tag(self):
        db = rundb.DB()
        data = db.query_by_tag("test", page_num=1)
        # pprint(data)
        self.assertTrue(len(data) > 0)


if __name__ == "__main__":
    unittest.main()
