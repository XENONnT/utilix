import unittest

from utilix import *


class TestGet(unittest.TestCase):

    def test_doc_by_number(self):
        db = rundb.DB()
        data = db.get_doc(2000)
        # pprint(data)
        self.assertTrue("data" in data)
        self.assertTrue("start" in data)
        self.assertTrue("source" in data)

    def test_doc_by_number_str(self):
        db = rundb.DB()
        data = db.get_doc("2000")
        # pprint(data)
        self.assertTrue("data" in data)
        self.assertTrue("start" in data)
        self.assertTrue("source" in data)

    def test_doc_by_name(self):
        db = rundb.DB()
        data = db.get_doc("170917_1819")
        # pprint(data)
        self.assertTrue("data" in data)
        self.assertTrue("start" in data)
        self.assertTrue("source" in data)

    def test_data_by_number(self):
        db = rundb.DB()
        data = db.get_data(2000)
        # pprint(data)
        self.assertTrue(len(data) > 0)

    def test_data_by_name(self):
        db = rundb.DB()
        data = db.get_data("170917_1819")
        # pprint(data)
        self.assertTrue(len(data) > 0)


if __name__ == "__main__":
    unittest.main()
