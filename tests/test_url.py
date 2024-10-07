import utilix
import pymongo

uconfig = utilix.uconfig


def test_mongo_urls_without_secondary_preferred(collection="runs", raise_errors=True, **kwargs):
    """Test that we can do a naive initiation of the rundb."""
    # Get the url for the config
    uri = "mongodb://{user}:{pw}@{url}"
    url = uconfig.get("RunDB", "pymongo_url")
    user = uconfig.get("RunDB", "pymongo_user")
    pw = uconfig.get("RunDB", "pymongo_password")
    database = uconfig.get("RunDB", "pymongo_database")
    uri = uri.format(user=user, pw=pw, url=url)

    # Do a naive client initialization
    c = pymongo.MongoClient(uri, serverSelectionTimeoutMS=500, **kwargs)
    DB = c[database]
    coll = DB[collection]

    # This test might fail! See the corresponding error message and:
    # https://github.com/XENONnT/utilix/pull/14. The point of this test
    # is to see if we can just plug the pymongo_url into a mongo client
    # (often not the case depending on the access to one or more of the
    # mirrors).
    # If CI is ever integrated to Utilix, we should disable the
    # raise_errors for this test.
    utilix.rundb.test_collection(coll, url, raise_errors=raise_errors)


def test_mongo_urls_secondary_preferred():
    """Do it the right way, use secondary preferred."""
    # Raise errors if this fails because then we have a big problem!
    test_mongo_urls_without_secondary_preferred(
        **dict(readPreference="secondaryPreferred"), raise_errors=True
    )
