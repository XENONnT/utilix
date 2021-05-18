__version__ = '0.6.0'

from . import config

# try loading config, if it doesn't work then set uconfig to None
# this is needed so that strax(en) CI  tests will work even without a config file
uconfig = config.Config()

if uconfig.is_configured:
    logger = config.setup_logger(uconfig.logging_level)

else:
    uconfig = None
    logger = config.setup_logger()

from .rundb import DB, xent_collection, xe1t_collection
from .mongo_files import MongoUploader, MongoDownloader, APIUploader, APIDownloader

