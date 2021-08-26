__version__ = '0.6.1'

from . import config

# try loading config, if it doesn't work then set uconfig to None
# this is needed so that strax(en) CI  tests will work even without a config file
uconfig = config.Config()

if uconfig.is_configured:
    logger = config.setup_logger(uconfig.logging_level)

else:
    uconfig = None
    logger = config.setup_logger()

from .rundb import xent_collection, xe1t_collection, DBapi, DBmongo
from .mongo_files import MongoUploader, MongoDownloader, APIUploader, APIDownloader
from .rundoc_data import RunDocUpload, upload_doc_from_file

# initialize runDB instance if we can
if uconfig:
    if uconfig.get('RunDB', 'api_or_mongo', fallback='api') == 'api':
        db = DBapi()
    else:
        db = DBmongo()
