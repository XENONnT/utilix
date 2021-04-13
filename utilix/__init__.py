__version__ = '0.5.1'

from warnings import warn
import logging

def _setup_logger(logging_level):
    logger = logging.getLogger("utilix")
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

from utilix.config import Config
uconfig = Config()
logger = _setup_logger(uconfig.logging_level)

if uconfig.config_path is not None:
    if uconfig.getboolean('utilix', 'initialize_db_on_import', fallback=True):
        from utilix.rundb import DB
        db = DB()

    else:
        print("Warning: DB class NOT initialized on import. You cannot do `from utilix import db`")
        print("If you want to initialize automatically on import, add the following to your utilix config:\n\n"
          "[utilix]\n"
          "initialize_db_on_import=true\n")

from . import mongo_files
