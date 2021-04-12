__version__ = '0.5.0'

from warnings import warn
import logging


try:
    from utilix.config import Config
    uconfig = Config()

    logger = logging.getLogger("utilix")
    ch = logging.StreamHandler()
    ch.setLevel(uconfig.logging_level)
    logger.setLevel(uconfig.logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

except FileNotFoundError as e:
    uconfig = None
    warn(f'Utilix cannot find config file:\n {e}\nWithout it, you cannot '
         f'access the database. See https://github.com/XENONnT/utilix.')

if uconfig is not None and uconfig.getboolean('utilix', 'initialize_db_on_import',
                                              fallback=True):
    from utilix.rundb import DB
    db = DB()
else:
    print("Warning: DB class NOT initialized on import. You cannot do `from utilix import db`")
    print("If you want to initialize automatically on import, add the following to your utilix config:\n\n"
          "[utilix]\n"
          "initialize_db_on_import=true\n")

from . import mongo_files
