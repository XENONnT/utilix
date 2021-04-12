__version__ = '0.4.1'
# instantiate here so we just do it once
from warnings import warn

try:
    from utilix.config import Config
    uconfig = Config()
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
