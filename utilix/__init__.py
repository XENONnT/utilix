__version__ = '0.4.1'
# instantiate here so we just do it once
from warnings import warn

try:
    from utilix.config import Config
    uconfig = Config()

    from utilix.rundb import DB
    db = DB()
except FileNotFoundError as e:
    uconfig = None
    warn(f'Utilix cannot find config file:\n {e}\nWithout it, you cannot '
         f'access the database. See https://github.com/XENONnT/utilix.')
