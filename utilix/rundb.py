import os
import requests
import re
import json
import datetime
import logging
import pymongo
from warnings import warn
import time

from . import uconfig, io
from .config import setup_logger


# Config the logger:
if uconfig is not None:
    PREFIX = uconfig.get('RunDB', 'rundb_api_url', fallback=None)
    BASE_HEADERS = {'Content-Type': "application/json", 'Cache-Control': "no-cache"}
    logger = setup_logger(uconfig.logging_level)

else:
    logger = setup_logger()


class NewTokenError(Exception):
    pass


class APIError(Exception):
    pass


def Responder(func):
    def func_wrapper(*args, **kwargs):
        st = func(*args, **kwargs)
        if st.status_code != 200:
            logger.error("\n\tAPI Call was {0}\n\tReturn code: {1}\n\tReason: {2} ".format(
                args[1],
                st.status_code,
                st.text,
            ))

            if st.status_code == 401:
                raise APIError(
                    "Error 401 is an authentication error. This is likely an issue with your token. "
                    "Can you do 'rm ~/.dbtoken' and try again? ")
            else:
                raise APIError("API called failed")
        return st

    return func_wrapper


class Token:
    """
    Object handling tokens for runDB API access.

    """
    token_string = None
    user = None
    creation_time = None

    def __init__(self, path):
        self.path = path

        # if token path exists, read it in. Otherwise make a new one
        if os.path.exists(path):
            logger.debug(f'Token exists at {path}')
            with open(path) as f:
                try:
                    json_in = json.load(f)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f'Cannot open {path}, please report to https://github.com/XENONnT/utilix/issues. '\
                        f'To continue do "rm {path}" and restart notebook/utilix') from e
                self.token_string = json_in['string']
                self.creation_time = json_in['creation_time']
            # some old token files might not have the user field
            if 'user' in json_in:
                self.user = json_in['user']
            # if not, make a new token
            else:
                logger.debug(f'Creating new token')
                self.new_token()
        else:
            logger.debug(f'No token exists at {path}. Creating new one.')
            self.new_token()

        # check if the user in the token matches the user in the config
        if self.user != uconfig.get('RunDB', 'rundb_api_user'):
            logger.info(
                f"Username in {uconfig.config_path} does not match token. Overwriting the token.")
            self.new_token()

        # refresh if needed
        if not self.is_valid:
            self.new_token()
            #self.refresh()
        else:
            logger.debug("Token is valid.")

    def __call__(self):
        return self.token_string

    def new_token(self):
        path = PREFIX + "/login"
        username = uconfig.get('RunDB', 'rundb_api_user')
        pw = uconfig.get('RunDB', 'rundb_api_password')
        data = json.dumps({"username": username,
                           "password": pw})
        logger.debug('Creating a new token: doing API call now')
        # try making a new token 3 times
        success = False
        for _try in range(3):
            try:
                response = requests.post(path, data=data, headers=BASE_HEADERS)
                response_json = json.loads(response.text)
                success = True
                break
            except json.decoder.JSONDecodeError:
                logger.info(f"Login attempt #{_try+1} failed. "
                            f"Sleeping for {10**_try} seconds and trying again.")
                time.sleep(10**_try)

        if not success:
            raise NewTokenError("Error in creating a token.")


        logger.debug(f'The response contains these keys: {list(response_json.keys())}')
        token = response_json.get('access_token', 'CALL_FAILED')
        if token == 'CALL_FAILED':
            logging.error(
                f"API call to create new token failed. Here is the response:\n{response.text}")
            raise RuntimeError("Creating a new token failed.")
        self.token_string = token
        self.user = username
        self.creation_time = datetime.datetime.now().timestamp()
        self.write()

    @property
    def is_valid(self):
        # TODO do an API call for this instead?
        diff = datetime.datetime.now().timestamp() - self.creation_time
        return diff < 24 * 60 * 60

    @property
    def json(self):
        return dict(string=self.token_string, creation_time=self.creation_time, user=self.user)

    def refresh(self):
        # update the token string
        url = PREFIX + "/refresh"
        headers = BASE_HEADERS.copy()
        headers['Authorization'] = f"Bearer {self.token_string}"
        logger.debug(f"Refreshing your token with API call {url}")
        response = requests.get(url, headers=headers)
        response_json = json.loads(response.text)
        logger.debug(f'The response contains these keys: {list(response_json.keys())}')
        # if renew fails, try logging back in
        if response.status_code != 200:
            if response_json['error'] != 'EarlyRefreshError':
                logger.warning("Refreshing token failed for some reason, so making a  new one")
                self.new_token()
                self.creation_time = datetime.datetime.now().timestamp()
                logger.debug("Token refreshed")
        else:
            self.creation_time = datetime.datetime.now().timestamp()
        self.write()

    def write(self):
        logger.debug(f"Dumping token to disk at {self.path}.")
        with open(self.path, "w") as f:
            json.dump(self.json, f)


class DB():
    """Wrapper around the RunDB API"""

    def __init__(self, token_path=None):

        if token_path is None:
            if 'HOME' not in os.environ:
                logger.error('$HOME is not defined in the environment')
                if 'USERPROFILE' in os.environ:
                    # Are you on windows?
                    token_path = os.path.join(os.environ['USERPROFILE'], '.dbtoken')
            else:
                token_path = os.path.join(os.environ['HOME'], ".dbtoken")

        # Takes a path to serialized token object
        token = Token(token_path)

        self.headers = BASE_HEADERS.copy()
        self.headers['Authorization'] = "Bearer {token}".format(token=token())

    # Helper:
    @Responder
    def _get(self, url):
        return requests.get(PREFIX + url, headers=self.headers)

    @Responder
    def _put(self, url, data):
        return requests.put(PREFIX + url, data=data, headers=self.headers)

    @Responder
    def _post(self, url, data):
        return requests.post(PREFIX + url, data=data, headers=self.headers)

    @Responder
    def _delete(self, url, data):
        return requests.delete(PREFIX + url, data=data, headers=self.headers)

    def _is_run_number(self, identifier):
        '''
        Takes a string and classifies it as a run number (as opposed to a
        run name)
        '''
        if re.search('^[0-9]+$', identifier):
            return True
        return False

    def _get_from_results(self, name_or_number, key):
        url = "/runs/number/{name_or_number}/filter/detector".format(name_or_number=name_or_number)
        response = json.loads(self._get(url).text)
        if (response is None
                or 'results' not in response
                or key not in response['results']):
            logger.warning(f'Cannot get {name_or_number} from {url}')
        else:
            return response['results'][key]

    def get_name(self, name):
        return self._get_from_results(name, 'name')

    def get_number(self, number):
        return self._get_from_results(number, 'number')

    def get_did(self, identifier, type='raw_records'):
        doc = self.get_doc(identifier)
        for d in doc['data']:
            if not ('host' in d and 'type' in d and 'did' in d):
                # This ddoc is not in the format of rucio
                continue
            if d['host'] == 'rucio-catalogue' and d['type'] == type:
                return d['did']
        raise ValueError(f'No {identifier} for {type}')

    def get_doc(self, identifier):
        '''
        Retrieves a document from the database. The identifier
        could be a run number of run name - the disambiguation
        takes place automatically.
        '''

        # map from all kinds of types (int, np int, ...)
        identifier = str(identifier)

        url = '/runs/name/{num}'.format(num=identifier)
        if self._is_run_number(identifier):
            url = '/runs/number/{num}'.format(num=identifier)
        # TODO what should be default
        return json.loads(self._get(url).text).get('results', None)

    def get_data(self, identifier, **filters):
        '''
        Retrieves the data portion of a document from the
        database. The identifier could be a run number of
        run name - the disambiguation takes place
        automatically.
        '''

        # map from all kinds of types (int, np int, ...)
        identifier = str(identifier)

        url = '/runs/name/{num}/data'.format(num=identifier)
        if self._is_run_number(identifier):
            url = '/runs/number/{num}/data'.format(num=identifier)

        data = json.loads(self._get(url).text).get('results', {})
        if 'data' not in data:
            raise RuntimeError('The requested document does not have a data key/value')

        data = data['data']

        ret = []
        for d in data:
            passes_filter = True
            for key, val in filters.items():
                if d.get(key) != val:
                    passes_filter = False
            if passes_filter:
                ret.append(d)

        return ret

    def update_data(self, identifier, datum):
        '''
        Updates a data entry. Identifier can be run number of name.
        '''

        datum = cleanup_datadict(datum)

        # map from all kinds of types (int, np int, ...)
        identifier = str(identifier)

        datum = json.dumps(datum)

        url = '/run/name/{num}/data/'.format(num=identifier)
        if self._is_run_number(identifier):
            url = '/run/number/{num}/data/'.format(num=identifier)

        return self._post(url, data=datum)

    def delete_data(self, identifier, datum):
        '''
        Updates a datum for a document with a matching identifier
        (name or run number)
        '''

        # map from all kinds of types (int, np int, ...)
        identifier = str(identifier)

        datum = json.dumps(datum)

        url = '/run/name/{num}/data/'.format(num=identifier)
        if self._is_run_number(identifier):
            url = '/run/number/{num}/data/'.format(num=identifier)

        return self._delete(url, data=datum)

    def query(self, page_num):
        url = '/runs/page/{page_num}'.format(page_num=page_num)
        response = json.loads(self._get(url).text)
        return response.get('results', {})

    def query_by_source(self, source, page_num):
        url = '/runs/source/{source}/page/{page_num}'.format(source=source, page_num=page_num)
        response = json.loads(self._get(url).text)
        return response.get('results', {})

    def query_by_tag(self, tag, page_num):
        url = '/runs/tag/{tag}/page/{page_num}'.format(tag=tag, page_num=page_num)
        response = json.loads(self._get(url).text)
        return response.get('results', {})

    def get_hash(self, context, datatype, straxen_version):
        if '.' in straxen_version:
            straxen_version = straxen_version.replace('.', '_')
        url = '/contexts/{straxen_version}/{context}/{dtype}'.format(context=context,
                                                                     dtype=datatype,
                                                                     straxen_version=straxen_version)
        response = json.loads(self._get(url).text)
        return response.get('results', {})

    def update_context_collection(self, data):
        context = data.get('name')
        straxen_version = data.get('straxen_version')
        straxen_version = straxen_version.replace('.', '_')
        url = '/contexts/{straxen_version}/{context}/'.format(context=context,
                                                              straxen_version=straxen_version)
        data['date_added'] = data['date_added'].isoformat()
        response = json.loads(self._post(url, data=json.dumps(data)).text)
        return response.get('results', {})

    def delete_context_collection(self, context, straxen_version):
        straxen_version = straxen_version.replace('.', '_')
        url = '/contexts/{straxen_version}/{context}/'.format(context=context,
                                                              straxen_version=straxen_version)
        response = json.loads(self._delete(url, data=None).text)
        return response.get('results', {})

    def get_context(self, context, straxen_version):
        straxen_version = straxen_version.replace('.', '_')
        url = '/contexts/{straxen_version}/{context}/'.format(context=context,
                                                              straxen_version=straxen_version)
        response = json.loads(self._get(url).text)
        return response.get('results', {})

    def get_rses(self, run_number, dtype, hash):
        data = self.get_data(run_number)
        rses = []
        for d in data:
            assert 'host' in d and 'type' in d, (
                f"invalid data-doc retrieved for {run_number} {dtype} {hash}")
            # Did is only in rucio-cataloge, hence don't ask for it to
            # be in all docs in data
            if (d['host'] == "rucio-catalogue" and d['type'] == dtype and
                    hash in d['did'] and d['status'] == 'transferred'):
                rses.append(d['location'])

        return rses

    # TODO
    def get_all_contexts(self):
        """Loads all contexts"""
        raise NotImplementedError

    # TODO
    def get_context_info(self, dtype, strax_hash):
        """Returns context name and strax(en) versions for a given dtype and hash"""
        raise NotImplementedError

    def get_mc_documents(self):
        '''
        Returns all MC documents.
        '''
        url = '/mc/documents/'
        return self._get(url)

    def add_mc_document(self, document):
        '''
        Adds a document to the MC database.
        '''
        doc = json.dumps(document)
        url = '/mc/documents/'
        return self._post(url, data=doc)

    def delete_mc_document(self, document):
        '''
        Deletes a document from the MC database. The document must be passed exactly.
        '''
        doc = json.dumps(document)
        url = '/mc/documents/'
        return self._delete(url, data=doc)

    def download_file(self, filename, save_dir='./', force=False):
        """Downloads file from GridFS"""
        url = f'/files/{filename}'
        os.makedirs(save_dir, exist_ok=True)
        write_to = os.path.join(save_dir, filename)
        if os.path.exists(write_to) and not force:
            logger.debug(f"{filename} already exists at {write_to} and the 'force' flag is not set.")
        else:
            logger.debug(f"Downloading {filename} from gridfs...")
            response = self._get(url)
            with open(write_to, 'wb') as f:
                f.write(response.content)
            logger.debug(f'DONE. {filename} downloaded to {write_to}')
        return write_to

    def load_file(self, filename, save_dir=None, force=False):
        if save_dir is None:
            save_dir = os.path.join(os.environ.get("HOME"), '.gridfs_cache')
        path = self.download_file(filename, save_dir=save_dir, force=force)
        return io.read_file(path)

    def upload_file(self, filepath, filename=None):
        with open(filepath, 'rb') as f:
            fb = f.read()
        # if no specific filename passed, just get it from the path
        if not filename:
            filename = os.path.basename(filepath)
        url = f'/files/{filename}'
        return self._post(url, data=fb)

    def get_files(self, query: dict, projection=None):
        """Do a general query on the fs.files collection"""
        url = '/files/query'
        # the projection needs to be a dict for the flask app
        if projection is None:
            projection = {}
        data = json.dumps(dict(query=query, projection=projection))
        result = self._post(url, data=data)
        response = result.json()
        return response.get('results', [])

    def count_files(self, query: dict)->int:
        """Perform colection.count_documents on the fs.files-collection using the query"""
        """<URL MAGIC>"""
        docs = self.get_files(query)
        return len(docs)

    def delete_file(self, filename):
        resp = input(f"HUGE GIGANTIC CRITICAL WARNING: this will delete all files of the name {filename} in GridFS. "
                     "Confirm by typing again the name of the file you want to delete: \n")
        if resp != filename:
            print("Probably the safe choice. Exiting.")
            return
        url = f'/files/{filename}'
        return self._delete(url, data="")

    def get_file_md5(self, filename):
        url = f"/files/{filename}/md5"
        response = self._get(url).json()
        return response['results']



class PyMongoCannotConnect(Exception):
    """Raise error when we cannot connect to the pymongo client"""
    pass


def test_collection(collection, url, raise_errors=False):
    """
    Warn user if client can be troublesome if read preference is not specified
    :param collection: pymongo client
    :param url: the mongo url we are testing (for the error message)
    :param raise_errors: if False (default) warn, otherwise raise an error
    """
    try:
        # test the collection by doing a light query
        collection.find_one({}, {'_id': 1})
    except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.OperationFailure) as e:
        # This happens when trying to connect to one or more mirrors
        # where we cannot decide on who is primary
        message = (
            f'Cannot get server info from "{url}". Check your config at {uconfig.config_path}')
        if not raise_errors:
            warn(message)
        else:
            message += (
                'This usually happens when trying to connect to multiple '
                'mirrors when they cannot decide which is primary. Also see:\n'
                'https://github.com/XENONnT/straxen/pull/163#issuecomment-732031099')
            raise PyMongoCannotConnect(message) from e


def pymongo_collection(collection='runs', **kwargs):
    # default collection is the XENONnT runsDB
    # for 1T, pass collection='runs_new'
    print("WARNING: pymongo_collection is deprecated. Please use xent_collection or xe1t_collection instead")
    uri = 'mongodb://{user}:{pw}@{url}'
    url = kwargs.get('url')
    user = kwargs.get('user')
    pw = kwargs.get('password')
    database = kwargs.get('database')

    if not url:
        url = uconfig.get('RunDB', 'pymongo_url')
    if not user:
        user = uconfig.get('RunDB', 'pymongo_user')
    if not pw:
        pw = uconfig.get('RunDB', 'pymongo_password')
    if not database:
        database = uconfig.get('RunDB', 'pymongo_database')
    uri = uri.format(user=user, pw=pw, url=url)
    c = pymongo.MongoClient(uri, readPreference='secondaryPreferred')
    DB = c[database]
    coll = DB[collection]
    # Checkout the collection we are returning and raise errors if you want
    # to be realy sure we can use this URL.
    # test_collection(coll, url, raise_errors=False)

    return coll


def _collection(experiment, collection, url=None, user=None, password=None, database=None):
    if experiment not in ['xe1t', 'xent']:
        raise ValueError(f"experiment must be 'xe1t' or 'xent'. You passed f{experiment}")

    if not url:
        url = uconfig.get('RunDB', f'{experiment}_url')
    if not user:
        user = uconfig.get('RunDB', f'{experiment}_user')
    if not password:
        password = uconfig.get('RunDB', f'{experiment}_password')
    if not database:
        database = uconfig.get('RunDB', f'{experiment}_database')

    uri = f"mongodb://{user}:{password}@{url}"
    c = pymongo.MongoClient(uri, readPreference='secondaryPreferred')
    DB = c[database]
    coll = DB[collection]
    return coll


def xent_collection(collection='runs', **kwargs):
    return _collection('xent', collection, **kwargs)


def xe1t_collection(collection='runs_new', **kwargs):
    return _collection('xe1t', collection, **kwargs)


def cleanup_datadict(ddict):
    good_keys = ['creation_time', 'host', 'location',
                 'type', 'status', 'meta', 'did', 'protocol']
    new_dict = ddict.copy()

    for key in ddict:
        if key not in good_keys:
            new_dict.pop(key)

    return new_dict

