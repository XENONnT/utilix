import os
import tempfile
from datetime import datetime
from warnings import warn
import pytz
from typing import Tuple, Dict, Any, List, Optional, Union
import gridfs
from tqdm import tqdm
from shutil import move
import hashlib
from pymongo.collection import Collection
from utilix.rundb import DB, xent_collection
from utilix.utils import to_str_tuple
from utilix import uconfig, logger


class GridFsBase:
    """Base class for GridFS operations."""

    def __init__(self, config_identifier: str = "config_name", **kwargs: Any) -> None:
        self.config_identifier = config_identifier

    def get_query_config(self, config: str) -> Dict[str, str]:
        """Generate query identifier for a config."""
        return {self.config_identifier: config}

    def document_format(self, config):
        """Format of the document to upload.

        :param config: str, name of the file of interest
        :return: dict, that will be used to add the document

        """
        doc = self.get_query_config(config)
        doc.update(
            {
                "added": datetime.now(tz=pytz.utc),
            }
        )
        return doc

    def config_exists(self, config: str) -> bool:
        """Check if a config exists."""
        raise NotImplementedError

    def md5_stored(self, abs_path: str) -> bool:
        """Check if file with given MD5 is stored."""
        raise NotImplementedError

    def test_find(self) -> None:
        """Test the find operation."""
        raise NotImplementedError

    def list_files(self) -> List[str]:
        """List all files in the database."""
        raise NotImplementedError

    @staticmethod
    def compute_md5(abs_path: str) -> str:
        """Compute MD5 hash of a file.

        RAM intensive operation.

        """
        if not os.path.exists(abs_path):
            return ""
        # bandit: disable=B303
        hash_md5 = hashlib.md5()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class GridFsInterfaceMongo(GridFsBase):
    """
    Class to upload/download the files to a database using GridFS
    for PyMongo:
    https://pymongo.readthedocs.io/en/stable/api/gridfs/index.html#module-gridfs

    This class does the basic shared initiation of the downloader and
    uploader classes.

    """

    def __init__(
        self,
        readonly=True,
        file_database="files",
        config_identifier="config_name",
        collection=None,
        _test_on_init=False,
    ):
        """GridFsInterface.

        :param readonly: bool, can one read or also write to the
            database.
        :param file_database: str, name of the database. Default should
            not be changed.
        :param config_identifier: str, header of the files that are
            saved in Gridfs
        :param collection: pymongo.collection.Collection, (Optional)
            PyMongo DataName Collection to bypass normal initiation
            using utilix. Should be an object of the form:
                pymongo.MongoClient(..).DATABASE_NAME.COLLECTION_NAME
        :param _test_on_init: Test if the collection is empty on init
            (only deactivate if you are using a brand new database)!

        """

        if collection is None:
            if not readonly:
                # We want admin access to start writing data!
                mongo_url = uconfig.get("rundb_admin", "mongo_rdb_url")
                mongo_user = uconfig.get("rundb_admin", "mongo_rdb_username")
                mongo_password = uconfig.get("rundb_admin", "mongo_rdb_password")
            else:
                # We can safely use the Utilix defaults
                mongo_url = mongo_user = mongo_password = None

            # If no collection arg is passed, it defaults to the 'files'
            # collection, see for more details:
            # https://github.com/XENONnT/utilix/blob/master/utilix/rundb.py
            mongo_kwargs = {
                "url": mongo_url,
                "user": mongo_user,
                "password": mongo_password,
                "database": file_database,
            }
            # We can safely hard-code the collection as that is always
            # the same with GridFS.
            collection = xent_collection(**mongo_kwargs, collection="fs.files")
        else:
            # Check the user input is fine for what we want to do.
            if not isinstance(collection, Collection):
                raise ValueError("Provide PyMongo collection (see docstring)!")
            if file_database is not None:
                raise ValueError("Already provided a collection!")

        # Set collection and make sure it can at least do a 'find' operation
        self.collection = collection
        if _test_on_init:
            self.test_find()

        # This is the identifier under which we store the files.
        self.config_identifier = config_identifier

        # The GridFS used in this database
        self.grid_fs = gridfs.GridFS(collection.database)

    def get_query_config(self, config):
        """Generate identifier to query against.

        This is just the configs name.
        :param config: str, name of the file of interest
        :return: dict, that can be used in queries

        """
        return {self.config_identifier: config}

    def config_exists(self, config):
        """Quick check if this config is already saved in the collection.

        :param config: str, name of the file of interest
        :return: bool, is this config name stored in the database

        """
        query = self.get_query_config(config)
        return self.collection.count_documents(query) > 0

    def md5_stored(self, abs_path):
        """
        NB: RAM intensive operation!
        Carefully compare if the MD5 identifier is the same as the file
        as stored under abs_path.

        :param abs_path: str, absolute path to the file name
        :return: bool, returns if the exact same file is already stored
            in the database

        """
        if not os.path.exists(abs_path):
            # A file that does not exist does not have the same MD5
            return False
        query = {"md5": self.compute_md5(abs_path)}
        return self.collection.count_documents(query) > 0

    def test_find(self):
        """Test the connection to the self.collection to see if we can perform a collection.find
        operation."""
        if self.collection.find_one(projection="_id") is None:
            raise ConnectionError("Could not find any data in this collection")

    def list_files(self):
        """Get a complete list of files that are stored in the database.

        :return: list, list of the names of the items stored in this database

        """
        return [
            doc[self.config_identifier]
            for doc in self.collection.find(projection={self.config_identifier: 1})
            if self.config_identifier in doc
        ]

    @staticmethod
    def compute_md5(abs_path):
        """
        NB: RAM intensive operation!
        Get the md5 hash of a file stored under abs_path

        :param abs_path: str, absolute path to a file
        :return: str, the md5-hash of the requested file
        """
        # This function is copied from:
        # stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file

        if not os.path.exists(abs_path):
            # if there is no file, there is nothing to compute
            return ""
        # Also, disable all the use of insecure MD2, MD4, MD5, or SHA1
        # hash function violations in this function.
        # disable bandit
        hash_md5 = hashlib.md5()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class MongoUploader(GridFsInterfaceMongo):
    """Class to upload files to GridFs."""

    def __init__(self, readonly=False, *args, **kwargs):
        # Same as parent. Just check the readonly_argument
        if readonly:
            raise PermissionError("How can you upload if you want to operate in readonly?")
        super().__init__(*args, readonly=readonly, **kwargs)

    def upload_from_dict(self, file_path_dict):
        """Upload all files in the dictionary to the database.

        :param file_path_dict: dict, dictionary of paths to upload. The
            dict should be of the format:
            file_path_dict = {'config_name': '/the_config_path', ...}

        :return: None

        """
        if not isinstance(file_path_dict, dict):
            raise ValueError(
                "file_path_dict must be dict of form "
                '"dict(NAME=ABSOLUTE_PATH,...)". Got '
                f"{type(file_path_dict)} instead"
            )

        for config, abs_path in tqdm(file_path_dict.items()):
            # We need to do this expensive check here. It is not enough
            # to just check that the file is stored under the
            # 'config_identifier'. What if the file changed? Then we
            # want to upload a new file! Otherwise we could have done
            # the self.config_exists-query. If it turns out we have the
            # exact same file, forget about uploading it.
            if self.config_exists(config) and self.md5_stored(abs_path):
                continue
            else:
                # This means we are going to upload the file because its
                # not stored yet.
                try:
                    self.upload_single(config, abs_path)
                except (CouldNotLoadError, ConfigTooLargeError):
                    # Perhaps we should fail then?
                    warn(f"Cannot upload {config}")

    def upload_single(self, config, abs_path):
        """Upload a single file to gridfs.

        :param config: str, the name under which this file should be stored
        :param abs_path: str, the absolute path of the file

        """
        doc = self.document_format(config)
        doc["md5"] = self.compute_md5(abs_path)
        if not os.path.exists(abs_path):
            raise CouldNotLoadError(f"{abs_path} does not exits")

        print(f"uploading {config}")
        with open(abs_path, "rb") as file:
            self.grid_fs.put(file, **doc)


class MongoDownloader(GridFsInterfaceMongo):
    """Class to download files from GridFs."""

    _instances: Dict[Tuple, "MongoDownloader"] = {}
    _initialized: Dict[Tuple, bool] = {}

    def __new__(cls, *args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super(MongoDownloader, cls).__new__(cls)
            cls._initialized[key] = False
        return cls._instances[key]

    def __init__(self, *args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if not self._initialized[key]:
            self._instances[key].initialize(*args, **kwargs)
            self._initialized[key] = True
        return

    def initialize(self, store_files_at=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We are going to set a place where to store the files. It's
        # either specified by the user or we use these defaults:
        if store_files_at is None:
            store_files_at = (
                "./resource_cache",
                "/tmp/straxen_resource_cache",
            )
        elif not isinstance(store_files_at, (tuple, str, list)):
            raise ValueError(f"{store_files_at} should be tuple of paths!")
        elif isinstance(store_files_at, str):
            store_files_at = to_str_tuple(store_files_at)

        self.storage_options = store_files_at

    def download_single(self, config_name: str, human_readable_file_name=False):
        """Download the config_name if it exists.

        :param config_name: str, the name under which the file is stored
        :param human_readable_file_name: bool, store the file also under it's human readable name.
            It is better not to use this as the user might not know if the version of the file is
            the latest.
        :return: str, the absolute path of the file requested

        """
        if self.config_exists(config_name):
            # Query by name
            query = self.get_query_config(config_name)
            try:
                # This could return multiple since we upload files if
                # they have changed again! Therefore just take the last.
                fs_object = self.grid_fs.get_last_version(**query)
            except gridfs.NoFile as e:
                raise CouldNotLoadError(f"{config_name} cannot be downloaded from GridFs") from e

            # Ok, so we can open it. We will store the file under it's
            # md5-hash as that allows to easily compare if we already
            # have the correct file.
            if human_readable_file_name:
                target_file_name = config_name
            else:
                target_file_name = fs_object.md5

            if not human_readable_file_name:
                for cache_folder in self.storage_options:
                    possible_path = os.path.join(cache_folder, target_file_name)
                    if os.path.exists(possible_path):
                        # Great! This already exists. Let's just return
                        # where it is stored.
                        return possible_path

            # Apparently the file does not exist, let's find a place to
            # store the file and download it.
            store_files_at = self._check_store_files_at(self.storage_options)
            destination_path = os.path.join(store_files_at, target_file_name)

            # Let's open a temporary directory, download the file, and
            # try moving it to the destination_path. This prevents
            # simultaneous writes of the same file.
            with tempfile.TemporaryDirectory() as temp_directory_name:
                temp_path = os.path.join(temp_directory_name, target_file_name)

                with open(temp_path, "wb") as stored_file:
                    # This is were we do the actual downloading!
                    warn(f"Downloading {config_name} to {destination_path}", DownloadWarning)
                    stored_file.write(fs_object.read())

                if not os.path.exists(destination_path):
                    # Move the file to the place we want to store it.
                    move(temp_path, destination_path)
            return destination_path

        else:
            raise ValueError(f"Config {config_name} cannot be downloaded since it is not stored")

    def get_abs_path(self, config_name):
        return self.download_single(config_name)

    def download_all(self):
        """Download all the files that are stored in the mongo collection."""
        for config in self.list_files():
            print(config, self.download_single(config))

    @staticmethod
    def _check_store_files_at(cache_folder_alternatives):
        """Iterate over the options in cache_options until we find a folder where we can store data.
        Order does matter as we iterate until we find one folder that is willing.

        :param cache_folder_alternatives: tuple, this tuple must be a list of paths one can try to
            store the downloaded data
        :return: str, the folder that we can write to.

        """
        if not isinstance(cache_folder_alternatives, (tuple, list)):
            raise ValueError("cache_folder_alternatives must be tuple")
        for folder in cache_folder_alternatives:
            if not os.path.exists(folder):
                try:
                    os.makedirs(folder)
                except (PermissionError, OSError):
                    continue
            if os.access(folder, os.W_OK):
                return folder
        raise PermissionError(
            f"Cannot write to any of the cache_folder_alternatives: {cache_folder_alternatives}"
        )


class GridFsInterfaceAPI(GridFsBase):
    """Interface to gridfs using the runDB API."""

    def __init__(self, config_identifier: str = "config_name") -> None:
        super().__init__(config_identifier=config_identifier)
        self.db = DB()

    def config_exists(self, config: str) -> bool:
        """Check if config is saved in the collection."""
        query = self.get_query_config(config)
        return self.db.count_files(query) > 0

    def md5_stored(self, abs_path: str) -> bool:
        """Check if file with same MD5 is stored.

        RAM intensive.

        """
        if not os.path.exists(abs_path):
            return False
        query = {"md5": self.compute_md5(abs_path)}
        return self.db.count_files(query) > 0

    def test_find(self) -> None:
        """Test the connection to the collection."""
        if self.db.get_files({}, projection={"_id": 1}) is None:
            raise ConnectionError("Could not find any data in this collection")

    def list_files(self) -> List[str]:
        """Get list of files stored in the database."""
        return [
            doc[self.config_identifier]
            for doc in self.db.get_files({}, projection={self.config_identifier: 1})
            if self.config_identifier in doc
        ]


class APIUploader(GridFsInterfaceAPI):
    """Upload files to gridfs using the runDB API."""

    def __init__(self, config_identifier: str = "config_name") -> None:
        super().__init__(config_identifier=config_identifier)

    def upload_single(self, config: str, abs_path: str) -> None:
        """Upload a single file to gridfs.

        :param config: str, the name under which this file should be stored
        :param abs_path: str, the absolute path of the file

        """
        if not os.path.exists(abs_path):
            raise CouldNotLoadError(f"{abs_path} does not exist")

        logger.info(f"uploading file {config} from {abs_path}")
        self.db.upload_file(abs_path, config)


class APIDownloader(GridFsInterfaceAPI):
    """Download files from gridfs using the runDB API."""

    _instances: Dict[Tuple, "APIDownloader"] = {}
    _initialized: Dict[Tuple, bool] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> "APIDownloader":
        key = (args, frozenset(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super(APIDownloader, cls).__new__(cls)
            cls._initialized[key] = False
        return cls._instances[key]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        key = (args, frozenset(kwargs.items()))
        if not self._initialized[key]:
            self._instances[key].initialize(*args, **kwargs)
            self._initialized[key] = True

    def initialize(
        self,
        config_identifier: str = "config_name",
        store_files_at: Optional[Union[str, Tuple[str, ...], List[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_identifier=config_identifier)

        if store_files_at is None:
            store_files_at = (
                "./resource_cache",
                "/tmp/straxen_resource_cache",
            )
        elif isinstance(store_files_at, str):
            store_files_at = (store_files_at,)
        elif isinstance(store_files_at, list):
            store_files_at = tuple(store_files_at)
        elif not isinstance(store_files_at, tuple):
            raise ValueError(f"{store_files_at} should be a string, list, or tuple of paths!")

        self.storage_options: Tuple[str, ...] = store_files_at

    def download_single(
        self,
        config_name: str,
        write_to: Optional[str] = None,
        human_readable_file_name: bool = False,
    ) -> str:
        """Download the config_name if it exists."""
        target_file_name = (
            config_name if human_readable_file_name else self.db.get_file_md5(config_name)
        )

        # check if self.storage_options is None or empty
        if not self.storage_options:
            raise ValueError("No storage options available")

        if write_to is None:
            if not human_readable_file_name:
                for cache_folder in self.storage_options:
                    possible_path = os.path.join(cache_folder, target_file_name)
                    if os.path.exists(possible_path):
                        return possible_path

            store_files_at = self._check_store_files_at(self.storage_options)
        else:
            store_files_at = write_to

        # make sure store_files_at is a string
        if not isinstance(store_files_at, str):
            raise TypeError(f"Expected string for store_files_at, got {type(store_files_at)}")

        destination_path = os.path.join(store_files_at, target_file_name)

        with tempfile.TemporaryDirectory() as temp_directory_name:
            temp_path = self.db.download_file(config_name, save_dir=temp_directory_name)
            if not os.path.exists(destination_path):
                move(temp_path, destination_path)
            else:
                warn(f"File {destination_path} already exists. Not overwriting.")
        return destination_path

    def _check_store_files_at(self, options: Union[str, Tuple[str, ...]]) -> str:
        """Check and return a valid storage location."""
        if isinstance(options, str):
            return options
        for option in options:
            if os.path.isdir(option):
                return option
        raise ValueError("No valid storage location found")


class DownloadWarning(UserWarning):
    pass


class CouldNotLoadError(Exception):
    """Raise if we cannot load this kind of data."""

    # Disable the inspection of 'Unnecessary pass statement'
    # pylint: disable=unnecessary-pass
    pass


class ConfigTooLargeError(Exception):
    """Raise if the data is to large to be uploaded into mongo."""

    # Disable the inspection of 'Unnecessary pass statement'
    # pylint: disable=unnecessary-pass
    pass
