"""
Downloader and uploader for mongo files
https://straxen.readthedocs.io/en/latest/config_storage.html
https://github.com/XENONnT/straxen/blob/master/straxen/mongo_storage.py

This module contains 5 classes:
 - GridFsInterface: General Gridfs base class (not used)
 - MongoDownloader: subclass of GridFsInterface for downloading files
    using a MongoClient
 - MongoUploader: subclass of GridFsInterface for uploading files
    using a MongoClient
 - DownloadAPI: subclass of MongoDownloader where all the MongoClient
    interactions have been replaced by API calls.
 - UploadAPI: subclass of MongoUploader where all the MongoClient
    interactions have been replaced by API calls.
"""

import os
import tempfile
from datetime import datetime
from warnings import warn
import gridfs
from tqdm import tqdm
from shutil import move
import hashlib
from pymongo.collection import Collection
import typing as ty
import numpy as np
import pandas as pd

from . import uconfig, logger
from .rundb import xent_collection, DB


class GridFsBase:
    """Base class to upload/download the files to a database using GridFS.

    It is subclassed by a PyMongo version as well as one that goes through the runDB API

    """

    def __init__(self, config_identifier="config_name", **kwargs):
        # This is the identifier under which we store the files.
        self.config_identifier = config_identifier

    def get_query_config(self, config):
        """Generate identifier to query against. This is just the configs name.

        :param config: str, name of the file of interest
        :return: dict, that can be used in queries

        """
        return {self.config_identifier: config}

    def document_format(self, config):
        """Format of the document to upload.

        :param config: str, name of the file of interest
        :return: dict, that will be used to add the document

        """
        doc = self.get_query_config(config)
        doc.update({"added": datetime.utcnow()})
        return doc

    def config_exists(self, config):
        raise NotImplementedError

    def md5_stored(self, abs_path):
        raise NotImplementedError

    def test_find(self):
        raise NotImplementedError

    def list_files(self):
        raise NotImplementedError

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
        # Also, disable all the  Use of insecure MD2, MD4, MD5, or SHA1
        # hash function violations in this function.
        # bandit: disable=B303
        hash_md5 = hashlib.md5()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class GridFsUploadBase:
    """Base class for uploads."""

    def upload_single(self, config, abs_path):
        raise NotImplementedError

    def upload_from_dict(self, file_path_dict):
        raise NotImplementedError


class GridFsDownloadBase:
    """Base class for downloads."""

    def __init__(self, store_files_at=None):
        # We are going to set a place where to store the files. It's
        # either specified by the user or we use these defaults:
        if store_files_at is None:
            store_files_at = ("./resource_cache",)
        elif not isinstance(store_files_at, (tuple, str, list)):
            raise ValueError(f"{store_files_at} should be tuple of paths!")
        elif isinstance(store_files_at, str):
            store_files_at = to_str_tuple(store_files_at)

        self.storage_options = store_files_at

    def download_single(self, config_name: str, write_to=None, human_readable_file_name=False):
        raise NotImplementedError

    def download_all(self):
        """Download all the files that are stored in the mongo collection."""
        raise NotImplementedError("This feature is disabled for now")
        # Disable the inspection of `Unreachable code`
        # pylint: disable=unreachable
        # for config in self.list_files():
        #     self.download_single(config)

    @staticmethod
    def _check_store_files_at(cache_folder_alternatives):
        """Iterate over the options in cache_options until we find a folder where we can store data.
        Order does matter as we iterate until we find one folder that is willing.

        :param cache_folder_alternatives: tuple, this tuple must be a list of paths one can try to
            store the downloaded data.
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
            f"Cannot write to any of the cache_folder_alternatives: " f"{cache_folder_alternatives}"
        )


class GridFsInterfaceMongo(GridFsBase):
    """
    Base class to upload/download the files to a database using GridFS
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
    ):
        """GridFs-Mongo Interface.

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

        """
        super().__init__(config_identifier=config_identifier)
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
            collection = xent_collection(collection="fs.files", **mongo_kwargs)
        else:
            # Check the user input is fine for what we want to do.
            if not isinstance(collection, Collection):
                raise ValueError("Provide PyMongo collection (see docstring)!")
            assert file_database is None, "Already provided a collection!"

        # Set collection and make sure it can at least do a 'find' operation
        self.collection = collection
        self.test_find()

        # The GridFS used in this database
        self.grid_fs = gridfs.GridFS(collection.database)

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


class MongoUploader(GridFsInterfaceMongo, GridFsUploadBase):
    """Class to upload files to GridFs."""

    def __init__(self, readonly=False, *args, **kwargs):
        # Same as parent. Just check the readonly_argument
        if readonly:
            raise PermissionError("How can you upload if you want to operate in readonly?")
        GridFsInterfaceMongo.__init__(self, *args, readonly=readonly, **kwargs)

    def upload_from_dict(self, file_path_dict):
        """Upload all files in the dictionary to the database.

        :param file_path_dict: dict, dictionary of paths to upload. The
            dict should be of the format:
            file_path_dict = {'config_name':  '/the_config_path', ...}

        :return: None

        """
        if not isinstance(file_path_dict, dict):
            raise ValueError(
                f"file_path_dict must be dict of form "
                f'"dict(NAME=ABSOLUTE_PATH,...)". Got '
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


class MongoDownloader(GridFsInterfaceMongo, GridFsDownloadBase):
    """Class to download files from GridFs."""

    def __init__(self, *args, **kwargs):
        GridFsDownloadBase.__init__(self, *args)
        GridFsInterfaceMongo.__init__(self, **kwargs)

    def get_gridfs_object(
        self,
        config_name: str,
    ):
        if self.config_exists(config_name):
            # Query by name
            query = self.get_query_config(config_name)
            try:
                # This could return multiple since we upload files if
                # they have changed again! Therefore just take the last.
                fs_object = self.grid_fs.get_last_version(**query)
                return fs_object
            except gridfs.NoFile as e:
                raise CouldNotLoadError(f"{config_name} cannot be downloaded from GridFs") from e
        else:
            raise ValueError(
                f"Config {config_name} cannot be downloaded " f"since it is not stored"
            )

    def open_response(self, fs_object):
        """Open the fs_object for writing to file."""
        return fs_object.read()

    def download_single(self, config_name: str, write_to=None, human_readable_file_name=False):
        """Download the config_name if it exists.

        :param config_name: str, the name under which the file is stored
        :param human_readable_file_name: bool, store the file also under it's human readable name.
            It is better not to use this as the user might not know if the version of the file is
            the latest.
        :return: str, the absolute path of the file requested

        """
        fs_object = self.get_gridfs_object(config_name)
        # Ok, so we can open it. We will store the file under it's
        # md5-hash as that allows to easily compare if we already
        # have the correct file.
        if human_readable_file_name:
            target_file_name = config_name
        else:
            target_file_name = fs_object.md5

        if write_to is None:
            for cache_folder in self.storage_options:
                possible_path = os.path.join(cache_folder, target_file_name)
                if os.path.exists(possible_path):
                    # Great! This already exists. Let's just return
                    # where it is stored.
                    return possible_path

            # Apparently the file does not exist, let's find a place to
            # store the file and download it.
            store_files_at = self._check_store_files_at(self.storage_options)
        else:
            store_files_at = write_to
        destination_path = os.path.join(store_files_at, target_file_name)

        # Let's open a temporary directory, download the file, and
        # try moving it to the destination_path. This prevents
        # simultaneous writes of the same file.
        with tempfile.TemporaryDirectory() as temp_directory_name:
            temp_path = os.path.join(temp_directory_name, target_file_name)

            with open(temp_path, "wb") as stored_file:
                # This is were we do the actual downloading!
                warn(f"Downloading {config_name} to {destination_path}")
                stored_file.write(self.open_response(fs_object))

            if not os.path.exists(destination_path):
                # Move the file to the place we want to store it.
                move(temp_path, destination_path)
        return destination_path

    def get_abs_path(self, config_name):
        return self.download_single(config_name)


class GridFsInterfaceAPI(GridFsBase):
    """Interface to gridfs using the runDB API."""

    def __init__(self, config_identifier="config_name"):
        super().__init__(config_identifier=config_identifier)
        # all the credentials logic is handled by the utilix config, so don't need lengthy setup
        self.db = DB()

    def config_exists(self, config):
        """Quick check if this config is already saved in the collection.

        :param config: str, name of the file of interest
        :return: bool, is this config name stored in the database

        """
        query = self.get_query_config(config)
        return self.db.count_files(query) > 0

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
        return self.db.count_files(query) > 0

    def test_find(self):
        """Test the connection to the self.collection to see if we can perform a collection.find
        operation."""
        if self.db.get_files({}, projection={"_id": 1}) is None:
            raise ConnectionError("Could not find any data in this collection")

    def list_files(self):
        """Get a complete list of files that are stored in the database.

        Note that the config_identifier attribute is used to filter the results.

        :return: list, list of the names of the items stored in this database

        """
        return [
            doc[self.config_identifier]
            for doc in self.db.get_files({}, projection={self.config_identifier: 1})
            if self.config_identifier in doc
        ]


class APIDownloader(GridFsInterfaceAPI, GridFsDownloadBase):
    """Download files using the runDB API."""

    def __init__(self, config_identifier="config_name", store_files_at=None):
        GridFsDownloadBase.__init__(self, store_files_at=store_files_at)
        GridFsInterfaceAPI.__init__(self, config_identifier=config_identifier)

    def download_single(self, config_name: str, write_to=None, human_readable_file_name=False):
        """Download the config_name if it exists.

        :param config_name: str, the name under which the file is stored
        :param human_readable_file_name: bool, store the file also under it's human readable name.
            It is better not to use this as the user might not know if the version of the file is
            the latest.
        :return: str, the absolute path of the file requested

        """

        if human_readable_file_name:
            target_file_name = config_name
        else:
            target_file_name = self.db.get_file_md5(config_name)

        if write_to is None:
            for cache_folder in self.storage_options:
                possible_path = os.path.join(cache_folder, target_file_name)
                if os.path.exists(possible_path):
                    # Great! This already exists. Let's just return
                    # where it is stored.
                    return possible_path

            # Apparently the file does not exist, let's find a place to
            # store the file and download it.
            store_files_at = self._check_store_files_at(self.storage_options)
        else:
            store_files_at = write_to
        destination_path = os.path.join(store_files_at, target_file_name)

        # Let's open a temporary directory, download the file, and
        # try moving it to the destination_path. This prevents
        # simultaneous writes of the same file.
        with tempfile.TemporaryDirectory() as temp_directory_name:
            temp_path = self.db.download_file(config_name, save_dir=temp_directory_name)
            if not os.path.exists(destination_path):
                # Move the file to the place we want to store it.
                move(temp_path, destination_path)
            else:
                warn(f"File {destination_path} already exists. Not overwriting.")
        return destination_path


class APIUploader(GridFsInterfaceAPI, GridFsUploadBase):
    """Upload files to gridfs using the runDB API."""

    def __init__(self, config_identifier="config_name"):
        GridFsInterfaceAPI.__init__(self, config_identifier=config_identifier)

    def upload_single(self, config, abs_path):
        """Upload a single file to gridfs.

        :param config: str, the name under which this file should be stored
        :param abs_path: str, the absolute path of the file

        """
        if not os.path.exists(abs_path):
            raise CouldNotLoadError(f"{abs_path} does not exist")

        logger.info(f"uploading file {config} from {abs_path}")
        self.db.upload_file(abs_path, config)


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


def to_str_tuple(x) -> ty.Tuple[str]:
    """Convert x to tuple of string Stolen from strax."""
    if isinstance(x, str):
        return (x,)
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, pd.Series):
        return tuple(x.values.tolist())
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    raise TypeError(f"Expected string or tuple of strings, got {type(x)}")
