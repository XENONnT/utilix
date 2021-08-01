"""
Simple module to register a data entry in the rundoc from a file
"""
import utilix
import os
from pymongo.collection import Collection
from bson import json_util
import json

# Should match github.com/AxFoundation/strax/blob/a57e7688c9c2ca3a6492c3042297a2f1db0f9bb3/strax/storage/files.py#L15  # noqa
RUN_METADATA_PATTERN = '%s-metadata.json'


class RunDocUpload:
    """
    Simple class to init once and keep uploading files from their path
    to the runs database
    """
    def __init__(self, base_doc: dict, run_col: Collection = None):
        if run_col is None:
            run_col = utilix.xent_collection()
        self._check_base_doc(base_doc)
        self.run_col = run_col
        self.base_doc = base_doc

    def upload_from_file(self,
                         path: str,
                         location: str = None,
                         ) -> None:
        """
        Use upload_doc_from_file to upload a data-document based on the
        folder

        :param path: where is the data stored (should be a folder)
        :param location: how should the location be called? If no
        location is specified, we'll assume the path is the location
        :return: None
        """
        upload_doc_from_file(self.base_doc, path, self.run_col, location=location)

    @staticmethod
    def _check_base_doc(doc):
        """Check that the base at least contains the required fields"""
        missing = []
        # I'm just hardcoding it here as this should *always* be true
        for field in 'host protocol'.split():
            if field not in doc:
                missing += [field]
        if missing:
            raise ValueError(f'base doc misses {missing}')


def upload_doc_from_file(base_doc: dict,
                         path: str,
                         run_col: Collection,
                         check_for_overwrite=True,
                         location=None,
                         ) -> None:
    """
    Given a path to a file, upload an entry to the Runs Database

    :param base_doc: The basis for the document, should be a dict
        with the fields that are the same for the current host
    :param path: path to the folder where the data is stored
    :param run_col: The (runs) collection where to write the data to
    :param check_for_overwrite: Check the collection if we cannot find
        an entry that matches the current document that would otherwise
        be inserted.
    :return: None
    :raises ValueError: if either of:
        - The path does not exists
        - The document already exists (if check_for_overwrite)
        - Not exactly one document is inserted.
    """
    if not os.path.exists(path):
        raise ValueError(f'{path} does not exist')

    metadata = get_md(path)
    run_id = metadata['run_id']
    lineage_hash = metadata['lineage_hash']
    data_type = metadata['data_type']
    files = os.listdir(path)

    for f in files:
        if 'temp' in f:
            raise ValueError(f'One or more temp files in {path}, we aren\'t '
                             f'finished writing!')

    if location is None:
        location = path

    # Make a new document for this data
    ddoc = base_doc.copy()

    # basic entries
    ddoc['file_count'] = len(files)
    ddoc['did'] = f'xnt_{run_id}:{data_type}:{lineage_hash}'
    ddoc['type'] = data_type
    ddoc['location'] = location

    # Meta entries
    ddoc['meta'] = {}
    ddoc['meta']['strax_version'] = metadata['strax_version']
    ddoc['meta']['compressor'] = metadata['compressor']
    chunk_mb = [chunk['nbytes'] / 1e6 for chunk in metadata['chunks']]
    ddoc['meta']['size_mb'] = int(sum(chunk_mb))
    ddoc['meta']['lineage_hash'] = lineage_hash

    # TODO
    #  Disabled, do we need any of these (open for discussion:
    #  meta.lineage
    #  meta.straxen_version
    #  meta.avg_chunk_mb
    #  lifetime
    #  status

    if check_for_overwrite:
        # Check that querying the current ddoc does not exist already
        if run_col.find_one({'number': int(run_id),
                             'data':
                                 {'$elemMatch': ddoc}
                             }
                            ) is not None:
            raise ValueError(f'Duplicating rundoc for {path}!')

    # If you don't have write access, this fails.
    insertion_result = run_col.find_one_and_update(
        {'number': int(run_id)},
        {'$push': {'data': ddoc}})

    if insertion_result is None:
        raise ValueError(f'inserted {insertion_result} is None. This means the '
                         f'entry for {run_id} does not exist?!')


def read_md(path):
    """Given an absolute path to a (metadata) json file, open it"""
    with open(path, mode='r') as f:
        md = json.loads(f.read(),
                        object_hook=json_util.object_hook)
    return md


def get_md(base_dir):
    """
    Get the metadata file from a strax-folder containing data from one
    data type of a run
    """
    for p in os.listdir(base_dir):
        if 'metadata' in p:
            abs_p = os.path.join(base_dir, p)
            return read_md(abs_p)
