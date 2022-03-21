import os
from typing import Literal
import rframe
import datetime
import getpass
import pydantic
import requests
from warnings import warn

CACHE = {}
DEFAULT_ENV = '2022.03.5'
ENV_TAGS_URL = 'https://api.github.com/repos/xenonnt/base_environment/git/matching-refs/tags/'

API_URL = 'http://api.cmt.yossisprojects.com'


def xeauth_user():
    return os.environ.get('XEAUTH_USER', 'UNKNOWN')

def get_envs():
    r = requests.get(ENV_TAGS_URL)
    if not r.ok:
        return []
    tags = r.json()
    tagnames = [tag['ref'][10:] for tag in tags if tag['ref'] ]
    return tagnames

def default_env():
    cond_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if cond_env.startswith('XENONnT_'):
        return cond_env.replace('XENONnT_', '')
    envs = get_envs()
    if envs:
        return envs[-1]
    return DEFAULT_ENV

class ProcessingRequest(rframe.BaseSchema):
    _NAME = 'processing_requests'

    data_type: str = rframe.Index()
    lineage_hash: str = rframe.Index()
    run_id: str = rframe.Index()
    
    user: str = pydantic.Field(default_factory=xeauth_user)
    request_date: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.utcnow)
    
    priority: int = -1

    comments: str = ''
    
    
    def pre_update(self, datasource, new):
        if new.user != self.user:
            raise ValueError(new.user)
        if new.run != self.run:
            raise ValueError(new.run)


class ProcessingJob(rframe.BaseSchema):
    _NAME = 'processing_jobs'

    job_id: str = rframe.Index()
    location: Literal['OSG','DALI','OTHER'] = rframe.Index()
    env: str = rframe.Index()
    context: str = rframe.Index()
    data_type: str = rframe.Index()
    run_id: str = rframe.Index()
    lineage_hash: str = rframe.Index()
    
    submission_time: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.utcnow)
    completed: bool = False
    progress: int = 0
    error: str = ''


def xeauth_login(readonly=True):
    try:
        import xeauth
        scope = 'read:all' if readonly else 'write:all'
        xetoken = xeauth.cmt_login(scope=scope)
        username = xetoken.profile.get('name', None)
        if username is not None:
            os.environ['XEAUTH_USER'] = username

        return xetoken.access_token
    except ImportError: 
        warn('xeauth not installed, cannot retrieve token automatically.')


def processing_api(token=None, readonly=True):
    cache_key = f'api_token_readonly_{readonly}'

    if token is None:
        token = CACHE.get(cache_key, None)
    
    if token is None:
        token = os.environ.get('PROCESSING_API_TOKEN', None)
    
    if token is None:
        token = xeauth_login(readonly=readonly)
    
    if token is None:
        token = getpass.getpass('API token: ')

    headers = {}
    if token:
        headers['Authorization'] = f"Bearer {token}"
        CACHE[cache_key] = token

    client = rframe.RestClient(f'{API_URL}/processing_requests',
                                 headers=headers,)
    return client


try:
    import strax
    import tqdm

    @strax.Context.add_method
    def request_processing(context, run_ids, data_type, priority=-1, comments='', token=None, submit=True):
        client = processing_api(token=token, readonly=False)

        run_ids = strax.to_str_tuple(run_ids)
        requests = []
        for run_id in tqdm.tqdm(run_ids, desc='Requesting processing'):
            
            lineage_hash = context.key_for(run_id, data_type).lineage_hash

            kwargs = dict(data_type=data_type, 
                          lineage_hash=lineage_hash,
                          run_id=run_id,
                          priority=priority,
                          comments=comments)
            
            request = ProcessingRequest(**kwargs)
            requests.append(request)
            if submit:
                request.save(client)
                
        if len(requests) == 1:
            return requests[0]
        return requests

except ImportError:
    pass