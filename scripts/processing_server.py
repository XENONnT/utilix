import os
import xeauth
import pymongo
import rframe
import requests
from fastapi import FastAPI, Request, Depends, status, HTTPException
from fastapi.security import HTTPBearer
from typing import Any, Optional, Union, List 

from utilix.processing_requests import ProcessingRequest, ProcessingJob

schemas = {
    'processing_request': ProcessingRequest,
    'processing_job': ProcessingJob,
}

mongo_user = os.getenv('MONGO_USER')
mongo_pass = os.getenv('MONGO_PASS')

url = f'mongodb://{mongo_user}:{mongo_pass}@xenon1t-daq.lngs.infn.it:27017/cmt2'

client = pymongo.MongoClient(url)
db = client['cmt2']

app = FastAPI()

MAX_RESULTS = 2000

token_auth_scheme = HTTPBearer()


def verfiy_read_auth(auth: str = Depends(token_auth_scheme)):
    try:
        claims = xeauth.certs.extract_verified_claims(auth.credentials)
        assert 'https://api.cmt.xenonnt.org' in claims.get('aud', [])
        assert 'read:all' in claims.get('scope', '').split(' ')

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verfiy_write_auth(auth: str = Depends(token_auth_scheme)):
    try:
        claims = xeauth.certs.extract_verified_claims(auth.credentials)
        assert 'https://api.cmt.xenonnt.org' in claims.get('aud', [])
        assert 'write:all' in claims.get('scope', '').split(' ')

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unauthorized: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Serve each schema at its own endpoint
for name, schema in schemas.items():

    collection = db[name]

    router = rframe.SchemaRouter(
             schema,
             collection,
             prefix = name,
             can_read = Depends(verfiy_read_auth),
             can_write = Depends(verfiy_write_auth))
    app.include_router(router)

# Root endpoint serves list of schema names
@app.get("/", response_model=List[str])
def list_schemas():
    return list(schemas)

# ------------ TODO: Automated job submission logic ------------- #
# 
# Step 1: Find environment that can produce the requested data

DEFAULT_ENV = '2022.03.5'
ENV_TAGS_URL = 'https://api.github.com/repos/xenonnt/base_environment/git/matching-refs/tags/'

# helper function to read all tags from github
def get_envs():
    r = requests.get(ENV_TAGS_URL)
    if not r.ok:
        return []
    tags = r.json()
    tagnames = [tag['ref'][10:] for tag in tags if tag['ref'] ]
    return tagnames

# Option 1: just go from newest to last env until one has the correct hash?
# Option 2: Lookup contexts from mongodb and use env listed there.

# step 2: Find best site for the requested job, e.g by location  of dependencies

# Step 3: Submit job to site with environment 
