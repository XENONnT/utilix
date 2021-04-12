# utilix
[![PyPI version shields.io](https://img.shields.io/pypi/v/utilix.svg)](https://pypi.python.org/pypi/utilix/)

``utilix`` is a utility package for XENON software, mainly relating to analysis. It currently has two main features: (1) a general XENON configuration framework and (2) easy access to the runsDB by wrapping python calls to a RESTful API. Eventually, we would like to include easy functions for interacting with the Midway and OSG batch queues. 

## Configuration file

This tool expects a configuration file given by the environment variable `XENON_CONFIG`, defaulting to `$HOME/.xenon_config` if it is empty. Note that
environment variables can be used in the form `$HOME`. Example:

    [RunDB]
    rundb_api_url = [ask Evan]
    rundb_api_user = [ask Evan]
    rundb_api_password = [ask Evan]


The idea is that analysts could use this single config for multiple purposes/analyses.
You just need to add a (unique) section for your own purpose and then you can use the `utilix.Config` 
easily. For example, if you made a new section called `WIMP` with `detected = yes` under it:

    from utilix.config import Config
    cfg = Config()
    value = cfg.get('WIMP', 'detected') # value = 'yes'
    
For more information, see the [ConfigParser](https://docs.python.org/3.6/library/configparser.html)
documentation, from which `utilix.config.Config` inherits.

## Runs Database
Nearly every analysis requires access to the XENON runsDB. The goal of utilix is to simplify the usage of this resource as much as possible. The ``rundb`` module includes two ways to access the runsDB:

  1. A RESTful API: a Flask app running at Chicago that queries the runDB in a controlled manner. This is the recommended way to query the database if the specific query is supported. The source code for this app can be found [here](https://github.com/XENONnT/xenon_runsDB_api).
  2. A wrapper around ``pymongo``, which sets up the Mongo client for you, similarly to how we did queries in XENON1T. In that case each package usually needed its own copy + pasted boilerplate code; that code is now just included in utilix where it can be easily imported by other packages. 
  
  
-------------
### RunDB API

#### RunDB API Authentication
The API authenticates using a token system. `utilix` makes the creation and renewal of these tokens easy with the `utilix.rundb.Token` class. When you specify a user/password in your utilix configuration file, as shown above, a token is saved locally at `~/.dbtoken` that contains this information. This token is used/renewed as needed, depending on the users specified in the config file. 

Different API users have different permissions, with the general analysis user only able to read from the runDB and not write. This is an additional layer of security around the RunDB. 

#### Setting up the runDB
The goal of utilix is to make access to the runDB trivial. If using the runDB API, all you need to do to setup the runDB in your local script/shell is

    from utilix import db
    
This instantiates the RunDB class, allowing for easy queries. Below we go through some examples of the type of queries currently supported by the runDB API wrapper in utilix. 

**If there is functionality missing that you think would be useful, please contact Evan or make a new issue (or even better, a pull request).**



#### Query for runs by source

Note that the interface returns pages of 1,000 entries, with the first page being 1.

    from utilix import db
    data = db.query_by_source('neutron_generator', page_num=1)

#### Get a full document

You can also grab the full run document using the run number. A run name is also supported (from XENON1T days), 
but not going to be used for XENONnT

    doc = db.get_doc(7200)

#### Get only the data entry of a document

    data = db.get_data(2000)
    
    
#### Strax(en) Contexts
In XENONnT we need to track the hash (or lineage) that specifies a configuration for each datatype. We keep that information in a specific collection of the runDB. We can access that collection using the runDB API as shown below.

For a given context name and straxen version, we can get the hash for each dataype. For example, for the xenonnt_online context and straxen version 0.11.0:
    
    >>> db.get_context('xenonnt_online', '0.11.0')
    
    {'_id': '5f89f588d33cced1fd104ea5',
     'date_added': '2020-10-16T19:33:28.913000',
     'hashes': {'aqmon_hits': '4gwju6gdto',
                'corrected_areas': 'wbgyvcbq7i',
                'distinct_channels': 'a6orjqoffa',
                'energy_estimates': 'plxyjbnnui',
                'event_basics': 'pjql7a36yb',
                'event_info': 's4mprxr7qq',
                'event_info_double': 'hodh2726fi',
                'event_positions': 'tzzwhaf4gy',
                'events': 'aycpwrvco2',
                'hitlets_nv': 'pd2fuwzbjt',
                'led_calibration': 'lsdigsccxn',
                'lone_hits': 'nagx3zzuiv',
                'lone_raw_record_statistics_nv': 'vx2cbuxcxo',
                'lone_raw_records_nv': 'vx2cbuxcxo',
                'merged_s2s': '274qjtnjto',
                'merged_s2s_he': 'n5tib6rljj',
                'peak_basics': '2gq3vm6b2t',
                'peak_basics_he': 'ucjr5lvvbg',
                'peak_positions': 'c7vzetbnqw',
                'peak_proximity': '7nvjls2mab',
                'peaklet_classification': 'gpnp6dzxc4',
                'peaklet_classification_he': 'tmlx5dkfcf',
                'peaklets': 'nagx3zzuiv',
                'peaklets_he': 'mu26cr25vf',
                'peaks': 'dmavalropc',
                'peaks_he': 'hzamxeoer6',
                'pulse_counts': 'jxkqp76kam',
                'pulse_counts_he': '5mepav2zzf',
                'raw_records': 'rfzvpzj4mf',
                'raw_records_aqmon': 'rfzvpzj4mf',
                'raw_records_aqmon_nv': 'rfzvpzj4mf',
                'raw_records_coin_nv': 'vx2cbuxcxo',
                'raw_records_he': 'rfzvpzj4mf',
                'raw_records_mv': 'rfzvpzj4mf',
                'raw_records_nv': 'rfzvpzj4mf',
                'records': 'jxkqp76kam',
                'records_he': '5mepav2zzf',
                'records_nv': 'btmwmjvgdp',
                'veto_intervals': '7q35r23nba',
                'veto_regions': 'jxkqp76kam'},
     'name': 'xenonnt_online',
     'strax_version': '0.12.2',
     'straxen_version': '0.11.0'}

If you know the specific datatype whose hash you need, use instead `get_hash`:

    >>> db.get_hash('xenonnt_online', 'peaklets',  '0.11.0')
    'nagx3zzuiv'
    

If you are deemed worthy to have write permissions to the runDB (you have a corresponding user/password with write access in your config file), you can also add documents to the context collection with

    >>> db.update_context_collection(document_data)
    
where `document_data` is a dictionary that contains the context name, straxen version, hash information, and more as shown in the example above. 

 
### Boilerplate pymongo setup
The runDB API is the recommended option for most database queries, but sometimes a specific query isn't supported or you might want to do complex aggregations, etc. For that reason, `utilix` also includes a wrapper around `pymongo` to setup the MongoClient. To use this, you need to specify in your config file 

    [RunDB]
    pymongo_url = [ask someone]
    pymongo_database = [ask someone]
    pymongo_user = [ask someone]
    pymongo_password =  [ask someone] 

Note that this is needed in addition to the runDB API fields. Given the correct user/password, you can setup the XENONnT runDB collection for queries as follows:

    >>> from utilix.rundb import pymongo_collection
    >>> collection = pymongo_collection()
    
Then you can query the runDB using normal pymongo commands. For example:

    >>> collection.find_one({'number': 9000}, {'number': 1})
    {'_id': ObjectId('5f2d999448350bff030d2d3b'), 'number': 9000}
    
You can also access different collections by passing an argument to `pymongo_collection`. The analogous query to the contexts collection shown above would be:

    >>> collection = pymongo_collection('contexts')
    >>> collection.find_one({'name': 'xenonnt_online', 'straxen_version': '0.11.0'})
    
If you need to use different databases or do not want to use the information listed in your utilix configuration file, you can also pass keyword arguments to overwrite the information in the config file. This is useful if you need to e.g. use both XENONnT and XENON1T collections. To use the XENON1T collection, for example:
    
    >>> xe1t_coll, xe1t_db, xe1t_user, xe1t_pw, xe1t_url = [ask someone]
    >>> xe1t_collection = pymongo_collection(xe1t_coll, database=xe1t_coll, user=xe1t_user, password=xe1t_pw, url=xe1t_url)
       
       
## TODO
We want to implement functionality for easy job submission to the Midway batch queue.
Eventually we want to do the same for OSG. 

It would be nice to port e.g. the admix database wrapper to utilix, which can then be used 
easily by all analysts. 
