import datetime
import time
import json

import gnome.basic_types
from gnome import environment
from gnome.persist import environment_schema, util

"""
helper functions to dump/load json or json like data to file
"""
def save(data, fname):
    with open(fname,'w') as outfile:
        json.dump(data, outfile, indent = True)

def load(fname):
    with open(fname,'r') as infile:
        return json.load(infile)

"""
Test case, Wind object 
state to be saved in text file in json format
"""
f =r'/Users/jasmine.sandhu/Documents/projects/gnome/py_gnome/tests/unit_tests/SampleData/WindDataFromGnome.WND'
w = environment.Wind(file=f)
w.latitude = 10
w.longitude = 100


"""
============================
Use Colander to save state - 
this is a possible approach if monkey-patch is successful for colander
"""
savfile_ = 'cdata.txt'

"""
Using colander to serialize to json then save data in cdata.txt
NOTE: this is not valid json format since everything is a string
"""
save( environment_schema.WindState().serialize(w.state_to_dict()), savfile_)

"""
Load json-like dict from file, deserialize and validate
Create new object from dict
"""
c_dict = environment_schema.WindState().deserialize(load(savfile_) )   # must deserialize through schema
new_w1 = environment.Wind.new_from_dict(c_dict)             # now use dict to create new object

"""
============================
CURRENT APPROACH - 
Use Colander to validate state
use util.* methods to convert to plain python objects to JSONify
"""
savfile_ = 'data.txt'
state_dict = w.state_to_dict()

"""
Following functionality could be moved to 'to_dict' so output is already in plain python objects
"""
state_dict['updated_at'] = state_dict['updated_at'].isoformat()
state_dict['timeseries'] = util.datetime_value_2d_to_list(state_dict['timeseries'])

save( state_dict, savfile_)     # WRITE JSON
load_state = load( savfile_)    # LOAD JSON from file, should be same as state_dict
environment_schema.WindState().deserialize( state_dict )    # Validate dict after loading data

"""
Following two operations to convert plain python objects can be moved to 
from_dict method for Wind object
"""
load_state['updated_at'] = util.str_to_date(load_state['updated_at'])
load_state['timeseries'] = util.list_to_datetime_value_2d(load_state['timeseries'])

new_w2 = environment.Wind.new_from_dict(load_state)          # now use dict to create new object
