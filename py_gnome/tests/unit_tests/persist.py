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
#w.latitude = 10
#w.longitude = 100


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
save( environment_schema.CreateWind().serialize(w.to_dict('create')), savfile_)

"""
Load json-like dict from file, deserialize and validate
Create new object from dict
"""
c_dict = environment_schema.CreateWind().deserialize(load(savfile_) )# must deserialize through schema
new_w1 = environment.Wind.new_from_dict(c_dict)                     # now use dict to create new object