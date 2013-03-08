'''
Created on Mar 7, 2013

load / save a py_gnome scenario
'''
import os
import json
import glob

import gnome
from gnome.persist import environment_schema, model_schema, movers_schema

def save(model, saveloc=None):
    """
    given a model and a saveloc, persist the model
    """
    if saveloc is None:
        raise ValueError("Provide a valid location for saving scenario")
    
    if not os.path.exists(saveloc):
        raise ValueError("Invalid location for saving scenario. {0} does not exist".format(saveloc))
    
    # first save model info
    model_to_json = gnome.persist.model_schema.CreateModel().serialize( model.to_dict('create') )
    _save_to_file(model_to_json,
                  os.path.join( saveloc, 'model_{0}.txt'.format(model.id)))
    
    _save_collection(model.movers,'movers_schema',saveloc)
    _save_collection(model.environment, 'environment_schema', saveloc)
    

def load(saveloc, filename=None):
    """
    look for model_*.txt for model to load
    
    (WIP) - currently not working
    """
    if not os.path.exists(saveloc):
        raise ValueError("Invalid location for saving scenario. {0} does not exist".format(saveloc))
    
    if filename is None:
        model_file = glob.glob(os.path.join(saveloc,'model_*.txt'))
        if len(model_file) > 1:
            raise ValueError("multiple model_*.txt files found in {0}. Please provide 'filename'".format(saveloc))
        else:
            model_file = model_file[0]
    else:
        if not os.path.exists( os.path.join(saveloc,filename) ):
            raise ValueError("{0} file not found, enter valid filename".format(os.path.join(saveloc,filename)))
        model_file = os.path.join(saveloc,filename) 
            
    # first load model, then create other objects and add to model
    model_json = _load_from_file(model_file)
    model_dict = gnome.persist.model_schema.CreateModel().deserialize( model_json )
    gnome.model.Model.new_from_dict(**model_dict)
    print "created base model ..."
    

def _save_collection(coll_,schema_module, saveloc):
    for obj in coll_:
        dict_ = obj.to_dict('create')
        to_eval = '{0}.Create{1}().serialize(dict_)'.format( schema_module, obj.__class__.__name__ )
        to_json = eval(to_eval)
        _save_to_file(to_json, os.path.join( saveloc, '{0}_{1}.txt'.format( obj.__class__.__name__, obj.id)) )
    
def _save_to_file(data, fname):
    with open(fname,'w') as outfile:
        json.dump(data, outfile, indent = True)

def _load_from_file(fname):
    with open(fname,'r') as infile:
        return json.load(infile)