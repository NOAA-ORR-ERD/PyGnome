'''
Created on Mar 7, 2013

load / save a py_gnome scenario
'''
import os
import json
import glob
import string
import shutil

import gnome
from gnome.persist import environment_schema, model_schema, movers_schema, spills_schema, map_schema


def save(model, saveloc=None):
    """
    given a model and a saveloc, persist the model
    """
    if saveloc is None:
        raise ValueError("Provide a valid location for saving scenario")
    
    if not os.path.exists(saveloc):
        raise ValueError("Invalid location for saving scenario. {0} does not exist".format(saveloc))
    
    # first save model info
    dict_ = model.to_dict('create')
    
    model_to_json = gnome.persist.model_schema.Model().serialize( dict_ )
    
    _save_to_file(model_to_json,
                  os.path.join( saveloc, '{0}_{1}.txt'.format( model.__class__.__name__, model.id)))
    
    _save_collection( [model.map, model.output_map],'map_schema',saveloc)
    _save_collection(model.movers,'movers_schema',saveloc)
    _save_collection(model.environment, 'environment_schema', saveloc)
    
    for sc in model.spills.items():
        _save_collection( sc.spills,'spills_schema',saveloc)
    

def load(saveloc, filename=None):
    """
    look for model_*.txt for model to load
    """
    if not os.path.exists(saveloc):
        raise ValueError("Invalid location for saving scenario. {0} does not exist".format(saveloc))
    
    if filename is None:
        model_file = glob.glob(os.path.join(saveloc,'Model_*.txt'))
        if len(model_file) > 1:
            raise ValueError("multiple Model_*.txt files found in {0}. Please provide 'filename'".format(saveloc))
        else:
            model_file = model_file[0]
    else:
        if not os.path.exists( os.path.join(saveloc,filename) ):
            raise ValueError("{0} file not found, enter valid filename".format(os.path.join(saveloc,filename)))
        model_file = os.path.join(saveloc,filename) 
            
    # first load model, then create other objects and add to model
    model_json = _load_from_file(model_file)
    model_dict = gnome.persist.model_schema.Model().deserialize( model_json )
    
    maps = _load_dict(model_dict['maps'], 'map_schema', saveloc)
    for key in maps.keys():
        model_dict.update({key:maps[key]})
    
    model = gnome.model.Model.new_from_dict(model_dict)
    print "created base model ..."
    
    print "add spills .."
    c_spills = _load_collection(model_dict['spills']['certain_spills'], 'spills_schema', saveloc)
    if model.uncertain:
        u_spills = _load_collection(model_dict['spills']['uncertain_spills'], 'spills_schema', saveloc)
        obj_list = zip(c_spills, u_spills)
    else:
        obj_list = c_spills
    
    [model.spills.add(obj) for obj in obj_list]
    
    #first add environment collection - since movers depend on this
    print "add environment .."    
    obj_list = _load_collection(model_dict['environment'], 'environment_schema', saveloc)
    [model.environment.add(obj) for obj in obj_list]
    
    print "add movers .."    
    _add_movers(model_dict['movers'], saveloc, model)

    return model
    
def _save_collection(coll_,schema_module, saveloc):
    for obj in coll_:
        dict_ = obj.to_dict('create')
        to_eval = '{0}.{1}().serialize(dict_)'.format( schema_module, obj.__class__.__name__ )
        to_json = eval(to_eval)
        _save_to_file(to_json, os.path.join( saveloc, '{0}_{1}.txt'.format( obj.__class__.__name__, obj.id)) )
    
def _save_to_file(data, fname):
    data = _move_data_file(data, os.path.dirname(fname) ) # if there is a
    with open(fname,'w') as outfile:
        json.dump(data, outfile, indent = True)

def _load_from_file(fname):
    with open(fname,'r') as infile:
        return json.load(infile)

def _add_movers(movers_dict, saveloc, model):
    for type_, id_ in movers_dict['id_list']:
        
        obj_json = _find_and_load_json_file( type_, id_, saveloc)
        obj_name= string.rsplit( type_, '.', 1)[-1]
        
        obj_dict = _dict_from_json(type_, 'movers_schema', obj_json)
            
        if obj_name == 'WindMover':
            obj_dict.update({'wind': _get_obj(model.environment, obj_dict['wind_id']) })
            
        elif obj_name == 'CatsMover' and obj_dict.get('tide_id') is not None:
            obj_dict.update({'tide': _get_obj(model.environment, obj_dict['tide_id']) })
            
        obj = _obj_from_dict( obj_dict, type_)
        model.movers += obj

def _get_obj( coll_, id):
    try:
        return coll_[id] # get object associated with this Id
    except KeyError, e:
        raise KeyError("Collection does not contain an object with id: {0}".format(e.message))

def _load_collection(coll_dict, schema_module, saveloc):
    """
    Load collection
    """
    obj_list = []
    for type_, id_ in coll_dict['id_list']:
        obj_json = _find_and_load_json_file( type_, id_, saveloc)
        obj = _obj_from_json( type_, schema_module, obj_json)
        obj_list.append( obj) 
    return obj_list

def _load_dict(dict_, schema_module, saveloc):
    """ 
    each keyword contains a tuple (type, id). For instance,
    'map': ("gnome.map.MapFromBNA", id)
    
    Use this to re-create gnome.map.MapFromBNA object from file
    Return a dict with same keywords but values now containing recreated objects
    'map' : recreated MapFromBNA object 
    """
    obj_dict = {}
    for key in dict_.keys():
        type_ = dict_[key][0]
        id_ = dict_[key][1]
        obj_json = _find_and_load_json_file( type_, id_, saveloc)
        obj = _obj_from_json( type_, schema_module, obj_json)
        obj_dict.update({key:obj})

    return obj_dict

def _find_and_load_json_file( type_, id_, saveloc):
    obj_file = glob.glob( os.path.join(saveloc, '*_{0}.txt'.format(id_) ) )
    if len(obj_file) == 0:
        raise IOError("No filename containing *_{0}.txt found in {1}".format(id_, saveloc))
    elif len(obj_file) > 1:
        raise IOError("Cannot have two objects with same Id. Multiple filenames containing *_{0}.txt found in {1}".format(id_, saveloc))
    
    obj_file = obj_file[0]
    obj_json = _load_from_file(os.path.abspath( obj_file) )
    return obj_json

def _obj_from_json( type_, schema_module, obj_json):
    obj_dict = _dict_from_json(type_, schema_module, obj_json)
    obj = _obj_from_dict(obj_dict, type_) 
    return obj

def _dict_from_json(type_, schema_module, obj_json):
    obj_name= string.rsplit( type_, '.', 1)[-1]
    to_eval = '{0}.{1}().deserialize( obj_json )'.format( schema_module, obj_name)
    obj_dict = eval( to_eval)
    return obj_dict

def _obj_from_dict(obj_dict, type_):
    to_eval = "{0}.new_from_dict(obj_dict)".format(type_)
    obj = eval( to_eval)
    return obj

def _move_data_file(to_json, saveloc):
    """if there is a 'filename' field, move the data file to saveloc, update 'filename' and return to_json"""
    if 'filename' in to_json:
        shutil.copy(to_json['filename'], saveloc)
        to_json['filename'] = os.path.join( saveloc, os.path.split(to_json['filename'])[1] )
    
    return to_json