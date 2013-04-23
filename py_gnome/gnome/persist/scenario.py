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

from gnome.persist import (
   modules_dict,
   environment_schema, 
   model_schema, 
   movers_schema, 
   spills_schema, 
   map_schema, 
   outputters_schema)


class Scenario(object):
    """ Create a class that contains functionality to load/save scenario"""
    
    def __init__(self, saveloc, model=None):
        """
        If user wants to save a scenario, then model must be set 
        """
        if not os.path.exists(saveloc):
            raise ValueError("Invalid location for saving scenario. {0} does not exist".format(saveloc))
    
        self.saveloc = saveloc
        self.model = model
        
    def save(self):
        if self.model is None:
            raise AttributeError("A model needs to be defined before it can be saved")
            
        dict_ = self.model.to_dict('create')
        self._save_json_to_file( self.dict_to_json(dict_), self.model)
        
        dict_ = self.model.map.to_dict('create')
        self._save_json_to_file( self.dict_to_json(dict_), self.model.map)
        
        self._save_collection(self.model.movers)
        self._save_collection(self.model.environment)
        self._save_collection(self.model.outputters)
        
        for sc in self.model.spills.items():
            self._save_collection( sc.spills)
        
    def load(self):
        """
        look for model_*.txt for model to load
        """
        # For all other objects, change directory to saveloc, load files, then change back to cur_dir
        try:
            cur_dir = os.getcwd()
            os.chdir(self.saveloc)  # do this since datafiles are stored relative to saveloc
            
            model_dict = self.load_model_dict()
            
            # pop lists that are not used for model initialization
            l_movers = model_dict.pop('movers')
            l_environment = model_dict.pop('environment')
            l_outputters = model_dict.pop('outputters')
            l_spills = model_dict.pop('spills')
                        
            self.model = self.dict_to_obj(model_dict)
            print "created base model ..."
            
            #first add environment collection - since l_movers depend on this
            print "add environment .."    
            obj_list = self._load_collection( l_environment)
            [self.model.environment.add(obj) for obj in obj_list]
            
            print "add outputters .."
            obj_list = self._load_collection( l_outputters)
            [self.model.outputters.add(obj) for obj in obj_list]
            
            print "add spills .."
            self._add_spills(l_spills)
            
            print "add movers .."    
            self._add_movers(l_movers)
            os.chdir(cur_dir)
            
        except (KeyError,IOError, ValueError) as err:
            os.chdir(cur_dir)   # return to original location
            print "\n\n!!! Following error occurred while loading model from {0}".format(self.saveloc)
            raise err   
        
    
    def dict_to_json(self,dict_):
        """ convert the dict returned by object's to_dict method to valid json format via colander schema """
        gnome_mod, obj_name = dict_['obj_type'].rsplit('.',1)
        to_eval = "{0}.{1}().serialize(dict_)".format( modules_dict[gnome_mod], obj_name)
        _to_json = eval(to_eval)
        return _to_json
    
    def _save_collection(self,coll_):
        for obj in coll_:
            dict_ = obj.to_dict('create')
            self._save_json_to_file( self.dict_to_json(dict_), obj )
        
    def _save_json_to_file(self, data, obj):
        fname = os.path.join( self.saveloc, '{0}_{1}.txt'.format( obj.__class__.__name__, obj.id))
        data = self._move_data_file(data) # if there is a
        with open(fname,'w') as outfile:
            json.dump(data, outfile, indent = True)
    
    def _move_data_file(self,to_json):
        """if there is a 'filename' field, move the data file to saveloc, update 'filename' and return to_json"""
        for key, value in to_json.items():
            if isinstance(value, basestring):
                if os.path.exists(value) and os.path.isfile(value):
                    shutil.copy(value, self.saveloc)
                    to_json[key] = os.path.split(to_json['filename'])[1]
                     
        #=======================================================================
        # if 'filename' in to_json:
        #    shutil.copy(to_json['filename'], self.saveloc)
        #    to_json['filename'] = os.path.split(to_json['filename'])[1]
        #=======================================================================
        
        return to_json
    
    """ LOADING FUNCTIONS """
    def json_to_dict(self, json_):
        """ convert the dict returned by object's to_dict method to valid json format via colander schema """
        gnome_mod, obj_name = json_['obj_type'].rsplit('.',1)
        to_eval = "{0}.{1}().deserialize(json_)".format( modules_dict[gnome_mod], obj_name)
        _to_dict = eval(to_eval)
        return _to_dict
    
    def dict_to_obj( self, obj_dict):
        """ create object from a dict. The dict contains (keyword,value) pairs used to create new object """
        type_   = obj_dict.pop('obj_type')
        to_eval = "{0}.new_from_dict(obj_dict)".format(type_)
        obj = eval( to_eval)
        return obj
    
    def load_model_dict(self):
        """ Load model dict from *.txt file. Pop 'map' key, create 'map' object and add to model dict 
            This dict is used in Model.new_from_dict(dict_) to create new Model """
        model_file = glob.glob( 'Model_*.txt')
        if model_file == []:
            raise ValueError("No Model_*.txt files find in {0}".format(self.saveloc))
        elif len(model_file) > 1:
            raise ValueError("multiple Model_*.txt files found in {0}. Please provide 'filename'".format(self.saveloc))
        else:
            model_file = model_file[0]
        
        model_json = self._load_json_from_file( model_file)
        model_dict = self.json_to_dict( model_json)
        
        # create map object and add to model_dict
        map_type, map_id = model_dict['map']
        obj_json = self._find_and_load_json_file( map_id)
        
        dict_ = self.json_to_dict(obj_json)
        map = self.dict_to_obj(dict_)
        
        model_dict['map'] = map # replace map object in the dict
        
        return model_dict
        
    
    def _load_json_from_file(self, fname):
        with open(fname,'r') as infile:
            return json.load(infile)
    
    def _find_and_load_json_file( self, id_):
        obj_file = glob.glob( '*_{0}.txt'.format(id_) )
        if len(obj_file) == 0:
            raise IOError("No filename containing *_{0}.txt found in {1}".format(id_, os.path.abspath('.')))
        elif len(obj_file) > 1:
            raise IOError("Cannot have two objects with same Id. Multiple filenames containing *_{0}.txt found in {1}".format(id_, os.path.abspath(self.saveloc)))
        
        obj_file = obj_file[0]
        obj_json = self._load_json_from_file(os.path.abspath( obj_file) )
        return obj_json
        
    def _load_collection( self, coll_dict):
        """
        Load collection - dict contains the output of OrderedCollection.to_dict()
        'dtype' - not used for anything
        'id_list' - for each object in list, use this to find and load the json file, convert it to 
                    a valid dict, then create a new object using new_from_dict
                    'id_list' contains a list of tuples (object_type, id of object)
        returns a list of objects corresponding with the data in 'id_list'
        """
        obj_list = []
        for type_, id_ in coll_dict['id_list']:
            obj_json = self._find_and_load_json_file( id_)
            dict_ = self.json_to_dict(obj_json)
            obj = self.dict_to_obj(dict_)
            obj_list.append( obj) 
        return obj_list
        
        
    def _add_movers( self, movers_dict):
        """
        add movers to the model - dict contains the output of OrderedCollection.to_dict()
        'dtype' - not used for anything
        'id_list' - for each object in list, use this to find and load the json file, convert it to 
                    a valid dict, then create a new object using new_from_dict
                    'id_list' contains a list of tuples (object_type, id of object)
        
        If Wind object and Tide object are present, they must already be added to self.model.environment
        prior to calling _add_movers
        """
        for type_, id_ in movers_dict['id_list']:
            
            obj_json = self._find_and_load_json_file( id_)
            obj_name= string.rsplit( type_, '.', 1)[-1]
            
            obj_dict = self.json_to_dict(obj_json)
                
            if obj_name == 'WindMover':
                obj_dict.update({'wind': self._get_obj(self.model.environment, obj_dict['wind_id']) })
                
            elif obj_name == 'CatsMover' and obj_dict.get('tide_id') is not None:
                obj_dict.update({'tide': self._get_obj(self.model.environment, obj_dict['tide_id']) })
                
            obj = self.dict_to_obj( obj_dict)
            self.model.movers += obj
    
    def _get_obj( self, coll_, id):
       try:
           return coll_[id] # get object associated with this Id
       except KeyError, e:
           raise KeyError("Collection does not contain an object with id: {0}".format(e.message))
    
    def _add_spills( self, l_spills):
        """ add spills from spills dict (uncertain and certain) to provided model """
        c_spills = self._load_collection(l_spills['certain_spills'])
        if self.model.uncertain:
            u_spills = self._load_collection(l_spills['uncertain_spills'])
            obj_list = zip(c_spills, u_spills)
        else:
            obj_list = c_spills
        [self.model.spills.add(obj) for obj in obj_list]