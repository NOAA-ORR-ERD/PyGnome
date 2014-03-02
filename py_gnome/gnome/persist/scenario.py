'''
Created on Mar 7, 2013

load / save a py_gnome scenario
'''

import os
import json
import glob
import string
import shutil

#import netCDF4 as nc

import gnome    # used implicitly by eval()
from gnome.outputters import NetCDFOutput

from gnome.persist import (
    modules_dict,
    environment_schema,     # used implicitly by eval()
    model_schema,           # used implicitly by eval()
    movers_schema,          # used implicitly by eval()
    weatherers_schema,      # used implicitly by eval()
    spills_schema,          # used implicitly by eval()
    map_schema,             # used implicitly by eval()
    outputters_schema       # used implicitly by eval()
    )


class Scenario(object):
    """
    Create a class that contains functionality to load/save a model scenario
    """
    def __init__(self, saveloc, model=None):
        """
        Constructor for a Scenario object. It's main function is to either
        'save' and model or to 'load' an already existing model. If a model is
        loaded from saveloc, the 'model' attribute will contain the re-created
        model object.

        'saveloc' is given as /path_to_save_directory/save_directory
        if /path_to_save_directory does not exist, then a ValueError is raised
        if save_directory does not exist, it is created and is empty

        If save_directory exists but is not empty, then the save() method
        will clobber all the files in here. The save_directory only contains
        all the files associated with a single PyGnome Model

        All object's being persisted must use the serializable.Serializable
        class as a mixin or they must define the same methods/attributes.

        :param saveloc: A valid directory. Model files are either persisted
                        here or a new model is re-created from the files
                        stored here. The files are clobbered when save() is
                        called.
        :type saveloc: A path as a string or unicode
        :param model: A model object. Only required if save method is invoked.
        :type model: gnome.model.Model object

        .. note:: If user wants to save a scenario, then model must be set
        """
        path_, savedir = os.path.split(saveloc)
        if path_ == '':
            path_ = '.'

        if not os.path.exists(path_):
            raise ValueError('"{0}" does not exist. \nCannot create "{1}"'
                             .format(path_, savedir))

        if not os.path.exists(saveloc):
            os.mkdir(saveloc)

        self.saveloc = saveloc
        self.model = model
        self._certainspill_data = os.path.join(self.saveloc,
                                               'spills_data_arrays.nc')

        # will be updated when _certainspill_data is saved
        self._uncertainspill_data = None

    def save(self):
        """
        Method used to save the model _state to saveloc
        It saves the _state of each object, including the model in JSON format
        in *.json files.

        Clobber any existing files in 'saveloc'. It will only contain the
        files associated with 'model' that user wishes to save.
        """
        self._empty_save_dir()

        if self.model is None:
            raise AttributeError('A model needs to be defined before it'\
                                 ' can be saved')

        json_ = self.model.serialize('create')
        self._save_json_to_file(json_, self.model, self.model.id)

        json_ = self.model.map.serialize('create')
        self._save_json_to_file(json_, self.model.map)

        self._save_collection(self.model.movers)
        self._save_collection(self.model.weatherers)
        self._save_collection(self.model.environment)
        self._save_collection(self.model.outputters)

        for sc in self.model.spills.items():
            self._save_collection(sc.spills)

        # persist model _state since middle of run
        if self.model.current_time_step > -1:
            self._save_spill_data()

#==============================================================================
#     def load(self):
#         """
#         reconstruct the model from saveloc. It stores the re-created model
#         inside 'model' attribute. Function also returns the recreated model.
# 
#         :returns: a model object re-created from the save files
#         """
#         model_dict = self.load_model_dict()
# 
#         # pop lists that correspond with ordered collections
#         # create a list of associated objects and put it back into model_dict
# 
#         l_movers = model_dict.pop('movers')
#         l_weatherers = model_dict.pop('weatherers')
#         l_environment = model_dict.pop('environment')
#         l_outputters = model_dict.pop('outputters')
#         l_spills = model_dict.pop('spills')
# 
#         # load objects in a list and add that back to model_dict
# 
#         model_dict['environment'] = self._load_collection(l_environment)
#         model_dict['outputters'] = self._load_collection(l_outputters)
#         model_dict['certain_spills'] = \
#             self._load_collection(l_spills['certain_spills'])
#         if ('uncertain' in model_dict and model_dict['uncertain']):
#             model_dict['uncertain_spills'] = \
#                 self._load_collection(l_spills['uncertain_spills'])
# 
#         model_dict['movers'] = \
#             self._load_movers_collection(l_movers, model_dict['environment'])
# 
#         model_dict['weatherers'] = self._load_collection(l_weatherers)
# 
#         self.model = self.dict_to_obj(model_dict)
# 
#         self._load_spill_data()
#         return self.model
#==============================================================================

    def _save_collection(self, coll_):
        """
        Function loops over an orderedcollection or any other iterable
        containing a list of objects. It calls the to_dict method for each
        object, then converts it o valid json (dict_to_json),
        and finally saves it to file (_save_json_to_file)

        :param OrderedCollection coll_: ordered collection to be saved
        """
        for count, obj in enumerate(coll_):
            json_ = obj.serialize('create')
            self._save_json_to_file(json_, obj, '_{0}'.format(count))

    def _save_json_to_file(self, data, obj, append=''):
        """
        write json data to file

        :param dict data: JSON data to be saved
        :param obj: gnome object corresponding w/ data
        """

        fname = os.path.join(self.saveloc,
                     '{0}{1}.json'.format(obj.__class__.__name__, append))
        data = self._move_data_file(data)  # if there is a

        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=True)

    def _move_data_file(self, to_json):
        """
        Look at _state attribute of object. Find all fields with 'isdatafile'
        attribute as True. If there is a key in to_json corresponding with
        'name' of the fields with True 'isdatafile' attribute then move that
        datafile and update the key in the to_json to point to new location
        """
        _state = eval('{0}._state'.format(to_json['obj_type']))
        fields = _state.get_field_by_attribute('isdatafile')

        for field in fields:
            if field.name not in to_json:
                continue

            value = to_json[field.name]

            if os.path.exists(value) and os.path.isfile(value):
                shutil.copy(value, self.saveloc)
                to_json[field.name] = os.path.split(to_json[field.name])[1]

        return to_json

#==============================================================================
#     def json_to_dict(self, json_):
#         """
#         Function used when loading a model scenario.
#         convert the dict returned by object's to_dict method to valid json
#         format via colander schema
# 
#         :param dict json_: JSON data to be converted
#         """
#         gnome_mod, obj_name = json_['obj_type'].rsplit('.', 1)
# 
#         to_eval = ('{0}.{1}().deserialize(json_)'
#                    .format(modules_dict[gnome_mod], obj_name))
#         _to_dict = eval(to_eval)
# 
#         return _to_dict
# 
#     def dict_to_obj(self, obj_dict):
#         '''
#         create object from a dict. The dict contains (keyword,value) pairs
#         used to create new object
#         '''
#         type_ = obj_dict.pop('obj_type')
#         to_eval = '{0}.new_from_dict(obj_dict)'.format(type_)
#         obj = eval(to_eval)
# 
#         return obj
# 
#     def load_model_dict(self):
#         '''
#         Load model dict from *.json file.
#         Pop 'map' key, create 'map' object and add to model dict.
#         This dict is used in Model.new_from_dict(dict_) to create new Model
#         '''
#         model_file = glob.glob(os.path.join(self.saveloc, 'Model_*.json'
#                                ))
#         if model_file == []:
#             raise ValueError('No Model_*.json files find in {0}'\
#                              .format(self.saveloc))
#         elif len(model_file) > 1:
#             raise ValueError("multiple Model_*.json files found in {0}. Please"
#                              " provide 'filename'".format(self.saveloc))
#         else:
#             model_file = model_file[0]
# 
#         model_json = self._load_json_from_file(model_file)
#         model_dict = self.json_to_dict(model_json)
# 
#         # create map object and add to model_dict
# 
#         map_type, map_id = model_dict['map']
#         obj_json = self._find_and_load_json_file(map_id)
# 
#         dict_ = self.json_to_dict(obj_json)
#         map_ = self.dict_to_obj(dict_)
# 
#         model_dict['map'] = map_  # replace map object in the dict
# 
#         return model_dict
# 
#     def _load_json_from_file(self, fname):
#         '''
#         Look at _state attribute of object. Find all fields with
#         'isdatafile' attribute as True. If there is a key in json_data
#         corresponding with 'name' of the fields with True 'isdatafile'
#         attribute, then append the saveloc path to the value
#         '''
#         with open(fname, 'r') as infile:
#             json_data = json.load(infile)
# 
#         _state = eval('{0}._state'.format(json_data['obj_type']))
#         fields = _state.get_field_by_attribute('isdatafile')
# 
#         for field in fields:
#             if field.name not in json_data:
#                 continue
#             json_data[field.name] = os.path.join(self.saveloc,
#                     json_data[field.name])
# 
#         return json_data
# 
#     def _find_and_load_json_file(self, id_):
#         """
#         Given the id of the object, find the *_{id}.json file that contains
#         json of the object and load it.
#         """
# 
#         obj_file = glob.glob(os.path.join(self.saveloc,
#                              '*_{0}.json'.format(id_)))
# 
#         if len(obj_file) == 0:
#             msg = 'No filename containing *_{0}.json found in {1}'
#             raise IOError(msg.format(id_, os.path.abspath('.')))
#         elif len(obj_file) > 1:
#             msg = 'Cannot have two objects with same Id. Multiple'\
#                   ' filenames containing *_{0}.json found in {1}'
#             raise IOError(msg.format(id_, os.path.abspath(self.saveloc)))
# 
#         obj_file = obj_file[0]
#         obj_json = self._load_json_from_file(os.path.abspath(obj_file))
# 
#         return obj_json
# 
#     def _load_collection(self, coll_dict):
#         """
#         Load collection - dict contains output of OrderedCollection.to_dict()
# 
#         'dtype' - currently not used for anything
#         'id_list' - for each object in list, use this to find and load the
#                     json file, convert it to a valid dict, then create a new
#                     object using new_from_dict 'id_list' contains a list of
#                     tuples (object_type, id of object)
# 
#         :returns: a list of objects corresponding with the data in 'id_list'
# 
#         .. note:: while this applies to ordered collections. It can work for
#                   any iterable that contains 'id_list' in the dict with above
#                   format.
#         """
#         obj_list = []
# 
#         for info in coll_dict['id_list']:
#             id_ = info[1]
#             obj_json = self._find_and_load_json_file(id_)
#             dict_ = self.json_to_dict(obj_json)
#             obj = self.dict_to_obj(dict_)
#             obj_list.append(obj)
# 
#         return obj_list
# 
#     def _load_movers_collection(self, movers_dict, l_env):
#         """
#         add movers to the model - dict contains the output of
#         OrderedCollection.to_dict()
# 
#         'dtype' - not used for anything
#         'id_list' - for each object in list, use this to find and load the json
#                     file, convert it to a valid dict, then create a new object
#                     using new_from_dict 'id_list' contains a list of tuples
#                     (object_type, id of object)
# 
#         .. note:: If Wind object and Tide object are present, the objects must
#                   be created and part of a list passed in as l_env
#         """
#         obj_list = []
# 
#         for (type_, id_) in movers_dict['id_list']:
#             obj_json = self._find_and_load_json_file(id_)
#             obj_name = string.rsplit(type_, '.', 1)[-1]
# 
#             obj_dict = self.json_to_dict(obj_json)
# 
#             if obj_name == 'WindMover':
#                 obj_dict.update({'wind': self._get_obj(l_env,
#                                 obj_dict['wind_id'])})
#             elif (obj_name == 'CatsMover' and
#                   obj_dict.get('tide_id') is not None):
#                 obj_dict.update({'tide': self._get_obj(l_env,
#                                 obj_dict['tide_id'])})
# 
#             obj = self.dict_to_obj(obj_dict)
#             obj_list.append(obj)
# 
#         return obj_list
# 
#     def _get_obj(self, list_, id_):
#         """
#         Get object by ID from list of objects
#         """
#         obj = [obj for obj in list_ if id_ in obj.id]
#         if len(obj) == 0:
#             msg = 'List does not contain an object with id_: {0}'
#             raise ValueError(msg.format(id_))
# 
#         if len(obj) > 1:
#             msg = 'List contains more than one object with id_: {0}'
#             raise ValueError(msg.format(id_))
# 
#         return obj[0]
# 
#     def _save_spill_data(self):
#         """ save the data arrays for current timestep to NetCDF """
#         nc_out = NetCDFOutput(self._certainspill_data,
#                               which_data='all',
#                               cache=self.model._cache)
#         nc_out.prepare_for_model_run(model_start_time=self.model.start_time,
#                                      uncertain=self.model.uncertain,
#                                      spills=self.model.spills)
#         nc_out.write_output(self.model.current_time_step)
#         self._uncertainspill_data = nc_out._u_netcdf_filename
# 
#     def _load_spill_data(self):
#         """ load NetCDF file and add spill data back in """
#         if not os.path.exists(self._certainspill_data):
#             return
# 
#         array_types = {}
# 
#         for m in self.model.movers:
#             array_types.update(m.array_types)
# 
#         for w in self.model.weatherers:
#             array_types.update(w.array_types)
# 
#         for sc in self.model.spills.items():
#             if sc.uncertain:
#                 data = NetCDFOutput.read_data(self._uncertainspill_data,
#                                               time=None,
#                                               which_data='all')
#             else:
#                 data = NetCDFOutput.read_data(self._certainspill_data,
#                                               time=None,
#                                               which_data='all')
# 
#             sc.current_time_stamp = data.pop('current_time_stamp').item()
#             sc._data_arrays = data
#             sc._array_types.update(array_types)
#

    def _empty_save_dir(self):
        '''
        Remove all files, directories under self.saveloc

        First clean out directory, then add new save files
        This should only be called by self.save()
        '''
        (dirpath, dirnames, filenames) = os.walk(self.saveloc).next()

        if dirnames:
            for dir_ in dirnames:
                shutil.rmtree(os.path.join(dirpath, dir_))

        if filenames:
            for file_ in filenames:
                os.remove(os.path.join(dirpath, file_))
