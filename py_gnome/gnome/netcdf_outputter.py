'''
NetCDF outputter - follows the interface defined by gnome.Outputter for a NetCDF output writer
'''
import copy
import os
from datetime import datetime
from collections import OrderedDict

import netCDF4 as nc
import numpy as np

import gnome
from gnome.outputter import Outputter
from gnome.utilities import serializable, time_utils

class NetCDFOutput(Outputter, serializable.Serializable):

    cf_attributes={'comment' : 'Particle output from the NOAA PyGnome model',
                   'source' : "PyGnome version x.x.x",
                   'references' : 'TBD',
                   'feature_type' : "particle_trajectory" ,
                   'institution' : "NOAA Emergency Response Division",
                   'conventions' : "CF-1.6",
                   }
    
    """ let's keep order the same as original NetCDF """
    data_vars = OrderedDict()
    var = OrderedDict()
    
    # longitude
    var['dtype'] = np.float32
    var['long_name'] = 'longitude of the particle'
    var['units'] = 'degrees_east'
    data_vars['longitude'] = copy.deepcopy(var)
    
    # latitude
    var['long_name'] = 'latitude of the particle'
    var['units'] = 'degrees_north'
    data_vars['latitude'] = copy.deepcopy(var)
    
    # latitude
    var['long_name'] = 'particle depth below sea surface'
    var['units'] = 'meters'
    var['axis'] = 'z positive down'
    data_vars['depth'] = copy.deepcopy(var)
    
    # mass
    var.clear()
    var['dtype'] = np.float32
    var['units'] = 'grams'
    data_vars['mass'] = copy.deepcopy(var)
    
    # age
    var.clear()
    var['dtype'] = np.int32
    var['long_name'] = 'from age at time of release'
    var['units'] = 'seconds'
    data_vars['age'] = copy.deepcopy(var)
    
    # flag
    var.clear()
    var['dtype'] = np.int8
    var['long_name'] = 'particle status flag'
    var['valid_range'] = [0, 5]
    var['flag_values'] = [1, 2, 3, 4],
    var['flag_meanings'] = 'on_land off_maps evaporated below_surface'
    data_vars['flag'] = copy.deepcopy(var)
    
    # status
    var['long_name'] = 'particle status flag'
    var['valid_range'] = [0, 10]
    var['flag_values'] = [2, 3, 7, 10],
    var['flag_meanings'] = '2:in_water 3:on_land 7:off_maps 10:evaporated'
    data_vars['status'] = copy.deepcopy(var)
    
    # id
    var.clear()
    var['dtype'] = np.int8
    var['long_name'] = 'particle ID'
    var['units'] = '1'
    data_vars['id'] = copy.deepcopy(var)

    # This is data that has already been written in standard format
    standard_data = ['positions','current_time_stamp','status_codes','spill_num','age','mass','flag']
    
    def __init__(self, netcdf_filename, cache=None, all_data=False, id=None):
        """
        should netcdf_filename be overwritten of if it already exists?
        """
        self.netcdf_filename = netcdf_filename
        self.cache = cache
        self._uncertain = False
        self._u_netcdf_filename = None
        self.all_data = all_data
        self.arr_types = None   # this is only updated in prepare_for_model_run if all_data is True
        #self.del_on_rewind = del_on_rewind
        
        if os.path.isdir(netcdf_filename):
            raise ValueError("netcdf_filename must be a file not a directory.")
        
        if os.path.exists(netcdf_filename):
            raise ValueError("{0} file exists. Enter a filename that does not exist in which to save data.".format(netcdf_filename))
        
        if not os.path.exists( os.path.realpath(os.path.dirname(netcdf_filename))):
            raise ValueError("{0} does not appear to be a valid path".format(os.path.dirname(netcdf_filename)))
        
        self._gnome_id = gnome.GnomeId(id)
    
    @property
    def id(self):
        return self._gnome_id.id
    
    def prepare_for_model_run(self, cache=None, model_start_time=None, num_time_steps=None, uncertain=False, spills=None, **kwargs):
        """ 
        Write global attributes
        
        :param cache=None: Sets the cache object to be used for the data.
                           If None, it will use the one already set up.
        
        This takes more than standard 'cache' argument. These are required arguments - they contain None for defaults
        because XXX
        :model_start_time: 
        :num_time_steps:
        
        Does not take any other input arguments; however, to keep the interface the same for all outputters,
        define **kwargs for now.
        """
        if cache is not None:
            self.cache = cache
        
        if model_start_time is None or num_time_steps is None:
            raise TypeError("model_start_time and num_time_steps cannot be NoneType")
        
        self._uncertain = uncertain
        
        if self._uncertain:
            name, ext = os.path.splitext(self.netcdf_filename)
            self._u_netcdf_filename = "{0}_uncertain{1}".format(name,ext)
            filenames = (self.netcdf_filename, self._u_netcdf_filename)
        else:
            filenames = (self.netcdf_filename,)
        
        for file_ in filenames:
            with nc.Dataset(file_, 'w', format='NETCDF4') as rootgrp:
                """ Global variables """
                rootgrp.comment = self.cf_attributes['comment']
                rootgrp.creation_date = time_utils.round_time(datetime.now(),roundTo=1).isoformat().replace('T',' ')
                rootgrp.source = self.cf_attributes['source']
                rootgrp.references = self.cf_attributes['references']
                rootgrp.feature_type = self.cf_attributes['feature_type']
                rootgrp.institution = self.cf_attributes['institution']
                rootgrp.convention = self.cf_attributes['conventions']
                
                """ Dimensions """
                rootgrp.createDimension('time', num_time_steps)
                rootgrp.createDimension('data', 0)
                
                """ Variables """
                time_ = rootgrp.createVariable('time', np.double, ('time',))
                time_.units = 'seconds since {0}'.format(model_start_time.isoformat().replace('T',' '))
                time_.long_name = 'time'
                time_.standard_name = 'time'
                time_.calendar = 'gregorian'
                time_.comment = 'unspecfied time zone'
                
                pc = rootgrp.createVariable('particle_count',np.int32, ('time',))
                pc.units = '1'
                pc.long_name = "number of particles in a given timestep"
                pc.ragged_row_count = "particle count at nth timestep"
                
                for key,val in self.data_vars.iteritems():
                    var = rootgrp.createVariable(key, val.get('dtype'), ('data',))  # don't pop since it maybe required twice
                    # iterate over remaining attributes
                    [setattr(var,key2,val2) for key2,val2 in val.iteritems() if key2 != 'dtype']
                    
                """ End standard data. Next create variables for remaining arrays if all_data is True """ 
                if self.all_data:
                    rootgrp.createDimension('world_point', 3)
                    self.arr_types = dict()
                    for spill in spills:
                        at = spill.array_types
                        [self.arr_types.update({key:atype}) for key,atype in at.iteritems() if key not in self.arr_types and key not in self.standard_data]
                    
                    # create variables
                    for key,val in self.arr_types.iteritems():
                        if len(val.shape) == 0:
                            rootgrp.createVariable(key, val.dtype,('data'))    
                        elif val.shape[0] == 3:
                            rootgrp.createVariable(key, val.dtype,('data','world_point'))
                        else:
                            raise ValueError("{0} has an undefined dimension: {1}".format(key,val.shape))
                        
    
    def write_output(self, step_num):
        """ write output at the end of the step"""
        if self.cache is None:
            raise ValueError("cache object is not defined. It is required prior to calling write_output")
        
        for sc in self.cache.load_timestep(step_num).items():
            if sc.uncertain and self._u_netcdf_filename is not None:
                file_ = self._u_netcdf_filename
            else:
                file_ = self.netcdf_filename
            
            time_stamp = sc['current_time_stamp'].item()
            
            with nc.Dataset(file_, 'a') as rootgrp:
                rootgrp.variables['time'][step_num] = nc.date2num( time_stamp, 
                                                                   rootgrp.variables['time'].units,
                                                                   rootgrp.variables['time'].calendar)
                pc = rootgrp.variables['particle_count']
                pc[step_num] = len(sc['status_codes'])
                
                """ write keys that don't map directly to sc variable names """
                ixs = step_num * pc[step_num]   # starting index for writing data in this timestep
                ixe = ixs + pc[step_num]        # ending index for writing data in this timestep
                rootgrp.variables['longitude'][ixs:ixe] = sc['positions'][:,0]
                rootgrp.variables['latitude'][ixs:ixe] = sc['positions'][:,1]
                rootgrp.variables['depth'][ixs:ixe] = sc['positions'][:,2]
                rootgrp.variables['status'][ixs:ixe] = sc['status_codes'][:]
                rootgrp.variables['id'][ixs:ixe] = sc['spill_num'][:]
                
                # write remaining data
                if self.all_data:
                    for key, val in self.arr_types.iteritems():
                        if len(val.shape) == 0:
                            rootgrp.variables[key][ixs:ixe] = sc[key]
                        else:
                            rootgrp.variables[key][ixs:ixe,:] = sc[key]
                    
                
        return {'step_num': step_num,
                'netcdf_filename': (self.netcdf_filename, self._u_netcdf_filename),
                'time_stamp': time_stamp}
            
    
    def rewind(self):
        """ if rewound, delete the file and start over? """
        if os.path.exists(self.netcdf_filename):
            os.remove(self.netcdf_filename)
            
        if self._u_netcdf_filename is not None and os.path.exists(self._u_netcdf_filename):
            os.remove(self._u_netcdf_filename)