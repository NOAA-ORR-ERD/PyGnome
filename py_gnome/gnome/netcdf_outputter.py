'''
NetCDF outputter - follows the interface defined by gnome.Outputter for a NetCDF output writer
'''
import copy
import os
from datetime import datetime

import netCDF4 as nc
import numpy as np

import gnome
from gnome.outputter import Outputter
from gnome.utilities import serializable, time_utils

class NetCDFOutput(Outputter, serializable.Serializable):

    cf_attributes={'comment' : 'Particle output from the NOAA PyGnome model',
                   'source' : "PyGnome version x.x.x",
                   'references' : 'TBD',
                   'CF:featureType' : "particle_trajectory" ,
                   'institution' : "NOAA Emergency Response Division",
                   'conventions' : "CF-1.6",
                   'history' : "Evolved with discussion on CF-metadata listserve",
                   #'title' : "Sample data/file for particle trajectory format",
                   }
    
    data_vars= { 'longitude':{'dtype':np.float32,
                              'units':'degrees_east',
                              'long_name':'longitude of the particle'},
                 
                 'latitude': {'dtype':np.float32,
                              'units':'degrees_north',
                              'long_name':'latitude of the particle'},
                 
                 'depth':    {'dtype':np.float32,
                              'units':'meters',
                              'long_name':'particle depth below sea surface',
                              'axis': 'z positive down'},
                 
                 'mass':     {'dtype':np.float32,
                              'units':'grams'},
                 
                 'age':      {'dtype':np.int32,
                              'units':'seconds',
                              'long_name':'from age at time of release'},
                 
                 'status':   {'dtype':np.int8,
                              'long_name': 'particle status flag',
                              'valid_range': '0, 10',
                              'flag_values': '2, 3, 7, 10',
                              'flag_meanings': '2:in_water 3:on_land 7:off_maps 10:evaporated'
                              },
                 
                 'id':       {'dtype':np.int8,
                              'long_name': 'particle ID',
                              'units' : '1'
                              },
                 }


    def __init__(self, netcdf_filename, cache=None, write_alldata=False):
        """
        should netcdf_filename be overwritten of if it already exists?
        """
        self.netcdf_filename = netcdf_filename
        self.cache = cache
        self._uncertain = False
        self._u_netcdf_filename = None
        self.write_alldata = write_alldata
        #self.del_on_rewind = del_on_rewind
        
        if os.path.exists(netcdf_filename):
            raise ValueError("{0} file exists. Enter a filename that does not exist in which to save data.".format(netcdf_filename))
        
        if os.path.isdir(netcdf_filename):
            raise ValueError("netcdf_filename must be a file not a directory.")
        
        if not os.path.exists( os.path.realpath(os.path.dirname(netcdf_filename))):
            raise ValueError("{0} does not appear to be a valid path".format(os.path.dirname(netcdf_filename)))
        
    
    def prepare_for_model_run(self, cache=None, **kwargs):
        """ 
        Write global attributes
        
        :param cache=None: Sets the cache object to be used for the data.
                           If None, it will use the one already set up.
        
        Since this takes more than standard 'cache' argument, those are in kwargs.
        These are required arguments, so they do not contain defaults.
        :model_start_time: 
        :num_time_steps:
        
        Does not take anyother input arguments; however, to keep the interface the same for all outputters,
        define **kwargs for now.
        """
        if cache is not None:
            self.cache = cache
        
        model_start_time = kwargs.pop('model_start_time')
        num_time_steps = kwargs.pop('num_time_steps')
        
        self._uncertain = kwargs.pop('uncertain',self._uncertain)
        
        if self._uncertain:
            name, ext = os.path.splitext(self.netcdf_filename)
            self._u_netcdf_filename = "{0}_uncertain{1}".format(name,ext)
            filenames = (self.netcdf_filename, self._u_netcdf_filename)
        else:
            filenames = (self.netcdf_filename,)
        
        for file_ in filenames:
            with nc.Dataset(file_, 'w', format='NETCDF4') as rootgrp:
                """ Global variables """
                rootgrp.convention = self.cf_attributes['conventions']
                rootgrp.institution = self.cf_attributes['institution']
                rootgrp.source = self.cf_attributes['source']
                rootgrp.history = self.cf_attributes['history']
                rootgrp.comment = self.cf_attributes['comment']
                rootgrp.creation_date = time_utils.round_time(datetime.now(),roundTo=1).isoformat().replace('T',' ')
                
                """ Dimensions """
                rootgrp.createDimension('time', num_time_steps)
                rootgrp.createDimension('data', 0)
                
                """ Variables """
                time_ = rootgrp.createVariable('time', np.double, ('time',))
                time_.units = 'seconds since {0}'.format(model_start_time.isoformat().replace('T',' '))
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
                    
                #if self.write_alldata:
                    # write all data
            
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
                
            return {'step_num': step_num,
                    'netcdf_filename': (self.netcdf_filename, self._u_netcdf_filename),
                    'time_stamp': time_stamp}
            
    
    def rewind(self):
        """ if rewound, delete the file and start over? """
        os.remove(self.netcdf_filename)
        if self._u_netcdf_filename is not None:
            os.remove(self._u_netcdf_filename)