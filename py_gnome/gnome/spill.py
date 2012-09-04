#!/usr/bin/env python

"""
spill.py

a new implementation of the spill class(s)

keeps all the data in separate arrays, so we only store and move around the
data that is needed

This is the "magic" class -- it handles the smart allocation of arrays, etc.

"""
import numpy as np

from gnome.cy_gnome import basic_types

class Spill(object):
    """
    Base class for all spills
    
    Needs to be subclassed to do anything useful

    Many of the "fields" associated with a collection of LEs are optional,
    or used only by some movers, so only the ones required will be requested
    by each mover.
    
    The data for the LEs is stored in the _data_arrays dict. They can be
    accessed by indexing:
      
    positions = Spill['positions'] : returns a (num_LEs, 3) array of world_point_types
    """
    
    def __init__(self, num_LEs, initial_positions=(0.0,0.0,0.0)):
        
        self.num_LEs = num_LEs
        self._data_arrays = {}
        
        self._data_arrays['positions'] = np.zeros((num_LEs, 3),
                                                  dtype=basic_types.world_point_type)
        self._data_arrays['positions'][:,:] = initial_positions
        
        self._data_arrays['next_positions'] =  np.zeros_like(self['positions'])

        self._data_arrays['status_codes'] = ( np.zeros((num_LEs,),
                                                       dtype=basic_types.status_code_type)
                                             )
        self._data_arrays['status_codes'][:] = basic_types.status_in_water             
 
    def __getitem__(self, data_name):
        """
        The basic way to access data for the LEs
        
        a KeyError will be raised if the data is not there
        """
        return self._data_arrays[data_name]
    
    def __setitem__(self, data_name, array):
        """
        sets the data item
        
        careful -- this needs to be compatible with the needs of any mover that uses it.
        
        It will be checked to at least be size-consistent with the rest of the
        data, and type-consistent if the data array is being replaced
        """
        array = np.asarray(array)
        
        if len(array) != self.num_LEs:
            raise ValueError("Length of new data arrays must be number of LEs in the Spill")
                        
        if data_name in self._data_arrays:
            # if the array is already here, the type should match        
            if array.dtype !=  self._data_arrays[data_name].dtype:
                raise ValueError("new data array must be the same type")
            # and the shape should match        
            if array.shape !=  self._data_arrays[data_name].shape:
                raise ValueError("new data array must be the same shape")
                    
        self._data_arrays[data_name] = array


class PointReleaseSpill(Spill):
    """
    The simplest real spill class  --  a point release of floating
    non-weathering particles

    """
    def __init__(self, num_LEs, start_position, release_time, windage=(0.01, 0.04)):
        """
        :param num_LEs: number of LEs used for this spill
        :param start_position: location the LEs are released (long, lat, z) (floating point)
        :param release_time: time the LEs are released (datetime object)
        :param windage: the windage range of the LEs (min, max) -- default is (0.01, 0.04) --1% to 4%
        """
        Spill.__init__(self, num_LEs)

        self.release_time = release_time
        self.start_position = start_position
        self.windage = windage

        self.__init_LEs()

    def __init_LEs(self):
        """
        called internally to initialize the data arrays needed
        """
        self._data_arrays['next_positions'] =  np.zeros_like(self['positions'])
        self._data_arrays['windages'] =  np.zeros((self.num_LEs, ),
                                                  dtype = basic_types.windage_type)
        self._data_arrays['last_water_pts'] = np.zeros_like(self['positions'])
        self.reset()

    def release_LEs(self, current_time):
        """
        Release the LEs -- i.e. change their status to in_water
        if the current time is greater than or equal to the release time
        
        :param current_time: datetime object for current time
        """
        if current_time >= self.release_time:
            self['positions'][:] = self.start_position
            self['status_codes'][:] = basic_types.status_in_water
        return None

    def reset(self):
        """
        reset to initial conditions -- i.e. not released, and at the start position
        """
        self['positions'][:] = self.start_position
        self['status_codes'][:] = basic_types.status_not_released
        

## fixme -- is there a need for this, or should we use a flag in the regular
##          version instead?
class PointReleaseSpillUncert(PointReleaseSpill):
    """
    The "uncertainty" version of a Floating Spill
    """  
    def __init__ (self, *args, **kwargs):
        """
        same __init__ as the FloatingSpill
        """
        FloatingSpill.___init__(self, *args, **kwargs)
        
        # what other parameters do we need here?
        # stuff for Uncetainty storage...
        
        
