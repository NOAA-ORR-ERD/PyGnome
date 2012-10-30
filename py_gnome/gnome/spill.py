#!/usr/bin/env python

"""
spill.py

a new implementation of the spill class(s)

keeps all the data in separate arrays, so we only store and move around the
data that is needed

This is the "magic" class -- it handles the smart allocation of arrays, etc.

"""
import numpy as np

from gnome import basic_types
from gnome.utilities import rand    # not to confuse with python random module

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
    
    def __init__(self, num_LEs, initial_positions=(0.0,0.0,0.0), uncertain=False):
        
        self.num_LEs = num_LEs
        self.is_uncertain = False   # uncertainty spill - same information as basic_types.spill_type
        
        self._data_arrays = {}
        
        self._data_arrays['positions'] = np.zeros((num_LEs, 3),
                                                  dtype=basic_types.world_point_type)
        self._data_arrays['positions'][:,:] = initial_positions
        self._data_arrays['next_positions'] =  np.zeros_like(self['positions'])
        self._data_arrays['last_water_positions'] = np.zeros_like(self['positions'])

        self._data_arrays['status_codes'] = ( np.zeros((num_LEs,),
                                                       dtype=basic_types.status_code_type)
                                             )
        self._data_arrays['status_codes'][:] = basic_types.oil_status.in_water
        
        self._data_arrays['windages'] =  np.zeros((self.num_LEs, ),
                                                  dtype = basic_types.windage_type)             
 
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

    def prepare_for_model_step(self, current_time, time_step=None, uncertain_on=None):
        """
        Do whatever needs to be done at the beginning of a time step:
        
        In this case nothing.
        """
        return None

    def __str__(self):
        msg = ["gnome.spill.Spill(num_LEs=%i)\n"%self.num_LEs]
        msg.append("spill LE attributes: %s"%self._data_arrays.keys())
        return "".join(msg)

    __repr__ = __str__ # should write a better one, I suppose

class PointReleaseSpill(Spill):
    """
    The simplest real spill class  --  a point release of floating
    non-weathering particles

    """
    def __init__(self, num_LEs, start_position, release_time, windage=(0.01, 0.04), persist=900):
        """
        :param num_LEs: number of LEs used for this spill
        :param start_position: location the LEs are released (long, lat, z) (floating point)
        :param release_time: time the LEs are released (datetime object)
        :param windage: the windage range of the LEs (min, max, persistence, step_length). 
        Default is (0.01, 0.04, -1, 900) from 1% to 4%. The -1 means the persistence is infinite so it is only set 
        at the beginning of the model run and not updated at each time step. The last parameter is the step size 
        of the simulation. The default is 15mins.
        """
        Spill.__init__(self, num_LEs)

        self.release_time = release_time
        self.start_position = start_position
        self.windage_range  = windage[0:2]
        self.windage_persist= persist

        if persist <= 0:
            # if it is anything less than 0, treat it as -1 flag
            self.update_windage(0)

        self.__init_LEs()

    def __init_LEs(self):
        """
        called internally to initialize the data arrays needed
        """
        self.reset()
    
    def prepare_for_model_step(self, current_time, time_step, uncertain_on=False):
        """
        Do whatever needs to be done at the beginning of a time step:
        
        In this case:
        
        Release the LEs -- i.e. change their status to in_water
        if the current time is greater than or equal to the release time
        
        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds
        :param uncertain_on: Boolean flag indicating whether uncertainty is on in the model

        """
        if current_time >= self.release_time and not self.released:
            self['status_codes'][:] = basic_types.oil_status.in_water
            self.released = True
            
        if self.windage_persist > 0:
            self.update_windage(time_step)
        
        return None

    def update_windage(self, time_step):
        """
        Update windage for each LE for each time step
        May want to cythonize this to speed it up
        """
        for le in range(0, self.num_LEs):
            self['windages'][le] = rand.random_with_persistance(self.windage_range[0],
                                                                self.windage_range[1],
                                                                self.windage_persist,
                                                                time_step)

    def reset(self):
        """
        reset to initial conditions -- i.e. not released, and at the start position
        """
        self['positions'][:] = self.start_position
        self['status_codes'][:] = basic_types.oil_status.not_released
        self.released = False

        
