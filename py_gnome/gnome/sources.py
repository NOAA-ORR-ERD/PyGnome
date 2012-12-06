#!/usr/bin/env python

"""
sources.py

classes for various types of sources of elements for GNOME

these are managed by the spill class
"""
import sys
import numpy as np
from gnome import basic_types

class Spill(object):
    """
    base class for a source of elements

    """

    # info about the array types
    # used to construct and expand data arrays.
    # this specifies both what arrays are there, and their types, etc.
    # this kept as a class atrribure so all properties are accesable everywhere.
    # subclasses should add new particle properties to this dict
    #             name           shape (not including first axis)       dtype         
    _array_info = {'positions':      ( (3,), basic_types.world_point_type),
                  'next_positions': ( (3,), basic_types.world_point_type),
                  'last_water_positions': ( (3,), basic_types.world_point_type),
                  'status_codes': ( (), basic_types.world_point_type),
                  'spill_id': ( (), basic_types.id_type)
                  }

    __all_ids = set() # set of all the in-use ids

    def __init__(self):

        self.reset()
        self.__set_id()

        # self._data_arrays['positions'] = np.zeros((0, 3),
        #                                           dtype=basic_types.world_point_type)
        # self._data_arrays['positions'][:,:] = initial_positions
        # self._data_arrays['next_positions'] =  np.zeros_like(self['positions'])
        # self._data_arrays['last_water_positions'] = np.zeros_like(self['positions'])

        # self._data_arrays['status_codes'] = ( np.zeros((0,),
        #                                                dtype=basic_types.status_code_type)
        #                                      )
        # self._data_arrays['status_codes'][:] = basic_types.oil_status.in_water
        
        # self._data_arrays['windages'] =  np.zeros((self.num_LEs, ),
        #                                           dtype = basic_types.windage_type)    

    def __set_id(self):
        """
        returns an id that is not already in use

        inefficient, but who cares?
        """
        id = 1
        while id < 65536: # just so it will eventually terminate!
            if id not in self.__all_ids:
                self.id = id
                self.__all_ids.add(id)
                break
            else:
                id+=1

    def __del__(self):
        """
        called when instance is deleted:

        removes its id from Spill.__all_ids
        """
        self.__all_ids.remove(self.id)

    def reset(self):
        pass
        
        # # create a set of zero-size arrays
        # for name, (shape, dtype, initilizer) in self._array_info.items():
        #     self._data_arrays[name] = np.empty( (0,)+shape, dtype=dtype)
        #     self._data_arrays[name][:] = initilizer

    def create_new_elements(self, num_elements):
        """
        create new arrays for the various types and 
        return a dict of the set

        :param num_elements: number of new elements to add
        """
        new_arrays = {}
        for name, (shape, dtype) in self._array_info.items():
            new = np.zeros( (0,)+shape, dtype=dtype)
            new_arrays[name] = new
        self.initialize_new_elements(arrays)
        return new_arrays

    def initialize_new_elements(self, arrays):
        """
        initilize the new elements just created
        This is probably need to be overridden by subclasses
        """
        pass


class FloatingSpill(Spill):
    """
    spill for floating objects

    all this does is add the 'windage' parameter
    """
    def __init__(self,
                 windage_range=(0.01, 0.04),
                 windage_persist=900):

        super(FloatingSpill, self).__init__()

        self._array_info['windages'] = ( (), basic_types.windage_type, 0.0 )

class SurfaceReleaseSpill(FloatingSpill):
    """
    The simplest spill source class  --  a point release of floating
    non-weathering particles

    """
    def __init__(self,
                 num_elements,
                 start_position,
                 release_time,
                 end_position=None,
                 end_release_time=None,
                 windage_range=(0.01, 0.04),
                 windage_persist=900,
                 ):
        """
        :param num_elements: total number of elements used for this spill
        :param start_position: location the LEs are released (long, lat, z) (floating point)
        :param release_time: time the LEs are released (datetime object)
        :param end_position=None: optional -- for a moving source, the end position
        :param end_release_time=None: optional -- for a release over time, the end release time
        :param windage: the windage range of the LEs (min, max). Default is (0.01, 0.04) from 1% to 4%.
        :param persist: Default is 900s, so windage is updated every 900 sec.
                        The -1 means the persistence is infinite so it is only set at the beginning of the run.
        :param uncertain: flag determines whether spill is uncertain or not
        """
        super(SurfaceReleaseSpill, self).__init__(windage_range, windage_persist)

        self.release_time = release_time
        if end_release_time is None:
            self.end_release_time = release_time
        else:
            self.end_release_time = end_release_time
        self.start_position = start_position
        if end_position is None:
            self.end_position = start_position
        else:
            self.end_position = end_position
        self.windage_range  = windage[0:2]
        self.windage_persist= persist

        if persist <= 0:
            # if it is anything less than 0, treat it as -1 flag
            self.update_windage(0)
    
    def get_new_elements(self, current_time, time_step):
        """
        Release any new elements to be added to the spill
                
        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds

        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements

        """
        if current_time >= self.release_time and not self.released:
            #self['status_codes'][:] = basic_types.oil_status.in_water
            #self.released = True

            # compute how many elements to release:
            dt = (self.end_release_time - self.release_time).total_seconds
            num = self.num_elements / (dt / time_step)

            return create_elements(self, num)

        else:
            #nothing to be done...
            return None

    def update_windage(self, time_step):
        """
        Update windage for each LE for each time step
        May want to cythonize this to speed it up
        """
        self['windages'][:] = rand.random_with_persistance(self.windage_range[0],
                                                           self.windage_range[1],
                                                           self.windage_persist,
                                                           time_step,
                                                           array_len=self.num_LEs)

    def reset(self):
        """
        reset to initial conditions -- i.e. not released, and at the start position
        """
        self['positions'][:] = self.start_position
        self['status_codes'][:] = basic_types.oil_status.not_released
        self.released = False




