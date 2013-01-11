#!/usr/bin/env python

"""
spill.py - A new implementation of the spill class(s)

We now keep all the data in separate arrays, so we only store and move around the
data that is needed

This is the "magic" class -- it handles the smart allocation of arrays, etc.

these are managed by the SpillContainer class
"""

import sys
import copy
import numpy as np
from gnome import basic_types

class Spill(object):
    """
    base class for a source of elements


    NOTE: It's important to dereive all Spills from this base class, as all sorts of
          trickery to keep track of spill ids, and what instances there are of
          derived classes, so that we can keep track of whatdata arrays are needed, etc.

    """

    # info about the array types
    # used to construct and expand data arrays.
    # this specifies both what arrays are there, and their types, etc.
    # this kept as a class atrribure so all properties are accesable everywhere.
    # subclasses should add new particle properties to this dict
    #             name           shape (not including first axis)       dtype         
    _array_info = {}

    __all_ids = set() # set of all the in-use ids
    __all_subclasses = {} # keys are the instance id -- values are the subclass object
    def __init__(self, num_elements=0):

        self.num_elements = num_elements

        self.is_active = True       # sets whether the spill is active or not

        self.__set_id()
        self.__all_subclasses[ id(self) ] = self.__class__
        Spill.reset_array_types()

    def __deepcopy__(self, memo=None):
        """
        the deepcopy implementation

        we need this, as we don't want the ids copied, but do want everything else.

        got the method from:

        http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy() ends up using recursion
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.deepcopy(self.__dict__, memo)
        obj_copy.__set_id()
        return obj_copy

    def __copy__(self):
        """
        might as well have copy, too.
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = self.__dict__.copy()
        obj_copy.__set_id()
        return obj_copy


    @classmethod
    def reset_array_types(cls):
        cls._array_info.clear()
        for subclass in cls.__all_subclasses.values():
            subclass.add_array_types()

    @classmethod
    def add_array_types(cls):
        cls._array_info.update({'positions':      ( (3,), basic_types.world_point_type),
                                'next_positions': ( (3,), basic_types.world_point_type),
                                'last_water_positions': ( (3,), basic_types.world_point_type),
                                'status_codes': ( (), basic_types.status_code_type),
                                'spill_id': ( (), basic_types.id_type)
                                })

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

        removes its id from Spill.__all_ids and instance from Spill.__all_subclass_instances
        """
        self.__all_ids.remove(self.id)
        del self.__all_subclasses[ id(self) ]

    def reset(self):
        """
        reset the Spill to original status
        """
        Spill.reset_array_types()        

    def release_elements(self, current_time, time_step=None):
        """
        probably overridden by a subclass
        """
        return None

    def create_new_elements(self, num_elements):
        """
        create new arrays for the various types and 
        return a dict of the set

        :param num_elements: number of new elements to add
        """
        arrays = {}
        for name, (shape, dtype) in self._array_info.items():
            arrays[name] = np.zeros( (num_elements,)+shape, dtype=dtype)
        self.initialize_new_elements(arrays)
        return arrays

    def initialize_new_elements(self, arrays):
        """
        initilize the new elements just created
        This is probably need to be extended by subclasses
        """
        arrays['spill_id'][:] = self.id
        arrays['status_codes'][:] = basic_types.oil_status.in_water

class FloatingSpill(Spill):
    """
    spill for floating objects

    all this does is add the 'windage' parameter
    """
    def __init__(self,
                 windage_range=(0.01, 0.04),
                 windage_persist=900):

        super(FloatingSpill, self).__init__()

        self.add_array_types()

    @classmethod
    def add_array_types(cls):
        # need to get the superclasses types added too.
        super(FloatingSpill, cls).add_array_types()
        cls._array_info['windages'] = ( (), basic_types.windage_type )

    ##fixme: where does this go????
    @classmethod
    def update_windage(self, windages, time_step):
        """
        Update windage for each LE for each time step
        May want to cythonize this to speed it up
        """
        windages[:] = rand.random_with_persistance(self.windage_range[0],
                                                           self.windage_range[1],
                                                           self.windage_persist,
                                                           time_step,
                                                           array_len=len(wiindages),
                                                           )

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
        """
        super(SurfaceReleaseSpill, self).__init__(windage_range, windage_persist)
        
        self.num_elements = num_elements
        
        self.release_time = release_time
        if end_release_time is None:
            self.end_release_time = release_time
        else:
            if release_time > end_release_time:
                raise ValueError("end_release_time must be greater than release_time")
            self.end_release_time = end_release_time

        if end_position is None:
            end_position = start_position
        self.start_position = np.asarray(start_position, dtype=basic_types.world_point_type).reshape((3,))
        self.end_position = np.asarray(end_position, dtype=basic_types.world_point_type).reshape((3,))
        self.windage_range    = windage_range[0:2]
        self.windage_persist  = windage_persist

#        if windage_persist <= 0:
#            # if it is anything less than 0, treat it as -1 flag
#            self.update_windage(0)

        self.num_released = 0
        self.prev_release_pos = self.start_position

    def initialize_new_elements(self, arrays):
        """
        initilize the new elements just created
        This is probably need to be extended by subclasses
        """
        super(SurfaceReleaseSpill, self).initialize_new_elements(arrays)
        arrays['positions'][:] = self.start_position

    def release_elements(self, current_time, time_step=None):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds -- this version doesn't use this

        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements

        NOTE: this version releases elements spread out along the line from
              start_position to end_position. If the release spans multiple
              time steps, the first will be spread out a little more, so that
              there will be an element released at botht he start and end,
              but no duplicate points. 
        """

        if current_time >= self.release_time:
            if self.num_released >= self.num_elements:
                return None

            #total release time
            release_delta = (self.end_release_time - self.release_time).total_seconds()

            # time since release began
            if current_time >= self.end_release_time:
                dt = release_delta
            else:
                dt = max( (current_time - self.release_time).total_seconds(), 0.0)
            
            # this here in case there is less than one released per time step.
            # or if the relase time is before the model start time
            if release_delta == 0: #instantaneous release
                num = self.num_elements - self.num_released #num_released shoudl always be 0?
            else:
                total_num = (dt / release_delta) * self.num_elements
                num = int(total_num - self.num_released)

            if num <= 0:
                return None

            self.num_released += num

            arrays = self.create_new_elements(num)

            #compute the position of the elements:
            if release_delta == 0: # all released at once:
                x1, y1 = self.start_position[:2]
                x2, y2 = self.end_position[:2]
                arrays['positions'][:,0] = np.linspace(x1, x2, num)
                arrays['positions'][:,1] = np.linspace(y1, y2, num)
            else:
                x1, y1 = self.prev_release_pos[:2]
                dx = self.end_position[0] - self.start_position[0]
                dy = self.end_position[1] - self.start_position[1]

                fraction = min (1, dt / release_delta)
                x2 = (fraction * dx) + self.start_position[0]
                y2 = (fraction * dy) + self.start_position[1]
                    

                if np.array_equal(self.prev_release_pos, self.start_position):
                    # we want both the first and last points
                    arrays['positions'][:,0] = np.linspace(x1, x2, num)
                    arrays['positions'][:,1] = np.linspace(y1, y2, num)
                else:
                    # we don't want to duplicate the first point.
                    arrays['positions'][:,0] = np.linspace(x1, x2, num+1)[1:]
                    arrays['positions'][:,1] = np.linspace(y1, y2, num+1)[1:]
                self.prev_release_pos = (x2, y2, 0.0)
            return arrays
        else:
            return None

    def reset(self):
       """
       reset to initial conditions -- i.e. nothing released. 
       """
       super(SurfaceReleaseSpill, self).reset()

       self.num_released = 0
       self.prev_release_pos = self.start_position

class SpatialReleaseSpill(FloatingSpill):
    """
    A simple spill  class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified

    """
    def __init__(self,
                 start_positions,
                 release_time,
                 windage_range=(0.01, 0.04),
                 windage_persist=900,
                 ):
        """
        :param start_positions: locations the LEs are released (num_elements X 3): (long, lat, z) (floating point)
        :param release_time: time the LEs are released (datetime object)
        :param windage: the windage range of the LEs (min, max). Default is (0.01, 0.04) from 1% to 4%.
        :param persist: Default is 900s, so windage is updated every 900 sec.
                        The -1 means the persistence is infinite so it is only set at the beginning of the run.
        """
        super(SpatialReleaseSpill, self).__init__(windage_range, windage_persist)
        
        self.start_positions = np.asarray(start_positions,
                                          dtype=basic_types.world_point_type).reshape((-1, 3))
        self.num_elements = self.start_positions.shape[0]
        
        self.release_time = release_time

        self.elements_not_released = True

        self.windage_range    = windage_range[0:2]
        self.windage_persist  = windage_persist

    def initialize_new_elements(self, arrays):
        """
        initilize the new elements just created (i.e set their default values)
        This is probably need to be extended by subclasses
        """
        super(SpatialReleaseSpill, self).initialize_new_elements(arrays)
        #arrays['positions'][:] = self.start_position

    def release_elements(self, current_time, time_step=None):
        """
        Release any new elements to be added to the SpillContainer
                
        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds -- this version doesn't use this

        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements

        NOTE: this releases all the elements at their initial positions at the release_time
        """
        if self.elements_not_released and current_time >= self.release_time:
            self.elements_not_released = False
            arrays = self.create_new_elements(self.num_elements)
            arrays['positions'][:,:] = self.start_positions
            return arrays
        else:
            return None

    def reset(self):
       """
       reset to initial conditions -- i.e. nothing released. 
       """
       super(SpatialReleaseSpill, self).reset()



