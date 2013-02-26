#!/usr/bin/env python

"""
spill.py - A new implementation of the spill class(s)

We now keep all the data in separate arrays, so we only store and move around the
data that is needed

This is the "magic" class -- it handles the smart allocation of arrays, etc.

these are managed by the SpillContainer class
"""
import numpy as np
from gnome import basic_types
from gnome.gnomeobject import GnomeObject

class ArrayType(object):
    def __init__(self, shape, dtype, initial_value=None):
        self.shape = shape
        self.dtype = dtype
        self.initial_value = initial_value


class Spill(GnomeObject):
    """
    base class for a source of elements
    """
    positions = ArrayType( (3,), basic_types.world_point_type)
    next_positions = ArrayType( (3,), basic_types.world_point_type)
    last_water_positions = ArrayType( (3,), basic_types.world_point_type)
    status_codes = ArrayType( (), basic_types.status_code_type,
                              basic_types.oil_status.in_water)
    spill_num = ArrayType( (), basic_types.id_type)

    @property
    def array_types(self):
        return dict([(name, getattr(self, name))
                for name in dir(self)
                if name != 'array_types'
                and type(getattr(self, name)) == ArrayType])

    def __new__(cls, *args, **kwargs):
        obj = super(Spill, cls).__new__(cls, *args, **kwargs)
        return obj

    def __init__(self, num_elements=0):
        self.num_elements = num_elements
        self.on = True       # sets whether the spill is active or not

    def __deepcopy__(self, memo=None):
        """
        the deepcopy implementation

        we need this, as we don't want the spill_nums copied, but do want everything else.

        got the method from:

        http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy() ends up using recursion
        """
        obj_copy = super(Spill, self).__deepcopy__(memo)
        return obj_copy

    def __copy__(self):
        """
        might as well have copy, too.
        """
        obj_copy = super(Spill, self).__copy__()
        return obj_copy

    def uncertain_copy(self):
        """
        Returns a copy of this spill for the uncertainty runs

        The copy has eveything the same, including the spill_num,
        but should have a new id.

        Not much to this method, but it could be overridden to do something
        fancier in the future or a subclass.
        """
        u_copy = super(Spill, self).__copy__()
        return u_copy

    def rewind(self):
        """
        rewinds the Spill to original status (before anything has been released)
        Nothing needs to be done for the base class, but this method will be
        overloaded by subclasses and defined to fit their implementation
        """
        pass

    def release_elements(self, current_time, time_step, array_types=None):
        """
        probably overridden by a subclass
        """
        return None

    def create_new_elements(self, num_elements, array_types=None):
        arrays = {}
        if not array_types:
            array_types = self.array_types

        for name, array_type in array_types.iteritems():
            arrays[name] = np.zeros( (num_elements,)+array_type.shape, dtype=array_type.dtype)
        self.initialize_new_elements(arrays, array_types)
        return arrays

    def initialize_new_elements(self, arrays, array_types=None):
        if not array_types:
            array_types = self.array_types

        for name, array_type in array_types.iteritems():
            if array_type.initial_value != None:
                arrays[name][:] = array_type.initial_value

class FloatingSpill(Spill):
    """
    spill for floating objects

    all this does is add the 'windage' parameter
    """
    windages = ArrayType( (), basic_types.windage_type)
    
    def __init__(self,
                 windage_range=(0.01, 0.04),
                 windage_persist=900):

        super(FloatingSpill, self).__init__()
        self.windage_range = windage_range
        self.windage_persist = windage_persist

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
        self.positions.initial_value = self.start_position

        self.windage_range    = windage_range[0:2]
        self.windage_persist  = windage_persist

        self.num_released = 0
        self.prev_release_pos = self.start_position

    def release_elements(self, current_time, time_step, array_types=None):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds --
                          sometimes used to decide how many should get released.

        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements
        """
        if not array_types:
            array_types = self.array_types

        if current_time >= self.release_time:
            if self.num_released >= self.num_elements:
                return None

            #total release time
            release_delta = (self.end_release_time - self.release_time).total_seconds()

            # time since release began
            if current_time >= self.end_release_time:
                dt = release_delta
            else:
                dt = max( (current_time - self.release_time).total_seconds() + time_step, 0.0)
            # this here in case there is less than one released per time step.
            # or if the release time is before the model start time
            if release_delta == 0: #instantaneous release
                num = self.num_elements - self.num_released #num_released should always be 0?
            else:
                total_num = (dt / release_delta) * self.num_elements
                num = int(total_num - self.num_released)

            if num <= 0:
                return None

            self.num_released += num

            arrays = self.create_new_elements(num, array_types)

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

    def rewind(self):
        """
        rewind to initial conditions -- i.e. nothing released.
        """
        super(SurfaceReleaseSpill, self).rewind()

        self.num_released = 0
        self.prev_release_pos = self.start_position


class SubsurfaceSpill(Spill):
    """
    spill for underwater objects

    all this does is add the 'water_currents' parameter
    """
    water_currents = ArrayType( (3,), basic_types.water_current_type)

    def __init__(self):
        super(SubsurfaceSpill, self).__init__()
        # it is not clear yet (to me anyway) what we will want to add to a subsurface spill


class SubsurfaceReleaseSpill(SubsurfaceSpill):
    """
    The second simplest spill source class  --  a point release of underwater
    non-weathering particles

    .. todo::
        'gnome.cy_gnome.cy_basic_types.oil_status' does not currently have an underwater status.
        for now we will just keep the in_water status, but we will probably want to change this
        in the future.
    """
    def __init__(self,
                 num_elements,
                 start_position,
                 release_time,
                 end_position=None,
                 end_release_time=None,
                 ):
        """
        :param num_elements: total number of elements used for this spill
        :param start_position: location the LEs are released (long, lat, z) (floating point)
        :param release_time: time the LEs are released (datetime object)
        :param end_position=None: optional -- for a moving source, the end position
        :param end_release_time=None: optional -- for a release over time, the end release time
        """
        super(SubsurfaceReleaseSpill, self).__init__()

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
        self.positions.initial_value = self.start_position

        self.num_released = 0
        self.prev_release_pos = self.start_position

    def release_elements(self, current_time, time_step, array_types=None):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds -- used to decide how many should get released.

        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements
        """
        if not array_types:
            array_types = self.array_types

        if current_time >= self.release_time:
            if self.num_released >= self.num_elements:
                return None

            #total release time
            release_delta = (self.end_release_time - self.release_time).total_seconds()

            # time since release began
            if current_time >= self.end_release_time:
                dt = release_delta
            else:
                dt = max( (current_time - self.release_time).total_seconds() + time_step, 0.0)
            # this here in case there is less than one released per time step.
            # or if the release time is before the model start time
            if release_delta == 0: #instantaneous release
                num = self.num_elements - self.num_released #num_released should always be 0?
            else:
                total_num = (dt / release_delta) * self.num_elements
                num = int(total_num - self.num_released)

            if num <= 0:
                return None

            self.num_released += num

            arrays = self.create_new_elements(num, array_types)

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

    def rewind(self):
        """
        rewind to initial conditions -- i.e. nothing released. 
        """
        super(SubsurfaceReleaseSpill, self).rewind()

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

    def release_elements(self, current_time, time_step, array_types=None):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: datetime object for current time
        :param time_step=None: the time step, in seconds --
                          this version doesn't use this, but it's part of the API
        :param array_types=None:
        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements

        NOTE: this releases all the elements at their initial positions at the release_time
        """
        if not array_types:
            array_types = self.array_types

        if self.elements_not_released and current_time >= self.release_time:
            self.elements_not_released = False
            arrays = self.create_new_elements(self.num_elements, array_types)
            arrays['positions'][:,:] = self.start_positions
            return arrays
        else:
            return None
