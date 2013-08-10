#!/usr/bin/env python

"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes provide
the logic about where an when the elements are released 

"""
# ## fixme: this needs a lot of re-factoring:

# base Spill class should be simpler

# need an element_type class to captue which data_arrays, etc are needed (or jsut a dict?)

# create_new_elements() should not be a Spill method (maybe Spill_contianer or element_type class)
# initialize_new_elements shouldn't be here either...

import os   # used to get path to database
import copy
import math
from datetime import timedelta

import numpy as np

## fixme: following used by get_oil_props only - may get moved
import sqlalchemy
#from sqlalchemy.orm import sessionmaker, scoped_session
from oillibrary.models import Oil, DBSession
from hazpy import unit_conversion
from itertools import chain
##

from gnome import basic_types, element_types
from gnome import GnomeId
from gnome.utilities import serializable


class Spill(object):
    """
    base class for a source of elements
    
    .. note:: This class is not serializable since it will not be used in PyGnome. It does not release any elements
    """

    _update = ['num_elements','on']
    _create = []
    _create.extend(_update)
    state = copy.deepcopy(serializable.Serializable.state)
    state.add( create=_create, update=_update)

    valid_vol_units = list(chain.from_iterable([item[1] for item in unit_conversion.ConvertDataUnits['Volume'].values()]))
    valid_vol_units.extend(unit_conversion.GetUnitNames('Volume'))

    @property
    def id(self):
        return self._gnome_id.id
    
    def __init__(self, num_elements=0, on=True, id=None, volume=0, volume_units='m^3', oil='oil_conservative'):
        """
        Base spill class. Spill used by a gnome model derive from this base class
        
        :param num_elements: number of LEs - default is 0.
        :type num_elements: int
        
        Optional parameters (kwargs):
        
        :param on: Toggles the spill on/off (bool). Default is 'on'.
        :type on: bool
        :param id: Unique Id identifying the newly created mover (a UUID as a string), used when loading from a persisted model
        :type id: str
        :param volume: volume of oil spilled (used to compute mass per particle)
        :type volume: float
        :param volume_units=m^3: volume units
        :type volume_units: str
        :param oil='oil_conservative': Type of oil spilled. 
                         If this is a string, or an oillibrary.models.Oil object, then create gnome.spill.OilProps(oil) object
                         If this is a gnome.spill.OilProps object, then simply instance oil_props variable to it: self.oil_props = oil
        :type oil: either str, or oillibrary.models.Oil object or gnome.spill.OilProps
        """
        self.num_elements = num_elements
        self.on = on       # sets whether the spill is active or not
        self._gnome_id = GnomeId(id)
        
        self.array_types = dict(element_types.all_spills)
        if isinstance( oil, OilProps):
            self.oil_props = oil    # OilProps object is already defined and passed in
        else:
            self.oil_props = OilProps(oil)  # construct OilProps object from str or from oillibrary.models.Oil object
        
        self._check_units(volume_units)
        self._volume_units = volume_units
        self._volume = unit_conversion.convert('Volume', volume_units, 'm^3', volume) 

    def __deepcopy__(self, memo=None):
        """
        the deepcopy implementation

        we need this, as we don't want the spill_nums copied, but do want everything else.

        got the method from:

        http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy() ends up using recursion
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.deepcopy(self.__dict__, memo)  # recursively calls deepcopy on GnomeId object
        return obj_copy

    def __copy__(self):
        """
        Make a shallow copy of the object
        
        It makes a shallow copy of all attributes defined in __dict__
        Since it is a shallow copy of the dict, the _gnome_id object is not copied, but merely referenced
        This seems to be standard python copy behavior so leave as is. 
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.copy(self.__dict__)
        return obj_copy

    def _check_units(self, units):
        """
        Checks the user provided units are in list Wind.valid_vel_units 
        """
        if units not in self.valid_vol_units:
            raise unit_conversion.InvalidUnitError('Volume units must be from following list to be valid: {0}'.format(self.valid_vol_units))
        
    @property
    def volume_units(self):
        """
        default units in which volume data is returned
        """
        return self._volume_units
    
    @volume_units.setter
    def volume_units(self, units):
        """
        set default units in which volume data is returned
        """
        self._check_units(units)    # check validity before setting
        self._volume_units = units
    
    # todo: Does setting a volume midway through the run require a rewind??
    #===========================================================================
    # def set_volume(self, volume, units):
    #     """
    #     set volume of spill released. Units are required! 
    #     """
    #     self._check_units(units)
    #     self._volume = unit_conversion.convert('Volume', units, 'm^3', volume)
    #===========================================================================
        
    def get_volume(self, units=None):
        """
        return the volume released during the spill. The default units for volume are
        as defined in 'volume_units' property. User can also specify desired output units in the
        function.
        """
        if units is None:
            return unit_conversion.convert('Volume', 'm^3', self.volume_units, self._volume)
        else:
            self._check_units(units)
            return unit_conversion.convert('Volume', 'm^3', units, self._volume)

    def uncertain_copy(self):
        """
        Returns a deepcopy of this spill for the uncertainty runs

        The copy has everything the same, including the spill_num,
        but it is a new object with a new id.

        Not much to this method, but it could be overridden to do something
        fancier in the future or a subclass.
        """
        u_copy = copy.deepcopy(self)
        return u_copy

    def rewind(self):
        """
        rewinds the Spill to original status (before anything has been released)
        Nothing needs to be done for the base class, but this method will be
        overloaded by subclasses and defined to fit their implementation
        """
        pass

    def release_elements(self, current_time, time_step):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: current time
        :type current_time: datetime.datetime 

        :param time_step: the time step, sometimes used to decide how many should get released.
        :type time_step: integer seconds

        :returns : None if there are no new elements released. A dict of arrays if there are new elements

        """
        return None

    def create_new_elements(self, num_elements):
        arrays = {}

        for name, array_type in self.array_types.iteritems():
            arrays[name] = np.zeros( (num_elements,)+array_type.shape, dtype=array_type.dtype)
            if name == 'mass' and num_elements > 0:
                # want mass in units of grams
                _total_mass = self.oil_props.get_density('kg/m^3') * self.get_volume('m^3') * 1000
                #_total_mass = unit_conversion.convert('mass', 'kg', 'g', _total_mass)
                arrays[name][:] = _total_mass/num_elements
            else:
                arrays[name][:] = array_type.initial_value
        return arrays
    

class FloatingSpill(Spill):
    """
    Spill for floating objects

    all this does is add the 'windage' parameters
    
    NOTE: This class is not serializable since it can't be used in PyGnome
    """

    _update= ['windage_range','windage_persist']
    _create= []
    _create.extend(_update) 
    state  = copy.deepcopy(Spill.state)
    state.add(update=_update, create=_create)
    
    def __init__(self,
                 windage_range=(0.01, 0.04),
                 windage_persist=900, **kwargs):
        """
        Object constructor. 
        
        Note on windage_range:
            The windage is computed by randomly sampling between this range and 
            normalizing it by windage_persist so windage is independent of model time_step.
            
        windage_persist:
            The 0 or -1 means the persistence is infinite so it is only set at the beginning of the run.
            
        Optional arguments:
        :param windage_range: A tuple defining the min/max % of wind acting on each LE. Default (0.01, 0.04)
        :type windage_range: a tuple of size 2 (min, max)
        :param windage_persist: Duration over which windage persists - this is given in seconds. Default is 900s.
        :type windage_persist: integer
        
        .. note:: Remaining kwargs are passed onto Spill's __init__ using super.  See base class documentation for remaining valid kwargs.
        """
        super(FloatingSpill, self).__init__(**kwargs)
        self.windage_range = windage_range
        self.windage_persist = windage_persist

         
class RiseVelocitySpill(Spill):
    """
    Parameters used to compute 
    """
    _update= ['distribution','range','use_dropletsize']
    _create= []
    _create.extend(_update) 
    state  = copy.deepcopy(Spill.state)
    state.add(update=_update, create=_create)
    
    def __init__(self, distribution='uniform', range=[0, 1],
                 use_dropletsize = False,
                 **kwargs):
        """
        todo: Default values??
        
        Similar to the FloatingSpill, this class is used as a mixin' and it simply
        houses 
        
        Can either set the rise velocity parameters to be sampled from a distribution or
        the droplet size parameters can be sampled from the distribution.
        
        if use_dropletsize is False, then use distribution to define rise_vel, otherwise,
        sample the dropletsize from the distribution.
        
        :param risevel_dist: could be 'uniform' or 'normal'
        :type distribution: str
        :param range: for 'uniform' dist, it is [min_val, max_val]. 
                      For 'normal' dist, it is [mean, sigma] where sigma is 1 standard deviation 
        """
        super(RiseVelocitySpill, self).__init__(**kwargs)
        
        self.distribution = distribution
        self.range = range
        self._use_dropletsize = use_dropletsize
        
        if use_dropletsize:
            self.array_types.update( element_types.droplet_size )
        else:
            self.array_types.update( element_types.rise_vel )


class PointSourceSpill(Spill):
    """
    The primary spill source class  --  a point release of floating
    non-weathering particles, can be instantaneous or continuous, and be
    released at a single point, or over a line.
    
    This serves as a base class for PointSourceSurfaceRelease and PointSourceSurfaceRelease
    """
    _update= ['start_position','release_time','end_position','end_release_time']
    _create= ['num_released', 'not_called_yet', 'prev_release_pos','delta_pos'] # not sure these should be user update able
    _create.extend(_update)
    state  = copy.deepcopy(Spill.state)
    state.add(update=_update, create=_create)
    
    @classmethod
    def new_from_dict(cls, dict_):
        """ 
        create object using the same settings as persisted object.
        In addition, set the state of other properties after initialization
        """
        new_obj = cls(num_elements=dict_.pop('num_elements'),
                      start_position=dict_.pop('start_position'),
                      release_time=dict_.pop('release_time'),
                      end_position=dict_.pop('end_position',None),
                      end_release_time=dict_.pop('end_release_time',None),
                      windage_range=dict_.pop('windage_range'),
                      windage_persist=dict_.pop('windage_persist'),
                      id=dict_.pop('id') )
        
        for key in dict_.keys():
            setattr(new_obj, key, dict_[key])
            
        return new_obj
     
    def __init__(self,
                 num_elements,
                 start_position,
                 release_time,
                 end_position=None,
                 end_release_time=None,
                 **kwargs):
        """
        :param num_elements: total number of elements to be released
        :type num_elements: integer

        :param start_position: initial location the elements are released
        :type start_position: 3-tuple of floats (long, lat, z)

        :param release_time: time the LEs are released (datetime object)
        :type release_time: datetime.datetime

        :param end_position=None: optional -- for a moving source, the end position
        :type end_position: 3-tuple of floats (long, lat, z)

        :param end_release_time=None: optional -- for a release over time, the end release time
        :type end_release_time: datetime.datetime

        :param windage_range=(0.01, 0.04): the windage range of the elements default is (0.01, 0.04) from 1% to 4%.
        :type windage_range: tuple: (min, max)

        :param windage_persist=-1: Default is 900s, so windage is updated every 900 sec.
                                -1 means the persistence is infinite so it is only set at the beginning of the run.
        :type windage_persist: integer seconds

        Remaining kwargs are passed onto base class __init__ using super. 
        See :class:`FloatingSpill` documentation for remaining valid kwargs.
        """
        super(PointSourceSpill, self).__init__( **kwargs )
        
        self.num_elements = num_elements
        
        self.release_time = release_time
        if end_release_time is None:
            self.end_release_time = release_time    # also sets self._end_release_time
        else:
            if release_time > end_release_time:
                raise ValueError("end_release_time must be greater than release_time")
            self.end_release_time = end_release_time

        if end_position is None:
            end_position = start_position   # also sets self._end_position
        self.start_position = np.array(start_position, dtype=basic_types.world_point_type).reshape((3,))
        self.end_position = np.array(end_position, dtype=basic_types.world_point_type).reshape((3,))
        if self.num_elements == 1:
            self.delta_pos = np.array( (0.0,0.0,0.0) , dtype=basic_types.world_point_type)
        else:
            self.delta_pos = (self.end_position - self.start_position) / (self.num_elements-1)
        self.delta_release = (self.end_release_time - self.release_time).total_seconds() 
        self.start_position = np.asarray(start_position, dtype=basic_types.world_point_type).reshape((3,))
        self.end_position = np.asarray(end_position, dtype=basic_types.world_point_type).reshape((3,))
        #self.positions.initial_value = self.start_position

        #self.windage_range    = windage_range[0:2]
        #self.windage_persist  = windage_persist

        self.num_released = 0
        self.not_called_yet = True
        self.prev_release_pos = self.start_position.copy()

    """
    Following properties were added primarily for setting values correctly when json input
    form webgnome is convereted to dict which is then used by from_dict to update variables.
    In this case, if user does not set end_positions or end_release_time, they become None in the
    dict. If these are None, then they should be updated to match release_time and start_position. 
    """
    @property
    def end_position(self):
        return self._end_position
    
    @end_position.setter
    def end_position(self, val):
        if val is None:
            self._end_position = self.start_position
        else:
            self._end_position = val
            
    @property
    def end_release_time(self):
        return self._end_release_time
    
    @end_release_time.setter
    def end_release_time(self, val):
        if val is None:
            self._end_release_time = self.release_time
        else:
            self._end_release_time = val        

    def release_elements(self, current_time, time_step):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: current time
        :type current_time: datetime.datetime 

        :param time_step: the time step, sometimes used to decide how many should get released.
        :type time_step: integer seconds

        :returns : None if there are no new elements released. A dict of arrays if there are new elements
        """

        if self.num_released >= self.num_elements:
            # nothing left to release
            return None

        if current_time > self.release_time and self.not_called_yet:
            # NOTE: JS - July 16th, 2013
            # This is intentional but needs to be revisited. If model run begins 
            # after the release_time, then do not release any elements!
            #first call after release time -- don't release anything
            #self.not_called_yet = False
            return None
        # it's been called before the release_time
        self.not_called_yet = False

        if current_time+timedelta(seconds=time_step) <= self.release_time: # don't want to barely pick it up...
            # not there yet...
            print "not time to release yet"
            return None

        if self.delta_release <= 0:
            num = self.num_elements
            arrays = self.create_new_elements(num)
            self.num_released = num
            if np.array_equal(self.delta_pos, (0.0,0.0,0.0)):
                #point release
                arrays['positions'][:,:] = self.start_position
            else:
                arrays['positions'][:,0] = np.linspace(self.start_position[0],self.end_position[0] , num)
                arrays['positions'][:,1] = np.linspace(self.start_position[1],self.end_position[1] , num)
                arrays['positions'][:,2] = np.linspace(self.start_position[2],self.end_position[2] , num)
            return arrays

        n_0 = self.num_released # always want to start at previous released
        #index of end of current time step
        n_1 = int( ( (current_time - self.release_time).total_seconds() + time_step) /
                      self.delta_release * (self.num_elements-1) ) # a tiny bit to make it open on the right.

        n_1 = min(n_1, self.num_elements-1) # don't want to go over the end.
        if n_1 == self.num_released-1: # indexes from zero
            # none to release this time step
            return None

        num = n_1 - n_0 + 1
        self.num_released = n_1+1 # indexes from zero
        
        arrays = self.create_new_elements(num)

        #compute the position of the elements:
        if np.array_equal(self.delta_pos, (0.0,0.0,0.0) ):
            # point release
            arrays['positions'][:,:] = self.start_position
        else:
            n = np.arange(n_0, n_1+1).reshape((-1,1))
            if self.num_elements == 1: # special case this one
                pos = np.array( [self.start_position,] )
            else:
                pos = self.start_position + n*self.delta_pos
            arrays['positions'] = pos

        return arrays

    def rewind(self):
        """
        reset to initial conditions -- i.e. nothing released.
        """
        super(PointSourceSpill, self).rewind()

        self.num_released = 0
        self.not_called_yet = True
        self.prev_release_pos = self.start_position


class PointSourceSurfaceRelease( PointSourceSpill, FloatingSpill, serializable.Serializable):
    
    # Let's add all the fields of ancestors - for now aggregate fields here
    # fixme: need a better way to do this - either put this function inside serializable module or better yet
    #        think about overriding __new__ in Serializable class to automatically traverse the hierarchy, aggretate the
    #        fields in the state object and return the object
    state  = copy.deepcopy( PointSourceSpill.state )
    [state.add_field(field) for field in FloatingSpill.state.fields if field not in state.fields]
    
    def __init__(self,
                 num_elements,
                 start_position,
                 release_time,
                 end_position=None,
                 #==============================================================
                 # end_release_time=None,
                 # windage_range=(0.01, 0.04),
                 # windage_persist=900,
                 #==============================================================
                 **kwargs):
        
        
        if len(start_position) == 2:
            start_position = [start_position[0], start_position[1], 0]
        elif len(start_position) == 3:
            if start_position[2] != 0:
                raise TypeError( "The 'z' coordinate for start_position for this type of release must be 0" )
        
        start_position = np.array( (start_position[0], start_position[1], 0), dtype=basic_types.world_point_type).reshape((3,))
        
        if end_position is not None:
            if len(end_position) == 2:
                end_position = [end_position[0], end_position[1], 0]
            elif len(end_position) == 3:
                if end_position[2] != 0:
                    raise TypeError( "The 'z' coordinate for end_position for this type of release must be 0" )
        
            self.end_position = np.array( (end_position[0], end_position[1], 0), dtype=basic_types.world_point_type).reshape((3,))

        super( PointSourceSurfaceRelease, self).__init__( num_elements,
                                                          start_position,
                                                          release_time,
                                                          end_position,
                                                          **kwargs)
        
class PointSource3DRelease( PointSourceSpill, FloatingSpill, RiseVelocitySpill, serializable.Serializable):
    # Let's add all the fields of ancestors - for now aggregate fields here
    # fixme: need a better way to do this - either put this function inside serializable module or better yet
    #        think about overriding __new__ in Serializable class to automatically traverse the hierarchy, aggretate the
    #        fields in the state object and return the object
    state  = copy.deepcopy( PointSourceSpill.state )
    [state.add_field(field) for field in FloatingSpill.state.fields if field not in state.fields]
    [state.add_field(field) for field in RiseVelocitySpill.state.fields if field not in state.fields]
    
    def __init__( self, num_elements, start_position, release_time, **kwargs):
        # simply use this to initialize all other properties of the mixins
        super(PointSource3DRelease, self).__init__( num_elements, start_position, release_time, **kwargs )

class SubsurfaceSpill(Spill):
    """
    spill for underwater objects

    all this does is add the 'water_currents' parameter
    """

    def __init__(self, **kwargs):
        super(SubsurfaceSpill, self).__init__(**kwargs)
        # it is not clear yet (to me anyway) what we will want to add to a subsurface spill


class SubsurfaceRelease(SubsurfaceSpill):
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
                 **kwargs):
        """
        :param num_elements: total number of elements used for this spill
        :param start_position: location the LEs are released (long, lat, z) (floating point)
        :param release_time: time the LEs are released (datetime object)
        :param end_position=None: optional -- for a moving source, the end position
        :param end_release_time=None: optional -- for a release over time, the end release time
        
        **kwargs contain keywords passed up the heirarchy
        """
        super(SubsurfaceRelease, self).__init__(**kwargs)

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
        #self.positions.initial_value = self.start_position

        self.num_released = 0
        self.prev_release_pos = self.start_position

    def release_elements(self, current_time, time_step):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: datetime object for current time
        :param time_step: the time step, in seconds -- used to decide how many should get released.

        :returns : None if there are no new elements released
                   a dict of arrays if there are new elements
        """
        
        if current_time >= self.release_time:
            if self.num_released >= self.num_elements:
                return None

            #total release time
            release_delta = (self.end_release_time - self.release_time).total_seconds()
            if release_delta == 0: #instantaneous release
                num = self.num_elements - self.num_released #num_released should always be 0?
            else:
                # time since release began
                if current_time >= self.end_release_time:
                    dt = release_delta
                else:
                    dt = max( (current_time - self.release_time).total_seconds() + time_step, 0.0)
                    total_num = (dt / release_delta) * self.num_elements
                    num = int(total_num - self.num_released)

            if num <= 0: # all released
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

    def rewind(self):
        """
        rewind to initial conditions -- i.e. nothing released. 
        """
        super(SubsurfaceRelease, self).rewind()

        self.num_released = 0
        self.prev_release_pos = self.start_position


class SpatialRelease(FloatingSpill):
    """
    A simple spill class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified

    """
    def __init__(self,
                 start_positions,
                 release_time,
                 windage_range=(0.01, 0.04),
                 windage_persist=900,
                 **kwargs):
        """
        :param start_positions: locations the LEs are released
        :type start_positions: (num_elements, 3) numpy array of float64 -- (long, lat, z)

        :param release_time: time the LEs are released
        :type release_time: datetime.datetime

        :param windage=(0.01, 0.04): the windage range of the LEs  Default is from 1% to 4%.
        :param windage: tuple of floats: (min, max). 

        :param persist=900: Default is 900s, so windage is updated every 900 sec. The -1 means the persistence is infinite so it is only set at the beginning of the run.
        :type persist: integer secondsDefault 
                        The -1 means the persistence is infinite so it is only set at the beginning of the run.
        """
        super(SpatialRelease, self).__init__(windage_range, windage_persist, **kwargs)
        
        self.start_positions = np.asarray(start_positions,
                                          dtype=basic_types.world_point_type).reshape((-1, 3))
        self.num_elements = self.start_positions.shape[0]

        self.release_time = release_time

        self.elements_not_released = True
        self.not_called_yet = True
        self.windage_range    = windage_range[0:2]
        self.windage_persist  = windage_persist

    def release_elements(self, current_time, time_step):
        """
        Release any new elements to be added to the SpillContainer

        :param current_time: current time
        :type current_time: datetime.datetime 

        :param time_step: the time step, sometimes used to decide how many should get released.
        :type time_step: integer seconds

        :returns : None if there are no new elements released. A dict of arrays if there are new elements

        .. note:: this releases all the elements at their initial positions at the release_time
        """
        
        if current_time > self.release_time and self.not_called_yet:
            #first call after release time -- don't release anything
            return None
        # it's been called before the release_time
        self.not_called_yet = False

        if self.elements_not_released and current_time >= self.release_time:
            self.elements_not_released = False
            arrays = self.create_new_elements(self.num_elements)
            arrays['positions'][:,:] = self.start_positions
            return arrays
        else:
            return None

    def rewind(self):
        """
        rewind to initial conditions -- i.e. nothing released. 
        """
        self.elements_not_released = True
        self.not_called_yet = True


class OilProps():
    """
    Class gets the oil properties in user specified units
    """
    # Some standard oils
    _sample_oils = {'oil_gas': 
                    {'Oil Name': 'oil_gas','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 0.75)},
                    
                    'oil_jetfuels': 
                    {'Oil Name': 'oil_jetfuels','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 0.81)},
                    
                    'oil_diesel': 
                    {'Oil Name':'oil_diesel','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 0.87)},
                    
                    'oil_4': 
                    {'Oil Name': 'oil_4','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 0.90)},
                    
                    'oil_crude': 
                    {'Oil Name':'oil_crude','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 0.90)},
                    
                    'oil_6': 
                    {'Oil Name':'oil_6','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 0.99)},
                    
                    'oil_conservative': 
                    {'Oil Name':'oil_conservative','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 1)},
                    
                    'chemical': 
                    {'Oil Name':'chemical','API':unit_conversion.convert('Density', 'gram per cubic centimeter', 'API degree', 1)},
                   }
    
    
    valid_density_units = list(chain.from_iterable([item[1] for item in unit_conversion.ConvertDataUnits['Density'].values()]))
    valid_density_units.extend(unit_conversion.GetUnitNames('Density'))
    
    def __init__(self, oil_):
        """
        Should user be able to provide an oil with density/properties?
        
        If oil_ is amongst self._sample_oils dict, then use the properties defined here. 
        If not, then query the Oil database to check if oil_ exists and get the properties from DB
        Otherwise define Oil object with 'Oil Name' set to 'unknown' and its 'API' set to '1' 
        
        :param oil_: name of the oil that spilled. If it is one of the names that exist in the oil database, the
                         associated properties stored in the DB are returned; otherwise, a default set of properties
                         are returned(?)
        :type oil_: str
        """
        
        if isinstance( oil_, basestring):
            if oil_ in self._sample_oils:
                self.oil = Oil( **self._sample_oils[oil_] )
            else:
                db_file = os.path.join( os.path.split( os.path.realpath(__file__))[0], '../../web/adios/OilLibrary/OilLibrary.db')
                if os.path.exists(db_file):
                    engine = sqlalchemy.create_engine('sqlite:///'+db_file)  # path relative to spill.py
                    DBSession.bind = engine # not sure we want to do it this way - but let's use for now
                    #let's use DBSession defined in oillibrary
                    #session_factory = sessionmaker(bind=engine)
                    #DBSession = scoped_session(session_factory)
                    
                    try:
                        self.oil = DBSession.query(Oil).filter(Oil.name==oil_).one()
                    except sqlalchemy.orm.exc.NoResultFound as ex: #or sqlalchemy.orm.exc.MultipleResultsFound as ex:
                        ex.message = "oil with name '{0}' not found in database. {1}".format(oil_, ex.message)
                        ex.args = (ex.message,)
                        raise ex
                    #    props={'Oil Name': 'unknown',
                    #           'API': 1.0}
                    #    self.oil = Oil( **props)
                else:
                    raise IOError('OilLibrary database not found at: '.format( db_file) )
            
        elif isinstance( oil_, Oil):
            self.oil = oil_
        else:
            raise TypeError( "Initialization requires either a string containing the oil name or a valid oillibrary.models.Oil object")
    
    name = property( lambda self: self.oil.name )
    
    def get_density(self, units='kg/m^3'):
        """
        :param units=kg/m^3: optional input if output units should be something other than kg/m^3
        """
        if self.oil.api is None:
            raise ValueError( "Oil with name '{0}' does not contain 'api' property.".format(self.oil.name)) 
        
        if units not in self.valid_density_units:
            raise unit_conversion.InvalidUnitError('Desired density units must be from following list to be valid: {0}'.format(self.valid_density_units))
        
        # since Oil object can have various densities depending on temperature, lets return API in correct units
        return unit_conversion.convert('Density', 'API degree', units, self.oil.api)
