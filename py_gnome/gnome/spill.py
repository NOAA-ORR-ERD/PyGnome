"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes provide
the logic about where an when the elements are released

"""

import copy
import inspect
from datetime import timedelta
from itertools import chain

import numpy
np = numpy

from hazpy import unit_conversion
uc = unit_conversion

import gnome    # required by new_from_dict
from gnome.persist import modules_dict
from gnome import elements, GnomeId
from gnome.basic_types import world_point_type
from gnome.utilities import serializable
from gnome.utilities.plume import Plume, PlumeGenerator
from gnome.persist import spills_schema

from gnome.db.oil_library.oil_props import OilProps


class Release(object):
    """
    base class for Release classes.

    It contains interface for Release objects
    """
    _update = ['num_elements']

    # obj_type is no longer part of default so add it explicitly here
    _create = ['num_released', 'start_time_invalid']
    _create.extend(_update)
    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(create=_create, update=_update)

    def __init__(self, num_elements=0, release_time=None):
        self.num_elements = num_elements
        self.release_time = release_time

        # number of new particles released at each timestep
        self.num_released = 0

        # flag determines if the first time is valid. If the first call to
        # self.num_elements_to_release(current_time, time_step) has
        # current_time > self.release_time, then no particles are ever released
        # if current_time <= self.release_time, then toggle this flag since
        # model start time is valid
        self.start_time_invalid = True

    def num_elements_to_release(self, current_time, time_step):
        """
        Determines the number of elements to be released during:
        current_time + time_step. Base class has partial (incomplete)
        implementation.

        This base class method checks if current_time in first step
        is valid and toggles the self.start_time_invalid flag if it is valid.
        If current_time <= self.release_time the first time this is called,
        then toggle start_time_invalid to True.

        Subclasses should define the complete implementation and return number
        of new particles to be released once this check passes. Be sure to call
        the base class method first if start_time_invalid flag should be
        checked.

        :param current_time: current time
        :type current_time: datetime.datetime
        :param time_step: the time step, sometimes used to decide how many
            should get released.
        :type time_step: integer seconds

        :returns: the number of elements that will be released. This is taken
            by SpillContainer to initialize all data_arrays.

        self.num_released is updated after self.set_newparticle_values is
        called. Particles are considered released after the values are set.
        """
        if self.release_time is None:
            raise ValueError("release_time attribute cannot be None")

        if (current_time <= self.release_time and
            self.start_time_invalid):
            # start time is valid
            # It's fine to release elements in subsequent steps
            self.start_time_invalid = False

        return 0    # base class does not release any particles

    def set_new_particle_positions(self,
        num_new_particles,
        current_time,
        time_step,
        data_arrays):
        """
        derived object should set the 'positions' array for the data_arrays
        base class has no implementation
        """
        pass

    def rewind(self):
        """
        rewinds the Release to original status (before anything has been
        released).

        Base class sets 'num_released'=0 and 'start_time_invalid'=True
        properties to original _state.
        Subclasses should overload for additional functions required to reset
        _state.
        """
        self.num_released = 0
        self.start_time_invalid = True


class PointLineRelease(Release, serializable.Serializable):

    """
    The primary spill source class  --  a release of floating
    non-weathering particles, can be instantaneous or continuous, and be
    released at a single point, or over a line.
    """

    _update = ['start_position',
               'release_time',
               'end_position',
               'end_release_time']

    # not sure these should be user update able
    _create = ['prev_release_pos']
    _create.extend(_update)
    _state = copy.deepcopy(Release._state)
    _state.add(update=_update, create=_create)

    @classmethod
    def new_from_dict(cls, dict_):
        """
        create object using the same settings as persisted object.
        In addition, set the _state of other properties after initialization
        """

        new_obj = cls(
            num_elements=dict_.pop('num_elements'),
            start_position=dict_.pop('start_position'),
            release_time=dict_.pop('release_time'),
            end_position=dict_.pop('end_position', None),
            end_release_time=dict_.pop('end_release_time', None),
            )

        for key in dict_.keys():
            setattr(new_obj, key, dict_[key])

        return new_obj

    def __init__(
        self,
        num_elements,
        start_position,
        release_time,
        end_position=None,
        end_release_time=None,
        ):
        """
        :param num_elements: total number of elements to be released
        :type num_elements: integer

        :param release_time: time the LEs are released (datetime object)
        :type release_time: datetime.datetime

        :param start_position: initial location the elements are released
        :type start_position: 3-tuple of floats (long, lat, z)

        :param end_position=None: optional. For moving source, the end position
        :type end_position: 3-tuple of floats (long, lat, z)

        :param end_release_time=None: optional -- for a release over time, the
            end release time
        :type end_release_time: datetime.datetime

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """

        super(PointLineRelease, self).__init__(num_elements,
                                               release_time)

        if end_release_time is None:
            # also sets self._end_release_time
            self.end_release_time = release_time
        else:
            if release_time > end_release_time:
                raise ValueError('end_release_time must be greater than'
                    ' release_time')
            self.end_release_time = end_release_time

        if self.release_time == self.end_release_time:
            self.set_newparticle_positions = \
                self._init_positions_instantaneous_release
        else:
            self.set_newparticle_positions = \
                self._init_positions_timevarying_release

        self.start_position = np.array(start_position,
                            dtype=world_point_type).reshape((3, ))
        if end_position is None:
            # also sets self._end_position
            end_position = start_position

        self.end_position = np.array(end_position,
                dtype=world_point_type).reshape((3, ))

        # only needs to be computed once
        self.delta_pos = ((self.end_position - self.start_position) /
                          max(1, self.num_elements - 1))

        self.delta_release = (self.end_release_time
                              - self.release_time).total_seconds()

        # number of new particles released at each timestep
        self.prev_release_pos = self.start_position.copy()

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

    def num_elements_to_release(self, current_time, time_step):
        """
        return number of particles released in current_time + time_step
        """
        # call base class method to check if start_time is valid
        super(PointLineRelease, self).num_elements_to_release(current_time,
                                                             time_step)
        if self.start_time_invalid:
            return 0

        if self.num_released >= self.num_elements:
            # nothing left to release
            return 0

        # it's been called before the release_time
        if current_time + timedelta(seconds=time_step) \
            <= self.release_time:  # don't want to barely pick it up...
            # not there yet...
            #print 'not time to release yet'
            return 0

        delta_release = (self.end_release_time
                              - self.release_time).total_seconds()
        # instantaneous release. All particles released at this timestep
        if delta_release <= 0:
            return self.num_elements

        # time varying release
        n_0 = self.num_released  # always want to start at previous released

        # index of end of current time step
        # a tiny bit to make it open on the right.
        n_1 = int(((current_time - self.release_time).total_seconds()
                  + time_step) / delta_release
                  * (self.num_elements - 1))

        n_1 = min(n_1, self.num_elements - 1)  # don't want to go over the end.
        if n_1 == self.num_released - 1:  # indexes from zero
            # none to release this time step
            return 0

        # JS: not sure why we want to release 1 extra particle?
        # but leave algorithm as it is. Since n_0 = 0 at first iteration,
        # _num_new_particles at 1st step is 1 more than _num_new_particles in
        # subsequent steps for a fixed time_step
        _num_new_particles = n_1 - n_0 + 1
        return _num_new_particles

    def _init_positions_instantaneous_release(
        self,
        num_new_particles,
        current_time,
        time_step,
        data_arrays):
        """
        initialize all elements in the very first instant (timestep) of the run
        The particles can be released at a single 'point' or along a 'line'

        For each axis (x,y,z), it evenly spaces all elements along a line:
            np.linspace( self.start_position, self.end_position,
                         self._num_new_particles)

        If self.start_position == self.end_position, then all particles are
        released at self.start_position.
        """
        if num_new_particles == 0:
            return

        if np.all(self.start_position == self.end_position):
            # point release
            data_arrays['positions'][-num_new_particles:, :] = \
                self.start_position
        else:
            # line release
            data_arrays['positions'][-num_new_particles:, 0] = \
                np.linspace(self.start_position[0],
                            self.end_position[0],
                            num_new_particles)
            data_arrays['positions'][-num_new_particles:, 1] = \
                np.linspace(self.start_position[1],
                            self.end_position[1],
                            num_new_particles)
            data_arrays['positions'][-num_new_particles:, 2] = \
                np.linspace(self.start_position[2],
                            self.end_position[2],
                            num_new_particles)

        # expect self.num_released to be 0 before the instantaneous release
        self.num_released += num_new_particles

    def _init_positions_timevarying_release(self, num_new_particles,
                                            current_time, time_step,
                                            data_arrays):
        """
        Time varying release of particles. Initialize particles as they are
        released in each timestep
        """
        if num_new_particles == 0:
            return

        if np.all(self.start_position == self.end_position):
            # point release
            data_arrays['positions'][-num_new_particles:] = \
                self.start_position
        else:
            # continuous line release
            n_0 = self.num_released
            n_1 = self.num_released + num_new_particles
            n = np.arange(n_0, n_1).reshape((-1, 1))
            data_arrays['positions'][-num_new_particles:] = \
                self.start_position + n * self.delta_pos

        self.num_released += num_new_particles

    def rewind(self):
        """
        reset to initial conditions -- i.e. nothing released.
        """
        super(PointLineRelease, self).rewind()
        self.prev_release_pos = self.start_position


class SpatialRelease(Release, serializable.Serializable):

    """
    A simple release class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified
    """

    _state = copy.deepcopy(Release._state)
    _state.add(update=['start_position'], create=['start_position'])

    def __init__(
        self,
        start_position,
        release_time,
        ):
        """
        :param release_time: time the LEs are released
        :type release_time: datetime.datetime

        :param start_positions: locations the LEs are released
        :type start_positions: (num_elements, 3) numpy array of float64
            -- (long, lat, z)

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """

        self.start_position = np.asarray(start_position,
                dtype=world_point_type).reshape((-1, 3))
        super(SpatialRelease, self).__init__(self.start_position.shape[0],
                                             release_time)

    def num_elements_to_release(self, current_time, time_step):
        """
        return number of particles released in current_time + time_step
        """
        # call base class method to check if start_time is valid
        super(SpatialRelease, self).num_elements_to_release(current_time,
                                                             time_step)
        if self.start_time_invalid:
            return 0

        if (self.num_released >= self.num_elements or
            current_time + timedelta(seconds=time_step) <= self.release_time):
            return 0

        return self.num_elements

    def set_newparticle_positions(self, num_new_particles, current_time,
                               time_step, data_arrays):
        """
        set positions for new elements added by the SpillContainer

        .. note:: this releases all the elements at their initial positions at
            the release_time
        """
        # call the base Spill class set_newparticle_values()
        #super(SpatialRelease, self).set_newparticle_values(num_new_particles,
        #                                                    current_time,
        #                                                    time_step,
        #                                                    data_arrays)
        self.num_released = self.num_elements
        data_arrays['positions'][:, :] = self.start_position


class VerticalPlumeRelease(Release):
    '''
    An Underwater Plume spill class -- a continuous release of particles,
    controlled by a contained spill generator object.
    - plume model generator will have an iteration method.  This will provide
      flexible looping and list comprehension behavior.
    '''
    def __init__(self,
                 num_elements,
                 start_position,
                 release_time,
                 plume_data,
                 end_release_time,
                 **kwargs):
        '''
        :param num_elements: total number of elements to be released
        :type num_elements: integer

        :param start_position: initial location the elements are released
        :type start_position: 3-tuple of floats (long, lat, z)

        :param release_time: time the LEs are released
        :type release_time: datetime.datetime

        :param start_positions: locations the LEs are released
        :type start_positions: (num_elements, 3) numpy array of float64
            -- (long, lat, z)
        '''

        super(VerticalPlumeRelease, self).__init__(num_elements, release_time)

        self.start_position = np.array(start_position,
                            dtype=world_point_type).reshape((3, ))

        plume = Plume(position=start_position,
                      plume_data=plume_data)
        time_step_delta = timedelta(hours=1).total_seconds()
        self.plume_gen = PlumeGenerator(release_time=release_time,
                                        end_release_time=end_release_time,
                                        time_step_delta=time_step_delta,
                                        plume=plume)

        if self.num_elements:
            self.plume_gen.set_le_mass_from_total_le_count(self.num_elements)

    def _plume_elem_coords(self, current_time, time_step):
        '''
        Return a list of positions for all elements released within
        current_time + time_step
        '''
        next_time = current_time + timedelta(seconds=time_step)
        elem_counts = self.plume_gen.elems_in_range(current_time, next_time)

        for coord, count in zip(self.plume_gen.plume.coords, elem_counts):
            print coord, count
            for c in (coord,) * count:
                yield tuple(c)

    def num_elements_to_release(self, current_time, time_step):
        '''
        Return number of particles released in current_time + time_step
        '''
        return len([e for e in self._plume_elem_coords(current_time,
                                                       time_step)])

    def set_newparticle_positions(self, num_new_particles, current_time,
                               time_step, data_arrays):
        '''
        Set positions for new elements added by the SpillContainer
        '''
        coords = [e for e in self._plume_elem_coords(current_time, time_step)]
        self.coords = np.asarray(tuple(coords),
                                 dtype=world_point_type).reshape((-1, 3))

        if self.coords.shape[0] != num_new_particles:
            raise RuntimeError('The Specified number of new particals does not'
                               ' match the number calculated from the '
                               'time range.')

        # call the base Spill class set_newparticle_values()
        #super(VerticalPlumeRelease,
        #      self).set_newparticle_positions(num_new_particles,
        #                                   current_time, time_step,
        #                                   data_arrays)
        self.num_released += num_new_particles
        data_arrays['positions'][-self.coords.shape[0]:, :] = self.coords

    def rewind(self):
        '''
        Rewind to initial conditions -- i.e. nothing released.
        '''
        self.num_released = 0
        self.start_time_invalid = True


class Spill(serializable.Serializable):

    """
    base class for a source of elements

    .. note:: This class is not serializable since it will not be used in
              PyGnome. It does not release any elements
    """

    _update = ['on', 'release', 'element_type']
    _create = []
    _create.extend(_update)
    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(create=_create, update=_update, read=['num_released'])

    valid_vol_units = list(chain.from_iterable([item[1] for item in
                           unit_conversion.ConvertDataUnits['Volume'
                           ].values()]))
    valid_vol_units.extend(unit_conversion.GetUnitNames('Volume'))

    valid_mass_units = list(chain.from_iterable([item[1] for item in
                           uc.ConvertDataUnits['Mass'].values()]))
    valid_mass_units.extend(uc.GetUnitNames('Mass'))

    @classmethod
    def new_from_dict(cls, dict_):
        """
        create object using the same settings as persisted object.
        In addition, set the _state of other properties after initialization
        """
        # create release object
        # create element_type object
        # then create Spill object
        for name in ['release', 'element_type']:
            obj_dict = dict_.pop(name)
            obj_type = obj_dict.pop('obj_type')
            obj = eval(obj_type).new_from_dict(obj_dict)
            dict_[name] = obj

        new_obj = super(Spill, cls).new_from_dict(dict_)

        return new_obj

    def __init__(
        self,
        release,
        element_type=None,
        on=True,
        volume=None,
        volume_units='m^3',
        # Is this total mass of the spill?
        mass=None,
        mass_units='g',
        id=None,
        ):
        """
        Base spill class. Spill used by a gnome model derive from this class

        :param num_elements: number of LEs - default is 0.
        :type num_elements: int

        Optional parameters (kwargs):

        :param on: Toggles the spill on/off (bool). Default is 'on'.
        :type on: bool
        :type id: str
        :param volume: oil spilled volume (used to compute mass per particle)
            Default is None.
        :type volume: float
        :param volume_units=m^3: volume units
        :type volume_units: str
        :param windage_range=(0.01, 0.04): the windage range of the elements
            default is (0.01, 0.04) from 1% to 4%.
        :type windage_range: tuple: (min, max)
        :param windage_persist=-1: Default is 900s, so windage is updated every
            900 sec. -1 means the persistence is infinite so it is only set at
            the beginning of the run.
        :type windage_persist: integer seconds
        :param id: Unique Id identifying the newly created mover (a UUID as a
            string), used when loading from a persisted model
        :param element_type=None: list of various element_type that are
            released. These are spill specific properties of the elements.
        :type element_type: list of gnome.element_type.* objects
        """

        #self.num_elements = num_elements
        self.release = release
        if element_type is None:
            element_type = elements.floating()
        self.element_type = element_type

        self.on = on    # spill is active or not

        # mass/volume, type of oil spilled
        self._check_units(volume_units)
        self._volume_units = volume_units   # user defined for display
        self._volume = volume
        if volume is not None:
            self._volume = unit_conversion.convert('Volume', volume_units,
                'm^3', volume)

        self._check_units(mass_units, 'Mass')
        self._mass_units = mass_units   # user defined for display
        self._mass = mass
        if mass is not None:
            self._mass = uc.convert('Mass', mass_units, 'g', mass)

        if mass is not None and volume is not None:
            raise ValueError("'mass' and 'volume' cannot both be set")

#==============================================================================
#         if windage_range is not None:
#             if 'windages' not in self.element_type.initializers:
#                 raise TypeError("'windage_range' cannot be set for specified"
#                                 " element_type: {0}".format(element_type))
#             (self.element_type.initializers['windages']).windage_range = \
#                     windage_range
# 
#         if windage_persist is not None:
#             if 'windages' not in self.element_type.initializers:
#                 raise TypeError("'windage_persist' cannot be set for specified"
#                                 " element_type: {0}".format(element_type))
#             (self.element_type.initializers['windages']).windage_persist = \
#                 windage_persist
#==============================================================================

        self._gnome_id = GnomeId(id)

    def __deepcopy__(self, memo=None):
        """
        the deepcopy implementation

        we need this, as we don't want the spill_nums copied, but do want
        everything else.

        got the method from:

        http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy()
        ends up using recursion
        """

        obj_copy = object.__new__(type(self))

        # recursively calls deepcopy on GnomeId object
        obj_copy.__dict__ = copy.deepcopy(self.__dict__, memo)
        return obj_copy

    def __copy__(self):
        """
        Make a shallow copy of the object

        It makes a shallow copy of all attributes defined in __dict__
        Since it is a shallow copy of the dict, the _gnome_id object is not
        copied, but merely referenced
        This seems to be standard python copy behavior so leave as is.
        """

        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.copy(self.__dict__)
        return obj_copy

    def __eq__(self, other):
        """
        over ride base == operator defined in Serializable class.
        Spill object contains nested objects like ElementType and Release
        objects. Check all properties here so nested objects properties
        can be checked in the __eq__ implementation within the nested objects
        """
        if (self._state.get_field_by_attribute('create') !=
            other._state.get_field_by_attribute('create')):
            return False

        for name in self._state.get_names('create'):
            if not hasattr(self, name):
                """
                for an attribute like obj_type, base class has
                obj_type_to_dict method so let base class convert the attribute
                to dict, then compare
                """
                if (self.attr_to_dict(name) != other.attr_to_dict(name)):
                    return False

            elif getattr(self, name) != getattr(other, name):
                return False

        return True

    def _check_units(self, units, unit_type='Volume'):
        """
        Checks the user provided units are in list of valid volume
        or mass units
        """

        if unit_type == 'Volume':
            if units not in self.valid_vol_units:
                raise uc.InvalidUnitError('Volume units must be from '
                                          'following list to be valid: '
                                          '{0}'.format(self.valid_vol_units))
        elif unit_type == 'Mass':
            if units not in self.valid_mass_units:
                raise uc.InvalidUnitError('Mass units must be from '
                                          'following list to be valid: '
                                          '{0}'.format(self.valid_mass_units))

    @property
    def id(self):
        return self._gnome_id.id

    def set(self, prop, val):
        """
        sets an existing property. The property could be of one of the
        contained objects like 'Release' or 'ElementType'
        It can also be a property of one of the initializers contained in
        the 'ElementType' object.

        If the property doesn't exist for any of these, then an error is raised
        since user cannot set a property that does not exist using this method
        """
        if prop == 'num_released':
            raise AttributeError("cannot set attribute")

        # we don't want to add an attribute that doesn't already exist
        # first check to see that the attribute exists, then change it else
        # raise error
        if hasattr(self.release, prop):
            setattr(self.release, prop, val)
        elif hasattr(self.element_type, prop):
            setattr(self.element_type, prop, val)
        else:
            for init in self.element_type.initializers.values():
                if hasattr(init, prop):
                    setattr(init, prop, val)
                    break
                else:
                    raise AttributeError("{0} attribute does not exist"
                        " in element_type or release object".format(prop))

    def get(self, prop=None):
        """
        if prop is None then return all the user defined properties of
        'release' object and list of initializers in 'element_type' object

        for get(), return all properties of embedded release object and
        element_type initializer objects
        """
        'Return all properties'
        if prop is None:
            all_props = []

            # release properties
            rel_props = inspect.getmembers(self.release,
                            predicate=lambda props: \
                                (False, True)[not inspect.ismethod(props)])
            'remove _state - update this after we change _state to _state'
            rel_props = [a[0] for a in rel_props
                            if not a[0].startswith('_') and a[0] != '_state']

            all_props.extend(rel_props)

            # element_type properties
            et_props = inspect.getmembers(self.element_type,
                            predicate=lambda props: \
                                (False, True)[not inspect.ismethod(props)])
            'remove _state - update this after we change _state to _state'
            et_props = [a[0] for a in et_props
                            if not a[0].startswith('_') and a[0] != '_state']

            all_props.extend(et_props)

            # properties for each of the initializer objects
            i_props = []
            for val in self.element_type.initializers.values():
                toadd = inspect.getmembers(val, lambda props: \
                                    (False, True)[not inspect.ismethod(props)])
                i_props.extend([a[0] for a in toadd
                            if not a[0].startswith('_') and a[0] != '_state'])

            all_props.extend(i_props)
            return all_props

        if hasattr(self.release, prop):
            return getattr(self.release, prop)

        if hasattr(self.element_type, prop):
            return getattr(self.element_type, prop)

        for init in self.element_type.initializers.values():
            if hasattr(init, prop):
                return getattr(init, prop)

        # nothing returned, then property was not found - raise exception or
        # return None?
        raise AttributeError("{0} attribute does not exist in element_type"
            " or release object".format(prop))

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
        self._check_units(units)  # check validity before setting
        self._volume_units = units

    # returns the volume in volume_units specified by user
    volume = property(lambda self: self.get_volume(),
                      lambda self, value: self.set_volume(value, 'm^3'))

    @property
    def mass_units(self):
        return self._mass_units

    @mass_units.setter
    def mass_units(self, units):
        self._check_units(units, 'Mass')  # check validity before setting
        self._mass_units = units

    # returns the mass in mass_units specified by user
    mass = property(lambda self: self.get_mass(),
                      lambda self, value: self.set_mass(value, 'g'))

    def get_volume(self, units=None):
        """
        return the volume released during the spill. The default units for
        volume are as defined in 'volume_units' property. User can also specify
        desired output units in the function.
        """
        if self._volume is None:
            return self._volume

        if units is None:
            return unit_conversion.convert('Volume', 'm^3',
                                           self.volume_units, self._volume)
        else:
            self._check_units(units)
            return unit_conversion.convert('Volume', 'm^3', units,
                    self._volume)

    def set_volume(self, volume, units):
        """
        set the volume released during the spill. The default units for
        volume are as defined in 'volume_units' property. User can also specify
        desired output units in the function.
        """
        self._check_units(units)
        self._volume = unit_conversion.convert('Volume', units, 'm^3', volume)
        self.volume_units = units

    def get_mass(self, units=None):
        '''
        Return the mass released during the spill.
        The default units for mass are as defined in 'mass_units' property.
        User can also specify desired output units in the function.
        '''
        if self._mass is None:
            return self._mass

        if units is None:
            return uc.convert('Mass', 'g', self.mass_units, self._mass)
        else:
            self._check_units(units, 'Mass')
            return uc.convert('Mass', 'g', units, self._mass)

    def set_mass(self, mass, units):
        '''
        Set the mass released during the spill.
        The default units for mass are as defined in 'mass_units' property.
        User can also specify desired output units in the function.
        '''
        self._check_units(units, 'Mass')
        self._mass = uc.convert('Mass', units, 'g', mass)
        self.mass_units = units

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
        rewinds the release to original status (before anything has been
        released).
        """
        self.release.rewind()

    def num_elements_to_release(self, current_time, time_step):
        """
        Determines the number of elements to be released during:
        current_time + time_step

        It invokes the num_elements_to_release method for the the unerlying
        release object: self.release.num_elements_to_release()

        :param current_time: current time
        :type current_time: datetime.datetime
        :param time_step: the time step, sometimes used to decide how many
            should get released.
        :type time_step: integer seconds

        :returns: the number of elements that will be released. This is taken
            by SpillContainer to initialize all data_arrays.
        """
        return self.release.num_elements_to_release(current_time, time_step)

    def set_newparticle_values(self, num_new_particles, current_time,
                               time_step, data_arrays):
        """
        SpillContainer will release elements and initialize all data_arrays
        to default initial value. The SpillContainer gets passed as input and
        the data_arrays for 'position' get initialized correctly by the release
        object: self.release.set_newparticle_positions()

        :param num_new_particles: number of new particles that were added
        :type num_new_particles: int
        :param current_time: current time
        :type current_time: datetime.datetime
        :param time_step: the time step, sometimes used to decide how many
            should get released.
        :type time_step: integer seconds
        :param data_arrays: dict of data_arrays provided by the SpillContainer.
            Look for 'positions' array in the dict and update positions for
            latest num_new_particles that are released
        :type data_arrays: dict containing numpy arrays for values

        Also, the set_newparticle_values() method for all element_type gets
        called so each element_type sets the values for its own data correctly
        """
        if self.element_type is not None:
            self.element_type.set_newparticle_values(num_new_particles, self,
                                                 data_arrays)

        self.release.set_newparticle_positions(num_new_particles, current_time,
                                               time_step, data_arrays)

    def serialize(self, do='update'):
        """
        override base serialize implementation
        Need to add node for release object and element_type object
        """
        dict_ = self.to_dict(do)
        schema = spills_schema.Spill()

        rel_type = '{0}()'.format(self.release.__class__.__name__)
        schema.add(eval('spills_schema.{0}'.format(rel_type)))

        json_ = schema.serialize(dict_)
        json_['element_type'] = self.element_type.serialize(do)

        return json_

    @classmethod
    def deserialize(cls, json_):
        """
        need to create schema dynamically for Spill() before deserialization
        """
        schema = spills_schema.Spill()

        rel_name = json_['release']['obj_type'].rsplit('.', 1)[1]
        rel_type = '{0}()'.format(rel_name)
        schema.add(eval('spills_schema.{0}'.format(rel_type)))

        dict_ = schema.deserialize(json_)
        et_type = json_['element_type']['obj_type']
        to_eval = "{0}.deserialize(json_['element_type'])".format(et_type)
        dict_['element_type'] = eval(to_eval)

        return dict_

""" Helper functions """


def point_line_release_spill(num_elements,
        start_position,
        release_time,
        end_position=None,
        end_release_time=None,
        element_type=None,
        on=True,
        volume=None,
        volume_units='m^3'):
    release = PointLineRelease(num_elements, start_position, release_time,
                               end_position, end_release_time)
    return Spill(release, element_type, on, volume, volume_units)
