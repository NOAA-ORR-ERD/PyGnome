"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes provide
the logic about where an when the elements are released

"""
import copy
from inspect import getmembers, ismethod
from itertools import chain

import numpy
np = numpy

from hazpy import unit_conversion
uc = unit_conversion
from colander import (SchemaNode, Bool, String)

import gnome    # required by new_from_dict
from gnome.utilities import serializable
from gnome.persist.base_schema import ObjType

from . import elements
from .release import PointLineRelease


class SpillSchema(ObjType):
    'Spill class schema'
    on = SchemaNode(Bool(), default=True, missing=True,
        description='on/off status of spill')


class Spill(serializable.Serializable):
    """
    Models a spill
    """
    _update = ['on', 'release', 'element_type']

    _create = []
    _create.extend(_update)

    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(save=_create, update=_update)
    _schema = SpillSchema

    valid_vol_units = list(chain.from_iterable([item[1] for item in
                           uc.ConvertDataUnits['Volume'].values()]))
    valid_vol_units.extend(unit_conversion.GetUnitNames('Volume'))

    valid_mass_units = list(chain.from_iterable([item[1] for item in
                            uc.ConvertDataUnits['Mass'].values()]))
    valid_mass_units.extend(uc.GetUnitNames('Mass'))

    def __init__(self, release,
                 element_type=None,
                 on=True,
                 volume=None, volume_units='m^3',
                 # Is this total mass of the spill?
                 mass=None, mass_units='kg',
                 name='Spill'):
        """
        Spills used by the gnome model. It contains a release object, which
        releases elements. It also contains an element_type object which
        contains the type of substance spilled and it initializes data arrays
        to non-default values (non-zero).

        :param release: an object defining how elements are to be released
        :type release: derived from a gnome.spill.Release object

        **Optional parameters (kwargs):**

        :param element_type=None: list of various element_type that are
            released. These are spill specific properties of the elements.
        :type element_type: list of gnome.element_type.* objects
        :param on=True: Toggles the spill on/off (bool).
        :type on: bool
        :param volume=None: oil spilled volume (used for mass per particle)
        :type volume: float
        :param volume_units=m^3: volume units
        :type volume_units: str
        :param mass=None:
        :type mass: float
        :param mass_units='kg':
        :type mass_units: str
        :param name='Spill': a name for the spill
        :type name: str

        ::note: Define either 'volume' or 'mass', cannot set both.
        """
        self.release = release
        if element_type is None:
            element_type = elements.floating()
        self.element_type = element_type

        self.on = on    # spill is active or not

        # mass/volume, type of oil spilled
        if mass is not None and volume is not None:
            raise ValueError("'mass' and 'volume' cannot both be set")

        self._check_units(volume_units)
        self._volume_units = volume_units   # user defined for display
        self._check_units(mass_units, 'Mass')
        self._mass_units = mass_units   # user defined for display
        self._volume = None
        self._mass = None

        if volume is not None:
            self._volume = uc.convert('Volume', volume_units,
                'm^3', volume)
            self._mass = (self.element_type.substance.get_density('kg/m^3') *
                self._volume)
        elif mass is not None:
            self._mass = uc.convert('Mass', mass_units, 'kg', mass)
            self._volume = (self._mass /
                    self.element_type.substance.get_density('kg/m^3'))

        self.name = name

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release={0.release!r}, '
                'element_type={0.element_type}, '
                'on={0.on}, '
                'volume={0.volume}, '
                'volume_units="{0.volume_units}", '
                'mass={0.mass}, '
                'mass_units="{0.mass_units}"'
                ')'.format(self))

    def __eq__(self, other):
        """
        over ride base == operator defined in Serializable class.
        Spill object contains nested objects like ElementType and Release
        objects. Check all properties here so nested objects properties
        can be checked in the __eq__ implementation within the nested objects
        """
        if not self._check_type(other):
            return False

        if (self._state.get_field_by_attribute('save') !=
            other._state.get_field_by_attribute('save')):
            return False

        for name in self._state.get_names('save'):
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

    def set(self, prop, val):
        """
        sets an existing property. The property could be of one of the
        contained objects like 'Release' or 'ElementType'
        It can also be a property of one of the initializers contained in
        the 'ElementType' object.

        If the property doesn't exist for any of these, then an error is raised
        since user cannot set a property that does not exist using this method

        For example: set('windage_range', (0.4, 0.4)) sets the windage_range
        assuming the element_type is floating

        .. fixme: There is an issue in that if two initializers have the same
        property - could be the case if they both define a 'distribution', then
        it does not know which one to return
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
                    raise AttributeError('{0} attribute does not exist '
                                         'in element_type '
                                         'or release object'.format(prop))

    def get(self, prop=None):
        """
        for get(), return all properties of embedded release object and
        element_type initializer objects. If 'prop' is not None, then return
        the property

        For example: get('windage_range') returns the 'windage_range' assuming
        the element_type = floating()

        .. fixme: There is an issue in that if two initializers have the same
        property - could be the case if they both define a 'distribution', then
        it does not know which one to return

        """
        'Return all properties'
        if prop is None:
            all_props = []

            # release properties
            rel_props = getmembers(self.release,
                                   predicate=lambda p: (not ismethod(p)))
            rel_props = [a[0] for a in rel_props if not a[0].startswith('_')]

            all_props.extend(rel_props)

            # element_type properties
            et_props = getmembers(self.element_type,
                                  predicate=lambda p: (not ismethod(p)))
            'remove _state - update this after we change _state to _state'
            et_props = [a[0] for a in et_props
                            if not a[0].startswith('_') and a[0] != '_state']

            all_props.extend(et_props)

            # properties for each of the initializer objects
            i_props = []
            for val in self.element_type.initializers.values():
                toadd = getmembers(val, lambda p: (not ismethod(p)))
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

    def is_initializer(self, key):
        '''
        a way to test whether an initializer for the given 'key' or data_array
        exists
        '''
        if key in self.element_type.initializers:
            return True
        else:
            return False

    def get_initializer(self, key):
        '''
        returns the initializer associated with 'key'.

        The 'key' refers to the data_array that a mover requires and that the
        initializer is setting. For instance,

            {'rise_vel' : InitRiseVelFromDist()}

        is an initializer that sets the 'rise_vel' if a RiseVelocityMover is
        included in the Model.
        '''
        if self.is_initializer(key):
            return self.element_type.initializers[key]

    def set_initializer(self, key, init):
        '''
        set the given initializer. The 'key' refers to the data_array that a
        mover requires and that the initializer is setting. For instance,

            {'rise_vel' : InitRiseVelFromDist()}

        is an initializer that sets the 'rise_vel' if a RiseVelocityMover is
        included in the Model.
        '''
        self.element_type.initializers[key] = init

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
                      lambda self, value: self.set_mass(value, 'kg'))

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
            return uc.convert('Mass', 'kg', self.mass_units, self._mass)
        else:
            self._check_units(units, 'Mass')
            return uc.convert('Mass', 'kg', units, self._mass)

    def set_mass(self, mass, units):
        '''
        Set the mass released during the spill.
        The default units for mass are as defined in 'mass_units' property.
        User can also specify desired output units in the function.
        '''
        self._check_units(units, 'Mass')
        self._mass = uc.convert('Mass', units, 'kg', mass)
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

    def serialize(self, json_='webapi'):
        """
        override base serialize implementation
        Need to add node for release object and element_type object
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        #schema = self.__class__._schema(
        #    release=self.release.__class__._schema(json_))

        o_json_ = schema.serialize(toserial)
        o_json_['element_type'] = self.element_type.serialize(json_)
        o_json_['release'] = self.release.serialize(json_)

        return o_json_

    @classmethod
    def deserialize(cls, json_):
        """
        Instead of creating schema dynamically for Spill() before
        deserialization, call nested object's serialize/deserialize methods
        """
        schema = cls._schema()

        dict_ = schema.deserialize(json_)
        element_type = json_['element_type']['obj_type']
        dict_['element_type'] = eval(element_type).deserialize(
                                                        json_['element_type'])
        rel = json_['release']['obj_type']
        dict_['release'] = eval(rel).deserialize(json_['release'])

        if json_['json_'] == 'save':
            '''
            convert nested dict back into objects. For the 'webapi', we're not
            always creating a new object so do this only for 'save' files
            '''
            for name in ['release', 'element_type']:
                obj_dict = dict_.pop(name)
                obj_type = obj_dict.pop('obj_type')
                obj = eval(obj_type).new_from_dict(obj_dict)
                dict_[name] = obj

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
        volume_units='m^3',
        name='Point/Line Release'):
    release = PointLineRelease(release_time, num_elements, start_position,
                               end_position, end_release_time)
    return Spill(release, element_type, on, volume, volume_units,
                 name=name)
