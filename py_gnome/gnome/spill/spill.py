"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes provide
the logic about where an when the elements are released

"""

import copy
import inspect
from itertools import chain

import numpy
np = numpy

from hazpy import unit_conversion
uc = unit_conversion
from colander import (SchemaNode, Bool)

import gnome    # required by new_from_dict
from gnome.utilities import serializable
from gnome.persist.base_schema import ObjType

from . import elements
from .release import PointLineRelease


class SpillSchema(ObjType):
    'Spill class schema'
    on = SchemaNode(Bool(), default=True, missing=True,
        description='on/off status of spill')

    #==========================================================================
    # def __init__(self, **kwargs):
    #     'add release object to schema instance'
    #     rel = kwargs.pop('release')
    #     self.add(rel)
    #     super(SpillSchema, self).__init__(**kwargs)
    #==========================================================================


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
    _state.add(save=_create, update=_update)
    _schema = SpillSchema

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
        if dict_['json_'] == 'save':
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

    def __init__(self, release,
                 element_type=None,
                 on=True,
                 volume=None,
                 volume_units='m^3',
                 # Is this total mass of the spill?
                 mass=None,
                 mass_units='g'):
        """
        Base spill class. Spill used by a gnome model derive from this class

        :param num_elements: number of LEs - default is 0.
        :type num_elements: int

        Optional parameters (kwargs):

        :param on: Toggles the spill on/off (bool). Default is 'on'.
        :type on: bool
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

    def serialize(self, json_='webapi'):
        """
        override base serialize implementation
        Need to add node for release object and element_type object
        """
        toserial = self.to_dict(json_)
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
        #======================================================================
        # rel_name = json_['release']['obj_type']
        # rel_schema = eval(rel_name)._schema(json_['json_'])
        # schema = cls._schema(release=rel_schema)
        #======================================================================
        schema = cls._schema()

        dict_ = schema.deserialize(json_)
        element_type = json_['element_type']['obj_type']
        dict_['element_type'] = eval(element_type).deserialize(
                                                        json_['element_type'])
        rel = json_['release']['obj_type']
        dict_['release'] = eval(rel).deserialize(json_['release'])

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
    release = PointLineRelease(release_time, num_elements, start_position,
                               end_position, end_release_time)
    return Spill(release, element_type, on, volume, volume_units)
