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
from colander import (SchemaNode, Bool, String, Float, drop)

import gnome    # required by new_from_dict
from gnome.utilities import serializable
from gnome.persist.base_schema import ObjType

from . import elements
from .release import PointLineRelease
from .. import _valid_units


class SpillSchema(ObjType):
    'Spill class schema'
    on = SchemaNode(Bool(), default=True, missing=True,
                    description='on/off status of spill')
    amount = SchemaNode(Float(), missing=drop)
    units = SchemaNode(String(), missing=drop)
    amount_uncertainty_scale = SchemaNode(Float(), missing=drop)


class Spill(serializable.Serializable):
    """
    Models a spill
    """
    _update = ['on', 'release',
               'amount', 'units', 'amount_uncertainty_scale']

    _create = ['frac_coverage', 'frac_water']
    _create.extend(_update)

    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(save=_create, update=_update)
    _state += serializable.Field('element_type',
                                 save=True,
                                 save_reference=True,
                                 update=True)
    _schema = SpillSchema

    valid_vol_units = _valid_units('Volume')
    valid_mass_units = _valid_units('Mass')

    def __init__(self, release,
                 element_type=None,
                 on=True,
                 amount=None,   # could be volume or mass
                 units=None,
                 amount_uncertainty_scale=0.0,
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
        :param amount=None: mass or volume of oil spilled
        :type amount: float
        :param units=None: must provide units for amount spilled
        :type units: str
        :param name='Spill': a name for the spill
        :type name: str

        ::note: Define either 'volume' or 'mass'; if both are given, then use
        volume to set the mass and ignore the input mass. Defines
        default element_type as floating elements (with windages) and
        mass if the Spill's 'mass' property is not None. If 'mass' property is
        None, then just floating elements (with 'windages')
        """
        self.release = release
        if element_type is None:
            if amount is None:
                # default element type sets substance='oil_conservative'
                element_type = elements.floating()
            else:
                element_type = elements.floating_mass()

        self.element_type = element_type

        self.on = on    # spill is active or not
        self._units = None
        self.amount = amount

        if amount is not None:
            if units is None:
                raise TypeError("Units must be provided with amount spilled")
            else:
                self.units = units

        self.amount_uncertainty_scale = amount_uncertainty_scale

        '''
        fractional water content in the emulsion
        fraction of area covered by oil
        '''
        self.frac_coverage = 1.0
        self.frac_water = 0.0
        self.name = name

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release={0.release!r}, '
                'element_type={0.element_type}, '
                'on={0.on}, '
                'amount={0.amount}, '
                'units="{0.units}", '
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

    def _check_units(self, units):
        """
        Checks the user provided units are in list of valid volume
        or mass units
        """

        if (units in self.valid_vol_units or
                units in self.valid_mass_units):
            return True
        else:
            msg = ('Units for amount spilled must be in volume or mass units. '
                   'Valid units for volume: {0}, for mass: {1} ').format(
                       self.valid_vol_units, self.valid_mass_units)
            #raise uc.InvalidUnitError(msg)
            #self.logger.exception(msg)
            ex = uc.InvalidUnitError(msg)
            self.logger.exception(ex, exc_info=True)
            raise ex    # this should be raised since run will fail otherwise

    def _get_all_props(self):
        'return all properties accessible through get'
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
        for val in self.element_type.initializers:
            toadd = getmembers(val, lambda p: (not ismethod(p)))
            i_props.extend([a[0] for a in toadd
                            if not a[0].startswith('_') and a[0] != '_state'])

            all_props.extend(i_props)
        return all_props

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
            self.logger.warning("cannot set 'num_released' attribute")

        # we don't want to add an attribute that doesn't already exist
        # first check to see that the attribute exists, then change it else
        # raise error
        if hasattr(self.release, prop):
            setattr(self.release, prop, val)
        elif hasattr(self.element_type, prop):
            setattr(self.element_type, prop, val)
        else:
            for init in self.element_type.initializers:
                if hasattr(init, prop):
                    setattr(init, prop, val)
                    break
                else:
                    self.logger.warning('{0} attribute does not exist '
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
            return self._get_all_props()

        try:
            return getattr(self.release, prop)
        except AttributeError:
            pass

        try:
            return getattr(self.element_type, prop)
        except AttributeError:
            pass

        for init in self.element_type.initializers:
            try:
                return getattr(init, prop)
            except AttributeError:
                pass

        # nothing returned, then property was not found - raise exception or
        # return None?
        self.logger.warning("{0} attribute does not exist in element_type"
                            " or release object or initializers".format(prop))
        return None

    def get_initializer_by_name(self, name):
        ''' get first initializer in list whose name matches 'name' '''
        init = [i for i in enumerate(self.element_type.initializers)
                if i.name == name]

        if len(init) == 0:
            return None
        else:
            return init[0]

    def is_initializer(self, key):
        '''
        Returns True if an initializer is present in the list which sets the
        data_array corresponding with 'key', otherwise returns False
        '''
        for i in self.element_type.initializers:
            if key in i.array_types:
                return True

        return False

    def get_initializer(self, key=None):
        '''
        if key is None, return list of all initializers else return initializer
        that sets given 'key'. 'key' refers to the data_array initialized by
        initializer. For instance, if key='rise_vel', function will look in
        all initializers to find the one whose array_types contain 'rise_vel'.

        If multiple initializers set 'key', then return the first one in the
        list. Although nothing prevents the user from having two initializers
        for the same data_array, it doesn't make much sense.

        The default 'name' of an initializer is the data_array that a mover
        requires and that the initializer is setting. For instance,

            init = InitRiseVelFromDist()
            init.name is 'rise_vel' by default

        is an initializer that sets the 'rise_vel' if a RiseVelocityMover is
        included in the Model. User can change the name of the initializer
        '''
        if key is None:
            return self.element_type.initializers

        init = None
        for i in self.element_type.initializers:
            if key in i.array_types:
                return i

        return init

    def set_initializer(self, init):
        '''
        set/add given initializer. Function looks for first initializer in list
        with same array_types and replaces it if found else it appends this to
        list of initializers.

        .. note:: nothing prevents user from defining two initializers that
        set the same data_arrays; however, there isn't a use case for it
        '''
        ix = [ix for ix, i in enumerate(self.element_type.initializers)
              if sorted(i.array_types.keys()) ==
              sorted(init.array_types.keys())]
        if len(ix) == 0:
            self.element_type.initializers.append(init)
        else:
            self.element_type.initializers[ix[0]] = init

    def del_initializer(self, name):
        '''
        delete the initializer with given 'name'

        The default 'name' of an initializer is the data_array that a mover
        requires and that the initializer is setting. For instance,
        the following is an initializer that sets the 'rise_vel' if a
        RiseVelocityMover is included in the Model.

            init = InitRiseVelFromDist()
            init.name is 'rise_vel' by default

        If name = 'rise_vel', all initializers with this name will be deleted
        '''
        ixs = [ix for ix, i in enumerate(self.element_type.initializers)
               if i.name == name]
        for ix in ixs:
            del self.element_type.initializers[ix]

    @property
    def units(self):
        """
        Default units in which amount of oil spilled was entered by user.
        The 'amount' property is returned in these 'units'
        """
        return self._units

    @units.setter
    def units(self, units):
        """
        set default units in which volume data is returned
        """
        self._check_units(units)  # check validity before setting
        self._units = units

    def get_mass(self, units=None):
        '''
        Return the mass released during the spill.
        The default units for mass are as defined in 'mass_units' property.
        User can also specify desired output units in the function.
        If volume is given, then use density to find mass. Density is always
        at 15degC, consistent with API definition
        '''
        if self.amount is None:
            return self.amount

        # first convert amount to 'kg'
        if self.units in self.valid_mass_units:
            mass = uc.convert('Mass', self.units, 'kg', self.amount)
        elif self.units in self.valid_vol_units:
            vol = uc.convert('Volume', self.units, 'm^3', self.amount)
            mass = self.element_type.substance.get_density() * vol

        if units is None or units == 'kg':
            return mass
        else:
            self._check_units(units)
            return uc.convert('Mass', 'kg', units, mass)

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

    def set_amount_uncertainty(self, up_or_down=None):
        '''
            This function shifts the spill amount based on a scale value
            in the range [0.0 ... 1.0].  The maximum uncertainty scale value
            is (2/3) * spill_amount.
            We determine either an upper uncertainty or a lower uncertainty
            multiplier.  Then we shift our spill amount value based on it.

            Since we are irreversibly changing the spill amount value,
            we should probably do this only once.
        '''
        if (self.amount_uncertainty_scale <= 0.0 or
                self.amount_uncertainty_scale > 1.0):
            return False

        if up_or_down == 'up':
            scale = (1.0 + (2.0 / 3.0) * self.amount_uncertainty_scale)
        elif up_or_down == 'down':
            scale = (1.0 - (2.0 / 3.0) * self.amount_uncertainty_scale)
        else:
            return False

        self.amount *= scale

        return True

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

        We also need to accept sparse json objects, in which case we will
        not treat them, but just send them back.
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()

            dict_ = schema.deserialize(json_)
            rel = json_['release']['obj_type']
            dict_['release'] = eval(rel).deserialize(json_['release'])

            if json_['json_'] == 'webapi':
                '''
                save files store a reference to element_type so it will get
                deserialized, created and added to this dict by load method
                '''
                element_type = json_['element_type']['obj_type']
                dict_['element_type'] = (eval(element_type).deserialize(
                                                        json_['element_type']))

            else:
                '''
                Convert nested dict (release object) back into object. The
                ElementType is now saved as a reference so it is taken care of
                by load method
                For the 'webapi', we're not always creating a new object
                so do this only for 'save' files
                '''
                obj_dict = dict_.pop('release')
                obj_type = obj_dict.pop('obj_type')
                obj = eval(obj_type).new_from_dict(obj_dict)
                dict_['release'] = obj

            return dict_
        else:
            return json_

""" Helper functions """


def point_line_release_spill(num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             element_type=None,
                             on=True,
                             amount=None,
                             units=None,
                             name='Point/Line Release'):
    '''
    Helper function returns a Spill object containing a point or line release
    '''
    release = PointLineRelease(release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_position=end_position,
                               end_release_time=end_release_time)
    return Spill(release,
                 element_type,
                 on,
                 amount,
                 units,
                 name=name)
