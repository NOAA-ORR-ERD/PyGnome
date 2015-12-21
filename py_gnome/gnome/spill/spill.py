"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes provide
the logic about where an when the elements are released

"""
import copy
from inspect import getmembers, ismethod
from datetime import timedelta

import unit_conversion as uc
from colander import (SchemaNode, Bool, String, Float, drop)

from gnome.utilities import serializable
from gnome.persist import class_from_objtype
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

    _create = ['frac_coverage']
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
                 substance=None,
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
        :type release: derived from :class:`~gnome.spill.Release`

        **Optional parameters (kwargs):**

        :param element_type: ElementType object defining the type
            of elements that are released. These are spill specific properties
            of the elements.
        :type element_type:
            :class:`~gnome.spill.elements.element_type.ElementType`
        :param bool on=True: Toggles the spill on/off.
        :param float amount=None: mass or volume of oil spilled.
        :param str units=None: must provide units for amount spilled.
        :param float amount_uncertainty_scale=0.0: scale value in range 0-1
                                                   that adds uncertainty to the
                                                   spill amount.
                                                   Maximum uncertainty scale
                                                   is (2/3) * spill_amount.
        :param str name='Spill': a name for the spill.

        .. note::
            Define either volume or mass in 'amount' attribute and provide
            appropriate 'units'. Defines default element_type as floating
            elements with mass if the Spill's 'amount' property is not None.
            If amount property is None, then just floating elements
            (ie. 'windages')
        """

        self.release = release
        if element_type is None:
            element_type = elements.floating(substance=substance)
        elif substance is not None:
            raise ValueError('Substance and element_type cannot both be '
                             'specified')

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
        fraction of area covered by oil
        '''
        self.frac_coverage = 1.0
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
            ex = uc.InvalidUnitError(msg)
            self.logger.exception(ex, exc_info=True)
            raise ex  # this should be raised since run will fail otherwise

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

    def _elem_mass(self, num_new_particles, current_time, time_step):
        '''
        get the mass of each element released in duration specified by
        'time_step'
        Function is only called if num_new_particles > 0 - no check is made
        for this case
        '''
        # set 'mass' data array if amount is given
        le_mass = 0.
        _mass = self.get_mass('kg')
        self.logger.debug(self._pid + "spill mass (kg): {0}".format(_mass))

        if _mass is not None:
            rd_sec = self.get('release_duration')
            if rd_sec == 0:
                try:
                    le_mass = _mass / self.get('num_elements')
                except TypeError:
                    le_mass = _mass / self.get('num_per_timestep')
            else:
                time_at_step_end = current_time + timedelta(seconds=time_step)
                if self.get('release_time') > current_time:
                    # first time_step in which particles are released
                    time_step = (time_at_step_end -
                                 self.get('release_time')).total_seconds()

                if self.get('end_release_time') < time_at_step_end:
                    time_step = (self.get('end_release_time') -
                                 current_time).total_seconds()

                _mass_in_ts = _mass/rd_sec * time_step
                le_mass = _mass_in_ts / num_new_particles

        self.logger.debug(self._pid + "LE mass (kg): {0}".format(le_mass))

        return le_mass

    def contains_object(self, obj_id):
        for o in (self.element_type, self.release):
            if o.id == obj_id:
                return True

            if (hasattr(o, 'contains_object') and
                    o.contains_object(obj_id)):
                return True

        return False

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

        .. todo::
            There is an issue in that if two initializers have the same
            property - could be the case if they both define a 'distribution',
            then it does not know which one to return
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
                                        'in element_type or release object'
                                        .format(prop))

    def get(self, prop=None):
        """
        for get(), return all properties of embedded release object and
        element_type initializer objects. If 'prop' is not None, then return
        the property

        For example: get('windage_range') returns the 'windage_range' assuming
        the element_type = floating()

        .. todo::
            There is an issue in that if two initializers have the same
            property - could be the case if they both define a 'distribution',
            then it does not know which one to return
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

    def has_initializer(self, name):
        '''
        Returns True if an initializer is present in the list which sets the
        data_array corresponding with 'name', otherwise returns False
        '''
        for i in self.element_type.initializers:
            if name in i.array_types:
                return True

        return False

    def get_initializer(self, name=None):
        '''
        If name is None, return list of all initializers else return
        initializer that sets given 'name'.
        'name' refers to the data_array initialized by initializer.
        For instance, if name='rise_vel', function will look in all
        initializers to find the one whose array_types contain 'rise_vel'.

        If multiple initializers set 'name', then return the first one in the
        list. Although nothing prevents the user from having two initializers
        for the same data_array, it doesn't make much sense.

        The default 'name' of an initializer is the data_array that a mover
        requires and that the initializer is setting. For instance,

            init = InitRiseVelFromDist()
            init.name is 'rise_vel' by default

        is an initializer that sets the 'rise_vel' if a RiseVelocityMover is
        included in the Model. User can change the name of the initializer

        '''
        if name is None:
            return self.element_type.initializers

        init = None
        for i in self.element_type.initializers:
            if name in i.array_types:
                return i

        return init

    def set_initializer(self, init):
        '''
        set/add given initializer. Function looks for first initializer in list
        with same array_types and replaces it if found else it appends this to
        list of initializers.

        .. note::
            nothing prevents user from defining two initializers that
            set the same data_arrays; however, there isn't a use case for it

        '''
        ix = [ix for ix, i in enumerate(self.element_type.initializers)
              if sorted(i.array_types) == sorted(init.array_types)]
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
        User can also specify desired output units in the function.
        If units are not specified, then return in 'SI' units ('kg')
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

        There are a number of python objects that cannot be deepcopied.
        - Logger objects

        So we copy them temporarily to local variables before we deepcopy
        our Spill object.
        """
        u_copy = copy.deepcopy(self)
        self.logger.debug(self._pid + "deepcopied spill {0}".format(self.id))

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
        :param int time_step: the time step, sometimes used to decide how many
            should get released.

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

        If a Spill Amount is given, the Spill object also sets the 'mass' data
        array; else 'mass' array remains '0'

        :param int num_new_particles: number of new particles that were added.
            Always greater than 0
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

        data_arrays['mass'][-num_new_particles:] = \
            self._elem_mass(num_new_particles, current_time, time_step)

        # set arrays that are spill specific - 'frac_coverage'
        if 'frac_coverage' in data_arrays:
            data_arrays['frac_coverage'][-num_new_particles:] = \
                self.frac_coverage

    def serialize(self, json_='webapi'):
        """
        override base serialize implementation
        Need to add node for release object and element_type object
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

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
            relcls = class_from_objtype(json_['release']['obj_type'])
            dict_['release'] = relcls.deserialize(json_['release'])

            if json_['json_'] == 'webapi':
                '''
                save files store a reference to element_type so it will get
                deserialized, created and added to this dict by load method
                '''
                etcls = \
                    class_from_objtype(json_['element_type']['obj_type'])
                dict_['element_type'] = \
                    etcls.deserialize(json_['element_type'])

            else:
                '''
                Convert nested dict (release object) back into object. The
                ElementType is now saved as a reference so it is taken care of
                by load method
                For the 'webapi', we're not always creating a new object
                so do this only for 'save' files
                '''
                obj = relcls.new_from_dict(dict_.pop('release'))
                dict_['release'] = obj

            return dict_
        else:
            return json_

""" Helper functions """


def surface_point_line_spill(num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             substance=None,
                             amount=None,
                             units=None,
                             windage_range=(.01, .04),
                             windage_persist=900,
                             name='Surface Point/Line Release'):
    '''
    Helper function returns a Spill object

    :param num_elements: total number of elements to be released
    :type num_elements: integer
    :param start_position: initial location the elements are released
    :type start_position: 3-tuple of floats (long, lat, z)
    :param release_time: time the LEs are released (datetime object)
    :type release_time: datetime.datetime
    :param end_position=None: Optional. For moving source, the end position
                              If None, then release is from a point source
    :type end_position: 3-tuple of floats (long, lat, z)
    :param end_release_time=None: optional -- for a time varying release,
        the end release time. If None, then release is instantaneous
    :type end_release_time: datetime.datetime
    :param substance=None: Type of oil spilled.
    :type substance: str or OilProps
    :param float amount=None: mass or volume of oil spilled
    :param str units=None: units for amount spilled
    :param tuple windage_range=(.01, .04): Percentage range for windage.
                                           Active only for surface particles
                                           when a mind mover is added
    :param int windage_persist=900: Persistence for windage values in seconds.
                                    Use -1 for inifinite, otherwise it is
                                    randomly reset on this time scale
    :param str name='Surface Point/Line Release': a name for the spill
    '''
    release = PointLineRelease(release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_position=end_position,
                               end_release_time=end_release_time)

    element_type = elements.floating(windage_range=windage_range,
                                     windage_persist=windage_persist,
                                     substance=substance)

    return Spill(release,
                 element_type=element_type,
                 amount=amount,
                 units=units,
                 name=name)


def subsurface_plume_spill(num_elements,
                           start_position,
                           release_time,
                           distribution,
                           distribution_type='droplet_size',
                           end_release_time=None,
                           substance=None,
                           density=None,
                           density_units='kg/m^3',
                           amount=None,
                           units=None,
                           windage_range=(.01, .04),
                           windage_persist=900,
                           name='Subsurface plume'):
    '''
    Helper function returns a Spill object

    :param num_elements: total number of elements to be released
    :type num_elements: integer
    :param start_position: initial location the elements are released
    :type start_position: 3-tuple of floats (long, lat, z)
    :param release_time: time the LEs are released (datetime object)
    :type release_time: datetime.datetime
    :param distribution=None: An object capable of generating a probability
                              distribution.  Right now, we have:
                               * UniformDistribution
                               * NormalDistribution
                               * LogNormalDistribution
                               * WeibullDistribution
    :type distribution: gnome.utilities.distribution
    :param str distribution_type=droplet_size: What is being sampled from the
                                               distribution.  Options are:
                                                * droplet_size - Rise velocity
                                                                 is then
                                                                 calculated
                                                * rise_velocity - No droplet
                                                                  size is
                                                                  computed
    :param end_release_time=None: End release time for a time varying release.
                                  If None, then release is instantaneous
    :type end_release_time: datetime.datetime
    :param substance='None': Required unless density specified.
                             Type of oil spilled.
    :type substance: str or OilProps
    :param float density=None: Required unless substance specified.
                               Density of spilled material.
    :param str density_units='kg/m^3':
    :param float amount=None: mass or volume of oil spilled.
    :param str units=None: must provide units for amount spilled.
    :param tuple windage_range=(.01, .04): Percentage range for windage.
                                           Active only for surface particles
                                           when a mind mover is added
    :param windage_persist=900: Persistence for windage values in seconds.
                                Use -1 for inifinite, otherwise it is
                                randomly reset on this time scale.
    :param str name='Surface Point/Line Release': a name for the spill.
    '''

    release = PointLineRelease(release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_release_time=end_release_time)

    # This helper function is just passing parameters thru to the plume
    # helper function which will do the work.
    # But this way user can just specify all parameters for release and
    # element_type in one go...
    element_type = elements.plume(distribution_type=distribution_type,
                                  distribution=distribution,
                                  substance_name=substance,
                                  windage_range=windage_range,
                                  windage_persist=windage_persist,
                                  density=density,
                                  density_units=density_units)

    return Spill(release,
                 element_type=element_type,
                 amount=amount,
                 units=units,
                 name=name)


def point_line_release_spill(num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             element_type=None,
                             substance=None,
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
                 substance,
                 on,
                 amount,
                 units,
                 name=name)
