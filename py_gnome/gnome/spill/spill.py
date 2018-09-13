"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes combine

Releases: where and when elements are released
and
Element_types -- what the types of the elements are.

"""
from datetime import timedelta
import copy
from inspect import getmembers, ismethod


import unit_conversion as uc
from gnome.utilities.time_utils import asdatetime

from colander import (SchemaNode, Bool, String, Float, drop, SequenceSchema)

from gnome.gnomeobject import GnomeId
from gnome.persist.base_schema import ObjTypeSchema, GeneralGnomeObjectSchema
from gnome.spill.initializers import (InitWindagesSchema,
                                    DistributionBaseSchema,
                                    floating_initializers,
                                    plume_from_model_initializers,
                                    plume_initializers)
from gnome.persist.base_schema import ObjType

from .release import (PointLineRelease,
                      ContinuousRelease,
                      GridRelease,
                      SpatialRelease)
from .. import _valid_units
from gnome.spill.release import BaseReleaseSchema, PointLineReleaseSchema,\
    ContinuousReleaseSchema, SpatialReleaseSchema
from gnome.environment.water import WaterSchema
from gnome.spill.le import LEData


class BaseSpill(GnomeId):
    """
    A base class for spills -- so we can check for them, etc.

    and as a spec for the API.
    """
    def __init__(self, **kwargs):
        """
        initialize -- sub-classes will probably have a lot more to do
        """
        super(BaseSpill, self).__init__(**kwargs)
        self.data = LEData()
        from oil_library import get_oil_props
        self.get_oil_props = get_oil_props
        self.initializers = []

    @property
    def release_time(self):
        return self._release_time

    @release_time.setter
    def release_time(self, rt):
        self._release_time = asdatetime(rt)

    @property
    def substance(self):
        return None

    @property
    def array_types(self):
        '''
        compile/return dict of array_types set by all initializers contained
        by ElementType object
        '''
        at = set()
        for init in self.initializers:
            at.update(init.array_types)

        return at

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}()'
                .format(self))

    # what is this for??
    def get_mass(self, units=None):
        '''
        Return the mass released during the spill.
        User can also specify desired output units in the function.
        If units are not specified, then return in 'SI' units ('kg')
        If volume is given, then use density to find mass. Density is always
        at 15degC, consistent with API definition
        '''
        # first convert amount to 'kg'
        if self.units in self.valid_mass_units:
            mass = uc.convert('Mass', self.units, 'kg', self.amount_released)

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

    def rewind(self):
        """
        rewinds the release to original status (before anything has been
        released).
        """
        raise NotImplementedError

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
        if not self.on:
            return 0

        raise NotImplementedError

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
        raise NotImplementedError

    def has_initializer(self, name):
        '''
        Returns True if an initializer is present in the list which sets the
        data_array corresponding with 'name', otherwise returns False

        Override with a real implimentaitn if you are using initializers
        '''

        return False


class SpillSchema(ObjTypeSchema):
    'Spill class schema'
    on = SchemaNode(
        Bool(), default=True, missing=True,
        description='on/off status of spill',
        save=True, update=True
    )
    release = GeneralGnomeObjectSchema(
        acceptable_schemas=[BaseReleaseSchema,
                            PointLineReleaseSchema,
                            ContinuousReleaseSchema,
                            SpatialReleaseSchema],
        save=True, update=True, save_reference=True
    )
    amount = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    units = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    amount_uncertainty_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    water = WaterSchema(
        missing=drop, save=True, update=True, save_reference=True
    )
    initializers = SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[InitWindagesSchema,
                                DistributionBaseSchema
                                ]
        ),
        save=True, update=True, save_reference=True
    )
    standard_density = SchemaNode(
        Float(), read_only=True
    )
    def __init__(self, unknown='preserve', *args, **kwargs):
        super(SpillSchema, self).__init__(*args, **kwargs)
        self.typ = ObjType(unknown)


class Spill(BaseSpill):
    """
    Models a spill by combining Release and ElementType objects
    """
    _schema = SpillSchema

    valid_vol_units = _valid_units('Volume')
    valid_mass_units = _valid_units('Mass')
    # attributes that need to be there for the __setattr__ magic to work
    # release = None  # just to make sure it's there.
    # element_type = None
    # this is so the properties in teh base classes work -- arrgg!
    # _name = 'Spill'

    def __init__(self,
                 release=None,
                 water=None,
                 substance=None,
                 on=True,
                 amount=None,  # could be volume or mass
                 units=None,
                 amount_uncertainty_scale=0.0,
                 **kwargs):
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
        super(Spill, self).__init__(**kwargs)
        self.water = water
        self.release = release

        self.substance = substance

        self.on = on  # spill is active or not
        # raise Exception("stopping")

        # fixme: shouldn't units default to 'kg'?
        self.units = None
        # fixme -- and amount always be in kg?
        self.amount = amount

        if amount is not None:
            if units is None:
                raise TypeError("Units must be provided with amount spilled")
            else:
                self.units = units

        self.amount_uncertainty_scale = amount_uncertainty_scale

        # fixme: why is fractional area part of spill???
        # fraction of area covered by oil
        self.frac_coverage = 1.0


# fixme: a bunch of these properties should really be defined in subclasses
# that use them
    @property
    def release_time(self):
        return self.release.release_time

    @release_time.setter
    def release_time(self, rt):
        self.release.release_time = asdatetime(rt)

    @property
    def end_release_time(self):
        return self.release.end_release_time

    @end_release_time.setter
    def end_release_time(self, rt):
        self.release.end_release_time = asdatetime(rt)

    @property
    def release_duration(self):
        return self.release.release_duration

    @property
    def start_time_invalid(self):
        return self.release.start_time_invalid
    # any reason to set this on a spill??
    # @start_time_invalid.setter
    # def start_time_invalid(self, rd):
    #     self.release.start_time_invalid = rd

    @property
    def num_elements(self):
        return self.release.num_elements

    @num_elements.setter
    def num_elements(self, ne):
        self.release.num_elements = ne

    # doesn't seem like this should be set on the spill object!
    @property
    def num_released(self):
        return self.release.num_released
    # @num_released.setter
    # def num_released(self, ne):
    #     self.release.num_released = ne

    @property
    def start_position(self):
        return self.release.start_position

    @start_position.setter
    def start_position(self, sp):
        self.release.start_position = sp

    @property
    def end_position(self):
        return self.release.end_position

    @end_position.setter
    def end_position(self, sp):
        self.release.end_position = sp

    @property
    def array_types(self):
        return self.element_type.array_types

    @array_types.setter
    def array_types(self, at):
        self.element_type.array_types = at

    @property
    def windage_range(self):
        return self.element_type.windage_range

    @windage_range.setter
    def windage_range(self, at):
        self.element_type.windage_range = at

    @property
    def windage_persist(self):
        return self.element_type.windage_persist

    @windage_persist.setter
    def windage_persist(self, wp):
        self.element_type.windage_persist = wp

    @property
    def substance(self):
        return self._substance

    @substance.setter
    def substance(self, val):
        '''
        first try to use get_oil_props using 'val'. If this fails, then assume
        user has provided a valid OilProps object and use it as is
        '''
        if val is None:
            self._substance = None
            return
        try:
            self._substance = self.get_oil_props(val)
        except Exception:
            if isinstance(val, basestring):
                raise

            self.logger.info('Failed to get_oil_props for {0}. Use as is '
                             'assuming has OilProps interface'.format(val))
            self._substance = val

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release={0.release!r}, '
                'element_type={0.element_type}, '
                'on={0.on}, '
                'amount={0.amount}, '
                'units="{0.units}", '
                ')'.format(self))

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
                   '{} was provided.'
                   'Valid units for volume: {0}, for mass: {1} ').format(
                         units,
                         self.valid_vol_units,
                         self.valid_mass_units)
            ex = ValueError(msg)
            self.logger.exception(ex, exc_info=True)
            raise ex  # this should be raised since run will fail otherwise

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
            rd_sec = self.release_duration
            if rd_sec == 0:
                try:
                    le_mass = _mass / self.num_elements
                except TypeError:
                    le_mass = _mass / self.num_per_timestep
            else:
                time_at_step_end = current_time + timedelta(seconds=time_step)
                if self.release_time > current_time:
                    # first time_step in which particles are released
                    time_step = (time_at_step_end -
                                 self.release_time).total_seconds()
                if self.end_release_time < time_at_step_end:
                    time_step = (self.end_release_time -
                                 current_time).total_seconds()

                _mass_in_ts = _mass / rd_sec * time_step
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
        if units is not None:
            self._check_units(units)  # check validity before setting
        self._units = units

    def get_mass(self, units=None):
        """
        Return the mass released during the spill.
        User can also specify desired output units in the function.
        If units are not specified, then return in 'SI' units ('kg')
        If volume is given, then use density to find mass. Density is always
        at 15degC, consistent with API definition
        """
        # fixme: This really should be re-factored to always store mass.
        if self.amount is None:
            return self.amount

        if self.units in self.valid_mass_units:
            # first convert amount to 'kg'
            mass = uc.convert('Mass', self.units, 'kg', self.amount)
        elif self.units in self.valid_vol_units:
            # need to convert to mass
                # DO NOT change this back!
                # for the UI to be consistent, the conversion needs to use
                # standard density -- not the current water temp.
                # water_temp = self.water.get('temperature')
                # substance has a "standard_density" attribute
                # for this.
            std_rho = self.element_type.standard_density

            vol = uc.convert('Volume', self.units, 'm^3', self.amount)
            mass = std_rho * vol
        else:
            raise ValueError("{} is not a valid mass or Volume unit"
                             .format(self.units))

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
        self.data.rewind()
        self.initializers = []

    @classmethod
    def new_from_dict(cls, dict_):
        return super(Spill, cls).new_from_dict(dict_)

    def to_dict(self, json_=None):
        dict_ = super(Spill, self).to_dict(json_=json_)
        #append substance because no good schema exists for it
        if json_ != 'save':
            if self.substance is None:
                dict_['substance'] = None
            else:
                dict_['substance'] = self.substance_to_dict()
        else:
            if self.substance is not None:
                dict_['substance'] = self.substance_to_dict()['name']
        return dict_

    def update_from_dict(self, dict_, refs=None):
        rv = super(Spill, self).update_from_dict(dict_, refs)
        if 'substance' in dict_:
            self.substance = dict_['substance']
            rv = True
        return rv


    def substance_to_dict(self):
        '''
        Call the tojson() method on substance

        An Oil object that has been queried from the database
        contains a lot of unnecessary relationships that we do not
        want to represent in our JSON output,

        So we prune them by first constructing an Oil object from the
        JSON payload of the queried Oil object.

        This creates an Oil object in memory that does not have any
        database links. Then output the JSON from the unlinked object.
        '''
        if self._substance is not None:
            return self._prune_substance(self._substance.tojson())


    def _prune_substance(self, substance_json):
        '''
            Whether saving to a savefile or outputting to the web client,
            the id of the substance objects is not necessary, and in fact
            not even wanted.

            Except for the main oil ID from the database.
        '''
        del substance_json['imported_record_id']
        del substance_json['estimated_id']

        for attr in ('kvis', 'densities', 'cuts',
                     'molecular_weights',
                     'sara_densities', 'sara_fractions'):
            for item in substance_json[attr]:
                for sub_item in ('id', 'oil_id', 'imported_record_id'):
                    if sub_item in item:
                        del item[sub_item]

        return substance_json

    def release_elements(self, current_time, time_step):
        """
        Releases and partially initializes new LEs
        """
        to_rel = self.release.num_elements_to_release(current_time, time_step)
        self.data.extend_data_arrays(self, to_rel)

        #Partial initialization from various objects
        self.data['mass'][-to_rel:] = self._elem_mass(to_rel, current_time, time_step)
        self.release.set_newparticle_positions(to_rel, current_time, time_step, self.data)

        if 'frac_coverage' in self.data:
            self.data['frac_coverage'][-to_rel:] = self.frac_coverage

        for init in self.initializers:
            init.initialize(to_rel, self, self.data, self.substance)

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


""" Helper functions """


def surface_point_line_spill(num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             substance=None,
                             amount=None,
                             units=None,
                             water=None,
                             on=True,
                             windage_range=(.01, .04),
                             windage_persist=900,
                             name='Surface Point or Line Release'):
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

    inits = floating_initializers(windage_range=windage_range,
                                  windage_persist=windage_persist)

    return Spill(release=release,
                 water=water,
                 initializers=inits,
                 substance=substance,
                 amount=amount,
                 units=units,
                 name=name,
                 on=on,
                 units=units)


def grid_spill(bounds,
               resolution,
               release_time,
               substance=None,
               amount=None,
               units=None,
               on=True,
               water=None,
               windage_range=(.01, .04),
               windage_persist=900,
               name='Surface Grid Spill'):
    '''
    Helper function returns a Grid Spill object

    :param bounds: bounding box of region you want the elements in:
                   ((min_lon, min_lat),
                    (max_lon, max_lat))
    :type bounds: 2x2 numpy array or equivalent

    :param resolution: resolution of grid -- it will be a resoluiton X
                       resolution grid
    :type resolution: integer

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
    release = GridRelease(release_time,
                          bounds,
                          resolution)

    inits = floating_initializers(windage_range=windage_range,
                                  windage_persist=windage_persist)

    return Spill(release=release,
                 water=water,
                 initializers=inits,
                 substance=substance,
                 amount=amount,
                 units=units,
                 name=name,
                 on=on,
                 units=units)


def subsurface_plume_spill(num_elements,
                           start_position,
                           release_time,
                           distribution,
                           distribution_type='droplet_size',
                           end_release_time=None,
                           substance=None,
                           amount=None,
                           units=None,
                           water=None,
                           on=True,
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
                    * droplet_size - Rise velocity is then calculated
                    * rise_velocity - No droplet size is computed

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
    inits = plume_initializers(distribution_type=distribution_type,
                               distribution=distribution,
                               windage_range=windage_range,
                               windage_persist=windage_persist)

    return Spill(release=release,
                 water=water,
                 initializers=inits,
                 substance=substance,
                 amount=amount,
                 units=units,
                 name=name,
                 on=on,
                 units=units)


def continuous_release_spill(initial_elements,
                             num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             water=None,
                             substance=None,
                             on=True,
                             amount=None,
                             units=None,
                             windage_range=(.01, .04),
                             windage_persist=900,
                             name='Point or Line Release'):
    '''
    Helper function returns a Spill object containing a point or line release
    '''
    release = ContinuousRelease(initial_elements=initial_elements,
                                release_time=release_time,
                                start_position=start_position,
                                num_elements=num_elements,
                                end_position=end_position,
                                end_release_time=end_release_time)
    inits = floating_initializers(windage_range=windage_range,
                                  windage_persist=windage_persist)

    return Spill(release=release,
                 water=water,
                 initializers=inits,
                 substance=substance,
                 amount=amount,
                 units=units,
                 name=name,
                 on=on,
                 units=units)


def point_line_release_spill(num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             water=None,
                             substance=None,
                             on=True,
                             amount=None,
                             units=None,
                             windage_range=(.01, .04),
                             windage_persist=900,
                             name='Point or Line Release'):
    '''
    Helper function returns a Spill object containing a point or line release
    '''
    release = PointLineRelease(release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_position=end_position,
                               end_release_time=end_release_time)
    inits = floating_initializers(windage_range=windage_range,
                                  windage_persist=windage_persist)
    return Spill(release=release,
                 water=water,
                 initializers=inits,
                 substance=substance,
                 amount=amount,
                 units=units,
                 name=name,
                 on=on,
                 units=units)


def spatial_release_spill(start_positions,
                          release_time,
                          substance=None,
                          water=None,
                          on=True,
                          amount=None,
                          units=None,
                          windage_range=(.01, .04),
                          windage_persist=900,
                          name='spatial_release'):
    '''
    Helper function returns a Spill object containing a spatial release

    A spatial release is a spill that releases elements at known locations.
    '''
    release = SpatialRelease(release_time=release_time,
                             start_position=start_positions,
                             name=name)
    inits = floating_initializers(windage_range=windage_range,
                                  windage_persist=windage_persist)
    return Spill(release=release,
                 water=water,
                 initializers=inits,
                 substance=substance,
                 amount=amount,
                 units=units,
                 name=name,
                 on=on,
                 units=units)
