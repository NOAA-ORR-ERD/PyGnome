"""
spill.py - An implementation of the spill class(s)

A "spill" is essentially a source of elements. These classes combine

Releases: where and when elements are released
and
Substance -- what the types of the elements are.

(currently there are only two substances: GnomeOIl and NonWeatheringSubstance)

"""

from datetime import datetime, timedelta
import copy

import nucos as uc
from gnome.utilities.time_utils import asdatetime
from gnome.utilities.appearance import SpillAppearanceSchema

from colander import (SchemaNode, Bool, String, Float, drop)

from gnome.gnomeobject import GnomeId
from gnome.persist.base_schema import ObjTypeSchema, GeneralGnomeObjectSchema

from gnome import _valid_units

from gnome.environment.water import WaterSchema

from .release import (Release,
                      PointLineRelease,
                      GridRelease,
                      PolygonRelease,
                      SubsurfaceRelease,
                      BaseReleaseSchema,
                      PointLineReleaseSchema,
                      PolygonReleaseSchema,
                      SubsurfaceReleaseSchema)

from .substance import (Substance,
                        SubstanceSchema,
                        NonWeatheringSubstance,
                        NonWeatheringSubstanceSchema
                        )
from .gnome_oil import (GnomeOil, GnomeOilSchema)

from .initializers import plume_initializers


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
                            PolygonReleaseSchema],
        save=True, update=True, save_reference=True
    )
    substance = GeneralGnomeObjectSchema(
        acceptable_schemas=[GnomeOilSchema,
                            NonWeatheringSubstanceSchema],
        save=True, update=True, save_reference=True
    )
    water = WaterSchema(
        missing=drop, save=True, update=True, save_reference=True
    )
    units = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    amount = SchemaNode(
        Float(), missing=0, save=True, update=True
    )
    amount_uncertainty_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    _appearance = SpillAppearanceSchema(
        save=True, update=True, missing=drop,
        test_equal=False
    )


class BaseSpill(GnomeId):
    """
    A base class for spill, with only what we really need

    Note: this should have all the methods that the model needs, to define the interface.

    for now, the real base is specified by Spill, but this gives us something to derive
    from to make a whole new one, and still be able to add it to the model.
    """
    pass


class Spill(BaseSpill):
    """
    Models a spill by combining Release and Substance objects
    """
    _schema = SpillSchema

    valid_vol_units = _valid_units('Volume')
    valid_mass_units = _valid_units('Mass')
    # attributes that need to be there for the __setattr__ magic to work
    # release = None  # just to make sure it's there.
    # element_type = None
    # this is so the properties in the base classes work -- arrgg!
    # _name = 'Spill'

    def __init__(self,
                 on=True,  # fixme: this shouldn't be the first parameter!
                 num_elements=1000,
                 amount=0,  # could be volume or mass
                 units='kg',
                 substance=None,
                 release=None,
                 water=None,
                 amount_uncertainty_scale=0.0,
                 **kwargs):
        """
        Spills used by the gnome model. It contains a release object, which
        releases elements. It also contains a Substance which
        contains the type of substance spilled and it initializes data arrays
        to non-default values (non-zero).

        :param release: an object defining how elements are to be released
        :type release: derived from :class:`~gnome.spills.release.Release`

        :param substance: an object defining the substance of this spill. Defaults to :class:`~gnome.spills.substance.NonWeatheringSubstance`
        :type substance: derived from :class:`~gnome.spills.substance.Substance`

        **Optional parameters (kwargs):**

        :param name: Human-usable Name of this spill
        :type name: str

        :param on=True: Toggles the spill on/off.
        :type on: bool

        :param amount=None: mass or volume of oil spilled.
        :type amount: double (volume or mass)

        :param units=None: must provide units for amount spilled.
        :type units: str

        :param amount_uncertainty_scale=0.0: scale value in range 0-1
                                             that adds uncertainty to the
                                             spill amount.
                                             Maximum uncertainty scale
                                             is (2/3) * spill_amount.
        :type amount_uncertainty_scale: float

        .. note::
            Define either volume or mass in 'amount' attribute and provide
            appropriate 'units'.
        """
        super(Spill, self).__init__(**kwargs)
        self.on = on
        self.substance = substance
        if release is None:
            release = PointLineRelease(release_time=datetime.now(),
                                       start_position=(0, 0, 0),
                                       num_elements=num_elements
                                       )
            self.release = release
        else:
            self.release = release
            num_elements = release.num_elements
        self.num_elements = num_elements

        self.units = units
        self.amount = amount

#         self.data = LEData()
        self.water = water

        self.amount_uncertainty_scale = amount_uncertainty_scale

        # fixme: why is fractional area part of spill???
        # fraction of area covered by oil
        self.frac_coverage = 1.0
        self._num_released = 0

    @property
    def all_array_types(self):
        '''
        Need to add array types from Release and Substance
        '''
        arr = self.array_types.copy()
        arr.update(self.release.all_array_types)
        arr.update(self.substance.all_array_types)
        return arr

    @property
    def substance(self):
        return self._substance

    # fixme: this should be obsolete -- hopefully it can be removed!
    @substance.setter
    def substance(self, val):
        '''
        first try to use get_oil_props using 'val'. If this fails, then assume
        user has provided a valid OilProps object and use it as is
        '''
        # a bit of type checking -- this bit me, and took a while to debug
        #  e.g. passing in NonWeatheringSubstance instead of NonWeatheringSubstance()
        if isinstance(val, type):
            raise TypeError(f"{val} is a class -- it should be an instance of a class")

        if val is None:
            self._substance = NonWeatheringSubstance()
            return
        #elif isinstance(val, Substance):
        elif isinstance(val, NonWeatheringSubstance):
            self._substance = val
            return
        elif isinstance(val, GnomeOil):
            self._substance = val
            return
        try:
            self._substance = GnomeOil.get_GnomeOil(val)
        except Exception:
            if isinstance(val, str):
                raise

            self.logger.info('Failed to make a GnomeOil for {0}. Use as is '
                             'assuming it has a Substance interface'.format(val))
            self._substance = val

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
    def num_elements(self):
        return self.release.num_elements

    @num_elements.setter
    def num_elements(self, ne):
        self.release.num_elements = ne

    # doesn't seem like this should be set on the spill object!
#     @property
#     def num_released(self):
#         return len(self.data)
    # @num_released.setter
    # def num_released(self, ne):
    #     self.release.num_released = ne

    @property
    def start_position(self):
        return self.release.centroid if not hasattr(self.release, 'start_position') else self.release.start_position

    @start_position.setter
    def start_position(self, sp):
        raise ValueError('Setting release start_position on spill is deprecated')

    @property
    def end_position(self):
        return self.release.centroid if not hasattr(self.release, 'end_position') else self.release.end_position

    @end_position.setter
    def end_position(self, sp):
        raise ValueError('Setting release end_position on spill is deprecated')

    # fixme: We store in standard units! i.e. kilograms!
    #        so the getter should just return that value.
    @property
    def amount(self):
        rel_mass = self.release.release_mass #kg

        if self.units in self.valid_vol_units:
            std_density = self.substance.standard_density #kg/m3
            vol = rel_mass / std_density
            return uc.convert('m^3', self.units, vol)

        if self.units in self.valid_mass_units:
            return uc.convert('kg', self.units, rel_mass)

    @amount.setter
    def amount(self, val):
        if val < 0:
            raise ValueError('amount cannot be less than 0')
        rel_mass = val

        if self.units in self.valid_vol_units:
            #need to get mass
            vol = uc.convert(self.units, 'm^3', val)
            std_density = self.substance.standard_density #kg/m3
            rel_mass = vol * std_density

        if self.units in self.valid_mass_units:
            rel_mass = uc.convert(self.units, 'kg', val)
        self.release.release_mass = rel_mass

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release={0.release!r}, '
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

    def get_mass(self):
        """
        Return the total mass released during the spill.
        """
        return self.release.release_mass

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
        '''
        TODO: Refactor spill container so it calls this only when the model is about to run
        in prepare_for_model_run, and cleans up afterwards. Currently it calls it when a
        spill is added to the spill container pair(?)
        '''
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
        self.array_types = {}
        self._num_released = 0
        self.release.rewind()
#         self.data.rewind()

    def prepare_for_model_run(self, timestep):
        '''
        array_types comes from all the other objects above in the model such as
        movers, weatherers, etc. The ones from the substance still need to be added
        '''
        self.release.prepare_for_model_run(timestep)

    def release_elements(self, sc, start_time, end_time, environment=None):
        """
        Releases and partially initializes new LEs
        """
        if not self.on:
            return 0
        idx = sc.spills.index(self)
        expected_num_release = self.release.num_elements_after_time(end_time)
        actual_num_release = self._num_released
        to_rel = expected_num_release - actual_num_release
        if to_rel <= 0:
            return 0  # nothing to release, so end early
        sc._append_data_arrays(to_rel)
        self._num_released += to_rel

        sc['spill_num'][-to_rel:] = idx

        # Partial initialization from various objects
        self.release.initialize_LEs(to_rel, sc, start_time, end_time)

        if 'frac_coverage' in sc:
            sc['frac_coverage'][-to_rel:] = self.frac_coverage

        self.substance.initialize_LEs(to_rel, sc, environment=environment)

        self.release.initialize_LEs_post_substance(to_rel, sc,
                                                   start_time, end_time,
                                                   environment=environment)

        return to_rel

    def num_elements_to_release(self, current_time, time_step):
        """
        Determines the number of elements to be released during:
        current_time + time_step

        It invokes the num_elements_to_release method for the the underlying
        release object: self.release.num_elements_to_release()

        :param current_time: current time
        :type current_time: datetime.datetime
        :param int time_step: the time step, sometimes used to decide how many
            should get released.

        :returns: the number of elements that will be released. This is taken
            by SpillContainer to initialize all data_arrays.
        """
        return self.release.num_elements_to_release(current_time, time_step)

    def _attach_default_refs(self, ref_dict):
        self.release._attach_default_refs(ref_dict)
        self.substance._attach_default_refs(ref_dict)
        return GnomeId._attach_default_refs(self, ref_dict)


""" Helper functions """

def _setup_spill(release,
                 water,
                 substance,
                 amount,
                 units,
                 name,
                 on,
                 windage_range,
                 windage_persist,
                 ):
    """
    set the windage on the substance, if it's provided

    otherwise simply passes everything in to the Spill
    """
    spill = Spill(release=release,
                  water=water,
                  substance=substance,
                  amount=amount,
                  units=units,
                  name=name,
                  on=on)

    # If windages are provided, they override what's in the Substance
    # The defaults are in the Substance base class
    if windage_range is not None:
        spill.substance.windage_range = windage_range
    if windage_persist is not None:
        spill.substance.windage_persist = windage_persist

    return spill

def surface_point_line_spill(num_elements,
                             start_position,
                             release_time,
                             end_position=None,
                             end_release_time=None,
                             substance=None,
                             amount=0,
                             units='kg',
                             water=None,
                             on=True,
                             windage_range=None,
                             windage_persist=None,
                             name='Surface Point or Line Release'):
    '''
    Helper function returns a Spill object

    :param num_elements: total number of elements to be released
    :type num_elements: integer

    :param start_position: initial location the elements are released
    :type start_position: 2-tuple of floats (long, lat)

    :param release_time: time the LEs are released (datetime object)
    :type release_time: datetime.datetime

    :param end_position=None: Optional. For moving source, the end position
                              If None, then release is from a point source
    :type end_position: 2-tuple of floats (long, lat)

    :param end_release_time=None: optional -- for a time varying release,
        the end release time. If None, then release is instantaneous
    :type end_release_time: datetime.datetime

    :param substance=None: Type of oil spilled.
    :type substance: Substance object

    :param amount=None: mass or volume of oil spilled
    :type amount: float

    :param units=None: units for amount spilled
    :type units: str

    :param tuple windage_range=(.01, .04): Percentage range for windage.
                                           Active only for surface particles
                                           when a mind mover is added
    :type windage_range: tuple

    :param windage_persist=900: Persistence for windage values in seconds.
                                    Use -1 for inifinite, otherwise it is
                                    randomly reset on this time scale
    :type windage_persist: int

    :param name='Surface Point/Line Spill': a name for the spill
    :type name: str
    '''
    # make positions 3d, with depth = 0 if they are not already
    start_position = *start_position[:2], 0

    end_position = (*end_position[:2], 0) if end_position is not None else None

    release = PointLineRelease(release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_position=end_position,
                               end_release_time=end_release_time)

    spill = _setup_spill(release=release,
                         water=water,
                         substance=substance,
                         amount=amount,
                         units=units,
                         name=name,
                         on=on,
                         windage_range=windage_range,
                         windage_persist=windage_persist,
                         )


    return spill


def grid_spill(bounds,
               resolution,
               release_time,
               substance=None,
               amount=1.,
               units='kg',
               on=True,
               water=None,
               windage_range=None,
               windage_persist=None,
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

    :param substance=None: Type of oil spilled.
    :type substance: str or OilProps

    :param float amount=None: mass or volume of oil spilled

    :param str units=None: units for amount spilled

    :param release_time: time the LEs are released (datetime object)
    :type release_time: datetime.datetime

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

    spill = _setup_spill(release=release,
                         water=water,
                         substance=substance,
                         amount=amount,
                         units=units,
                         name=name,
                         on=on,
                         windage_range=windage_range,
                         windage_persist=windage_persist
                         )

    return spill


def subsurface_spill(num_elements,
                     start_position,
                     release_time,
                     distribution,
                     distribution_type='droplet_size',
                     end_release_time=None,
                     substance=None,
                     amount=0,
                     units='kg',
                     water=None,
                     on=True,
                     windage_range=None,
                     windage_persist=None,
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

    :param substance=None: Required unless density specified.
                             Type of oil spilled.

    :type substance: GnomeOil

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

    :param str name='Subsurface Release': a name for the spill.
    '''

    release = SubsurfaceRelease(distribution_type=distribution_type,
                               distribution=distribution,
                               release_time=release_time,
                               start_position=start_position,
                               num_elements=num_elements,
                               end_release_time=end_release_time)

    spill = _setup_spill(release=release,
                         water=water,
                         substance=substance,
                         amount=amount,
                         units=units,
                         name=name,
                         on=on,
                         windage_range=windage_range,
                         windage_persist=windage_persist
                         )

    return spill

# def continuous_release_spill(initial_elements,
#                              num_elements,
#                              start_position,
#                              release_time,
#                              end_position=None,
#                              end_release_time=None,
#                              water=None,
#                              substance=None,
#                              on=True,
#                              amount=None,
#                              units=None,
#                              windage_range=(.01, .04),
#                              windage_persist=900,
#                              name='Point or Line Release'):
#     '''
#     Helper function returns a Spill object containing a point or line release
#     '''
#     release = ContinuousRelease(initial_elements=initial_elements,
#                                 release_time=release_time,
#                                 start_position=start_position,
#                                 num_elements=num_elements,
#                                 end_position=end_position,
#                                 end_release_time=end_release_time)
#     retv = Spill(release=release,
#                  water=water,
#                  substance=substance,
#                  amount=amount,
#                  units=units,
#                  name=name,
#                  on=on)
#     retv.substance.windage_range = windage_range
#     retv.substance.windage_persist = windage_persist
#     return retv


def spatial_release_spill(start_positions,
                          release_time,
                          substance=None,
                          water=None,
                          on=True,
                          amount=0,
                          units='kg',
                          windage_range=None,
                          windage_persist=None,
                          name='spatial_release'):
    '''
    Helper function returns a Spill object containing a spatial release

    A spatial release is a spill that releases elements at known locations.
    '''
    release = Release(release_time=release_time,
                             custom_positions=start_positions,
                             name=name)
    spill = _setup_spill(release=release,
                         water=water,
                         substance=substance,
                         amount=amount,
                         units=units,
                         name=name,
                         on=on,
                         windage_range=windage_range,
                         windage_persist=windage_persist
                         )

    return spill
