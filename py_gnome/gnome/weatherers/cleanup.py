'''
oil removal from various cleanup options
add these as weatherers
'''
from datetime import timedelta
import copy
import os

import numpy as np
from colander import (SchemaNode, Float, String, drop)

from gnome.basic_types import oil_status, fate as bt_fate
from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.inf_datetime import InfDateTime

from .core import WeathererSchema
from .. import _valid_units

import unit_conversion as uc


class SkimmerSchema(WeathererSchema):
    amount = SchemaNode(Float())
    units = SchemaNode(String())
    efficiency = SchemaNode(Float())


class CleanUpBase(Weatherer):
    '''
    Just need to add a few internal methods for Skimmer + Burn common code
    Currently defined as a base class.
    '''
    # todo: following is same as Spill code so rework to make it DRY
    valid_vol_units = _valid_units('Volume')
    valid_mass_units = _valid_units('Mass')

    def __init__(self, **kwargs):
        '''
        add 'frac_water' to array_types and pass **kwargs to base class
        __init__ using super
        '''
        super(CleanUpBase, self).__init__(**kwargs)
        self.array_types.update({'frac_water'})

    def _get_mass(self, substance, amount, units):
        '''
        return 'amount' in units of 'kg' for specified substance
        uses the density corresponding with API temperature
        '''
        if units in self.valid_mass_units:
            rm_mass = uc.convert('Mass', units, 'kg', amount)
        else:   # amount must be in volume units
            rm_vol = uc.convert('Volume', units, 'm^3', amount)
            rm_mass = substance.get_density() * rm_vol

        return rm_mass

    def _get_substance(self, sc):
        '''
        return a single substance - cleanup operations only know about the
        total amount removed. Unclear how to assign this to multiple substances
        For now, just log an error if more than one substance present
        '''
        substance = sc.get_substances(complete=False)
        if len(substance) > 1:
            msg = ('Found more than one type of Oil - not supported. '
                   'Results will be incorrect')
            self.logger.error(msg)
        return substance[0]

    def _update_LE_status_codes(self, sc, fate, substance, mass_to_remove):
        '''
        Need to mark LEs for skimming/burning. Mark LEs based on mass
        Mass to remove is the oil_water mixture so we need to find the
        oil_amount given the water_frac:

            volume = sc['mass']/API_density
            (1 - sc['frac_water']) * oil_water_vol = volume
            oil_water_vol = volume / (1 - sc['frac_water'])

        Now, do a cumsum of oil_water_mass and find where
            np.cumsum(oil_water_vol) >= vol_to_remove
        and change the status_codes of these LEs. Can just as easily multiple
        everything by API_density to get
            np.cumsum(oil_water_mass) >= mass_to_remove
            mass_to_remove = sc['mass'] / (1 - sc['frac_water'])
        This is why the input is 'mass_to_remove' instead of 'vol_to_remove'
        - less computation
        '''
        status = getattr(bt_fate, fate)
        arrays = {'fate_status', 'mass', 'frac_water'}
        data = sc.substancefatedata(substance, arrays, 'surface_weather')
        oil_water = data['mass'] / (1 - data['frac_water'])

        # (1 - frac_water) * mass_to_remove
        if mass_to_remove >= oil_water.sum():
            data['fate_status'][:] = status
            self.logger.debug(self._pid + 'marked ALL ({0}) LEs with mass: '
                              '{1}'.format(len(data['fate_status']),
                                           data['mass'].sum()))
        else:
            # sum up mass until threshold is reached, find index where
            # total_mass_removed is reached or exceeded
            ix = np.where(np.cumsum(oil_water) >= mass_to_remove)[0][0]
            # change status for elements upto and including 'ix'
            data['fate_status'][:ix + 1] = status

            self.logger.debug(self._pid + 'marked {0} LEs with mass: '
                              '{1}'.format(ix, data['mass'][:ix].sum()))

        sc.update_from_fatedataview(substance, 'surface_weather')

    def _set__timestep(self, time_step, model_time):
        '''
        For cleanup operations we may know the start time pretty precisely.
        Use this to set _timestep to less than time_step resolution. Mostly
        done for testing right now so if XXX amount is skimmed between
        active_start and active_stop, the rate * duration gives the correct
        amount. Object must be active before invoking this, else
        self._timestep will give invalid results
        '''
        if not self.active:
            return

        self._timestep = time_step
        dt = timedelta(seconds=time_step)

        if (model_time < self.active_start):
            self._timestep = \
                time_step - (self.active_start -
                             model_time).total_seconds()

        if (self.active_stop < model_time + dt):
            self._timestep = (self.active_stop -
                              model_time).total_seconds()

    def _avg_frac_oil(self, data):
        '''
        find weighted average of frac_water array, return (1 - avg_frac_water)
        since we want the average fraction of oil in this data
        '''
        avg_frac_water = ((data['mass'] * data['frac_water']).
                          sum()/data['mass'].sum())
        return (1 - avg_frac_water)


class Skimmer(CleanUpBase, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('amount', save=True, update=True),
               Field('units', save=True, update=True),
               Field('efficiency', save=True, update=True)]

    _schema = SkimmerSchema

    def __init__(self,
                 amount,
                 units,
                 efficiency,
                 active_start,
                 active_stop,
                 **kwargs):
        '''
        initialize Skimmer object - calls base class __init__ using super()
        active_start and active_stop time are required
        cleanup operations must have a valid datetime - cannot use -inf and inf
        active_start/active_stop is used to get the mass removal rate
        '''

        super(Skimmer, self).__init__(active_start=active_start,
                                      active_stop=active_stop,
                                      **kwargs)
        self._units = None
        self.amount = amount
        self.units = units
        self.efficiency = efficiency

        # get the rate as amount/sec, use this to compute amount at each step
        # set in prepare_for_model_run()
        self._rate = None

        # let prepare_for_model_step set timestep to use when active_start or
        # active_stop is between a timestep. Generally don't do subtimestep
        # resolution; however, in this case we want numbers to add up correctly
        self._timestep = 0.0

        if self.units is None:
            raise TypeError('Need valid mass or volume units for amount')

    def _validunits(self, value):
        'checks if units are either valid_vol_units or valid_mass_units'
        if value in self.valid_vol_units or value in self.valid_mass_units:
            return True
        return False

    @property
    def units(self):
        'return units for amount skimmed'
        return self._units

    @units.setter
    def units(self, value):
        if self._validunits(value):
            self._units = value
        else:
            msg = ('{0} are not valid volume or mass units.'
                   ' Not updated').format('value')
            self.logger.warn(msg)

    def prepare_for_model_run(self, sc):
        '''
        no need to call base class since no new array_types were added
        '''
        self._rate = self.amount/(self.active_stop -
                                  self.active_start).total_seconds()
        if self.on:
            sc.weathering_data['skimmed'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Do sub timestep resolution here so numbers add up correctly
        Mark LEs to be skimmed - do them in order right now. Assume all LEs
        that are released together will be skimmed together since they would
        be closer to each other in position.

        Assumes: there is more mass in water than amount of mass to be
        skimmed. The LEs marked for Skimming are marked only once -
        code checks to see if any LEs are marked for skimming and if
        none are found, it marks them.
        '''
        if not self.on:
            self._active = False
            return

        if (model_time + timedelta(seconds=time_step) > self.active_start and
            self.active_stop > model_time):
            self._active = True
        else:
            self._active = False
            return

        # only when it is active, update the status codes
        self._set__timestep(time_step, model_time)
        if (sc['fate_status'] == bt_fate.skim).sum() == 0:
            substance = self._get_substance(sc)
            total_mass_removed = (self._get_mass(substance, self.amount,
                                                 self.units) *
                                  self.efficiency)
            self._update_LE_status_codes(sc, 'skim',
                                         substance, total_mass_removed)

    def _mass_to_remove(self, substance):
        '''
        use density at 15C, ie corresponding with API to do mass/volume
        conversion
        '''
        amount = self._rate * self._timestep
        rm_mass = self._get_mass(substance, amount, self.units)

        return rm_mass

    def weather_elements(self, sc, time_step, model_time):
        '''
        Assumes there is only ever 1 substance being modeled!
        remove mass equally from LEs marked to be skimmed
        '''
        if not self.active:
            return

        if len(sc) == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='skim'):
            if len(data['mass']) is 0:
                continue

            rm_amount = \
                self._rate * self._avg_frac_oil(data) * self._timestep
            rm_mass = self._get_mass(substance,
                                     rm_amount,
                                     self.units) * self.efficiency

            rm_mass_frac = min(rm_mass / data['mass'].sum(), 1.0)

            # if elements are also evaporating following could be true
            # need to include weathering for skimmed particles, then test and
            # add if following is required.
            # if rm_mass_frac > 1:
            #     rm_mass_frac = 1.0

            data['mass_components'] = \
                (1 - rm_mass_frac) * data['mass_components']
            data['mass'] = data['mass_components'].sum(1)

            sc.weathering_data['skimmed'] += rm_mass
            self.logger.debug(self._pid + 'amount skimmed for {0}: {1}'.
                              format(substance.name, rm_mass))

        sc.update_from_fatedataview(fate='skim')


class BurnSchema(WeathererSchema):
    area = SchemaNode(Float())
    thickness = SchemaNode(Float())
    _oil_thickness = SchemaNode(Float(), missing=drop)
    area_units = SchemaNode(String())
    thickness_units = SchemaNode(String())


class Burn(CleanUpBase, Serializable):
    _schema = BurnSchema

    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('area', save=True, update=True),
               Field('thickness', save=True, update=True),
               Field('area_units', save=True, update=True),
               Field('thickness_units', save=True, update=True),
               Field('_oil_thickness', save=True)]

    valid_area_units = _valid_units('Area')
    valid_length_units = _valid_units('Length')

    def __init__(self,
                 area,
                 thickness,
                 active_start,
                 area_units='m^2',
                 thickness_units='m',
                 **kwargs):
        '''
        Set the area of boomed oil to be burned. Assumes the area and thickness
        are in SI units so 'area' is in 'm^2' and 'thickness' in 'm'. There
        is no unit conversion.

        Set intial thickness of this oil as specified by user.
        cleanup operations must have a valid datetime - cannot use -inf
        '''
        if 'active_stop' in kwargs:
            # user cannot set 'active_stop'
            kwargs.pop('active_stop')

        super(Burn, self).__init__(active_start=active_start,
                                   **kwargs)

        # store in SI units - internally, object uses these
        self._si_thickness = None  # in SI units
        self._si_area = None       # in SI units

        # validate user units before setting _area_units/_thickness_units
        # if any of the following are updated via setters, then update
        # _si_thickness/_si_area
        self._area = area
        self._thickness = thickness

        # initialize user units to valid units - setters following this will
        # initialize area_units and thickness_units per input values
        self._area_units = 'm^2'
        self._thickness_units = 'm'

        # setters will validate the units
        self.area_units = area_units
        self.thickness_units = thickness_units

        # thickness of burned/boomed oil which is updated at each timestep
        # this will be set once we figure out how much oil will be burned
        # in prepare_for_model_step()
        self._oil_thickness = None
        self._min_thickness = 0.002     # stop burn threshold

        # burn rate is defined as a volume rate in m^3/sec
        # where rate = 0.000058 * self.area * (1 - frac_water_content)
        # However, the area will cancel out when we compute the burn time:
        # burn_time = area/burn_rate * (thickness - 0.002)
        self._burn_rate_constant = 0.000058
        self._burn_duration = None

    def _update_si_area(self):
        '''
        update internal _area variable. Called if user sets the 'area' or
        the 'area_units'
        '''
        if self.area_units != 'm^2':
            value = uc.Convert('Area', self.area_units, 'm^2', self.area)
        else:
            value = self.area

        self._si_area = value

    def _update_si_thickness(self):
        '''
        update internal _thickness variable. Called if user sets 'thickness'
        or the 'thickness_units'
        '''
        if self.thickness_units != 'm':
            value = uc.Convert('Length', self.thickness_units, 'm',
                               self.thickness)
        else:
            value = self.thickness

        self._si_thickness = value

    @property
    def area(self):
        '''
        return area in user specified area_units
        '''
        return self._area

    @area.setter
    def area(self, value):
        self._area = value
        self._update_si_area()

    @property
    def thickness(self):
        '''
        return thickness in user specified area_units
        '''
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        self._update_si_thickness()

    @property
    def area_units(self):
        return self._area_units

    @area_units.setter
    def area_units(self, value):
        if value not in self.valid_area_units:
            self.logger.error(self._pid + "ignoring invalid area units: {0}".
                              format(value))
        else:
            self._area_units = value
            self._update_si_area()

    @property
    def thickness_units(self):
        return self._thickness_units

    @thickness_units.setter
    def thickness_units(self, value):
        if value not in self.valid_length_units:
            self.logger.error(self._pid + "ignoring invalid thickness units: "
                              "{0}".format(value))
        else:
            self._thickness_units = value
            self._update_si_thickness()

    def prepare_for_model_run(self, sc):
        '''
        resets internal _oil_thickness variable to initial thickness specified
        by user. Also resets _burn_duration to None
        '''
        # reset current thickness to initial thickness whenever model is rerun
        self._oil_thickness = None
        self._burn_duration = None
        if self.on:
            sc.weathering_data['burned'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Do sub timestep resolution here so numbers add up correctly
        Mark LEs to be burned - do them in order right now. Assume all LEs
        that are released together will be burned together since they would
        be closer to each other in position.

        Assumes: there is more mass in water than amount of mass to be
        skimmed. The LEs marked for Burning are marked only once -
        code checks to see if any LEs are marked and if
        none are found, it marks them.
        '''
        if not self.on:
            self._active = False
            return

        if model_time + timedelta(seconds=time_step) > self.active_start:
            self._active = True
        else:
            self._active = False
            return

        # only when it is active, update the status codes
        self._set__timestep(time_step, model_time)
        if (sc['fate_status'] == bt_fate.burn).sum() == 0:
            substance = self._get_substance(sc)
            mass_to_remove = self._get_mass(substance,
                                            self._si_area * self._si_thickness,
                                            'm^3')
            self._update_LE_status_codes(sc, 'burn',
                                         substance, mass_to_remove)
            # now also set up self._oil_thickness
            self._oil_thickness = \
                (sc['mass'][sc['fate_status'] == bt_fate.burn].sum() /
                 (substance.get_density() * self._si_area))

        # check _oil_thickness after property is set in code block above
        if self._oil_thickness <= self._min_thickness:
            self._active = False
            return

    def weather_elements(self, sc, time_step, model_time):
        '''
        Given burn rate and fixed area, we have the rate of change of thickness
        Do a weighted average of frac_water array for the elements marked as
        burn. This is the avg_frac_water to use when computing the burn rate.

            burn_rate_th := burn_rate/area = 0.000058 * (1 - avg_frac_water)
            burn_time = ((self._oil_thickness - 0.002) / burn_rate_th
            _timestep = min(burn_time, time_step)
            remove_volume = burn_rate_th * _timestep * area

        The data is masked to operate only on LEs marked for burning:
            mask = data['status_codes'] == burn

        Convert remove_volume to remove_mass. Find fraction of mass to remove:
            rm_mass_frac = remove_mass/data['mass'][mask].sum()

        Then update the 'mass_components' array per:
            data['mass_components'][mask] = \
                (1 - rm_mass_frac) * data['mass_components'][mask]

        update the 'mass' array and the amount burned in weathering_data dict
        Also update the _burn_duration attribute.
        '''
        if not self.active or len(sc) == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types,
                                                    fate='burn'):
            if len(data['mass']) is 0:
                continue

            # keep updating thickness
            burn_th_rate = \
                self._burn_rate_constant * self._avg_frac_oil(data)
            burn_time = ((self._oil_thickness - self._min_thickness) /
                         burn_th_rate)

            self._timestep = min(burn_time, self._timestep)
            if self._timestep > 0:
                th_burned = burn_th_rate * self._timestep
                rm_mass = self._get_mass(substance, th_burned * self._si_area,
                                         'm^3')
                rm_mass_frac = rm_mass / data['mass'].sum()
                data['mass_components'] = \
                    (1 - rm_mass_frac) * data['mass_components']
                data['mass'] = data['mass_components'].sum(1)

                # new thickness
                self._oil_thickness -= th_burned

                sc.weathering_data['burned'] += rm_mass
                self.logger.debug(self._pid + 'amount burned for {0}: {1}'.
                                  format(substance.name, rm_mass))

                # update burn duration at each timestep
                self._burn_duration = \
                    (model_time + timedelta(seconds=self._timestep) -
                     self.active_start).total_seconds()

        sc.update_from_fatedataview(fate='burn')


class Dispersion(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def prepare_for_model_run(self, sc):
        if sc.spills:
            sc.weathering_data['dispersed'] = 0.0

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active and len(sc) > 0:
            for substance, data in sc.itersubstancedata(self.array_types):
                if len(data['mass']) is 0:
                    continue

                # take out 0.25% of the mass
                pct_per_le = (1 - 0.015/data['mass_components'].shape[1])
                mass_remain = pct_per_le * data['mass_components']
                sc.weathering_data['dispersed'] += \
                    np.sum(data['mass_components'] - mass_remain[:, :])
                data['mass_components'] = mass_remain
                data['mass'] = data['mass_components'].sum(1)

            sc.update_from_fatedataview()
