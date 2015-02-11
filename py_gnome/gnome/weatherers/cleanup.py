'''
oil removal from various cleanup options
add these as weatherers
'''
from datetime import timedelta
import copy
import os

import numpy as np
from colander import (SchemaNode, Float, String, drop)

from gnome.basic_types import oil_status
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

    def _update_LE_status_codes(self, sc, status, substance, mass_to_remove):
        ''' Need to mark LEs for skimming/burning. Mark LEs based on mass '''
        data = sc.substancedata(substance, ['status_codes', 'mass'])

        if mass_to_remove >= data['mass'].sum():
            data['status_codes'][:] = status
        else:
            # sum up mass until threshold is reached, find index where
            # total_mass_removed is reached or exceeded
            ix = np.where(np.cumsum(data['mass']) >=
                          mass_to_remove)[0][0]
            # change status for elements upto and including 'ix'
            data['status_codes'][:ix + 1] = status

        sc.update_from_substancedata(self.array_types, substance)

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
        '''

        super(Skimmer, self).__init__(active_start=active_start,
                                      active_stop=active_stop,
                                      **kwargs)
        self._units = None
        self.amount = amount
        self.units = units
        self.efficiency = efficiency

        # get the rate as amount/sec, use this to compute amount at each step
        self._rate = self.amount/(self.active_stop -
                                  self.active_start).total_seconds()
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
        if sc.spills:
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
        if (sc['status_codes'] == oil_status.skim).sum() == 0:
            substance = self._get_substance(sc)
            total_mass_removed = (self._get_mass(substance, self.amount,
                                                 self.units) *
                                  self.efficiency)
            self._update_LE_status_codes(sc, oil_status.skim,
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

        for substance, data in sc.itersubstancedata(self.array_types):
            rm_mass = (self._mass_to_remove(substance) * self.efficiency)

            self.logger.info('{0} - Amount skimmed: {1}'.
                             format(os.getpid(), rm_mass))

            # following should work even if all elements go to zero so
            # ~c_to_zero is all False since rm_mass_per_c is a scalar
            mask = data['status_codes'] == oil_status.skim
            rm_mass_frac = rm_mass / data['mass'][mask].sum()

            # if elements are also evaporating following could be true
            # need to include weathering for skimmed particles, then test and
            # add if following is required.
            # if rm_mass_frac > 1:
            #     rm_mass_frac = 1.0

            data['mass_components'][mask, :] = \
                (1 - rm_mass_frac) * data['mass_components'][mask, :]
            data['mass'][mask] = data['mass_components'][mask, :].sum(1)

            sc.weathering_data['skimmed'] += rm_mass

        sc.update_from_substancedata(self.array_types)


class BurnSchema(WeathererSchema):
    area = SchemaNode(Float())
    thickness = SchemaNode(Float())
    _curr_thickness = SchemaNode(Float(), missing=drop)


class Burn(CleanUpBase, Serializable):
    _schema = BurnSchema

    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('area', save=True, update=True),
               Field('thickness', save=True, update=True),
               Field('_curr_thickness', save=True)]

    def __init__(self,
                 area,
                 thickness,
                 active_start,
                 **kwargs):
        '''
        Set the area of boomed oil to be burned. Assumes the area and thickness
        are in SI units so 'area' is in 'm^2' and 'thickness' in 'm'. There
        is no unit conversion.

        Set intial thickness of this oil as specified by user.
        '''
        if 'active_stop' in kwargs:
            # user cannot set 'active_stop'
            kwargs.pop('active_stop')

        super(Burn, self).__init__(active_start=active_start,
                                   **kwargs)
        self.area = area
        self.thickness = thickness

        self._init_volume = self.area * self.thickness
        self._min_thickness = 0.002

        # thickness of burned/boomed oil which is updated at each timestep
        self._curr_thickness = thickness

        # burn rate is defined as a volume rate in m^3/sec
        # where rate = 0.000058 * self.area * (1 - frac_water_content)
        # However, the area will cancel out when we compute the burn time:
        # burn_time = area/burn_rate * (thickness - 0.002)
        self._burn_rate_constant = 0.000058
        self._burn_duration = None

        self.array_types.update({'frac_water'})

    def prepare_for_model_run(self, sc):
        '''
        resets internal _curr_thickness variable to initial thickness specified
        by user. Also resets _burn_duration to None
        '''
        # reset current thickness to initial thickness whenever model is rerun
        self._curr_thickness = self.thickness
        self._burn_duration = None
        if sc.spills:
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

        if self._curr_thickness <= self._min_thickness:
            self._active = False
            return

        if model_time + timedelta(seconds=time_step) > self.active_start:
            self._active = True
        else:
            self._active = False
            return

        # only when it is active, update the status codes
        self._set__timestep(time_step, model_time)
        if (sc['status_codes'] == oil_status.burn).sum() == 0:
            substance = self._get_substance(sc)
            mass_removed = self._get_mass(substance, self._init_volume,
                                          'm^3')
            self._update_LE_status_codes(sc, oil_status.burn,
                                         substance, mass_removed)

    def weather_elements(self, sc, time_step, model_time):
        '''
        Given burn rate and fixed area, we have the rate of change of thickness
        Do a weighted average of frac_water array for the elements marked as
        burn. This is the avg_frac_water to use when computing the burn rate.

            burn_rate_th := burn_rate/area = 0.000058 * (1 - avg_frac_water)
            burn_time = ((self._curr_thickness - 0.002) / burn_rate_th
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

        for substance, data in sc.itersubstancedata(self.array_types):
            # keep updating thickness
            mask = data['status_codes'] == oil_status.burn
            avg_frac_water = ((data['mass'][mask] * data['frac_water'][mask]).
                              sum()/data['mass'][mask].sum())
            burn_th_rate = self._burn_rate_constant * (1 - avg_frac_water)
            burn_time = ((self._curr_thickness - self._min_thickness) /
                         burn_th_rate)

            self._timestep = min(burn_time, self._timestep)
            if self._timestep > 0:
                th_burned = burn_th_rate * self._timestep
                rm_mass = self._get_mass(substance, th_burned * self.area,
                                         'm^3')
                rm_mass_frac = rm_mass / data['mass'][mask].sum()
                data['mass_components'][mask, :] = \
                    (1 - rm_mass_frac) * data['mass_components'][mask, :]
                data['mass'][mask] = data['mass_components'][mask, :].sum(1)

                # new thickness
                self._curr_thickness -= th_burned

                sc.weathering_data['burned'] += rm_mass

                # update burn duration at each timestep
                self._burn_duration = \
                    (model_time + timedelta(seconds=self._timestep) -
                     self.active_start).total_seconds()

        sc.update_from_substancedata(self.array_types)


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
                mask = data['status_codes'] == oil_status.in_water
                # take out 0.25% of the mass
                pct_per_le = (1 - 0.015/data['mass_components'].shape[1])
                mass_remain = pct_per_le * data['mass_components'][mask, :]
                sc.weathering_data['dispersed'] += \
                    np.sum(data['mass_components'][mask, :] - mass_remain[:, :])
                data['mass_components'][mask, :] = mass_remain
                data['mass'][mask] = data['mass_components'][mask, :].sum(1)

            sc.update_from_substancedata(self.array_types)