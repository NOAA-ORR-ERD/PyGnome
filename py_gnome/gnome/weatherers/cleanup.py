'''
oil removal from various cleanup options
add these as weatherers
'''
from datetime import timedelta
import copy
import os

import numpy as np
from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable
from .core import WeathererSchema
from .. import _valid_units

from hazpy import unit_conversion as uc


class Skimmer(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    # todo: following is same as Spill code so rework to make it DRY
    valid_vol_units = _valid_units('Volume')
    valid_mass_units = _valid_units('Mass')

    def __init__(self,
                 amount,
                 units,
                 efficiency,
                 **kwargs):
        '''
        initialize Skimmer object - calls base class __init__ using super()
        '''
        self._units = None
        self.amount = amount
        self.units = units
        self.efficiency = efficiency
        self.thickness_lim = 0.002

        super(Skimmer, self).__init__(**kwargs)

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
        if sc.spills:
            sc.weathering_data['skimmed'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        '''
        have to do sub timestep resolution here so numbers add up correctly
        '''
        if not self.on:
            self._active = False
            return

        self._timestep = time_step
        dt = timedelta(seconds=time_step)

        if (model_time_datetime + dt > self.active_start and
            self.active_stop > model_time_datetime):
            self._active = True

            if (model_time_datetime < self.active_start):
                self._timestep = \
                    time_step - (self.active_start -
                                 model_time_datetime).total_seconds()

            if (self.active_stop < model_time_datetime + dt):
                self._timestep = (self.active_stop -
                                  model_time_datetime).total_seconds()
        else:
            self._active = False

    def _amount_removed(self, substance):
        '''
        use density at 15C, ie corresponding with API to do mass/volume
        conversion
        '''
        amount = self._rate * self._timestep
        if self.units in self.valid_mass_units:
            rm_mass = uc.convert('Mass', self.units, 'kg', amount)
        else:   # amount must be in volume units
            rm_vol = uc.convert('Volume', self.units, 'm^3', amount)
            rm_mass = substance.get_density() * rm_vol

        self.logger.info('{0} - Amount skimmed before efficiency: {1}'.
                         format(os.getpid(), rm_mass))
        return rm_mass

    def weather_elements(self, sc, time_step, model_time):
        '''
        Assumes there is only ever 1 substance being modeled!
        remove mass equally from all elements and all components
        with thickness > 2mm?
        '''
        if not self.active:
            return

        if len(sc) == 0:
            return

        for substance, data in sc.itersubstancedata(self._arrays):
            rm_mass = (self._amount_removed(substance) * self.efficiency)
            frac_mass_left = 1 - (rm_mass / data['mass'].sum())
            if frac_mass_left < 0.:
                self.logger.info('{0} - removing more mass {1} than '
                                 'available {2}, remove only remaining mass'.
                                 format(os.getpid(), rm_mass,
                                        data['mass'].sum()))
                frac_mass_left = 0.

            self.logger.info('{0} - frac_mass_left: {1}'.
                             format(os.getpid(), frac_mass_left))
            data['mass_components'][:, :] *= frac_mass_left
            data['mass'][:] = data['mass_components'][:, :].sum(1)

            sc.weathering_data['skimmed'] += rm_mass

        sc.update_from_substancedata(self._arrays)


class Burn(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def prepare_for_model_run(self, sc):
        if sc.spills:
            sc.weathering_data['burned'] = 0.0

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active and len(sc) > 0:
            for substance, data in sc.itersubstancedata(self._arrays):
                # take out 0.25% of the mass
                pct_per_le = (1 - 0.25/data['mass_components'].shape[1])
                mass_remain = pct_per_le * data['mass_components']
                sc.weathering_data['burned'] += \
                    np.sum(data['mass_components'][:, :] - mass_remain[:, :])
                data['mass_components'] = mass_remain
                data['mass'][:] = data['mass_components'].sum(1)

            sc.update_from_substancedata(self._arrays)


class Dispersion(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def prepare_for_model_run(self, sc):
        if sc.spills:
            sc.weathering_data['dispersed'] = 0.0

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active and len(sc) > 0:
            for substance, data in sc.itersubstancedata(self._arrays):
                # take out 0.25% of the mass
                pct_per_le = (1 - 0.015/data['mass_components'].shape[1])
                mass_remain = pct_per_le * data['mass_components']
                sc.weathering_data['dispersed'] += \
                    np.sum(data['mass_components'][:, :] - mass_remain[:, :])
                data['mass_components'] = mass_remain
                data['mass'][:] = data['mass_components'].sum(1)

            sc.update_from_substancedata(self._arrays)