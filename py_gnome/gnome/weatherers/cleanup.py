'''
oil removal from various cleanup options
add these as weatherers
'''
import copy

import numpy as np
from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable
from .core import WeathererSchema


class Skimmer(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def prepare_for_model_run(self, sc):
        if sc.spills:
            sc.weathering_data['skimmed'] = 0.0

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active:
            # take out 0.1% of the mass
            pct_per_le = (1 - 0.1/sc['mass_components'].shape[1])
            mass_remain = pct_per_le * sc['mass_components']
            sc.weathering_data['skimmed'] += \
                np.sum(sc['mass_components'][:, :] - mass_remain[:, :])
            sc['mass_components'] = mass_remain
            sc['mass'][:] = sc['mass_components'].sum(1)


class Burn(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def prepare_for_model_run(self, sc):
        if sc.spills:
            sc.weathering_data['burned'] = 0.0

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active:
            # take out 0.25% of the mass
            pct_per_le = (1 - 0.25/sc['mass_components'].shape[1])
            mass_remain = pct_per_le * sc['mass_components']
            sc.weathering_data['burned'] += \
                np.sum(sc['mass_components'][:, :] - mass_remain[:, :])
            sc['mass_components'] = mass_remain
            sc['mass'][:] = sc['mass_components'].sum(1)


class Dispersion(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = WeathererSchema

    def prepare_for_model_run(self, sc):
        if sc.spills:
            sc.weathering_data['dispersed'] = 0.0

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active:
            # take out 0.015% of the mass
            pct_per_le = (1 - 0.015/sc['mass_components'].shape[1])
            mass_remain = pct_per_le * sc['mass_components']
            sc.weathering_data['dispersed'] += \
                np.sum(sc['mass_components'][:, :] - mass_remain[:, :])
            sc['mass_components'] = mass_remain
            sc['mass'][:] = sc['mass_components'].sum(1)
