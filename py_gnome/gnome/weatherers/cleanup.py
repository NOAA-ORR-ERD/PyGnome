'''
oil removal from various cleanup options
add these as weatherers
'''
import copy

import numpy as np
from gnome.weatherers import Weatherer
from gnome.utilities.serializable import Serializable
from gnome.persist.base_schema import ObjType


class Skimmer(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _schema = ObjType

    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active:
            # take out 0.1% of the mass
            pct_per_le = (1 - 0.1/sc['mass_components'].shape[1])
            mass_remain = pct_per_le * sc['mass_components']
            sc.weathering_data['skimmed'] = \
                np.sum(sc['mass_components'][:, :] - mass_remain[:, :])
            sc['mass_components'] = mass_remain
            sc['mass'][:] = sc['mass_components'].sum(1)


class Burn(Weatherer, Serializable):
    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active:
            # take out 0.25% of the mass
            pct_per_le = (1 - 0.25/sc['mass_components'].shape[1])
            mass_remain = pct_per_le * sc['mass_components']
            sc.weathering_data['burned'] = \
                np.sum(sc['mass_components'][:, :] - mass_remain[:, :])
            sc['mass_components'] = mass_remain
            sc['mass'][:] = sc['mass_components'].sum(1)


class Disperse(Weatherer, Serializable):
    def weather_elements(self, sc, time_step, model_time):
        'for now just take away 0.1% at every step'
        if self.active:
            # take out 0.15% of the mass
            pct_per_le = (1 - 0.15/sc['mass_components'].shape[1])
            mass_remain = pct_per_le * sc['mass_components']
            sc.weathering_data['dispersed'] = \
                np.sum(sc['mass_components'][:, :] - mass_remain[:, :])
            sc['mass_components'] = mass_remain
            sc['mass'][:] = sc['mass_components'].sum(1)
