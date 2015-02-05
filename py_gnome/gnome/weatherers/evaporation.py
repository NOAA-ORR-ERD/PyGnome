'''
model evaporation process
'''
import copy
import os

import numpy as np

from gnome.basic_types import oil_status
from gnome.utilities.serializable import Serializable, Field
from gnome import constants
from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.environment import (WindSchema,
                               WaterSchema)


class Evaporation(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('water', save=True, update=True, save_reference=True),
               Field('wind', save=True, update=True, save_reference=True)]
    _schema = WeathererSchema

    def __init__(self,
                 water=None,
                 wind=None,
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param wind: wind object for obtaining speed at specified time
        :type wind: Wind API, specifically must have get_value(time) method
        '''
        self.water = water
        self.wind = wind

        super(Evaporation, self).__init__(**kwargs)
        self.array_types.update({'area', 'mol', 'evap_decay_constant',
                                 'frac_water', 'frac_lost', 'init_mass'})

    def prepare_for_model_run(self, sc):
        '''
        add evaporated key to weathering_data
        for now also add 'density' key here
        Assumes all spills have the same type of oil
        '''
        # create 'evaporated' key if it doesn't exist
        # let's only define this the first time
        if self.active:
            sc.weathering_data['evaporated'] = 0.0

    def _mass_transport_coeff(self, model_time):
        '''
        Is wind a function of only model_time? How about time_step?
        at present yes since wind only contains timeseries data

            K = c * U ** 0.78 if U <= 10 m/s
            K = 0.06 * c * U ** 2 if U > 10 m/s

        If K is expressed in m/sec, then Buchanan and Hurford set c = 0.0025
        U is wind_speed 10m above the surface
        '''
        wind_speed = self.wind.get_value(model_time)[0]
        c_evap = 0.0025     # if wind_speed in m/s
        if wind_speed <= 10.0:
            return c_evap * wind_speed ** 0.78
        else:
            return 0.06 * c_evap * wind_speed ** 2

    def _set_evap_decay_constant(self, model_time, data, substance):
        # used to compute the evaporation decay constant
        K = self._mass_transport_coeff(model_time)
        water_temp = self.water.get('temperature', 'K')

        f_diff = 1.0
        if 'frac_water' in data:
            # frac_water content in emulsion will be a per element but is
            # currently not being set by anything. Fix once we initialize
            # and properly set frac_water
            f_diff = (1.0 - data['frac_water'])

        vp = substance.vapor_pressure(water_temp)

        # A more verbose description of the equation:
        # -------------------------------------------
        # d_numer = -(data['area'] * f_diff).reshape(-1, 1) * K * vp
        # d_denom = (constants['gas_constant'] * water_temp *
        #            data['mol']).reshape(-1, 1)
        # d_denom = np.repeat(d_denom, d_numer.shape[1], axis=1)
        # data['evap_decay_constant'][:] = d_numer/d_denom
        if len(data['evap_decay_constant']) > 0:
            # set/update mols -- happens the same way for new or old particles
            mw = substance.molecular_weight
            data['mol'][:] = \
                np.sum(data['mass_components'][:, :len(mw)]/mw, 1)

            data['evap_decay_constant'][:, :len(vp)] = \
                -(((data['area'] * f_diff).reshape(-1, 1) * K * vp) /
                  np.repeat((constants.gas_constant * water_temp *
                             data['mol']).reshape(-1, 1), len(vp), axis=1))

            # only elements 'in_water' experience evaporation
            inwater = data['status_codes'] == oil_status.in_water
            data['evap_decay_constant'][~inwater, :len(vp)] = 0

            self.logger.info('{0} - Max decay: {1}, Min decay: {2}'.
                             format(os.getpid(),
                                    np.max(data['evap_decay_constant']),
                                    np.min(data['evap_decay_constant'])))
        if np.any(data['evap_decay_constant'] > 0.0):
            raise ValueError("Error in Evaporation routine. One of the"
                             " exponential decay constant is positive")

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'evaporation' in sc.weathering_data
        - currently also sets 'density' in sc.weathering_data but may update
          this as we add more weatherers and perhaps density gets set elsewhere
        '''
        if not self.active:
            return
        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            # set evap_decay_constant array
            self._set_evap_decay_constant(model_time, data, substance)
            mass_remain = \
                self._exp_decay(data['mass_components'],
                                data['evap_decay_constant'],
                                time_step)

            sc.weathering_data['evaporated'] += \
                np.sum(data['mass_components'][:, :] - mass_remain[:, :])
            data['mass_components'][:] = mass_remain
            data['mass'][:] = data['mass_components'].sum(1)
            self.logger.info('{0} - Amount Evaporated for {1}: {2}'.
                             format(os.getpid(),
                                    substance.name,
                                    sc.weathering_data['evaporated']))

            # add frac_lost
            data['frac_lost'][:] = 1 - data['mass']/data['init_mass']
        sc.update_from_substancedata(self.array_types)

    def serialize(self, json_='webapi'):
        """
        Since 'wind'/'water' property is saved as references in save file
        need to add appropriate node to WindMover schema for 'webapi'
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        if json_ == 'webapi':
            if self.wind:
                # add wind schema
                schema.add(WindSchema(name='wind'))
            if self.water:
                schema.add(WaterSchema(name='water'))

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema()
        if 'wind' in json_:
            schema.add(WindSchema(name='wind'))

        if 'water' in json_:
            schema.add(WaterSchema(name='water'))
        _to_dict = schema.deserialize(json_)

        return _to_dict