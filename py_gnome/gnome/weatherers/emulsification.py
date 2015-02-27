'''
model emulsification process
'''
from __future__ import division

import copy

import numpy as np

import gnome    # required by new_from_dict

from gnome.array_types import (frac_lost,  # due to evaporation and dissolution
                               age,
                               bulltime,
                               interfacial_area,
                               frac_water)

from gnome.utilities.serializable import Serializable, Field
from gnome import constants
from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.cy_gnome.cy_weatherers import emulsify_oil


class Emulsification(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('waves', save=True, update=True, save_reference=True)]
    _schema = WeathererSchema

    def __init__(self,
                 waves=None,
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param waves: waves object for obtaining emulsification wind speed at specified time
        :type waves: get_emulsifiation_wind(model_time)
        '''
        self.waves = waves

        super(Emulsification, self).__init__(**kwargs)
        self.array_types.update({'age', 'bulltime', 'frac_water',
                                 'interfacial_area', 'frac_lost'})

    def prepare_for_model_run(self, sc):
        '''
        add water_content key to weathering_data
        Assumes all spills have the same type of oil
        '''
        # create 'water_content' key if it doesn't exist
        # let's only define this the first time
        if self.on:
            sc.weathering_data['water_content'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Set/update arrays used by emulsification module for this timestep:

        '''

        # do we need this?
        super(Emulsification, self).prepare_for_model_step(sc,
                                                           time_step,
                                                           model_time)
        if not self.active:
            return

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'water_content' in sc.weathering_data
        '''

        if not self.active:
            return
        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            k_emul = self._water_uptake_coeff(model_time, substance)

            # bulltime is not in database, but could be set by user
            #emul_time = substance.get_bulltime()
            emul_time = substance.bulltime

            # get from database bullwinkle (could be overridden by user)
            #emul_constant = substance.get('bullwinkle_fraction')
            emul_constant = substance.bullwinkle

            # max water content fraction - get from database
            Y_max = substance.get('emulsion_water_fraction_max')

            # doesn't emulsify, avoid the nans
            if Y_max <= 0:
                continue
            S_max = (6. / constants.drop_min) * (Y_max / (1.0 - Y_max))

            emulsify_oil(time_step,
                         data['frac_water'],
                         data['interfacial_area'],
                         data['frac_lost'],
                         data['age'],
                         data['bulltime'],
                         k_emul,
                         emul_time,
                         emul_constant,
                         S_max,
                         Y_max,
                         constants.drop_max)

            #sc.weathering_data['water_content'] += \
                #np.sum(data['frac_water'][:]) / sc.num_released
            # just average the water fraction each time - it is not per time
            # step value but at a certain time value
            sc.weathering_data['water_content'] = \
                np.sum(data['frac_water'][:]) / sc.num_released
            self.logger.info('Amount water_content: {0}'.
                             format(sc.weathering_data['water_content']))

        sc.update_from_substancedata(self.array_types)

    def serialize(self, json_='webapi'):
        """
        Since 'wind'/'waves' property is saved as references in save file
        need to add appropriate node to WindMover schema for 'webapi'
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        serial = schema.serialize(toserial)

        if json_ == 'webapi':
            if self.waves is not None:
                serial['waves'] = self.waves.serialize(json_)
#             if self.wind is not None:
#                 serial['wind'] = self.wind.serialize(json_)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for waves object
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()

            dict_ = schema.deserialize(json_)
#             if 'wind' in json_:
#                 obj = json_['wind']['obj_type']
#                 dict_['wind'] = (eval(obj).deserialize(json_['wind']))
            if 'waves' in json_:
                obj = json_['waves']['obj_type']
                dict_['waves'] = (eval(obj).deserialize(json_['waves']))
            return dict_

        else:
            return json_

    def _water_uptake_coeff(self, model_time, substance):
        '''
        Use higher of wind or pseudo wind corresponding to wave height

            if (H0 > 0) HU = 2.0286 * sqrt(g * H0)
            if (HU < 4.429) HU = pow(HU / .71, .813)
            if (U < HU) U = HU
            k_emul = 6.0 * K0Y * U * U / d_max

        '''

        ## higher of real or psuedo wind
        wind_speed = self.waves.get_emulsifiation_wind(model_time)

        # water uptake rate constant - get this from database
        K0Y = substance.get('k0y')

        k_emul = 6.0 * K0Y * wind_speed * wind_speed / constants.drop_max

        return k_emul
