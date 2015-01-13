'''
model emulsification process
'''
from __future__ import division

import copy

import numpy as np

import gnome    # required by new_from_dict

from gnome.array_types import (frac_lost,	# due to evaporation and dissolution
                               age,
                               droplet_diameter,
                               bulltime,
                               interfacial_area,
                               frac_water)
                               
from gnome.utilities.serializable import Serializable, Field

from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.environment import (constants,
                               WindSchema,
                               WaterSchema,
                               WavesSchema)


from gnome.cy_gnome.cy_weatherers import emulsify_oil

class Emulsification(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('wind', save=True, update=True, save_reference=True),
               Field('waves', save=True, update=True, save_reference=True)]
    _schema = WeathererSchema

    def __init__(self,
                 waves=None,
                 wind=None,
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param wind: wind object for obtaining speed at specified time
        :type wind: Wind API, specifically must have get_value(time) method
        '''
        self.waves = waves
        self.wind = wind

        super(Emulsification, self).__init__(**kwargs)
        self.array_types.update({'age': age,
                                 'droplet_diameter': droplet_diameter,
                                 'bulltime': bulltime,
                                 'frac_water': frac_water,
                                 'interfacial_area': interfacial_area,
                                 'frac_lost': frac_lost,
                                 })

    def prepare_for_model_run(self, sc):
        '''
        add emulsified key to weathering_data
        Assumes all spills have the same type of oil
        '''
        # create 'emulsified' key if it doesn't exist - are we tracking this ?
        # let's only define this the first time
        if self.active:
            sc.weathering_data['emulsified'] = 0.0


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
        - sets 'evaporation' in sc.weathering_data
        - currently also sets 'density' in sc.weathering_data but may update
          this as we add more weatherers and perhaps density gets set elsewhere
        '''

        if not self.active:
            return
        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self._arrays):
            k_emul = self._water_uptake_coeff(model_time, substance)
            emul_time = substance.get_bulltime()	#bulltime is not in database, but could be set by user
            emul_constant = substance.get('bullwinkle_fraction') # get from database bullwinkle (could be overridden by user)
            Y_max = substance.get('emulsion_water_fraction_max') # max water content fraction - get from database 
            if Y_max <= 0:	# doesn't emulsify, avoid the nans
                continue
            S_max = (6. / constants['drop_min']) * (Y_max / (1.0 - Y_max))
            #emulsify_oil(time_step,sc['frac_water'],sc['interfacial_area'],sc['frac_lost'],sc['droplet_diameter'],sc['age'],sc['bulltime'],k_emul,emul_time, emul_constant, Y_max,S_max,constants['drop_max'])
            emulsify_oil(time_step,
                         data['frac_water'],
                         data['interfacial_area'],
                         data['frac_lost'],
                         data['droplet_diameter'],
                         data['age'],
                         data['bulltime'],
                         k_emul,
                         emul_time,
                         emul_constant,
                         S_max,
                         Y_max,constants['drop_max'])

            #sc.weathering_data['emulsified'] += \
                #np.sum(data['frac_water'][:]) / sc.num_released
            # just average the water fraction each time - it is not per time step value but at a certain time value
            sc.weathering_data['emulsified'] = \
                np.sum(data['frac_water'][:]) / sc.num_released
            self.logger.info('Amount Emulsified: {0}'.
                             format(sc.weathering_data['emulsified']))

        sc.update_from_substancedata(self._arrays)
                

    def serialize(self, json_='webapi'):
        """
        Since 'wind'/'waves' property is saved as references in save file
        need to add appropriate node to WindMover schema for 'webapi'
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        serial = schema.serialize(toserial)
        
        if json_ == 'webapi':
            if self.waves:
                serial['waves'] = self.waves.serialize(json_)
            if self.wind:
                serial['wind'] = self.wind.serialize(json_)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()

            dict_ = schema.deserialize(json_)
            if 'wind' in json_:
                #obj = json_['wind'].pop('obj_type')
                obj = json_['wind']['obj_type']
                dict_['wind'] = (eval(obj).deserialize(json_['wind']))
            if 'waves' in json_:
                #obj = json_['waves'].pop('obj_type')
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
        wind_speed = self.wind.get_value(model_time)[0] # what happens if no wind or wave height data? returns zero?
        #wave_height = self.water.get('wave_height','m'] # should be time dependent
        # wave_height = self.waves.get_value(model_time)[0] # from the waves module
        K0Y = substance.get('k0y')    # water uptake rate constant - get this from database
        
        #note should probably use comp_psuedo_wind from the waves module (pass in wave_height)
        pseudo_wind = self.waves.get_pseudo_wind(model_time)

        # if wave_height > 0.0:
        #     pseudo_wind = 2.0286 * np.sqrt(constants['gravity'] * wave_height) 
        # else:
        #     pseudo_wind = 0
 
        # if pseudo_wind > 4.429:
        #     pseudo_wind = np.power(pseudo_wind / .71, .813) 

        if wind_speed < pseudo_wind:
            wind_speed = pseudo_wind 

        k_emul = 6.0 * K0Y * wind_speed * wind_speed / constants['drop_max']
        
        return k_emul
        

