'''
model dispersion process
'''
from __future__ import division

import copy

import numpy as np

import gnome    # required by new_from_dict

from gnome.array_types import (viscosity,	
                               #area,
                               thickness,
                               mass,
                               density,
                               frac_water)
                               
from gnome.utilities.serializable import Serializable, Field

from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome import constants
from gnome.environment import (WaterSchema,
                               WavesSchema)


from gnome.cy_gnome.cy_weatherers import disperse_oil

class NaturalDispersion(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('water', save=True, update=True, save_reference=True),
               Field('waves', save=True, update=True, save_reference=True)]
    _schema = WeathererSchema

    def __init__(self,
                 waves=None,
                 water=None,
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param waves: waves object for obtaining wave_height, etc at given time
        '''
        self.waves = waves
        self.water = water

        super(NaturalDispersion, self).__init__(**kwargs)
        self.array_types.update({'viscosity': viscosity,
                                 #'area': area,
                                 'thickness':  mass,
                                 'mass':  mass,
                                 'density': density,
                                 'frac_water': frac_water,
                                 })

    def prepare_for_model_run(self, sc):
        '''
        add dispersion key to weathering_data
        Assumes all spills have the same type of oil
        '''
        # create 'dispersed' key if it doesn't exist - are we tracking this ?
        # let's only define this the first time
        if self.active:
            sc.weathering_data['dispersed_natural'] = 0.0


    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Set/update arrays used by dispersion module for this timestep:

        '''

        # do we need this?
        super(NaturalDispersion, self).prepare_for_model_step(sc,
                                                        time_step,
                                                        model_time)
        if not self.active:
            return


    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'dispersion_natural' in sc.weathering_data
        - currently also sets 'density' in sc.weathering_data but may update
          this as we add more weatherers and perhaps density gets set elsewhere
        '''

        if not self.active:
            return

        if sc.num_released == 0:
            return

        wave_height = self.waves.get_value(model_time)[0] # from the waves module
        frac_breaking_waves = self.waves.get_value(model_time)[2] # from the waves module
        disp_wave_energy = self.waves.get_value(model_time)[3] # from the waves module
        visc_w = self.water.kinematic_viscosity
        rho_w = self.water.density
        sediment = self.water.sediment

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['frac_water']) == 0:
                # substance does not contain any surface_weathering LEs
                continue

            V_entrain = constants.volume_entrained
            ka = constants.ka # oil sticking term

            disp = np.zeros((len(data['frac_water'])), dtype=np.float64)

            disperse_oil(time_step,
                         data['frac_water'],
                         data['thickness'],
                         data['mass'],
                         #data['area'],
                         data['viscosity'],
                         data['density'],
                         disp,
                         frac_breaking_waves,
                         disp_wave_energy,
                         wave_height,
                         visc_w,
                         rho_w,
                         sediment,
                         V_entrain,
                         ka)

            #sc.weathering_data['emulsified'] += \
                #np.sum(data['frac_water'][:]) / sc.num_released
            # just average the water fraction each time - it is not per time step value but at a certain time value
            #sc.weathering_data['dispersed_natural'] = \
                #np.sum(disp[:]) / len(data['frac_water']
            sc.weathering_data['dispersed_natural'] += np.sum(disp[:])

            disp_mass_frac = np.sum(disp[:]) / data['mass'].sum()
            data['mass_components'] = \
                (1 - disp_mass_frac) * data['mass_components']
            data['mass'] = data['mass_components'].sum(1)

            self.logger.info('Amount Dispersed: {0}'.
                             format(sc.weathering_data['dispersed_natural']))

        sc.update_from_fatedataview()
                

    def serialize(self, json_='webapi'):
        """
        'water'/'waves' property is saved as references in save file
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        serial = schema.serialize(toserial)
        
        if json_ == 'webapi':
            if self.waves:
                serial['waves'] = self.waves.serialize(json_)
            if self.water:
                serial['water'] = self.water.serialize(json_)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for water / waves
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()

            dict_ = schema.deserialize(json_)
            if 'water' in json_:
                #obj = json_['wind'].pop('obj_type')
                obj = json_['water']['obj_type']
                dict_['water'] = (eval(obj).deserialize(json_['water']))
            if 'waves' in json_:
                #obj = json_['waves'].pop('obj_type')
                obj = json_['waves']['obj_type']
                dict_['waves'] = (eval(obj).deserialize(json_['waves']))
            return dict_

        else:
            return json_
        

