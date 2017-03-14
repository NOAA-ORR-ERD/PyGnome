'''
model emulsification process
'''
from __future__ import division

import copy

import numpy as np

import gnome

from gnome.array_types import (frac_lost,  # due to evaporation and dissolution
                               age,
                               mass,
                               oil_density,
                               density,
                               bulltime,
                               interfacial_area,
                               oil_viscosity,
                               viscosity,
                               frac_water)

from gnome.utilities.serializable import Serializable, Field
from gnome import constants
from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.cy_gnome.cy_weatherers import emulsify_oil
from gnome.persist import class_from_objtype


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
        :type waves: get_emulsification_wind(model_time)
        '''
        self.waves = waves

        self._bw = 0
        if waves is not None:
            kwargs['make_default_refs'] = \
                kwargs.pop('make_default_refs', False)

        super(Emulsification, self).__init__(**kwargs)
        self.array_types.update({'age', 'bulltime', 'frac_water',
                                 'density', 'viscosity', 
                                 'oil_density', 'oil_viscosity', 
                                 'mass', 'interfacial_area', 'frac_lost'})

    def prepare_for_model_run(self, sc):
        '''
        add water_content key to mass_balance
        Assumes all spills have the same type of oil
        '''
        # create 'water_content' key if it doesn't exist
        # let's only define this the first time
        if self.on:
            super(Emulsification, self).prepare_for_model_run(sc)
            sc.mass_balance['water_content'] = 0.0
            self._bw = 0

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

    # eventually switch this in
    def new_weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'water_content' in sc.mass_balance
        '''

        if not self.active:
            return
        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['age']) == 0:
            #if len(data['frac_water']) == 0:
                # substance does not contain any surface_weathering LEs
                continue
                
            product_type = substance.get('product_type')
            if product_type == 'Refined':
                data['frac_water'][:] = 0.0	# since there can only be one product type this could be return...
                continue	# since there can only be one product type this could be return...

			# compute energy dissipation rate (m^2/s^3) based on wave height
            wave_height = self.waves.get_value(model_time)[0]
            if wave_height > 0:
                eps = (.0355 * wave_height ** .215) / ((np.log(6.31 / wave_height ** 1.45)) ** 3)
            else:
                #eps = 0.
                continue

            water_temp = self.waves.water.get('temperature', 'K')
            rho_oil = substance.density_at_temp(water_temp)
            dens_emul = data['density']
            visc_emul = data['viscosity']
            dens_oil = data['oil_density']
            visc_oil = data['oil_viscosity']
            sigma_ow = substance.oil_water_surface_tension() # does this vary in time?
            v0 = substance.kvis_at_temp(water_temp)	#viscosity is calculated in weathering_data
            if wave_height > 0:
                delta_T_emul = 1630 + 450 / wave_height ** (1.5)
            else:
                continue
			
            visc_min = .00001 # 10 cSt
            visc_max = .01 # 10000 cSt
            sigma_min = .01 # 10 dyne/com
            # new suggestion .03 <= f_asph <= .2
            f_min = .03
            f_max = .2
            r_min = .2
            r_max = 1.4
            rho_min = 600	#kg/m^3
            drop_min = .000008	# 8 microns
            
            k_emul2 = 2.3 / delta_T_emul
            k_emul = self._water_uptake_coeff(model_time, substance)

            # bulltime is not in database, but could be set by user
            #emul_time = substance.get_bulltime()
            emul_time = substance.bulltime

            resin_mask = substance._sara['type'] == 'Resins'
            asphaltene_mask = substance._sara['type'] == 'Asphaltenes'
            saturates_mask = substance._sara['type'] == 'Saturates'
            aromatics_mask = substance._sara['type'] == 'Aromatics'

            f_sat = (saturates_mask * data['mass_components']).sum(axis=1) / data['mass'].sum()
            f_arom = (aromatics_mask * data['mass_components']).sum(axis=1) / data['mass'].sum()

            # will want to use mass_components to update over time
            f_res1 = resin_mask * substance._sara['fraction']
            f_res = np.sum(resin_mask * substance._sara['fraction'])
            f_asph = np.sum(asphaltene_mask * substance._sara['fraction'])
            rho_asph = np.sum(asphaltene_mask * substance._sara['density']) # 1100 kg/m^3

            f_res2 = resin_mask * data['mass_components']
            if data['mass'].sum() == 0:
                continue
            f_res3 = (resin_mask * data['mass_components']).sum(axis=1) / data['mass'].sum()
            f_asph3 = (asphaltene_mask * data['mass_components']).sum(axis=1) / data['mass'].sum()

            if f_res > 0:	
                r_oil = f_asph / f_res	
            else:	
                #r_oil = 0
                continue
            if f_asph <= 0:	
                continue	
            r_oil3 = np.where(f_res3 > 0, f_asph3 / f_res3, 0)	# check if limits are just for S_b calculation

            Y_max = .61 + .5 * r_oil - .28 * r_oil **2
            # limit on r_oil3 values or just final Y_max or set Y_max = 0 if out of bounds?
            if Y_max > .9:
                Y_max = .9
                
            m = .5 * (visc_max + visc_min)
            x_visc = (visc_oil - m) / (visc_max - visc_min)
            
            x_sig_min = (sigma_ow[0] - sigma_min) / sigma_ow[0]

            m = .5 * (f_max + f_min)
            x_fasph = (f_asph3 - m) / (f_max - f_min)

            m = .5 * (r_max + r_min)
            x_r = (r_oil - m) / (r_max - r_min)

            x_s = 0 		# placeholder since this isn't used
		
            # decide which factors use initial value and which use current value
            # once Bw is set it stays on
            Bw = self._Bw(x_visc,x_sig_min,x_fasph,x_r,x_s)

            T_week = 604800

            S_b = ((dens_oil * visc_oil**.25 / sigma_ow[0]) * r_oil * np.exp(-2 * r_oil**2))**(1/6)
            S_b[S_b > 1] = 1.
            S_b[S_b < 0] = 0.
            T_week = 604800

            
            k_lw = np.where(data['frac_water'] > 0, (1 - S_b) / T_week, 0.)
            
            data['frac_water'] += (Bw * (k_emul2 * (Y_max - data['frac_water'])) - k_lw * data['frac_water']) * time_step
            data['frac_water'] = np.where(data['frac_water']>Y_max,Y_max,data['frac_water'])
            # get from database bullwinkle (could be overridden by user)
            #emul_constant = substance.get('bullwinkle_fraction')
            # will want to check if user set a value, and change the interface since there is no longer a bullwinkle
            emul_constant = substance.bullwinkle

            # max water content fraction - get from database
            Y_max = substance.get('emulsion_water_fraction_max')

            # doesn't emulsify, avoid the nans
            if Y_max <= 0:
                continue
            S_max = (6. / constants.drop_min) * (Y_max / (1.0 - Y_max))

            #sc.mass_balance['water_content'] += \
                #np.sum(data['frac_water'][:]) / sc.num_released
            # just average the water fraction each time - it is not per time
            # step value but at a certain time value
            # todo: probably should be weighted avg
            if data['mass'].sum() > 0:
                sc.mass_balance['water_content'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['frac_water'])

            self.logger.debug(self._pid + 'water_content for {0}: {1}'.
                              format(substance.name,
                                     sc.mass_balance['water_content']))

        sc.update_from_fatedataview()


    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'water_content' in sc.mass_balance
        '''

        if not self.active:
            return
        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['age']) == 0:
            #if len(data['frac_water']) == 0:
                # substance does not contain any surface_weathering LEs
                continue

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

            #sc.mass_balance['water_content'] += \
                #np.sum(data['frac_water'][:]) / sc.num_released
            # just average the water fraction each time - it is not per time
            # step value but at a certain time value
            # todo: probably should be weighted avg
            if data['mass'].sum() > 0:
                sc.mass_balance['water_content'] = \
                    np.sum(data['mass']/data['mass'].sum() * data['frac_water'])

            self.logger.debug(self._pid + 'water_content for {0}: {1}'.
                              format(substance.name,
                                     sc.mass_balance['water_content']))

        sc.update_from_fatedataview()

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
            if 'waves' in json_:
                obj = json_['waves']['obj_type']
                dict_['waves'] = (eval(obj).deserialize(json_['waves']))
#             if 'waves' in json_:
#                 waves = class_from_objtype(json_['waves'].pop('obj_type'))
#                 dict_['waves'] = waves.deserialize(json_['waves'])
            return dict_

        else:
            return json_


    def _H_log(self, k, x):
        '''
        logistic function for turning on emulsification
        '''
        H_log = 1 / (1 + np.exp(-1*k*x))
        
        return H_log

    def _H_4(self, k, x):
        '''
        symmetric function for turning on emulsification
        '''
        H_4 = 1 / (1 + x**(2*k))
        
        return H_4

    def _Bw(self, x_visc, x_sig_min, x_fasph, x_r, x_s):
        '''
        '''
        k_v = 4
        k_sig = 4
        k_fasph = 3
        k_r = 2
        k_s = 1.5
        
        # for now, I think P_min will be determined elsewhere
        U = 0
        P_min = .03
        #P_min = P_min - U*P_min
        #P_min = P_min + U*(1 - P_min)

        k = 4
        P_1 = self._H_log(k,x_sig_min)

        k = 4
        P_2 = self._H_4(k,x_visc)
        
        k = 3
        P_3 = self._H_4(k,x_fasph)
        
        k = 2
        P_4 = self._H_4(k,x_r)
        
        k = 1.5
        #P_5 = self._H_log(k,x_s)
        P_5 = 1 # placeholder until Bill comes up with a good option
        
        P_all = P_1 * P_2 * P_3 * P_4 * P_5
        #P_all = self._H_log(k_v,x_v_min) * self._H_log(k_v,x_v_max) * self._H_log(k_sig,x_sig_min) * self._H_log(k_fasph,x_fasph) * self._H_log(k_r,x_r_min) * self._H_log(k_r,x_r_max) * self._H_log(k_s,x_s_min)

        #if (P_all.any() < P_min):
        if (P_all.all() < P_min):
            Bw = 0 
        else:
            Bw = 1
             
        Bw =  np.where(P_all < .03, 0, 1)
        
        return Bw

    def _water_uptake_coeff(self, model_time, substance):
        '''
        Use higher of wind or pseudo wind corresponding to wave height

            if (H0 > 0) HU = 2.0286 * sqrt(g * H0)
            if (HU < 4.429) HU = pow(HU / .71, .813)
            if (U < HU) U = HU
            k_emul = 6.0 * K0Y * U * U / d_max

        '''

        ## higher of real or psuedo wind
        wind_speed = self.waves.get_emulsification_wind(model_time)

        # water uptake rate constant - get this from database
        K0Y = substance.get('k0y')

        k_emul = 6.0 * K0Y * wind_speed * wind_speed / constants.drop_max

        return k_emul
