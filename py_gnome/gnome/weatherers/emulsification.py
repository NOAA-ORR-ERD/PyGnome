'''
model emulsification process
'''

import numpy as np

from gnome.array_types import gat

from gnome import constants
from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.cy_gnome.cy_weatherers import emulsify_oil
from gnome.environment.waves import WavesSchema


class EmulsificationSchema(WeathererSchema):
    waves = WavesSchema(
        save=True, update=True, save_reference=True
    )


class Emulsification(Weatherer):
    _schema = EmulsificationSchema
    _ref_as = 'emulsification'
    _req_refs = ['waves']

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
        self.array_types.update({'age': gat('age'),
                                 'bulltime': gat('bulltime'),
                                 'frac_water': gat('frac_water'),
                                 'density': gat('density'),
                                 'viscosity': gat('viscosity'),
                                 'positions': gat('positions'),
                                 'oil_density': gat('oil_density'),
                                 'oil_viscosity': gat('oil_viscosity'),
                                 'mass': gat('mass'),
                                 'interfacial_area': gat('interfacial_area'),
                                 'frac_evap': gat('frac_evap')})

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
    # this will have to be updated; SARA is being refactored out of gnome_oil
    def weather_elements_lehr(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'water_content' in sc.mass_balance
        '''

        if not self.active or sc.num_released == 0 or not sc.substance.is_weatherable:
            return

        for substance, data in sc.itersubstancedata(self.array_types):

            if len(data['age']) == 0:
            #if len(data['frac_water']) == 0:
                # substance does not contain any surface_weathering LEs
                return

            product_type = substance.get('product_type')
            if product_type == 'Refined':
                data['frac_water'][:] = 0.0	# since there can only be one product type this could be return...
                return	# since there can only be one product type this could be return...

            # compute energy dissipation rate (m^2/s^3) based on wave height
            wave_height = self.waves.get_value(model_time)[0]
            if wave_height > 0:
                eps = (.0355 * wave_height ** .215) / ((np.log(6.31 / wave_height ** 1.45)) ** 3)
            else:
                #eps = 0.
                return

            water_temp = self.waves.water.get('temperature', 'K')
            rho_oil = substance.density_at_temp(water_temp)
            dens_emul = data['density']
            visc_emul = data['viscosity']
            dens_oil = data['oil_density']
            visc_oil = data['oil_viscosity']
            sigma_ow = substance.oil_water_surface_tension() # does this vary in time?
            print("sigma_ow")
            print(sigma_ow[0])
            v0 = substance.kvis_at_temp(water_temp)	#viscosity is calculated in weathering_data
            if wave_height > 0:
                delta_T_emul = 1630 + 450 / wave_height ** (1.5)
            else:
                return

            visc_min = .00001 # 10 cSt
            visc_max = .01 # 10000 cSt
            sigma_min = .01 # 10 dyne/com
            # new suggestion .03 <= f_asph <= .2
            # latest update, min only .03 <= f_asph
            f_min = .03
            f_max = .2
            r_min = .2
            r_max = 1.4
            rho_min = 600	#kg/m^3
            drop_min = .000008	# 8 microns

            #k_emul2 = 2.3 / delta_T_emul
            k_emul2 = 1. / delta_T_emul
            k_emul = self._water_uptake_coeff(model_time, substance)

            emul_time = substance.bullwinkle_time

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
                return
            f_res3 = (resin_mask * data['mass_components']).sum(axis=1) / data['mass'].sum()
            f_asph3 = (asphaltene_mask * data['mass_components']).sum(axis=1) / data['mass'].sum()

            if f_res > 0:
                r_oil = f_asph / f_res
            else:
                #r_oil = 0
                return
            if f_asph <= 0:
                return
            r_oil3 = np.where(f_res3 > 0, f_asph3 / f_res3, 0)	# check if limits are just for S_b calculation

            Y_max = .61 + .5 * r_oil - .28 * r_oil **2
            # limit on r_oil3 values or just final Y_max or set Y_max = 0 if out of bounds?
            if Y_max > .9:
                Y_max = .9

            m = .5 * (visc_max + visc_min)
            x_visc = (visc_oil - m) / (visc_max - visc_min)

            x_sig_min = (sigma_ow[0] - sigma_min) / sigma_ow[0]

            #m = .5 * (f_max + f_min)
            #x_fasph = (f_asph3 - m) / (f_max - f_min)
            # changed to one-sided, add check for f_asph3 = 0
            x_fasph = (f_asph3 - f_min) / (f_asph3)

            m = .5 * (r_max + r_min)
            x_r = (r_oil - m) / (r_max - r_min)

            x_s = 0 		# placeholder since this isn't used

            # decide which factors use initial value and which use current value
            # once Bw is set it stays on
            Bw = self._Bw(x_visc,x_sig_min,x_fasph,x_r,x_s)

            T_week = 604800

            # Bill's calculation uses sigma_ow[0] in dynes/cm, visc in cSt and a fudge factor of .478834
            # so we need to convert and scale
            print("dens_oil")
            print(dens_oil)
            print("visc_oil")
            print(visc_oil)
            print("r_oil")
            print(r_oil)
            S_b = .478834 * ((dens_oil * (1000000*visc_oil)**.25 / (1000*sigma_ow[0])) * r_oil * np.exp(-2 * r_oil**2))**(1/6)
            S_b[S_b > 1] = 1.
            S_b[S_b < 0] = 0.
            print("S_b")
            print(S_b)
            T_week = 604800


            k_lw = np.where(data['frac_water'] > 0, (1 - S_b) / T_week, 0.)

            #data['frac_water'] += (Bw * (k_emul2 * (Y_max - data['frac_water'])) - k_lw * data['frac_water']) * time_step
            Y_prime = 1.582 * Y_max  # Y_max / (1 - 1/e)
            data['frac_water'] += (Bw * (k_emul2 * (Y_prime - data['frac_water'])) - k_lw * data['frac_water']) * time_step
            data['frac_water'] = np.where(data['frac_water']>Y_max,Y_max,data['frac_water'])
            # get from database bullwinkle (could be overridden by user)
            #emul_constant = substance.get('bullwinkle_fraction')
            # will want to check if user set a value, and change the interface since there is no longer a bullwinkle
            emul_constant = substance.bullwinkle

            # max water content fraction - get from database
            Y_max = substance.get('emulsion_water_fraction_max')

            # doesn't emulsify, avoid the nans
            if Y_max <= 0:
                return
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

    def weather_elements_adios2(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'water_content' in sc.mass_balance
        '''

        if not self.active or sc.num_released == 0 or not sc.substance.is_weatherable:
            return

        for substance, data in sc.itersubstancedata(self.array_types):

            # what is this check for? why age?
            if len(data['age']) == 0:
                return

            points = data['positions']
            k_emul = self._water_uptake_coeff(points, model_time, substance)

            emul_time = substance.bullwinkle_time

            emul_constant = substance.bullwinkle_fraction

            # max water content fraction - get from database
            Y_max = substance.get('emulsion_water_fraction_max')

            # doesn't emulsify, avoid the nans
            if Y_max <= 0:
                return
            S_max = (6. / constants.drop_min) * (Y_max / (1.0 - Y_max))

            emulsify_oil(time_step,
                         data['frac_water'],
                         data['interfacial_area'],
                         data['frac_evap'],
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

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'water_content' in sc.mass_balance
        '''

        if not self.active or sc.num_released == 0 or not sc.substance.is_weatherable:
            return

        use_new_algorithm = False
        # only use new algorithm if all substances have measured SARA totals
#         for substance in sc.get_substances():
#             if substance.record.imported is not None:
#                 sat = substance.record.imported.saturates
#                 arom = substance.record.imported.aromatics
#                 if sat is not None and arom is not None:
#                     use_new_algorithm = True
#                 else:
#                     use_new_algorithm = False
#                     break
#             else:
#                 use_new_algorithm = False	#use old algorithm
#                 break

        #self.weather_elements_lehr(sc, time_step, model_time)
        self.weather_elements_adios2(sc, time_step, model_time)

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
        # in his AMOP paper he is using slick thickness...

        P_all = P_1 * P_2 * P_3 * P_4 * P_5
        #P_all = self._H_log(k_v,x_v_min) * self._H_log(k_v,x_v_max) * self._H_log(k_sig,x_sig_min) * self._H_log(k_fasph,x_fasph) * self._H_log(k_r,x_r_min) * self._H_log(k_r,x_r_max) * self._H_log(k_s,x_s_min)

        #if (P_all.any() < P_min):
        if (P_all.all() < P_min):
            Bw = 0
        else:
            Bw = 1

        Bw =  np.where(P_all < .03, 0, 1)

        return Bw

    def _water_uptake_coeff(self, points, model_time, substance):
        '''
        Use higher of wind or pseudo wind corresponding to wave height

            if (H0 > 0) HU = 2.0286 * sqrt(g * H0)
            if (HU < 4.429) HU = pow(HU / .71, .813)
            if (U < HU) U = HU
            k_emul = 6.0 * K0Y * U * U / d_max

        '''

        ## higher of real or psuedo wind
        wind_speed = self.waves.get_emulsification_wind(points, model_time).reshape(-1)

        # water uptake rate constant - get this from database
        #K0Y = substance.get('k0y')
        K0Y = 2.02E-06

        k_emul = 6.0 * K0Y * wind_speed * wind_speed / constants.drop_max

        return k_emul
