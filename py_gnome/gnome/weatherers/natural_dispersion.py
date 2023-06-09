'''
model dispersion process
'''





import numpy as np

from gnome import constants
from gnome.cy_gnome.cy_weatherers import disperse_oil
from gnome.array_types import gat


from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.environment.water import WaterSchema
from gnome.environment.waves import WavesSchema
from gnome.persist import String, SchemaNode

g = constants.gravity  # the gravitational constant.


class NaturalDispersionSchema(WeathererSchema):
    water = WaterSchema(save=True, update=True, save_reference=True)
    waves = WavesSchema(save=True, update=True, save_reference=True)
    algorithm = SchemaNode(String(),
                           save=True,
                           update=True,
                           description='Algorithm used for dispersion')

class NaturalDispersion(Weatherer):
    _schema = NaturalDispersionSchema
    _ref_as = 'dispersion'
    _req_refs = ['waves', 'water']
    _algorithms_opts = {'D&S1988', 'Li2017'}

    def __init__(self,
                 waves=None,
                 water=None,
                 algorithm='D&S1988',
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param waves: waves object for obtaining wave_height, etc at given time
        '''
        self.waves = waves
        self.water = water
        if algorithm not in self._algorithms_opts:
           raise ValueError(f'{algorithm} not valid, must be one of:{self._algorithm_opts}')
        self.algorithm = algorithm

        super(NaturalDispersion, self).__init__(**kwargs)
        self.array_types.update({'viscosity': gat('viscosity'),
                                 'mass': gat('mass'),
                                 'density': gat('density'),
                                 'positions': gat('positions'),
                                 'area': gat('area'),
                                 'frac_water': gat('frac_water'),
                                 'droplet_avg_size': gat('droplet_avg_size'),
                                 })

    def prepare_for_model_run(self, sc):
        '''
        add dispersion and sedimentation keys to mass_balance
        Assumes all spills have the same type of oil
        '''
        # create 'natural_dispersion' and 'sedimentation keys
        # if they doesn't exist
        # let's only define this the first time
        if self.on:
            super(NaturalDispersion, self).prepare_for_model_run(sc)
            sc.mass_balance['natural_dispersion'] = 0.0
            sc.mass_balance['sedimentation'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        Set/update arrays used by dispersion module for this timestep:

        '''
        super(NaturalDispersion, self).prepare_for_model_step(sc,
                                                              time_step,
                                                              model_time)

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'natural_dispersion' and 'sedimentation' in sc.mass_balance
        '''
        if not self.active:
            return

        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            #print('mass', sc['mass'], data['mass'])
            #print('area', sc['area'], data['area'])

            if len(data['mass']) == 0:
                # substance does not contain any surface_weathering LEs
                continue
            points = data['positions']
            # from the waves module
            waves_values = self.waves.get_value(points, model_time)
            wave_height = waves_values[0]
            frac_breaking_waves = waves_values[2]
            disp_wave_energy = waves_values[3]

            visc_w = self.waves.water.kinematic_viscosity
            rho_w = self.waves.water.density

            # web has different units
            sediment = self.waves.water.get('sediment', unit='kg/m^3')
            V_entrain = constants.volume_entrained
            ka = constants.ka  # oil sticking term

            disp = np.zeros((len(data['mass'])), dtype=np.float64)
            sed = np.zeros((len(data['mass'])), dtype=np.float64)
            droplet_avg_size = data['droplet_avg_size']

            # print ('dispersion: mass_components = {}'
            #        .format(data['mass_components'].sum(1)))
            if self.algorithm == 'D&S1988':
                 dis_fun = self.disperse_oil_DS
            elif self.algorithm == 'Li2017':
                 dis_fun = self.disperse_oil_Li
            else:
                 raise ValueError(f'Dispersion options {self.algorithm} are not "D&S1988" or "Li2017"')

            disp, droplet_avg_size, sed = dis_fun(time_step,
                        data['frac_water'],
                        data['mass'],
                        data['viscosity'],
                        data['density'],
                        data['area'],
                        disp,
                        sed,
                        droplet_avg_size,
                        frac_breaking_waves,
                        disp_wave_energy,
                        wave_height,
                        visc_w,
                        rho_w,
                        sediment,
                        V_entrain,
                        ka)

            sc.mass_balance['natural_dispersion'] += np.sum(disp[:])

            if data['mass'].sum() > 0:
                disp_mass_frac = np.sum(disp[:]) / data['mass'].sum()

                if disp_mass_frac > 1:
                    disp_mass_frac = 1
            else:
                disp_mass_frac = 0

            data['mass_components'] = ((1 - disp_mass_frac) *
                                       data['mass_components'])
            data['mass'] = data['mass_components'].sum(1)

            sc.mass_balance['sedimentation'] += np.sum(sed[:])

            if data['mass'].sum() > 0:
                sed_mass_frac = np.sum(sed[:]) / data['mass'].sum()

                if sed_mass_frac > 1:
                    sed_mass_frac = 1
            else:
                sed_mass_frac = 0

            data['mass_components'] = ((1 - sed_mass_frac) *
                                       data['mass_components'])
            data['mass'] = data['mass_components'].sum(1)

            self.logger.debug('{0} Amount Dispersed for {1}: {2}'
                              .format(self._pid,
                                      substance.name,
                                      sc.mass_balance['natural_dispersion']))
            # print ('dispersion: mass_components = {}'
            #        .format(data['mass_components'].sum(1)))

        sc.update_from_fatedataview()

    def disperse_oil_Li(self, time_step,
                     frac_water,
                     mass,
                     viscosity,
                     density,
                     area,
                     disp_out,
                     sed_out,
                     droplet_avg_size,
                     frac_breaking_waves,
                     disp_wave_energy,
                     wave_height,
                     visc_w,
                     rho_w,
                     sediment,
                     V_entrain,
                     ka):
        '''
        Oil natural dispersion algorithm developed by Li et al., (2017)
        '''
        # typical range of interfacial tension between oil and water 30-40 dyne/cm (10-5, 10-2)
        sigma_o_w = 3.5e-2 # unit N/m
        H0 = wave_height / 0.707 # significant wave height

        d_oil = 4.0 * np.sqrt(sigma_o_w / (constants.gravity * (rho_w - density))) # maximum stable droplet diameter
        dynamic_visc = viscosity * density

        # parameter values estimated by Li et al., (2017)
        a =  4.604e-10
        b =  1.805
        c = -1.023

        r =  1.791
        p =  0.460
        q = -0.518
        # a factor added to adjust volume fraction from 200 micron to 70 micron
        factor = 0.11

        Weber = rho_w * constants.gravity * H0 * d_oil / sigma_o_w # Weber number
        Ohnesorge = dynamic_visc / np.sqrt(density * sigma_o_w * d_oil) # Ohnesorge number
        Q_oil = a * (Weber**b) * (Ohnesorge**c)

        s_mask = area > 0
        Vemul = (mass[s_mask] / density[s_mask])  / (1.0 - frac_water[s_mask]);
        thickness = Vemul / area[s_mask];

        Q_disp = density[s_mask] * thickness * frac_breaking_waves[s_mask] * Q_oil[s_mask] * (1.0 - frac_water[s_mask]) * area[s_mask]
        # print(area[s_mask], area)
        disp_out[s_mask] = factor * Q_disp * time_step

        droplet_avg_size[s_mask] = d_oil[s_mask] * r * (Ohnesorge[s_mask]**p) * (Weber[s_mask]**q)



        # sedimentation algorithm below
        droplet = 0.613 * thickness
        # droplet average rising velocity
        speed = (droplet * droplet * constants.gravity * (1.0 - density / rho_w) / (18.0 * visc_w))

        # vol of refloat oil/wave p
        V_refloat = 0.588 * (np.power(thickness, 1.7) - 5.0e-8)
        V_refloat[V_refloat < 0.0] = 0.0

        # dispersion term at current time.
        C_disp = disp_wave_energy ** 0.57 * frac_breaking_waves

        # Roy's constant
        C_Roy = 2400.0 * np.exp(-73.682 * np.sqrt(viscosity))

        q_refloat = C_Roy * C_disp * V_refloat * area

        C_oil = (q_refloat * time_step / (speed * time_step + 1.5 * wave_height))
        #print('C_oil', C_oil[0])
        # mass rate of oil loss due to sedimentation
        Q_sed = (1.6 * ka * np.sqrt(wave_height * disp_wave_energy * frac_breaking_waves / (rho_w * visc_w)) * C_oil * sediment)


        s_mask = np.logical_and(sediment > 0.0, thickness >= 1.0e-4)
        sed_out[s_mask] = (1.0 - frac_water[s_mask]) * Q_sed[s_mask] * time_step

        s_mask = (disp_out + sed_out) > mass
        disp_out[s_mask] = (disp_out[s_mask] / (disp_out[s_mask] + sed_out[s_mask])) * mass[s_mask]
        sed_out[s_mask] = mass[s_mask] - disp_out[s_mask]

        return disp_out, droplet_avg_size, sed_out


    def disperse_oil_DS(self, time_step,
                     frac_water,
                     mass,
                     viscosity,
                     density,
                     area,
                     disp_out, # output
                     sed_out,  # output
                     droplet_avg_size, # output
                     frac_breaking_waves,
                     disp_wave_energy,
                     wave_height,
                     visc_w,
                     rho_w,
                     sediment,
                     V_entrain,
                     ka):
        '''
            Oil natural dispersion model developed by Delvgine and Sweeney (1988)
            Right now we just want to recreate what the lib_gnome dispersion
            function is doing...but in python.
            This will allow us to more easily refactor, and we can always
            then put it back into lib_gnome if necessary.
            (TODO: Not quite finished with the function yet.)
        '''
        D_e = disp_wave_energy
        f_bw = frac_breaking_waves
        H_rms = wave_height

        # dispersion term at current time.
        C_disp = D_e ** 0.57 * f_bw
        # Roy's constant
        C_Roy = 2400.0 * np.exp(-73.682 * np.sqrt(viscosity))

        for i, (rho, mass_elem, visc, Y, A) in enumerate(zip(density, mass,
                                                        viscosity, frac_water,
                                                        area)):
            if Y >= 1:
               disp_out[i] = 0.0
               sed_out[i] = 0.0
               droplet_avg_size[i] = 0.0
               continue
            # shouldn't happen

            # natural dispersion computation below
            if A > 0:
               # emulsion volume (m3)
               Vemul = (mass_elem / rho) / (1.0 - Y)
               thickness = Vemul / A
            else:
               thickness = 0.0

            # mass rate of oil driven into the first 1.5 wave height (kg/sec)
            Q_disp = C_Roy[i] * C_disp[i] * V_entrain * (1.0 - Y) * A

            d_disp_out = Q_disp * time_step

            # sedimentation computation below
            droplet = 0.613 * thickness
            if sediment > 0.0 and thickness >= 1.0e-4:
               # droplet average rising velocity
               speed = (droplet * droplet * constants.gravity * (1.0 - rho / rho_w) / (18.0 * visc_w))

               # vol of refloat oil/wave p
               V_refloat = 0.588 * (np.power(thickness, 1.7) - 5.0e-8)
               if V_refloat < 0.0:
                  V_refloat = 0.0;

               # (kg/m2-sec) mass rate of emulsion
               q_refloat = C_Roy[i] * C_disp[i] * V_refloat * A

               C_oil = (q_refloat * time_step / (speed * time_step + 1.5 * H_rms[i]))
               # print('C_oil', C_oil)
               # mass rate of oil loss due to sedimentation
               Q_sed = (1.6 * ka * np.sqrt(H_rms[i] * D_e[i] * f_bw[i] / (rho_w * visc_w)) * C_oil * sediment)

               d_sed_out = (1.0 - Y) * Q_sed * time_step
            else:
               d_sed_out = 0.0

            if (d_disp_out + d_sed_out) > mass_elem:
               ratio = d_disp_out / (d_disp_out + d_sed_out)

               d_disp_out = ratio * mass_elem
               d_sed_out = mass_elem - d_disp_out

            disp_out[i] = d_disp_out
            sed_out[i] = d_sed_out
            droplet_avg_size[i] = droplet

        return disp_out, droplet_avg_size, sed_out