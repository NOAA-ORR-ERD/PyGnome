'''
model dispersion process
'''
from __future__ import division

import numpy as np

from gnome import constants
from gnome.cy_gnome.cy_weatherers import disperse_oil
from gnome.array_types import (viscosity,
                               mass,
                               density,
                               positions,
                               area,
                               frac_water,
                               droplet_avg_size)


from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.environment.water import WaterSchema
from gnome.environment.waves import WavesSchema

g = constants.gravity  # the gravitational constant.


class NaturalDispersionSchema(WeathererSchema):
    water = WaterSchema(save=True, update=True, save_reference=True)
    waves = WavesSchema(save=True, update=True, save_reference=True)


class NaturalDispersion(Weatherer):
    _schema = NaturalDispersionSchema

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
                                 'mass':  mass,
                                 'density': density,
                                 'positions': positions,
                                 'area': area,
                                 'frac_water': frac_water,
                                 'droplet_avg_size': droplet_avg_size,
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

            disperse_oil(time_step,
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

    def disperse_oil(self, time_step,
                     frac_water,
                     mass,
                     viscosity,
                     density,
                     area,
                     disp_out,
                     sed_out,
                     frac_breaking_waves,
                     disp_wave_energy,
                     wave_height,
                     visc_w,
                     rho_w,
                     sediment,
                     V_entrain,
                     ka):
        '''
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

        for i, (rho, mass, visc, Y, A) in enumerate(zip(density, mass,
                                                        viscosity, frac_water,
                                                        area)):
            pass
