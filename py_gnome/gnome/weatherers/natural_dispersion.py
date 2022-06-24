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

g = constants.gravity  # the gravitational constant.


class NaturalDispersionSchema(WeathererSchema):
    water = WaterSchema(save=True, update=True, save_reference=True)
    waves = WavesSchema(save=True, update=True, save_reference=True)


class NaturalDispersion(Weatherer):
    _schema = NaturalDispersionSchema
    _ref_as = 'dispersion'
    _req_refs = ['waves', 'water']

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
            try:
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
            except:
                import pdb
                pdb.post_mortem()

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
        
    def weather_elements_2(self, sc, time_step, model_time):
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
            
            disp = self.disperse_oil2(time_step,
                        data['frac_water'],
                        data['mass'],
                        data['viscosity'],
                        data['density'],
                        data['area'],
                        disp,
                        sed,
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
 

    
    def disperse_oil2(self, time_step,
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
        # typical range of interfacial tension between oil and water 30-40 dyne/cm (10-5, 10-2)
        sigma_o_w = 3.5e-2 # unit N/m         
        H0 = wave_height / 0.707 
        d_oil = 4.0 * np.sqrt(sigma_o_w / (constants.gravity * (rho_w - density)))
        dynamic_visc = viscosity * density
        
        a = 4.604e-10 
        b = 1.805
        c = -1.023
        
        Weber = rho_w * constants.gravity * H0 * d_oil / sigma_o_w        
        Ohnesorge = dynamic_visc / np.sqrt(density * sigma_o_w * d_oil)         
        Q_oil = a * (Weber**b) * (Ohnesorge**c)  
        
        s_mask = area > 0
        Vemul = (mass[s_mask] / density[s_mask])  / (1.0 - frac_water[s_mask]);
        thickness = Vemul / area[s_mask];
#        print('LE mass', mass)
#        print('Vemul', Vemul)
#        print('area', area)
#        print('thickness', thickness)        
#        print('fraction-waves', frac_breaking_waves)
#        print('Q_oil', Q_oil)
        
        Q_disp = density[s_mask] * thickness * frac_breaking_waves * Q_oil[s_mask] * (1.0 - frac_water[s_mask]) * area[s_mask]
        # print(area[s_mask], area)
        disp_out[s_mask] = Q_disp * time_step
#        print(mass, disp_out)
        return disp_out          

    

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
