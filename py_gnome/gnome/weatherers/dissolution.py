'''
model dissolution process
'''
from __future__ import division
from collections import Iterable
import copy

import numpy as np

import gnome  # required by deserialize

from gnome import constants
from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.weathering import (LeeHuibers, Stokes,
                                        DingFarmer, DelvigneSweeney)

from gnome.array_types import (viscosity,
                               mass,
                               density,
                               partition_coeff,
                               droplet_avg_size)

from .core import WeathererSchema
from gnome.weatherers import Weatherer

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2, width=120)


class Dissolution(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('waves', save=True, update=True, save_reference=True)]

    _schema = WeathererSchema

    def __init__(self, waves=None, **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
                           things like water temperature
        :param waves: waves object for obtaining wave_height, etc at given time

        TODO: we still need to validate all the inputs that this weatherer
              requires
        '''
        self.waves = waves

        super(Dissolution, self).__init__(**kwargs)

        self.array_types.update({'viscosity': viscosity,
                                 'mass':  mass,
                                 'density': density,
                                 'partition_coeff': partition_coeff,
                                 'droplet_avg_size': droplet_avg_size
                                 })

    def prepare_for_model_run(self, sc):
        '''
            Add dissolution key to mass_balance if it doesn't exist.
            - Assumes all spills have the same type of oil
            - let's only define this the first time
        '''
        if self.on:
            super(Dissolution, self).prepare_for_model_run(sc)
            sc.mass_balance['dissolution'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
            Set/update arrays used by dispersion module for this timestep
        '''
        super(Dissolution, self).prepare_for_model_step(sc,
                                                        time_step,
                                                        model_time)

        if not self.active:
            return

    def initialize_data(self, sc, num_released):
        '''
            initialize the newly released portions of our data arrays:

            If on is False, then arrays should not be included
            - dont' initialize
        '''
        if not self.on:
            return

        self._initialize_k_ow(sc, num_released)

    def _initialize_k_ow(self, sc, num_released):
        '''
            Initialize the molar averaged oil/water partition coefficient.

            Note: we are assuming that each substance gets a distinct
                  slice from our data arrays (maybe not a good assumption)
        '''
        num_initialized = 0

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['partition_coeff']) == 0:
                # no particles released yet
                continue

            mask = data['partition_coeff'] == 0

            num_initialized += len(mask)

            for s_num in np.unique(data['spill_num'][mask]):
                s_mask = np.logical_and(mask, data['spill_num'] == s_num)

                data['partition_coeff'][s_mask] = 0

        assert num_initialized == num_released

    def dissolve_oil(self, data, substance, **kwargs):
        '''
            Here is where we calculate the dissolved oil.
            We will outline the steps as we go along, but off the top of
            my head:
            - recalculate the partition coefficient (K_ow)
              TODO: This requires a molar average of the aromatic components.
            - use VDROP to calculate the shift in the droplet distribution
            - for each droplet size category:
                - calculate the water phase transfer velocity (k_w) (Stokes)
                - calculate the mass xfer rate coefficient (beta)
                - calculate the water column time fraction (f_wc)
                - calculate the volume dissolved
            - subtract the mass of smallest droplets in our distribution
              that are below a threshold.
        '''
        model_time = kwargs.get('model_time')
        time_step = kwargs.get('time_step')

        fmasses = data['mass_components']
        droplet_avg_sizes = data['droplet_avg_size']
        areas = data['area']
        print 'areas = ', areas

        arom_mask = substance._sara['type'] == 'Aromatics'
        not_arom_mask = arom_mask ^ True

        mol_wt = substance.molecular_weight
        rho = substance.component_density

        assert mol_wt.shape == rho.shape

        # calculate the partition coefficient (K_ow) for all aromatics.
        # K_ow for non-aromatics should be masked to 0.0
        K_ow_comp = arom_mask * LeeHuibers.partition_coeff(mol_wt, rho)
        print 'K_ow_comp = ', K_ow_comp

        mass_dissolved_in_wc = []
        mass_dissolved_in_slick = []
        for idx, (m, drop_sizes, area) in enumerate(zip(fmasses,
                                                        droplet_avg_sizes,
                                                        areas)):
            # This will eventually be a droplet distribution, but for now
            # we are receiving the average droplet size from the dispersion
            # weatherer.  So we turn the scalar into an iterable.
            if not isinstance(drop_sizes, Iterable):
                drop_sizes = np.array(drop_sizes)

            print '\nm = ', m
            print 'drop_sizes = ', drop_sizes, 'in meters diameter'
            # overall K_ow value
            K_ow = (np.sum(m * K_ow_comp / mol_wt) /
                    np.sum(m / mol_wt))

            print '\nK_ow = ', K_ow
            data['partition_coeff'][idx] = K_ow

            avg_rho = self.oil_avg_density(m, rho)
            water_rho = self.waves.water.get('density')
            print 'water_rho, avg_rho = ', (water_rho, avg_rho)

            k_w = Stokes.water_phase_xfer_velocity(water_rho - avg_rho,
                                                   drop_sizes)
            print 'k_w = ', k_w, 'm/s'
            print '    = ', k_w * (60.0 * 60.0), 'm/hr'

            total_volume = (m / rho).sum()
            aromatic_volume = ((m / rho) * arom_mask).sum()
            S_RA_volume = ((m / rho) * not_arom_mask).sum()
            X = aromatic_volume / S_RA_volume

            print 'total volume = ', total_volume, '(m^3)'
            print 'aromatic volume = ', aromatic_volume, '(m^3)'
            print 'S_RA volume = ', S_RA_volume, '(m^3)'
            print 'X = ', X

            assert np.isclose(aromatic_volume + S_RA_volume, total_volume)

            beta = self.beta_coeff(k_w, K_ow, S_RA_volume)
            print 'beta = ', beta

            dX_dt = beta * X / (X + 1) ** (1.0 / 3.0)
            print 'dX_dt = ', dX_dt, 'kg/s'
            print '      = ', dX_dt * 1000.0, 'g/s'

            f_wc = self.water_column_time_fraction(model_time, k_w)
            T_calm = self.calm_between_wave_breaks(model_time)
            print 'f_wc = ', f_wc
            print 'T_calm = ', T_calm

            time_spent_in_wc = f_wc * time_step
            print 'time_spent_in_wc = ', time_spent_in_wc

            #
            # OK, here it is, the mass dissolved in the water column.
            #
            mass_dissolved_in_wc.append(dX_dt * time_spent_in_wc)

            #
            # Now we need to calculate the mass dissolved in the surface slick
            # dV_surf = N_s * rho_dis * time_step
            #
            oil_concentration = self.oil_concentration(m, rho)

            N_s = self.slick_subsurface_mass_xfer_rate(model_time,
                                                       oil_concentration,
                                                       K_ow_comp,
                                                       area)
            N_s = np.nan_to_num(N_s * arom_mask)
            print 'N_s = ', N_s
            print 'time_step = ', time_step

            mass_dissolved_in_slick.append(N_s.sum() * T_calm)

        print 'mass_dissolved in water column = ', mass_dissolved_in_wc
        print 'mass_dissolved in slick = ', mass_dissolved_in_slick
        print

        diss = np.zeros((len(data['mass'])), dtype=np.float64)
        return diss

    def oil_avg_density(self, masses, densities):
        assert masses.shape == densities.shape

        return (masses / masses.sum() * densities).sum()

    def beta_coeff(self, k_w, K_ow, v_inert):
        return 4.84 * k_w / K_ow * v_inert ** (2.0 / 3.0)

    def water_column_time_fraction(self, model_time,
                                   water_phase_xfer_velocity):
        wave_period = self.waves.peak_wave_period(model_time)
        wave_height = self.waves.get_value(model_time)[0]
        wind_speed = self.waves.wind.get_value(model_time)[0]
        print ('wind_speed, wave_height, wave_period = ',
               wind_speed, wave_height, wave_period)

        f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)
        print 'f_bw = ', f_bw

        f_wc = DingFarmer.water_column_time_fraction(f_bw,
                                                     wave_period,
                                                     wave_height,
                                                     water_phase_xfer_velocity)

        return f_wc

    def calm_between_wave_breaks(self, model_time):
        wave_period = self.waves.peak_wave_period(model_time)
        wind_speed = self.waves.wind.get_value(model_time)[0]

        f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)

        T_calm = DingFarmer.calm_between_wave_breaks(f_bw, wave_period)

        return T_calm

    def oil_concentration(self, masses, densities):
        mass_fractions = masses / masses.sum()

        print 'mass_fractions = ', mass_fractions
        print 'densities = ', densities

        aggregate_rho = (mass_fractions * densities).sum()
        print 'aggregate_rho = ', aggregate_rho

        C_oil = aggregate_rho * mass_fractions
        return C_oil

    def slick_subsurface_mass_xfer_rate(self, model_time,
                                        oil_concentration,
                                        partition_coeff,
                                        slick_area,
                                        schmidt_number=1000.0):
        U_10 = self.waves.wind.get_value(model_time)[0]
        c_oil = oil_concentration
        k_ow = partition_coeff
        s_c = schmidt_number

        print 'U_10 = ', U_10
        print 'c_oil = ', c_oil
        print 'k_ow = ', k_ow
        print 'slick_area = ', slick_area

        # mass xfer rate (per unit area) in units (kg / s * m^2)
        N_s_a = (0.01 * U_10 ** (7.0 / 9.0) *
                 (c_oil / k_ow) *
                 (4 * slick_area / np.pi) ** (-1.0 / 18.0) *
                 s_c ** (-2.0 / 3.0))
        print 'N_s_a = ', N_s_a

        # return mass xfer rate in units (kg / s)
        return N_s_a * slick_area

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'dissolution' in sc.mass_balance
        '''
        if not self.active:
            return

        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) == 0:
                # data does not contain any surface_weathering LEs
                continue

            diss = self.dissolve_oil(model_time=model_time,
                                     time_step=time_step,
                                     data=data,
                                     substance=substance)

            print 'mass_balance:'
            pp.pprint(sc.mass_balance)
            sc.mass_balance['dissolution'] += np.sum(diss[:])

            if data['mass'].sum() > 0:
                diss_mass_frac = np.sum(diss[:]) / data['mass'].sum()
                if diss_mass_frac > 1:
                    diss_mass_frac = 1
            else:
                diss_mass_frac = 0

            data['mass_components'] = ((1 - diss_mass_frac) *
                                       data['mass_components'])
            data['mass'] = data['mass_components'].sum(1)

            self.logger.debug('{0} Amount dissolved for {1}: {2}'
                              .format(self._pid,
                                      substance.name,
                                      sc.mass_balance['dissolution']))

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

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        Append correct schema for water / waves
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()
            dict_ = schema.deserialize(json_)

            if 'waves' in json_:
                obj = json_['waves']['obj_type']
                dict_['waves'] = (eval(obj).deserialize(json_['waves']))

            return dict_
        else:
            return json_
