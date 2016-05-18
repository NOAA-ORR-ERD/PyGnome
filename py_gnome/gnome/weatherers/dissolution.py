'''
model dissolution process
'''
from __future__ import division
from collections import Iterable
import copy

import numpy as np

import gnome  # required by deserialize

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
            :param waves: waves object for obtaining wave_height, etc. at a
                          given time
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

        for _substance, data in sc.itersubstancedata(self.array_types):
            if len(data['partition_coeff']) == 0:
                # print 'no particles released yet...'
                continue

            mask = data['partition_coeff'] == 0

            num_initialized += len(mask)

            for s_num in np.unique(data['spill_num'][mask]):
                s_mask = np.logical_and(mask, data['spill_num'] == s_num)

                data['partition_coeff'][s_mask] = 0

    def dissolve_oil(self, data, substance, **kwargs):
        '''
            Here is where we calculate the dissolved oil.
            We will outline the steps as we go along, but off the top of
            my head:
            - recalculate the partition coefficient (K_ow)
            - droplet distribution per LE should be calculated by the
              natural dispersion process and saved in the data arrays before
              the dissolution weathering process.
            - for each LE:
                (Note: right now the natural dispersion process only
                       calculates a single average droplet size. But we still
                       treat it as an iterable.)
                - for each droplet size category:
                    - calculate the water phase transfer velocity (k_w)
                    - calculate the mass xfer rate coefficient (beta)
                    - calculate the water column time fraction (f_wc)
                    - calculate the mass dissolved during refloat period
                - calculate the mass dissolved from the slick during the
                  calm period.
            - the mass dissolved in the water column and the slick is summed
              per mass fraction (should only be aromatic fractions)
            - the sum of dissolved masses are compared to the existing mass
              fractions and adjusted to make sure we don't dissolve more
              mass than exists in the mass fractions.
        '''
        model_time = kwargs.get('model_time')
        time_step = kwargs.get('time_step')

        fmasses = data['mass_components']
        droplet_avg_sizes = data['droplet_avg_size']
        areas = data['area']

        arom_mask = substance._sara['type'] == 'Aromatics'
        not_arom_mask = arom_mask ^ True

        mol_wt = substance.molecular_weight
        rho = substance.component_density

        assert mol_wt.shape == rho.shape

        # calculate the partition coefficient (K_ow) for all aromatics.
        # K_ow for non-aromatics are masked to 0.0
        K_ow_comp = arom_mask * LeeHuibers.partition_coeff(mol_wt, rho)

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

            # overall K_ow value
            K_ow = (np.sum(m * K_ow_comp / mol_wt) /
                    np.sum(m / mol_wt))

            data['partition_coeff'][idx] = K_ow

            avg_rho = self.oil_avg_density(m, rho)
            water_rho = self.waves.water.get('density')

            k_w = Stokes.water_phase_xfer_velocity(water_rho - avg_rho,
                                                   drop_sizes)

            total_volume = (m / rho).sum()
            aromatic_volume = ((m / rho) * arom_mask).sum()
            S_RA_volume = ((m / rho) * not_arom_mask).sum()
            X = aromatic_volume / S_RA_volume

            assert np.isclose(aromatic_volume + S_RA_volume, total_volume)

            beta = self.beta_coeff(k_w, K_ow, S_RA_volume)

            dX_dt = beta * X / (X + 1.0) ** (1.0 / 3.0)

            f_wc = self.water_column_time_fraction(model_time, k_w)
            T_calm = self.calm_between_wave_breaks(model_time)

            time_spent_in_wc = f_wc * time_step

            #
            # OK, here it is, the mass dissolved in the water column.
            #
            aromatic_mass = m * arom_mask
            aromatic_fractions = aromatic_mass / aromatic_mass.sum()

            mass_dissolved_in_wc.append(np.nan_to_num(aromatic_fractions *
                                                      dX_dt *
                                                      time_spent_in_wc))

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

            mass_dissolved_in_slick.append(N_s * T_calm)

        mass_dissolved_in_wc = np.vstack(mass_dissolved_in_wc)
        mass_dissolved_in_slick = np.vstack(mass_dissolved_in_slick)
        total_mass_dissolved = mass_dissolved_in_wc + mass_dissolved_in_slick

        # adjust any masses that might go negative
        total_mass_dissolved += np.clip(fmasses - total_mass_dissolved,
                                        -np.inf, 0.0)

        return total_mass_dissolved

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

        f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)

        return DingFarmer.water_column_time_fraction(f_bw,
                                                     wave_period,
                                                     wave_height,
                                                     water_phase_xfer_velocity)

    def calm_between_wave_breaks(self, model_time):
        wave_period = self.waves.peak_wave_period(model_time)
        wind_speed = self.waves.wind.get_value(model_time)[0]

        f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)

        return DingFarmer.calm_between_wave_breaks(f_bw, wave_period)

    def oil_concentration(self, masses, densities):
        assert masses.shape == densities.shape

        mass_fractions = masses / masses.sum()
        aggregate_rho = (mass_fractions * densities).sum()

        return aggregate_rho * mass_fractions

    def slick_subsurface_mass_xfer_rate(self, model_time,
                                        oil_concentration,
                                        partition_coeff,
                                        slick_area):
        '''
            Here we are implementing something similar to equation 1.21
            of our dissolution document.

            The Cohen equation (eq. 1.1), I believe, is actually expressed
            in kg/(m^2 * hr).  So we need to convert our time units.

            We return the mass xfer rate in units (kg/s)
        '''
        U_10 = self.waves.wind.get_value(model_time)[0]
        c_oil = oil_concentration
        k_ow = partition_coeff

        # mass xfer rate (per unit area)
        N_s_a = (0.01 *
                 (U_10 / 3600.0) *
                 (c_oil / k_ow))

        return N_s_a * slick_area

    def weather_elements(self, sc, time_step, model_time):
        '''
            weather elements over time_step
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

            # TODO: We should probably only modify the floating LEs
            data['mass_components'] -= diss

            sc.mass_balance['dissolution'] += diss.sum()

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
