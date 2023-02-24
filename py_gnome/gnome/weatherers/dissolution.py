'''
model dissolution process
'''

import copy
import contextlib

import numpy as np


from gnome.utilities.weathering import (BanerjeeHuibers, Stokes,
                                        DingFarmer, DelvigneSweeney,
                                        PiersonMoskowitz)

from gnome.array_types import gat

from .core import WeathererSchema
from gnome.weatherers import Weatherer

from pprint import PrettyPrinter
from gnome.environment.waves import WavesSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.environment.wind import WindSchema
from gnome.environment.gridded_objects_base import VectorVariableSchema
pp = PrettyPrinter(indent=2, width=120)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


class DissolutionSchema(WeathererSchema):
    waves = WavesSchema(
        save=True, update=True, save_reference=True
    )
    wind = GeneralGnomeObjectSchema(
        acceptable_schemas=[WindSchema, VectorVariableSchema],
        save=True, update=True, save_reference=True
    )


class Dissolution(Weatherer):
    """
    Dissolution is still under development and not recommended for use.
    """

    _schema = DissolutionSchema

    _ref_as = 'dissolution'
    _req_refs = ['waves', 'wind']

    def __init__(self, waves=None, wind=None, **kwargs):
        '''
            :param waves: waves object for obtaining wave_height, etc. at a
                          given time
        '''
        self.waves = waves
        self.wind = wind

        if waves is not None and wind is not None:
            make_default_refs = False
        else:
            make_default_refs = True

        super(Dissolution, self).__init__(make_default_refs=make_default_refs, **kwargs)

        self.array_types.update({'area': gat('area'),
                                 'mass': gat('mass'),
                                 'density': gat('density'),
                                 'positions': gat('positions'),
                                 'viscosity': gat('viscosity'),
                                 'partition_coeff': gat('partition_coeff'),
                                 'droplet_avg_size': gat('droplet_avg_size')
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

            Actually, there is nothing to do to initialize our partition
            coefficient, as it is recalculated in dissolve_oil()
        '''
        pass

    # this will have to be updated; SARA is being refactored out of gnome_oil
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

                .. note::
                    right now the natural dispersion process only
                    calculates a single average droplet size. But we still
                    treat it as an iterable.

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
        points = data['positions']

        # print 'droplet_avg_sizes = ', droplet_avg_sizes

        #arom_mask = substance._sara['type'] == 'Aromatics'
        sara = np.asarray(substance.sara_type)
        arom_mask = sara == 'Aromatics'

        mol_wt = substance.molecular_weight
        rho = substance.component_density

        assert mol_wt.shape == rho.shape

        # calculate the partition coefficient (K_ow) for all aromatics
        # for each LE.
        # K_ow for non-aromatics are masked to 0.0
        K_ow_comp = arom_mask * BanerjeeHuibers.partition_coeff(mol_wt, rho)
        data['partition_coeff'] = ((fmasses * K_ow_comp / mol_wt).sum(axis=1) /
                                   (fmasses / mol_wt).sum(axis=1))

        avg_rhos = self.oil_avg_density(fmasses, rho)
        # print ('oil density at temp = {}'
        #        .format(substance.get_density(self.waves.water
        #                                      .get('temperature'))))
        # print 'avg_rhos = ', avg_rhos
        water_rhos = np.zeros(avg_rhos.shape) + self.waves.water.get('density')

        k_w_i = Stokes.water_phase_xfer_velocity(water_rhos - avg_rhos,
                                                 droplet_avg_sizes)
        k_diffusion = 0.134  # Thorpe turbulent diffusion coefficient

        total_volumes = self.oil_total_volume(fmasses, rho)

        f_wc_i = self.water_column_time_fraction(points,model_time, k_w_i)
        T_wc_i = f_wc_i * time_step
        # print 'T_wc_i = ', T_wc_i

        T_calm_i = self.calm_between_wave_breaks(points,model_time, time_step, T_wc_i)
        # print 'T_calm_i = ', T_calm_i

        assert np.alltrue(T_calm_i <= float(time_step))
        assert np.alltrue(T_wc_i <= float(time_step))
        assert np.alltrue(T_wc_i + T_calm_i <= float(time_step))

        oil_concentrations = self.oil_concentration(fmasses, rho)

        N_drop_i = self.droplet_subsurface_mass_xfer_rate(droplet_avg_sizes,
                                                          k_w_i + k_diffusion,
                                                          oil_concentrations,
                                                          K_ow_comp,
                                                          arom_mask,
                                                          total_volumes
                                                          )
        # with printoptions(precision=2):
        #     print 'N_drop_i = ', N_drop_i
        #
        # print 'T_wc_i = ', T_wc_i

        #
        # OK, here it is, the mass dissolved in the water column.
        #
        mass_dissolved_in_wc = (N_drop_i.T * T_wc_i).T

        # with printoptions(precision=2):
        #     print 'mass_dissolved_in_wc = ', mass_dissolved_in_wc

        N_s_i = self.slick_subsurface_mass_xfer_rate(points,
                                                     model_time,
                                                     oil_concentrations,
                                                     K_ow_comp,
                                                     areas,
                                                     arom_mask)
        # with printoptions(precision=2):
        #     print 'N_s_i = ', N_s_i

        #
        # OK, here it is, the mass dissolved in the slick.
        #
        mass_dissolved_in_slick = (N_s_i.T * T_calm_i).T

        # with printoptions(precision=2):
        #     print 'mass_dissolved_in_slick = ', mass_dissolved_in_slick

        mass_dissolved_in_wc = np.vstack(mass_dissolved_in_wc)
        mass_dissolved_in_slick = np.vstack(mass_dissolved_in_slick)
        total_mass_dissolved = mass_dissolved_in_wc + mass_dissolved_in_slick

        # adjust any masses that might go negative
        total_mass_dissolved += np.clip(fmasses - total_mass_dissolved,
                                        - np.inf, 0.0)

        return total_mass_dissolved

    def oil_avg_density(self, masses, densities):
        # oil component count needs to match
        assert masses.shape[-1] == densities.shape[-1]
        assert len(densities.shape) == 1  # single dimension

        if len(masses.shape) == 1:
            # a single LE of mass components
            avg_rho = (masses / masses.sum() * densities).sum()
        else:
            # multiple LE mass components in a 2D array
            avg_rho = (((masses.T / masses.sum(axis=1).T).T * densities)
                       .sum(axis=1))

        return np.nan_to_num(avg_rho)

    def oil_total_volume(self, masses, densities):
        # oil component count needs to match
        assert masses.shape[-1] == densities.shape[-1]
        assert len(densities.shape) == 1  # single dimension

        if len(masses.shape) == 1:
            # a single LE of mass components
            return (masses / densities).sum()
        else:
            # multiple LE mass components in a 2D array
            return (masses / densities).sum(axis=1)

    def state_variable(self, masses, densities, arom_mask):
        # oil component count needs to match
        assert masses.shape[-1] == densities.shape[-1]
        assert len(densities.shape) == 1  # single dimension
        assert len(arom_mask.shape) == 1  # single dimension

        not_arom_mask = arom_mask ^ True

        if len(masses.shape) == 1:
            # a single LE of mass components
            aromatic_volume = (masses / densities * arom_mask).sum()
            S_RA_volume = (masses / densities * not_arom_mask).sum()

            return aromatic_volume / S_RA_volume, S_RA_volume
        else:
            # multiple LE mass components in a 2D array
            aromatic_volume = (masses / densities * arom_mask).sum(axis=1)
            S_RA_volume = (masses / densities * not_arom_mask).sum(axis=1)

            return aromatic_volume / S_RA_volume, S_RA_volume

    def beta_coeff(self, k_w, K_ow, v_inert):
        return 4.84 * k_w / K_ow * v_inert ** (2.0 / 3.0)

    def water_column_time_fraction(self,
                                   points,
                                   model_time,
                                   water_phase_xfer_velocity):
        wave_height = self.waves.get_value(points, model_time)[0]
        wind_speed = np.clip(self.get_wind_speed(points, model_time), 0.01, None)
        wave_period = PiersonMoskowitz.peak_wave_period(wind_speed)

        f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)

        return DingFarmer.water_column_time_fraction(f_bw,
                                                     wave_period,
                                                     wave_height,
                                                     water_phase_xfer_velocity)

    def calm_between_wave_breaks(self,
                                 points,
                                 model_time,
                                 time_step,
                                 time_spent_in_wc=0.0):
        #wind_speed = max(.1, self.waves.wind.get_value(model_time)[0])
        wind_speed = np.clip(self.get_wind_speed(points, model_time), 0.01, None)
        wave_period = PiersonMoskowitz.peak_wave_period(wind_speed)

        f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)

        T_calm = DingFarmer.calm_between_wave_breaks(f_bw, wave_period)

        return np.clip(T_calm, 0.0, float(time_step) - time_spent_in_wc)

    def oil_concentration(self, masses, densities):
        # oil component count needs to match
        assert masses.shape[-1] == densities.shape[-1]
        assert len(densities.shape) == 1  # single dimension

        if len(masses.shape) == 1:
            # a single LE of mass components
            mass_fractions = masses / masses.sum()
            aggregate_rho = (mass_fractions * densities).sum()

            return mass_fractions * aggregate_rho
        else:
            # multiple LE mass components in a 2D array
            mass_fractions = (masses.T / masses.sum(axis=1)).T
            aggregate_rho = (mass_fractions * densities).sum(axis=1)

            return (mass_fractions.T * aggregate_rho).T

    def droplet_subsurface_mass_xfer_rate(self,
                                          droplet_avg_size,
                                          k_w,
                                          oil_concentrations,
                                          partition_coeffs,
                                          arom_mask,
                                          total_volumes
                                          ):
        '''
            Here we are implementing something similar to equations

            - 1.26: this should estimate the mass xfer rate in kg/s

                    .. note::
                        For this equation to work, we need to estimate
                        the total surface area of all droplets, not just
                        a single one

            - 1.27: this should estimate the mass xfer rate per unit area in kg/(m^2 * s)
            - 1.28: combines equations 1.26 and 1.27
            - 1.29: estimates the surface area of a single droplet.

            We return the mass xfer rate in units (kg/s)

            .. note::
                The Cohen equation (eq. 1.1, 1.27), I believe, is actually
                expressed in kg/(m^2 * hr).  So we need to convert our
                time units.

            .. note::
                for now, we are receiving a single average droplet size,
                which we assume will account for 100% of the oil volume.
                In the future we will need to work with something like::
                    [(drop_size, vol_fraction, k_w_drop),
                     ...
                    ]

                This is because each droplet bin will represent a fraction
                of the total oil volume (or mass?), and will have its own
                distinct rise velocity.
                oil_concentrations and partition coefficients will be the
                same regardless of droplet size.
        '''
        K_ow = partition_coeffs
        C_dis = oil_concentrations * arom_mask

        # with printoptions(precision=2):
        #     print 'C_dis = ', C_dis
        #     print 'K_ow = ', K_ow

        # ok, first lets get the xfer rate per unit area (1.27)
        N_drop_a = ((C_dis / K_ow).T * (k_w / 3600.0)).T

        # with printoptions(precision=2):
        #     print 'N_drop_a = ', N_drop_a

        # now we calculate the xfer rate.  For this we need the total area
        # first the slow method, just to prove our equations.
        # print 'droplet_avg_sizes = ', droplet_avg_size

        A_drop = 4 * np.pi * (droplet_avg_size / 2.0) ** 2.0
        # print 'A_drop = ', A_drop

        V_drop = (4.0 / 3.0) * np.pi * (droplet_avg_size / 2.0) ** 3.0
        # print 'V_drop = ', V_drop

        num_droplets = total_volumes / V_drop
        # print 'num_droplets = ', num_droplets

        total_surface_area = A_drop * num_droplets
        # print 'total_surface_area = ', total_surface_area

        N_drop = (N_drop_a.T * total_surface_area).T

        return np.nan_to_num(N_drop)

    def slick_subsurface_mass_xfer_rate(self,
                                        points,
                                        model_time,
                                        oil_concentration,
                                        partition_coeff,
                                        slick_area,
                                        arom_mask):
        '''
            Here we are implementing something similar to equation 1.21
            of our dissolution document.

            The Cohen equation (eq. 1.1), I believe, is actually expressed
            in kg/(m^2 * hr).  So we need to convert our time units.

            We return the mass xfer rate in units (kg/s)
        '''
        # print 'slick_area = ', slick_area

        # oil component count needs to match
        assert oil_concentration.shape[-1] == partition_coeff.shape[-1]
        assert len(partition_coeff.shape) == 1  # single dimension

        #U_10 = max(.1, self.waves.wind.get_value(model_time)[0])
        U_10 = np.clip(self.get_wind_speed(points, model_time), 0.01, None).reshape(-1,1)
        c_oil = oil_concentration
        k_ow = partition_coeff

        # with printoptions(precision=2):
        #     print 'c_oil = ', c_oil
        #     print 'k_ow = ', k_ow

        if len(c_oil.shape) == 1:
            # a single LE of mass components
            # mass xfer rate (per unit area)
            N_s_a = (0.01 *
                     (U_10 / 3600.0) *
                     (c_oil / k_ow))

            N_s = N_s_a * slick_area
        else:
            # multiple LE mass components in a 2D array
            N_s_a = (0.01 * np.prod((U_10 / 3600.0))
                      *
                     (c_oil / k_ow))

            # with printoptions(precision=2):
            #     print 'N_s_a = ', N_s_a

            N_s = (N_s_a.T * slick_area).T

        return np.nan_to_num(N_s * arom_mask)

    def weather_elements(self, sc, time_step, model_time):
        '''
            weather elements over time_step
        '''
        if not self.active or sc.num_released == 0 or not sc.substance.is_weatherable:
            return

        for substance, data in sc.itersubstancedata(self.array_types):

            if len(data['mass']) == 0:
                # data does not contain any surface_weathering LEs
                return

            # print ('dissolution: mass_components = {}'
            #        .format(data['mass_components'].sum(1)))
            diss = self.dissolve_oil(model_time=model_time,
                                     time_step=time_step,
                                     data=data,
                                     substance=substance)

            # print 'diss = ', diss

            # TODO: We should probably only modify the floating LEs
            data['mass_components'] -= diss

            sc.mass_balance['dissolution'] += diss.sum()

            data['mass'] = data['mass_components'].sum(1)

            self.logger.debug('{0} Amount dissolved for {1}: {2}'
                              .format(self._pid,
                                      substance.name,
                                      sc.mass_balance['dissolution']))
            # print ('dissolution: mass_components = {}'
            #        .format(data['mass_components'].sum(1)))
        sc.update_from_fatedataview()

