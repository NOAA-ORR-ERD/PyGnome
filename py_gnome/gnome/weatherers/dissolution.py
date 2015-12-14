'''
model dissolution process
'''
from __future__ import division

import copy

import numpy as np

import gnome  # required by deserialize

from gnome import constants
from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.weathering import LeeHuibers

from gnome.array_types import (viscosity,
                               mass,
                               density,
                               mol_wt_components,
                               density_components)

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
                                 'mol_wt_components': mol_wt_components,
                                 'density_components': density_components
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

        self._initialize_mol_wt(sc, num_released)
        self._initialize_density(sc, num_released)

    def _initialize_mol_wt(self, sc, num_released):
        '''
            Initialize the aromatic molecular weight components.

            mol_wt_components has been prebuilt with a static size based on
            substance.num_components.
            We fill in the available aromatics from our substance,
            and leave the rest of the array as 0.0

            Note: we are assuming that each substance gets a distinct
                  slice from our data arrays (maybe not a good assumption)
        '''
        num_initialized = 0

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mol_wt_components']) == 0:
                # no particles released yet
                continue

            mol_weights = substance._component_mw('Aromatics')

            mask = data['mol_wt_components'][:, 0] == 0
            num_initialized += len(mask)

            for s_num in np.unique(data['spill_num'][mask]):
                s_mask = np.logical_and(mask, data['spill_num'] == s_num)

                row_size = len(mol_weights)

                num = len(s_mask)
                mol_weights = (mol_weights,) * num

                data['mol_wt_components'][s_mask, :row_size] = mol_weights

        assert num_initialized == num_released

    def _initialize_density(self, sc, num_released):
        '''
            Initialize the aromatic density components.

            density_components has been prebuilt with a static size based on
            substance.num_components.
            We fill in the available aromatics from our substance,
            and leave the rest of the array as 0.0

            Note: we are assuming that each substance gets a distinct
                  slice from our data arrays (maybe not a good assumption)
        '''
        num_initialized = 0

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['density_components']) == 0:
                # no particles released yet
                continue

            densities = substance._component_density('Aromatics')

            mask = data['density_components'][:, 0] == 0
            num_initialized += len(mask)

            for s_num in np.unique(data['spill_num'][mask]):
                s_mask = np.logical_and(mask, data['spill_num'] == s_num)

                row_size = len(densities)

                num = len(s_mask)
                densities = (densities,) * num

                data['density_components'][s_mask, :row_size] = densities

        assert num_initialized == num_released

    def dissolve_oil(self, **kwargs):
        '''
            Here is where we calculate the dissolved oil.
            We will outline the steps as we go along, but off the top of
            my head:
            - recalculate the partition coeffieieint (K_ow)
              TODO: This requires a molar average of the aromatic components.
            - use VDROP to calculate the shift in the droplet distribution
            - subtract the mass of smallest droplets in our distribution
              that are below a threshold.
        '''
        # pp.pprint(kwargs)
        data = kwargs['data']
        mol_wt = data['mol_wt_components']
        rho = data['density_components']

        # recalculate the partition coefficient (K_ow)
        # - density for the LE should be current weathered density
        print 'mol_wt: ', mol_wt
        print 'rho: ', rho
        K_ow = LeeHuibers.partition_coeff(mol_wt, rho)
        print 'K_ow: ', K_ow[np.where(K_ow != np.nan)]

        diss = np.zeros((len(data['mass'])), dtype=np.float64)
        return diss

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        - sets 'dissolution' in sc.mass_balance
        '''
        if not self.active:
            return

        if sc.num_released == 0:
            return

        # from the waves module
        wave_height = self.waves.get_value(model_time)[0]
        frac_breaking_waves = self.waves.get_value(model_time)[2]
        disp_wave_energy = self.waves.get_value(model_time)[3]

        visc_w = self.waves.water.kinematic_viscosity
        rho_w = self.waves.water.density

        print 'self.array_types:'
        pp.pprint(self.array_types)
        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) == 0:
                # substance does not contain any surface_weathering LEs
                continue

            ka = constants.ka  # oil sticking term

            diss = self.dissolve_oil(time_step=time_step,
                                     data=data,
                                     frac_breaking_waves=frac_breaking_waves,
                                     disp_wave_energy=disp_wave_energy,
                                     wave_height=wave_height,
                                     visc_w=visc_w,
                                     rho_w=rho_w,
                                     ka=ka)

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
