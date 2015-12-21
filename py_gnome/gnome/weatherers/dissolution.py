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
                               partition_coeff)

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
                                 'partition_coeff': partition_coeff
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

    def dissolve_oil(self, **kwargs):
        '''
            Here is where we calculate the dissolved oil.
            We will outline the steps as we go along, but off the top of
            my head:
            - recalculate the partition coefficient (K_ow)
              TODO: This requires a molar average of the aromatic components.
            - use VDROP to calculate the shift in the droplet distribution
            - subtract the mass of smallest droplets in our distribution
              that are below a threshold.
        '''
        data = kwargs['data']
        fmass = data['mass_components']

        substance = kwargs['substance']
        arom_mask = substance._sara['type'] == 'Aromatics'
        mol_wt = substance.molecular_weight
        rho = substance.component_density
        assert mol_wt.shape == rho.shape

        # calculate the partition coefficient (K_ow) for all aromatics.
        # K_ow for non-aromatics should be masked to 0.0
        K_ow_comp = arom_mask * LeeHuibers.partition_coeff(mol_wt, rho)

        for idx, m in enumerate(fmass):
            K_ow = (np.sum(m * K_ow_comp / mol_wt) /
                    np.sum(m / mol_wt))

            data['partition_coeff'][idx] = K_ow

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
                                     substance=substance,
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
