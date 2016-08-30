'''
model biodegradation process
'''

from __future__ import division
import copy

import numpy as np

import gnome  # required by deserialize
from gnome.utilities.serializable import Serializable, Field

from .core import WeathererSchema
from gnome.weatherers import Weatherer

from gnome.array_types import (mass,
                               mass_components,
                               droplet_avg_size)

from math import exp, pi

class Biodegradation(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('waves', save=True, update=True, save_reference=True)]

    _schema = WeathererSchema

    def __init__(self, waves=None, **kwargs):

        if 'arctic' not in kwargs:
            self.arctic = False # default is a temperate conditions (>6 deg C)
        else:
            self.arctic = kwargs.pop('arctic')

        self.waves = waves

        super(Biodegradation, self).__init__(**kwargs)

        self.array_types.update({'mass':  mass,
                                 'mass_components': mass_components,
                                 'droplet_avg_size': droplet_avg_size
                                 })

        # we need to keep previous time step mass ratio (d**2 / M)
        # for interative bio degradation formula:
        #
        #    m(j,n+1) = m(j,n) * exp(-pi * K(j) * 
        #               (d(n) ** 2 / M(n) - d(n-1)**2 / M(n-1)))
        # 
        #  where 
        #    m(j, t) - mass of pseudocomponent j at time t
        #    K(j) - biodegradation rate constant for pseudocomponent j
        #    d(t) - droplet size diameter at time t
        #    M(t) - total mass of oil at time t
        self.prev_mass_ratio = None


    def prepare_for_model_run(self, sc):
        '''
            Add biodegradation key to mass_balance if it doesn't exist.
            - Assumes all spills have the same type of oil
            - let's only define this the first time
        '''
        if self.on:
            super(Biodegradation, self).prepare_for_model_run(sc)
            sc.mass_balance['bio_degradation'] = 0.0

            self.prev_mass_ratio = 0.0


    def initialize_data(self, sc, num_released):
        '''
            Initialize needed weathering data arrays but only if 'on' is True
        '''
        if not self.on:
            return

        pass


    def initialize_K_comp_rates(self, sc, num_released):
        '''
            Initialize/calculate component bio degradation rate 
            coefficient for each substance 
        '''

        k_comp_rates = [] # work with list for the speed (append()) and simplicity

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) is 0:
                # data does not contain any surface_weathering LEs
                continue

            # we are going to calculate bio degradation rate coefficients
            # (K_comp_rates) just for saturates below C30 and aromatics 
            # components - other ones are masked to 0.0

            assert 'boiling_point' in substance._sara.dtype.names
            type_bp = substance._sara[['type','boiling_point']]
  
            #for _ in range(len(data['mass_components'])):  # use num_released instead???
            k_comp_rates.append(map(self.get_K_comp_rates, type_bp))

        # convert list to numpy array
        self.K_comp_rates = np.asarray(k_comp_rates)

        # TODO asserting
        #assert self.K_comp_rates[0].shape == sc['mass_components'][0].shape
        #assert len(self.K_comp_rates) == len(sc['mass'])
        #assert len(self.K_comp_rates[:,0]) == len(sc.get_substances())


    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
            Set/update arrays used by bio degradation module for this timestep
        '''
        super(Biodegradation, self).prepare_for_model_step(sc, 
                                                           time_step, 
                                                           model_time)

        if not self.active:
            return


    def bio_degradate_oil(self, K, data, mass_ratio):
        '''
            Calculate oil bio degradation
            1. K - biodegradation rate coefficients are calculated for temperate or 
            arctic emvironment conditions
            2. It uses pseudo component boiling point to select rate constant
            3. It must take into consideration saturates below C30 and aromatics only.
            4. Droplet distribution per LE should be calculated by the natural
            dispersion process and saved in the data arrays before the 
            biodegradation weathering process.
         '''

        comp_masses = data['mass_components']

        # 
        mass_biodegradated = (comp_masses *
                              np.exp(np.outer(mass_ratio - self.prev_mass_ratio,
                              -pi * K)))

        return mass_biodegradated


    def get_K_comp_rates(self, type_and_bp):
        '''
            Get bio degradation rate coefficient based on component type and 
            its boiling point for temparate or arctic environment conditions
            :param type_and_bp - a tuple ('type', 'boiling_point')
                - 'type': component type, string
                - 'boiling_point': float value
            :param boolean arctic = False - flag for arctic conditions (below 6 deg C)
        '''

        if type_and_bp[0] == 'Saturates':
            if type_and_bp[1] < 722.85:     # 722.85 - boiling point for C30 saturate (K)
                return 0.128807242 if self.arctic else 0.941386396
            else:
                return 0.0                  # zero rate for C30 and above saturates

        elif type_and_bp[0] == 'Aromatics':
            if type_and_bp[1] < 630.0:      # 
                return 0.126982603 if self.arctic else 0.575541103
            else:
                return 0.021054707 if self.arctic else 0.084840485
        else:
            return 0.0                      # zero rate for ather than saturates and aromatics
        

    def weather_elements(self, sc, time_step, model_time):
        '''
            weather elements over time_step
        '''
        if not self.active:
            return

        if sc.num_released == 0:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) is 0:
                # data does not contain any surface_weathering LEs
                continue

            # get the substance index
            indx = sc._substances_spills.substances.index(substance)

            # get bio degradation rate coefficient array for this substance
            # we are going to calculate bio degradation rate coefficients
            # (K_comp_rates) just for saturates below C30 and aromatics 
            # components - other ones are masked to 0.0

            assert 'boiling_point' in substance._sara.dtype.names
            type_bp = substance._sara[['type','boiling_point']]
            K_comp_rates = np.asarray(map(self.get_K_comp_rates, type_bp))

            # calculate the mass over time step
            mass_ratio = data['droplet_avg_size'] ** 2 / data['mass_components'].sum(1)
            bio_deg = self.bio_degradate_oil(K_comp_rates, data, mass_ratio)
            # update mass ration for the next time step
            self.prev_mass_ratio = mass_ratio

            # calculate mass ballance for bio degradation process - mass loss
            sc.mass_balance['bio_degradation'] += data['mass'].sum() - bio_deg.sum()

            # update masses
            data['mass_components'] = bio_deg
            data['mass'] = data['mass_components'].sum(1)

            # log bio degradated amount
            self.logger.debug('{0} Amount bio degradated for {1}: {2}'
                              .format(self._pid,
                                      substance.name,
                                      sc.mass_balance['bio_degradation']))

        sc.update_from_fatedataview()


    def serialize(self, json_='webapi'):

        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        serial = schema.serialize(toserial)

        if json_ == 'webapi':
            if self.waves:
                serial['waves'] = self.waves.serialize(json_)

        return serial

    @classmethod
    def deserialize(cls, json_):
 
        if not cls.is_sparse(json_):
            schema = cls._schema()
            dict_ = schema.deserialize(json_)

            if 'waves' in json_:
                obj = json_['waves']['obj_type']
                dict_['waves'] = (eval(obj).deserialize(json_['waves']))

            return dict_
        else:
            return json_
