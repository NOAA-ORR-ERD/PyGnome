'''
model bio degradation process
'''

import numpy as np

from gnome.weatherers import Weatherer
from gnome.array_types import gat

from .core import WeathererSchema
from gnome.environment.waves import WavesSchema


# FIXME: this shouldn't need waves -- though we may want to
#        do something with that in the future.
class BiodegradationSchema(WeathererSchema):
    waves = WavesSchema(
        save=True, update=True, save_reference=True
    )


class Biodegradation(Weatherer):

    _schema = BiodegradationSchema
    _ref_as = 'biodegradation'
    _req_refs = ['waves']

    def __init__(self, waves=None, **kwargs):

        if 'arctic' not in kwargs:
            self.arctic = False  # default is a temperate conditions (>6 deg C)
        else:
            self.arctic = kwargs.pop('arctic')

        self.waves = waves

        super(Biodegradation, self).__init__(**kwargs)

        self.array_types.update({'mass':  gat('mass'),
                                 'mass_components': gat('mass_components'),
                                 'droplet_avg_size': gat('droplet_avg_size'),
                                 'positions': gat('positions'),
                                 'yield_factor': gat('yield_factor'),
                                 })

        #
        # Original bio degradation formula:
        #
        #  m(j, t+1) = m(j, t0) * exp(-K(j) * A(t) / M(t))
        #
        #  where
        #    m(j, t + 1) - mass of pseudocomponent j at time step t + 1
        #    K(j) - biodegradation rate constant for pseudocomponent j
        #    A(t) - droplet surface area at time step t
        #    M(t) - droplet mass at time step t
        #
        # Since
        #
        #  A(t) / M(t) = 6 / (d(t) * ro(t))
        #
        # where
        #   d(t) - droplet diameter at time step t
        #   ro(t) - droplet density at time step t
        #
        # follows this formula for bio degradation:
        #
        #  m(j, t+1) = m(j, t0) * exp(-6 * K(j) / (d(t) * ro(t)))
        #
        # and then interative bio degradation formula:
        #
        #  m(j, t+1) = m(j, t) * exp(6 * K(j) *
        #    (1 / (d(t-1) * ro(t-1)) - 1 / (d(t) * ro(t))))
        #
        # where
        #  d(t-1) - droplet diameter at previous (t-1) time step
        #  ro(t-1) - droplet density at previous (t-1) time step
        #
        # So we will keep previous time step specific surface value
        # (squre meter per kilogram) or yield_factor =  1 / (d * ro)
        #

        self.prev_yield_factor = None

    def prepare_for_model_run(self, sc):
        '''
            Add biodegradation key to mass_balance if it doesn't exist.

            - Assumes all spills have the same type of oil

            - let's only define this the first time

        '''
        if self.on:
            super(Biodegradation, self).prepare_for_model_run(sc)
            sc.mass_balance['bio_degradation'] = 0.0

            self.prev_yield_factor = 0.0

    def initialize_data(self, sc, num_released):
        '''
            Initialize needed weathering data arrays but only if 'on' is True
        '''
        if not self.on:
            return

    # if this isn't doing anything, no need to define it
    # def prepare_for_model_step(self, sc, time_step, model_time):
    #     '''
    #         Set/update arrays used by bio degradation module for this timestep
    #     '''
    #     super(Biodegradation, self).prepare_for_model_step(sc,
    #                                                        time_step,
    #                                                        model_time)

    #     if not self.active:
    #         return

    def bio_degradate_oil(self, K, data, yield_factor):
        '''
            Calculate oil bio degradation

              K - biodegradation rate coefficients are calculated for

                  temperate or arctic environment conditions

              yield_factor - specific surface value (sq meter per kg)

                  yield_factor = 1 / ( d * ro) where

                  d - droplet diameter

                  ro - droplet density

              data['mass_components'] - mass of pseudocomponents
         '''

        mass_biodegradated = (data['mass_components']
                              * np.exp(np.outer(self.prev_yield_factor - yield_factor,
                              6.0 * K)))

        return mass_biodegradated

    # this will have to be updated, SARA is being refactored out of gnome_oil
    def get_K_comp_rates(self, type_and_bp):
        '''
            Get bio degradation rate coefficient based on component
            type and its boiling point for temparate or arctic
            environment conditions. It must take into consideration
            saturates below C30 and aromatics only.

              type_and_bp - a tuple ('type', 'boiling_point')
                - 'type': component type, string
                - 'boiling_point': float value
              self.arctic - flag for arctic conditions
                - TRUE if arctic conditions (below 6 deg C)
                - FALSE if temperate

            Rate units: kg/m^2 per day(!)
        '''

        if type_and_bp[0] == 'Saturates':
            # 722.85 - boiling point for C30 saturate (K)
            if type_and_bp[1] < 722.85:
                return 0.000128807242 if self.arctic else 0.000941386396
            else:
                # zero rate for C30 and above saturates
                return 0.0

        elif type_and_bp[0] == 'Aromatics':
            if type_and_bp[1] < 630.0:
                return 0.000126982603 if self.arctic else 0.000575541103
            else:
                return 0.000021054707 if self.arctic else 0.000084840485
        else:
            # zero rate for other than saturates and aromatics
            return 0.0

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

            # get pseudocomponent boiling point and its type
            if not hasattr(substance, 'boiling_point'):
                raise ValueError("Invalid Substance: no boiling point")
            # # assert hasattr(substance, 'boiling_point')
            # #type_bp = substance._sara[['type','boiling_point']]
            # type_bp = zip(substance.sara_type, substance.boiling_point)

            # print("type_bp:", list(type_bp))

            # raise Exception

            # get bio degradation rate coefficient array for this substance
            K_comp_rates = np.asarray([self.get_K_comp_rates(tbp) for tbp in
                                       zip(substance.sara_type,
                                           substance.boiling_point)])

            # (!) bio degradation rate coefficients are coming per day
            # so we need recalculate ones for the time step interval
            # Fixme: we should probably have the rates in more "Normal"
            #        units -- i.e. per second
            K_comp_rates = K_comp_rates / (60 * 60 * 24) * time_step

            self.previous_yield_factor = data['yield_factor']
            # calculate yield factor (specific surface)
            if np.any(data['droplet_avg_size']):
                data['yield_factor'] = 1.0 / (data['droplet_avg_size'] * data['density'])
            else:
                data['yield_factor'] = 0.0

            # calculate the mass over time step
            bio_deg = self.bio_degradate_oil(K_comp_rates, data, data['yield_factor'])


            # calculate mass balance for bio degradation process - mass loss
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


