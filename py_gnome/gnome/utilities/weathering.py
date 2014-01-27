#!/usr/bin/env python
#
# weathering.py
#
# - module to compute the weathering of an oil that contains one or more
#   "pseudo components".
#
# Built-in Oil Types are in the OilTypes dict.
#
# NOTE:
#   Right now we will support the three most common exponential decay methods.
#   These are:
#   - half life
#     - the amount of time required for a quantity to fall to half its value
#     - Basically our calculation is M_0 * (half ** (time / t_half))
#   - mean lifetime (tau)
#     - Average length of time that an element remains in the set.
#     - This is probably not as popular as half life, but we should cover it
#       just in case.
#     - Basically our calculation is M_0 * np.exp(-time / tau)
#       half-life = tau * np.log(2)
#       tau = half-life / np.log(2)
#   - decay constant (lambda)
#     - Exponential positive constant value which solves the differential
#       rate of change for our decaying quantity.
#     - This is probably not as popular as half life, but we should cover it
#       just in case.
#     - Basically our calculation is M_0 * np.exp(-time * lambda)
#       half-life = np.log(2) / lambda
#       lambda * half-life = np.log(2)
#       lambda = np.log(2) / half-life


from collections import namedtuple

import numpy
np = numpy


WeatheringComponent = namedtuple('WeatheringComponent',
                                 ''' fraction,
                                     factor,
                                 ''')


class weather_curve:
    '''
        This is an object designed to compute the weathering of an oil
        that contains one or more "pseudo components".
        - Each pseudo component is assumed to be a known substance that
          has a known rate of decay that can be expressed using an
          exponential decay function.
        - Each pseudo component has a quantitative value that represents a
          fraction of a total mass that adds up to 1.0.  Thus, we require that
          the sum of the component mass fractions adhere to this constraint.
        - It is assumed that all components have exponential decay factors
          that are solvable using a common functional method.
        - Right now we support the three most common exponential decay methods.
          These are:
          - half life.  This is the amount of time required for a quantity to
            fall to half its value
          - mean lifetime.  This is the average length of time that an element
            remains in the set.
          - decay constant.  Positive constant value which solves the
            differential rate of change for our decaying quantity.
    '''
    def __init__(self, components, method="halflife"):
        '''
           :param components: The properties of each component.
           :type components: Sequence of WeatheringComponents
                             (WC1, WC2, WC3, ....WCi).
                             The sum of the component fractional values must
                             add up to 1.0

           :param method: the method in which the decay_factor is to be used.
           :type method: set({'halflife', 'mean-lifetime', 'decay-constant'})
        '''
        fractions, factors = zip(*components)
        self.fractions = np.asarray(fractions, dtype=np.float32).reshape(-1,)
        self.factors = np.asarray(factors, dtype=np.float32).reshape(-1,)

        # only six digit, because float32
        if round(self.fractions.sum(), 6) != 1.0:
            raise ValueError('The sum of our components {0} must add up '
                             'to one'.format(self.fractions.sum()))

        methods = {'halflife': self._halflife,
                   'mean-lifetime': self._mean_lifetime,
                   'decay-constant': self._decay_constant,
                   }
        self.method = methods[method]

    def _xform_inputs(self, M_0, time):
        '''
           make sure our mass and time arguments are a good fit
           for our calculations
        '''
        M_0 = np.asarray(M_0, dtype=np.float32).reshape(-1, 1)
        time = np.asarray(time, dtype=np.float32).reshape(-1, 1)
        return M_0, time

    def _halflife(self, M_0, time):
        'Assumes our factors are half-life values'
        half = np.float32(0.5)

        self.total_mass = (self.fractions * M_0) * (half ** (time / self.factors))
        return self.total_mass.sum(1)

    def _mean_lifetime(self, M_0, time):
        'Assumes our factors are mean lifetime values (tau)'
        self.total_mass = (self.fractions * M_0) * np.exp(-time / self.factors)
        return self.total_mass.sum(1)

    def _decay_constant(self, M_0, time):
        'Assumes our factors are decay constant values'
        self.total_mass = (self.fractions * M_0) * np.exp(-time * self.factors)
        return self.total_mass.sum(1)

    def weather(self, M_0, time):
        'Compute the decayed mass at time specified'
        M_0, time = self._xform_inputs(M_0, time)
        return self.method(M_0, time)


## Parameters for combined weathering and bio-degradation for "medium crude"
## used for FL Staits TAP analysis
mass_fractions = [0.25, 0.1, 0.107, 0.2, 0.186, 0.109, 0.048]
combined_half_lives = [21.0, 422.0, 2.0, 1358.0, 1982.0, 7198.0, 14391.0]

OilTypes = {None: None,
            # Medium Crude parameters from OSSM
            'MediumCrude': weather_curve(((.22, 14.4),
                                          (.26, 48.6),
                                          (.52, 1e9)),
                                         ),
            "FL_Straits_MediumCrude": weather_curve(zip(mass_fractions,
                                                        combined_half_lives)),
            }
