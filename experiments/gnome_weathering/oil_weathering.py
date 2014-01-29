#!/usr/bin/env python
#
# OilWeathering.py
#
# - module to compute "pseudo component" weathering
# - primarily the weather_curve class
#
# Built-in Oil Types are in the OilTypes dict.
#
# NOTE:
#   To compute the half lives of components subject to multiple processes (linear),
#   such as evaporation and bio-degradation, you can use the following formula:
#
#   H_t = 1 / (1 / t_w + 1 / t_b)
#
#

import numpy
np = numpy


class weather_curve:
    def __init__(self, fractions, decay_factors, method="halflife"):
        '''
           Weathering computation for a multiple "component" oil.
           Each component is a fraction of the total mass and is assigned
           its own exponential decay value.

           :param fractions: The fractional values of each component.
           :type fractions: Sequence of items (C1, C2, C3, ....Ci).
                            The sum of all items must add up to 1.0

           :param decay_factors: The decay factors of each component.
           :type decay_factors: Sequence of items (F1, F2, F3, ....Fi).
                                The number of items must match fractions

           :param method: the method in which the decay_factor is to be used.
           :type method: set(['halflife', 'mean-lifetime', 'decay-constant'])
        '''
        self.fractions = np.asarray(fractions, dtype=np.float32).reshape(-1,)
        self.factors = np.asarray(decay_factors, dtype=np.float32).reshape(-1,)

        # only six digit, because float32
        if round(self.fractions.sum(), 6) != 1.0:
            raise ValueError('The sum of our components {0} must add up '
                             'to one'.format(self.fractions.sum()))
        if len(self.factors) != len(self.fractions):
            raise ValueError('There must be the same number of '
                             'component fractions as half lives')

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
FS_mass_fractions      = [0.25,   0.1, 0.107,    0.2,  0.186,  0.109, 0.048]
FS_combined_half_lives = [21.0, 422.0,   2.0, 1358.0, 1982.0, 7198.0, 14391.0]

OilTypes = {None: None,
            # Medium Crude parameters from OSSM
            'MediumCrude': weather_curve((.22, .26, .52),
                                         (14.4, 48.6, 1e9)),
            "FL_Straits_MediumCrude": weather_curve(FS_mass_fractions, FS_combined_half_lives),
            }


if __name__ == '__main__':
    print 'first, just a single data point...'
    wc = weather_curve((1.0,), (12,))
    print wc.weather((100, 200, 300), (12, 24, 36))
    assert wc.weather(100, 12) == 50.0

    print '\nNext, we split into thirds...'
    wc = weather_curve((0.333333, 0.333333, 0.333334),
                       (12, 12, 12))

    print '\nTesting multiple initial masses'
    res = wc.weather((100, 200, 300), 12)
    print res
    assert np.allclose(res, (50., 100., 150.))

    res = wc.weather((100, 200, 300), 24)
    print res
    assert np.allclose(res, (25., 50., 75.))

    print '\nTesting multiple times'
    res = wc.weather(100, (12, 24, 36))
    print res
    assert np.allclose(res, (50., 25., 12.5))

    # Test out our mean lifetime method
    # Basically our function is M_0 * exp(-time/tau)
    #     half-life = tau * ln(2)
    #     tau = half-life / ln(2)
    # So if our half life is 12 hrs,
    #     tau = (12 / np.log(2)) = 17.312340490667562
    print '\nTesting our mean lifetime method'
    wc = weather_curve((0.333333, 0.333333, 0.333334),
                       ((12 / np.log(2)),
                        (12 / np.log(2)),
                        (12 / np.log(2))),
                       method='mean-lifetime'
                       )
    res = wc.weather((100, 200, 300), 12)
    print res
    assert np.allclose(res, (50., 100., 150.))

    # Test out our decay constant method
    # Basically our function is M_0 * exp(-time * lambda)
    #     half-life = ln(2) / lambda
    #     lambda * half-life = ln(2)
    #     lambda = ln(2) / half-life
    # So if our half life is 12 hrs,
    #     lambda = (np.log(2) / 12) = 0.057762265046662105
    print '\nTesting our decay constant method'
    wc = weather_curve((0.333333, 0.333333, 0.333334),
                       ((np.log(2) / 12),
                        (np.log(2) / 12),
                        (np.log(2) / 12)),
                       method='decay-constant'
                       )
    res = wc.weather((100, 200, 300), 12)
    print res
    assert np.allclose(res, (50., 100., 150.))
