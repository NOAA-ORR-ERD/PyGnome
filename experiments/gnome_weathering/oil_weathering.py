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

import numpy as np

class weather_curve:
    def __init__(self, C, H):
        '''
           Simple weathering computation for a three "component" oil

           Each component is a fraction of the total mass and has its own half-life

           (C1, C2, C3, ... Ci) are the fractions of each component (must add up to 1.0)
           (H1, H2, H3, ....Hi) are the half lives of each component (in hours)
        '''
        self.C = np.asarray(C, dtype=np.float32).reshape(-1,)
        self.H = np.asarray(H, dtype=np.float32).reshape(-1,)

        if round(self.C.sum(), 6)  != 1.0: # only six digit, because float32
            raise ValueError('The sum of our components must add up to one. '
                             'These add up to: {0}'.format(self.C.sum()))
        if len(self.H) != len(self.C):
            raise ValueError("There must be the same number of component fractions as half lives")

    def weather(self, M_0, time):
        """
           Compute how much mass is left at time specified, given an initial mass

           returns the mass remaining
        """
        M_0 = np.asarray(M_0, dtype=np.float32).reshape(-1, 1)
        time = np.asarray(time, dtype=np.float32).reshape(-1, 1)
        half = np.float32(0.5)

        self.total_mass = (self.C * M_0) * (half ** (time / self.H))
        return self.total_mass.sum(1)


## Parameters for combined weathering and bio-degradation for "medium crude"
## used for FL Staits TAP analysis
mass_fractions =       [0.25,   0.1, 0.107,    0.2,  0.186,   0.109,    0.048]
combined_half_lives =  [21.0, 422.0,   2.0, 1358.0, 1982.0,  7198.0,  14391.0]
    
OilTypes = {None: None,
            # Medium Crude parameters from OSSM
            'MediumCrude': weather_curve((.22, .26, .52),
                                         (14.4, 48.6, 1e9)),
            "FL_Straits_MediumCrude": weather_curve(mass_fractions, combined_half_lives),
            }


if __name__ == '__main__':
    print 'first, just a single data point...'
    wc = weather_curve((1.0,), (12,))
    print wc.weather((100, 200, 300), (12, 24, 36))
    assert wc.weather(100, 12) == 50.0

    print 'next, we split into thirds...'
    wc = weather_curve( (0.333333, 0.333333, 0.333334),
                        ( 12,  12,  12))
    print 'testing multiple initial masses'
    print wc.weather((100, 200, 300), 12)
    print wc.weather((100, 200, 300), 24)
    print wc.weather((100, 200, 300), 36)
    print 'testing multiple times'
    print wc.weather(100, (12, 24, 36))
