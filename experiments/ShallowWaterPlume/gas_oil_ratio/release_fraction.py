#!/usr/bin/env python
# The Pipeline Oil Spill Volume Estimator (OCS Study MMS 2002-033)
# has a section for calculating the "Maximum released volume fraction".
# The method uses a series of tables in the calculations.
# This technique was chosen as it appears to be a reliable
# reference.
#
# References:
# OCS Study MMS 2002-033, "Pipeline Oil Spill Volume Estimator,
# Pocket Guide"

from collections import namedtuple

from common.interpolation import IArray


class ReleaseFraction(object):
    '''
       References:
           OCS Study MMS 2002-033, "Pipeline Oil Spill Volume Estimator,
           Pocket Guide"

       Table 1.3, page 11
       Maximum released volume fraction, [(Prel, (frel, Gmax)),
                                          ...
                                          ]
    '''
    table_results = namedtuple('ReleaseFractionResults',
                               ''' f_rel,
                                   g_max,
                               ''')

    points = [(1, (0.0, None)),
              (1.1, (0.08, 140)),
              (1.2, (0.17, 225)),
              (1.5, (0.3, 337)),
              (2, (0.4, 449)),
              (3, (0.47, 505)),
              (4, (0.5, 560)),
              (5, (0.55, 505)),
              (10, (0.64, 337)),
              (20, (0.71, 168)),
              (30, (0.74, 140)),
              (50, (0.76, 112)),
              (200, (0.77, 112)),
              ]

    def __init__(self):
        # we defined our table data above so that it is more easily
        # readable.  Here we organize it into a couple of arrays.
        self.f_rel = IArray([(p[0], p[1][0]) for p in self.points], method='leftmost')
        self.g_max = IArray([(p[0], p[1][1]) for p in self.points], method='leftmost')

    def __getitem__(self, p_rel):
        return self.table_results(self.f_rel[p_rel],
                                  self.g_max[p_rel])


if __name__ == '__main__':
    print '\nTesting out our packaged release fraction array object.'

    rfa = ReleaseFraction()

    print '\nFirst, we test our limits.'
    p_rel = -1
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.0, None)
    assert rfa[p_rel].f_rel == 0.0
    assert rfa[p_rel].g_max == None

    p_rel = 0
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.0, None)
    assert rfa[p_rel].f_rel == 0.0
    assert rfa[p_rel].g_max == None

    p_rel = 1
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.0, None)
    assert rfa[p_rel].f_rel == 0.0
    assert rfa[p_rel].g_max == None

    p_rel = 200
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.77, 112)
    assert rfa[p_rel].f_rel == 0.77
    assert rfa[p_rel].g_max == 112

    p_rel = 300
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.77, 112)
    assert rfa[p_rel].f_rel == 0.77
    assert rfa[p_rel].g_max == 112

    print '\nNext, we test our interpolation behavior.'
    p_rel = 4
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.5, 560)
    assert rfa[p_rel].f_rel == 0.5
    assert rfa[p_rel].g_max == 560

    p_rel = 4.5
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.5, 560)
    assert rfa[p_rel].f_rel == 0.5
    assert rfa[p_rel].g_max == 560

    p_rel = 4.9999
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.5, 560)
    assert rfa[p_rel].f_rel == 0.5
    assert rfa[p_rel].g_max == 560

    p_rel = 5
    print p_rel, rfa[p_rel],
    print '\tPrel = %s, frel = %s, Gmax = %s' % (p_rel,
                                                 rfa[p_rel].f_rel,
                                                 rfa[p_rel].g_max)
    assert rfa[p_rel] == (0.55, 505)
    assert rfa[p_rel].f_rel == 0.55
    assert rfa[p_rel].g_max == 505
