#!/usr/bin/env python
#
# The Pipeline Oil Spill Volume Estimator (OCS Study MMS 2002-033) has a
# section for calculating the "GOR reduction factor".  The method uses a
# series of tables in the calculations.  This technique was chosen to
# calculate the "GOR reduction factor" as it appears to be a reliable
# reference.
#
# References:
# OCS Study MMS 2002-033, "Pipeline Oil Spill Volume Estimator,
# Pocket Guide"

import numpy
np = numpy

from common.interpolation import IArray


class GasOilRatioReductionFactor(object):
    '''
        References:
            OCS Study MMS 2002-033, "Pipeline Oil Spill Volume Estimator,
            Pocket Guide"

        Table 1.4 GOR reduction factors, page 11
        TODO: The table is not clear if we accept values outside our bounds.
              For now we will be unbounded since that is what is shown
              in Debra's example work.
        The section of the table labeled 'will not occur' is going to
        return a value of None
    '''
    array = IArray([  # (-1, None),
                    (0, 1),
                    (225, 0.98),
                    (280, 0.97),
                    (340, 0.95),
                    (420, 0.9),
                    (560, 0.85),
                    (1100, 0.82),
                    (1700, 0.63),
                    (2800, 0.43),
                    (5600, 0.26),
                    #(11300, None),
                   ],
                   method='rightmost')

    def __init__(self):
        pass

    def _calculate_reduction_factor(self, gor, g_max):
        if gor < 0 or gor >= 560:
            return None
        else:
            return float(gor) / g_max

    def get_gas_oil_reduction_factor(self, gor, g_max):
        if gor < g_max:
            return self._calculate_reduction_factor(gor, g_max)
        else:
            return self.array[gor]


if __name__ == '__main__':
    print '\nTesting out our packaged GOR Reduction Factor array object.'

    gor_reduction_factor_lu = GasOilRatioReductionFactor()

    print '\nFirst, we test our calculation limits (GOR < Gmax)'
    gor, g_max = -1, 1
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == None

    gor, g_max = 0, 1
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 0.0

    gor, g_max = 0, 15000
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 0.0

    gor, g_max = 500, 1000
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 0.5

    gor, g_max = 559, 560
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    np.testing.assert_almost_equal(fgor, 0.998214285714, decimal=9)

    gor, g_max = 560, 1000
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == None

    gor, g_max = 1000, 15000
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == None

    print '\nNext, we test our table limits (GOR > Gmax)'
    gor, g_max = -1, -3
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    # depending on whether our table is bounded or unbounded,
    # we will either get a None value or a value of 1
    #assert fgor == None
    assert fgor == 1

    gor, g_max = 0, 0
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 1

    gor, g_max = 420, 400
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 0.9

    gor, g_max = 5600, 560
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 0.26

    gor, g_max = 11299, 560
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    assert fgor == 0.26

    gor, g_max = 11300, 560
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    # depending on whether our table is bounded or unbounded,
    # we will either get a None value or a value of 0.26
    #assert fgor == None
    assert fgor == 0.26

    gor, g_max = 15000, 560
    fgor = gor_reduction_factor_lu.get_gas_oil_reduction_factor(gor, g_max)
    print 'GOR = %s, Gmax = %s, fgor = %s' % (gor, g_max, fgor)
    # depending on whether our table is bounded or unbounded,
    # we will either get a None value or a value of 0.26
    #assert fgor == None
    assert fgor == 0.26
