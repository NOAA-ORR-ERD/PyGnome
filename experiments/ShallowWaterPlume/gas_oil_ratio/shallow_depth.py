#!/usr/bin/env python
# Simplified calculation to convert the surface GOR to the GOR at a
# specified depth.
# Originally developed in Matlab by:
#     Debra Simecek-Beatty, Physical Scientist
#     National Oceanic and Atmosperic Administration
#     7600 Sand Point Way NE
#     Seattle, WA 98115
# Ported to python by:
#     James L. Makela
#     NOAA Affiliate


# The total released volume of oil is strongly connected to the GOR or
# gas-oil-ratio.  The GOR is the ratio of the volume of gas that comes out
# of solution in standard conditions.  As the oil is brought to the
# surface, natural gas comes out of the solution.  The user provides
# the GOR which is typically valid at the surface.  The GOR is then
# adjusted for the sea floor (or the depth of the release).  The GOR is
# typically reported as standard cubic feet per standard oil barrel.
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

from collections import namedtuple

import numpy
np = numpy

from hazpy.unit_conversion import convert
from common.unit_conversion import Volume, Force

from release_fraction import ReleaseFraction
from gor_reduction_factor import GasOilRatioReductionFactor


class ShallowDepthGOR(object):
    ''' Class for calculating the Gas/Oil ratio
        at depths of 150 meters or less
        The SINTEF engineering document uses English units
        for calculation, so internally that is what we use
        as well.
    '''
    release_fraction_lu = ReleaseFraction()
    gor_reduction_factor_lu = GasOilRatioReductionFactor()
    gor_results = namedtuple('GasOilRatioResults',
                             ''' source_pressure,
                                 ambient_pressure_at_depth,
                                 relative_pressure_delta,
                                 max_release_fraction,
                                 max_release_occurrence,
                                 gor_reduction_factor
                             ''' )

    def __init__(self,
                 sea_level_pressure=14.7,
                 metric_inputs=True):
        # most of the time we will want to use the default
        # constant for pressure at sea level in psi units,
        # but there are some times we would like
        # to use 0 or a slightly adjusted value.
        # TODO: For now, we always accept psi units, but
        #       maybe we want to accept metric units based on
        #       the state of our metric_units flag.
        self.ambient_pressure_at_sea_level = sea_level_pressure

        # internally, our calculations are english units,
        # but we want to be able to use metric units for
        # our inputs.
        self.metric_inputs = metric_inputs

    def Calculate(self, depth, gas_oil_ratio,
                  oil_jet_velocity=None, oil_jet_density=None,
                  source_pressure=None,
                  output_metric=False):
        if oil_jet_velocity and oil_jet_density:
            # N/m^2  or Pa units (Pascals)
            # equavelent to 950 psi
            source_pressure = (oil_jet_density * (oil_jet_velocity ** 2)) / 2
        elif not source_pressure:
            raise ValueError('need either '
                             'oil_jet_velocity and oil_jet_density, '
                             'or source_pressure')

        if self.metric_inputs:
            depth = convert('Length', 'meter', 'foot', depth)
            gas_oil_ratio = Volume.CubicMeterRatioToScfPerStb(gas_oil_ratio)
            source_pressure = Force.PascalsToPsi(source_pressure)

        # Start-Equation 1.5, page 8
        # Calculating ambient pressure outside leak at depth in psi.
        # We will go off the document, but here are some considerations
        # regarding this calculation:
        # - The ambient atmospheric pressure at sea level is not constant.
        #   It varies with the weather, but averages around 100 kPa
        #   One bar is 100kPa or approximately ambient pressure at sea level
        # - One atmosphere(atm) is also approximately the ambient pressure
        #   at sea level and is equal to 14.7 psi or 1.01325 bar
        # - Ambient water pressure increases linearly with depth.
        #   Roughly, each 10 meters (33 ft) of depth adds another bar
        #   to the ambient pressure.  Assuming the density of sea water
        #   to be 1025 kg/m^3 (in fact it is slightly variable),
        #   pressure increases by 1 atm with each 10 m of depth
        ambient_pressure_at_depth = self.ambient_pressure_at_sea_level + (0.446533 * depth);

        # Start-Equation 1.4, page 8
        # The relative pressure, deltaPrel, difference over the leak point is
        relative_pressure_delta = source_pressure / ambient_pressure_at_depth

        # Start- Table 1.3, page 11
        # Maximum released volume fraction, frel
        max_release_fraction, max_release_occurrence = self.release_fraction_lu[relative_pressure_delta]

        # Start-Section 1.3.5
        #
        # Table 1.4 GOR reduction factors, page 11
        gor_reduction_factor = self.gor_reduction_factor_lu.get_gas_oil_reduction_factor(gas_oil_ratio,
                                                                                         max_release_occurrence)

        if output_metric:
            source_pressure = Force.PsiToPascals(source_pressure)
            ambient_pressure_at_depth = Force.PsiToPascals(ambient_pressure_at_depth)
            max_release_occurrence = Volume.ScfPerStbToCubicMeterRatio(max_release_occurrence)

        return self.gor_results(source_pressure,
                                ambient_pressure_at_depth,
                                relative_pressure_delta,
                                max_release_fraction,
                                max_release_occurrence,
                                gor_reduction_factor)


if __name__ == '__main__':
    #
    # OK, here is where we test our Gas/Oil Ratio calculation class
    #
    # For our initial values, we will choose the ones that Debra came up with
    # in her matlab calculations (filename: GORatDepth.m)
    depth = 30  # depth in meters

    # gas/oil ratio at sealevel (S m^3/ S m^3)
    # equivalant to 450 Scf/Stb
    # (standard cubic feet/ stock tank barrel)
    gasOilRatio = 2530

    # Velocity of the jet in meter per sec (Uo)
    # - 117.1 m/s is about 950 PSI
    # - 12.5 m/s is about 10 PSI
    oilJetVelocity = 117.1

    oilJetDensity = 995.81  # Jet density (kg/m^3) (Rhoi)

    #
    # let's calculate assuming sealevel pressure has been normalized to 0.
    #
    gor_at_depth = ShallowDepthGOR(sea_level_pressure=0)

    res = gor_at_depth.Calculate(depth, gasOilRatio, oilJetVelocity,
                                 oilJetDensity)
    print res

    # these values are very close to the results that Debra came up with in her
    # matlab calculations
    assert np.isclose(res.source_pressure, 990.24387120473)
    assert np.isclose(res.ambient_pressure_at_depth, 43.9500984251)
    assert np.isclose(res.relative_pressure_delta, 22.5310956445)

    assert np.isclose(res.max_release_fraction, 0.71)
    assert np.isclose(res.max_release_occurrence, 168)

    assert np.isclose(res.gor_reduction_factor, 0.26)

    #
    # now let's calculate again taking atmospheric pressure into account.
    #
    gor_at_depth = ShallowDepthGOR()

    res = gor_at_depth.Calculate(depth, gasOilRatio, oilJetVelocity,
                                 oilJetDensity)
    print res

    assert np.isclose(res.source_pressure, 990.24387120473)
    assert np.isclose(res.ambient_pressure_at_depth, 58.6500984251)
    assert np.isclose(res.relative_pressure_delta, 16.8839251389)

    assert np.isclose(res.max_release_fraction, 0.64)
    assert np.isclose(res.max_release_occurrence, 337)

    assert np.isclose(res.gor_reduction_factor, 0.26)
