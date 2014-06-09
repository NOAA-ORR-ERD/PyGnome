#!/usr/bin/env python
#
# ADIOS3 Shallow Water Plume (Well blowout) implementation
#
# This is coming from the document:
#     ShallowWaterWellBlowoutGuidev1.2.pdf
#     by: Debra Simecek-Beatty
#     version: 1.2 DRAFT
#     June 19, 2013 0915
#
# This document presents a set of equations solved by the Runge-Kutta method
# to simulate the behavior of an oil and gas plume in shallow water.
# These equations are intended for submerged pipeline releases
# and well blowouts in shallow water.
# This computer code will proved a quick estimate of the center line
# vertical velocity of the oil/gas plume and the radius of the plume
# with depth.
#

import math
import numpy
np = numpy

from common.interpolation import IArray
from common.unit_conversion import Time, Volume, Force

from gas_oil_ratio import ShallowDepthGOR
from model import Model, DischargeRate


if __name__ == '__main__':
    # To test our modules are doing the right things,
    # we will follow the example that is laid out in the document:
    # ShallowWaterWellBlowoutGuidev1.2.pdf

    ###
    ### Step 2a - Required data
    ###           User Input
    print 'defining our user inputs...'

    # Do = diameter of round hole (m)
    # Uo = discharge velocity (m/s)
    # Qt = total discharge rate (m^3/s)
    discharge_rate = DischargeRate(diameter=.09, velocity=20.2)
    print 'our discharge rate =', discharge_rate

    # dj = discharge depth(m) in the range 0-150 meters.
    # note: we will be using positive values for depth.
    depth = 106
    print 'depth =', depth

    # Pl = liquid or petroleum product density (kg/m^3)
    # We're using Troll Crude (note: couldn't find this in the oil database.)
    productDensity = 893.0
    print 'product density = ', productDensity

    # Pa = ambient water density at the discharge depth (kg/m^3)
    # note: There are two main factors that make ocean water
    #       more or less dense than about 1027 kg/m3:
    #       - the temperature of the water
    #       - the salinity of the water.
    #       Ocean water gets more dense as temperature goes down.
    #       So, the colder the water, the more dense it is.
    #       Increasing salinity also increases the density of sea water.
    # note: Pressure can be a factor in the density of water at
    #       extreme depths, but probably not at 150 meters or less.
    #       To calculate this, we would need to use the bulk modulus of
    #       water = 2.2e9 Pa
    # note: We probably need a calculation for this, but right here we will
    #       choose an easy arbitrary value.
    ambientWaterDensityAtDepth = 1027.0

    ###
    ### Step 2b - Required data
    ###           Model Parameters
    model = Model(depth, discharge_rate,
                  productDensity, ambientWaterDensityAtDepth)
    print 'model =', model
    print 'model.adiabatic_index = ', model.adiabatic_index
    print 'model.atmospheric_pressure = ', model.atmospheric_pressure
    print 'model.slip_velocity = ', model.slip_velocity
    print 'model.entrainment_coeff = ', model.entrainment_coeff
    print 'model.spreading_ratio = ', model.spreading_ratio
    print 'model.gas_density_at_surface = ', model.gas_density_at_surface

    ###
    ### Step 3 - Load and read the density gradient file for testing purposes
    ###
    density_gradient = IArray(((0,  1026500),
                               (10, 1026600),
                               (20, 1026700),
                               (45, 1027000),
                               (80, 1027200),
                               (120, 1027300),
                               ),
                              method='leftmost')
    print density_gradient.points
    print 'ambient water density at 100m =', density_gradient[100]
    print 'ambient water density at 140m =', density_gradient[140]

    ###
    ### Step 4 - calculate the starting depth, at the start of the
    ###          'zone of established flow', along the center line
    ###          of the jet path, S
    ###        - This should be done when our Model class is initialized.
    print 'model.start_depth = ', model.start_depth



    ###
    ### Step 5 - calculate gas density through the water column
    ###

    ###
    ### Step 6 - calculate the fraction of released product that is oil and gas
    ### at the discharge depth

    # GORsurface = Surface Gas/Oil ratio (m^3/m^3)
    # note: I think we are calculating this in a previous module.
    #       for now we just choose a number.
    gasOilRatioSurface = 67

    # Pdj = the discharge pressure at the release depth, not the surface
    #       (N/m^2 or Pascals)
    #       This is required if GORsurface is input.
    dischargePressureAtPipe = 6827477  # about 990psi

    # so we have GORsurface and Pdj...let's calculate:
    # - frel (release volume fraction)
    # - fGOR (GOR reduction factor)
    gor_at_depth = ShallowDepthGOR(sea_level_pressure=0)
    res = gor_at_depth.Calculate(depth=depth,
                                 gas_oil_ratio=gasOilRatioSurface,
                                 source_pressure=dischargePressureAtPipe,
                                 output_metric=True)
    print res
    releaseVolumeFraction = res.max_release_fraction
    gorReductionFactor = res.gor_reduction_factor

    # I believe that:
    # - Qi is the oil discharge rate
    # - Qgas is the gas discharge rate
    oilDischargeRate = totalDischargeRate * releaseVolumeFraction * gorReductionFactor
    gasDischargeRate = totalDischargeRate - oilDischargeRate

    ### Step 7 calculate the initial oil conditions

    # Po = density of air at the surface (kg/m^3)
    # note: air was used in the experiments, not methane.
    # note: At sea level and at 15 degrees Celsius , the ISA density of air is
    #       1.275 kg/m3, so I guess this is close.
    # note: the density of Methane at NTP is 0.668 kg/m^3, quite a big
    #       contrast to ordinary air.  Could this be a more reasonable
    #       assumption for the density of gas in the gas/oil mixture?
    airDensitySurface = 1.269

    # n = adiabatic index for dry air
    # note: at the top of the document, the description shows a value of 1.320,
    #       but the example value used was 1.4
    # note: Wikipedia (I know, is this reliable?) has a table that shows
    #       1.4 to be pretty accurate from 0-200 degrees Celsius
    adiabaticIndex = 1.4

    # Pa = Atmospheric Pressure (kg/m^2)
    # note: the document says that atmospheric pressure is 1033000 g/m^2.
    #       Converting to kg/m^2 gives us 1033.0 kg/m^2
    # note: using the source
    #       http://en.wikipedia.org/wiki/Pascal_(unit)#cite_note-4
    #       - If we are using Technical Atmosphere, the standard unit is
    #         1 kilogram-force/cm^2, which is 10000.0 kilogram-force/m^2
    #       - If we are using Standard Atmosphere, the unit is
    #         1.0332 kilogram-force/cm^2, which is 10332.0 kilogram-force/m^2
    # note: Based on the above, are we off by a factor of 10 here?
    atmosphericPressure = 1033.0

    # Qgas = Discharge Volume Flux (m^3/s)
    dischargeVolumeFlux = 0.1117
