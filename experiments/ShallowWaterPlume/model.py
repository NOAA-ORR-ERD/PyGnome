#!/usr/bin/env python
# Here is a quick mockup of the model that we want to implement
# for the shallow water well blowout.
# this is taken from the ShallowWaterWellBlowoutProgrammingGuide.pdf
# written by Debra Simecek-Beatty

import math
import numpy
np = numpy

from common.unit_conversion import Force, Volume


class DischargeRate(object):
    '''
        It is possible the user may know the total discharge rate (Qt)
        in barrels per day and either the diameter or velocity.
        If so, then the other variable can be calculated using the equation
        Qt = (pi/2) * Do^2 * Uo
        So I guess we can design a three-way solving of this equation based on
        the missing values.  But Do and Uo are the ones we need.
    '''
    def __init__(self, diameter=None, velocity=None, rate=None):
        '''
            Arguments are expected to be metric, although should work with
            **equivalent** english units:
                - diameter(m)
                - velocity(m/s)
                - rate(m^3/s)
        '''
        if (diameter and velocity):
            self.diameter = diameter
            self.velocity = velocity
            self._calculate_rate()
        elif (diameter and rate):
            self.diameter = diameter
            self.rate = rate
            self._calculate_velocity()
        elif (velocity and rate):
            self.velocity = velocity
            self.rate = rate
            self._calculate_diameter()
        else:
            raise "need some combination of diameter, velocity, or rate"

    def _calculate_rate(self):
        'Here we solve Qt = (pi/2) * Do^2 * Uo'
        Do = self.diameter
        Uo = self.velocity
        Qt = (math.pi / 2) * (Do ** 2) * Uo
        self.rate = Qt

    def _calculate_velocity(self):
        'Here we solve Uo = (2/pi) * Qt / Do^2'
        Do = self.diameter
        Qt = self.rate
        Uo = (2 / math.pi) * Qt / (Do ** 2)
        self.velocity = Uo

    def _calculate_diameter(self):
        'Here we solve Do = ((2/pi) * Qt / Uo)^(1/2)'
        Uo = self.velocity
        Qt = self.rate
        Do = ((2 / math.pi) * Qt / Uo) ** (1.0 / 2.0)
        self.diameter = Do

    def __repr__(self):
        ret = '<DischargeRate(diameter={0}, velocity={1}, rate={2})>'
        return ret.format(self.diameter, self.velocity, self.rate)

    def items(self):
        'pack our attributes in a tuple and send them back'
        return (self.diameter, self.velocity, self.rate)


class Model(object):
    gravity = Force.gravity
    adiabatic_index = 1.320  # no heat transfer at 1 at 20 degrees Celsius
    atmospheric_pressure = 1033000  # grams/m^2
    slip_velocity = 0.3  # m/s
    entrainment_coeff = 0.1
    spreading_ratio = 1.0  # aka Schmidt number

    # density of methane at 1 atm, 15 degrees Celsius (kg/m^3)
    gas_density_at_surface = 0.66

    def __init__(self, depth, discharge_rate, oil_density, water_density):
        self.depth = depth
        self.discharge_rate = discharge_rate
        self.oil_density = oil_density
        self.water_density = water_density

        self._set_established_flow_zone_start_depth()
        self._set_water_column_gas_density()

    def _set_established_flow_zone_start_depth(self):
        ''' calculate the starting depth, at the start of the
            'zone of established flow', along the center line
            of the jet path, S
        '''
        self.start_depth = self.depth - (self.discharge_rate.diameter * 6.2)

    def _set_water_column_gas_density(self):
        pass

    def sanity_check(self):
        ''' This is just a check to see if our model is healthy enough
            to be run
            (note: we will add to this as we flesh out our functionality)
        '''
        pass

    def __repr__(self):
        ret = '<Model(depth={0}, discharge_rate={1}, ' \
                     'oil_density={2}, water_density={3})>'
        return ret.format(self.depth, self.discharge_rate,
                          self.oil_density, self.water_density)


def test_discharge_rate():
    print 'testing out our discharge rate calculator...'
    # Do = diameter of round hole (m)
    # Uo = discharge velocity (m/s)

    # calculate our total discharge rate
    holeDiameter, dischargeVelocity, totalDischargeRate = DischargeRate(diameter=0.1, velocity=1.).items()
    print holeDiameter, dischargeVelocity, totalDischargeRate
    dr = DischargeRate(diameter=holeDiameter, velocity=dischargeVelocity)
    print dr
    print 'Our rate in bbl/day: %s' % (Volume.CubicMetersPerSecondToStbsPerDay(dr.rate))
    assert np.isclose(dr.rate, 0.0157079632679)

    dr = DischargeRate(diameter=holeDiameter, rate=0.0157079632679)
    print dr
    assert np.isclose(dr.velocity, 1.)

    dr = DischargeRate(velocity=dischargeVelocity, rate=0.0157079632679)
    print dr
    assert np.isclose(dr.diameter, 0.1)


if __name__ == '__main__':
    test_discharge_rate()

    discharge_rate = DischargeRate(diameter=.09, velocity=20.2)
    model = Model(106, discharge_rate, 893.0, 1027.0)

    print model.depth
    print model.discharge_rate
    print model.oil_density
    print model.water_density
