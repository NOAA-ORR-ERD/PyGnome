#!/usr/bin/env python

"""
the waves environment object.

Computes the wave hight and percent wave breaking

Uses the same approach as ADIOS 2

(code ported from old MATLAB prototype code)

"""
from __future__ import division
from math import sqrt
from gnome import environment
from gnome.utilities import serializable

g = environment.constants['gravity'] # the graviational contant.


class Waves(environment.Environment, serializable.Serializable):
    """
    class to compute the wave height for a time series

    At the moment, it only does a single point, non spatially
    variable, but may be extended in the future
    """
    def __init__(self, wind, fetch=None):
        """
        :param wind: A wind object to get the wind speed
        :type wind: a Wind type, or equivelent

        :param fetch: the limiting fetch for the wave generation
        :param type: floating point number, units of meters.
        """

        self.fetch = fetch
        self.wind = wind

    def get_value(self, time):
        """
        return the rms wave height, peak period and percent wave breaking
        at a given time. Does not currently support location-variable waves.

        :param time: the time you want the wave data for 
        :type time: datetime.datetime object

        :returns: wave_height, peak_period, percent_breaking

        wave_height is in units of meters, percent_breaking is unitless percent.
        """

        U = self.time.get_value(self, time)

        data = self.get_timeseries(time, 'm/s', 'r-theta')
        return tuple(data[0]['value'])

    def compute_H(self, U):
        """
        compute the wave height

        :param U: wind speed
        :type U: floating point number in m/s units

        :returns Hrms: RMS wave height in meters
        """

        ## wind stress factor
        ## Transition at U = 4.433049525859078 for linear scale with wind speed.
        ##   4.433049525859078 is where the solutions match
        ws = 0.71*U**1.23 if U < 4.433049525859078 else U # wind stress factor

        # 2268*ws**2 is limit of fetch limited case.
        if (self.fetch is not None) and (self.fetch < 2268*ws**2): ## fetch limited case
            H = 0.0016*sqrt(self.fetch/g)*ws
        else: # fetch unlimited
            H = 0.243*ws*ws/g

        Hrms = 0.707*H
    
        return Hrms

    def comp_psuedo_wind(self, H):
        """
        Compute the wind speed to use for the whitecap fraction

        Used if the wave height is specified.

        Unlimited fetch is assumed: this is the reverse of compute_H

        :param H: given wave height.
        """

        ##U_h = 2.0286*g*sqrt(H/g) # Bill's version
        U_h = sqrt(g * H / 0.243)
        if U_h < 4.433049525859078: # check if low wind case
            print "low wind case"
            U_h = (U_h/0.71)**0.813008
        else:
            print "high wind case"
        return U_h

    def comp_whitecap_fraction(self, U):
        """
        compute the white capping fraction

        This and wave height drives dispersion
        """

        ## fixme -- smooth this out toward zero?
        ## disontinuity at 3 m/s at about 1.5%
        if U < 3: # m/s
            fw = 0.0
        else:
            fw = 0.5 * (0.01*U + 0.01) # Ding and Farmer (JPO 1994)

        if fw > 1.0: # only with U > 200m/s!
            fw = 1.0
        return fw

    def comp_period(self, U):
        """
        Compute the mean wave period 
        """
        # wind stress factor
        ## fixme: check for discontinuity at large fetch..
        ##        Is this s bit low??? 32 m/s -> T=15.7 s
        ws = U * 0.71 * U**1.23 ## fixme -- linear for large windspeed?
        if (self.fetch is None) or (self.fetch >= 2268*ws**2): # fetch unlimited
            T = 0.83*ws
        else:
            T = 0.0624*(self.fetch*ws)**0.3333333333 # eq 3-34 (SPM?)
        return T












# wind.get_timeseries(self, datetime=None, units=None, format='r-theta')

