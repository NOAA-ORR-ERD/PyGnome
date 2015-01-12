#!/usr/bin/env python

"""
the waves environment object.

Computes the wave hight and percent wave breaking

Uses the same approach as ADIOS 2

(code ported from old MATLAB prototype code)

"""
from __future__ import division
from math import sqrt

import copy

#from gnome import environment
from gnome.utilities import serializable
from gnome.utilities.serializable import Field
from gnome.persist import base_schema
from .environment import Environment
from wind import WindSchema
from .environment import WaterSchema

#g = environment.constants['gravity'] # the gravitational constant.
g = 9.80665 # the gravitational constant.

class WavesSchema(base_schema.ObjType):
    'Colander Schema for Conditions object'
    name = 'Waves'
    description = 'waves schema base class'

class Waves(Environment, serializable.Serializable):
    """
    class to compute the wave height for a time series

    At the moment, it only does a single point, non spatially
    variable, but may be extended in the future
    """
    _state = copy.deepcopy(Environment._state)
    _state += [Field('water', save=True, update=True, save_reference=True),
               Field('wind', save=True, update=True, save_reference=True)]
    _schema = WavesSchema

    _state['name'].test_for_eq = False

    def __init__(self, wind, water):
        """
        :param wind: A wind object to get the wind speed.
                     This should be a moving average wind object.
        :type wind: a Wind type, or equivelent

        :param water: water properties, specifically fetch and wave height
        :type water: environment.Water object.
        """

        self.wind = wind
        self.water = water

    # def update_water(self):
    #     """
    #     updates values from water object

    #     this should be called when you want to make sure new data is Used

    #     note: yes, this is kludgy, but it avoids calling self.water.fetch all over the place
    #     """
    #     self.wave_height = self.water.wave_height
    #     self.fetch = self.water.fetch
    #     self.density = self.water.density

    def get_value(self, time):
        """
        return the rms wave height, peak period and percent wave breaking
        at a given time. Does not currently support location-variable waves.

        :param time: the time you want the wave data for 
        :type time: datetime.datetime object

        :returns: wave_height, peak_period, whitecap_fraction, dissipation_energy

        Units:
          wave_height: meters (RMS height)
          peak_perid: seconds
          whitecap_fraction: unit-less fraction
          dissipation_energy: not sure!! # fixme!
        """
        ## make are we are up to date with water object
        wave_height = self.water.wave_height

        if wave_height is None:
            U = self.wind.get_value(time)[0] # only need velocity
            H = self.compute_H(U)
        else: # user specified a wave height
            H = wave_height
            U = self.comp_psuedo_wind(H)
        Wf = self.comp_whitecap_fraction(U)
        T = self.comp_period(U)

        De = self.disp_wave_energy(H)

        return H, T, Wf, De

    def compute_H(self, U):
        """
        compute the wave height

        :param U: wind speed
        :type U: floating point number in m/s units

        :returns Hrms: RMS wave height in meters
        """
        fetch = self.water.fetch
        ## wind stress factor
        ## Transition at U = 4.433049525859078 for linear scale with wind speed.
        ##   4.433049525859078 is where the solutions match
        ws = 0.71*U**1.23 if U < 4.433049525859078 else U # wind stress factor

        # 2268*ws**2 is limit of fetch limited case.
        if (fetch is not None) and (fetch < 2268*ws**2): ## fetch limited case
            H = 0.0016*sqrt(fetch/g)*ws
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
            #print "low wind case"
            U_h = (U_h/0.71)**0.813008
        return U_h

    def comp_whitecap_fraction(self, U):
        """
        compute the white capping fraction

        This and wave height drives dispersion
        """

        ## fixme -- smooth this out toward zero?
        ## disontinuity at 3 m/s at about 1.5%
        if U < 3.0: # m/s
            fw = 0.0
        else:
            fw = 0.5 * (0.01*U + 0.01) # Ding and Farmer (JPO 1994)

        if fw > 1.0: # only with U > 200m/s!
            fw = 1.0
        return fw / 3.85 # Ding and Farmer time constant

    def comp_period(self, U):
        """
        Compute the mean wave period 
        """
        # wind stress factor
        ## fixme: check for discontinuity at large fetch..
        ##        Is this s bit low??? 32 m/s -> T=15.7 s
        wave_height = self.water.wave_height
        fetch = self.water.wave_height
        if wave_height is None:
            ws = U * 0.71 * U**1.23 ## fixme -- linear for large windspeed?
            if (fetch is None) or (fetch >= 2268*ws**2): # fetch unlimited
                T = 0.83*ws
            else:
                T = 0.06238*(fetch*ws)**0.3333333333 # eq 3-34 (SPM?)
        else: # user-specified wave height
            T = 7.508*sqrt(wave_height)
        return T

    def disp_wave_energy(self, H):
        """
        Compute the dissipative wave energy
        """
        # fixme: does this really only depend on height?
        0.0034*self.water.density*g*H**2




    def serialize(self, json_='webapi'):
        """
        Since 'wind'/'water' property is saved as references in save file
        need to add appropriate node to WindMover schema for 'webapi'
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        if json_ == 'webapi':
            if self.wind:
                # add wind schema
                schema.add(WindSchema(name='wind'))
            if self.water:
                schema.add(WaterSchema(name='water'))

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema()
        if 'wind' in json_:
            schema.add(WindSchema(name='wind'))

        if 'water' in json_:
            schema.add(WaterSchema(name='water'))
        _to_dict = schema.deserialize(json_)

        return _to_dict
        
        
# wind.get_timeseries(self, datetime=None, units=None, format='r-theta')

