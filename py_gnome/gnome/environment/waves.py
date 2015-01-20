#!/usr/bin/env python

"""
The waves environment object.

Computes the wave height and percent wave breaking

Uses the same approach as ADIOS 2

(code ported from old MATLAB prototype code)

"""
from __future__ import division
from math import sqrt

import copy

from gnome import constants
from gnome.utilities import serializable
from gnome.utilities.serializable import Field
from gnome.persist import base_schema
from .environment import Environment
from wind import WindSchema
from .environment import WaterSchema


g = constants.gravity  # the gravitational constant.


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
        # make sure are we are up to date with water object
        wave_height = self.water.wave_height

        if wave_height is None:
            U = self.wind.get_value(time)[0]  # only need velocity
            H = self.compute_H(U)
        else:  # user specified a wave height
            H = wave_height
            U = self.comp_psuedo_wind(H)
        Wf = self.comp_whitecap_fraction(U)
        T = self.comp_period(U)

        De = self.disp_wave_energy(H)

        return H, T, Wf, De

    def get_emulsifiation_wind(self, time):
        """
        Return the right wind for the wave climate

        If a wave height was specified, then you need the greater of the real or
        psuedo wind. 

        If not, then you need the actual wind.

        The idea here is that if there is a low wind, but the user specified waves,
        we really want emulsification that makes sense for the waves. But if the
        actual wind is stronger than that for the wave height give, we should use
        the actual wind.

        fixme: I'm not sure this is right -- if we stick with the wave energy given
        by the user for dispesion, why not for emulsification?

        """
        wave_height = self.water.wave_height
        U = self.wind.get_value(time)[0]  # only need velocity
        if wave_height is None:
            return U
        else:  # user specified a wave height
            return max( U, self.comp_psuedo_wind(wave_height) )


    # def get_pseudo_wind(self, time):
    #     wave_height = self.water.wave_height
    #     if wave_height is None:
    #         U = self.wind.get_value(time)[0]  # only need velocity
    #         H = self.compute_H(U)
    #     else:  # user specified a wave height
    #         H = wave_height
    #     U = self.comp_psuedo_wind(H)

    #     return U

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
        if (fetch is not None) and (fetch < 2268*ws**2):  # fetch limited case
            H = 0.0016*sqrt(fetch/g)*ws
        else:  # fetch unlimited
            H = 0.243*ws*ws/g

        Hrms = 0.707*H

        # arbitrary limit at 30 m -- about the largest waves recorded
        # fixme -- this really depends on water depth -- should take that into account?
        return Hrms if Hrms < 30.0 else 30.0

    def comp_psuedo_wind(self, H):
        """
        Compute the wind speed to use for the whitecap fraction

        Used if the wave height is specified.

        Unlimited fetch is assumed: this is the reverse of compute_H

        :param H: given wave height.
        """

        ##U_h = 2.0286*g*sqrt(H/g) # Bill's version
        U_h = sqrt(g * H / 0.243)
        if U_h < 4.433049525859078:  # check if low wind case
            U_h = (U_h/0.71)**0.813008
        return U_h

    def comp_whitecap_fraction(self, U):
        """
        compute the white capping fraction

        This and wave height drives dispersion

        This based on the formula in:
        Lehr and Simecek-Beatty
        The Relation of Langmuir Circulation Processes to the Standard Oil Spill Spreading, Dispersion and Transport Algorithms
        Spill Sci. and Tech. Bull, 6:247-253 (2000)
        (maybe typo -- didn't match)

        Should look in:  Ocean Waves Breaking and Marine Aerosol Fluxes
                         By Stanislaw R. Massel

        """

        ## Monahan(JPO, 1971) time constant characterizing exponential whitecap decay.
        ## The saltwater value for   is 3.85 sec while the freshwater value is 2.54 sec.
        #  interpolate with salinity:
        Tm = 0.03742857*self.water.salinity + 2.54

        if U < 4.0: # m/s
            ## linear fit from 0 to the 4m/s value from Ding and Farmer
            ## maybe should be a exponential / quadratic fit?
            ## or zero less than 3, then a sharp increase to 4m/s?
            fw = (0.0125*U) / Tm
        else:
            fw = (0.01*U + 0.01) / Tm  # Ding and Farmer (JPO 1994)

        return fw if fw <= 1.0 else 1.0  # only with U > 200m/s!

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
            ws = U * 0.71 * U**1.23  ## fixme -- linear for large windspeed?
            if (fetch is None) or (fetch >= 2268*ws**2):  # fetch unlimited
                T = 0.83*ws
            else:
                T = 0.06238*(fetch*ws)**0.3333333333  # eq 3-34 (SPM?)
        else:  # user-specified wave height
            T = 7.508*sqrt(wave_height)
        return T

    def disp_wave_energy(self, H):
        """
        Compute the dissipative wave energy
        """
        # fixme: does this really only depend on height?
        return 0.0034*self.water.density*g*H**2

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

