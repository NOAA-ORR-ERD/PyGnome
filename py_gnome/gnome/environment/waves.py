#!/usr/bin/env python

"""
The waves environment object.

Computes the wave height and percent wave breaking

Uses the same approach as ADIOS 2

(code ported from old MATLAB prototype code)

"""
from __future__ import division

import copy

from gnome import constants
from gnome.utilities import serializable
from gnome.utilities.serializable import Field
from gnome.utilities.weathering import Adios2, LehrSimecek, PiersonMoskowitz

from gnome.persist import base_schema
from gnome.exceptions import ReferencedObjectNotSet

from .environment import Environment
from .environment import WaterSchema

from wind import WindSchema

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
    _ref_as = 'waves'
    _state = copy.deepcopy(Environment._state)
    _state += [Field('water', save=True, update=True, save_reference=True),
               Field('wind', save=True, update=True, save_reference=True)]
    _schema = WavesSchema

    _state['name'].test_for_eq = False

    def __init__(self, wind=None, water=None, **kwargs):
        """
        wind and water must be set before running the model; however, these
        can be set after object construction

        :param wind: A wind object to get the wind speed.
                     This should be a moving average wind object.
        :type wind: a Wind type, or equivelent

        :param water: water properties, specifically fetch and wave height
        :type water: environment.Water object.

        .. note:: must take **kwargs since base class supports more inputs like
            'name'. The new_from_dict() alternate constructor will invoke
            __init__ will arguments that supported by baseclass
        """

        self.wind = wind
        self.water = water

        # turn off make_default_refs if references are defined and
        # make_default_refs is False
        if self.water is not None and self.wind is not None:
            kwargs['make_default_refs'] = \
                kwargs.pop('make_default_refs', False)

        super(Waves, self).__init__(**kwargs)

    # def update_water(self):
    #     """
    #     updates values from water object

    #     this should be called when you want to make sure new data is Used

    #     note: yes, this is kludgy, but it avoids calling self.water.fetch
    #           all over the place
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

        :returns: wave_height, peak_period, whitecap_fraction,
                  dissipation_energy

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
            U = self.pseudo_wind(H)
        Wf = self.whitecap_fraction(U)
        T = self.mean_wave_period(U)

        De = self.dissipative_wave_energy(H)

        return H, T, Wf, De

    def get_emulsification_wind(self, time):
        """
        Return the right wind for the wave climate

        If a wave height was specified, then you need the greater of the
        real or pseudo wind.

        If not, then you need the actual wind.

        The idea here is that if there is a low wind, but the user specified
        waves, we really want emulsification that makes sense for the waves.
        But if the actual wind is stronger than that for the wave height give,
        we should use the actual wind.

        fixme: I'm not sure this is right -- if we stick with the wave energy
               given by the user for dispersion, why not for emulsification?
        """
        wave_height = self.water.wave_height
        U = self.wind.get_value(time)[0]  # only need velocity
        if wave_height is None:
            return U
        else:  # user specified a wave height
            return max(U, self.pseudo_wind(wave_height))

    def compute_H(self, U):
        return Adios2.wave_height(U, self.water.fetch)

    def pseudo_wind(self, H):
        return Adios2.wind_speed_from_height(H)

    def whitecap_fraction(self, U):
        return LehrSimecek.whitecap_fraction(U, self.water.salinity)

    def mean_wave_period(self, U):
        return Adios2.mean_wave_period(U,
                                       self.water.wave_height,
                                       self.water.fetch)

    def peak_wave_period(self, time):
        '''
        :param time: the time you want the wave data for
        :type time: datetime.datetime object

        :returns: peak wave period (s)
        '''
        U = self.wind.get_value(time)[0]
        return PiersonMoskowitz.peak_wave_period(U)

    def dissipative_wave_energy(self, H):
        return Adios2.dissipative_wave_energy(self.water.density, H)

    def serialize(self, json_='webapi'):
        """
        Since 'wind'/'water' property is saved as references in save file
        need to add appropriate node to WindMover schema for 'webapi'
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        if json_ == 'webapi':
            if self.wind:
                schema.add(WindSchema(name='wind'))
            if self.water:
                schema.add(WaterSchema(name='water'))

        return schema.serialize(toserial)

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

        return schema.deserialize(json_)

    def prepare_for_model_run(self, model_time):
        if self.wind is None:
            msg = "wind object not defined for " + self.__class__.__name__
            raise ReferencedObjectNotSet(msg)

        if self.water is None:
            msg = "water object not defined for " + self.__class__.__name__
            raise ReferencedObjectNotSet(msg)
