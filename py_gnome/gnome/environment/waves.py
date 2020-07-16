#!/usr/bin/env python
"""
The waves environment object.

Computes the wave height and percent wave breaking

Uses the same approach as ADIOS 2

(code ported from old MATLAB prototype code)
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


from future import standard_library
standard_library.install_aliases()
from builtins import *
from past.utils import old_div
import copy
import numpy as np

from gnome import constants
from gnome.utilities.weathering import Adios2, LehrSimecek, PiersonMoskowitz

from gnome.persist import base_schema
from gnome.exceptions import ReferencedObjectNotSet

from .environment import Environment
from .water import WaterSchema

from .wind import WindSchema
from gnome.environment.gridded_objects_base import VectorVariableSchema
from gnome.environment.wind import Wind
from gnome.environment.water import Water

g = constants.gravity  # the gravitational constant.


class WavesSchema(base_schema.ObjTypeSchema):
    'Colander Schema for Conditions object'
    description = 'waves schema base class'
    water = WaterSchema(
        save=True, update=True, save_reference=True
    )
    wind = base_schema.GeneralGnomeObjectSchema(
            acceptable_schemas=[WindSchema, VectorVariableSchema],
            save_reference=True
    )


class Waves(Environment):
    """
    class to compute the wave height for a time series

    At the moment, it only does a single point, non spatially
    variable, but may be extended in the future
    """
    _ref_as = 'waves'
    _req_refs = ['wind', 'water']
    _schema = WavesSchema

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

    @property
    def data_start(self):
        '''
            The Waves object doesn't directly manage a time series of data,
            so it will not have a data range itself.  But it depends upon
            a Wind and a Water object.  The Water won't have a data range
            either, but the Wind will.
            So its data range will be that of the Wind it is associated with.
        '''
        return self.wind.data_start

    @property
    def data_stop(self):
        return self.wind.data_stop

    def get_value(self, points, time):
        """
        return the rms wave height, peak period and percent wave breaking
        at a given time. Does not currently support location-variable waves.

        :param time: the time you want the wave data for
        :type time: datetime.datetime object

        :returns: wave_height, peak_period, whitecap_fraction,
                  dissipation_energy

        Units:
          wave_height: meters (RMS height)
          peak_period: seconds
          whitecap_fraction: unit-less fraction
          dissipation_energy: not sure!! # fixme!
        """
        # make sure are we are up to date with water object
        wave_height = self.water.get('wave_height')

        if wave_height is None:
            # only need velocity
            U = self.get_wind_speed(points, time)
            H = self.compute_H(U)
        else:
            # user specified a wave height
            U = self.get_wind_speed(points, time)
            H = np.full_like(U, wave_height)
            #H = wave_height
            U = self.pseudo_wind(H)	#significant wave height used for pseudo wind
            H = .707 * H	#Hrms

        Wf = self.whitecap_fraction(U)
        T = self.mean_wave_period(U)

        De = self.dissipative_wave_energy(H)

        return H, T, Wf, De

    def get_wind_speed(self, points, model_time,
                       coord_sys='r', fill_value=1.0):
        '''
        Wrapper for the weatherers so they can extrapolate
        '''
        retval = self.wind.at(points, model_time, coord_sys=coord_sys)

        if isinstance(retval, np.ma.MaskedArray):
            return retval.filled(fill_value)
        else:
            return retval

    def get_emulsification_wind(self, points, time):
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
        wave_height = self.water.get('wave_height')
        U = self.get_wind_speed(points, time)  # only need velocity

        if wave_height is None:
            return U
        else:  # user specified a wave height
            U = np.where(U < self.pseudo_wind(wave_height),
                         self.pseudo_wind(wave_height),
                         U)
            return U

    def compute_H(self, U):
        U = np.array(U).reshape(-1)
        return Adios2.wave_height(U, self.water.get('fetch'))

    def pseudo_wind(self, H):
        H = np.array(H).reshape(-1)
        return Adios2.wind_speed_from_height(H)

    def whitecap_fraction(self, U):
        U = np.array(U).reshape(-1)
        return LehrSimecek.whitecap_fraction(U, self.water.salinity)

    def mean_wave_period(self, U):
        U = np.array(U).reshape(-1)
        return Adios2.mean_wave_period(U,
                                       self.water.get('wave_height'),
                                       self.water.get('fetch'))

    def peak_wave_period(self, points, time):
        '''
        :param time: the time you want the wave data for
        :type time: datetime.datetime object

        :returns: peak wave period (s)
        '''
        U = self.get_wind_speed(points, time)  # only need velocity

        return PiersonMoskowitz.peak_wave_period(U)

    def dissipative_wave_energy(self, H):
        return Adios2.dissipative_wave_energy(self.water.density, H)

    def energy_dissipation_rate(self, H, U):
        '''
        c_ub = 100 = dimensionless empirical coefficient to correct
        for non-Law-of-the-Wall results (Umlauf and Burchard, 2003)

        u_c = water friction velocity (m/s)
               sqrt(rho_air / rho_w) * u_a ~ .03 * u_a
        u_a = air friction velocity (m/s)
        z_0 = surface roughness (m) (Taylor and Yelland)
        c_p = peak wave speed for Pierson-Moskowitz spectrum
        w_p = peak angular frequency for Pierson-Moskowitz spectrum (1/s)

        TODO: This implementation should be in a utility function.
              It should not be part of the Waves management object itself.
        '''
        if H is 0 or U is 0:
            return 0

        c_ub = 100

        c_p = PiersonMoskowitz.peak_wave_speed(U)
        w_p = PiersonMoskowitz.peak_angular_frequency(U)

        z_0 = 1200 * H * ((old_div(H, (2*np.pi*c_p))) * w_p)**4.5
        u_a = old_div(.4 * U, np.log(old_div(10, z_0)))
        u_c = .03 * u_a
        eps = old_div(c_ub * u_c**3, H)

        return eps


    def prepare_for_model_run(self, _model_time):
        if self.wind is None:
            raise ReferencedObjectNotSet("wind object not defined for {}"
                                         .format(self.__class__.__name__))

        if self.water is None:
            raise ReferencedObjectNotSet("water object not defined for {}"
                                         .format(self.__class__.__name__))
