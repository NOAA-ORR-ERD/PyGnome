"""
tamoc_spill.py

This file contains the definitions to create a TAMOC simulation in GNOME and
run the simulation in a seemless integration with GNOME.

"""

from datetime import datetime, timedelta

import nucos as uc

from gnome.spills.release import Release
from gnome.spills.spill import SpillSchema, Spill
from gnome.spills.substance import Substance
from gnome.spills.gnome_oil import GnomeOil
from gnome.utilities.time_utils import asdatetime
from gnome.gnomeobject import GnomeId
from gnome import _valid_units

import numpy as np


class TamocSpill(Spill):
    """
    Models a TAMOC spill by combining the near-field model with Release
    and Substance objects

    # This really should be based in a BaseSpill Class!

    """
    # fixme: we'll need to make a proper schema for this at some point
    # but only to use with save files and/or the WebClient
    # _schema = SpillSchema

    # Attributes to overload, to remove properties from  Spill
    # this should really be factored out of a Spill Base Class
    release_time = None
    end_release_time = None
    start_position = None
    num_elements = None

    def __init__(self,
                 num_elements=1000,
                 num_per_timestep=None,
                 start_position=(0.0, 0.0, 1000.),
                 release_time=datetime.now(),  # just so it gets something...
                 release_rate=0.0,
                 release_duration=timedelta(hours=1),
                 units='bbl/day',
                 substance=None,
                 release=None,
                 water=None,
                 gor=None,
                 d0=0.0,
                 phi_0=-np.pi / 2.0,
                 theta_0=0.0,
                 windage_range=(0.01, 0.04),
                 windage_persist=900,
                 on=True,
                 name=None, **kwargs):

        self.substance = substance
        # compute the total release amount
        # this is all pretty kludgy
        # but I think we are expecting a Spill to have a total amount
        # we may be able to relax that later.
        release_time = asdatetime(release_time)
        end_release_time = release_time + release_duration

        try:
            # if it was provided in mass discharge units:
            release_rate = uc.convert(units, "kg/s", release_rate)
        except uc.UnitConversionError:  # Trying volume discharge units
            # must be in volume discharge
            release_rate = uc.convert(units, "m^3/s", release_rate)
            # convert to kg/s discharge with density
            release_rate = release_rate * self.substance.standard_density
        amount = release_rate * release_duration.total_seconds()
        self.release_mass = amount


        super(TamocSpill, self).__init__(num_elements=num_elements,
                                         amount=amount,  # could be volume or mass
                                         units='kg',
                                         substance=self.substance,
                                         release=None,
                                         on=on,
                                         **kwargs)

        # removing the release, so we can duplicate its functionality
        self.release = None

        self.release_time = release_time
        self.end_release_time = end_release_time


        self.frac_coverage = 1.0
        self._num_released = 0

        # Initialize internal element counters to None
        self._num_elements = self._num_per_timestep = None

        # Ensure either the number of elements or the release rate is given
        if num_elements is None and num_per_timestep is None:
            num_elements = 1000


        # Error check whether the number of elements is over specified
        if num_elements is not None and num_per_timestep is not None:
            msg = ('Either num_elements released or a release rate,'
                   'defined by num_per_timestep must be given, not both')
            raise TypeError(msg)

        self.start_position = start_position
        self.release_time=asdatetime(release_time)
        self.end_release_time = asdatetime(end_release_time)
        self.num_elements=num_elements



    def rewind(self):
        # do what we need to do to reset everything here
        pass

    ## this copied from Spill -- it really should be kept there
    ## but I needed it to get the density for the flow conversion
    @property
    def substance(self):
        return self._substance
    @substance.setter
    def substance(self, val):
        '''
        first try to use get_oil_props using 'val'. If this fails, then assume
        user has provided a valid OilProps object and use it as is
        '''
        if val is None:
            self._substance = NonWeatheringSubstance()
            return
        elif isinstance(val, Substance):
            self._substance = val
        try:
            self._substance = GnomeOil.get_GnomeOil(val)
        except Exception:
            if isinstance(val, str):
                raise

            self.logger.info('Failed to get_oil_props for {0}. Use as is '
                             'assuming has OilProps interface'.format(val))
            self._substance = val


# # I don't think we need this:
# class WellBlowoutRelease(Release):
#     """
#     The primary source class for Lagrangian elements

#     The primary source class that releases Lagrangian elements from a TAMOC
#     oil well blowout spill scenario.


#     """
#     def __init__(self,
#                  release_time=None,
#                  release_duration=None,
#                  start_position=None,
#                  num_elements=None,
#                  num_per_timestep=None,
#                  end_position=None,
#                  release_mass=0,
#                  **kwargs):

#         # Initialize internal element counters to None
#         self._num_elements = self._num_per_timestep = None

#         # Ensure either the number of elements or the release rate is given
#         if num_elements is None and num_per_timestep is None:
#             num_elements = 1000

#         # Pass variables on toward inherited objects
#         super(WellBlowoutRelease, self).__init__(release_time=release_time,
#                                                  num_elements=num_elements,
#                                                  release_mass = release_mass,
#                                                  **kwargs)

#         # Error check whether the number of elements is over specified
#         if num_elements is not None and num_per_timestep is not None:
#             msg = ('Either num_elements released or a release rate,'
#                    'defined by num_per_timestep must be given, not both')
#             raise TypeError(msg)

#         # Update the internal counter for element release rate
#         self._num_per_timestep = num_per_timestep

#         # Initialize remaining input variables
#         self.end_release_time = asdatetime(end_release_time)
#         self.start_position = start_position
#         self.end_position = end_position


# # or this
# def well_blowout(num_elements,
#                  start_position,
#                  release_time,
#                  release_duration,
#                  substance='AD01554',
#                  release_rate=0.,
#                  units='bbl/day',
#                  gor=0.,
#                  d0=0.,
#                  phi_0=-np.pi / 2.,
#                  theta_0=0.,
#                  windage_range=(0.01, 0.04),
#                  windage_persist=900,
#                  on=True,
#                  name='Oil Well Blowout'):
#     """
#     Helper function returns a Spill object containing a well blowout

#     Uses the TAMOC model to simulate the near-field of a well blowout and
#     passes Lagrangian Elements to GNOME, providing seamless integration
#     with GNOME.

#     Parameters
#     ----------
#     num_elements : int
#         Total number of Lagrangian elements to release in GNOME for this
#         spill (--)
#     start_position : tup
#         Initial position (lat, lon, z) in (deg, deg, m) for the blowout
#         release.  Here, z is positive down so that z is depth.
#     release_time : datetime
#         State time for this blowout release
#     end_release_time : datetime
#         End time for this blowout release
#     substance : Gnome Oil
#         Type of oil released from the blowout.  Normally, this would be an
#         Adios ID number for an oil in the GNOME Oil Library.
#     on : bool
#         Flag indicating that this spill object is currently active
#     amount : float
#         Flow rate of the release at standard conditions (bbl/day)
#     units : str
#         Units for the amount attribute
#     gor : float
#         Gas-to-oil ratio of the release at standard conditions (std ft^3/bbl)
#     d0 : float
#         Diameter of the equivalent circular area of the orifice at the
#         release (m)
#     phi_0 : float
#         Vertical orientation of the release relative to the horizontal plane
#         (rad). Since z is positive down, a vertical release would have
#         phi_0 = -np.pi/2.
#     theta_0 : float
#         Horizontal orientation of the release relative to the x-axis (East)
#         in (rad); positive angles are counter-clockwise from East.  For a
#         vertical release, this parameter has no effect.
#     windage_range : tup
#         Minimum and maximum windage coefficient values stored in a tuple (--)
#     windage_persist : int
#         ???
#     name : str
#         Name for this spill

#     Returns
#     -------
#     ts : gnome.spills.spill.Spill
#         A gnome.spills.spill.Spill object that integrates seamlessly with
#         the GNOME simulation environment.  This object determines 1.) when
#         and how many Lagrangian elements to create and 2.) specifies the
#         initial properties of these Lagrangian elements.

#     Notes
#     -----
#     This function is based on the gnome.spills.spill.surface_point_line_spill
#     function in the py_gnome package.

#     """
#     # Create the release object
#     release = WellBlowoutRelease(release_time=release_time,
#                                  start_position=start_position,
#                                  num_elements=num_elements,
#                                  release_duration=release_duration,
#                                  )

#     # Create the spill object, which includes the release
#     ts = TamocSpill(release=release,
#                     substance=substance,
#                     amount=amount,
#                     units=units,
#                     gor=gor,
#                     d0=d0,
#                     phi_0=phi_0,
#                     theta_0=theta_0,
#                     name=name,
#                     on=on)

#     # Make sure that is the substance is None, the model will not break
#     if substance is None:
#         ts.substance.windage_range=windage_range
#         ts.substance.windage_persist=windage_persist

#     return ts
