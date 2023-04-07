#!/usr/bin/env python

"""
tideflat_map.py

A map that handles tide flats:

It sets the oil_status code to "on_tideflat" when particles are on a tide flat.

Then the movers will not move the particles while that flag is set.

It is set back to in_water when the water depth indicates the tide flat
is no longer a tide flat.
"""

import numpy as np

from gnome.gnomeobject import GnomeId
from gnome.basic_types import oil_status
from gnome.utilities.time_utils import asdatetime

from gnome.utilities.geometry import (points_in_poly,
                                      )


class TideflatMap(GnomeId):
    """
    Checks for tidal flats, and sets and unsets the on_tidalflat
    oil_status code appropriately.

    Not sub-classed from a GnomeMap, as it delegates to the passed-in map
    instead
    """

    def __init__(self, land_map, tideflat):
        """
        initialize a TideflatMap

        :param land_map: A usual GnomeMap object -- for land-water, etc.
                         It will be used to do beaching/refloating before
                         the tidal flat is checked
        :type land_map: :class: GnomeMap

        :param tideflat: A tide-flat object that indicates when the water
                          depth is zero -- or other indicator that the element
                          is on a tideflat
        :type tideflat: :class: TideFlatBase object
        """

        self.land_map = land_map
        self.tideflat = tideflat

    def __getattr__(self, name):
        """
        Delegate everything that is not overridden to the enclosed GnomeMap
        """
        return getattr(self.land_map, name)

    # These are the methods that need to be overridden:
    def beach_elements(self, spill_container, model_time=None):

        # first check the usual beaching:
        self.land_map.beach_elements(spill_container, model_time)

        status_codes = spill_container['status_codes']
        positions = spill_container['next_positions']

        # only the elements that are on water
        #  note that they get moved off the tideflat in refloat_elements
        in_water_idx = np.nonzero(status_codes == oil_status.in_water)[0]

        if in_water_idx.size > 0:  # some elements in water
            in_water_pos = positions[in_water_idx]

            # check if they are now dry
            now_dry = self.tideflat.is_dry(in_water_pos, model_time)

            # set them to on_tideflat
            status_codes[in_water_idx[now_dry]] = oil_status.on_tideflat

    def refloat_elements(self, spill_container, time_step, model_time=None):
        """
        Checks whether elements that were on a tidal flat still are.
        If not, removes the flag

        Then passes off the the enclosed map to do the refloat logic

        :param spill_container: current SpillContainer
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`
        """
        status_codes = spill_container['status_codes']
        positions = spill_container['next_positions']

        tf_idx = np.nonzero(status_codes == oil_status.on_tideflat)[0]

        if tf_idx.size > 0:  # some elements on tidal flats
            on_tideflat = positions[tf_idx]
            # check if they are still dry
            now_wet = self.tideflat.is_wet(on_tideflat, model_time)
            # reset_them
            status_codes[tf_idx[now_wet]] = oil_status.in_water

        # Pass off to the map
        self.land_map.refloat_elements(spill_container, time_step)


class TideflatBase(GnomeId):
    """
    Base class for a Tideflat object to be used with TideflatBase

    Subclasses should override the ``is_dry()`` method.
    """
    pass

    def is_dry(self, points, time):
        """
        :param points: locations for testing if the locations are dry.are
        :type points: Nx3 numpy array or equivalent.

        :param time: time at which to check for wet/dry

        :return: numpy array of bools one for each point
        This should be over-ridden -- this version always returns
        FAlse for all points
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[1] != 3:
            raise ValueError("points should be a Nx3 array or equivalent")
        return np.zeros(points.shape[0], dtype=bool)

    def is_wet(self, points, time):
        return np.logical_not(self.is_dry(points, time))


class SimpleTideflat(TideflatBase):
    """
    Simple Tideflat implementation for testing and demo

    For a real case, a subclass of TideflatBase should be
    made that implements these methods in a meaningful way

    This provides a single polygon describing a tidal flat

    The flat is initially wet, then dry at a given time,
    then wet again after that.

    Just enough for testing.
    """

    def __init__(self, bounds, dry_start, dry_end):

        """
        :param bounds: polygon around the tidal flat

        :dry_start: initial time of flat being dry
        :dry_end: end time of flat being dry
        """

        self.bounds = np.array(bounds, dtype=np.float64).reshape((-1, 2))
        self.dry_start = asdatetime(dry_start)
        self.dry_end = asdatetime(dry_end)

    def is_dry(self, points, model_time):
        """
        :param points: locations for testing if the locations are dry.
        :type points: Nx3 numpy array or equivalent.

        :param time: time at which to check for wet/dry

        :return: numpy array of bools one for each point
        """
        points = np.array(points, dtype=np.float64).reshape((-1, 3))

        # check time first
        if model_time < self.dry_start or model_time > self.dry_end:
            return np.zeros(points.shape[0], dtype=bool)

        return points_in_poly(self.bounds, points)
