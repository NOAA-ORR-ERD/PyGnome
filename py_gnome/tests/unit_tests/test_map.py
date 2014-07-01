#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Tests of the map code.

Designed to be run with py.test

"""

from __future__ import division
import os

import numpy as np
import pytest

import gnome.map
from gnome.basic_types import oil_status, status_code_type
from gnome.utilities.projections import NoProjection

from gnome.map import MapFromBNA, RasterMap
from gnome.persist import load

from conftest import sample_sc_release

basedir = os.path.dirname(__file__)
datadir = os.path.join(basedir, r"sample_data")
testmap = os.path.join(basedir, '../sample_data', 'MapBounds_Island.bna'
                       )


##fixme: these two should maybe be in their own test file
##       -- for testing map_canvas.

### tests of depricated code -- port to new map code?
# def test_map_in_water():
#    '''
#    Test whether the location of a particle on the map
#    -- in or out of water
#    -- is determined correctly.
#    '''
#    # Create a 500x500 pixel map, with an LE refloat half-life of 2 hours
#    # (specified here in seconds).
#    m = gnome.map.lw_map([500,500],
#                         "../sample_data/MapBounds_Island.bna",
#                         2.*60.*60.,"1")
#
#    #Coordinate of a point within the water area of MapBounds_Island.bna.
#    LatInWater=48.1647
#    LonInWater=-126.78709
#
#    #Throw an error if the know in-water location returns false.
#    assert(m.in_water((LonInWater,LatInWater)))
#
#
# def test_map_on_land():
#    '''
#    Test whether the location of a particle on the map
#    (off or on land) is determined correctly.
#    '''
#    # Create a 500x500 pixel map, with an LE refloat half-life of 2 hours
#    # (specified here in seconds).
#    m = gnome.map.lw_map((500,500),
#                         "../sample_data/MapBounds_Island.bna",
#                         2.*60.*60.,
#                         color_mode = "1")
#
#    #Coordinate of a point on the island of MapBounds_Island.bna.  This point
#    #passes the test.  [Commented-out in favor of coordinate below.]
#    #OnLand = (-126.78709, 47.833333)
#
#    #Coordinate of a point that is outside of the "Map Bounds" polygon.
#    #Barker:  this should be failing! that's not on land == but it is
#              on the map.
#    Zelenke:  This point falls outside of the "Map Bounds" polygon.
#    OnLand = (-127, 47.4)
#
#    # Coordinate of a point in water that is within both the "Map Bounds" and
#    # "SpillableArea" polygons of of MapBounds_Island.bna.
#    # InWater = (-127, 47.7)
#    # This should fail.  Commented out in lieu of line below.
#    #assert(m.on_land( InWater ))
#
#    # Throw an error if the known on-land location returns false.
#    assert(m.on_land( OnLand ))

def test_in_water_resolution():
    '''
    Test the limits of the precision, to within an order of magnitude,
    defining whether a point is in or out of water.
    '''

    # Create an 500x500 pixel map, with an LE refloat half-life of 2 hours
    # (specified here in seconds).

    m = gnome.map.MapFromBNA(filename=testmap, refloat_halflife=2,
                             raster_size=500 * 500)  # in hours
                                                     # approx resolution

    # Specify coordinates of the two points that make up the
    # southeastern coastline segment of the island in the BNA map.

    x1 = -126.78709
    y1 = 47.666667
    x2 = -126.44218
    y2 = 47.833333

    # Define a point on the line formed by this coastline segment.

    slope = (y2 - y1) / (x2 - x1)
    b = y1 - slope * x1
    py = 47.7
    px = (py - b) / slope

    # Find the order of magnitude epsilon change in the latitude that causes
    # the given point to "move" from water to land.
    # eps == Distance between 1 and the nearest floating point number.

    eps = np.spacing(1)
    mag = 0.
    running = True
    while running:
        mag = mag + 1.
        print 'Order of magnitude: %g' % mag
        running = m.in_water((px, py + eps * 10.0 ** mag, 0.))

    # Difference in position within an order of magnitude in
    # degrees of latitude necessary to "move" point from water to land.

    dlatO0 = eps * 10.0 ** (mag - 1.)
    dlatO1 = eps * 10.0 ** mag

    msg = \
        '''A particle positioned on a coastline segment must be moved
    something more than {0} meters, but less than {1} meters,
    inland before pyGNOME acknowledges it's no longer in water.'''
    print msg.format(dlatO0 * 1852.0, dlatO1 * 1852.0)


## tests for GnomeMap -- the most basic version

class Test_GnomeMap:

    def test_on_map(self):
        gmap = gnome.map.GnomeMap()
        assert gmap.on_map((0., 0., 0.)) is True

        # too big latitude

        print gmap.on_map((0., 91.0, 0.))
        assert gmap.on_map((0., 91.0, 0.)) is False

        # too small latitude

        assert gmap.on_map((0., -91.0, 0.)) is False

        # too big langitude

        assert gmap.on_map((0., 361.0, 0.)) is False

        # too small langitude

        assert gmap.on_map((0., -361.0, 0.)) is False

    def test_on_land(self):
        gmap = gnome.map.GnomeMap()
        assert gmap.on_land((18.0, -87.0, 0.)) is False

    def test_in_water(self):
        gmap = gnome.map.GnomeMap()

        assert gmap.in_water((18.0, -87.0, 0.))

        assert gmap.in_water((370.0, -87.0, 0.)) is False

    def test_on_map_array(self):
        """
        a little bit more complex tests
        and test of arrays of points
        """

        # a concave map boundary

        map_bounds = ((-40.0, 50.0), (-40.0, 58.0), (-30.0, 58.0),
                      (-35.0, 53.0), (-30.0, 50.0))
        gmap = gnome.map.GnomeMap(map_bounds=map_bounds)

        points = ((-35, 55, 0.), (-45, 55, 0.))  # on map
                                                 # off map
        result = gmap.on_map(points)

        # some points on the map:

        assert np.array_equal(result, (True, False))

    def test_allowable_spill_position(self):
        gmap = gnome.map.GnomeMap()

        assert gmap.allowable_spill_position((18.0, -87.0, 0.)) is True

        assert gmap.allowable_spill_position((370.0, -87.0, 0.)) \
            is False

    def test_GnomeMap_from_dict(self):
        gmap = gnome.map.GnomeMap()
        dict_ = gmap.to_dict()
        dict_['map_bounds'] = ((-10, 10), (10, 10), (10, -10), (-10,
                               -10))
        gmap.update_from_dict(dict_)
        assert gmap.map_bounds == dict_['map_bounds']


class Test_RasterMap:

    """
    some tests for the raster map
    """

    # a very simple raster:

    (w, h) = (20, 12)
    raster = np.zeros((w, h), dtype=np.uint8)

    # set some land in middle:

    raster[6:13, 4:8] = 1

    def test_save_as_image(self):
        """
        only tests that it doesn't crash -- you need to look at the
        image to see if it's right
        """
        rmap = RasterMap(refloat_halflife=6,
                         bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),(50, -30)),
                         projection=NoProjection())  

        rmap.save_as_image('raster_map_image.png')

        assert True

    def test_on_map(self):
        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours
        assert gmap.on_map((0., 0., 0.))

        assert gmap.on_map((55.0, 0., 0.)) is False

    def test_on_land(self):
        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours
        print 'testing a land point:', (10, 6, 0.)
        print gmap.on_land((10, 6, 0.))
        assert gmap.on_land((10, 6, 0.))  # right in the middle

        print 'testing a water point:'
        assert not gmap.on_land((19.0, 11.0, 0.))

    def test_spillable_area(self):

        # anywhere not on land is spillable...
        # in this case

        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        print 'testing a land point:'

        # right in the middle of land

        assert not gmap.allowable_spill_position((10, 6, 0.))

        print 'testing a water point:'
        assert gmap.allowable_spill_position((19.0, 11.0, 0.))

    def test_spillable_area2(self):

        # a test with a polygon spillable area

        poly = ((5, 2), (15, 2), (15, 10), (10, 10), (10, 5))
        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection(),
                         spillable_area=poly)  # hours

        # cases that are spillable

        assert gmap.allowable_spill_position((11.0, 3.0, 0.))
        assert gmap.allowable_spill_position((14.0, 9.0, 0.))

        # in polygon, but on land:

        assert not gmap.allowable_spill_position((11.0, 6.0, 0.))

        # outside polygon, on land:

        assert not gmap.allowable_spill_position((8.0, 6.0, 0.))

        # outside polygon, off land:

        assert not gmap.allowable_spill_position((3.0, 3.0, 0.))


class TestRefloat:

    """
    only tests the refloat_elements interface and functionality
    for borderline cases like all elements in water, refloat_halflife = 0

    A raster map with only water is used, but since there isn't a land check,
    this is irrelevant
    """

    # make time_step = refloat_halflife so 50% probability of refloat

    time_step = 3600.

    # land/water irrelevant for this test

    map = RasterMap(refloat_halflife=time_step / 3600.,
                    bitmap_array=np.zeros((20, 12), dtype=np.uint8),
                    projection=NoProjection(),
                    map_bounds=((-50, -30),(-50, 30), (50, 30), (50, -30)))  # hours

    num_les = 1000
    spill = sample_sc_release(num_les)
    orig_pos = np.random.uniform(0, num_les, spill['positions'].shape)
    last_water = (1., 2., 0.)

    (spill['positions'])[:] = orig_pos
    spill['last_water_positions'] += last_water

    def reset(self):
        (self.spill['positions'])[:] = self.orig_pos
        (self.spill['last_water_positions'])[:] = self.last_water
        self.map.refloat_halflife = self.time_step / 3600.

    def test_all_elementsinwater(self):
        """
        all elements in water so do nothing
        """

        self.reset()  # reset _state
        (self.spill['status_codes'])[:] = oil_status.in_water
        self.map.refloat_elements(self.spill, self.time_step)
        assert np.all(self.spill['positions'] == self.orig_pos)
        assert np.all(self.spill['status_codes'] == oil_status.in_water)

    def test_refloat_halflife_0(self):
        """
        refloat_halflife is 0 so refloat all elements on land
        """

        self.reset()
        self.map.refloat_halflife = 0
        (self.spill['status_codes'])[5:] = oil_status.on_land
        self.map.refloat_elements(self.spill, self.time_step)
        assert np.all((self.spill['positions'])[:5]
                      == self.orig_pos[:5])
        assert np.all((self.spill['positions'])[5:] == self.last_water)

    def test_refloat_halflife_negative(self):
        """
        refloat_halflife is test_refloat_halflife_negative:

        this should mean totally sticky --no refloat

        """

        self.reset()
        self.map.refloat_halflife = -1
        (self.spill['status_codes'])[5:] = oil_status.on_land
        orig_status_codes = self.spill['status_codes'].copy()
        self.map.refloat_elements(self.spill, self.time_step)
        assert np.all((self.spill['positions']) == self.orig_pos)
        assert np.all( self.spill['status_codes'] == orig_status_codes)


    def test_refloat_some_onland(self):
        """
        refloat elements on land based on probability
        """

        self.reset()
        (self.spill['status_codes'])[:] = oil_status.in_water
        self.map.refloat_halflife = 3 * self.time_step / 3600.

        # say 500 out of 1000 are on_land, and we expect about 50% of these
        # to refloat

        # initial 25% LEs on_land, last 25% of LEs on_land

        init_ix = int(round(.25 * self.num_les))
        last_ix = self.num_les - (int(round(.5 * self.num_les))
                                  - init_ix)

        ix = range(init_ix)  # choose first 25% of indices
        ix.extend(range(last_ix, self.num_les, 1))  # last 25% of indices
        ix = np.asarray(ix)

        self.spill['status_codes'][ix] = oil_status.on_land
        self.map.refloat_elements(self.spill, self.time_step)

        expected = round(1. - .5 ** (self.time_step
                         / (self.map.refloat_halflife * 3600.)), 2) \
            * 100
        actual = np.count_nonzero(self.spill['status_codes'][ix]
                                  == oil_status.in_water) \
            / (self.num_les / 2) * 100
        print 'Expect {0}% refloat, actual refloated: {1}%'.format(expected,
                actual)

        # ensure some of the elements that were on land are back on water

        assert np.count_nonzero(self.spill['status_codes'][ix]
                                == oil_status.in_water) > 0

        refloat_ix = ix[np.where(self.spill['status_codes'][ix]
                        == oil_status.in_water)[0]]
        assert np.all(self.spill['positions'][refloat_ix]
                      == self.last_water)  # refloated elements
        assert np.all(self.spill['status_codes'][refloat_ix]
                      == oil_status.in_water)  # status is back in water

        # ensure elements that were in_water are not changed
        # these are original values that are not refloated

        mask = np.array([i not in refloat_ix for i in
                        range(self.num_les)], dtype=bool)
        assert np.all(self.spill['positions'][mask, :]
                      == self.orig_pos[mask, :])


class Test_MapfromBNA:

    bna_map = MapFromBNA(testmap, 6, raster_size=1000)

    def test_map_in_water(self):
        '''
        Test whether the location of a particle is:
          - in water
          - is determined correctly.
        '''

        InWater = (-126.78709, 48.1647, 0.)

        # Throw an error if the known in-water location returns false.

        assert self.bna_map.in_water(InWater)
        assert not self.bna_map.on_land(InWater)

    def test_map_in_water2(self):

        # in water, but inside land Bounding box

        InWater = (-126.971456, 47.935608, 0.)

        # Throw an error if the know in-water location returns false.

        assert self.bna_map.in_water(InWater)

    def test_map_on_land(self):
        '''
        Test whether the location of a particle on land is determined
        correctly.
        '''

        # Throw an error if the know on-land location returns false.

        OnLand = (-127, 47.8, 0.)
        assert self.bna_map.on_land(OnLand)

        # Throw an error if the know on-land location returns false.

        assert not self.bna_map.in_water(OnLand)

    def test_map_in_lake(self):
        '''
        Test whether the location of a particle in a lake
        is determined correctly.
        '''

        # Throw an error if the know on-land location returns false.

        InLake = (-126.8, 47.84, 0.)
        assert self.bna_map.in_water(InLake)

        # Throw an error if the know on-land location returns false.

        assert not self.bna_map.on_land(InLake)

    def test_map_spillable(self):
        point = (-126.984472, 48.08106, 0.)  # in water, in spillable

        # Throw an error if the know on-land location returns false.

        assert self.bna_map.allowable_spill_position(point)

    def test_map_spillable_lake(self):
        point = (-126.793592, 47.841064, 0.)  # in lake, should be spillable

        # Throw an error if the known on-land location returns false.

        assert self.bna_map.allowable_spill_position(point)

    def test_map_not_spillable(self):
        point = (-127, 47.8, 0.)  # on land should not be spillable

        # Throw an error if the know on-land location returns false.

        assert not self.bna_map.allowable_spill_position(point)

    def test_map_not_spillable2(self):

        # in water, but outside spillable area

        point = (127.244752, 47.585072, 0.)

        # Throw an error if the know on-land location returns false.

        assert not self.bna_map.allowable_spill_position(point)

    def test_map_not_spillable3(self):

        # off the map -- should not be spillable

        point = (127.643856, 47.999608, 0.)

        # Throw an error if the know on-land location returns false.

        assert not self.bna_map.allowable_spill_position(point)

    def test_map_on_map(self):
        point = (-126.12336, 47.454164, 0.)
        assert self.bna_map.on_map(point)

    def test_map_off_map(self):
        point = (-126.097336, 47.43962, 0.)
        assert not self.bna_map.on_map(point)


@pytest.mark.parametrize("json_", ('save', 'webapi'))
def test_serialize_deserialize(json_):
    """
    test create new object from to_dict
    """

    gmap = gnome.map.MapFromBNA(testmap, 6)
    serial = gmap.serialize('webapi')
    dict_ = gnome.map.MapFromBNA.deserialize(serial)
    map2 = gmap.new_from_dict(dict_)
    assert gmap == map2

    dict_['map_bounds'] = ((-10, 10), (10, 10), (10, -10), (-10, -10))
    dict_['spillable_area'] = ((-5, 5), (5, 5), (5, -5), (-5, -5))
    dict_['refloat_halflife'] = 2
    gmap.update_from_dict(dict_)
    assert gmap.map_bounds == dict_['map_bounds']
    assert gmap.spillable_area == dict_['spillable_area']
    assert gmap.refloat_halflife == dict_['refloat_halflife']


class Test_full_move:

    """
    A test to see if the full API is working for beaching

    It should check for land-jumping and return the "last known water point"
    """

    # a very simple raster:

    (w, h) = (20, 10)
    raster = np.zeros((w, h), dtype=np.uint8)

    # a single skinny vertical line:

    raster[10, :] = 1

    def test_on_map(self):
        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        # making sure the gmap is set up right

        assert gmap.on_map((100.0, 1., 0.)) is False
        assert gmap.on_map((0., 1., 0.))

    def test_on_land(self):
        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours
        assert gmap.on_land((10, 3, 0)) == 1
        assert gmap.on_land((9, 3, 0)) == 0
        assert gmap.on_land((11, 3, 0)) == 0

    def test_land_cross(self):
        """
        try a single LE that should be crossing land
        """

        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        spill = sample_sc_release(1)

        spill['positions'] = np.array(((5.0, 5.0, 0.), ),
                dtype=np.float64)
        spill['next_positions'] = np.array(((15.0, 5.0, 0.), ),
                dtype=np.float64)
        spill['status_codes'] = np.array((oil_status.in_water, ),
                dtype=status_code_type)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'][0], (10.0, 5.0,
                              0.))
        assert np.array_equal(spill['last_water_positions'][0], (9.0,
                              5.0, 0.))
        assert spill['status_codes'][0] == oil_status.on_land

    def test_land_cross_array(self):
        """
        test a few LEs
        """

        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        # one left to right
        # one right to left
        # one diagonal upper left to lower right
        # one diagonal upper right to lower left

        spill = sample_sc_release(4)

        spill['positions'] = np.array(((5.0, 5.0, 0.), (15.0, 5.0, 0.),
                (0., 0., 0.), (19.0, 0., 0.)), dtype=np.float64)
        spill['next_positions'] = np.array(((15.0, 5.0, 0.), (5.0, 5.0,
                0.), (10.0, 5.0, 0.), (0., 9.0, 0.)), dtype=np.float64)
        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'], ((10.0, 5.0,
                              0.), (10.0, 5.0, 0.), (10.0, 5.0, 0.),
                              (10.0, 4.0, 0.)))

        assert np.array_equal(spill['last_water_positions'], ((9.0,
                              5.0, 0.), (11.0, 5.0, 0.), (9.0, 4.0,
                              0.), (11.0, 4.0, 0.)))

        assert np.alltrue(spill['status_codes'] == oil_status.on_land)

    def test_some_cross_array(self):
        """
        test a few LEs
        """

        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        # one left to right
        # one right to left
        # diagonal that doesn't hit
        # diagonal that does hit

        spill = sample_sc_release(4)

        spill['positions'] = np.array(((5.0, 5.0, 0.), (15.0, 5.0, 0.),
                (0., 0., 0.), (19.0, 0., 0.)), dtype=np.float64)

        spill['next_positions'] = np.array(((9.0, 5.0, 0.), (11.0, 5.0,
                0.), (9.0, 9.0, 0.), (0., 9.0, 0.)), dtype=np.float64)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'], ((9.0, 5.0, 0.),
                              (11.0, 5.0, 0.), (9.0, 9.0, 0.), (10.0,
                              4.0, 0.)))

        # just the beached ones

        assert np.array_equal((spill['last_water_positions'])[3:],
                              ((11.0, 4.0, 0.), ))

        assert np.array_equal((spill['status_codes'])[3:],
                              (oil_status.on_land, ))

    def test_outside_raster(self):
        """
        test LEs starting form outside the raster bounds
        """

        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        # one left to right
        # one right to left
        # diagonal that doesn't hit
        # diagonal that does hit
        # spill = gnome.spill.Spill(num_LEs=4)

        spill = sample_sc_release(4)
        spill['positions'] = np.array(((30.0, 5.0, 0.), (-5.0, 5.0,
                0.), (5.0, -5.0, 0.), (-5.0, -5.0, 0.)),
                dtype=np.float64)  # outside right
                                   # outside left
                                   # outside top
                                   # outside upper left

        spill['next_positions'] = np.array(((15.0, 5.0, 0.), (5.0, 5.0,
                0.), (5.0, 15.0, 0.), (25.0, 15.0, 0.)),
                dtype=np.float64)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'], ((15.0, 5.0,
                              0.), (5.0, 5.0, 0.), (5.0, 15.0, 0.),
                              (10.0, 5.0, 0.)))

        # just the beached ones

        assert np.array_equal((spill['last_water_positions'])[3:],
                              ((9.0, 4.0, 0.), ))

        assert np.array_equal((spill['status_codes'])[3:],
                              (oil_status.on_land, ))

    def test_some_off_map(self):
        """
        Test LEs that go off the map

        should get off_map flag
        """

        gmap = RasterMap(refloat_halflife=6, bitmap_array=self.raster,
                         map_bounds=((-50, -30), (-50, 30), (50, 30),
                         (50, -30)), projection=NoProjection())  # hours

        spill = sample_sc_release(8)
        spill['positions'] = np.array((
            (45.0, 25.0, 0.),
            (45.0, 25.0, 0.),
            (45.0, -25.0, 0.),
            (45.0, -25.0, 0.),
            (-45.0, -25.0, 0.),
            (-45.0, -25.0, 0.),
            (-45.0, 25.0, 0.),
            (-45.0, 25.0, 0.),
            ), dtype=np.float64)

        spill['next_positions'] = np.array((  # off
                                              # still on
                                              # off
                                              # still on
                                              # off
                                              # still on
                                              # off
                                              # still on
            (55.0, 25.0, 0.),
            (49.0, 25.0, 0.),
            (45.0, -35.0, 0.),
            (45.0, -29.0, 0.),
            (-55.0, -25.0, 0.),
            (-49.0, -25.0, 0.),
            (-45.0, 35.0, 0.),
            (-45.0, 29.0, 0.),
            ), dtype=np.float64)

        gmap.beach_elements(spill)

        # off = np.ones(4,) * basic_types.oil_status.off_maps

        off = np.ones(4) * oil_status.to_be_removed
        assert np.array_equal(spill['status_codes'][0::2], off)

        on = np.ones(4) * oil_status.in_water
        assert np.array_equal(spill['status_codes'][1::2], on)


def test_resurface_airborne_elements():
    positions = np.array(((1, 2, 0.), (3, 4, 1.), (-3, 4, -1.), (3, 4,
                         0.1), (3, 4, 0.1)), dtype=np.float64)
    spill = {'next_positions': positions}
    m = gnome.map.GnomeMap()
    m.resurface_airborne_elements(spill)

    assert spill['next_positions'][:, 2].min() == 0.


if __name__ == '__main__':

#    tester = Test_GnomeMap()
#    tester.test_on_map()
#    tester.test_on_map_array()
#    tester.test_allowable_spill_position()

    tester = Test_full_move()
    tester.test_some_off_map()
