#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests of the map code.

Designed to be run with py.test
"""





import os

import pytest

from pprint import pprint
import numpy as np


# import gnome.maps.map
from gnome.basic_types import oil_status, status_code_type
from gnome.utilities.projections import NoProjection

from gnome.maps import GnomeMap, MapFromBNA, RasterMap, ParamMap
# MapFromUGrid

from gnome.gnomeobject import class_from_objtype

from ..conftest import sample_sc_release


# fixme: this should realy be in conftest
basedir = os.path.dirname(__file__)
basedir = os.path.split(basedir)[0]
datadir = os.path.normpath(os.path.join(basedir, "sample_data"))
output_dir = os.path.normpath(os.path.join(basedir, "output_dir"))
testbnamap = os.path.join(datadir, 'MapBounds_Island.bna')
bna_with_lake = os.path.join(datadir, 'florida_with_lake_small.bna')
test_tri_grid = os.path.join(datadir, 'small_trigrid_example.nc')


def test_in_water_resolution():
    '''
    Test the limits of the precision, to within an order of magnitude,
    defining whether a point is in or out of water.
    '''

    # Create an 500x500 pixel map, with an LE refloat half-life of 2 hours
    # (specified here in seconds).
    m = MapFromBNA(filename=testbnamap, refloat_halflife=2,
                   raster_size=500 * 500)

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
        print('Order of magnitude: %g' % mag)
        running = m.in_water((px, py + eps * 10.0 ** mag, 0.))

    # Difference in position within an order of magnitude in
    # degrees of latitude necessary to "move" point from water to land.

    dlatO0 = eps * 10.0 ** (mag - 1.)
    dlatO1 = eps * 10.0 ** mag

    msg = \
        '''A particle positioned on a coastline segment must be moved
    something more than {0} meters, but less than {1} meters,
    inland before pyGNOME acknowledges it's no longer in water.'''
    print(msg.format(dlatO0 * 1852.0, dlatO1 * 1852.0))


# tests for GnomeMap -- the most basic version

class Test_GnomeMap:

    def test_on_map(self):
        gmap = GnomeMap()
        assert gmap.on_map((0., 0., 0.)) is True

        # too big latitude

        print(gmap.on_map((0., 91.0, 0.)))
        assert gmap.on_map((0., 91.0, 0.)) is False

        # too small latitude

        assert gmap.on_map((0., -91.0, 0.)) is False

        # too big langitude

        assert gmap.on_map((0., 361.0, 0.)) is False

        # too small langitude

        assert gmap.on_map((0., -361.0, 0.)) is False

    def test_on_land(self):
        gmap = GnomeMap()
        assert gmap.on_land((18.0, -87.0, 0.)) is False

    def test_in_water(self):
        gmap = GnomeMap()

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
        gmap = GnomeMap(map_bounds=map_bounds)

        points = ((-35, 55, 0.), (-45, 55, 0.))

        result = gmap.on_map(points)

        # some points on the map:
        assert np.array_equal(result, (True, False))

    def test_allowable_spill_position(self):
        gmap = GnomeMap()

        assert gmap.allowable_spill_position((18.0, -87.0, 0.)) is True
        assert gmap.allowable_spill_position((370.0, -87.0, 0.)) is False

    def test_update_from_dict(self):
        gmap = GnomeMap()

        json_ = {'map_bounds': [(-10, 10), (10, 10),
                                (10, -10), (-10, -10)]}
        assert np.all(gmap.map_bounds != json_['map_bounds'])
        gmap.update_from_dict(json_)
        assert np.all(gmap.map_bounds == json_['map_bounds'])


class Test_ParamMap:
    '''
    WIP

    Not sure where to go with these.
    '''

    def test_on_map(self):
        pmap = ParamMap((0, 0), 10000, 90)
        assert pmap.on_map((0, 0, 0))
        assert pmap.on_map((15, 0, 0)) is False

    def test_on_land(self):
        pmap = ParamMap((0, 0), 10000, 90)
        assert pmap.on_land((0.3, 0, 0)) is True
        assert pmap.on_land((-0.3, 0, 0)) is False

    def test_in_water(self):
        pmap = ParamMap((0, 0), 10000, 90)
        assert pmap.in_water((-0.3, 0, 0)) is True
        assert pmap.in_water((0.3, 0, 0)) is False

    def test_land_generation(self):
        pmap1 = ParamMap((0, 0), 10000, 90)
        print(pmap1.land_points)

    def test_to_geojson(self):
        pmap = ParamMap((0, 0), 10000, 90)
        geo_json = pmap.to_geojson()

        assert geo_json['type'] == 'FeatureCollection'
        assert 'features' in geo_json

        for f in geo_json['features']:
            assert 'type' in f
            assert 'geometry' in f
            assert 'coordinates' in f['geometry']
            for coord_coll in f['geometry']['coordinates']:
                assert len(coord_coll) == 1

                # This is the level where the individual coordinates are
                assert len(coord_coll[0]) > 1
                for c in coord_coll[0]:
                    assert len(c) == 2

    def test_serialize_deserialize_param(self):
        """
        test create new ParamMap from deserialized dict
        """
        pmap = ParamMap((5, 5), 12000, 40)

        serial = pmap.serialize()
        pmap2 = ParamMap.deserialize(serial)

        assert pmap == pmap2

    def test_update_from_dict_param(self):
        """
        test create new ParamMap from deserialized dict
        """
        map1 = ParamMap((5, 5), 12000, 40)
        assert map1.center == (5, 5, 0)

        json_ = {'center': [6, 6]}
        map1.update_from_dict(json_)
        assert map1.center == (6, 6, 0)


class Test_RasterMap:
    """
    some tests for the raster map
    """
    # a very simple raster:

    (w, h) = (20, 12)
    raster = np.zeros((w, h), dtype=np.uint8)

    # set some land in middle:
    raster[6:13, 4:8] = 1

    def test__off_raster(self):
        """
        test the _on_raster method
        """
        # overkill for just the raster..
        rmap = RasterMap(refloat_halflife=6,
                         raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # the corners
        assert not rmap._off_raster((0, 0))
        assert not rmap._off_raster((19, 0))
        assert not rmap._off_raster((19, 11))
        assert not rmap._off_raster((0, 11))

        # in the middle somewhere
        assert not rmap._off_raster((10, 6))

        # just off the edges
        assert rmap._off_raster((-1, 0))
        assert rmap._off_raster((19, -1))
        assert rmap._off_raster((20, 11))
        assert rmap._off_raster((0, 12))

        # way off -- just for the heck of it.
        assert rmap._off_raster((-1000, -2000))
        assert rmap._off_raster((1000, 2000))

    def test_save_as_image(self, dump_folder):
        """
        only tests that it doesn't crash -- you need to look at the
        image to see if it's right
        """
        rmap = RasterMap(refloat_halflife=6,
                         raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        rmap.save_as_image(os.path.join(dump_folder, 'raster_map_image.png'))

        assert True

    def test_on_map(self):
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())  # hours

        assert gmap.on_map((0., 0., 0.))
        assert not gmap.on_map((55.0, 0., 0.))

    def test_on_land(self):
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # right in the middle
        print('testing a land point:', (10, 6, 0.), gmap.on_land((10, 6, 0.)))
        assert gmap.on_land((10, 6, 0.))

        print('testing a water point:')
        assert not gmap.on_land((19.0, 11.0, 0.))

    def test_spillable_area(self):
        # anywhere not on land is spillable...
        # in this case
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # right in the middle of land
        print('testing a land point:')
        assert not gmap.allowable_spill_position((10, 6, 0.))

        print('testing a water point:')
        assert gmap.allowable_spill_position((19.0, 11.0, 0.))

    def test_spillable_area2(self):
        # a test with a polygon spillable area
        poly = ((5, 2), (15, 2), (15, 10), (10, 10), (10, 5))
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection(),
                         spillable_area=[poly])

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
                    raster=np.zeros((20, 12), dtype=np.uint8),
                    projection=NoProjection(),
                    map_bounds=((-50, -30), (-50, 30),
                                (50, 30), (50, -30)))

    num_les = 1000
    spill = sample_sc_release(num_les)
    orig_pos = np.random.uniform(0, num_les, spill['positions'].shape)
    last_water = (1., 2., 0.)

    spill['positions'][:] = orig_pos
    spill['last_water_positions'] += last_water

    def reset(self):
        self.spill['positions'][:] = self.orig_pos
        self.spill['last_water_positions'][:] = self.last_water

        self.map.refloat_halflife = self.time_step / 3600.

    def test_all_elementsinwater(self):
        """
        all elements in water so do nothing
        """
        self.reset()  # reset _state
        self.spill['status_codes'][:] = oil_status.in_water

        self.map.refloat_elements(self.spill, self.time_step)

        assert np.all(self.spill['positions'] == self.orig_pos)
        assert np.all(self.spill['status_codes'] == oil_status.in_water)

    def test_refloat_halflife_0(self):
        """
        refloat_halflife is 0 so refloat all elements on land
        """
        self.reset()
        self.map.refloat_halflife = 0
        self.spill['status_codes'][5:] = oil_status.on_land

        self.map.refloat_elements(self.spill, self.time_step)

        assert np.all((self.spill['positions'])[:5] == self.orig_pos[:5])
        assert np.all((self.spill['positions'])[5:] == self.last_water)

    def test_refloat_halflife_negative(self):
        """
        refloat_halflife is test_refloat_halflife_negative:

        this should mean totally sticky --no refloat
        """
        self.reset()
        self.map.refloat_halflife = -1

        self.spill['status_codes'][5:] = oil_status.on_land
        orig_status_codes = self.spill['status_codes'].copy()

        self.map.refloat_elements(self.spill, self.time_step)

        assert np.all((self.spill['positions']) == self.orig_pos)
        assert np.all(self.spill['status_codes'] == orig_status_codes)

    def test_refloat_some_onland(self):
        """
        refloat elements on land based on probability
        """
        self.reset()

        self.spill['status_codes'][:] = oil_status.in_water
        self.map.refloat_halflife = 3 * self.time_step / 3600.

        # say 500 out of 1000 are on_land, and we expect about 50% of these
        # to refloat

        # initial 25% LEs on_land, last 25% of LEs on_land
        init_ix = int(round(.25 * self.num_les))
        last_ix = self.num_les - (int(round(.5 * self.num_les)) - init_ix)

        ix = list(range(init_ix))  # choose first 25% of indices
        ix.extend(range(last_ix, self.num_les, 1))  # last 25% of indices
        ix = np.asarray(ix)

        self.spill['status_codes'][ix] = oil_status.on_land

        self.map.refloat_elements(self.spill, self.time_step)

        expected = (round(1. - .5 ** (self.time_step /
                                      self.map.refloat_halflife *
                                      3600.), 2) * 100)

        actual = (np.count_nonzero(self.spill['status_codes'][ix] ==
                                   oil_status.in_water) /
                  (self.num_les / 2) * 100)

        print ('Expect {0}% refloat, actual refloated: {1}%'
               .format(expected, actual))

        # ensure some of the elements that were on land are back on water
        assert np.count_nonzero(self.spill['status_codes'][ix] ==
                                oil_status.in_water) > 0

        refloat_ix = ix[np.where(self.spill['status_codes'][ix] ==
                                 oil_status.in_water)[0]]

        assert np.all(self.spill['positions'][refloat_ix] ==
                      self.last_water)  # refloated elements
        assert np.all(self.spill['status_codes'][refloat_ix] ==
                      oil_status.in_water)  # status is back in water

        # ensure elements that were in_water are not changed
        # these are original values that are not refloated

        mask = np.array([i not in refloat_ix for i in
                         range(self.num_les)], dtype=bool)
        assert np.all(self.spill['positions'][mask, :] ==
                      self.orig_pos[mask, :])


class Test_MapfromBNA:

    print("instaniating map:", testbnamap)
    # NOTE: this is a pretty course map -- for testing
    bna_map = MapFromBNA(testbnamap, refloat_halflife=6, raster_size=1000)

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

    # def test_map_in_water2(self):
    #     # in water, but inside land Bounding box
    #     InWater = (-126.971456, 47.935608, 0.)
    #
    #     # Throw an error if the know in-water location returns false.
    #     assert self.bna_map.in_water(InWater)

    def test_map_on_land(self):
        '''
        Test whether the location of a particle on land is determined
        correctly.
        '''
        OnLand = (-127, 47.8, 0.)
        print("on land:", self.bna_map.on_land(OnLand))
        print(self.bna_map.raster)

        assert self.bna_map.on_land(OnLand)
        assert not self.bna_map.in_water(OnLand)

    def test_map_in_lake(self):
        '''
        Test whether the location of a particle in a lake
        is determined correctly.
        '''
        InLake = (-126.8, 47.84, 0.)

        assert self.bna_map.in_water(InLake)
        assert not self.bna_map.on_land(InLake)

    def test_map_spillable(self):
        in_water = (-126.984472, 48.08106, 0.)  # in water, in spillable

        assert self.bna_map.allowable_spill_position(in_water)

    def test_map_spillable_lake(self):
        in_lake = (-126.793592, 47.841064, 0.)  # in lake, should be spillable

        assert self.bna_map.allowable_spill_position(in_lake)

    def test_map_not_spillable(self):
        on_land = (-127, 47.8, 0.)  # on land should not be spillable

        # Throw an error if the know on-land location returns false.
        assert not self.bna_map.allowable_spill_position(on_land)

    def test_map_not_spillable2(self):
        # in water, but outside spillable area
        in_water_but_outside = (127.244752, 47.585072, 0.)

        assert not self.bna_map.allowable_spill_position(in_water_but_outside)

    def test_map_not_spillable3(self):
        # off the map -- should not be spillable
        off_map = (127.643856, 47.999608, 0.)

        assert not self.bna_map.allowable_spill_position(off_map)

    def test_map_on_map(self):
        point_on_map = (-126.12336, 47.454164, 0.)

        assert self.bna_map.on_map(point_on_map)

    def test_map_off_map(self):
        point_off_map = (-126.097336, 47.43962, 0.)

        assert not self.bna_map.on_map(point_off_map)

    def test_map_bounds(self):
        map_bounds = self.bna_map.map_bounds
        # these are the map_bounds in the BNA
        expected_bounds = np.array([[-127.465333, 48.3294],
                                    [-126.108847, 48.3294],
                                    [-126.108847, 47.44727],
                                    [-127.465333, 47.44727],
                                    ])
        assert np.allclose(map_bounds, expected_bounds)

    def test_to_geojson(self):
        geo_json = self.bna_map.to_geojson()

        assert geo_json['type'] == 'FeatureCollection'
        assert 'features' in geo_json

        for f in geo_json['features']:
            assert 'type' in f
            assert 'geometry' in f
            assert 'coordinates' in f['geometry']
            for coord_coll in f['geometry']['coordinates']:
                assert len(coord_coll) == 1

                # This is the level where the individual coordinates are
                assert len(coord_coll[0]) > 1
                for c in coord_coll[0]:
                    assert len(c) == 2

    def test_serialize_deserialize(self):
        """
        test create new object from to_dict
        """
        gmap = MapFromBNA(testbnamap, 6)

        serial = gmap.serialize()
        map2 = MapFromBNA.deserialize(serial)

        assert gmap == map2

    def test_update_from_dict_MapFromBNA(self):
        'test update_from_dict for MapFromBNA'
        gmap = MapFromBNA(testbnamap, 6)

        dict_ = {}
        dict_['map_bounds'] = [(-10, 10), (10, 10), (10, -10), (-10, -10)]
        dict_['spillable_area'] = [[(-5, 5), (5, 5), (5, -5), (-5, -5)]]
        dict_['refloat_halflife'] = 2
        assert np.all(gmap.map_bounds != dict_['map_bounds'])

        gmap.update_from_dict(dict_)
        assert gmap.map_bounds is not dict_['map_bounds']
        assert np.all(gmap.map_bounds == dict_['map_bounds'])


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
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # making sure the gmap is set up right
        assert not gmap.on_map((100.0, 1., 0.))
        assert gmap.on_map((0., 1., 0.))

    def test_on_land(self):
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        assert gmap.on_land((10, 3, 0)) == 1
        assert gmap.on_land((9, 3, 0)) == 0
        assert gmap.on_land((11, 3, 0)) == 0

    def test_starts_on_land(self):
        """
        try a single LE that starts on land

        it last water position should be the same point.
        """
        gmap = RasterMap(refloat_halflife=6,
                         raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        spill = sample_sc_release(1)

        spill['positions'] = np.array(((10.0, 5.0, 0.), ), dtype=np.float64)
        spill['last_water_positions'] = np.array(((0.0, 0.0, 0.), ),
                                                 dtype=np.float64)
        spill['next_positions'] = np.array(((15.0, 5.0, 0.), ),
                                           dtype=np.float64)
        spill['status_codes'] = np.array((oil_status.in_water, ),
                                         dtype=status_code_type)

        gmap.beach_elements(spill)

        # next position gets set to land location
        assert np.array_equal(spill['next_positions'][0], (10.0, 5.0, 0.))
        assert np.array_equal(spill['last_water_positions'][0],
                              (10.0, 5.0, 0.))
        assert spill['status_codes'][0] == oil_status.on_land

    def test_land_cross(self):
        """
        try a single LE that should be crossing land
        """
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        spill = sample_sc_release(1)

        spill['positions'] = np.array(((5.0, 5.0, 0.), ), dtype=np.float64)
        spill['next_positions'] = np.array(((15.0, 5.0, 0.), ),
                                           dtype=np.float64)
        spill['status_codes'] = np.array((oil_status.in_water, ),
                                         dtype=status_code_type)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'][0], (10.0, 5.0, 0.))
        assert np.array_equal(spill['last_water_positions'][0], (9.0, 5.0, 0.))
        assert spill['status_codes'][0] == oil_status.on_land

    def test_land_cross_array(self):
        """
        test a few LEs
        """
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # one left to right
        # one right to left
        # one diagonal upper left to lower right
        # one diagonal upper right to lower left

        spill = sample_sc_release(4)

        spill['positions'] = np.array(((5.0, 5.0, 0.), (15.0, 5.0, 0.),
                                       (0., 0., 0.), (19.0, 0., 0.)),
                                      dtype=np.float64)
        spill['next_positions'] = np.array(((15.0, 5.0, 0.), (5.0, 5.0, 0.),
                                            (10.0, 5.0, 0.), (0., 9.0, 0.)),
                                           dtype=np.float64)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'],
                              ((10.0, 5.0, 0.), (10.0, 5.0, 0.),
                               (10.0, 5.0, 0.), (10.0, 4.0, 0.)))

        assert np.array_equal(spill['last_water_positions'],
                              ((9.0, 5.0, 0.), (11.0, 5.0, 0.),
                               (9.0, 4.0, 0.), (11.0, 4.0, 0.)))

        assert np.alltrue(spill['status_codes'] == oil_status.on_land)

    def test_some_cross_array(self):
        """
        test a few LEs
        """
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # one left to right
        # one right to left
        # diagonal that doesn't hit
        # diagonal that does hit

        spill = sample_sc_release(4)

        spill['positions'] = np.array(((5.0, 5.0, 0.), (15.0, 5.0, 0.),
                                       (0., 0., 0.), (19.0, 0., 0.)),
                                      dtype=np.float64)

        spill['next_positions'] = np.array(((9.0, 5.0, 0.), (11.0, 5.0, 0.),
                                            (9.0, 9.0, 0.), (0., 9.0, 0.)),
                                           dtype=np.float64)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'],
                              ((9.0, 5.0, 0.), (11.0, 5.0, 0.),
                               (9.0, 9.0, 0.), (10.0, 4.0, 0.)))

        # just the beached ones
        assert np.array_equal((spill['last_water_positions'])[3:],
                              ((11.0, 4.0, 0.), ))

        assert np.array_equal((spill['status_codes'])[3:],
                              (oil_status.on_land, ))

    def test_outside_raster(self):
        """
        test LEs starting form outside the raster bounds
        """

        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        # one left to right
        # one right to left
        # diagonal that doesn't hit
        # diagonal that does hit
        # spill = gnome.spills.Spill(num_LEs=4)
        spill = sample_sc_release(4)
        spill['positions'] = np.array(((30.0, 5.0, 0.), (-5.0, 5.0, 0.),
                                       (5.0, -5.0, 0.), (-5.0, -5.0, 0.)),
                                      dtype=np.float64)

        spill['next_positions'] = np.array(((15.0, 5.0, 0.), (5.0, 5.0, 0.),
                                            (5.0, 15.0, 0.), (25.0, 15.0, 0.)),
                                           dtype=np.float64)

        gmap.beach_elements(spill)

        assert np.array_equal(spill['next_positions'],
                              ((15.0, 5.0, 0.), (5.0, 5.0, 0.),
                               (5.0, 15.0, 0.), (10.0, 5.0, 0.)))

        # just the beached ones
        assert np.array_equal((spill['last_water_positions'])[3:],
                              ((9.0, 4.0, 0.), ))

        assert np.array_equal((spill['status_codes'])[3:],
                              (oil_status.on_land, ))

    def test_some_off_map(self):
        """
        Test LEs that go off the map

        should get off_map flag - no longer setting to_be_removed flag. map
        simply sets the off_maps flag.
        """
        gmap = RasterMap(refloat_halflife=6, raster=self.raster,
                         map_bounds=((-50, -30), (-50, 30),
                                     (50, 30), (50, -30)),
                         projection=NoProjection())

        spill = sample_sc_release(8)
        spill['positions'] = np.array(((45.0, 25.0, 0.),
                                       (45.0, 25.0, 0.),
                                       (45.0, -25.0, 0.),
                                       (45.0, -25.0, 0.),
                                       (-45.0, -25.0, 0.),
                                       (-45.0, -25.0, 0.),
                                       (-45.0, 25.0, 0.),
                                       (-45.0, 25.0, 0.)),
                                      dtype=np.float64)

        # off
        # still on
        # off
        # still on
        # off
        # still on
        # off
        # still on
        spill['next_positions'] = np.array(((55.0, 25.0, 0.),
                                            (49.0, 25.0, 0.),
                                            (45.0, -35.0, 0.),
                                            (45.0, -29.0, 0.),
                                            (-55.0, -25.0, 0.),
                                            (-49.0, -25.0, 0.),
                                            (-45.0, 35.0, 0.),
                                            (-45.0, 29.0, 0.)),
                                           dtype=np.float64)

        gmap.beach_elements(spill)

        off = np.ones(4,) * oil_status.off_maps
        assert np.array_equal(spill['status_codes'][0::2], off)

        on = np.ones(4) * oil_status.in_water
        assert np.array_equal(spill['status_codes'][1::2], on)


def test_resurface_airborne_elements():
    positions = np.array(((1, 2, 0.),
                          (3, 4, 1.),
                          (-3, 4, -1.),
                          (3, 4, 0.1),
                          (3, 4, 0.1)),
                         dtype=np.float64)

    spill = {'next_positions': positions}

    m = GnomeMap()
    m.resurface_airborne_elements(spill)

    assert spill['next_positions'][:, 2].min() == 0.


def test_bna_no_map_bounds():
    """
    tests that the map bounds will get expanded to include
    the bounding box of the land and spillable area.
    """
    test_no_bounds_bna = os.path.join(datadir, 'no_map_bounds.bna')
    m = MapFromBNA(test_no_bounds_bna)

    assert np.array_equal(m.map_bounds, [(3., 10.),
                                         (3., 11.),
                                         (6., 11.),
                                         (6., 10.),
                                         ])


class Test_lake():
    """
    tests for handling a BNA with a lake

    The code should now return a polygon with a hole for lakes.

    And render properly both with the py_gnome renderer and the json output.

    """
    map = MapFromBNA(bna_with_lake)

    def test_polys(self):
        """
        Once loaded, polygons should be there
        """
        # There should always be map bounds
        assert self.map.map_bounds is not None

        # no spillable area in this one
        assert self.map.spillable_area is None

        # NOTE: current version puts land and lakes in the land_polys set
        assert len(self.map.land_polys) == 2

    def test_to_geojson(self):
        """
        make sure geojson is right
        """

        gj = self.map.to_geojson()

        # has only the land polys in there.
        assert len(gj['features']) == 2

        land_polys = gj['features'][0]
        assert land_polys['geometry']['type'] == "MultiPolygon"
        assert land_polys["properties"]["name"] == "Shoreline Polys"

        import json
        outfilename = os.path.join(output_dir, "florida_geojson.json")
        with open(outfilename, 'w') as outfile:
            json.dump(gj, outfile, indent=2)

class Test_serialize:

    map = MapFromBNA(bna_with_lake)

    def test_serialize(self):
        ser = self.map.to_dict()
        print(ser.keys())

        assert ser['spillable_area'] is None

    #need to add the file to file server
    @pytest.mark.xfail()
    def test_serialize_from_blob_old(self):
        # this one uses the "old" name, before moving the map module.
        json_data = {'approximate_raster_interval': 53.9608870724,
                     'filename': u'/Users/chris.barker/Hazmat/GitLab/pygnome/py_gnome/tests/unit_tests/sample_data/florida_with_lake_small.bna',
                     'id': u'b3590b7d-aab1-11ea-8899-1e00b098d304',
                     'map_bounds': [(-82.8609915978, 24.5472415066),
                                    (-82.8609915978, 28.1117673335),
                                    (-80.0313642811, 28.1117673335),
                                    (-80.0313642811, 24.5472415066)],
                     'name': u'MapFromBNA_8',
                     'obj_type': u'gnome.map.MapFromBNA',
                     'raster_size': 16777216.0,
                     'spillable_area': None,
                     'refloat_halflife': 1.0}

        cls = class_from_objtype(json_data['obj_type'])
    #   obj = cls.load(saveloc, fname, references)

        print("found class:", cls)
        map = cls.deserialize(json_data)

        # when we go to Python3 :-(
        # assert map.__class__.__qualname__ == "gnome.maps.map.MapFromBNA"
        assert map.__class__.__name__ == "MapFromBNA"
        assert map.__class__.__module__ == "gnome.maps.map"

        assert map.spillable_area is None
        assert len(map.map_bounds) == 4

    @pytest.mark.xfail()
    def test_serialize_from_blob_new(self):
        # this one uses the "new" name, after moving the map module.
        json_data = {'approximate_raster_interval': 53.9608870724,
                     'filename': u'/Users/chris.barker/Hazmat/GitLab/pygnome/py_gnome/tests/unit_tests/sample_data/florida_with_lake_small.bna',
                     'id': u'b3590b7d-aab1-11ea-8899-1e00b098d304',
                     'map_bounds': [(-82.8609915978, 24.5472415066),
                                    (-82.8609915978, 28.1117673335),
                                    (-80.0313642811, 28.1117673335),
                                    (-80.0313642811, 24.5472415066)],
                     'name': u'MapFromBNA_8',
                     'obj_type': u'gnome.maps.map.MapFromBNA',
                     'raster_size': 16777216.0,
                     'spillable_area': None,
                     'refloat_halflife': 1.0}

        cls = class_from_objtype(json_data['obj_type'])
    #   obj = cls.load(saveloc, fname, references)

        map = cls.deserialize(json_data)

        # when we go to Python3 :-(
        # assert map.__class__.__qualname__ == "gnome.maps.map.MapFromBNA"
        assert map.__class__.__name__ == "MapFromBNA"
        assert map.__class__.__module__ == "gnome.maps.map"

        assert map.spillable_area is None
        assert len(map.map_bounds) == 4

    def test_deserialize(self):
        ## fixme: this should fail with no spillable area ?!?
        jsblob = self.map.serialize()

        jsblob['spillable_area'] = None

        # {'approximate_raster_interval': 53.9608870724,
        #  'filename': u'/Users/chris.barker/Hazmat/GitLab/pygnome/py_gnome/tests/unit_tests/sample_data/florida_with_lake_small.bna',
        #  'id': u'b3590b7d-aab1-11ea-8899-1e00b098d304',
        #  'map_bounds': [(-82.8609915978, 24.5472415066),
        #                 (-82.8609915978, 28.1117673335),
        #                 (-80.0313642811, 28.1117673335),
        #                 (-80.0313642811, 24.5472415066)],
        #  'name': u'MapFromBNA_8',
        #  'obj_type': u'gnome.maps.map.MapFromBNA',
        #  'raster_size': 16777216.0,
        #  'refloat_halflife': 1.0}

        map = MapFromBNA.deserialize(jsblob)

        assert map.spillable_area is None
        assert len(map.map_bounds) == 4

    def test_update_spillable_area_none(self):
        map = MapFromBNA(bna_with_lake)

        map.update_from_dict({'spillable_area': None})

    def test_update_spillable_area_polygons(self):
        map = MapFromBNA(bna_with_lake)

        map.update_from_dict({'spillable_area': [[(-82.86099, 24.54724),
                                                  (-82.86099, 28.11176),
                                                  (-80.03136, 28.11176),
                                                  (-80.03136, 24.54724)],
                                                 [(-82.86099, 24.54724),
                                                  (-82.86099, 28.11176),
                                                  (-80.03136, 28.11176),
                                                  (-80.03136, 24.54724),
                                                  (-80.03136, 28.11176),
                                                  (-80.03136, 24.54724),
                                                  ],
                                                 ]
                              })

        assert len(map.spillable_area) == 2
        assert len(map.spillable_area[1]) == 6
