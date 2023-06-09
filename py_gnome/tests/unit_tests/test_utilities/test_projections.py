#!/usr/bin/env python

"""
tests of the gnome.utilities.projections module
"""

import pytest
import numpy as np

from gnome.utilities import projections
from gnome.utilities.projections import to_2d_coords


def test_NoProjection():
    proj = projections.NoProjection()

    coords = np.array(((45.5, 32.1, 0.0), (-18.3, -12.6, 0.0)))

    result = [[45, 32], [-18, -12]]

    proj_coords = proj.to_pixel(coords, asint=True)

    assert np.array_equal(proj_coords, result)

    proj_coords = proj.to_pixel(coords, asint=False)

    assert np.array_equal(proj_coords, coords[:, :2])


class Test_GeoProjection(object):

    bounding_box = ((-10.0, 23.0), (-5, 33.0))

    image_size = (500, 500)

    proj = projections.GeoProjection(bounding_box, image_size)

    def test_bounds(self):

        # corners of the BB:

        coords = ((-10.0, 23.0, 0.0), (-5, 33.0, 0.0), (-10, 33, 0.0),
                  (-5, 23.0, 0.0))

        proj_coords = self.proj.to_pixel(coords, asint=True)

        print(proj_coords)

        assert np.array_equal(proj_coords, [[125, 500], [375, 0], [125,
                              0], [375, 500]])

    def test_middle(self):

        # middle of the BB

        coords = ((-7.5, 28.0, 0.0), )

        proj_coords = self.proj.to_pixel(coords, asint=True)

        print(proj_coords)

        assert np.array_equal(proj_coords, [[250, 250]])

    def test_outside(self):
        """
        points outside the BB should still come back, but the pixel coords
        will be outside the image shape
        """

        # just outside the bitmap

        coords = ((-12.500001, 22.99999999, 0.0), (-2.499999,
                  33.00000001, 0.0))

        print('scale:', self.proj.scale)

        proj_coords = self.proj.to_pixel(coords, asint=True)
        print(proj_coords)

        assert np.array_equal(proj_coords, [[-1, 500], [500, -1]])

    def test_reverse(self):
        # middle of the BB
        # some non-round numbers
        # outside the bitmap
        coords = np.array(((-7.5, 28.0, 0.0), (-8.123, 25.345, 0.0),
                          (-2.500001, 33.00000001, 0.0), (-4.2345,
                          32.123, 0.0)))

        # first test without rounding
        proj_coords = self.proj.to_pixel(coords, asint=False)
        back_coords = self.proj.to_lonlat(proj_coords)
        print(coords)
        print(proj_coords)
        print(repr(back_coords))
        assert np.allclose(coords[:, :2], back_coords)

        # now with the pixel rounding
        proj_coords = self.proj.to_pixel(coords, asint=True)
        back_coords = self.proj.to_lonlat(proj_coords)
        print(coords)
        print(proj_coords)
        print(repr(back_coords))

        # tolerence set according to the scale:
        tol = 1. / self.proj.scale[0] / 1.5  # should be about 1/2 pixel

        # rtol tiny so it really doesn't matter
        assert np.allclose(coords[:, :2], back_coords, rtol=1e-100,
                           atol=tol)

    def test_equality(self):
        proj1 = projections.GeoProjection(self.bounding_box, self.image_size)
        proj2 = projections.GeoProjection(self.bounding_box, self.image_size)
        print(proj1)
        print(proj2)
        assert proj1 == proj2
        assert not proj1 != proj2

    def test_not_equal(self):
        proj1 = projections.GeoProjection(self.bounding_box, self.image_size)
        proj2 = projections.GeoProjection(self.bounding_box, self.image_size)
        proj2.image_size = (500, 510)
        proj2.set_scale(self.bounding_box)
        assert proj1 != proj2
        assert not proj1 == proj2


class Test_FlatEarthProjection(object):
    # bb with 60 degrees in the center: ( cos(60 deg) == 0.5 )
    bounding_box = ((20, 50.0), (40, 70.0))

    image_size = (500, 500)

    proj = projections.FlatEarthProjection(bounding_box, image_size)

    def test_bounds(self):
        # corners of the BB: (sqare in lat-long)
        coords = ((20.0, 50.0, 0.0), (20.0, 70.0, 0.0),
                  (40.0, 50.0, 0.0), (40.0, 70.0, 0.0))

        proj_coords = self.proj.to_pixel(coords, asint=True)
        print(proj_coords)

        assert np.array_equal(proj_coords,
                              [[124, 500], [124, 0],
                               [375, 500], [375, 0]])

    def test_middle(self):
        # middle of the BB
        coords = ((30, 60.0, 0.0), )

        proj_coords = self.proj.to_pixel(coords, asint=True)
        print(proj_coords)

        assert np.array_equal(proj_coords, [[250, 250]])

    def test_outside(self):
        """
        points outside the BB should still come back, but the pixel coords
        will be outside the image shape
        """
        # just outside the bitmap
        coords = ((9.9999999, 49.99999999, 0.0),
                  (50.0000001, 70.000000001, 0.0))

        print('scale:', self.proj.scale)

        proj_coords = self.proj.to_pixel(coords, asint=True)
        print(proj_coords)

        assert np.array_equal(proj_coords, [[-1, 500], [500, -1]])

    def test_reverse(self):
        # middle of the BB
        # some non-round numbers
        # outside the bitmap
        coords = np.array(((-7.5, 28.0, 0.0), (-8.123, 25.345, 0.0),
                           (-2.500001, 33.00000001, 0.0), (-4.2345, 32.123, 0.0)))

        # first test without rounding
        proj_coords = self.proj.to_pixel(coords, asint=False)
        back_coords = self.proj.to_lonlat(proj_coords)
        print(coords)
        print(proj_coords)
        print(repr(back_coords))
        assert np.allclose(coords[:, :2], back_coords)

        # now with the pixel rounding
        proj_coords = self.proj.to_pixel(coords, asint=True)
        back_coords = self.proj.to_lonlat(proj_coords)
        print(coords)
        print(proj_coords)
        print(repr(back_coords))

        # tolerence set according to the scale:
        tol = 1. / self.proj.scale[0] / 1.5  # should be about 1/2 pixel

        # rtol tiny so it really doesn't matter
        assert np.allclose(coords[:, :2], back_coords, rtol=1e-100,
                           atol=tol)

    def test_equality(self):
        proj1 = projections.FlatEarthProjection(self.bounding_box,
                                                self.image_size)
        proj2 = projections.FlatEarthProjection(self.bounding_box,
                                                self.image_size)
        print(proj1)
        print(proj2)
        assert proj1 == proj2
        assert not proj1 != proj2

    def test_not_equal(self):
        proj1 = projections.FlatEarthProjection(self.bounding_box,
                                                self.image_size)
        proj2 = projections.FlatEarthProjection(self.bounding_box,
                                                self.image_size)
        proj2.image_size = (500, 510)
        proj2.set_scale(self.bounding_box)
        assert proj1 != proj2
        assert not proj1 == proj2


# tests for meters_to_lonlat
m2l = projections.FlatEarthProjection.meters_to_lonlat
METERS_PER_DEGREE_GNOME = 111119.9994764


def test_meters_to_lonlat():
    """ distance at equator """
    assert np.allclose(m2l((METERS_PER_DEGREE, METERS_PER_DEGREE_GNOME, 0.0),
                           (0.0, 0.0, 0.0)), (1., 1., 0.0))


def test_meters_to_lonlat2():
    """ distance at 60 deg north (1/2) """
    assert np.allclose(m2l((METERS_PER_DEGREE_GNOME, METERS_PER_DEGREE_GNOME, 4.5),
                           (0.0, 60.0, 0.0)), (2.0, 1., 4.5))


def test_meters_to_lonlat3():
    """ distance at 90 deg north: it should get very large!"""
    dlonlat = m2l((METERS_PER_DEGREE_GNOME, METERS_PER_DEGREE_GNOME, 0.0),
                  (30.0, 90.0, 0.0))

    # somewhat arbitrary...it should be infinity, but apparently not
    # with fp rounding
    assert dlonlat[0, 0] > 1e16


# tests for lonlat_to_meters
l2m = projections.FlatEarthProjection.lonlat_to_meters
METERS_PER_DEGREE_GNOME = 111119.9994764


def test_meters_to_latlon():
    """ distance at equator """
    assert np.allclose(l2m((1., 1., 0.0), (0.0, 0.0, 0.0)),
                       (METERS_PER_DEGREE, METERS_PER_DEGREE_GNOME, 0.0))


def test_meters_to_latlon2():
    """ distance at 60 deg north (1/2) """
    assert np.allclose(l2m((2.0, 1., 4.5), (0.0, 60.0, 0.0)),
                       (METERS_PER_DEGREE_GNOME, METERS_PER_DEGREE_GNOME, 4.5))


def test_meters_to_latlon3():
    """ distance at 90 deg north: it should get very small!"""
    delta_meters = l2m((0.01, 1., 0.0), (30.0, 90.0, 0.0))
    print(delta_meters)

    # should be zero -- but with floating point...
    assert delta_meters[0][0] <= 1e-13
    assert np.allclose(delta_meters[0][1], METERS_PER_DEGREE_GNOME)


# Some tests for the round-trip -- meters to lon-lat and back
d_lonlat = [(1., 1., 0), (10.0, -10.0, 0), (-2.0, -10.0, 0)]

refs = [(0.0, 0.0, 0.0),
        (0.0, 30.0, 0.0),
        (0.0, -30.0, 0.0),
        (0.0, 60.0, 0.0),
        (0.0, -60.0, 0.0),
        (0.0, 90.0, 0.0)]

examples = [(l, r) for l in d_lonlat for r in refs]


@pytest.mark.parametrize(('d_lonlat', 'ref'), examples)
def test_round_trip(d_lonlat, ref):
    # d_lonlat = (1.0, 1.0, 0)
    # ref = (0.0, 0.0, 0.0)

    assert np.allclose(d_lonlat, m2l(l2m(d_lonlat, ref), ref))


@pytest.mark.parametrize(('d_meters', 'ref'), examples)
def test_round_trip_reverse(d_meters, ref):
    assert np.allclose(d_meters, l2m(m2l(d_meters, ref), ref))


##################################################
# test the geodesic on the sphere code:
##################################################
geodesic_sphere = projections.FlatEarthProjection.geodesic_sphere

# maybe a better value, but we want to match GNOME
# METERS_PER_DEGREE = 111195.11
METERS_PER_DEGREE = METERS_PER_DEGREE_GNOME


def test_near_equatorE():
    """ directly east on the equator """
    (lon, lat) = geodesic_sphere(30, 0.0, METERS_PER_DEGREE, 90.0)
    print(lon, lat)

    assert round(lon, 6) == 31.0
    assert round(lat, 15) == 0.0


def test_near_equatorW():
    """ directly west on the equator """
    (lon, lat) = geodesic_sphere(-30.0, 0.0, METERS_PER_DEGREE, 270.0)
    print(lon, lat)

    assert round(lon, 6) == -31.0
    assert round(lat, 15) == 0.0


def test_near_equatorN():
    """ directly north on the equator """
    (lon, lat) = geodesic_sphere(30, 0.0, METERS_PER_DEGREE, 0.0)
    print(lon, lat)

    assert round(lon, 6) == 30.0
    assert round(lat, 6) == 1.


def test_near_equatorS():
    """ directly south on the equator """
    (lon, lat) = geodesic_sphere(30, 0.0, METERS_PER_DEGREE, 180.0)
    print(lon, lat)

    assert round(lon, 6) == 30.0
    assert round(lat, 6) == -1.


def test_near_equatorNE():
    """ directly northeast from the equator """
    (lon, lat) = geodesic_sphere(0.0, 0.0, METERS_PER_DEGREE, 45.0)
    print(lon, lat)

    # these values from the online geodesic calculator
    # (which uses and elipsoidal earth)
    # http://geographiclib.sourceforge.net/cgi-bin/Geod
    acc = 2  # almost good to 3 decimal place -- still not great!

    assert round(lon, acc) == round(0.70635273, acc)
    assert round(lat, acc) == round(0.71105843, acc)


def test_north_NE():
    """ directly northeast from north of the equator """
    (lon, lat) = geodesic_sphere(0.0, 60.0, METERS_PER_DEGREE, 45.0)
    print(lon, lat)

    # these values from the online geodesic calculator
    # (which uses and elipsoidal earth)
    # http://geographiclib.sourceforge.net/cgi-bin/Geod
    acc = 2  # almost good to 3 decimal place -- still not great!

    assert round(lon, acc) == round(1.43959, acc)
    assert round(lat, acc) == round(60.69799, acc)


class Test_rectangular_grid_projection(object):
    """
    Test the RectangularGridProjection with a small irregularly spaced grid
    """
    # small irregular grid
    lon = (20, 22, 25, 29, 34)
    lat = (0, 2, 5, 9, 12, 14)
    proj = projections.RectangularGridProjection(lon, lat)

    def test_init(self):
        """
        can one be initialized ?
        """
        proj = projections.RectangularGridProjection(self.lon, self.lat)

    def test_to_pixel_corners(self):
        proj = self.proj

        # check the corners:
        assert np.array_equal(proj.to_pixel(((20, 0, 0),
                                             (22, 2, 0),
                                             (25, 5, 0),
                                             (29, 9, 0),
                                             (34, 12, 0),
                                             (29, 12, 0),
                                             (34, 14, 0),
                                             )),
                              ((0, 5),
                               (1, 4),
                               (2, 3),
                               (3, 2),
                               (4, 1),
                               (3, 1),
                               (4, 0),
                               ))

    def test_to_pixel_bounds(self):
        proj = self.proj

        # outer bounds:
        assert np.array_equal(proj.to_pixel(((34,  0, 0),
                                             (34, 14, 0),
                                             (20, 14, 0)
                                             )),
                              ((4, 5),
                               (4, 0),
                               (0, 0)
                               ))

    def test_to_pixel_mid_float(self):
        proj = self.proj

        # mid_pixel as float:
        assert np.allclose(proj.to_pixel(((21,  1, 0),
                                          (24,  4, 0),
                                          (33.9999999, 8.9999999, 0)
                                          ), asint=False),
                           ((0.5, 4.5),
                            (1.66666666, 3.33333333),
                            (3.99999999, 2.000000001),
                            ), rtol=1e-8)

    def test_to_pixel_mid_int(self):
        proj = self.proj

        # mid_pixel as integer:
        assert np.array_equal(proj.to_pixel(((21,  1, 0),
                                             (24,  4, 0),
                                             (33.9999999, 8.9999999, 0)
                                             ), asint=True),
                              ((0, 4),
                               (1, 3),
                               (3, 2),
                               ))

    # def test_to_pixel_out_of_bounds(self):
    #    #NOTE: this no longer returns nans -- now returns outer edge

    #     proj = self.proj

    #     assert np.all(np.isnan(  proj.to_pixel( ( (10,  -1, 0),
    #                                               (34.00001, -.000000001, 0),
    #                                              )
    #                                            )
    #                 ))

    def test_to_lonlat_int(self):
        proj = self.proj

        assert np.array_equal(proj.to_lonlat(((0, 0),
                                              (1, 1),
                                              (2, 4),
                                              (3, 4))),
                              ((21, 13),
                               (23.5, 10.5),
                               (27, 1),
                               (31.5, 1))
                              )

    def test_to_lonlat_float_corners(self):
        proj = self.proj

        assert np.array_equal(proj.to_lonlat(((0., 0.),
                                              (1., 1.),
                                              (2., 4.),
                                              (3., 4.),
                                              (4., 5.))),
                              ((20, 14),
                               (22, 12),
                               (25,  2),
                               (29,  2),
                               (34,  0))
                              )

    def test_to_lonlat_float_middle(self):
        proj = self.proj

        assert np.allclose(proj.to_lonlat(((0.5, 0.5),
                                           (1.3333333333333, 2.5),
                                           (2.5, 4.5))),
                           ((21., 13.),
                            (23., 7.),
                            (27., 1.),
                            ),
                           rtol=1e-8)

    def test_to_lonlat_out_of_bounds(self):
        proj = self.proj

        assert np.array_equal(proj.to_lonlat(((-1, -1),
                                              (5, 6))),
                              ((20, 14),
                               (34,  0))
                              )

    def test_to_pixel_out_of_bounds(self):
        proj = self.proj

        assert np.array_equal(proj.to_pixel((10, -1, 0)),
                              ((0, 5),)
                              )

        assert np.array_equal(proj.to_pixel((35, 15, 0)),
                              ((4, 0),)
                              )


def test_to_2d_coords_2point():
    assert np.array_equal(to_2d_coords((2, 3)), ((2.0, 3.0),))


def test_to_2d_coords_3point():
    assert np.array_equal(to_2d_coords((2, 3, 4)), ((2.0, 3.0),))


def test_to_2d_coords_2points():
    assert np.array_equal(to_2d_coords(((2, 3), (3, 4), (5, 6))),
                                       ((2, 3), (3, 4), (5, 6))
                          )


def test_to_2d_coords_3points():
    assert np.array_equal(to_2d_coords(((2, 3, 4), (3, 4, 5), (5, 6, 7))),
                                       ((2, 3), (3, 4), (5, 6))
                          )
