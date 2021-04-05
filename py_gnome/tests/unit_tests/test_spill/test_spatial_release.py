"""
tests for the spatial release from polygons:

e.g. from the NESDIS MPSR reports
"""
from __future__ import print_function

import os
import numpy as np
import datetime

from gnome.spill.release import NESDISRelease

data_dir = os.path.join(os.path.split(__file__)[0], "data_for_tests")

sample_shapefile = os.path.join(data_dir, "NESDIS_files.zip")


def check_valid_polygon(poly):
    """
    checks that a shapely Polygon object at least has valid values for coordinates
    """
    for point in poly.exterior.coords:
        assert -360 < point[0] < 360
        assert -90 < point[0] < 90


def test_load_shapefile():
    (release_time,
     all_oil_polys,
     all_oil_weights,
     all_oil_thicknesses,
     all_oil_areas,
     all_oil_types) = NESDISRelease.load_shapefile(sample_shapefile)

    assert release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')
    assert len(all_oil_polys) == 8
    assert len(all_oil_weights) == 8
    assert len(all_oil_thicknesses) == 2
    assert len(all_oil_areas) == 2
    assert len(all_oil_types) == 2

    for poly in all_oil_polys:
        check_valid_polygon(poly)

    # NOTE: these values are pulled from running the code
    #       they may not be correct, but this will let us catch changes
    assert np.allclose(all_oil_weights, [0.0019291097691711862, 0.0018247639782104231,
                                         0.09568991387647877, 0.00017874329138003873,
                                         9.309062636361091e-05, 0.005663950543120452,
                                         0.001098505440460224, 0.8935219224748153
                                         ], rtol=1e-12)

    assert np.allclose(all_oil_thicknesses, [5e-06, 0.0002
                                             ], rtol=1e-12)

def test_construct_from_shapefile():
    rel = NESDISRelease(filename=sample_shapefile)
    assert rel.release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')
    assert rel.end_release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')
    assert len(rel.polygons) == 8
    assert len(rel.weights) == 8
    assert len(rel.thicknesses) == 2

    for poly in rel.polygons:
        check_valid_polygon(poly)

    # NOTE: these values are pulled from running the code
    #       they may not be correct, but this will let us catch changes
    assert np.allclose(rel.weights, [0.0019291097691711862, 0.0018247639782104231,
                                         0.09568991387647877, 0.00017874329138003873,
                                         9.309062636361091e-05, 0.005663950543120452,
                                         0.001098505440460224, 0.8935219224748153
                                         ], rtol=1e-12)


