"""
tests for the spatial release from polygons:

e.g. from the NESDIS MPSR reports
"""
from __future__ import print_function

import os
import numpy as np

from gnome.spill.release import SpatialRelease

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
    (all_oil_polys,
     all_oil_weights,
     all_oil_thicknesses) = SpatialRelease.load_shapefile(sample_shapefile)

    assert len(all_oil_polys) == 8
    assert len(all_oil_weights) == 8
    assert len(all_oil_thicknesses) == 8

    for poly in all_oil_polys:
        check_valid_polygon(poly)

    # NOTE: these values are pulled from running the code
    #       they may not be correct, but this will let us catch changes
    assert np.allclose(all_oil_weights, [0.0019291097691711862, 0.0018247639782104231,
                                         0.09568991387647877, 0.00017874329138003873,
                                         9.309062636361091e-05, 0.005663950543120452,
                                         0.001098505440460224, 0.8935219224748153
                                         ], rtol=1e-12)

    assert np.allclose(all_oil_thicknesses, [5e-06, 5e-06, 5e-06, 5e-06,
                                             5e-06, 5e-06, 5e-06, 0.0002
                                             ], rtol=1e-12)



