"""
tests for the spatial release from polygons:

e.g. from the NESDIS MPSR reports
"""

import os
from pathlib import Path
import datetime
import numpy as np
import datetime
import shapely
import pytest
import zipfile
import shapefile

from gnome.utilities.geometry import geo_routines

from gnome.spills.release import (PolygonRelease, NESDISRelease)

data_dir = Path(__file__).parent / "data_for_tests"

sample_nesdis_shapefile = data_dir / "NESDIS_files.zip"

simplePolys = [shapely.geometry.Polygon([[0,0],[3,0],[3,3],[0,3]]),
               shapely.geometry.MultiPolygon(
                   [shapely.geometry.Polygon([[4,0],[5,0],[5,1],[4,1]]),
                    shapely.geometry.Polygon([[0,4],[1,4],[1,5],[0,5]])]
                )]
thicknesses = [0.0005,0.0001]
weights = [0.75, 0.25]

class TestPolygonRelease:

    def test_construction(self):

        sr_from_file = PolygonRelease(filename=sample_nesdis_shapefile)
        assert len(sr_from_file.features[:]) == 2

        sr_from_features = PolygonRelease(features=sr_from_file.__geo_interface__)
        assert len(sr_from_features.features[:]) == 2

        #NOTE It is worth pointing out here that sr_from_file and sr_from_features are
        #NOT EQUAL. This is because the release_time and other release attributes are NOT currently put
        #into the __geo_interface__

        sr = PolygonRelease(polygons=simplePolys)
        with pytest.raises(ValueError):
            sr2 = PolygonRelease(polygons=simplePolys, weights=[0.5,0.25,0.25])
        with pytest.raises(ValueError):
            sr3 = PolygonRelease(polygons=simplePolys, weights = [0.5, 0.5], thicknesses=thicknesses)

    def test_properties(self):

        sr = PolygonRelease(filename=sample_nesdis_shapefile)

        assert len(sr.polygons) == 2
        assert sr.weights == None
        assert sr.thicknesses == None
        assert len(sr.areas) == 2

        sr.weights = weights
        assert all([a == b for a, b in zip(sr.weights, [0.75, 0.25])])

        with pytest.raises(ValueError):
            #cannot have both thicknesses and weights
            sr.thicknesses = thicknesses

        sr.weights = None

        sr.thicknesses = thicknesses

        with pytest.raises(ValueError):
            sr.weights = [0.75, 0.25]

        assert all([a == b for a, b in zip(sr.thicknesses, [0.0005, 0.0001])])

    def test_serialize(self):
        sr = PolygonRelease(filename=sample_nesdis_shapefile)
        ser = sr.serialize()
        sr2 = PolygonRelease.deserialize(ser)
        assert sr == sr2

    def test_prepare(self):
        sr = PolygonRelease(filename=sample_nesdis_shapefile)
        sr.prepare_for_model_run(900)

    def test_feature_update(self):
        #polygons, weights, and thicknesses can be updated from the web client by passing
        #a new FeatureCollection through the feature attribute.

        sr = PolygonRelease(filename=sample_nesdis_shapefile)
        ser = sr.serialize()
        assert sr.weights == None
        ser['features']['features'][0]['properties']['weight'] = 0.75
        ser['features']['features'][1]['properties']['weight'] = 0.25
        sr.update_from_dict(ser)
        assert all([w1 == w2 for w1, w2 in zip(sr.weights, [0.75, 0.25])])

class TestNESDISRelease(object):

    def test_construction(self):
        nr_from_file = NESDISRelease(filename=sample_nesdis_shapefile)
        assert len(nr_from_file.features[:]) == 2

        nr_from_features = NESDISRelease(features=nr_from_file.__geo_interface__)
        assert len(nr_from_features.features[:]) == 2

        assert nr_from_file == nr_from_features

    def test_default_construction(self):
        features = geo_routines.load_shapefile(sample_nesdis_shapefile)

    def test_release_time(self):
        nr = NESDISRelease(filename=sample_nesdis_shapefile)

        assert nr.release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')
        assert nr.end_release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')

    def test_alter_release_time_and_save_load(self):
        nr = NESDISRelease(filename=sample_nesdis_shapefile)

        assert nr.release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')
        assert nr.end_release_time == datetime.datetime.strptime('5/14/2020 15:20', '%m/%d/%Y %H:%M')
        nr.end_release_time += datetime.timedelta(minutes=40)
        nr.release_time += datetime.timedelta(minutes=40)
        assert nr.release_time == datetime.datetime.strptime('5/14/2020 16:00', '%m/%d/%Y %H:%M')
        assert nr.end_release_time == datetime.datetime.strptime('5/14/2020 16:00', '%m/%d/%Y %H:%M')
        sv = nr.save(None) #.save(None)[0] gets the serialized form of this object as it would appear in a save file
        sv[1].close()
        nr2 = NESDISRelease.deserialize(sv[0])
        assert nr.release_time == nr2.release_time
        assert nr == nr2


    def test_coord_reprojection(self):
        nr = NESDISRelease(filename=sample_nesdis_shapefile)

        for poly in nr.polygons:
            geo_routines.check_valid_polygon(poly)

    def test_prepare(self):
        sr = NESDISRelease(filename=sample_nesdis_shapefile)
        sr.prepare_for_model_run(900)
        assert len(sr._weights) == len(sr._tris)
        assert np.isclose(sum(sr._weights), 1.0)
        assert np.isclose(sum([geo_routines.geo_area_of_polygon(t) for t in sr._tris]),
        sum([geo_routines.geo_area_of_polygon(p) for p in sr.polygons]))

def test_load_minimal_shapefile():
    """
    test loading a shapefile with just a polygon in it
    """
    shapefilename  = data_dir / "spatial_example.zip"
    sr = PolygonRelease(filename=shapefilename)

    # should maybe test more, but at least this will show it loaded
    assert len(sr.polygons) == 1


'''
def test_load_shapefile():
    (release_time,
     all_oil_polys,
     all_oil_weights,
     all_oil_thicknesses,
     all_oil_areas,
     all_oil_types) = NESDISRelease.load_nesdis(sample_nesdis_shapefile)

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
    rel = NESDISRelease(filename=sample_nesdis_shapefile)
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

'''
