'''
Helper functions to make it a little easier to use the
Geospatial Data Abstraction Libraries (GDAL/OGR).
'''
from osgeo import ogr


def ogr_drivers():
    return [ogr.GetDriver(i) for i in range(ogr.GetDriverCount())]


def ogr_driver_names():
    return [d.GetName() for d in ogr_drivers()]


def get_ogr_driver_by_name(name):
    return ogr.GetDriverByName(name)


def ogr_open_file(filename):
    return ogr.Open(filename)


def ogr_layers(infile):
    return [infile.GetLayerByIndex(i) for i in range(infile.GetLayerCount())]


def ogr_features(layer):
    layer.SetNextByIndex(0)
    return [feature for feature in layer]
