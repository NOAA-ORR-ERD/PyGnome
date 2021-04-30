import shapely
import pyproj
import geojson
import shapefile
import warnings
import zipfile
import trimesh

from shapely.geometry import Polygon

import numpy as np
import random

geod = pyproj.Geod(ellps='WGS84')
def geo_area_of_polygon(poly):
    '''
    :param poly: 
    :type poly: shapely or geojson MultiPolygon or Polygon

    :return: area of polygon in m^2
    '''
    if isinstance(poly, (geojson.MultiPolygon, geojson.Polygon)):
        poly = shapely.geometry.shape(poly)
    return abs(geod.geometry_area_perimeter(poly)[0])

def triangulate_poly(poly):
    '''
    :param poly: shapely or geojson MultiPolygon or Polygon

    :return: list of shapely.Polygon (triangles)
    '''
    if isinstance(poly, (geojson.MultiPolygon, geojson.Polygon)):
        poly = shapely.geometry.shape(poly)
    retval = []
    if isinstance(poly, shapely.geometry.MultiPolygon):
        for p in poly:
            pts, tris = trimesh.creation.triangulate_polygon(p, engine='earcut')
            retval = retval + [Polygon(k) for k in pts[tris]]
    else:
        pts, tris = trimesh.creation.triangulate_polygon(poly, engine='earcut')
        retval = [Polygon(k) for k in pts[tris]]
    return retval

def poly_area_weight(polys, geo_area=False):
    '''
    :param polys: iterable of shapely.Polygon
    :param geo_area: If true, calculates using geo area (slower, more accurate)
    '''
    areas = None
    if geo_area:
        areas = [geo_area_of_polygon(p) for p in polys]
    else:
        areas = [p.area for p in polys]
    t_area = sum(areas)
    weights = [s/t_area for s in areas]
    return weights

def mixed_polys_to_polygon(polys):
    '''
    :param polys: iterable containing mixed Polygon and MultiPolygon
    :return: iterable of shapely.Polygon
    '''
    rv = []
    for p in polys:
        p = shapely.geometry.shape(p) #to handle geojson.(Multi)Polygon objects
        if isinstance(p, shapely.geometry.MultiPolygon):
            for subp in p:
                rv.append(subp)
        else:
            rv.append(p)
    return rv

def check_valid_polygon(poly):
    """
    checks that a shapely Polygon object at least has valid values for coordinates
    """
    if isinstance(poly, shapely.geometry.MultiPolygon):
        for p in poly:
            for point in p.exterior.coords:
                assert -360 < point[0] < 360
                assert -90 < point[1] < 90
    else:
        for point in poly.exterior.coords:
            assert -360 < point[0] < 360
            assert -90 < point[1] < 90
            
#tri is a Shapely.Polygon, or 3x2 array of coords
#returns a 2D coordinate
def random_pt_in_tri(tri):
    coords = None
    if isinstance(tri, Polygon):
        coords = tri.exterior.coords
    else:
        coords = tri
    coords = np.array(coords)
    R = random.random()
    S = random.random()
    if R + S >= 1:
        R = 1 - R
        S = 1 - S
    A = coords[0]
    AB = coords[1] - coords[0]
    AC = coords[2] - coords[0]
    RPP = A + R*AB + S*AC
    return RPP

def load_shapefile(filename):
    """
    load up a generic shapefile into a FeatureCollection

    filename is a zip file
    returns a geojson.FeatureCollection
    """
    with zipfile.ZipFile(filename, 'r') as zsf:
        #need to hunt down the correct pair of shp/dbf. NESDIS files may
        #have a 'point' as well as a 'polygon' file...
        #Going to just go with eliminating choices that contain 'Point Source' and the like for now..
        shpfiles = [f for f in zsf.namelist() if f.split('.')[-1] == 'shp' and 'point' not in f.lower()]
        dbffiles = [f for f in zsf.namelist() if f.split('.')[-1] == 'dbf' and 'point' not in f.lower()]
        if len(shpfiles) == 0:
            raise ValueError('No .shp file found')
        elif len(shpfiles) > 1:
            warnings.warn('More than one .shp file found. Using {0}'.format(shpfiles[0]))
        shpfile = zsf.open(shpfiles[0], 'r')
        if len(dbffiles) == 0:
            raise ValueError('No .dbf file found')
        elif len(dbffiles) > 1:
            warnings.warn('More than one .shp file found. Using {0}'.format(shpfiles[0]))
        dbffile = zsf.open(dbffiles[0], 'r')
        sf = shapefile.Reader(shp=shpfile, dbf=dbffile)
        fc = geojson.loads(geojson.dumps(sf.__geo_interface__))
        return fc