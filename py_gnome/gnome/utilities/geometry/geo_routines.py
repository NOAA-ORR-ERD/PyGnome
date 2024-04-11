import functools
import warnings
import zipfile
import trimesh

# Spatial libs
import pyproj
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape
import geopandas as gpd

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
        poly = shape(poly)
    return abs(geod.geometry_area_perimeter(poly)[0])

def triangulate_poly(poly):
    '''
    :param poly: shapely or geojson MultiPolygon or Polygon

    :return: list of shapely.Polygon (triangles)
    '''
    if isinstance(poly, (geojson.MultiPolygon, geojson.Polygon)):
        poly = shape(poly)
    retval = []
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
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
        p = shape(p) #to handle geojson.(Multi)Polygon objects
        if isinstance(p, MultiPolygon):
            for subp in p.geoms:
                rv.append(subp)
        else:
            rv.append(p)
    return rv

def check_valid_polygon(poly):
    """
    checks that a shapely Polygon object at least has valid values for coordinates
    """
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
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

def load_shapefile(filename, transform_crs=True):
    """
    Use GeoPandas to load up a shapefile into a FeatureCollection

    :param filename: string path of a zip file
    :param transform_crs: attempts to read the .prj file if any and convert to EPSG:4326

    :return: geojson.FeatureCollection
    """
    # Open up the zip file so we can find any .shp we have
    with zipfile.ZipFile(filename, 'r') as zipper:
        # Use the namelist to find any shapefiles in the zip
        # We also reject any that have 'point' in the name
        shapefiles = [f for f in zipper.namelist() if f.split('.')[-1] == 'shp' and 'point' not in f.lower()]
        # If we did not find any, we need to toss an error
        if not shapefiles:
            raise ValueError(f'No shapefile found in zip {filename}!')
        # If we found more than one, we issue a warning and use the first one.
        if len(shapefiles) > 1:
            warnings.warn(f'More than one shapefile found in zip {filename}! Using {shapefiles[0]}')
        # Use GeoPandas to read the shapefile out of the zip
        shapefile = gpd.read_file(f'zip://{str(filename)}!{shapefiles[0]}', engine="pyogrio")
        # Force convert to 4326 if requested.  This will be a Noop if already in 4326
        if transform_crs:
            shapefile = shapefile.to_crs('epsg:4326')
        # Dump to json (dataframe -> json string -> json object)
        shapefile_json = geojson.loads(shapefile.to_json())
        # Add a bbox to the feature collection
        shapefile_json['bbox'] = list(shapefile.total_bounds)
        # Finally, hand the geojson back to the caller
        return shapefile_json

