import functools
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

def get_shapefile_args(filename):
    """
    :param filename: string path of a zipped shapefile
    :return: dict associating shapefile.Reader kwargs to names in the zip file
    """
    with zipfile.ZipFile(filename, 'r') as zsf:
        #need to hunt down the correct pair of shp/dbf. NESDIS files may
        #have a 'point' as well as a 'polygon' file...
        #Going to just go with eliminating choices that contain 'Point Source' and the like for now..
        sfile_ex = ['shp','dbf','shx','prj']
        reader_args = {}
        for arg in sfile_ex:
            files = [f for f in zsf.namelist() if f.split('.')[-1] == arg and 'point' not in f.lower()]
            if len(files) == 0:
                raise ValueError('No .{0} file found'.format(arg))
            elif len(files) > 1:
                warnings.warn('More than one .{0} file found. Using {1}'.format(arg, files[0]))
            reader_args[arg] = files[0]
        return reader_args


def open_shapefile(filename):
    """
    :param filename: string path of a zipped shapefile
    :return: shapefile.Reader
    """
    with zipfile.ZipFile(filename, 'r') as zsf:
        args = get_shapefile_args(filename)
        for k, v in args.items():
            args[k] = zsf.open(v, 'r')
        rv = shapefile.Reader(**args)
        return rv

def load_shapefile(filename, transform_crs=True):
    """
    load up a generic shapefile into a FeatureCollection

    :param filename: string path of a zip file
    :param transform_crs: attempts to read the .prj file if any and convert to EPSG:4326

    :return: geojson.FeatureCollection
    """
    rv = open_shapefile(filename)
    rv = geojson.loads(geojson.dumps(rv.__geo_interface__))
    args = get_shapefile_args(filename)
    pf = None
    with zipfile.ZipFile(filename, 'r') as zsf:
        pf = pyproj.CRS.from_wkt(zsf.open(args['prj'], 'r').readline().decode('utf-8'))
    if not transform_crs:
        if pf.to_epsg() != '4326':
            warnings.warn('shapefile is using epsg:{0} not epsg:4326!'.format(pf.to_epsg()))
    else:
        if pf.to_epsg() != '4326':
            if int(pyproj.__version__[0]) < 2:
                Proj1 = pyproj.Proj(init='epsg:3857')
                Proj2 = pyproj.Proj(init='epsg:4326')
                transformer = functools.partial(
                    pyproj.transform,
                    Proj1,
                    Proj2)
            else:
                transformer = pyproj.Transformer.from_crs(
                    "epsg:{0}".format(pf.to_epsg()),
                    "epsg:4326",
                    always_xy=True
                )
            if hasattr(rv, 'bbox'):
                xx, yy = transformer.transform([rv.bbox[0],rv.bbox[2]],[rv.bbox[1],rv.bbox[3]])
                rv.bbox = [xx[0], yy[0], xx[1], yy[1]]
            for feature in rv.features:
                old_geo = shapely.geometry.shape(feature.geometry)
                #Geometries can be MultiPolygons or Polygons
                #Each needs to be converted to EPSG:4326
                new_geo = shapely.ops.transform(transformer.transform, old_geo)
                feature.geometry = geojson.loads(geojson.dumps(new_geo.__geo_interface__))
        return rv
