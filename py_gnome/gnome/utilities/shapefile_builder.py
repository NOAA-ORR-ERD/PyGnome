"""Shapefile Builder"""

import geopandas as gpd
import os
import pandas as pd
import pathlib
import pyogrio
from shapely.geometry import Point
import shutil
import time
import warnings
import zipfile

from gnome.utilities.hull import calculate_hull

# Ignore a pyogrio warning that we dont care about
warnings.filterwarnings('ignore', message='.*Possibly due to too larger',
                        category=RuntimeWarning, module='pyogrio')

class ShapefileBuilder(object):
    def __init__(self, filename, zip_output=True, **kwargs):
        '''
        :param filename: Full path and basename of the shape file.
        :param zip: If we should zip the final shapefile results.
        '''
        pathlib_path = pathlib.Path(filename)
        self.zip_output = zip_output
        if zip_output:
            self.fullfilename = pathlib_path.with_suffix('.zip')
        else:
            # Else we return .shp
            self.fullfilename = pathlib_path.with_suffix('.shp')
        self.filenamestem = pathlib_path.stem
        self.filenamepath = str(pathlib_path.parent)
        # Data frame array to hold the step data
        self.data_frames = []
        self.geometry_type = None

    def append(self, sc):
        # Must be implemented by child classes
        pass

    def write(self):
        """Actually write out the shapefile as .shp or .zip"""
        # Default the geometry_type to None so that geopandas can introspect
        # the data.  We set to Point if we have an empty shapefile
        if self.data_frames:
            full_gdf = gpd.GeoDataFrame(pd.concat(self.data_frames,
                                                  ignore_index=True))
        else:
            full_gdf = gpd.GeoDataFrame(geometry=[], crs='epsg:4326')
        full_gdf.set_crs('epsg:4326')
        if self.zip_output:
            shz_name = self.fullfilename.with_suffix('.shz')
            # Write out the zipped shapefile
            full_gdf.to_file(shz_name, driver='ESRI Shapefile',
                             engine="pyogrio", geometry_type=self.geometry_type)
            shutil.move(shz_name, self.fullfilename)
        else:
            # Write out the raw shapefile
            full_gdf.to_file(self.fullfilename, driver='ESRI Shapefile',
                             engine="pyogrio", geometry_type=self.geometry_type)
        return self.fullfilename

    @property
    def filename(self):
        return self.fullfilename

class ParticleShapefileBuilder(ShapefileBuilder):
    def __init__(self, filename, zip_output=True, **kwargs):
        '''
        :param filename: Full path and basename of the shape file.
        :param zip: If we should zip the final shapefile results.
        '''
        super(ParticleShapefileBuilder, self).__init__(filename, zip_output, **kwargs)

    def append(self, sc):
        """Given a spill container, write out the current particle data to a data frame"""
        super(ParticleShapefileBuilder, self).append(sc)
        frame_data = {'LE_id': sc['id'],
                      'Spill_id': sc['spill_num'],
                      'Depth': [pos[2] for pos in sc['positions']],
                      'Mass': sc['mass'],
                      'Age': sc['age'],
                      'StatusCode': sc['status_codes'],
                      'Time': sc.current_time_stamp.strftime('%Y-%m-%dT%H:%M:%S'),
                      'Position': [Point(pt) for pt in sc['positions']]
        }
        # Some elements are optional... check those here:
        if 'surface_concentration' in sc and not sc.uncertain:
            frame_data['Surf_Conc'] = sc['surface_concentration']
        if 'viscosity' in sc:
            frame_data['Viscosity'] = sc['viscosity']
        if 'frac_water' in sc:
            frame_data['FracWater'] = sc['frac_water']
        if 'density' in sc:
            frame_data['Density'] = sc['density']
        gdf = gpd.GeoDataFrame(frame_data, crs='epsg:4326', geometry='Position')
        self.data_frames.append(gdf)

    def write(self):
        """Actually write out the shapefile as .shp or .zip"""

        # If we dont have any data... we need to set the geom type of the
        # empty shapefile that will be written.  Its point for particles, but
        # polygon for boundary.
        if not self.data_frames:
            self.geometry_type = 'Point'
        super(ParticleShapefileBuilder, self).write()

def area_in_meters(this_hull):
    schema = {'geometry':[this_hull]}
    geodataframe = gpd.GeoDataFrame(schema, crs='epsg:4326', geometry='geometry')
    geodataframe_3857 = geodataframe.to_crs(epsg=3857)
    return geodataframe_3857.area[0]

class BoundaryShapefileBuilder(ShapefileBuilder):
    def __init__(self, filename, zip_output=True, **kwargs):
        '''
        :param filename: Full path and basename of the shape file.
        :param zip: If we should zip the final shapefile results.
        '''
        super(BoundaryShapefileBuilder, self).__init__(filename, zip_output, **kwargs)

    def append(self, sc, separate_by_spill=True, hull_ratio=0.5, hull_allow_holes=False):
        """Given a spill container, write out the current boundary data to a data frame"""
        super(BoundaryShapefileBuilder, self).append(sc)
        # Calculate a concave hull
        hull = calculate_hull(sc, separate_by_spill=separate_by_spill, ratio=hull_ratio,
                              allow_holes=hull_allow_holes)
        # Only process it if we get a hull back.  There are cases where the hull is not
        # a polygon, and we skip those.
        if hull:
            frame_data = {'geometry': [hull],
                          'area': [area_in_meters(hull)],
                          'time': sc.current_time_stamp.strftime('%Y-%m-%dT%H:%M:%S')}
            gdf = gpd.GeoDataFrame(frame_data, crs='epsg:4326', geometry='geometry')
            self.data_frames.append(gdf)

    def write(self):
        """Actually write out the shapefile as .shp or .zip"""

        # If we dont have any data... we need to set the geom type of the
        # empty shapefile that will be written.  Its point for particles, but
        # polygon for boundary.
        if not self.data_frames:
            self.geometry_type = 'Polygon'
        super(BoundaryShapefileBuilder, self).write()
