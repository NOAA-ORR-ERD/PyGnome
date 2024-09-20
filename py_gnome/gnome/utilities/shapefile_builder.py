"""Shapefile Builder"""

import pathlib
import shutil
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from gnome.utilities.hull import calculate_hull, calculate_contours

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
                             engine="pyogrio",
                             geometry_type=self.geometry_type)
            shutil.move(shz_name, self.fullfilename)
        else:
            # Write out the raw shapefile
            full_gdf.to_file(self.fullfilename, driver='ESRI Shapefile',
                             engine="pyogrio",
                             geometry_type=self.geometry_type)
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
        super(ParticleShapefileBuilder, self).__init__(filename, zip_output,
                                                       **kwargs)

    def append(self, sc):
        """
        Given a spill container, write out the current particle data to a
        data frame
        """
        super(ParticleShapefileBuilder, self).append(sc)
        frame_data = {
            'LE_id': sc['id'],
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
        gdf = gpd.GeoDataFrame(frame_data, crs='epsg:4326',
                               geometry='Position')
        # If we have surface concentration sort by it...
        if 'surface_concentration' in sc and not sc.uncertain:
            gdf = gdf.loc[gdf['Surf_Conc'].sort_values().index]
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
    schema = {'geometry': [this_hull]}
    geodataframe = gpd.GeoDataFrame(schema, crs='epsg:4326',
                                    geometry='geometry')
    geodataframe_3857 = geodataframe.to_crs(epsg=3857)
    return geodataframe_3857.area[0]


class BoundaryShapefileBuilder(ShapefileBuilder):
    def __init__(self, filename, zip_output=True, **kwargs):
        '''
        :param filename: Full path and basename of the shape file.
        :param zip: If we should zip the final shapefile results.
        '''
        super(BoundaryShapefileBuilder, self).__init__(filename, zip_output,
                                                       **kwargs)

    def append(self, sc, separate_by_spill=True, hull_ratio=0.5,
               hull_allow_holes=False):
        """
        Given a spill container, write out the current boundary data to a
        data frame
        """
        super(BoundaryShapefileBuilder, self).append(sc)
        # If we have a tuple or list for sc, we need to grab the timestep
        # out of the first
        current_time_stamp = None
        if isinstance(sc, (tuple, list)):
            current_time_stamp = sc[0].current_time_stamp
        else:
            current_time_stamp = sc.current_time_stamp

        # Calculate a concave hull
        hull = calculate_hull(sc, separate_by_spill=separate_by_spill,
                              ratio=hull_ratio,
                              allow_holes=hull_allow_holes)
        # Only process it if we get a hull back.
        # There are cases where the hull is not a polygon, and we skip those.
        if hull:
            frame_data = {
                'geometry': [hull],
                'area': [area_in_meters(hull)],
                'time': current_time_stamp.strftime('%Y-%m-%dT%H:%M:%S')
            }

            gdf = gpd.GeoDataFrame(frame_data, crs='epsg:4326',
                                   geometry='geometry')
            self.data_frames.append(gdf)

    def write(self):
        """Actually write out the shapefile as .shp or .zip"""

        # If we dont have any data... we need to set the geom type of the
        # empty shapefile that will be written.  Its point for particles, but
        # polygon for boundary.
        if not self.data_frames:
            self.geometry_type = 'Polygon'
        super(BoundaryShapefileBuilder, self).write()


class ContourShapefileBuilder(ShapefileBuilder):
    def __init__(self, filename, zip_output=True, **kwargs):
        '''
        :param filename: Full path and basename of the shape file.
        :param zip: If we should zip the final shapefile results.
        '''
        super(ContourShapefileBuilder, self).__init__(filename, zip_output,
                                                      **kwargs)

    def append(self, sc, cutoff_struct=None, hull_ratio=0.5,
               hull_allow_holes=False):
        """
        Given a spill container, write out the contours based on the
        cutoffs provided
        """
        # Cutoffs defined in a structure like the following (by spill_num):
        # cutoff_struct = {1: {'param': 'mass',
        #                      'cutoffs': [{'cutoff': 100,
        #                                   'label': 'low'},
        #                                  {'cutoff': 500,
        #                                   'label': 'medium'},
        #                                  {'cutoff': 1000,
        #                                   'label': 'heavy'}]},
        #                  2: {'param': 'surf_conc',
        #                      'cutoffs': [{'cutoff': 0.3,
        #                                   'label': 'low'},
        #                                  {'cutoff': 0.5,
        #                                   'label': 'medium'},
        #                                  {'cutoff': 1.0,
        #                                   'label': 'high'}]
        #                      }
        #                  }
        # The cutoffs are ordered array... low to high
        # First "low" bin is always [data] < cutoffs[current]
        # Middle bins are cutoffs[previous] < [data] < cutoffs[current]
        # Last "high" bin is always cutoffs[previous] < [data]

        super(ContourShapefileBuilder, self).append(sc)
        # Look in the spill container and get a list of spills
        # Make sure they are defined in the cutoffs_per_spill
        # If they are, we loop through each spill
        #   In each spill we loop through the cutoffs and create subsets
        #   of the data.
        #   Make a hull around each subset
        #   Create a geodataframe based on the array of generated hulls
        #   Append to the data_frames

        # Calculate the contours
        contours = calculate_contours(sc, cutoff_struct=cutoff_struct,
                                      ratio=hull_ratio,
                                      allow_holes=hull_allow_holes)
        if contours:
            frame_data = {
                'geometry': [c['contour'] for c in contours],
                'spill_num': [c['spill_num'] for c in contours],
                'cutoff': [c['cutoff'] for c in contours],
                'cutoff_id': [c['cutoff_id'] for c in contours],
                'color': [c['color'] for c in contours],
                'label': [c['label'] for c in contours],
                'time': sc.current_time_stamp.strftime('%Y-%m-%dT%H:%M:%S')
            }
            gdf = gpd.GeoDataFrame(frame_data, crs='epsg:4326',
                                   geometry='geometry')
            self.data_frames.append(gdf)

    def write(self):
        """Actually write out the shapefile as .shp or .zip"""

        # If we dont have any data... we need to set the geom type of the
        # empty shapefile that will be written.  Its point for particles, but
        # polygon for boundary.
        if not self.data_frames:
            self.geometry_type = 'Polygon'
        super(ContourShapefileBuilder, self).write()
