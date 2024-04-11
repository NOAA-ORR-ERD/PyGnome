import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint, LineString
from shapely import concave_hull, union_all, buffer

# Buffer by a number of meters
def buffer_hull(this_hull, buffer_distance=1):
    schema = {'geometry':[this_hull]}
    geodataframe = gpd.GeoDataFrame(schema, crs='epsg:4326', geometry='geometry')
    geodataframe_3857 = geodataframe.to_crs(epsg=3857)
    geodataframe_3857_buffered = geodataframe_3857.buffer(buffer_distance)
    geodataframe_3857_buffered_back_to_4326 = geodataframe_3857_buffered.to_crs(epsg=4326)
    return geodataframe_3857_buffered_back_to_4326[0]

def calculate_hull(spill_container, ratio=0.5, allow_holes=False,
                   separate_by_spill=False):
    hulls_found = []
    if separate_by_spill:
        # Make a dataframe
        schema = {'positions':[Point(point) for point in spill_container['positions']],
                  'spill_num':spill_container['spill_num']}
        data_frame = gpd.GeoDataFrame(schema, crs='epsg:4326', geometry='positions')
        spill_nums = data_frame['spill_num'].unique()
        for spill_num in spill_nums:
            this_spill = data_frame[data_frame['spill_num'] == spill_num]
            this_mpt = MultiPoint([Point(point) for point in this_spill['positions']])
            this_hull = concave_hull(this_mpt, ratio=ratio, allow_holes=allow_holes)
            if (isinstance(this_hull, Point) or isinstance(this_hull, LineString)):
                this_hull = buffer_hull(this_hull)
            if (isinstance(this_hull, Polygon) or
                isinstance(this_hull, MultiPolygon)):
                hulls_found.append(this_hull)
    else:
        # We want a hull around everything
        schema = {'positions':[Point(point) for point in spill_container['positions']]}
        data_frame = gpd.GeoDataFrame(schema, crs='epsg:4326')
        this_mpt = MultiPoint([Point(point) for point in spill_container['positions']])
        this_hull = concave_hull(this_mpt, ratio=ratio, allow_holes=allow_holes)
        if (isinstance(this_hull, Point) or isinstance(this_hull, LineString)):
            this_hull = buffer_hull(this_hull)
        if (isinstance(this_hull, Polygon) or
            isinstance(this_hull, MultiPolygon)):
            hulls_found.append(this_hull)
    final_geom = union_all(hulls_found)
    return final_geom if (isinstance(final_geom, Polygon) or
                          isinstance(final_geom, MultiPolygon)) else None


# Cutoffs defined in a structure like the following (by spill_num):
#cutoff_struct = {1: {'param': 'mass',
#                     'cutoffs': [{'cutoff': 100,
#                                  'label': 'low'},
#                                 {'cutoff': 500,
#                                  'label': 'medium'},
#                                 {'cutoff': 1000,
#                                  'label': 'heavy'}]},
#                 2: {'param': 'surf_conc',
#                     'cutoffs': [{'cutoff': 0.3,
#                                  'label': 'low'},
#                                 {'cutoff': 0.5,
#                                  'label': 'medium'},
#                                 {'cutoff': 1.0,
#                                  'label': 'high'}]
#                     }
#                 }
# The cutoffs are ordered array... low to high
# First "low" bin is always [data] < cutoffs[current]
# Middle bins are cutoffs[previous] < [data] < cutoffs[current]
# Last "high" bin is always cutoffs[previous] < [data]

def calculate_contours(spill_container, cutoff_struct=None,
                       ratio=0.5, allow_holes=False):
    contours_found = []
    # The spills requested are the keys
    spills = cutoff_struct.keys()
    # Make a dataframe
    schema = {'positions':[Point(point) for point in spill_container['positions']],
              'spill_num':spill_container['spill_num'],
              'mass': spill_container['mass'],
              'age': spill_container['age'],
              'status_codes': spill_container['status_codes']}
    # Some elements are optional... check those here:
    if 'surface_concentration' in spill_container:
        schema['surf_conc'] = spill_container['surface_concentration']
    if 'viscosity' in spill_container:
        schema['viscosity'] = spill_container['viscosity']
    if 'frac_water' in spill_container:
        schema['frac_water'] = spill_container['frac_water']
    if 'density' in spill_container:
        schema['density'] = spill_container['density']
    data_frame = gpd.GeoDataFrame(schema, crs='epsg:4326', geometry='positions')
    for spill_num in spills:
        this_spill = data_frame[data_frame['spill_num'] == spill_num]
        this_struct = cutoff_struct[spill_num]
        param = this_struct['param']
        for idx, cutoff in enumerate(this_struct['cutoffs']):
            if idx == 0:
                # First one... we just do anything less than the current cutoff
                this_contour_set = this_spill[this_spill[param] <= cutoff['cutoff']]
            elif idx == len(this_struct['cutoffs'])-1:
                # Last one... we just do anything greater than the previous cutoff
                this_contour_set = this_spill[this_spill[param] >= this_struct['cutoffs'][idx-1]['cutoff']]
            else:
                # Else we bracket it...
                this_contour_set = this_spill[this_spill[param] <= cutoff['cutoff']]
                this_contour_set = this_contour_set[this_contour_set[param] >= this_struct['cutoffs'][idx-1]['cutoff']]
            if len(this_contour_set['positions']) > 0:
                this_contour_mpt = MultiPoint([Point(point) for point in this_contour_set['positions']])
                this_hull = concave_hull(this_contour_mpt, ratio=ratio, allow_holes=allow_holes)
                if (isinstance(this_hull, Point) or isinstance(this_hull, LineString)):
                    this_hull = buffer_hull(this_hull)
                if (isinstance(this_hull, Polygon) or
                    isinstance(this_hull, MultiPolygon)):
                    contours_found.append({'spill_num': spill_num,
                                           'cutoff': cutoff['cutoff'],
                                           'cutoff_id': cutoff['cutoff_id'],
                                           'color': cutoff['color'],
                                           'label': cutoff['label'],
                                           'contour': this_hull})
    return contours_found
