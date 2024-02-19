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
