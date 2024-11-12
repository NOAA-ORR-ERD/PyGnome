import sys
import time

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import geopandas as gpd

from gnome.environment.environment_objects import GridCurrent
from gnome.environment.gridded_objects_base import Grid_U, Grid_R, Grid_S

from gridded.utilities import convert_mask_to_numpy_mask
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
transformer.transform(12, 12)

#test_filename = 'C:\\Users\\jahen\\Downloads\\CIOFS.nc'
test_filename = 'C:\\Users\\jahen\\Downloads\\wcofs.t03z.20241003.fields.f060.nc'

class GridGeoGenerator(object):
    
    node_line_appearance = {
        'color': 'black',
        'alpha': 1,
        'linewidth': 0.5,
    }
    
    node_line_masked_apperance = {
        'color': 'black',
        'alpha': 0.25,
        'linewidth': 0.5,
    }
    
    node_marker_appearance = {
        'color': 'black',
        'alpha': 0.75,
        'marker': 'o',
    }
    
    node_marker_masked_appearance = {
        'color': 'black',
        'alpha': 0.25,
        'marker': 'X',
    }
    
    edge1_line_appearance = {
        'color': 'blue',
        'alpha': 0.5,
        'linewidth': 0.5,
    }
    
    edge1_line_masked_appearance = {
        'color': 'blue',
        'alpha': 0.25,
        'linewidth': 0.5,
    }
    
    edge1_marker_appearance = {
        'color': 'blue',
        'alpha': 0.5,
        'marker': '<',
    }
    
    edge1_marker_masked_appearance = {
        'color': 'brown',
        'alpha': 0.25,
        'marker': '3',
    }
    
    edge2_line_appearance = {    
        'color': 'red',
        'alpha': 0.5,
        'linewidth': 0.5,
    }
    
    edge2_line_masked_appearance = {
        'color': 'red',
        'alpha': 0.25,
        'linewidth': 0.5,
    }
    
    edge2_marker_appearance = {
        'color': 'blue',
        'alpha': 0.5,
        'marker': '^',
    }
    
    edge2_marker_masked_appearance = {
        'color': 'brown',
        'alpha': 0.25,
        'marker': '2',
    }
    
    center_line_appearance = {
        'color': 'green',
        'alpha': 0.75,
        'linewidth': 0.5,
    }
    
    center_line_masked_appearance = {
        'color': 'green',
        'alpha': 0.25,
        'linewidth': 0.5,
    }
    
    center_marker_appearance = {
        'color': 'blue',
        'alpha': 0.75,
        'marker': 's',
    }
    
    center_marker_masked_appearance = {
        'color': 'brown',
        'alpha': 0.25,
        'marker': 'd',
    }
    
    def __init__(self, crs=None):
        self.crs = crs

    def get_gc(self, filename):
        gc = GridCurrent.from_netCDF(filename)
        return gc

    def get_lonlat(self, grid_obj, location_name):
        lon_var_name = location_name+'_lon'
        lat_var_name = location_name+'_lat'
        lons = getattr(grid_obj, lon_var_name)
        lats = getattr(grid_obj, lat_var_name)
        return lons, lats

    def get_mask_raw(self, grid_obj, location_name, scale=30):
        mask_var_name = location_name+'_mask'
        mask = getattr(grid_obj, mask_var_name)
        return mask[:] * scale
    
    def get_mask_raw_invert(self, grid_obj, location_name, scale=30):
        mask_var_name = location_name+'_mask'
        mask = getattr(grid_obj, mask_var_name)
        return (mask[:] * scale + scale) % (2*scale)

    def get_mask_bool(self, grid_obj, location_name):
        mask_var_name = location_name+'_mask'
        mask = getattr(grid_obj, mask_var_name)
        return convert_mask_to_numpy_mask(mask)

    def gen_grid_lines(self, grid_obj, location='node', use_mask=True):
        #Returns a GeoSeries object representing the grid lines
        if isinstance(grid_obj, Grid_S):
            if use_mask:
                return self.gen_grid_lines_S_masked(grid_obj, location)
            else: 
                return self.gen_grid_lines_S_unmasked(grid_obj, location)
        elif isinstance(grid_obj, Grid_U):
            return self.gen_grid_lines_U(grid_obj, location)
        elif isinstance(grid_obj, Grid_R):
            return self.gen_grid_lines_R(grid_obj, location)
        pass

    def gen_grid_lines_S_unmasked(self, grid_obj, location='node'):
        #Returns a GeoSeries object representing the grid lines of a Grid_S object (no masking)
        lons, lats = self.get_lonlat(grid_obj, location)
        h_lines = [sgeom.LineString([(lons[i, j], lats[i, j]) for j in range(lons.shape[1])]) for i in range(lons.shape[0])]
        v_lines = [sgeom.LineString([(lons[i, j], lats[i, j]) for i in range(lons.shape[0])]) for j in range(lons.shape[1])]
        multiline = sgeom.MultiLineString(h_lines + v_lines)
        return gpd.GeoSeries([multiline], crs=self.crs.to_proj4())

    def gen_grid_lines_S_masked(self, grid_obj, location='node'):
        #Returns a GeoSeries object representing the grid lines of a Grid_S object (with masking)
        lons, lats = self.get_lonlat(grid_obj, location)
        mask = self.get_mask_bool(grid_obj, location)
        lines = []
        for i in range(lons.shape[0]):
            for j in range(lons.shape[1]-1):
                if not mask[i, j] and not mask[i, j+1]:
                    lines.append(sgeom.LineString([(lons[i, j], lats[i, j]), (lons[i, j+1], lats[i, j+1])]))
        multiline = sgeom.MultiLineString(lines)
        return gpd.GeoSeries([multiline], crs=self.crs)

    def gen_grid_lines_U(self, grid_obj, location='node'):
        pass

    def gen_grid_lines_R(self, grid_obj, location='node'):
        pass

    def get_max_extent(self, grid_obj, margin=0.1):
        #Returns the maximum extent of the grid (lon_min, lon_max, lat_min, lat_max)
        #adds a margin of 10% by default
        w, e, s, n = grid_obj.node_lon.min(), grid_obj.node_lon.max(), grid_obj.node_lat.min(), grid_obj.node_lat.max()
        w -= (e - w) * margin
        e += (e - w) * margin
        s -= (n - s) * margin
        n += (n - s) * margin
        return w, e, s, n

    def gen_grid_points(self, grid_obj, location='node'):
        #Returns a GeoSeries object representing the grid points. Does not respect masking.
        lon_var_name = location+'_lon'
        lat_var_name = location+'_lat'
        lons = getattr(grid_obj, lon_var_name)
        lats = getattr(grid_obj, lat_var_name)
        pts = np.stack((lons, lats), axis=-1)
        points = gpd.GeoSeries([sgeom.Point(lon, lat) for lon, lat in np.reshape(pts, (-1, 2))], crs=self.crs)
        return points

    def gen_grid_point_ij_labels(self, grid_obj, location='node'):
        pass

    def gen_masked_grid_points(self, grid_obj, location='node'):
        #Returns a pair of GeoSeries objects representing the masked and unmasked grid points
        pass

    def gen_mask(self, grid_obj, location='node'):
        #Returns a True/False mask of the grid points
        pass


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        fig = Figure(figsize=(15, 9))
        static_canvas = FigureCanvas(fig)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout.addWidget(NavigationToolbar(static_canvas, self))
        layout.addWidget(static_canvas)
        
        plot_crs_class = ccrs.Orthographic
        g3_crs_class = ccrs.PlateCarree
        
        g3 = GridGeoGenerator()
        gc = g3.get_gc(test_filename)
        grid_obj = gc.grid
        extent = g3.get_max_extent(grid_obj)
        g3.crs = g3_crs_class()
        plot_crs = plot_crs_class(extent[0] + (extent[1] - extent[0])/2, extent[2] + (extent[3] - extent[2])/2)
        
        ctrans = Transformer.from_proj(g3.crs, plot_crs)
        ws = ctrans.transform(extent[0], extent[2])
        en = ctrans.transform(extent[1], extent[3])
        
        ax = fig.add_subplot(1, 1, 1, projection=plot_crs)
        ax.set_extent(extent, crs=g3.crs)

        ax.add_feature(cfeature.OCEAN.with_scale('50m'), zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('50m'), zorder=0, edgecolor='black')
        
        ax.gridlines()
        
        node_lines = g3.gen_grid_lines(grid_obj, 'node', use_mask=False)
        node_lines.to_crs(plot_crs).plot(ax=ax, **g3.node_line_appearance)
        
        markerscale = 30
        
        center_markers = g3.gen_grid_points(grid_obj, 'center').to_crs(plot_crs)
        center_mask_raw = g3.get_mask_raw(grid_obj, 'center', scale=markerscale)
        center_mask_invert = g3.get_mask_raw_invert(grid_obj, 'center', scale=markerscale)
        ax.scatter(center_markers.x, center_markers.y, s=center_mask_raw, **g3.center_marker_appearance)
        ax.scatter(center_markers.x, center_markers.y, s=center_mask_invert, **g3.center_marker_masked_appearance)
        
        edge1_markers = g3.gen_grid_points(grid_obj, 'edge1').to_crs(plot_crs)
        edge1_mask_raw = g3.get_mask_raw(grid_obj, 'edge1', scale=markerscale)
        edge1_mask_invert = g3.get_mask_raw_invert(grid_obj, 'edge1', scale=markerscale)
        ax.scatter(edge1_markers.x, edge1_markers.y, s=edge1_mask_raw, **g3.edge1_marker_appearance)
        ax.scatter(edge1_markers.x, edge1_markers.y, s=edge1_mask_invert, **g3.edge1_marker_masked_appearance)
        
        edge2_markers = g3.gen_grid_points(grid_obj, 'edge2').to_crs(plot_crs)
        edge2_mask_raw = g3.get_mask_raw(grid_obj, 'edge2', scale=markerscale)
        edge2_mask_invert = g3.get_mask_raw_invert(grid_obj, 'edge2', scale=markerscale)
        ax.scatter(edge2_markers.x, edge2_markers.y, s=edge2_mask_raw, **g3.edge2_marker_appearance)
        ax.scatter(edge2_markers.x, edge2_markers.y, s=edge2_mask_invert, **g3.edge2_marker_masked_appearance)
        
        spill_pos = sgeom.Point(ctrans.transform(-123.96, 45.6))
        ax.scatter(spill_pos.x, spill_pos.y, s=100, color='red', marker='*')


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()