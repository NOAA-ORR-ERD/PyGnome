import sys
import time
import pathlib
import argparse

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons
import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import geopandas as gpd

from gnome.environment.environment_objects import GridCurrent
from gnome.environment.gridded_objects_base import Grid_U, Grid_R, Grid_S

from gridded.utilities import convert_mask_to_numpy_mask
from pyproj import Transformer

#test_filename = 'C:\\Users\\jahen\\Downloads\\CIOFS.nc'
# test_filename = 'C:\\Users\\jahen\\Downloads\\wcofs.t03z.20241003.fields.f060.nc'
# spill_location = (-123.96, 45.6)


parser=argparse.ArgumentParser(
    description='''Small QT/MPL app to draw grids, examine masks, and query grid cells.'''
    )
parser.add_argument('filename', type=pathlib.Path, help='Path to file')
parser.add_argument('-spill', nargs=2, help='Spill location in lon lat')
args=parser.parse_args()

test_filename = args.filename
spill_location = args.spill
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
    
    def __init__(self, crs=None, filename=None):
        self.crs = crs
        if filename is not None:
            self.gc = GridCurrent.from_netCDF(filename)
            self.grid_obj = self.gc.grid

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

    def index_query(self, grid_obj, position):
        #Returns the index of the grid cell that contains the given position
        #along with the indices of any relevant interpolation points.
        index = grid_obj.locate_faces(position, _memo=False, _copy=False, _hash=False)
        if np.all(index == -1):
            return None
        
        rv = {'cell_index': index}
        if isinstance(grid_obj, Grid_S):
            #compute the u and v interpolation indices
            e1_padding = grid_obj.get_padding_by_location('edge1')
            index = index.reshape(-1,2)
            edge1_interp_idx = grid_obj.apply_padding_to_idxs(index.copy(), padding=e1_padding)
            rv['edge1_interp_idx'] = edge1_interp_idx
            rv['edge1_u0_pos'] = (grid_obj.get_variable_at_index(grid_obj.edge1_lon, edge1_interp_idx.reshape(-1,2))[0][0],
                                  grid_obj.get_variable_at_index(grid_obj.edge1_lat, edge1_interp_idx.reshape(-1,2))[0][0])
            rv['edge1_u0_value'] = grid_obj.get_variable_at_index(self.gc.u.data[0,-1,:], edge1_interp_idx.reshape(-1,2))[0][0]
            u1_offset = [0, 1]
            edge1_interp_u1_idx = edge1_interp_idx + u1_offset
            rv['edge1_u1_pos'] = (grid_obj.get_variable_at_index(grid_obj.edge1_lon, edge1_interp_u1_idx.reshape(-1,2))[0][0],
                                  grid_obj.get_variable_at_index(grid_obj.edge1_lat, edge1_interp_u1_idx.reshape(-1,2))[0][0])
            rv['edge1_u1_value'] = grid_obj.get_variable_at_index(self.gc.u.data[0,-1,:], edge1_interp_u1_idx.reshape(-1,2))[0][0]
            
            e2_padding = grid_obj.get_padding_by_location('edge2')
            edge2_interp_idx = grid_obj.apply_padding_to_idxs(index.copy(), padding=e2_padding)
            rv['edge2_interp_idx'] = edge2_interp_idx
            rv['edge2_v0_pos'] = (grid_obj.get_variable_at_index(grid_obj.edge2_lon, edge2_interp_idx.reshape(-1,2))[0][0],
                                  grid_obj.get_variable_at_index(grid_obj.edge2_lat, edge2_interp_idx.reshape(-1,2))[0][0])
            rv['edge2_v0_value'] = grid_obj.get_variable_at_index(self.gc.v.data[0,-1,:], edge2_interp_idx.reshape(-1,2))[0][0]
            v1_offset = [1, 0]
            edge2_interp_v1_idx = edge2_interp_idx + v1_offset
            rv['edge2_v1_pos'] = (grid_obj.get_variable_at_index(grid_obj.edge2_lon, edge2_interp_v1_idx.reshape(-1,2))[0][0],
                                  grid_obj.get_variable_at_index(grid_obj.edge2_lat, edge2_interp_v1_idx.reshape(-1,2))[0][0])
            rv['edge2_v1_value'] = grid_obj.get_variable_at_index(self.gc.v.data[0,-1,:], edge2_interp_v1_idx.reshape(-1,2))[0][0]
            rv['node_lons'] = grid_obj.get_variable_by_index(grid_obj.node_lon, index.reshape(-1,2))[0]
            rv['node_lats'] = grid_obj.get_variable_by_index(grid_obj.node_lat, index.reshape(-1,2))[0]
            return rv
       


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        #Setup the main window
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)
        self.fig = fig = Figure(figsize=(15, 18))
        self.static_canvas = static_canvas = FigureCanvas(fig)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        layout.addWidget(NavigationToolbar(static_canvas, self)) #pan, zoom, save controls
        layout.addWidget(static_canvas)
        buttons = QtWidgets.QGroupBox()
        buttons.sizePolicy().setHorizontalPolicy(QtWidgets.QSizePolicy.Minimum)
        buttons.setFixedHeight(50)
        hblayout = QtWidgets.QHBoxLayout(buttons)
        self.center_markers_water_checkbox = QtWidgets.QCheckBox('Center Markers (Water)')
        self.center_markers_land_checkbox = QtWidgets.QCheckBox('Center Markers (Land)')
        self.edge1_markers_water_checkbox = QtWidgets.QCheckBox('Edge1 Markers (Water)')
        self.edge1_markers_land_checkbox = QtWidgets.QCheckBox('Edge1 Markers (Land)')
        self.edge2_markers_water_checkbox = QtWidgets.QCheckBox('Edge2 Markers (Water)')
        self.edge2_markers_land_checkbox = QtWidgets.QCheckBox('Edge2 Markers (Land)')
        hblayout.addWidget(self.center_markers_land_checkbox)
        hblayout.addWidget(self.center_markers_water_checkbox)
        hblayout.addWidget(self.edge1_markers_land_checkbox)
        hblayout.addWidget(self.edge1_markers_water_checkbox)
        hblayout.addWidget(self.edge2_markers_land_checkbox)
        hblayout.addWidget(self.edge2_markers_water_checkbox)
        layout.addWidget(buttons)
        
        #Define the projections in use
        self.plot_crs_class = ccrs.Orthographic
        self.g3_crs_class = ccrs.PlateCarree
        
        #Open the file and get the grid
        self.g3 = g3 = GridGeoGenerator(crs = self.g3_crs_class(), filename=test_filename)
        self.gc = g3.gc
        self.grid_obj = grid_obj = self.gc.grid
        
        #Set up the map extent and projection transformation
        extent = g3.get_max_extent(grid_obj)
        g3.crs = self.g3_crs_class()
        self.plot_crs = plot_crs = self.plot_crs_class(extent[0] + (extent[1] - extent[0])/2, extent[2] + (extent[3] - extent[2])/2)
        
        self.ctrans = ctrans = Transformer.from_proj(g3.crs, plot_crs)
        ws = ctrans.transform(extent[0], extent[2])
        en = ctrans.transform(extent[1], extent[3])
        
        self.map_ax = ax = fig.add_subplot(1, 1, 1, projection=plot_crs)
        ax.set_extent(extent, crs=g3.crs)
        
        #Add the map features from cartopy
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('50m'), zorder=0, edgecolor='black')

        cid = fig.canvas.mpl_connect('button_press_event', self.cell_index_lookup)
        
        #add the node lines
        print("Drawing node lines")
        node_lines = g3.gen_grid_lines(grid_obj, 'node', use_mask=False)
        node_lines.to_crs(plot_crs).plot(ax=ax, **g3.node_line_appearance)
        
        #add the node markers callbacks. Perhaps move markerscale to the appearances?
        self.markerscale = 60
        self.center_markers_land = self.center_markers_water = self.edge1_markers_land = self.edge1_markers_water = self.edge2_markers_land = self.edge2_markers_water = None
        self.center_markers_land_checkbox.stateChanged.connect(self.callback_wrapper(self.toggle_markers, 'center', 'land'))
        self.center_markers_water_checkbox.stateChanged.connect(self.callback_wrapper(self.toggle_markers, 'center', 'water'))
        self.edge1_markers_land_checkbox.stateChanged.connect(self.callback_wrapper(self.toggle_markers, 'edge1', 'land'))
        self.edge1_markers_water_checkbox.stateChanged.connect(self.callback_wrapper(self.toggle_markers, 'edge1', 'water'))
        self.edge2_markers_land_checkbox.stateChanged.connect(self.callback_wrapper(self.toggle_markers, 'edge2', 'land'))
        self.edge2_markers_water_checkbox.stateChanged.connect(self.callback_wrapper(self.toggle_markers, 'edge2', 'water'))
        
        #add the spill position
        if spill_location is not None:
            self.draw_spill(spill_location)
        
    def callback_wrapper(self, func, location, land_or_water, *args, **kwargs):
        return lambda state: func(state, self, location, land_or_water, *args, **kwargs)
    
    @staticmethod
    def toggle_markers(state, self, location, land_or_water):
        markers = getattr(self, location+'_markers_' + land_or_water)
        print(markers)
        print(self)
        print(location)
        g3 = self.g3
        grid_obj = self.grid_obj
        if state:
            if markers is None:
                print("Drawing cell centers")
                markers = g3.gen_grid_points(grid_obj, location).to_crs(self.plot_crs)
                mask_raw = appearance = None
                if land_or_water == 'water':
                   mask_raw = g3.get_mask_raw(grid_obj, location, scale=self.markerscale)
                   appearance = getattr(g3, location+'_marker_appearance')
                else:
                   mask_raw = g3.get_mask_raw_invert(grid_obj, location, scale=self.markerscale)
                   appearance = getattr(g3, location+'_marker_masked_appearance')
                c1 = self.map_ax.scatter(markers.x, markers.y, s=mask_raw, **appearance)
                setattr(self, location+'_markers_' + land_or_water, c1)
                markers = c1
            markers.set_visible(True)
        else:
            if markers is not None:
                markers.set_visible(False)
        self.fig.canvas.draw()

    def cell_index_lookup(self, event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                 ('double' if event.dblclick else 'single', event.button,
                  event.x, event.y, event.xdata, event.ydata))
            coord = sgeom.Point(self.ctrans.transform(event.xdata, event.ydata, direction='INVERSE'))
            print(coord)
            print(dir(event))
            if (event.button == 1 and event.dblclick):
                query_result = self.g3.index_query(self.grid_obj, (coord.x, coord.y))
                print(query_result)
                self.draw_query_result(query_result)
                #Double click to query a position.

    def draw_query_result(self, query_result):
        #query result is a dict with the following keys:
        #cell_index, edge1_interp_idx, edge1_u0_pos, edge1_u1_pos, edge2_interp_idx, edge2_v0_pos, edge2_v1_pos, node_lons, node_lats
        node_coords = self.ctrans.transform(query_result['node_lons'], query_result['node_lats'])
        print(node_coords)
        self.map_ax.scatter(*node_coords, s=100, color='green', marker='s')
        u_coords = zip(self.ctrans.transform(*query_result['edge1_u0_pos']), self.ctrans.transform(*query_result['edge1_u1_pos']))
        self.map_ax.scatter(*u_coords, s=100, color='blue', marker='^')
        self.map_ax.annotate(
                f"{query_result['edge1_interp_idx']}\n" +
                "{:.3f},{:.3f}\n".format(*query_result['edge1_u0_pos']) +
                "u = {:.3f}".format(query_result['edge1_u0_value']),
                self.ctrans.transform(*query_result['edge1_u0_pos']),
                xytext=(0, -25),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
        )
        self.map_ax.annotate(
                f"{query_result['edge1_interp_idx'] + [0,1]}\n" +
                "{:.3f},{:.3f}\n".format(*query_result['edge1_u1_pos']) +
                "u = {:.3f}".format(query_result['edge1_u1_value']),
                self.ctrans.transform(*query_result['edge1_u1_pos']),
                xytext=(0, -25),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.5,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
        )
        #plot v stuff
        v_coords = zip(self.ctrans.transform(*query_result['edge2_v0_pos']), self.ctrans.transform(*query_result['edge2_v1_pos']))
        self.map_ax.scatter(*v_coords, s=100, color='red', marker='^')
        self.map_ax.annotate(
                f"{query_result['edge2_interp_idx']}\n" +
                "{:.3f},{:.3f}\n".format(*query_result['edge2_v0_pos']) +
                "v = {:.3f}".format(query_result['edge2_v0_value']),
                self.ctrans.transform(*query_result['edge2_v0_pos']),
                xytext=(0, -25),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
        )
        self.map_ax.annotate(
                f"{query_result['edge2_interp_idx'] + [0,1]}\n" +
                "{:.3f},{:.3f}\n".format(*query_result['edge2_v1_pos']) +
                "v = {:.3f}".format(query_result['edge2_v1_value']),
                self.ctrans.transform(*query_result['edge2_v1_pos']),
                xytext=(0, -25),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.5,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
        )
        self.fig.canvas.draw()
        pass
    
    def draw_spill(self, spill_location):
        #Draws the spill location on the map
        spill_pos = sgeom.Point(self.ctrans.transform(*spill_location))
        self.map_ax.scatter(spill_pos.x, spill_pos.y, s=100, color='red', marker='*')
        pass

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