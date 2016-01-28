import warnings

import netCDF4 as nc4
import numpy as np

from gnome.utilities.geometry.cy_point_in_polygon import points_in_polys
from datetime import datetime, timedelta
from dateutil import parser

import pyugrid
import pysgrid


def tri_vector_field(filename=None, dataset=None):
    if dataset is None:
        dataset = nc4.Dataset(filename)

    nodes = np.ascontiguousarray(
        np.column_stack((dataset['lon'], dataset['lat']))).astype(np.double)
    faces = np.ascontiguousarray(np.array(dataset['nv']).T - 1)
    boundaries = np.ascontiguousarray(np.array(dataset['bnd'])[:, 0:2] - 1)
    neighbors = np.ascontiguousarray(np.array(dataset['nbe']).T - 1)
    edges = None
    grid = pyugrid.UGrid(nodes,
                         faces,
                         edges,
                         boundaries,
                         neighbors)
    u = pyugrid.UVar('u', 'node', dataset['u'])
    v = pyugrid.UVar('v', 'node', dataset['v'])
    time = Time(dataset['time'])
    variables = {'velocities': pyugrid.UMVar('velocities', 'node', [u, v])}
    type = dataset.grid_type
    return VectorField(grid, time=time, variables=variables, type=type)


def ice_field(filename=None, dataset=None):
    if dataset is None:
        dataset = nc4.Dataset(filename)
    y_size = len(dataset.dimensions['y'])
    x_size = len(dataset.dimensions['x'])
    dims = {'x': x_size, 'y': y_size}
    faces = np.array([np.array([[x, x + 1, x + x_size + 1, x + x_size]
                                for x in range(0, x_size - 1, 1)]) + y * x_size for y in range(0, y_size - 1)])
    faces = np.ascontiguousarray(
        faces.reshape(((y_size - 1) * (x_size - 1), 4)))
    nodes = np.column_stack((dataset['lon'][:].reshape(
        y_size * x_size), dataset['lat'][:].reshape(y_size * x_size)))
    nodes = np.ascontiguousarray(nodes)
    grid = pyugrid.UGrid(nodes,
                         faces,
                         curvilinear=True)
    time = Time(dataset['time'])
    w_u = pyugrid.UVar('water_u', 'node', dataset['water_u'], curvilinear=True)
    w_v = pyugrid.UVar('water_v', 'node', dataset['water_v'], curvilinear=True)
    mask = pyugrid.UVar('mask', 'node', dataset['mask'], curvilinear=True)
    i_u = pyugrid.UVar('ice_u', 'node', dataset['ice_u'], curvilinear=True)
    i_v = pyugrid.UVar('ice_v', 'node', dataset['ice_v'], curvilinear=True)
    thickness = pyugrid.UVar(
        'ice_thickness', 'node', dataset['ice_thickness'], curvilinear=True)
    fraction = pyugrid.UVar(
        'ice_fraction', 'node', dataset['ice_fraction'], curvilinear=True)
    variables = {'water_vel': pyugrid.UMVar('water_vel', 'node', [w_u, w_v]),
                 'ice_vel': pyugrid.UMVar('ice_vel', 'node', [i_u, i_v]),
                 'ice_thickness': thickness,
                 'ice_fraction': fraction,
                 'mask': mask,
                 'time': time}
    type = dataset.grid_type
    return VectorField(grid, time=time, variables=variables, type=type, dimensions=dims)


def roms_field(filename=None, dataset=None):
    if dataset is None:
        dataset = nc4.Dataset(filename)
    grid = pysgrid.from_nc_dataset(dataset)
    grid.build_edges()

    time = Time(dataset['ocean_time'])
    u = pyugrid.SUVar('u', data=dataset['u'])
    v = pyugrid.SUVar('v', data=dataset['v'])
    u_nodes = np.stack((dataset['lon_u'][:], dataset['lat_u'][:]), -1)
    v_nodes = np.stack((dataset['lon_v'][:], dataset['lat_v'][:]), -1)
    u_mask = dataset['mask_u'][:]
    v_mask = dataset['mask_v'][:]
    land_mask = dataset['mask_psi'][:]
    variables = {'u': u,
                 'v': v,
                 'u_nodes': u_nodes,
                 'v_nodes': v_nodes,
                 'u_mask': u_mask,
                 'v_mask': v_mask,
                 'land_mask': land_mask,
                 'time': time}
    return SField(grid, 'u', 'v', time=time, variables=variables, dimensions=grid.nodes.shape)


class VectorField(object):
    '''
    This class takes a netCDF file containing current or wind information on an unstructured grid
    and provides an interface to retrieve this information.
    '''

    def __init__(self, grid,
                 time=None,
                 variables=None,
                 name=None,
                 type=None,
                 dimensions=None,
                 velocities=None,
                 appearance={}
                 ):
        self.grid = grid
        if grid.edges is None:
            self.grid.build_edges()
#         if grid.face_face_connectivity is None:
#             self.grid.build_face_face_connectivity()
        self.grid_type = type
        self.time = time
        self.variables = variables
        for k, v in self.variables.items():
            setattr(self, k, v)

        if not hasattr(self, 'velocities'):
            self.velocities = velocities
        self._appearance = {}
        self.set_appearance(**appearance)

    def set_appearance(self, **kwargs):
        self._appearance.update(kwargs)

    @property
    def appearance(self):
        d = {'on': False,
             'color': 'grid_1',
             'width': 1,
             'filled': False,
             'mask': None,
             'n_size': 2,
             'curvilinear': False}
        d.update(self._appearance)
        return d

    @property
    def nodes(self):
        return self.grid.nodes

    @property
    def faces(self):
        return self.grid.faces

    @property
    def triangles(self):
        return self.grid.nodes[self.grid.faces]

    def interpolated_velocities(self, time, points):
        """
        Returns the velocities at each of the points at the specified time, using interpolation
        on the nodes of the triangle that the point is in.
        :param time: The time in the simulation
        :param points: a numpy array of points that you want to find interpolated velocities for
        :return: interpolated velocities at the specified points
        """
        indices = self.grid.locate_faces(points)
        pos_alphas = self.grid.interpolation_alphas(points, indices)
        # map the node velocities to the faces specified by the points
        t_alpha = self.time.interp_alpha(time)
        t_index = self.time.indexof(time)
        v0 = self.velocities[t_index]
        v1 = self.velocities[t_index + 1]
        node_vels = v0 + (v1 - v0) * t_alpha
        time_interp_vels = node_vels[self.grid.faces[indices]]

        # scaled vels = [us,vs] = [(u1*alpha1 + u2*a2 + u3*a3), (v1*a1 + v2*a2
        # + v3*a3)]
        return np.sum(time_interp_vels * pos_alphas[:, :, np.newaxis], axis=1)

    def interpolate(self, time, points, field):
        """
        Returns the velocities at each of the points at the specified time, using interpolation
        on the nodes of the triangle that the point is in.
        :param time: The time in the simulation
        :param points: a numpy array of points that you want to find interpolated velocities for
        :param field: the value field that you want to interpolate over. 
        :return: interpolated velocities at the specified points
        """
        indices = self.grid.locate_faces(points)
        pos_alphas = self.grid.interpolation_alphas(points, indices)
        # map the node velocities to the faces specified by the points
        t_alpha = self.time.interp_alpha(time)
        t_index = self.time.indexof(time)
        f0 = field[t_index]
        f1 = field[t_index + 1]
        node_vals = f0 + (f1 - f0) * t_alpha
        time_interp_vels = node_vels[self.grid.faces[indices]]

        return np.sum(time_interp_vels * pos_alphas[:, :, np.newaxis], axis=1)

    def get_edges(self, bounds=None):
        """

        :param bounds: Optional bounding box. Expected is lower left corner and top right corner in a tuple
        :return: array of pairs of lon/lat points describing all the edges in the grid, or only those within
        the bounds, if bounds is specified.
        """
        return self.grid.edges
        if bounds is None:
            return self.grid.nodes[self.grid.edges]
        else:
            lines = self.grid.nodes[self.grid.edges]

            def within_bounds(line, bounds):
                pt1 = (bounds[0][0] <= line[0, 0] * line[0, 0] <= bounds[1][0] and
                       bounds[0][1] <= line[0, 1] * line[:, 0, 1] <= bounds[1][1])
                pt2 = (bounds[0][0] <= line[1, 0] <= bounds[1][0] and
                       bounds[0][1] <= line[1, 1] <= bounds[1][1])
                return pt1 or pt2
            pt1 = ((bounds[0][0] <= lines[:, 0, 0]) * (lines[:, 0, 0] <= bounds[1][0]) *
                   (bounds[0][1] <= lines[:, 0, 1]) * (lines[:, 0, 1] <= bounds[1][1]))
            pt2 = ((bounds[0][0] <= lines[:, 1, 0]) * (lines[:, 1, 0] <= bounds[1][0]) *
                   (bounds[0][1] <= lines[:, 1, 1]) * (lines[:, 1, 1] <= bounds[1][1]))
            return lines[pt1 + pt2]

    def masked_nodes(self, time, variable):
        """
        This allows visualization of the grid nodes with relation to whether the velocity is masked or not.
        :param time: a time within the simulation
        :return: An array of all the nodes, masked with the velocity mask.
        """
        if hasattr(variable, 'name') and variable.name in self.variables:
            if time < self.time.max_time:
                return np.ma.array(self.grid.nodes, mask=variable[self.time.indexof(time)].mask)
            else:
                return np.ma.array(self.grid.nodes, mask=variable[self.time.indexof(self.time.max_time)].mask)
        else:
            variable = np.array(variable, dtype=bool).reshape(-1, 2)
            return np.ma.array(self.grid.nodes, mask=variable)


class Time(object):

    def __init__(self, data, base_dt_str=None):
        """

        :param data: A netCDF, biggus, or dask source for time data
        :return:
        """
        self.time = nc4.num2date(data[:], units=data.units)

    @property
    def min_time(self):
        return self.time[0]

    @property
    def max_time(self):
        return self.time[-1]

    def get_time_array(self):
        return self.time[:]

    def time_in_bounds(self, time):
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def indexof(self, time):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.
        :param time:
        :return:
        '''
        self.valid_time(time)
        index = np.searchsorted(self.time, time) - 1
        return index

    def interp_alpha(self, time):
        i0 = self.indexof(time)
        t0 = self.time[i0]
        t1 = self.time[i0 + 1]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()


class SField(VectorField):

    def __init__(self, grid, u_name, v_name, **kwargs):
        super(SField, self).__init__(grid, **kwargs)
        self.set_appearance(curvilinear=True)

    def s_poly(self, index, var):
        arr = var[:]
        x = index[:, 0]
        y = index[:, 1]
        return np.stack((arr[x, y], arr[x + 1, y], arr[x + 1, y + 1], arr[x, y + 1]), axis=1)

    def interp_alphas(self, points, grid=None, indices=None, translation=None):
        if grid is None:
            grid = self.grid
            pos_alphas = grid.interpolation_alphas(points, indices)
            return pos_alphas
        if indices is None:
            if translation is not None:
                warnings.warn(
                    "indices not provided, translation ignored", UserWarning)
                translation = None
            indices = grid.locate_faces(points)
        if translation is not None:
            indices = pysgrid.utils.translate_index(
                points, indices, grid, translation)
        pos_alphas = grid.interpolation_alphas(points, indices)
        return pos_alphas

    def interpolated_velocities(self, time, points, indices=None, depth=0):
        points = np.ascontiguousarray(points)
        psi_ind = None
        if indices == None:
            psi_ind = self.grid.locate_faces(points)
        else:
            psi_ind = indices

        u_grid = pysgrid.SGrid2D(nodes=self.u_nodes)
        v_grid = pysgrid.SGrid2D(nodes=self.v_nodes)

        u_alphas = self.interp_alphas(points, u_grid, psi_ind, 'psi2u')
        v_alphas = self.interp_alphas(points, v_grid, psi_ind, 'psi2v')

        t_index = self.time.indexof(time)
        u_vels = self.u[t_index, depth]
        v_vels = self.v[t_index, depth]

        u_interp = np.sum(u_vels * u_alphas)
        v_interp = np.sum(v_vels * v_alphas)

        return np.column_stack(u_interp, v_interp)

    def get_edges(self, bounds=None):
        """

        :param bounds: Optional bounding box. Expected is lower left corner and top right corner in a tuple
        :return: array of pairs of lon/lat points describing all the edges in the grid, or only those within
        the bounds, if bounds is specified.
        """
        return self.grid.edges
