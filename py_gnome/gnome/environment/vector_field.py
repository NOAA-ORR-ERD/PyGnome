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

    grid = pysgrid.load_grid(dataset)

    time = Time(dataset['ocean_time'])
    u = grid.u
    v = grid.v
    u_mask = grid.mask_u
    v_mask = grid.mask_v
    r_mask = grid.mask_rho
    land_mask = grid.mask_psi
    variables = {'u': u,
                 'v': v,
                 'u_mask': u_mask,
                 'v_mask': v_mask,
                 'land_mask': land_mask,
                 'time': time}
    return SField(grid, time=time, variables=variables, dimensions=grid.node_lon.shape)


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
             'type': 'unstructured'}
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
        t_alpha = self.time.interp_alpha(time)
        t_index = self.time.indexof(time)
        v0 = self.velocities[t_index]
        v1 = self.velocities[t_index + 1]
        node_vels = v0 + (v1 - v0) * t_alpha

        return self.grid.interpolate_var_to_points(points, node_vels)

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

    def __init__(self, grid, **kwargs):
        super(SField, self).__init__(grid, **kwargs)
        self.set_appearance(curvilinear=True)

    @property
    def appearance(self):
        d = {'on': False,
             'color': 'grid_1',
             'width': 1,
             'filled': False,
             'mask': None,
             'n_size': 2,
             'type': 'curvilinear'}
        d.update(self._appearance)
        return d

    def s_poly(self, index, var):
        '''
        Function to get the node values of a given face index.
        Emulates the self.grid.nodes[self.grid.faces[index]] paradigm of unstructured grids
        '''
        arr = var[:]
        x = index[:, 0]
        y = index[:, 1]
        return np.stack((arr[x, y], arr[x + 1, y], arr[x + 1, y + 1], arr[x, y + 1]), axis=1)

    def interp_alphas(self, points, grid=None, indices=None, translation=None):
        '''
        Find the interpolation alphas for the four points of the cells that contains the points
        This function is meant to be a universal way to get these alphas, including translating across grids

        If grid is not specified, it will default to the grid contained in self, ignoring any translation specified

        If the grid is specified and indicies is not, it will use the grid's cell location
        function to find the indices of the points. This may incur extra memory usage if the
        grid needs to construct a cell_tree

        If the grid is specified and indices is specified, it will use those indices and points to
        find interpolation alphas. If translation is specified, it will translate the indices
        beforehand.
        :param points: Numpy array of 2D points
        :param grid: The SGrid object that you want to interpolate over
        :param indices: Numpy array of the x,y indices of each point
        :param translation: String to specify an index translation.
        '''
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
        '''
        Finds velocities at the points at the time specified, interpolating in 2D
        over the u and v grids to do so.
        :param time: The time in the simulation
        :param points: a numpy array of points that you want to find interpolated velocities for
        :param indices: Numpy array of indices of the points, if already known.
        :return: interpolated velocities at the specified points
        '''
        points = np.ascontiguousarray(points)

        t_index = self.time.indexof(time)
        t_alphas = self.time.interp_alpha(time)
        u0, u1 = self.u[t_index:t_index + 2, depth]
        v0, v1 = self.v[t_index:t_index + 2, depth]
        u_mask = self.u_mask[:].astype(np.bool)
        v_mask = self.v_mask[:].astype(np.bool)
        land_mask = self.land_mask[:]

        ind = self.grid.locate_faces(points)

        u_vels = u0 + (u1 - u0) * t_alphas
        v_vels = v0 + (v1 - v0) * t_alphas

        u_interp = self.grid.interpolate_var_to_points(
            points, u_vels, ind)
        v_interp = self.grid.interpolate_var_to_points(
            points, v_vels, ind)

        ang_ind = ind + [1, 1]
        angs = self.grid.angles[:][ang_ind[:, 0], ang_ind[:, 1]]
        rotations = np.array(
            ([np.cos(angs), -np.sin(angs)], [np.sin(angs), np.cos(angs)]))

        vels = np.column_stack((u_interp, v_interp))
        return np.matmul(rotations.T, vels[:, :, np.newaxis]).reshape(-1, 2)

    def get_edges(self, bounds=None):
        """

        :param bounds: Optional bounding box. Expected is lower left corner and top right corner in a tuple
        :return: array of pairs of lon/lat points describing all the edges in the grid, or only those within
        the bounds, if bounds is specified.
        """
        return self.grid.get_grid()
