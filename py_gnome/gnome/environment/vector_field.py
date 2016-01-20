import netCDF4 as nc4
import numpy as np
from gnome.utilities.geometry.cy_point_in_polygon import ps_in_poly
from datetime import datetime, timedelta
from dateutil import parser
import pyugrid


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
    xi_psi = len(dataset.dimensions['xi_psi'])
    eta_psi = len(dataset.dimensions['eta_psi'])
    dims = {'xi_psi': xi_psi, 'eta_psi': eta_psi}
    faces = np.array([np.array([[x, x + 1, x + xi_psi + 1, x + xi_psi]
                                for x in range(0, xi_psi - 1, 1)]) + y * xi_psi for y in range(0, eta_psi - 1)])
    faces = np.ascontiguousarray(
        faces.reshape(((eta_psi - 1) * (xi_psi - 1), 4)))
    nodes = np.column_stack((dataset['lon_psi'][:].reshape(
        eta_psi * xi_psi), dataset['lat_psi'][:].reshape(eta_psi * xi_psi)))
    nodes = np.ascontiguousarray(nodes)
    grid = pyugrid.UGrid(nodes,
                         faces,
                         curv_x=xi_psi,
                         curv_y=eta_psi,
                         curvilinear=True)
    time = Time(dataset['ocean_time'])
    u = pyugrid.SUVar('u', data=dataset['u'])
    v = pyugrid.SUVar('v', data=dataset['v'])
    u_lon = dataset['lon_u'][:]
    u_lat = dataset['lat_u'][:]
    v_lon = dataset['lon_v'][:]
    v_lat = dataset['lat_v'][:]
    u_mask = pyugrid.SUVar('u_mask', data=dataset['mask_u'])
    v_mask = pyugrid.SUVar('v_mask', data=dataset['mask_v'])
    land_mask = pyugrid.SUVar('psi_mask', data=dataset['mask_psi'])
    variables = {'u': u,
                 'v': v,
                 'u_lon': u_lon,
                 'u_lat': u_lat,
                 'v_lon': v_lon,
                 'v_lat': v_lat,
                 'u_mask': u_mask,
                 'v_mask': v_mask,
                 'land_mask': land_mask,
                 'time': time}
    type = 'curvilinear'
    return SField(grid, 'u', 'v', time=time, variables=variables, type=type, dimensions=dims)


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
                 appearance={}
                 ):
        self.grid = grid
        if grid.edges is None:
            self.grid.build_edges()
        if grid.face_face_connectivity is None:
            self.grid.build_face_face_connectivity()
        self.grid.mesh_name = name
        self.grid_type = type
        self.time = time
        self.variables = variables
        for k, v in self.variables.items():
            setattr(self, k, v)
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
             'n_size': 2}
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

    def get_node_velocities(self, time):
        '''
        TODO: implement and check a cache to avoid excessive disk lookup
        Returns a numpy array containing the velocities at each node at the specified time.
        :param time: a datetime object within the bounds of the time
        :type time: datetime.datetime

        '''
        t_alpha = self.time.interp_alpha(time)
        t_index = self.time.indexof(time)
        v0 = self.velocities[t_index]
        v1 = self.velocities[t_index + 1]
        return v0 + (v1 - v0) * t_alpha

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
        time_interp_vels = self.get_node_velocities(
            time)[self.grid.faces[indices]]

        # scaled vels = [us,vs] = [(u1*alpha1 + u2*a2 + u3*a3), (v1*a1 + v2*a2
        # + v3*a3)]
        return np.sum(time_interp_vels * pos_alphas[:, :, np.newaxis], axis=1)

    def get_edges(self, bounds=None):
        """

        :param bounds: Optional bounding box. Expected is lower left corner and top right corner in a tuple
        :return: array of pairs of lon/lat points describing all the edges in the grid, or only those within
        the bounds, if bounds is specified.
        """
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

    def s_poly(self, index, var):
        arr = var[:]
        x = index[:, 0]
        y = index[:, 1]
        return np.stack((arr[x, y], arr[x + 1, y], arr[x + 1, y + 1], arr[x, y + 1]), axis=1)

    def interpolated_velocities(self, time, points, depth=0):
        '''
        For every point at time, find the index on the psi grid.
        Then, compute the index on the u and v grids.
        Retrieve the information, then interpolate
        '''
        indices = self.grid.locate_faces(points)
        psi_x = indices % (self.u_lon.shape[1] - 1)
        psi_y = indices / (self.u_lon.shape[1] - 1)
        psi_ind = np.column_stack((psi_y, psi_x))

        u_ind = np.copy(psi_ind)
        v_ind = np.copy(psi_ind)
#         polys = self.s_poly(psi_ind, self.v_pts)
#         t = np.logical_and(points[..., 0] >= polys[..., 0, 0],
#                            points[..., 0] < polys[..., 2, 0])
        x = psi_x
        y = psi_y
        t = np.logical_and(points[..., 1] >= (self.u_lat[x, y]),
                           points[..., 1] < (self.u_lat[x + 1, y]))

        u_ind[~t] += [1, 0]
#         t = np.logical_and(points[..., 1] >= polys[..., 0, 1],
#                            points[..., 1] < polys[..., 1, 1])
        t = np.logical_and(points[..., 0] >= self.v_lon[x, y],
                           points[..., 0] < self.v_lon[x, y + 1])
        v_ind[~t] += [0, 1]

        '''add get time index here'''
        u_vals = self.u[0, depth]
        u_coords = self.s_poly(u_ind, np.stack((self.u_lon, self.u_lat), -1))
        u_interp_vals = self.s_poly(u_ind, u_vals)

        pass
