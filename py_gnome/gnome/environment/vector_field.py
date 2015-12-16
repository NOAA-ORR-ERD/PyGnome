import netCDF4 as nc4
import numpy as np
from datetime import datetime, timedelta
import pyugrid
import gnome.utilities.profiledeco as pd
import cell_tree2d.cell_tree2d as ct


class VectorField(object):

    def __init__(self, grid=None, variables=None):
        self.grid = grid
        self.variables = variables

class TriVectorField(object):
    '''
    This class takes a netCDF file containing current or wind information on an unstructured grid
    and provides an interface to retrieve this information.
    '''

    v_desc = {'bnd': 'Nodes of a boundary edge, and edge info (n1, n2, island #, land(0)/water(1))',
              'time': 'Time offset from a base date',
              'lon': 'Longitude in decimal degrees',
              'lat': 'Latitude in decimal degrees',
              'nbe': 'Indices of neighbor face across edge in same order as nodes',
              'nv': 'Indices of nodes that define a face',
              'u': 'E/W velocity on node or face',
              'v': 'N/S velocity on node or face'}

    vars = {'bnd': {'desc': v_desc['bnd'], 'dtype': np.int32, 'dims': ['nbnd', 'nbi']},
            'time': {'desc': v_desc['time'], 'dtype': np.float32, 'dims': ['time']},
            'lon': {'desc': v_desc['lon'], 'dtype': np.float32, 'dims': ['node']},
            'lat': {'desc': v_desc['lat'], 'dtype': np.float32, 'dims': ['node']},
            'nbe': {'desc': v_desc['nbe'], 'dtype': np.int32, 'dims': ['three', 'nele']},
            'nv': {'desc': v_desc['nv'], 'dtype': np.int32, 'dims': ['three', 'nele']},
            'u': {'desc': v_desc['u'], 'dtype': np.float32, 'dims': ['time', 'node']},
            'v': {'desc': v_desc['v'], 'dtype': np.float32, 'dims': ['time', 'node']}}
    dims = ['node', 'nele', 'nbnd', 'nbi', 'time', 'three']
    tri_attributes = {'grid_type': 'Triangular',
                      'dimensions': dims,
                      'vars': vars}
    vel_on_nodes = True

    def __init__(self, filename=None, dataset=None):
        if dataset is None:
            dataset = nc4.Dataset(filename)

        self.validate_tri_grid(dataset)
        nodes = np.ascontiguousarray(np.column_stack((dataset['lon'], dataset['lat']))).astype(np.double)
        faces = np.ascontiguousarray(np.array(dataset['nv']).T - 1)
        boundaries = np.ascontiguousarray(np.array(dataset['bnd'])[:,0:2] - 1)
        neighbors = np.ascontiguousarray(np.array(dataset['nbe']).T -1)
        pyugrid.UGrid.__init__(self,nodes, faces, boundaries=boundaries, face_face_connectivity=neighbors)
        self.build_edges()
        self._data = {'time': pyugrid.UVar('time', data=dataset['time']), 'u': dataset['u'], 'v': dataset['v']}
        self.grid_type = dataset.grid_type
        self._base_time = datetime.strptime(self.data['time'].base_date, '%Y-%m-%d %H:%M:%S %Z')
        self._min_time = self.base_time + timedelta(seconds=int(self.data['time'][0]))
        self._max_time = self.base_time + timedelta(seconds=int(self.data['time'][len(self.data['time'])-1]))

    def validate_tri_grid(self, dataset):
        for k in self.vars:
            if k not in dataset.variables.keys():
                raise ValueError(
                    'Necessary variable {} not in dataset'.format(k))
            if self.vars[k]['dtype'] != dataset.variables[k].dtype:
                raise ValueError('dtype for {} inconsistent; Expected {}, got {}'.format(
                    k, self.vars[k]['dtype'], dataset.variables[k].dtype))
        if 'nele' in dataset['u'].dimensions:
            self.vars['u']['dims'] = ['time', 'nele']
            self.vars['v']['dims'] = ['time', 'nele']
            self.vel_on_nodes = False

    @property
    def base_time(self):
        return self._base_time

    @property
    def min_time(self):
        return self._min_time

    @property
    def max_time(self):
        return self._max_time

    def get_time_array(self):
        return self.data['time'][:]

    def time_in_bounds(self, time):
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the data ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def time_index(self, time):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.
        :param time:
        :return:
        '''
        self.valid_time(time)
        delta_t = (time - self.base_time).total_seconds()
        index = np.searchsorted(self.get_time_array(),delta_t) - 1
        return index

    @pd.profile
    def get_node_velocities(self, time):
        '''
        TODO: implement and check a cache to avoid excessive disk lookup
        Returns a numpy array containing the velocities at each node at the specified time.
        :param time: a datetime object within the bounds of the data
        :type time: datetime.datetime

        '''
        self.valid_time(time)
        delta_t = (time - self.base_time).total_seconds()
        index = np.searchsorted(self.get_time_array(),delta_t)
        # if len(self.vel_cache) == 0:
        #     cache = {index: np.column_stack((self.data['u'][index].data, self.data['v'][index].data))}
        return np.column_stack((self.data['u'][index].data, self.data['v'][index].data))

    @pd.profile
    def interpolated_velocities(self, time, points):
        """
        Returns the velocities at each of the points at the specified time, using interpolation
        on the nodes of the triangle that the point is in.
        :param time:
        :param points:
        :return:
        """

        # cyvf is the cython portion of the class....only does celltree interfacing right now
        indices = np.ma.array(self.locate_faces(points), shrink=False)
        mask = indices.mask
        nodes = np.ma.array(self.nodes, mask=mask)
        faces = np.ma.array(self.faces, mask=mask)
        node_positions = nodes[faces[indices]]
        (lon1,lon2,lon3) = node_positions[:,:,0].T
        (lat1,lat2,lat3) = node_positions[:,:,1].T
        reflats = points[:,1]
        reflons = points[:,0]

        # denom = (vertex3.v-vertex1.v)*(vertex2.h-vertex1.h)-(vertex3.h-vertex1.h)*(vertex2.v-vertex1.v);
        denoms = ((lat3 - lat1) * (lon2 - lon1) - (lon3 - lon1) * (lat2 - lat1))
        # alphas should all add up to 1
        alpha1s = (reflats - lat3) * (lon3 - lon2) - (reflons - lon3) * (lat3 - lat2)
        alpha2s = (reflons - lon1) * (lat3 - lat1) - (reflats - lat1) * (lon3 - lon1)
        alpha3s = (reflats - lat1) * (lon2 - lon1) - (reflons - lon1) * (lat2 - lat1)
        alphas = np.column_stack((alpha1s / denoms, alpha2s / denoms, alpha3s / denoms))

        i0 = self.time_index(time)
        t0 = self.base_time + timedelta(seconds=int(self.data['time'][i0]))
        t1 = self.base_time + timedelta(seconds=int(self.data['time'][i0 + 1]))
        base_vels = self.get_node_velocities(t0)[self.faces[indices]]
        next_vels = self.get_node_velocities(t1)[self.faces[indices]]
        t_interval = t1-t0
        t_alpha = ((time - t0).total_seconds()/(t1-t0).total_seconds())
        time_interp_vels = base_vels + (next_vels - base_vels) * t_alpha
        # scaled vels = [us,vs] = [(u1*alpha1 + u2*a2 + u3*a3), (v1*a1 + v2*a2 + v3*a3)]


        return np.column_stack((np.sum(time_interp_vels[:,:,0] * alphas, axis=1), np.sum(time_interp_vels[:,:,1] * alphas, axis=1)))


    def get_gridlines(self):
        return self.nodes[self.edges]

if __name__ == "__main__":
    vf = TriVectorField('data\COOPSu_CREOFS24.nc')
    # vf = TriVectorField('data\\21_tri_mesh.nc')
    result = vf.locate_faces((0.,0.))
    print type(result)
    print result
    result = vf.locate_faces(np.array(([0.,0.],)))
    print type(result)
    print result

    # result = vf.locate_faces((0.,0.),simple=True)
    # print type(result)
    # print result
    tris = vf.nodes[vf.faces]
    cents = (tris[:,0,:] + tris[:,1,:] + tris[:,2,:])/3
    indices = vf.locate_faces(cents.astype(np.double))
    dts = [datetime(2015,9,24,3,0,) + timedelta(hours= x, minutes=30) for x in range(-2,47)]

    for i in range(0,48):
        vels = vf.interpolated_velocities(dts[1],cents)

    pd.print_stats(.1)
    pass