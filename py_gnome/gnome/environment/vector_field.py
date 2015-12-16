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

    def __init__(self, filename=None, dataset=None):
        if dataset is None:
            dataset = nc4.Dataset(filename)

        nodes = np.ascontiguousarray(np.column_stack((dataset['lon'], dataset['lat']))).astype(np.double)
        faces = np.ascontiguousarray(np.array(dataset['nv']).T - 1)
        boundaries = np.ascontiguousarray(np.array(dataset['bnd'])[:,0:2] - 1)
        neighbors = np.ascontiguousarray(np.array(dataset['nbe']).T -1)
        self.grid = pyugrid.UGrid(nodes, faces, boundaries=boundaries, face_face_connectivity=neighbors)
        self.grid.build_edges()
        self.grid_type = dataset.grid_type
        u = pyugrid.UVar('u','node', dataset['u'])
        v = pyugrid.UVar('v','node', dataset['v'])
        time = pyugrid.UVar('time','none', dataset['time'])
        self.variables = {'time': time, 'velocities': pyugrid.UMVar('velocities', 'node', [u,v])}
        for k,v in self.variables.items():
            setattr(self,k,v)
        self._base_time = datetime.strptime(self.time.base_date, '%Y-%m-%d %H:%M:%S %Z')
        self._min_time = self.base_time + timedelta(seconds=int(self.time[0]))
        self._max_time = self.base_time + timedelta(seconds=int(self.time[len(self.time)-1]))

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
        return self.time[:]

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
        return self.velocities[index]

    def get_interpolated_node_velocities(self, time):
        i0 = self.time_index(time)
        t0 = self.base_time + timedelta(seconds=int(self.time[i0]))
        t1 = self.base_time + timedelta(seconds=int(self.time[i0 + 1]))
        base_vels = self.get_node_velocities(t0)
        next_vels = self.get_node_velocities(t1)
        t_alpha = ((time - t0).total_seconds()/(t1-t0).total_seconds())
        time_interp_vels = base_vels + (next_vels - base_vels) * t_alpha
        return time_interp_vels

    @pd.profile
    def interpolated_velocities(self, time, points):
        """
        Returns the velocities at each of the points at the specified time, using interpolation
        on the nodes of the triangle that the point is in.
        :param time: The time in the simulation
        :param points: a numpy array of points that you want to find interpolated velocities for
        :return: interpolated velocities at the specified points
        """
        indices = self.grid.locate_faces(points)
        alphas = self.grid.interpolation_alphas(points,indices)
        time_interp_vels = self.get_interpolated_node_velocities(time)[self.grid.faces[indices]]

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