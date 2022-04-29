
import os
import datetime as dt

import pytest
import tempfile

import numpy as np
import netCDF4 as nc
import nucos as uc

from gnome.environment.gridded_objects_base import (Variable,
                                                    VectorVariable,
                                                    PyGrid,
                                                    Time)
from gnome.environment.timeseries_objects_base import (TimeseriesData,
                                                       TimeseriesVector)

from .gen_analytical_datasets import gen_all


@pytest.fixture(scope='class')
def dates():
    return np.array([dt.datetime(2000, 1, 1, 0),
                     dt.datetime(2000, 1, 1, 2),
                     dt.datetime(2000, 1, 1, 4),
                     dt.datetime(2000, 1, 1, 6),
                     dt.datetime(2000, 1, 1, 8), ])


@pytest.fixture(scope='class')
def series_data():
    return np.array([1, 3, 6, 10, 15])


@pytest.fixture(scope='class')
def series_data2():
    return np.array([2, 6, 12, 20, 30])


class TestTime(object):
    test_class = Time

    def test_construction(self, dates):
        t = Time(dates)

        assert t.min_time == t.data[0] == dt.datetime(2000, 1, 1, 0)
        assert t.max_time == t.data[-1] == dt.datetime(2000, 1, 1, 8)

        dates = [dt.datetime(2000, 1, 1, 0),
                 dt.datetime(2000, 1, 1, 2),
                 dt.datetime(2000, 1, 1, 4),
                 dt.datetime(2000, 1, 1, 6),
                 dt.datetime(2000, 1, 1, 8)]

        t2 = self.test_class(dates)

        assert t == t2

    def test_constant_time(self):
        t1 = self.test_class.constant_time()
        t2 = self.test_class.constant_time()
        t3 = self.test_class.constant_time()

        assert t1 == t2 == t3
        assert t1 is t2 is t3

    def test_index_of(self, dates):
        t = Time(dates)
        before = t.min_time - dt.timedelta(hours=1)
        after = t.max_time + dt.timedelta(hours=1)

        assert t.index_of(before, True) == 0
        assert t.index_of(after, True) == 5
        assert t.index_of(t.data[-1], True) == 4
        assert t.index_of(t.data[0], True) == 0

        with pytest.raises(ValueError):
            t.index_of(before, False)

        with pytest.raises(ValueError):
            t.index_of(after, False)

        assert t.index_of(t.max_time, True) == 4
        assert t.index_of(t.min_time, True) == 0

    def test_interp_alpha(self, dates):
        t = Time(dates)
        test_time = dt.datetime(2000, 1, 1, 1)

        assert np.isclose(t.interp_alpha(test_time), 0.5)

    def test_serialize(self, dates):
        t = Time(dates)
        web_ser = t.serialize()
        t2 = self.test_class.deserialize(web_ser)
        assert t == t2

    def test_save_load(self, dates):
        t = Time(dates)
        saveloc = tempfile.mkdtemp()
        saveloc = os.path.join(saveloc, 'test.zip')

        _references = t.save(saveloc)
        new_instance = self.test_class.load(saveloc)

        assert t == new_instance


class TestTimeseriesData(object):
    test_class = TimeseriesData

    def get_tsd_instance(self, dates, series_data):
        return TimeseriesData(time=Time(dates), data=series_data, units=None)

    def test_construction(self, dates, series_data):
        t = self.get_tsd_instance(dates, series_data)
        t2 = self.test_class(time=dates, data=series_data, units=None)

        assert t.time == t2.time
        assert t == t2

    def test_at(self, dates, series_data):
        t = self.get_tsd_instance(dates, series_data)

        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1)),
                           np.array([2]))

        assert np.allclose(t.at(np.array([0, 0]),
                                dt.datetime(1999, 1, 1, 0)),
                           np.array([1]))

        assert np.allclose(t.at(np.array([0, 0]),
                                dt.datetime(2000, 2, 1, 0)),
                           np.array([15]))

        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 0),
                                extrapolate=False),
                           np.array([1]))

        with pytest.raises(ValueError):
            t.at(np.array([0, 0, 0, 0]),
                 dt.datetime(2000, 2, 1, 1),
                 extrapolate=False)

        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1),
                                units='m'),
                        np.array([2, 2, 2]))

        t.units = 'cm/s'
        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1)),
                           np.array([2, 2, 2]))

        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1),
                                units='m/s'),
                           np.array([0.02, 0.02, 0.02]))


    def test_serialize(self, dates, series_data):
        t = self.get_tsd_instance(dates, series_data)
        web_ser = t.serialize()
        t2 = self.test_class.deserialize(web_ser)
        assert t == t2

    def test_save_load(self, dates, series_data):
        t = self.get_tsd_instance(dates, series_data)
        saveloc = tempfile.mkdtemp()
        saveloc = os.path.join(saveloc, 'test.zip')

        _references = t.save(saveloc)
        new_instance = self.test_class.load(saveloc)
        assert t == new_instance


class TestTimeseriesVector(object):
    test_class = TimeseriesVector

    def get_tsv_instance(self, dates, series_data, series_data2):
        _t = Time(dates)

        return TimeseriesVector(variables=[TimeseriesData(name='u', time=_t,
                                                          data=series_data),
                                           TimeseriesData(name='v', time=_t,
                                                          data=series_data2)],
                                units='m/s')

    def test_construction(self, dates, series_data, series_data2):
        _t = self.get_tsv_instance(dates, series_data, series_data2)

        # assert len(t.variables) == 2
        # assert t.time == t.variables[0].time == t.variables[1].time
        # assert t.units == t.variables[0].units == t.variables[1].units

    def test_at(self, dates, series_data, series_data2):
        t = self.get_tsv_instance(dates, series_data, series_data2)

        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1)),
                           np.array([(2, 4)]))

        assert np.allclose(t.at(np.array([0, 0]),
                                dt.datetime(1999, 1, 1, 0)),
                           np.array([(1, 2)]))

        assert np.allclose(t.at(np.array([0,0]),
                                dt.datetime(2000, 2, 1, 0)),
                           np.array([(15, 30)]))

        assert np.allclose(t.at(np.array([0,0,0]),
                                dt.datetime(2000, 1, 1, 0),
                                extrapolate=False),
                           np.array([(1, 2)]))

        with pytest.raises(ValueError):
            t.at(np.array([0, 0, 0, 0]),
                 dt.datetime(2000, 2, 1, 1),
                 extrapolate=False)


        with pytest.raises(uc.UnitConversionError):
            assert np.allclose(t.at(np.array([0, 0, 0]),
                                    dt.datetime(2000, 1, 1, 1),
                                    units='cm'),
                            np.array([(200, 400), (200, 400), (200, 400)]))

        t.units = 'cm/s'
        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1)),
                           np.array([(2, 4), (2, 4), (2, 4)]))

        assert np.allclose(t.at(np.array([0, 0, 0]),
                                dt.datetime(2000, 1, 1, 1),
                                units='m/s'),
                           np.array([(0.02, 0.04), (0.02, 0.04), (0.02, 0.04)]))

    def test_serialize(self, dates, series_data, series_data2):
        t = self.get_tsv_instance(dates, series_data, series_data2)
        web_ser = t.serialize()
        t2 = self.test_class.deserialize(web_ser)
        assert t == t2

    def test_save_load(self, dates, series_data, series_data2):
        tsv = self.get_tsv_instance(dates, series_data, series_data2)
        saveloc = tempfile.mkdtemp()
        saveloc = os.path.join(saveloc, 'test.zip')

        _references = tsv.save(saveloc)
        new_instance = self.test_class.load(saveloc)

        assert tsv == new_instance


# class TestGrid(TestBase):
#     pass
#
#
# class TestDepth(TestBase):
#     pass
#
#
# class TestVariable(TestBase):
#     pass
#
#
# class TestVectorVariable(TestBase):
#     pass

'''
Need to hook this up to existing test data infrastructure
'''

base_dir = os.path.dirname(__file__)

s_data = os.path.join(base_dir, 'sample_data')
#gen_all(base_path=s_data)

sinusoid = os.path.join(s_data, 'staggered_sine_channel.nc')
sinusoid = nc.Dataset(sinusoid)

circular_3D = os.path.join(s_data, '3D_circular.nc')
circular_3D = nc.Dataset(circular_3D)

tri_ring = os.path.join(s_data, 'tri_ring.nc')
tri_ring = nc.Dataset(tri_ring)


# class TestS_Depth:
#
#     def test_construction(self):
#
#         test_grid = Grid_S(node_lon=np.array([[0, 1, 2, 3],
#                                               [0, 1, 2, 3],
#                                               [0, 1, 2, 3],
#                                               [0, 1, 2, 3]]),
#                            node_lat=np.array([[0, 0, 0, 0],
#                                               [1, 1, 1, 1],
#                                               [2, 2, 2, 2],
#                                               [3, 3, 3, 3]]))
#
#         u = np.zeros((3, 4, 4), dtype=np.float64)
#         u[0, :, :] = 0
#         u[1, :, :] = 1
#         u[2, :, :] = 2
#
#         w = np.zeros((4, 4, 4), dtype=np.float64)
#         w[0, :, :] = 0
#         w[1, :, :] = 1
#         w[2, :, :] = 2
#         w[3, :, :] = 3
#
#         bathy_data = -np.array([[1, 1, 1, 1],
#                                 [1, 2, 2, 1],
#                                 [1, 2, 2, 1],
#                                 [1, 1, 1, 1]], dtype=np.float64)
#
#         Cs_w = np.array([1.0, 0.6667, 0.3333, 0.0])
#         s_w = np.array([1.0, 0.6667, 0.3333, 0.0])
#         Cs_r = np.array([0.8333, 0.5, 0.1667])
#         s_rho = np.array([0.8333, 0.5, 0.1667])
#         hc = np.array([1])
#
#         b = Variable(name='bathymetry', data=bathy_data, grid=test_grid,
#                        time=Time.constant_time())
#
#         zeta = Variable(name='zeta', data=)
#         dep = S_Depth(bathymetry=b,
#                          terms=dict(zip(S_Depth.default_terms[0],
#                                         [Cs_w, s_w, hc, Cs_r, s_rho])),
#                          dataset='dummy')
#         assert dep is not None
#
#         corners = np.array([[0, 0, 0],
#                             [0, 3, 0],
#                             [3, 3, 0],
#                             [3, 0, 0]], dtype=np.float64)
#
#         res, alph = dep.interpolation_alphas(corners, Time.constant_time(),
#                                              w.shape)
#         assert res is None  # all particles on surface
#         assert alph is None  # all particles on surface
#
#         res, alph = dep.interpolation_alphas(corners, Time.constant_time(),
#                                              u.shape)
#         assert res is None  # all particles on surface
#         assert alph is None  # all particles on surface
#
#         pts2 = corners + (0, 0, 2)
#         res = dep.interpolation_alphas(pts2, Time.constant_time(), w.shape)
#         assert all(res[0] == 0)  # all particles underground
#         assert np.allclose(res[1], -2.0)  # all particles underground
#
#         res = dep.interpolation_alphas(pts2, Time.constant_time(), u.shape)
#         assert all(res[0] == 0)  # all particles underground
#         assert np.allclose(res[1], -2.0)  # all particles underground
#
#         layers = np.array([[0.5, 0.5, .251],
#                            [1.5, 1.5, 1.0],
#                            [2.5, 2.5, 1.25]])
#
#         res, alph = dep.interpolation_alphas(layers, Time.constant_time(),
#                                              w.shape)
#         print res
#         print alph
#         assert all(res == [3, 2, 1])
#         assert np.allclose(alph, np.array([0.397539, 0.5, 0]))

'''
Analytical cases:

Triangular
    grid shape: (nodes = nv, faces = nele)
    data_shapes: (time, depth, nv),
                 (time, nv),
                 (depth, nv),
                 (nv)
    depth types: (None),
                 (constant),
                 (sigma v1),
                 (sigma v2),
                 (levels)
    test points: 2D surface (time=None, depth=None)
                     - nodes should be valid
                     - off grid should extrapolate with fill value or Error
                     - interpolation elsewhere
                 2D surface (time=t, depth=None)
                     - as above, validate time interpolation



Quad
    grid shape: (nodes:(x,y))
                (nodes:(x,y), faces(xc, yc))
                (nodes:(x,y), faces(xc, yc), edge1(x, yc), edge2(xc, y))
    data_shapes: (time, depth, x, y),
                 (time, x, y),
                 (depth, x, y),
                 (x,y)
    depth types: (None),
                 (constant),
                 (sigma v1),
                 (sigma v2),
                 (levels)

'''


class TestGriddedProp(object):
    def test_construction(self):
        data = sinusoid['u'][:]
        grid = PyGrid.from_netCDF(dataset=sinusoid)
        time = None

        u = Variable(name='u',
                     units='m/s',
                     data=data,
                     grid=grid,
                     time=time,
                     data_file='staggered_sine_channel.nc',
                     grid_file='staggered_sine_channel.nc')

        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')

        k = Variable.from_netCDF(filename=curr_file, varname='u', name='u')

        assert k.name == u.name
        assert k.units == 'm/s'

        # fixme: this was failing
        # assert k.time == u.time
        assert k.data[0, 0] == u.data[0, 0]

    # def test_at(self):
    #     curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
    #     u = Variable.from_netCDF(filename=curr_file, varname='u_rho')
    #     v = Variable.from_netCDF(filename=curr_file, varname='v_rho')

    #     points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
    #     time = dt.datetime.now()

    #     assert np.all(u.at(points, time) == np.array([1, 1, 1]).T)

    #     print np.cos(points[:, 0] / 2) / 2
    #     assert np.all(np.isclose(v.at(points, time),
    #                           np.cos(points[:, 0] / 2) / 2))


class TestGridVectorProp(object):
    def test_construction(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        u = Variable.from_netCDF(filename=curr_file, varname='u_rho')
        v = Variable.from_netCDF(filename=curr_file, varname='v_rho')

        gvp = VectorVariable(name='velocity', units='m/s', time=u.time,
                             variables=[u, v])
        assert gvp.name == 'velocity'
        assert gvp.units == 'm/s'
        assert gvp.varnames[0] == 'u_rho'

    # def test_at(self):
    #     curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
    #     gvp = VectorVariable.from_netCDF(filename=curr_file,
    #                                      varnames=['u_rho', 'v_rho'])
    #     points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
    #     time = dt.datetime.now()

    #     assert np.all(np.isclose(gvp.at(points, time)[:, 1],
    #                           np.cos(points[:, 0] / 2) / 2).T)

    def test_gen_varnames(self):
        import netCDF4 as nc4
        from gnome.environment import GridCurrent, GridWind, IceVelocity

        ds = nc4.Dataset('testname', 'w', diskless=True, persist=False)
        ds.createDimension('y', 5)
        ds.createDimension('x', 5)

        ds.createVariable('x', 'f8', dimensions=('x', 'y'))
        ds['x'].standard_name = 'eastward_sea_water_velocity'

        ds.createVariable('y', 'f8', dimensions=('x', 'y'))
        ds['y'].standard_name = 'northward_sea_water_velocity'

        ds.createVariable('xw', 'f8', dimensions=('x', 'y'))
        ds['xw'].long_name = 'eastward_wind'

        ds.createVariable('yw', 'f8', dimensions=('x', 'y'))
        ds['yw'].long_name = 'northward_wind'

        ds.createVariable('ice_u', 'f8', dimensions=('x', 'y'))
        ds.createVariable('ice_v', 'f8', dimensions=('x', 'y'))

        names = GridCurrent._gen_varnames(dataset=ds)
        assert names[0] == names.u == 'x'
        assert names[1] == names.v == 'y'

        names = GridWind._gen_varnames(dataset=ds)
        assert names[0] == names.u == 'xw'
        assert names[1] == names.v == 'yw'

        names = IceVelocity._gen_varnames(dataset=ds)
        assert names[0] == names.u == 'ice_u'
        assert names[1] == names.v == 'ice_v'

        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        gc = GridCurrent.from_netCDF(filename=curr_file)
        assert gc.u == gc.variables[0]
        assert gc.varnames[0] == 'u'


if __name__ == '__main__':
    pass
