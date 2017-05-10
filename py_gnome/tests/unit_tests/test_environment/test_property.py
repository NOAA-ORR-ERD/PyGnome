import os
import sys
import pytest
import datetime as dt
import numpy as np
import pysgrid
import datetime
from gnome.environment.property import Time
from gnome.environment import GriddedProp, GridVectorProp
from gnome.environment.ts_property import TimeSeriesProp, TSVectorProp
from gnome.environment.environment_objects import (VelocityGrid,
                                                   VelocityTS,
                                                   Bathymetry,
                                                   S_Depth_T1)
from gnome.environment.grid import PyGrid, PyGrid_S, PyGrid_U
from gnome.utilities.remote_data import get_datafile
from unit_conversion import NotSupportedUnitError
import netCDF4 as nc
import unit_conversion
import pprint as pp

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, 'sample_data'))
from gen_analytical_datasets import gen_all


'''
Need to hook this up to existing test data infrastructure
'''

s_data = os.path.join(base_dir, 'sample_data')
gen_all(path=s_data)

sinusoid = os.path.join(s_data, 'staggered_sine_channel.nc')
sinusoid = nc.Dataset(sinusoid)

circular_3D = os.path.join(s_data, '3D_circular.nc')
circular_3D = nc.Dataset(circular_3D)

tri_ring = os.path.join(s_data, 'tri_ring.nc')
tri_ring = nc.Dataset(tri_ring)


class TestTime:
    time_var = circular_3D['time']
    time_arr = nc.num2date(time_var[:], units=time_var.units)

    def test_construction(self):

        t1 = Time(TestTime.time_var)
        assert all(TestTime.time_arr == t1.time)

        t2 = Time(TestTime.time_arr)
        assert all(TestTime.time_arr == t2.time)

        t = Time(TestTime.time_var, tz_offset=dt.timedelta(hours=1))
        print TestTime.time_arr
        print t.time
        print TestTime.time_arr[0] + dt.timedelta(hours=1)
        assert t.time[0] == (TestTime.time_arr[0] + dt.timedelta(hours=1))

        t = Time(TestTime.time_arr.copy(), tz_offset=dt.timedelta(hours=1))
        assert t.time[0] == TestTime.time_arr[0] + dt.timedelta(hours=1)

        diff = t.time[1] - t.time[0]
        now = dt.datetime.now()
        t = Time(TestTime.time_arr.copy(), origin=now)
        assert t.time[0] == now
        assert t.time[1] - diff == t.time[0]

        t = Time(TestTime.time_arr.copy(), displacement=dt.timedelta(hours=1))
        assert t.time[0] == TestTime.time_arr[0] + dt.timedelta(hours=1)

    def test_save_load(self):
        t1 = Time(TestTime.time_var)
        fn = 'time.txt'
        t1._write_time_to_file('time.txt')
        t2 = Time.from_file(fn)
#         pytest.set_trace()
        assert all(t1.time == t2.time)
        os.remove(fn)

    def test_extrapolation(self):
        ts = Time(TestTime.time_var)
        before = TestTime.time_arr[0] - dt.timedelta(hours=1)
        after = TestTime.time_arr[-1] + dt.timedelta(hours=1)
        assert ts.index_of(before, True) == 0
        assert ts.index_of(after, True) == 11
        assert ts.index_of(ts.time[-1], True) == 10
        assert ts.index_of(ts.time[0], True) == 0
        with pytest.raises(ValueError):
            ts.index_of(before, False)
        with pytest.raises(ValueError):
            ts.index_of(after, False)
        assert ts.index_of(ts.time[-1], True) == 10
        assert ts.index_of(ts.time[0], True) == 0

    @pytest.mark.parametrize('_json_', ['save', 'webapi'])
    def test_serialization(self, _json_):
        ts = Time(TestTime.time_var)
        ser = ts.serialize(_json_)
        if _json_ == 'webapi':
            deser = Time.deserialize(ser)
            t2 = Time.new_from_dict(deser)
            assert all(ts.data == t2.data)
            assert 'data' in ser
        else:
            assert 'data' in ser


class TestS_Depth_T1:

    def test_construction(self):

        test_grid = PyGrid_S(node_lon=np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
                            node_lat=np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))

        u = np.zeros((3, 4, 4), dtype=np.float64)
        u[0, :, :] = 0
        u[1, :, :] = 1
        u[2, :, :] = 2

        w = np.zeros((4, 4, 4), dtype=np.float64)
        w[0, :, :] = 0
        w[1, :, :] = 1
        w[2, :, :] = 2
        w[3, :, :] = 3

        bathy_data = -np.array([[1, 1, 1, 1],
                               [1, 2, 2, 1],
                               [1, 2, 2, 1],
                               [1, 1, 1, 1]], dtype=np.float64)

        Cs_w = np.array([1.0, 0.6667, 0.3333, 0.0])
        s_w = np.array([1.0, 0.6667, 0.3333, 0.0])
        Cs_r = np.array([0.8333, 0.5, 0.1667])
        s_rho = np.array([0.8333, 0.5, 0.1667])
        hc = np.array([1])

        b = Bathymetry(name='bathymetry', data=bathy_data, grid=test_grid, time=None)

        dep = S_Depth_T1(bathymetry=b, terms=dict(zip(S_Depth_T1.default_terms[0], [Cs_w, s_w, hc, Cs_r, s_rho])), dataset='dummy')
        assert dep is not None

        corners = np.array([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0]], dtype=np.float64)
        res, alph = dep.interpolation_alphas(corners, w.shape)
        assert res is None  # all particles on surface
        assert alph is None  # all particles on surface
        res, alph = dep.interpolation_alphas(corners, u.shape)
        assert res is None  # all particles on surface
        assert alph is None  # all particles on surface

        pts2 = corners + (0, 0, 2)
        res = dep.interpolation_alphas(pts2, w.shape)
        assert all(res[0] == 0)  # all particles underground
        assert np.allclose(res[1], -2.0)  # all particles underground
        res = dep.interpolation_alphas(pts2, u.shape)
        assert all(res[0] == 0)  # all particles underground
        assert np.allclose(res[1], -2.0)  # all particles underground

        layers = np.array([[0.5, 0.5, .251], [1.5, 1.5, 1.0], [2.5, 2.5, 1.25]])
        res, alph = dep.interpolation_alphas(layers, w.shape)
        print res
        print alph
        assert all(res == [3, 2, 1])
        assert np.allclose(alph, np.array([0.397539, 0.5, 0]))




class TestTSprop:

    def test_construction(self):

        u = None
        v = None
        with pytest.raises(ValueError):
            # mismatched data and dates length
            dates = []
            u = TimeSeriesProp('u', 'm/s', [datetime.datetime.now(), datetime.datetime.now()], [5, ])

        u = TimeSeriesProp('u', 'm/s', [datetime.datetime.now()], [5, ])

        assert u is not None
        assert u.name == 'u'
        assert u.units == 'm/s'

        v = None
        with pytest.raises(ValueError):
            v = TimeSeriesProp('v', 'nm/hr', [datetime.datetime.now()], [5, ])

        assert v is None

        constant = TimeSeriesProp.constant('const', 'm/s', 5)
        assert constant.data[0] == 5
        assert all(constant.at(np.array((0, 0)), datetime.datetime.now()) == 5)

    def test_unit_conversion(self):
        u = TimeSeriesProp('u', 'm/s', [datetime.datetime.now()], [5, ])

        t = u.in_units('km/hr')

        assert t.data is not u.data
        assert round(t.data[0], 2) == 18.0

        with pytest.raises(unit_conversion.NotSupportedUnitError):
            # mismatched data and dates length
            t = u.in_units('nm/hr')

    def test_at(self):

        dates2 = np.array([dt.datetime(2000, 1, 1, 0),
                           dt.datetime(2000, 1, 1, 2),
                           dt.datetime(2000, 1, 1, 4),
                           dt.datetime(2000, 1, 1, 6),
                           dt.datetime(2000, 1, 1, 8), ])
        u_data = np.array([2., 4., 6., 8., 10.])
        u = TimeSeriesProp(name='u', units='m/s', time=dates2, data=u_data)

        corners = np.array(((1, 1), (2, 2)))
        t1 = dt.datetime(1999, 12, 31, 23)
        t2 = dt.datetime(2000, 1, 1, 0)
        t3 = dt.datetime(2000, 1, 1, 1)
        t4 = dt.datetime(2000, 1, 1, 8)
        t5 = dt.datetime(2000, 1, 1, 9)

        # No extrapolation. out of bounds time should fail
        with pytest.raises(ValueError):
            u.at(corners, t1)
        assert (u.at(corners, t2) == np.array([2])).all()
        assert (u.at(corners, t3) == np.array([3])).all()
        assert (u.at(corners, t4) == np.array([10])).all()
        with pytest.raises(ValueError):
            u.at(corners, t5)

        # turn extrapolation on
        assert (u.at(corners, t1, extrapolate=True) == np.array([2])).all()
        assert (u.at(corners, t5, extrapolate=True) == np.array([10])).all()

# class TestTSVectorProp:
#
#     def test_construction(self, u, v):
#         vp = None
#         vp = TSVectorProp(name='vp', units='m/s', time=dates2, variables=[u_data, v_data])
#         pytest.set_trace()
#         assert vp.variables[0].data == u_data
#
#         # 3 components
#         vp = TSVectorProp(name='vp', units='m/s', time=dates2, variables=[u_data, v_data, u_data])
#
#         # Using TimeSeriesProp
#         vp = TSVectorProp(name='vp', variables=[u, v])
#         assert vp.time == vp.variables[0].time == vp.variables[1].time
#
#         # SHORT TIME
#         with pytest.raises(ValueError):
#             vp = TSVectorProp(name='vp', units='m/s', time=dates, variables=[u_data, v_data])
#
#         # DIFFERENT LENGTH VARS
#         with pytest.raises(ValueError):
#             vp = TSVectorProp(name='vp', units='m/s', time=dates2, variables=[s_data, v_data])
#
#         # UNSUPPORTED UNITS
#         with pytest.raises(ValueError):
#             vp = TSVectorProp(name='vp', units='km/s', time=dates2, variables=[s_data, v_data, u_data])
#
#     def test_unit_conversion(self, vp):
#         nvp = vp.in_units('km/hr')
#         assert round(nvp.variables[0].data[0], 2) == 7.2
#
#         with pytest.raises(unit_conversion.NotSupportedUnitError):
#             # mismatched data and dates length
#             nvp = vp.in_units('nm/hr')
#
#         assert nvp != vp
#         assert all(nvp.variables[0].data != vp.variables[0].data)

#     def test_set_variables(self, vp):
#         print u_data
#         vp.variables = [u_data, v_data, u_data]
#         assert (vp._variables[0].data == u_data).all()
#
#         with pytest.raises(ValueError):
#             # mismatched data and time length
#             vp.variables = [[5], [6], [7]]
#
#     def test_set_attr(self, vp):
#
#         # mismatched data and time length
#         with pytest.raises(ValueError):
#             vp.set_attr(time=dates2, variables=[s_data, s_data])
#
#         vp.set_attr(name='vp1')
#         assert vp.name == 'vp1'
#
#         with pytest.raises(ValueError):
#             vp.set_attr(time=dates, variables=[u_data, v_data])
#
#         vp.set_attr(time=dates, variables=[s_data, s_data])
#         assert vp.variables[0].data[0] == 20
#
#         vp.set_attr(variables=[[50, 60, 70], s_data])
#         assert vp.variables[0].data[0] == 50
#
#         vp.set_attr(time=[datetime.datetime(2000, 1, 3, 1),
#                          datetime.datetime(2000, 1, 3, 2),
#                          datetime.datetime(2000, 1, 3, 3)])
#
#         vp.set_attr(units='km/hr')
#
#         assert vp.units == 'km/hr'
#
#         with pytest.raises(ValueError):
#             vp.set_attr(units='nm/hr')
#
#     def test_at(self, vp):
#         corners = np.array(((1, 1, 0), (2, 2, 0)))
#         t1 = dt.datetime(1999, 12, 31, 23)
#         t2 = dt.datetime(2000, 1, 1, 0)
#         t3 = dt.datetime(2000, 1, 1, 1)
#         t4 = dt.datetime(2000, 1, 1, 8)
#         t5 = dt.datetime(2000, 1, 1, 9)
#
#         # No extrapolation. out of bounds time should fail
#         with pytest.raises(ValueError):
#             vp.at(corners, t1)
#
#         print vp.name
#         assert (vp.at(corners, t2) == np.array([2, 5])).all()
#         assert (vp.at(corners, t3) == np.array([3, 6])).all()
#         assert (vp.at(corners, t4) == np.array([10, 13])).all()
#         with pytest.raises(ValueError):
#             vp.at(corners, t5)
#
#         # turn extrapolation on
#         assert (vp.at(corners, t1, extrapolate=True) == np.array([2, 5])).all()
#         assert (vp.at(corners, t5, extrapolate=True) == np.array([10, 13])).all()

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


class TestGriddedProp:


    def test_construction(self):

        data = sinusoid['u'][:]
        grid = PyGrid.from_netCDF(dataset=sinusoid)
        time = None

        u = GriddedProp(name='u',
                        units='m/s',
                        data=data,
                        grid=grid,
                        time=time,
                        data_file='staggered_sine_channel.nc',
                        grid_file='staggered_sine_channel.nc')

        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        k = GriddedProp.from_netCDF(filename=curr_file, varname='u', name='u')
        assert k.name == u.name
        assert k.units == 'm/s'
        # fixme: this was failing
        # assert k.time == u.time
        assert k.data[0, 0] == u.data[0, 0]

    def test_at(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        u = GriddedProp.from_netCDF(filename=curr_file, varname='u_rho')
        v = GriddedProp.from_netCDF(filename=curr_file, varname='v_rho')

        points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
        time = datetime.datetime.now()

        assert all(u.at(points, time) == [1, 1, 1])
        print np.cos(points[:, 0] / 2) / 2
        assert all(np.isclose(v.at(points, time), np.cos(points[:, 0] / 2) / 2))

    def test_time_offset(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        now = dt.datetime.now()
        u = GriddedProp.from_netCDF(filename=curr_file, varname='u_rho', time_origin=now)
        v = GriddedProp.from_netCDF(filename=curr_file, varname='v_rho')
        assert all(u.time.data > v.time.data)

class TestGridVectorProp:

    def test_construction(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        u = GriddedProp.from_netCDF(filename=curr_file, varname='u_rho')
        v = GriddedProp.from_netCDF(filename=curr_file, varname='v_rho')
        gvp = GridVectorProp(name='velocity', units='m/s', time=u.time, variables=[u, v])
        assert gvp.name == 'velocity'
        assert gvp.units == 'm/s'
        assert gvp.varnames[0] == 'u_rho'
#         pytest.set_trace()

    def test_at(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        gvp = GridVectorProp.from_netCDF(filename=curr_file,
                                         varnames=['u_rho', 'v_rho'])
        points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
        time = datetime.datetime.now()

        assert all(np.isclose(gvp.at(points, time)[:, 1], np.cos(points[:, 0] / 2) / 2))

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

if __name__ == "__main__":
    pass
