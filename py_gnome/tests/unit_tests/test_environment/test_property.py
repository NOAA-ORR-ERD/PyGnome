import os
import pytest
import datetime as dt
import numpy as np
import pysgrid
import datetime
from gnome.environment.property import TimeSeriesProp, GriddedProp, Time, TSVectorProp
from gnome.utilities.remote_data import get_datafile
import netCDF4 as nc
import unit_conversion

base_dir = os.path.dirname(__file__)
'''
Need to hook this up to existing test data infrastructure
'''
s_data = os.path.join(base_dir, 'sample_data')
curr_dir = os.path.join(s_data, 'currents')
curr_file = get_datafile(os.path.join(curr_dir,'tbofs_example.nc'))
dataset = nc.Dataset(curr_file)
node_lon = dataset['lonc']
node_lat = dataset['latc']
grid_u = dataset['water_u']
grid_v = dataset['water_v']
grid_time = dataset['time']
test_grid = pysgrid.SGrid(node_lon=node_lon,
                          node_lat=node_lat)

dates = np.array([dt.datetime(2000, 1, 1, 0),
                  dt.datetime(2000, 1, 1, 2),
                  dt.datetime(2000, 1, 1, 4)])
dates2 = np.array([dt.datetime(2000, 1, 1, 0),
                   dt.datetime(2000, 1, 1, 2),
                   dt.datetime(2000, 1, 1, 4),
                   dt.datetime(2000, 1, 1, 6),
                   dt.datetime(2000, 1, 1, 8), ])
uv_units = 'm/s'
u_data = np.array([2, 4, 6, 8, 10])
v_data = np.array([5, 7, 9, 11, 13])

s_data = np.array([20,30,40])


@pytest.fixture()
def u():
    return TimeSeriesProp(name='u', units='m/s', time=dates2, data=u_data)

@pytest.fixture()
def v():
    return TimeSeriesProp(name='v', units='m/s', time=dates2, data=v_data)

@pytest.fixture()
def vp():
    return TSVectorProp(name='vp', units='m/s', time=dates2, variables=[u_data,v_data], extrapolate=False)

class TestTSprop:

    def test_construction(self):

        u = None
        v = None
        with pytest.raises(ValueError):
            # mismatched data and dates length
            u = TimeSeriesProp('u', 'm/s', dates, u_data)

        assert u is None

        u = TimeSeriesProp('u', 'm/s', dates2, u_data)

        assert u is not None
        assert u.name == 'u'
        assert u.units == 'm/s'
        assert u.time == Time(dates2)
        assert (u.data == u_data).all()

        v = None
        with pytest.raises(ValueError):
            v = TimeSeriesProp('v', 'nm/hr', dates2, v_data)

        assert v is None

    def test_unit_conversion(self,u):

        t = u.in_units('km/hr')

        assert t.data is not v_data
        assert round(t.data[0],2) == 7.2

        with pytest.raises(unit_conversion.UnitConversionError):
            # mismatched data and dates length
            t = u.in_units('nm/hr')

    def test_set_data(self,u):
        u.data = v_data
        assert (u.data == v_data).all()

        with pytest.raises(ValueError):
            # mismatched data and time length
            u.data = [5,6,7]

    def test_set_time(self,u):
        with pytest.raises(ValueError):
            # mismatched data and time length
            u.time = dates

        u.time = dates2
        assert u.time == Time(dates2)

    def test_set_ts(self,u):

        #mismatched data and time length
        with pytest.raises(ValueError):
            u.set_ts(time = dates2, data=s_data)

        u.set_ts(name = 'v')
        assert u.name == 'v'

        with pytest.raises(ValueError):
            u.set_ts(time = dates,data=u_data)

        u.set_ts(time=dates, data=s_data)
        assert u.data[0] == 20

        u.set_ts(data = [50,60,70])
        assert u.data[0] == 50

        u.set_ts(time = [datetime.datetime(2000,1,3,1),
                         datetime.datetime(2000,1,3,2),
                         datetime.datetime(2000,1,3,3)])

        u.set_ts(extrapolate=True, units='km/hr')

        assert u.extrapolate == True
        assert u.units == 'km/hr'

        with pytest.raises(ValueError):
            u.set_ts(units='nm/hr')

    def test_at(self,u):
        pts = np.array(((1,1), (2,2)))
        t1 = dt.datetime(1999, 12,31,23)
        t2 = dt.datetime(2000, 1, 1, 0)
        t3 = dt.datetime(2000, 1, 1, 1)
        t4 = dt.datetime(2000,1, 1, 8)
        t5 = dt.datetime(2000,1, 1, 9)

        #No extrapolation. out of bounds time should fail
        with pytest.raises(ValueError):
            u.at(pts, t1)
        assert (u.at(pts, t2) == np.array([2])).all()
        assert (u.at(pts, t3) == np.array([3])).all()
        assert (u.at(pts, t4) == np.array([10])).all()
        with pytest.raises(ValueError):
            u.at(pts, t5)

        #turn extrapolation on
        u.set_ts(extrapolate=True)
        print u.time
        print u.time.time
        print u.time.extrapolate
        assert (u.at(pts, t1) == np.array([2])).all()
        assert (u.at(pts, t5) == np.array([10])).all()

class TestTSVectorProp:

    def test_construction(self):
        vp = None
        vp = TSVectorProp(name='vp', units='m/s', time=dates2, variables=[u_data,v_data], extrapolate=False)

        assert (vp.variables[0] == u_data).all()

        vp = TSVectorProp(name='vp', units='m/s', time=dates2, variables=[u_data,v_data, u_data], extrapolate=False)


        with pytest.raises(ValueError):
            vp = TSVectorProp(name='vp', units='m/s', time=dates, variables=[u_data,v_data], extrapolate=False)
        with pytest.raises(ValueError):
            vp = TSVectorProp(name='vp', units='m/s', time=dates2, variables=[s_data,v_data], extrapolate=False)
        with pytest.raises(ValueError):
            vp = TSVectorProp(name='vp', units='m/s', time=dates, variables=[s_data,v_data, u_data], extrapolate=False)

    def test_unit_conversion(self, vp):
        nvp = vp.in_units('km/hr')
        assert round(nvp.variables[0][0],2) == 7.2

        with pytest.raises(unit_conversion.UnitConversionError):
            # mismatched data and dates length
            nvp = vp.in_units('nm/hr')

        assert nvp != vp
        assert (nvp.variables[0] != vp.variables[0]).all()

    def test_set_variables(self,vp):
        print u_data
        vp.variables = [u_data,v_data,u_data]
        assert (vp._variables[0].data == u_data).all()

        with pytest.raises(ValueError):
            # mismatched data and time length
            vp.variables = [[5],[6],[7]]

    def test_set_time(self,vp):
        with pytest.raises(ValueError):
            # mismatched data and time length
            vp.time = dates

        vp.time = dates2
        assert vp.time == Time(dates2)

    def test_set_ts(self,vp):

        #mismatched data and time length
        with pytest.raises(ValueError):
            vp.set_ts(time = dates2, variables=[s_data, s_data])

        vp.set_ts(name = 'vp1')
        assert vp.name == 'vp1'

        with pytest.raises(ValueError):
            vp.set_ts(time = dates, variables=[u_data,v_data])

        vp.set_ts(time=dates, variables=[s_data, s_data])
        print vp.variables[0][0]
        assert vp.variables[0][0] == 20

        vp.set_ts(variables = [[50,60,70],s_data])
        assert vp.variables[0][0] == 50

        vp.set_ts(time = [datetime.datetime(2000,1,3,1),
                         datetime.datetime(2000,1,3,2),
                         datetime.datetime(2000,1,3,3)])

        vp.set_ts(extrapolate=True, units='km/hr')

        assert vp.extrapolate == True
        assert vp.units == 'km/hr'

        with pytest.raises(ValueError):
            vp.set_ts(units='nm/hr')

    def test_at(self,vp):
        pts = np.array(((1,1), (2,2)))
        t1 = dt.datetime(1999, 12,31,23)
        t2 = dt.datetime(2000, 1, 1, 0)
        t3 = dt.datetime(2000, 1, 1, 1)
        t4 = dt.datetime(2000,1, 1, 8)
        t5 = dt.datetime(2000,1, 1, 9)

        #No extrapolation. out of bounds time should fail
        with pytest.raises(ValueError):
            vp.at(pts, t1)

        print vp.name
        assert (vp.at(pts, t2) == np.array([2,5])).all()
        assert (vp.at(pts, t3) == np.array([3,6])).all()
        assert (vp.at(pts, t4) == np.array([10,13])).all()
        with pytest.raises(ValueError):
            vp.at(pts, t5)

        #turn extrapolation on
        vp.set_ts(extrapolate=True)
        print vp.time
        print vp.time.time
        print vp.time.extrapolate
        assert (vp.at(pts, t1) == np.array([2,5])).all()
        assert (vp.at(pts, t5) == np.array([10,13])).all()



def test_gridprop_construction():
    u = GriddedProp(name='u',
                    units='m/s',
                    data=grid_u,
                    grid=test_grid,
                    time=grid_time,
                    data_file='tbofs_example.nc',
                    grid_file='tbofs_example.nc')
#     print u.at(np.array((-82.75, 27.5)), datetime.datetime(2015,11,30,19,30))
#     print u.at(np.array((-82.75, 27.5)), datetime.datetime(2015,11,30,19,30), units='km/hr')
    pass

if __name__ == "__main__":
    test_tsprop_construction()
    test_tsprop_unit_conversion()
    test_tsprop_set_ts()
    test_gridprop_construction()
