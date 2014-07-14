'''
test cython wrappers around grid objects
'''
import os
from datetime import datetime

import netCDF4 as nc
import numpy as np

from gnome.cy_gnome.cy_grid_rect import CyTimeGridWindRect
from gnome.cy_gnome.cy_grid_curv import CyTimeGridWindCurv
from gnome.utilities.time_utils import date_to_sec
from gnome.basic_types import velocity_rec
from gnome.utilities.remote_data import get_datafile

wind_base = os.path.join(os.path.dirname(__file__), 'sample_data/winds/')
f = get_datafile(os.path.join(wind_base, 'test_wind.cdf'))
f2 = get_datafile(os.path.join(wind_base, 'WindSpeedDirSubset.nc'))
t2 = get_datafile(os.path.join(wind_base, 'WindSpeedDirSubsetTop.dat'))


def test_grid_wind_rect():
    '''
    check CyTimeGridWindRect correctly parses data
    '''
    idx = 5
    with nc.Dataset(f) as data:
        time = nc.num2date(data.variables['time'][idx],
                           units=data.variables['time'].units)
        long_lat = (data.variables['lon'][idx],
                    data.variables['lat'][idx])
        exp_vel = (data.variables['air_u'][idx][idx][idx],
                   data.variables['air_v'][idx][idx][idx])

    c = CyTimeGridWindRect(f)
    time = date_to_sec(time)
    vel = c.get_value(time, long_lat)
    print "\nRect grid - vel: {0}".format(vel)
    print "Rect grid - expected_vel: {0}".format(exp_vel)
    assert (vel.item() == exp_vel)


def test_grid_wind_curv():
    # curvlinear grid
    curv = CyTimeGridWindCurv(f2, t2)
    time = date_to_sec(datetime(2006, 3, 31, 21))
    vel = curv.get_value(time, (-122.934656, 38.27594))
    print "Curv grid - vel: {0}\n".format(vel)
    assert vel.item() != 0
