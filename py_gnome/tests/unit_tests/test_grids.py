'''
test grid objects
'''





from datetime import datetime

import netCDF4 as nc

from gnome.environment import Grid

from gnome.utilities.time_utils import date_to_sec
from .conftest import testdata

wind_file = testdata['c_GridWindMover']['wind_curv']
topology_file = testdata['c_GridWindMover']['top_curv']


def test_grid_wind_rect():
    '''
    check TimeGridWindRect correctly parses data
    '''
    idx = 5
    file_ = testdata['c_GridWindMover']['wind_rect']
    with nc.Dataset(file_) as data:
        time = nc.num2date(data.variables['time'][idx],
                           units=data.variables['time'].units)
        long_lat = (data.variables['lon'][idx],
                    data.variables['lat'][idx])
        exp_vel = (data.variables['air_u'][idx][idx][idx],
                   data.variables['air_v'][idx][idx][idx])

    rect = Grid(file_)
    time = date_to_sec(time)
    vel = rect.get_value(time, long_lat)
    print("\nRect grid - vel: {0}".format(vel))
    print("Rect grid - expected_vel: {0}".format(exp_vel))
    assert (vel.item() == exp_vel)


def test_grid_wind_curv():
    # curvlinear grid
    curv = Grid(wind_file, topology_file, grid_type=2)
    time = date_to_sec(datetime(2006, 3, 31, 21))
    vel = curv.get_value(time, (-122.934656, 38.27594))
    print("Curv grid - vel: {0}\n".format(vel))
    assert vel.item() != 0

