'''
test cython wrappers around grid objects
'''
import os
from datetime import datetime

from gnome.cy_gnome.cy_grid_rect import CyTimeGridWindRect
from gnome.cy_gnome.cy_grid_curv import CyTimeGridWindCurv
from gnome.utilities.time_utils import date_to_sec

wind_base = 'sample_data/winds/'
f = os.path.join(wind_base, 'test_wind.cdf')
f2 = os.path.join(wind_base, 'WindSpeedDirSubset.nc')
t2 = os.path.join(wind_base, 'WindSpeedDirSubsetTop.dat')


def test_grid_wind_rect():
    c = CyTimeGridWindRect(f)
    time = date_to_sec(datetime(1999, 11, 29, 21))
    vel = c.get_value(time, (3.104588, 52.016468))
    print "\nRect grid - vel: {0}".format(vel)
    for key in vel:
        assert vel[key] != 0


def test_grid_wind_curv():
    # curvlinear grid
    curv = CyTimeGridWindCurv(f2, t2)
    time = date_to_sec(datetime(2006, 3, 31, 21))
    vel = curv.get_value(time, (-122.934656, 38.27594))
    print "Curv grid - vel: {0}\n".format(vel)
    for key in vel:
        assert vel[key] != 0
