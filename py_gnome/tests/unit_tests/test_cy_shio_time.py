from gnome.cy_gnome import cy_shio_time
from datetime import datetime, timedelta
from gnome.utilities import time_utils

file=r"SampleData/long_island_sound/CLISShio.txt"

s = cy_shio_time.CyShioTime(file)

info = s.get_info()

dt = [datetime.now() + timedelta(hours=(i*4)) for i in range(5)]

dtsec = time_utils.date_to_sec(dt)
#print "get_time_value"
#s.get_time_value(dtsec)

#print "get_height"
#s.get_height(dtsec)
