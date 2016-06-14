#!/usr/bin/env python

"""
some code to profile loading a raster map from a BNA

too slow!
"""

import time


# a fairly big map
map_filename = "../../scripts/script_chesapeake_bay/ChesapeakeBay.bna"

from gnome.map import MapFromBNA

# make a map:

start = time.clock()
map = MapFromBNA(map_filename)
print "it took %s seconds"%(time.clock() - start)

