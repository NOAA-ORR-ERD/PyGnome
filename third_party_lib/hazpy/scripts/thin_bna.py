#!/usr/bin/env python

"""
script to "thin" a BNA file, using the "re-scale" approach

NOTE: there is a version in py_gnome:
  gnome.utilities.geometry.polygons
"""

import sys

import numpy as np

from hazpy.file_tools import haz_files

USAGE = """Usage:  thin_bna scale filename

scale: a scale, in meters,  only accurate at the equator
filename: name of the input bna file
    
the output will be a file in the same place as the input file,
with "thinned" added to the name
"""

try:
    scale = sys.argv[1]
    infilename = sys.argv[2]
    scale = float(scale)
except (IndexError, ValueError):
    print USAGE
    sys.exit()

outfilename = infilename.rsplit(".")
outfilename = ".".join(outfilename[:-1]+['thinned-%s'%scale, outfilename[-1]]) 

print "Reading input BNA"
input = haz_files.ReadBNA(infilename, "PolygonSet")

print "number of input polys: %i"% len(input)
print "total number of input points: %i "%input.total_num_points

# convert scale in lat degrees:
lat_scale = (scale * 9e-06)
##fixme: add a scaling to the longitude for the midpoint of the BBox
lon_scale = lat_scale

scale = np.array((lon_scale, lat_scale), dtype=np.float64)

#Scale the bna:

def scaling_fun(arr):
    return np.round(arr / scale) * scale

    
def remove_dups(scaled, orig):
    """
    returns a new Polygonset instance, with points removed from orig that are
    duplicate in the scaled polygons

    If a polygon is reduced to a single point, it is removed.
    """
    from hazpy.geometry import polygons
    new = polygons.PolygonSet()
    for i in xrange(len(scaled)):
        sc_poly = scaled[i]
        orig_poly = orig[i]
        last_point = np.asarray(sc_poly[0])
        thinned = [orig_poly[0],]
        for j in xrange(len(sc_poly)):
            point = np.asarray(sc_poly[j])
            if not np.array_equal(point, last_point):
                thinned.append(orig_poly[j])
            last_point = point
        if len(thinned) > 1:
            #points = np.array(thinned)
            new.append(polygons.Polygon(thinned, metadata=orig_poly.metadata))
    print "new set: number of polys:", len(new)
    print "new set: number of points:", new.total_num_points

    return new

            
#        points, metadata=None
#        new.append(poly[0])

orig = input.Copy()        
input.TransformData(scaling_fun)
output = remove_dups(input, orig)

print "writing new BNA"
haz_files.WriteBNA(outfilename, output)


