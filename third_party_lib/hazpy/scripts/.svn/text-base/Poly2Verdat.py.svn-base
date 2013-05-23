#!/usr/bin/env python

"""
A script to generate a verdat file from the poly format used by the triangle
program.

"""

import sys
from hazpy.file_tools import haz_files, triangle_files

infilename1 = sys.argv[1]
infilename2 = infilename1.rsplit(".", 1)[0]+".node"
outfilename = infilename1.rsplit(".", 1)[0] + ".verdat"

print "infile:", infilename1
print "outfile:", outfilename

(PointData, Attributes, BoundaryMarkers) = triangle_files.ReadNodeFile(infilename2)
(Segs, Holes, BoundaryMarkers) = triangle_files.ReadPolyFile(infilename1)

(VerPointData, Boundaries) = triangle_files.SortForVerdat(PointData, Segs, Holes)

haz_files.WriteVerdatFile(outfilename, VerPointData, Boundaries)

