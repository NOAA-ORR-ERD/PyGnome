#!/usr/bin/env python

"""
A script to generate a "poly" file for the triangle program from a verdat.

NOTE: This creates  Poly and Node  files that are indexed from 0 

"""


import sys
from hazpy.file_tools import haz_files, triangle_files

infilename = sys.argv[1]
outfilename1 = ".".join(infilename.split(".")[:-1]+["node"])
outfilename2 = ".".join(infilename.split(".")[:-1]+["poly"])
print "infile:", infilename
print "outfiles:", outfilename1, outfilename2


(PointData, Boundaries) = haz_files.ReadVerdatFile(infilename)
(PointData, Segs, Holes) = triangle_files.Verdat2Poly(PointData, Boundaries)

triangle_files.WriteNodeFile(outfilename1,PointData[:,:2],PointData[:,2])
triangle_files.WritePolyFile(outfilename2, Segs, Holes)


