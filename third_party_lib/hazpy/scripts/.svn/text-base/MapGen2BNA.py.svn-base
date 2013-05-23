#!/usr/bin/env python

"""

NOTE: this is a start, but the Coastline extractor actually only provides the
coastline itself, not polygons surrounding the land -- so more needs to be done.

Perhaps code could be borrowed from GOODS?

MapGen2BNA

Simple script to Reads a "MapGen" style Ascii file for coastlines
(as produced by the NOAA "shoreline extractor")
and writes a GNOME-compatible BNA file.

USAGE:

python MapGen2BNA.py infilename [outfilename]

if outfilename is not specified, the output will be written to a file with
the same name as the infilename, with the extension replaced by "BNA".

"""
USAGE = """ python MapGen2BNA.py infilename [outfilename]"""

import sys

try: 
    infilename = sys.argv[1]
    try:
        outfilename = sys.argv[2]
    except IndexError: # no outfile specified
        outfilename = infilename.rsplit(".", 1)[0] + ".bna"
except:
    print USAGE
    raise

infile = file(infilename, 'U')
outfile = file(outfilename, 'w')


poly = []
count = 1
for line in infile:
    if line.strip() == "# -b": # start of a new polygon
        if poly: # write out the previous polygon
            # BNA has the last point being the same as the first
            poly.append(poly[0])
            outfile.write('"%s","%s", %i\n'%(count, 1, len(poly)) )
            for point in poly:
                outfile.write('%.8f, %.8f \n'%(point[0], point[1]) )
            
        poly = []
        count += 1
        continue
    else: # read the point in 
        poly.append(tuple([float(i) for i in line.split()]))

infile.close()
outfile.close()

