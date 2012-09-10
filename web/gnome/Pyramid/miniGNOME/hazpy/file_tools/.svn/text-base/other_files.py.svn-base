#!/usr/bin/env python

##fixme: untested since the reorginization

"""
File reading tools for assorted non-hazmat file types
"""


def ReadADCIRC(filename):
    file = open(filename,'rU')
    
    file.readline()
    # read first line:
    line = file.readline().strip().split()
    NTri, NPoints = map(int, line)
    
    Coords = zeros( (NPoints,3), Float )
    
    for n in xrange(NPoints):
        line = map(float, file.readline().strip().split())
        Coords[n,:] = line[1:4]
        n += 1
        
    C = zeros( (NPoints,3), Float )
    
    Triangles = zeros( (NTri,3), Int )
    for n in xrange(NTri):
        line = map(int, file.readline().strip().split())
        
        Triangles[n,:] = line[2:5]
    Triangles -= 1
    
    Unknown = []
    while 1:
        line = file.readline().strip()
        if not line: break
        if len( line.split() ) > 1: print "double points:", line
        Unknown.append(int(line.split()[0]))
    Unknown = array(Unknown)
    Unknown -= 1
    
    return (Coords, Triangles, Unknown) 

