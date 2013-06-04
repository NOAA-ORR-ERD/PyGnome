#!/usr/bin/env python

"""
Point in Polygon code.
"""

import numpy as np


def point_in_poly( pgon, (tx, ty) ):
    """
    Point in polygon test using the "Crossings" algorithm.
    
    point_in_polygon(pgon, (tx, ty))
    
    :param pgon: a NX2 numpy array of points (or something that can be turned into one)
    
    :param (tx, ty):  the coords of the point to check
    
    points on lines on the right and top are in the polygon,
    points on the lines on the left and bottom are out of the polygon
      - so a point should be in one and only one of two adjoining polygons.
    
        
    translated from C code from "Graphics Gems"
    
    from: http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
    
    "For most of the algorithms above there is a pathological case if the point
     being queried lies exactly on a vertex. The easiest way to cope with this
     is to test that as a separate process and make your own decision as to 
     whether you want to consider them inside or outside." 
    
    I've added that test -- and calling a point on the vertex inside.
    
    This could be compiled with cython nicely
    
    Note: This code will ignore the last point if the first and last points
          are the same.
    
    """
    # make it a numpy array if it isn't one
    pgon = np.asarray(pgon).reshape((-1, 2))
    
    if pgon[0,0] == pgon[-1,0] and pgon[0,1] == pgon[-1,1]:
        # first and last points are the same, so ignore the last point
        numverts = len(pgon) - 1
    else:
        numverts = len(pgon)
    vtx0 = pgon[numverts-1] # the last vertex
    # get test bit for above/below X axis 
    yflag0 = ( vtx0[1] >= ty )
    vtx1 = pgon[0]
    inside_flag = False
    for j in xrange(numverts):
        vtx1 = pgon[j]
        # check if the point is exactly on the vertex:
        if vtx1[0] == tx and vtx1[1] == ty:
            return True
        yflag1 = ( vtx1[1] >= ty ) 
        #check if endpoints straddle (are on opposite sides) of X axis
        #(i.e. the Y's differ); if so, +X ray could intersect this edge.
        if ( yflag0 != yflag1 ):
            xflag0 = ( vtx0[0] >= tx )
            # check if endpoints are on same side of the Y axis (i.e. X's
            # are the same); if so, it's easy to test if edge hits or misses.
            if ( xflag0 == ( vtx1[0] >= tx ) ) :
                # if edge's X values both right of the point, must hit
                if ( xflag0 ):
                    inside_flag = not inside_flag 
            else:
                # compute intersection of pgon segment with +X ray, note
                # if >= point's X; if so, the ray hits it.
                if ( (vtx1[0] - (vtx1[1]-ty)* ( vtx0[0]-vtx1[0])/(vtx0[1]-vtx1[1])) >= tx ):
                    inside_flag = not inside_flag 
        # move to next pair of vertices, retaining info as possible
        yflag0 = yflag1
        vtx0 = vtx1
    return inside_flag

def point_in_poly2(pgon, (testx, testy)):
## This is a C version from:
## http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
## run time is about the same -- I wonder if it's more optimizable in C?    
## it handles on-line differently than above -- though is it said to be consistant on the web page.
## not sure how it handles points on a vertex
##    
#int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
#    #nvert 	Number of vertices in the polygon. Whether to repeat the first vertex at the end is discussed below.
#    #vertx, verty 	Arrays containing the x- and y-coordinates of the polygon's vertices.
#    #testx, testy	X- and y-coordinate of the test point. 
#{
#  int i, j, c = 0;
#  for (i = 0, j = nvert-1; i < nvert; j = i++) {
#    if ( ((verty[i]>testy) != (verty[j]>testy)) &&
#         (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
#    c = !c;
#  }
#  return c;
#}

## another option here: http://softsurfer.com/Archive/algorithm_0103/algorithm_0103.htm
    
    # make it a numpy array if it isn't one
    pgon = np.asarray(pgon).reshape((-1, 2))
    vertx = pgon[:, 0]
    verty = pgon[:, 1]
    
    if pgon[0,0] == pgon[-1,0] and pgon[0,1] == pgon[-1,1]:
        # first and last points are the same, so ignore the last point
        numvert = len(pgon) - 1
    else:
        numvert = len(pgon)

    i = j = c = 0
    j = numvert - 1
    while i < numvert:
        if ( ((verty[i]>testy) != (verty[j]>testy)) and
             (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ):
            c =  not c;
        j = i
        i +=1

    return c

def point_in_tri( tri, (x, y) ):
    """
    
    This method is for only triangles -- but it's  method that works for all
     convex polygons it's 30% or so faster than point_in_poly above.
    
    Note: this method has been adapted so that points on the lines will be
          considered in the triangle, regarless of if it's wound CW or CCW
    
    from: http://paulbourke.net/geometry/polygonmesh/
    
    There are other solutions to this problem for polygons with special attributes.
    If the polygon is convex then one can consider the polygon as a "path"
    from the first vertex. A point is on the interior of this polygons if it
    is always on the same side of all the line segments making up the path.
    
    Given a line segment between P0 (x0,y0) and P1 (x1,y1), another point P (x,y)
    has the following relationship to the line segment:

    Compute:    (y - y0) (x1 - x0) - (x - x0) (y1 - y0)
    
    if it is less than 0 then P is to the right of the line segment, if greater
    than 0 it is to the left, if equal to 0 then it lies on the line segment. 
    """
    print "point_in_tri called"
    tri = np.asarray(tri, dtype=np.float64).reshape((3, 2))
    
    def sign(x):
        """
        returns +1 if x > 0
                -1 if x < 0
                 0 if x == 0
        (shouldn't this be built in?)
        """
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    x0 = tri[0,0] #index into the array once...
    y0 = tri[0,1]
    x1 = tri[1,0]
    y1 = tri[1,1]
    x2 = tri[2,0]
    y2 = tri[2,1]
    # first check the first line segment -- left or right:
    #check for CW or CCW:
    first_CW =  ( (y2 - y0) * (x1 - x0) - (x2 - x0)*(y1 - y0) ) <= 0 # if it's equal, the triangle is degenerate...
    print "is CW:", CW
    if first_CW == 0:
        raise ValueError("triangle is degenerate")
    first_sign = sign( ( (y - y0) * (x1 - x0) - (x - x0)*(y1 - y0) ) )
#    if first_side == 0.0:
#        return True  # on the line is considered in
#    else:
#        sign = first_side > 0.0
#    if sign ( ( (y - y1) * (x2 - x1) - (x - x1)*(y2 - y1) ) ) == first_sign:
#        side = 1.0
#    if side == 0.0:
#        return True  # on the line is considered in
#    elif (side > 0.0) is not sign:
#        return False
#    side = ( (y - y2)*(x0 - x2) - (x - x2)*(y0 - y2) )
#    if side == 0.0:
#        return True  # on the line is considered in
#    elif (side > 0.0) is not sign:
#        return False
#
    return True
    




