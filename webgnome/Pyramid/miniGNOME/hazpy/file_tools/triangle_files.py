#!/usr/bin/env python

import os
import numpy as np

"""

A module that contains various functions for reading and writing
the files used by the "triangle" code.

"""

def ReadNodeFile(filename):

    """
    
    This is code to read a "*.node" file as used by the Triangle program.
    
    It does not handle comment lines very well.
    
    """
    
    file = open(filename,'rU')
    
    # read first line:
    line = file.readline().strip().split()
    NPoints, dim, NumAttributes, NumBoundaryMarkers = map(int, line)
    
    Coords = np.zeros( (NPoints,2), np.float32 )
    Attributes = np.zeros( (NPoints, NumAttributes), np.float32 )
    if NumBoundaryMarkers:
        BoundaryMarkers = np.zeros( (NPoints, ), np.uint32 )
    else:
        BoundaryMarkers = None
    n = 0
    while n < NPoints:
        line = map(float, file.readline().strip().split())
        Coords[n,:] = line[1:3]
        Attributes[n,:] = line[3:3+NumAttributes]
        if NumBoundaryMarkers:
            BoundaryMarkers[n] = line[-1]
        n += 1
        
    return (Coords, Attributes, BoundaryMarkers) 
    
def WriteNodeFile(filename,Coords,Attributes, BoundaryMarkers =  None):

    if BoundaryMarkers is None:
        BoundaryMarkers = np.ones(Attributes.shape)
        
    if len(Attributes.shape) > 1 and len(Attributes[0]) > 1:
        raise Exception, "WriteNodeFile can only handle one attribute"
    if len(BoundaryMarkers.shape) > 1 and len(BoundaryMarkers[0]) > 1:
        raise Exception, "WriteNodeFile can only handle one BoundaryMarker"
        
    file = open(filename,'w')
    print len(Coords)
    file.write("%i  %i  %i  %i\n"%(len(Coords), 2, 1, 1 ) )
    
    for i in xrange(len(Coords)):
        file.write("%i\t%f\t%f\t%i\t%i\n"%(i, Coords[i][0], Coords[i][1], Attributes[i], BoundaryMarkers[i]) )
    file.close()
    
    
def ReadPolyFile(filename):

    """
    
    This is code to read a "*.poly" file as used by the Triangle program.
    It does not handle comment lines very well.
    
    """
    
    file = open(filename,'rU')
    
    # read first line:
    line = file.readline().strip().split()
    if int(line[0]) > 0:
        raise "Can't read a poly file with the nodes built in"
    line = file.readline().strip().split()
    NSegs,  NumBoundaryMarkers = map(int, line)
    
    Segs = np.zeros( (NSegs,2), np.int32 )
    if NumBoundaryMarkers:
        BoundaryMarkers = np.zeros( (NSegs, ), np.int32 )
    else:
        BoundaryMarkers = None
    n = 0
    ## read the Segment lines
    while n < NSegs:
        line = map(float, file.readline().strip().split())
        Segs[n,:] = line[1:3]
        if NumBoundaryMarkers:
            BoundaryMarkers[n] = line[-1]
        n += 1
        
    ## read holes line:
    NumHoles = int( file.readline().strip() )
    if NumHoles:
        Holes = np.zeros( (NumHoles,2), np.float )
    else:
        Holes = None
    for n in range(NumHoles):
        line = map(float, file.readline().strip().split())
        Holes[n,:] = line[1:3]
    return (Segs, Holes, BoundaryMarkers) 
    
def WritePolyFile(filename, Segs, Holes = None, BoundaryMarkers = None):

    """
    
    WritePolyFile write a "triangle" style *.poly file. It does not , at
    the moment, handle regional attributes.
    
    """
    
    if not BoundaryMarkers:
        BoundaryMarkers = np.zeros( (len(Segs),0) )
    else:
        BoundaryMarkers.shape = (len(Segs),-1) # forces it to be rank-2

    if Holes is None:
        Holes = []  
    file = open(filename,'w')
    file.write("%i  %i  %i  %i\n"%(0, 2, 1, 1 ) ) # this version only writes a .poly file with no vertices.
    NumBoundaryMarkers = BoundaryMarkers.shape[1]
    FormatString = (NumBoundaryMarkers * "\t%i") + "\n"
    file.write("%i  %i\n"%(len(Segs), NumBoundaryMarkers ) ) # this version only writes a .poly file with no vertices.
    for i in xrange(len(Segs)):
        file.write( "%i\t%i\t%i"%(i, Segs[i][0], Segs[i][1]) )
        file.write( FormatString % tuple(BoundaryMarkers[i]) )
        
    file.write( "%i\n"%len(Holes) )
    for i in xrange(len(Holes)):
        file.write("%i\t%f\t%f\n"%(i,Holes[i,0],Holes[i,1]) )
        
    file.close()
    
    
def ReadEleFile(filename):

    """
    
    This is code to read a "*.ele" file as used by the Triangle program.
    
    This is the file that has the triangles defined
    
    It does not handle comment lines very well.
    
    """
    
    file = open(filename,'rU')
    
    # read first line:
    line = file.readline().strip().split()
    NTri,  NodesPerTriangle, NAttributes = map(int, line)
    
    Triangles = np.zeros( (NTri,3), np.uint32 )
    for n in xrange(NTri):
        line = map(int, file.readline().strip().split())
        Triangles[n,:] = line[1:4]
    return Triangles 
    
def Verdat2Poly(PointData, Boundaries):
    """
    
    This function converts Verdat style description of a domain to a
    triangle *.poly style data. This involves generating the segment
    data for the poly, and defining holes to describe the topology of
    the domain.
    
    """

    from hazpy.geometry import PinP

    # Generate Segments
    Segs = []
    firstpoint = 0
    for i in range( len(Boundaries) ):
        lastpoint = Boundaries[i]
        for i in range(firstpoint, lastpoint):
            Segs.append((i,i+1))
        Segs.append( (lastpoint, firstpoint) )
        firstpoint = lastpoint + 1
        
        ## Generate holes
        
    Segs = np.array(Segs)

    ## holes: one for each polygon
    ##  fixme: should have a way to vary delta
    ##  Note that there is no guarantee that the hole for the
    ##  bounding box will be inside the convex hull, and I think it needs
    ##  to be.

    delta = .0001
    Holes = np.zeros( (len(Boundaries),2), np.float )
    firstpoint = 0
    for i in range( len(Boundaries) ):
        lastpoint = Boundaries[i]
        # the polygon, with the first and last points the same
        poly = np.concatenate( (PointData[firstpoint:lastpoint+1,:2],PointData[firstpoint:firstpoint+1,:2]) )
        firstpoint = lastpoint+1
        for ( (x1,y1),(x2,y2) ) in zip( poly[:-1], poly[1:]) :
            ## Try each segment, exit if a hole is found.
            ##
            ## This approach finds the center of the segment, then
            ## generates a vector in the complex plane rotated 90deg to
            ## the right, of length delta, and adds it to the center point
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            z1 = complex( (xc-x1), (yc-y1) )
            theta = np.arctan2(z1.imag, z1.real)
            theta -= (np.pi/2)
            z2 = delta * np.e**(1j*theta)
            hole = (z2.real + xc, z2.imag + yc)
            if i == 0: # the first one is the boundary, the hole should be outside
                
                if not PinP.CrossingsTest( poly, hole ):
                    Holes[i] = hole
                    break
            else: # the rest are islands, the hole should be inside
                if PinP.CrossingsTest( poly, hole ):
                    Holes[i] = hole
                    break
        else:
            print Holes
            raise VerdatError, "I couldn't find a hole in polygon: %i"%i

    return (PointData, np.array(Segs), Holes)

    
class VerdatError(Exception):
    pass
    
def SortForVerdat(PointData, Segs, Holes):
    """
    
    This function takes a set of points and segments as output by
    Triangle, and sorts them so that the segments are in Verdat order.
    
    Boy, this is a pain in the Tuckus!
    
    fixme: This is not doing any checking for which is the outside boundary
           It could use the holes to figure it out.
    """
    from  hazpy.geometry.PinP import CrossingsTest as PinP
    
    Boundaries = []
    PointNums = []
    
    if len(Segs) < 3:
        raise "A verdat must have a boundary of at least three segments"
        
    Segs = Segs.flatten().tolist()
    
    StartNewBoundary = 1
    while Segs:
    ##    print PointNums
    ##    temp = np.array(Segs)
    ##    temp.shape = (-1,2)
    ##    print temp
        if StartNewBoundary: # just add the first segment
            startpoint, endpoint = Segs[:2]
            ##        print "Starting bound, adding segment: ",startpoint, endpoint
            del Segs[:2]
            PointNums.append( startpoint )
            PointNums.append( endpoint )
            BoundStart = startpoint
            StartNewBoundary = 0
        else:
            # search for a connecting segment:
            try: # this will only get run if one is found
                i = Segs.index( endpoint )
                startpoint = Segs.pop(i) # do I need this?
                if i%2: # odd number: second point in segment
                    endpoint = Segs.pop(i-1)
                else: # even number, first point in segment
                    endpoint = Segs.pop(i)
                if (endpoint == BoundStart): # This closes the boundary, don't add it
                    #print "closing the boundary, not adding:", startpoint, endpoint
                    Boundaries.append( len(PointNums) - 1 )
                    #print "Boundaries:",Boundaries
                    StartNewBoundary = 1
                else:
                    #print "adding the next segment:",startpoint, endpoint
                    PointNums.append( endpoint  )
            except ValueError: # a connecting segment was not found
                raise "All boundaries must be closed, with no duplpicates, i = %i"%i
                ##fixme: should I check orientation, etc before puttingt them in verdat order???
                # Put the points in VerDat order:
                # fixme: can this be vectorized? 
    test = []
    NewPointData = []
    AllPointNums = range( len(PointData) )
    for index in PointNums:
        test.append(index)
        NewPointData.append( PointData[index] )
        AllPointNums.remove(index)
        # put the rest in
    for index in AllPointNums:
        try:
            NewPointData.append( PointData[index] )
        except:
            print "huh?", index
            return PointData, NewPointData
            
            ## Make sure the orientation is correct, and that the first boundary
            ## is the outer one
            
    NewPointData = np.array(NewPointData)
    Boundaries = np.array(Boundaries)
    
    hole_array = np.zeros( ( len(Holes), len(Boundaries) ) )
    bound_indexes = np.zeros( (len(Boundaries),) ) 
    
    ## Check which holes are in which boundaries
    for (i, point) in zip( range(len(Holes)), Holes )[:]:
        ## find which hole is in which boundary
        firstpoint = 0
        for (j, b) in enumerate(Boundaries): #zip( range(len(Boundaries)), Boundaries ):
            print "j, b", j, b
            poly = NewPointData[firstpoint:b+1,:2]
            if PinP(poly, point):
                hole_array[i,j] = 1
                if not bound_indexes[j]:
                    bound_indexes[j] = i
            firstpoint = b + 1
            
    # the outer boundary should have all the holes in it except the outer one:
    print "hole_array", hole_array
    num_holes = sum(hole_array)
    print "num_holes", num_holes
    outer = np.nonzero(num_holes == len(Holes)-1)
    if len(outer) > 1:
        raise VerdatError, "There is more than one polygon with all the holes in it!"
    rest = np.nonzero(num_holes == 1)
    if len(rest) <> len(Boundaries)-1:
        raise VerdatError, "there is not a 1:1 correspondence between holes and polygons"
        
        
    ## put the outer polygon at the front
    ## fixme: this has not been tested!
    print "outer:", outer
    outer = outer[0][0] # make this a scalar
    print "outer:", outer
    NewPointData, Boundaries = MoveBoundToFront(NewPointData, Boundaries, outer)
    bound_indexes = np.concatenate( (bound_indexes[outer:outer+1],bound_indexes[:outer],bound_indexes[outer+1:]) )
    
    
    ## fixme: should all this be moved to the inner loop above?
    ##        the problem is that we don't wnow till the end which is the outer bound.
    ## check for CC, CCW:
    # Outer Boundary should be CCW:
    poly = NewPointData[:Boundaries[0]+1]
    if IsClockwise(poly, Holes[0]):
        ReverseBound(NewPointData, Boundaries, 0)
        
        ## now the islands should be CW:
    for i in range(1,len(Boundaries)):
        poly = NewPointData[Boundaries[i-1]+1:Boundaries[i]+1]
        if not IsClockwise(poly,Holes[bound_indexes[i]]):
            ReverseBound(NewPointData, Boundaries, i)
            
    return NewPointData, Boundaries
    
    
    
def MoveBoundToFront(Points, Bounds, Bind):
    """
    Utility for SortForVerdat
    """
    if Bind < 0 or Bind >= len(Bounds):
        raise VerdatError, "Bind (%i) not valid for number of Boundaries"%Bind
    if not Bind == 0:
        poly = Points[ Bounds[Bind-1]+1:Bounds[Bind]+1 ]
        before = Points[ :Bounds[Bind-1]+1 ]
        after = Points[ Bounds[Bind]+1: ]
        Points = np.concatenate( (poly,before,after) )
        Boundlength = Bounds[Bind]-Bounds[Bind-1]
        Bounds = np.concatenate( ( np.array((Boundlength-1,)),
                                Bounds[:Bind-1]+ Boundlength,
                                Bounds[Bind:] ) )
    return Points, Bounds
    
def ReverseBound(Points, Bounds, Bind):
    """
    Utility for SortForVerdat
    """
    if Bind < 0 or Bind >= len(Bounds):
        raise VerdatError, "Bind (%i) not valid for number of Boundaries"%Bind
    if Bind == 0:
        start = 0
    else:
        start = Bounds[Bind-1]+1
    end = Bounds[Bind]+1
    Points[start:end] = take(Points, arange( end-1,start-1, -1 ) )
    
    return 1
    
    
    
def IsClockwise(poly, hole):
    """
    
    IsClockwise(poly, point) returns 1 if the polygon is oriented clockwise, and 0 otherwise
    point must a point inside the polygon

    fixme: Dan's code tha cmpute the area is a better way to do this! (see maproom SVN)
    
    """
    poly = poly[:,:2] # dump any extra data
    if poly[0][0] != poly[-1][0] or poly[0][1] != poly[-1][1]:
        # add the first point to the end if it isn't already there
        poly = np.concatenate( (poly,poly[:1]) )

    Diff =  poly - hole
    
    A = Diff[:-1]
    B = Diff[1:]
    length = np.sqrt(sum(Diff**2,1))
    
    a = length[:-1]
    b = length[1:]
    
    AdotB = sum(A*B,1)
    AXB = A[:,0]*B[:,1] - B[:,0]*A[:,1]
    
    Theta =   sum( np.arccos( AdotB / (a*b) ) ) # dot product determines the angle
    Theta2 =  sum( np.arcsin( AXB / (a*b) ) ) # cross product determines the sign
    Theta = np.where( (Theta2 < 0), -Theta, Theta )
    
    if Theta < -np.pi:
        return 1
    elif Theta > np.pi:
        return 0 
    else:
        raise VerdatError, "Can't compute CW-CCW, Perhaps the point is not in the polygon."
        
def angle(A,B): # this was really here just for developing the above function

    a = sqrt(A[0]**2 + A[1]**2)
    b = sqrt(B[0]**2 + B[1]**2)
    
    AdotB = sum( A * B ) 
    T = AdotB / (a*b)
    Theta1 = np.acos( T )
    
    AXB =  A[0]*B[1] - B[0]*A[1]
    Theta2 =  arcsin( AXB / (a*b) )
    
    if Theta2 < 0:
        Theta = -Theta1
    else:
        Theta = Theta1
        
    return Theta
