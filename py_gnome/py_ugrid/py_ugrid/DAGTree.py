#!/usr/bin/env python

"""

My first stab at a DAG tree implimentation

This should work for any triangle mesh that meets these requirements:

- The domain is covered in non-overlapping triangles
- The triangles vertices are ordered counter-clockwise
- None of the triangles are degenerate (three points on the same line, etc)

Note: This should work for small enough integers, and floats that are
far enough apart so that none of the triangles appear degenerate. It
really should use Skewchuck's "robust predicates" to do it right.

There is some test quadtree code in here too.

"""

import numpy as N

class DAGNode(object):
    def __init__(self, Segment=None, ParentNode = None, Triangles = []):

        self.Segment = Segment
        self.Triangles = Triangles

        self.ParentNode = ParentNode
        self.RightNode = None
        self.LeftNode = None

    def IsLeaf(self):
        #print "in IsLeaf:", self.Triangles
        ##fixme: should I do something more robust?
        if self.RightNode is None:
            return True

    def GetSubTree(self, Point):
        """
        returns the node of the subtree that the point is in
        """
        #print "In GetSubTree: ", self.Triangles
        side = SideOfLine(self.Segment, Point)
        if side  < 0 :
            return self.RightNode
        elif side == 0:
            return self.LeftNode
        else:
            return self.LeftNode
        
    def GetNodesTriangles(self, nodeType): # nodeType = "Parent", "Right", or "Left"

        return eval("self." + nodeType + "Node.Triangles")


class QuadNode(object):
    def __init__(self,
                 BoundingBox,
                 X = None, Y = None,
                 ParentNode = None,
                 Triangles = []):

        self.BB = BoundingBox
        self.X = X
        self.Y = Y
        self.YL = None
        self.YR = None
        self.Triangles = Triangles

        self.ParentNode = ParentNode

        self.ULNode = None
        self.URNode = None
        self.LLNode = None
        self.LRNode = None

    def IsLeaf(self):
        # fixme: is this robust? (and fast enough?)
        if self.ULNode is None:
            return True
        else:
            return False

    def IsNull(self):
        if len(self.Triangles) == 0:
            return True
        else:
            return False

    def GetSubTree(self, Point):
        """
        returns the node of the subtree that the point is in
        """
        for t in self.Triangles:
            print t.Index

        if Point[0] >= self.X:
            if point[1] >= self.YR:
                return self.URNode
            else:
                return self.LRNode
        else:
            if point[1] >= self.YL:
                return self.ULNode
            else:
                return self.LLNode


class Triangle(object):
    def __init__(self, Points, Vertices, Neighbors = None, Index = None):

        """
        The triangle class

        Points is an NX2 array of all the points in the mesh

        Vertices is a 3-tuple of indices into the Points array

        Neighbors is a 3-tuple of neighboring triangles

        """
       
        self.Points = Points
        self.Vertices = Vertices
        self.Neighbors = Neighbors
        self.Index = Index

        self.IsNull = False
        
##    def __eq__(self, other):
##        
##        return self.__dict__ == other.__dict__

    def GetSegment(self, SideNum):

        """
        returns the segment cooresponding to the SideNum:

        segment 0 --> (point0, point1)
        segment 1 --> (point1, point2)
        segment 2 --> (point2, point0)

        This keeps the counterclockwise ordering of points

        """
        Points = self.Points
        Ver = self.Vertices
        if SideNum == 0:
            return N.array((Points[Ver[0]], Points[Ver[1]] ))
        elif SideNum == 1:
            return N.array((Points[Ver[1]], Points[Ver[2]] ))
        elif SideNum == 2:
            return N.array((Points[Ver[2]], Points[Ver[0]] ))
        else:
            raise ValueError("SideNum can be one of: 0,1,2")
        
    def PointInTri(self, Point):
        """
        returns True if Point is in self (the current Triangle instance),
        False otherwise
        """
        for seg in range(3):
            if SideOfLine(self.GetSegment(seg), Point) < 0:
                # all points should be the the left of the segments
                # or it's not in the triangle.
                # On the line is considered inside
                return False
        return True
    
    def GetVertices(self):
        """
        returns the vertices of the triangles in an 3X2 array of coordinates

        """
        
        return N.array((self.Points[self.Vertices[0]],
                        self.Points[self.Vertices[1]],
                        self.Points[self.Vertices[2]]) )

    def TriSideOfLine(self, Seg):
        """
        
        returns -1 if entire triangle is to the right of line
        returns +1 if entire triangle is to the left of line
        returns  0 if part of triangle is to the right of line and part to the left
        
        NOTE: this allows two points on the same side, and one on the line
        to count as that side. Is that robust?
        
        fixme: This can probably be optimized!
        
        
        """
        on = 0
        count = 0
        for point in self.GetVertices():
            side = SideOfLine(Seg, point)
            count += side
            if side == 0: on += 1
        if count <= -2: return -1
        elif count >=  2: return 1
        else:
            if on == 2:
                return count
            else:
                return 0

    def TriAboveLine(self, X, Coord = 0):
        """
        Returns 1 if triangle is all above the line at X
                0 i the triangle stradles the line
               -1 if the tiangle is below the line

        If any point is the  line, it returns 0.  This assures that
        you will find the triangle when you search for it!

               
        """

        ## fixme: this could use some optimization
        ## and a unit test!
        above = 0
        below = 0
        on = 0
        for point in self.GetVertices():
            x = point[Coord]
            if x <  X: # above (or on) the line
                below += 1
            elif x > X:
                above += 1
            else:
                on += 1
        if above == 3:
            return 1
        elif below == 3:
            return -1
        elif above > 0 and below > 0:
            return 0
        elif on == 1 and above == 2:
            return 1
        elif on ==2 and above == 1:
            return 1
        else:
            return 0

    def GetCenter(self):
        """
        returns the mean of the vertices of the triangle
        """

        return N.sum(self.GetVertices())/3.0

        
        
    def __str__(self):
        return "Triangle: #%s "%self.Index

    def __repr__(self):
        return "T#%s "%self.Index


def TriCompare(Tri1, Tri2, Coord = 0):
    Max1 = N.maximum.reduce(Tri1.GetVertices()[:,Coord])
    Max2 = N.maximum.reduce(Tri2.GetVertices()[:,Coord])
    Min1 = N.minimum.reduce(Tri1.GetVertices()[:,Coord])
    Min2 = N.minimum.reduce(Tri2.GetVertices()[:,Coord])
    if Min1 >= Max2:
        return 1
    elif Max1 < Min2:
        return -1
    elif Min1 > Min2:
        return 1
    elif Max1 < Max2:
        return -1
    else:
        return 0

def TriCompareX(Tri1, Tri2):
    return TriCompare(Tri1, Tri2, 0)

def TriCompareY(Tri1, Tri2):
    return TriCompare(Tri1, Tri2, 1)

    
class NullTriangle(object):
    """
    class that represents the space outside of the domain

    It is on both the right and the left of all segments

    """

    def __init__(self):
        self.Points = None
        self.Vertices = None
        self.Neighbors = None
        self.Index = None

        self.IsNull = True

    def GetSegment(self, SideNum):
        return None

    def PointInTri(self, Point):
        ##fixme: should this do a real test
        ##       of outside the domain?
        return None

    def GetVertices(self):
        return None

    def TriSideOfLine(self, Seg):
        """
        always returns 0: it is on both sides of every line
        """
        return 0

    def __str__(self):
        return "Null Triangle"

OutOfDomain = NullTriangle()
    

class QuadTree(object):
    def __init__(self, Triangles):
        # fixme:
        # I think to do this, you'd need to use the neighbor info.
        #Triangles.append( NullTriangle() )

        self.Root = self.BuildTree(Triangles)

    def FindPointTri(self, Point):
        node = self.Root
        while True:
            if node.IsLeaf():
                ## Down to the last triangle, but is it inside the domain at all?
                ## another option would be to put a Null triangle in the tree
                ## that represents outside the domain.
                ## I can't figure out how to do that withouth neighbor info on the triangles
                tri = node.Triangles[0]
                if tri.PointInTri(Point):
                    return tri
                else:
                    return OutOfDomain

            node = node.GetSubTree(Point)

    def CalcBB(self, Triangles):
        t = Triangles[0]
        ## BB = ( (xmin, ymin), (xmax, ymax) )
        BB = N.array(((t.GetVertices()[0]),(t.GetVertices()[0])), N.float)
        for t in Triangles:
            for v in t.GetVertices():
                BB[0,0] = min(BB[0,0], v[0]) 
                BB[0,1] = min(BB[0,1], v[1]) 
                BB[1,0] = max(BB[1,0], v[0]) 
                BB[1,1] = max(BB[1,1], v[1]) 
        return BB
    
    def BuildTree(self, Triangles, BoundingBox, Parent = None):
        """
        Recursively builds a Quad tree
        """
        #print "Building a Quad Tree with these triangles:"
        for t in Triangles:
            print t.Index,
        print
        
        if len(Triangles) == 1: # This is a leaf node
            #print "returning a leaf with this triangle:", Triangles[0].Index
            return QuadNode(Triangles = Triangles)

        else: # Create Sub-trees
            Triangles.sort(TriCompareX)
            print "the X-sorted triangles are:", Triangles
            
            X = Triangles[len(Triangles)//2].GetCenter()[0]
            Right = []
            Left = []
            Both = []
            for t in Triangles:
                if t.TriAboveLine(X, 0) > 0:
                    Right.append(t)
                if t.TriAboveLine(X, 0) < 0:
                    Left.append(t)
                else:
                    Both.append(t)

            print "X: %f"%(X)
            print "Right Triangles are:", Right
            print "Left Triangles are:", Left
            print "Both Triangles are:", Both
            
            Right.sort(TriCompareY)
            print "Right Triangles sorted by Y:", Right
            Y = Triangles[len(Right)//2].GetCenter()[1]
            print "Y: %f"%(Y)


            UpperRight = []
            LowerRight = []
            for t in Right:
                if t.TriAboveLine(Y, 1) >= 0:
                    UpperRight.append(t)
                if t.TriAboveLine(Y, 1) <= 0:
                    LowerRight.append(t)

            Left.sort(TriCompareY)
            Y = Triangles[len(Left)//2].GetCenter()[0]
            UpperLeft = []
            LowerLeft = []
            ##divide the Left:
            for t in Left:
                if t.TriAboveLine(Y, 1) >= 0:
                    UpperLeft.append(t)
                if t.TriAboveLine(Y, 1) <= 0:
                    LowerLeft.append(t)
            print "UpperRight:", UpperRight
            print "UpperLeft:", UpperLeft
            print "LowerRight:", LowerRight
            print "LowerLeft:", LowerLeft

            raise(Exception)

            ## fixme: Do I need to store the triangles in the node?
            node = QuadNode(X, Y, Parent, Triangles)

            ## fixme: is this circular reference a problem?
            print " Building the UR tree:"
            node.URNode = self.BuildTree(UpperRight, node)
            print " Building the UL tree:"
            node.ULNode = self.BuildTree(UpperLeft, node)
            print " Building the LR tree:"
            node.LRNode = self.BuildTree(LowerRight, node)
            print " Building the UL tree:"
            node.LLNode = self.BuildTree(LowerLeft, node)

            return node


class DAGTree(object):
    def __init__(self, Triangles):
        # fixme:
        # I think to do this, you'd need to use the neighbor info.
        #Triangles.append( NullTriangle() )
        self.Root = self.BuildTree(Triangles)

    def FindPointTri(self, Point, Subtrees=None):
        """
        returns the Triangle instance in self (a DAGTree instance)
        in which argument Point lies, OutOfDomain if Point lies in 
        no Triangle in self.  Subtrees returns diagnostic information.
        """
#        print Point
        node = self.Root
        while True:
            if node.IsLeaf():
                ## Down to the last triangle, but is it inside the domain at all?
                ## another option would be to put a Null triangle in the tree
                ## that represents outside the domain.
                ## I can't figure out how to do that without neighbor info on the triangles
                tri = node.Triangles[0]
                if tri.PointInTri(Point):
                    return tri
                else:
                    if Subtrees is not None:
                        Subtrees.append((node.ParentNode.Segment, []))
#                    print "Returning OutOfDomain"
                    return OutOfDomain
                
            node = node.GetSubTree(Point)
            if Subtrees is not None:
                l = []
                for t in node.Triangles:
                   l.append(t.Index)
                Subtrees.append((node.ParentNode.Segment, l))
            
    def BuildTree(self, Triangles, Parent = None):
        """
        Recusively builds a DAG tree
        """
##        print "Building a Tree with these triangles:"
##        for t in Triangles:
##            print t.Index,
##        print 
        if len(Triangles) == 1: # This is a leaf node
            #print "returning a leaf with this triangle:", Triangles[0].Index
            return DAGNode(ParentNode=Parent, Triangles=Triangles)
        else: # Create Sub-tree
            # loop through all the sides of all the triangles
            Done = False
            ## fixme: the starting triangle is always on the left
            ## should I just add it at the end, so that the first one
            ## doesn't always get used first?
            for starttri in xrange(len(Triangles)):
                #print "Using starting triangle: %s"%Triangles[starttri].Index
                for side in range(3):
                    #print " Using side:", side
                    Segment = Triangles[starttri].GetSegment(side)
                    Right = []
                    Left = []
                    for tri in Triangles:
                        side = tri.TriSideOfLine(Segment)
                        if side < 0:
                            #print "tri: %s is on the right"%tri.Index
                            Right.append(tri)
                        elif side > 0:
                            #print "tri: %s is on the left"%tri.Index
                            Left.append(tri)
                        else:
                            #print "tri: %s is on both"%tri.Index
                            Right.append(tri)
                            Left.append(tri)
                    if len(Right) < len(Triangles) and len(Left) < len(Triangles):
                        ##fixme: maybe a check here for a more balanced tree? 
                        Done = True
                        break
                if Done:
                    break
            ## fixme: Do I need to store the triangles in the node?
            node = DAGNode(Segment, Parent, Triangles)
            ## fixme: is this circular reference a problem?
            #print " Building the Right tree:"
            node.RightNode = self.BuildTree(Right, node)
            #print " Building the Left tree:"
            node.LeftNode = self.BuildTree(Left, node)

            return node

def SideOfLine(Line, Point):
    """
    determines which side of a line segment a point is on.
    Right and left defined as though looing form point1 to point 2
    
    :param Line: -- the line segment, defined by two points: ( (x1,y1), (x2,y2)  )
    
    returns -1 if Point is to the right of Line
    returns +1 if Point is to the left of Line
    returns  0 if Point is on Line

    Note: NOT ROBUST with floats

    """

##    ##OLD:
##    ax = Line[0][0]
##    ay = Line[0][1]
##    bx = Line[1][0]
##    by = Line[1][1]
##    cx = Point[0]
##    cy = Point[1]

##    A = ax-cx
##    B = ay-cy
##    C = bx-cx
##    D = by-cy
##    Det = A*D - B*C

    ##NEW this is a little faster, but not much
    cx = Point[0]
    cy = Point[1]
    
    Det = (Line[0][0] - cx)*(Line[1][1] - cy) - (Line[0][1] - cy)*(Line[1][0] - cx)

    if Det > 0: return 1
    elif Det < 0: return -1
    else: return 0
            
def makeDAGTreepyTriFromBMout(dataObj, index=0):
    ''' makeDAGTreepyTriFromBMout makes a Chris Barker's DagTree.py 
        Triangle object from vertex data output from BaroModes
    '''
    points = N.transpose( N.array((dataObj.theData["vLons"],
                                    dataObj.theData["vLats"])) )#,
#                                    float) )
    return Triangle(points, 
                    tuple(dataObj.theData["tris"][index]), 
                    Index=index)
    
def makeDAGTree(dataObj):
    triList = [ makeDAGTreepyTriFromBMout(dataObj, i) \
                for i in range(len(dataObj.theData["tris"][:,0])) ]
    return DAGTree(triList)

def GetBMoutTriNumUsingDAGTree(dataObj, LonLat):
#    print vars(dataObj)
#    print LonLat
    tree = makeDAGTree(dataObj)
#    print vars(tree.Root)
#    print vars(tree.FindPointTri(LonLat))
    temp = tree.FindPointTri(LonLat).Index
#    print temp
    return temp

def GetBMoutTriNum(dataObj, LonLat):
    i = check = 0
    while (i < len(dataObj.theData["tris"][:,0])) and (not check): 
        check = makeDAGTreepyTriFromBMout(dataObj, i).PointInTri(LonLat)
        i += 1
    if check:
        return i-1
    else:
        return None
        
#TESTS:

class Mesh(object):
    """
    A very simple class to hold a whole mesh

    """
    def __init__(self, Points, Triangles, Type = "DAG"):
        self.Points = Points
        self.Triangles = Triangles
        if Type == "DAG":
            self.Tree = DAGTree(Triangles)
        elif Type == "Quad":
            self.Tree = QuadTree(Triangles)

##def LoadElementFile(filename):
##    if filename[-4:].lower() == ".ele":
##        elename = filename
##        nodename = filename[:-4] + ".node"
##    else:
##        raise Exception("%s does not appear to be an element file"%filename)
##    print "loading node file"
##    (Points, temp, temp, StartIndex) = FileTools.ReadNodeFile(nodename)
##    Points = Points
##    print "loading ele file"
##    TriIndexes = FileTools.ReadEleFile(elename)
##    if StartIndex == 1:
##        TriIndexes -= 1
##    Triangles = []
##    print "making triangles"
##    for i in xrange(len(TriIndexes)):
##        Triangles.append(Triangle(Points, TriIndexes[i],Index = i))
##    print "there are %i triangles"%len(Triangles)
##    print "Building DagTree"
##    M = Mesh(Points, Triangles)
##    print "DagTree built"
##    return M

def GetTestTriangles():
    Points = N.array((( 5, 10),
                      ( 9, 10),
                      ( 3,  7),
                      ( 6,  7),
                      (10,  7),
                      ( 6,  2),
                      (11,  3),
                      ),N.int32)

    Triangles = [Triangle(Points, (0,2,3) ),
                 Triangle(Points, (0,3,1) ),
                 Triangle(Points, (3,4,1) ),
                 Triangle(Points, (3,5,4) ),
                 Triangle(Points, (2,5,3) ),
                 Triangle(Points, (5,6,4) ),
                 ]


    # not using neighbors for anything at the moment
    #T0.Neighbors = (None, T4, T1)
    #T1.Neighbors = (T0, T2, None)
    #T2.Neighbors = (T3, None, T1)
    #T3.Neighbors = (T4, None, T2)
    #T4.Neighbors = (None, T3, T0)

    for i, tri in enumerate(Triangles):
        tri.Index = i

    return Triangles, Points

def GetTestMesh(Type = "DAG"):
    Triangles, Points = GetTestTriangles()
    if Type == "DAG":
        TestMesh = Mesh(Points, Triangles)
#        f = file("pickledTestDAGTree",'w')
#        pcklr = P.Pickler(f)
#        P.dump(TestMesh, f)
##        f.close()
##        print dir()
##        del TestMesh
##        print dir()
##        f = file("pickledTestDAGTree",'r')
###        upcklr = P.Unpickler(f)
##        TestMesh = P.load(f)
##        print dir()
##        f.close()
        
    elif Type == "Quad":
        TestMesh = Mesh(Points, Triangles, "Quad")
    return TestMesh

def SimpleTest():
    TestMesh = GetTestMesh()
    TestTree = TestMesh.Tree
    Triangles = TestMesh.Triangles
    Points = TestMesh.Points

    print " Checking SideOf Line:"
    print SideOfLine(N.array((Points[1],Points[3])),Points[0]) == -1
    print SideOfLine(N.array((Points[1],Points[3])),Points[4]) == 1

    print SideOfLine(N.array((Points[3],Points[2])),Points[4]) == 0
    print SideOfLine(N.array((Points[3],Points[2])),Points[5]) == 1
    print SideOfLine(N.array((Points[3],Points[2])),Points[0]) == -1

    print "Checking TriSideOfLine:"
    print Triangles[0].TriSideOfLine(N.array((Points[1],Points[3]))) == -1
    print Triangles[3].TriSideOfLine(N.array((Points[1],Points[3]))) == 1
    print Triangles[4].TriSideOfLine(N.array((Points[1],Points[3]))) == 0 
                                                                  
                                                                  
    print "Checking FindPointTri:"

    def CheckPointInTri(point, realtris):
        tri = TestTree.FindPointTri(point)
        if tri.Index in realtris:
            return True
        else:
            print "Point: %s is in Triangle %s, should be in %s"%(point, tri.Index, realtris )
            return False

    tests = (( (5,8), (0,) ),
             ( (6,8), (1,) ),
             ( (9,9), (2,) ),
             ( (9,6), (3,) ),
             ( (5,6), (4,) ),
             ( (5,7), (0, 4) ),
             ( (9,4), (5,) ),
             ( (1,1), (None,) ),
             )
    for test in tests:
       print CheckPointInTri(*test)

def QuadTest():
    TestMesh = GetTestMesh("Quad")
    TestTree = TestMesh.Tree
    Triangles = TestMesh.Triangles
    Points = TestMesh.Points

    print "Checking FindPointTri:"

    def CheckPointInTri(point, realtris):
        tri = TestTree.FindPointTri(point)
        if tri.Index in realtris:
            return True
        else:
            print "Point: %s is in Triangle %s, should be in %s"%(point, tri.Index, realtris )
            return False

    tests = (( (5,8), (0,) ),
             ( (6,8), (1,) ),
             ( (9,9), (2,) ),
             ( (9,6), (3,) ),
             ( (5,6), (4,) ),
             ( (5,7), (0, 4) ),
             ( (9,4), (5,) ),
             ( (1,1), (None,) ),
             )
    for test in tests:
       print CheckPointInTri(*test)

def ProfileTest():
    import profile
    import RandomArray

    global MyMesh, TestPoints
    
    elefile = "TestData/A.1.ele"
    MyMesh = LoadElementFile(elefile)

    minx = N.minimum.reduce(MyMesh.Points[:,0])
    maxx = N.maximum.reduce(MyMesh.Points[:,0])
    miny = N.minimum.reduce(MyMesh.Points[:,1])
    maxy = N.maximum.reduce(MyMesh.Points[:,1])

    xpoints = RandomArray.uniform(minx, maxx, (1000,1))
    ypoints = RandomArray.uniform(miny, maxy, (1000,1))
    TestPoints = N.concatenate((xpoints, ypoints),1)

    profile.run('DoLots(MyMesh.Tree, TestPoints)')

    #DoLots(MyMesh.Tree, TestPoints)

def DoLots(Tree, Points):
    print "checking a bunch of points"
    for point in Points:
        t = Tree.FindPointTri(point)
        #print " point is in tri:", t


if __name__ =="__main__":
    import sys
    if len(sys.argv) == 1:
        SimpleTest()
        #QuadTest()
    elif sys.argv[1] == "p":
       ProfileTest()

    
    


    
