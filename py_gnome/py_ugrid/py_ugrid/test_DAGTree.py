#!/usr/bin/env python

"""
test for DAGTree.py

designed to be run with py.test

"""

import pytest

import numpy as np

#TESTS:

from DAGTree import Triangle, DAGTree, SideOfLine

class Mesh(object):
    """
    A very simple class to hold a whole mesh

    """
    def __init__(self, Points, Triangles, Type = "DAG"):
        self.Points = Points
        self.Triangles = Triangles
        if Type == "DAG":
            self.Tree = DAGTree(Triangles)
#        elif Type == "Quad":
#            self.Tree = QuadTree(Triangles)

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
    Points = np.array((( 5, 10),
                      ( 9, 10),
                      ( 3,  7),
                      ( 6,  7),
                      (10,  7),
                      ( 6,  2),
                      (11,  3),
                      ),np.int32)

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
        
#    elif Type == "Quad":
#        TestMesh = Mesh(Points, Triangles, "Quad")
    return TestMesh


class Test_SideOfLine:

    TestMesh = GetTestMesh()
    Points = TestMesh.Points

    def test_right(self):
        assert SideOfLine(np.array( (self.Points[1], self.Points[3]) ), self.Points[0]) == -1
        assert SideOfLine(np.array( (self.Points[3], self.Points[2]) ), self.Points[0]) == -1
    
    def test_left(self):
        assert SideOfLine(np.array((self.Points[1],self.Points[3])),self.Points[4]) == 1
        assert SideOfLine(np.array((self.Points[3],self.Points[2])),self.Points[5]) == 1

    def test_on(self):
        assert SideOfLine(np.array((self.Points[3],self.Points[2])),self.Points[4]) == 0

class Test_TriSideOfLine:
    TestMesh = GetTestMesh()
    Triangles = TestMesh.Triangles
    Points = TestMesh.Points

    def test_right(self):
        assert self.Triangles[0].TriSideOfLine(np.array((self.Points[1],self.Points[3]))) == -1

    def test_left(self):
        assert self.Triangles[3].TriSideOfLine(np.array((self.Points[1],self.Points[3]))) == 1

    def test_on(self):
        assert self.Triangles[4].TriSideOfLine(np.array((self.Points[1],self.Points[3]))) == 0 

class Test_FindPointTri:
    TestMesh = GetTestMesh()
    TestTree = TestMesh.Tree
    Triangles = TestMesh.Triangles
    Points = TestMesh.Points

    tests = (( (5,8), (0,) ),
             ( (6,8), (1,) ),
             ( (9,9), (2,) ),
             ( (9,6), (3,) ),
             ( (5,6), (4,) ),
             ( (5,7), (0, 4) ),
             ( (9,4), (5,) ),
             ( (1,1), (None,) ),
             )
    
    @pytest.mark.parametrize( ("point", "realtris"), tests)
    def test_FindPointInTri(self, point, realtris):
        tri = self.TestTree.FindPointTri(point)
        print "Point: %s is in Triangle %s, should be in %s"%(point, tri.Index, realtris )
        assert tri.Index in realtris

#def QuadTest():
#    TestMesh = GetTestMesh("Quad")
#    TestTree = TestMesh.Tree
#    Triangles = TestMesh.Triangles
#    Points = TestMesh.Points
#
#    print "Checking FindPointTri:"
#
#    def CheckPointInTri(point, realtris):
#        tri = TestTree.FindPointTri(point)
#        if tri.Index in realtris:
#            return True
#        else:
#            print "Point: %s is in Triangle %s, should be in %s"%(point, tri.Index, realtris )
#            return False
#
#    tests = (( (5,8), (0,) ),
#             ( (6,8), (1,) ),
#             ( (9,9), (2,) ),
#             ( (9,6), (3,) ),
#             ( (5,6), (4,) ),
#             ( (5,7), (0, 4) ),
#             ( (9,4), (5,) ),
#             ( (1,1), (None,) ),
#             )
#    for test in tests:
#       print CheckPointInTri(*test)
#
#def ProfileTest():
#    import profile
#    import RandomArray
#
#    global MyMesh, TestPoints
#    
#    elefile = "TestData/A.1.ele"
#    MyMesh = LoadElementFile(elefile)
#
#    minx = np.minimum.reduce(MyMesh.Points[:,0])
#    maxx = np.maximum.reduce(MyMesh.Points[:,0])
#    miny = np.minimum.reduce(MyMesh.Points[:,1])
#    maxy = np.maximum.reduce(MyMesh.Points[:,1])
#
#    xpoints = RandomArray.uniform(minx, maxx, (1000,1))
#    ypoints = RandomArray.uniform(miny, maxy, (1000,1))
#    TestPoints = np.concatenate((xpoints, ypoints),1)
#
#    profile.run('DoLots(MyMesh.Tree, TestPoints)')
#
#    #DoLots(MyMesh.Tree, TestPoints)
#
#def DoLots(Tree, Points):
#    print "checking a bunch of points"
#    for point in Points:
#        t = Tree.FindPointTri(point)
#        #print " point is in tri:", t
#
#
#if __name__ =="__main__":
#    import sys
#    if len(sys.argv) == 1:
#        SimpleTest()
#        #QuadTest()
#    elif sys.argv[1] == "p":
#       ProfileTest()
#
    
    


    
