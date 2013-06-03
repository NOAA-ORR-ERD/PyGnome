#!/usr/bin/env python

"""
test code to try out the "edge up" method for searching a triangular mesh

This is NOT get very far at all -- in fact, no where!

could it be used for a quad mesh (curvilinear grid...)

NOTE: I've been thinking abou this, and I'm pretty sure the trick is that
      it will only work with a convex hull.

uses some of the code in the DAGTri module, too.

According to this non-authoritative post, it should be order log(N)
and could be fast if we generally can start from a nearby triangle.
(however, as far as I can tell, it should be order sqrt(N), but
 that's still good.)

http://scicomp.stackexchange.com/questions/2624/finding-which-triangles-points-are-in

Here is a paper about a similar method:

http://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-728.pdf

"""

DAGTree import Triangle, NullTriangle


class Mesh(object):
    """
    class to hold a triangular mesh
    
    and provide the edge_hop search method
    
    """
    def __init__(self, triangles):
        """
        creates a Mesh object from the input triangles
        
        param: triangles is an ordered list of DAGTree.Triangle objects
               i.e if a neightbor triangle is indexed i, that is triangles[i]
               triangles[0] is the null triangle
        """
        
        self.triangles = triangles
    
    def find_point(point, starting_tri=1):
        """
        returns the index of th triangle the point is in.
        param: point -- (x, y) tuple or numpy array point you want to know
               what triangle it's in.
        param: starting_tri=1  optional starting triangle -- if not given the first (non-null) triangle will be used.
        """
        
        tri = starting_tri
        while True:  # keeping looking 'till it's found!
            
        
    