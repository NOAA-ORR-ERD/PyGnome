#!/usr/bin/env python

"""
Some example UGRIDs to test, etc with

"""

from py_ugrid import ugrid

def get_triangle_1():
    """
    returns about the simplest triangle grid possible
    """
    nodes = [(0,0),
                  (2,0),
                  (1,2),
                  (3,2)]

    faces = [(0, 1, 2),
                  (1, 3, 2),]

    edges = [(0,1),
                  (1,3),
                  (3,2),
                  (2,0)]
    return ugrid.ugrid(nodes, faces, edges)


def get_triangle_2():
    """
    rerturns set up a basic triangle grid with a hole and a "tail"
    """
    nodes = [(5,1),
             (10,1),
             (3,3),
             (7,3),
             (9,4),
             (12,4),
             (5,5),
             (3,7),
             (5,7),
             (7,7),
             (9,7),
             (11,7),
             (5,9),
             (8,9),
             (11,9),
             (9,11),
             (11,11),
             (7,13),
             (9,13),
             (7,15),
             ]
    
    faces = [(0,1,3),
             (0,2,6),
             (0,3,6),
             (1,3,4),
             (1,4,5),
             (2,6,7),
             (6,7,8),
             (7,8,12),
             (6,8,9),
             (8,9,12),
             (9,12,13),
             (4,5,11),
             (4,10,11),
             (9,10,13),
             (10,11,14),
             (10,13,14),
             (13,14,15),
             (14,15,16),
             (15,16,18),
             (15,17,18),
             (17,18,19),
             ]
    
    edges = [(0,1),
             (1,5),
             (5,11),
             (11,14),
             (14,16),
             (16,18),
             (18,19),
             (19,17),
             (17,15),
             (15,13),
             (13,12),
             (12,7),
             (7,2),
             (2,0),
             (3,4),
             (4,10),
             (10,9),
             (9,6),
             (6,3),
             ]

    return ugrid.ugrid(nodes, faces, edges)

    
    
    
    







