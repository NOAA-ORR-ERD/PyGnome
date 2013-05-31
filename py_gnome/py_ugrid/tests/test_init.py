#!/usr/bin/env python

"""
Some test of the py_ugrid classes

Designed to be run with py.test

"""
import numpy as np

import pytest

from py_ugrid import ugrid

# will the class initialize?
def test_init():
    grid = ugrid.ugrid()
    assert True # if we got this far, you can create an empty one...

def test_simple_grid_1():
    """
    initilizing about a simple a grid as you can with data
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
    
    grid = ugrid.ugrid(nodes, faces, edges)

    assert grid.num_nodes == 4
    assert np.array_equal(grid.nodes[2], (1,2))

    assert grid.num_faces == 2
    assert np.array_equal(grid.faces[0], (0,1,2))
    assert np.array_equal(grid.faces[1], (1,3,2))

    assert grid.num_edges == 4
    assert np.array_equal(grid.edges[0], (0,1))
    assert np.array_equal(grid.edges[3], (2,0))


def test_simple_grid_2():
    """
    initilizing about a simple a grid as you can
    adding the data later.
    """
    grid = ugrid.ugrid()
    grid.nodes = [(0,0),
                  (2,0),
                  (1,2),
                  (3,2)]
    assert grid.num_nodes == 4
    assert np.array_equal(grid.nodes[0], (0,0))
    grid.faces = [(0, 1, 2),
                  (1, 3, 2),]
    print grid.faces
    assert grid.num_faces == 2
    assert np.array_equal(grid.faces[0], (0,1,2))
    assert np.array_equal(grid.faces[1], (1,3,2))
    
    grid.edges = [(0,1),
                  (1,3),
                  (3,2),
                  (2,0)]
    assert grid.num_edges == 4
    assert np.array_equal(grid.edges[0], (0,1))
    assert np.array_equal(grid.edges[3], (2,0))
    
def test_data_set_nodes():
    """
    test initializing a simple data set on nodes
    """
    grid = ugrid.ugrid()
    
    grid.nodes = [(0,0),
                  (2,0),
                  (1,2),
                  (3,2)]
    depths = (3.4, 5.6, 2.3, 4.1)
    grid.set_node_data('depth', depths)
    
    assert np.array_equal(grid.get_node_data('depth'), depths)

    assert grid.get_node_data('depth', 1) == 5.6 

    assert np.array_equal( grid.get_node_data('depth', (1, 3) ), (5.6, 4.1) )

    grid.set_node_data('depth', data=1.1, indexes=1)
    assert grid.get_node_data('depth', 1) == 1.1 
    
    grid.set_node_data('depth', data=1.1, indexes=(0 ,3) )
    assert np.array_equal( grid.get_node_data('depth', (0, 3) ), (1.1, 1.1) )
    
    grid.set_node_data('depth', data=(1.1, 4.5), indexes=(0 ,3) )
    assert np.array_equal( grid.get_node_data('depth', (0, 3) ), (1.1, 4.5) )
    
    with pytest.raises(ValueError): 
        grid.set_face_data('depth', (3.4, 6.7, 2.3, 4.1, 6.5) )

    with pytest.raises(ValueError): 
        grid.set_face_data('depth', (3.4,) )
    
    with pytest.raises(KeyError): ## note: should this raise a Value error?
        grid.get_node_data('velocity')

def test_data_set_edges():
    """
    test initializing a simple data set on edges
    """
    grid = ugrid.ugrid()
    grid.nodes = [(0,0),
                  (2,0),
                  (1,2),
                  (3,2)]
    grid.edges = [(0,1),
                  (1,3),
                  (3,2),
                  (2,0)]
    bound_types = np.array( (0, 0, 1, 2), dtype=np.uint8) # flags for boundary types
    grid.set_edge_data('boundary_type', bound_types)
    
    assert np.array_equal(grid.get_edge_data('boundary_type'), bound_types)

    assert grid.get_edge_data('boundary_type', 1) == 0 

    assert np.array_equal( grid.get_edge_data('boundary_type', (1, 3) ), (0, 2) )

    grid.set_edge_data('boundary_type', data=3, indexes=1)
    assert grid.get_edge_data('boundary_type', 1) == 3 
    
    grid.set_edge_data('boundary_type', data=4, indexes=(0, 3) )
    assert np.array_equal( grid.get_edge_data('boundary_type', (0, 3) ), (4, 4) )
    
    grid.set_edge_data('boundary_type', data=(1, 2), indexes=(0 ,3) )
    assert np.array_equal( grid.get_edge_data('boundary_type', (0, 3) ), (1, 2) )
    
    with pytest.raises(KeyError): ## note: should this raise a Value error instead?
        grid.get_edge_data('velocity')

    with pytest.raises(IndexError): 
        grid.get_edge_data('boundary_type', indexes = 6)

    with pytest.raises(IndexError): 
        grid.get_edge_data('boundary_type', indexes = [1, 0, 6, 4])

def test_data_set_faces():
    """
    test initializing a simple data set on faces
    """
    grid = ugrid.ugrid()
    grid.nodes = [(0,0),
                  (2,0),
                  (1,2),
                  (3,2)]
    grid.faces = [(0, 1, 2),
                  (1, 3, 2),
                  ]

    u_velocity = np.array( (3.2, 4.5), dtype=np.float32) 
    grid.set_face_data('u_velocity', u_velocity)
    
    assert np.array_equal(grid.get_face_data('u_velocity'), u_velocity)
    
    assert grid.get_face_data('u_velocity', 1) == 4.5

    assert np.array_equal( grid.get_face_data('u_velocity', (0, 1) ), np.array( (3.2, 4.5), np.float32) )

    grid.set_face_data('u_velocity', data=3.1, indexes=1)
    assert grid.get_face_data('u_velocity', 1) == 3.1 
    
    grid.set_face_data('u_velocity', data=4, indexes=(1, 0) )
    assert np.array_equal( grid.get_face_data('u_velocity', (0, 1) ), (4, 4) )
    
    grid.set_face_data('u_velocity', data=(1.1, 2.2), indexes=(1 ,0 ) ) # note out of order indexes...
    assert np.array_equal( grid.get_face_data('u_velocity', (0, 1) ), np.array( (2.2, 1.1), np.float32) )
    
    with pytest.raises(ValueError): 
        grid.set_face_data('u_velocity', (3.4, 6.7, 2.3, 4.1) )

    with pytest.raises(ValueError): 
        grid.set_face_data('u_velocity', (3.4,) )

    with pytest.raises(KeyError): ## note: should this raise a Value error instead?
        grid.get_face_data('something')

    with pytest.raises(IndexError): 
        grid.get_face_data('u_velocity', indexes = 6)

    with pytest.raises(IndexError): 
        grid.set_face_data('u_velocity', 4.5, indexes = 6)

