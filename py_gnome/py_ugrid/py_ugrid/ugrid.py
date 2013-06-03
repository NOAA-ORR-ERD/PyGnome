#!/usr/bin/env python
"""
ugrid classes

set of classes for working with unstructured model grids

The "ugrid" class is the base class: it stores eveything in memory

subclasses include a nc_ugrid, which points to a netcdf file (Or opendap url).

It provides the same API, but does not store the data in memory, rather
reading it on demand

NOTE: only full support for triangular mesh grids at the moment

"""

import numpy as np

# used for simple locate_face test
from .geometry.cy_point_in_polygon import point_in_poly as point_in_tri


class data_set(object):
    """
    a class to hold the data assocated with nodes, edges, etc
 
    """
    ##fixme: is this neccessary? -- only if there is more in it later..
    def __init__(self, name, type='node', data=None):
        """
        create a data_set object
        name: the name of the data (depth, u_velocity, etc)
        type: the type of grid element: node, edge, or face the data is assigned to
        data: the data
        """
        self.name = name
        if type not in ['node', 'edge', 'face']:
            raise ValueError("type must be one of: 'node', 'edge', 'face'")
        self.type = type # must be 'node', 'edge', of 'face' (eventually 'volume')

        if data is None:
            self.data = np.zeros((0,), dtype=np.float) # could be any data type
        else:
            self.data = data
     
class data_set_indexed(data_set):
    """
    a class to hold the arrays used to map data to indexes of the nodes, edges
    or faces they are assigned to, if there is not that data on all of the objects
    do we ever need this?
    
    """
    ## fixme: do we want a special case for when there is data on all the nodes, edges, etc?

    def __init__(self, name, type='node', indexes=None, data=None):
        """
        create a data_set object
        
        name: the name of the data (depth, u_velocity, etc)
        
        type: the type of grid element: node, edge, or face the data is assigned to
        
        """
        self.name = name
        self.type = type # must be 'node', 'edge', of 'face' (eventually 'volume')
        if (indexes is None) ^  (data is None):
            raise ValueError("indexes and data both need to be either None or have values")
        if indexes is None:
            self.indexes = np.zeros((0,), dtype=np.int32) 
        else:
            self.indexes = indexes
        if data is None:
            self.data = np.zeros((0,), dtype=np.float) # could be any data type
        else:
            self.data = data
     
    def check_consistent(self):
        """
        check if the indexes match the data, etc.
        """
        raise NotImplimentedError


class ugrid(object):
    """
    a basic class to hold an unstructred grid (triangular mesh)
    
    the internal structure mirrors the netcdf data standard.
    """

    def __init__(self, nodes=None, faces=[], edges=[], ):
        """
        ugrid class -- holds, saves, etc an unstructured grid

        :param nodes=None : the coordinates of the nodes -- (NX2) float array
        :param faces=[] : the faces of the grid -- (NX3) integer array of indexes into the nodes array
        :param edges=[] : the edges of the grid -- (NX2) integer array of indexes into the nodes array
        
        often this is too much data to pass in in a literal -- so usually
        specialized constructors will be used instead (load from file, etc.)
        """
        
        if nodes is None:
            self._nodes = np.zeros((0,2), dtype=np.float64)
        else:
            self._nodes = np.asarray(nodes, dtype=np.float64).reshape((-1, 2))

        self._faces = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
        self._edges = np.asarray(edges, dtype=np.int32).reshape((-1, 2))

        self._node_data = {}
        self._edge_data = {}
        self._face_data = {}

    def check_consistent(self):
        """
        check if the various data is consisent: the edges and faces reference
        existing nodes, etc.
        """
        raise NotImplimentedError

    @property
    def nodes(self):
        return self._nodes
    @nodes.setter
    def nodes(self, nodes_coords):
        # room here to do consistency checking, etc.
        # for now -- simply make sure it's a numpy array
        self._nodes = np.asarray(nodes_coords, dtype=np.float64).reshape((-1, 2))
    @nodes.deleter
    def nodes(self):
        ## if there are no nodes, there can't be any faces or edges
        self._nodes = np.zeros((0,2), dtype=np.float64)
        self._edges = np.zeros((0,2), dtype=np.int32)
        self._faces = np.zeros((0,3), dtype=np.int32)
        
    @property
    def num_nodes(self):
        return self._nodes.shape[0]
        
    @property
    def faces(self):
        return self._faces
    @faces.setter
    def faces(self, faces_indexes):
        # room here to do consistency checking, etc.
        # for now -- simply make sure it's a numpy array
        self._faces = np.asarray(faces_indexes, dtype=np.int32).reshape((-1, 3))
    @faces.deleter
    def faces(self):
        self._faces = np.zeros((0,3), dtype=np.int32)
    @property
    def num_faces(self):
        return self._faces.shape[0]
        
    @property
    def edges(self):
        return self._edges
    @edges.setter
    def edges(self, edges_indexes):
        # room here to do consistency checking, etc.
        # for now -- simply make sure it's a numpy array
        self._edges = np.asarray(edges_indexes, dtype=np.int32).reshape((-1, 2))
    @edges.deleter
    def edges(self):
        self._edges = np.zeros((0,2), dtype=np.int32)
    @property
    def num_edges(self):
        return self._edges.shape[0]
            
##fixme: repeated code here -- should these methods be combined?
    def set_node_data(self, name, data, indexes=None):
        if indexes is None:
            data = np.asarray(data)
            if not data.shape == (self.num_nodes,):
                raise ValueError("size of data should match number of nodes") # shape should match edges, data type can be anything
            self._node_data[name] = data
        else:
            indexes = np.array(indexes, dtype=np.int32).reshape((-1,))
            self._node_data[name][indexes] = data

    def get_node_data(self, name, indexes=None):
        if indexes is None:
            return self._node_data[name]
        else:
            indexes = np.array(indexes, dtype=np.int32).reshape((-1,))
            return self._node_data[name][indexes]

    def set_edge_data(self, name, data, indexes=None):
        if indexes is None:
            data = np.asarray(data)
            if not data.shape == (self.num_edges,):
                raise ValueError("size of data shold match number of edges") # shape should match edges, data type can be anything
            self._edge_data[name] = data
        else:
            indexes = np.array(indexes, dtype=np.int32).reshape((-1,))
            self._edge_data[name][indexes] = data

    def get_edge_data(self, name, indexes=None):
        if indexes is None:
            return self._edge_data[name]
        else:
            indexes = np.array(indexes, dtype=np.int32).reshape((-1,))
            return self._edge_data[name][indexes]

    def set_face_data(self, name, data, indexes=None):
        if indexes is None:
            data = np.asarray(data)
            if not data.shape == (self.num_faces,):
                raise ValueError("size of data shold match number of faces") # shape should match faces, data type can be anything
            self._face_data[name] = data
        else:
            indexes = np.array(indexes, dtype=np.int32).reshape((-1,))
            self._face_data[name][indexes] = data

    def get_face_data(self, name, indexes=None):
        if indexes is None:
            return self._face_data[name]
        else:
            indexes = np.array(indexes, dtype=np.int32).reshape((-1,))
            return self._face_data[name][indexes]


    def locate_face_simple(self, point):
        """
        returns the index of the face that the point is in
        
        returns None if the point is not in the mesh
        
        : param point :  the point that you want to locate -- (x, y)
        
        this is a very simple, look through all the faces search.
        It is slow ( O(N) ), but should be robust 
        """        
        for i, face in enumerate(self._faces):
            f = self._nodes[face]
            #print "checking:", point, "in", f 
            if point_in_tri(f, point):
                #print "got a hit:", i
                return i
        return None
            
            
    
    def save_as_netcdf(self, filename):
        """
        save the ugrid object as a netcdf file
        
        follows the convernsion established by the netcdf UGRID working group:
        
        http://publicwiki.deltares.nl/display/NETCDF/Deltares+CF+proposal+for+Unstructured+Grid+data+model
                
        """
        
        import netCDF4
        

