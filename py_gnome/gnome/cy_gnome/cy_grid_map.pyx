cimport numpy as cnp
import numpy as np
import os

from .type_defs cimport *
from .grids cimport GridMap_c
from gnome.cy_gnome.cy_helpers import filename_as_bytes

#cimport cy_mover

cdef class CyGridMap:

    cdef GridMap_c *map
    
    def __cinit__(self):
        self.map = new GridMap_c()
    
    def __dealloc__(self):
        del self.map
        #self.map = NULL
    

    def text_read(self, grid_map_file):
        """
        .. function::text_read
        
        """
        cdef OSErr err

        cdef bytes grid_map_file_b = filename_as_bytes(grid_map_file)
        err = self.map.TextRead(grid_map_file_b)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridMap_c.TextRead returned an error.")

    def __init__(self):
        #self.grid.fIsOptimizedForStep = 0
        self.map.fGrid = NULL

    
    def export_topology(self, topology_file):
        """
        .. function::export_topology
        
        """
        cdef OSErr err
        cdef bytes topology_file_b = filename_as_bytes(topology_file)
        err = self.map.ExportTopology(topology_file_b)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridMap_c.ExportTopology returned an error.")

    def save_netcdf(self, netcdf_file):
        """
        .. function::save_netcdf
        
        """
        cdef OSErr err

        cdef bytes netcdf_file_bytes = filename_as_bytes(netcdf_file)

        err = self.map.SaveAsNetCDF(netcdf_file_bytes)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridMap_c.SaveAsNetCDF returned an error.")

