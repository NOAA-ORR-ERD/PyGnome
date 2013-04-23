cimport numpy as cnp
import numpy as np

from type_defs cimport *
from movers cimport GridMap_c
#cimport cy_mover

cdef class CyGridMap:

    cdef GridMap_c *map
    
    def __cinit__(self):
        self.map = new GridMap_c()
    
    def __dealloc__(self):
        del self.map
        self.map = NULL
    

    def text_read(self, grid_map_file):
        """
        .. function::text_read
        
        """
        cdef OSErr err
        err = self.map.TextRead(grid_map_file)
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
        err = self.map.ExportTopology(topology_file)
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
        err = self.map.SaveAsNetCDF(netcdf_file)
        if err != 0:
            """
            For now just raise an OSError - until the types of possible errors are defined and enumerated
            """
            raise OSError("GridMap_c.SaveAsNetCDF returned an error.")

