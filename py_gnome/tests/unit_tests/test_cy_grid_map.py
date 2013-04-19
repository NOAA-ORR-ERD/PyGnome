"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
import numpy as np

from gnome import basic_types
from gnome.cy_gnome import cy_grid_map

import pytest

here = os.path.dirname(__file__)

# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         cy_grid_map.CyGridMap()
# 
@pytest.mark.slow
class TestGridMap():
    gcm = cy_grid_map.CyGridMap()    
        
    def test_grid_map_curv(self):
        """
        Test a grid map - read and write out
        """
		#curvilinear grid
        grid_map_file = os.path.join(here, r'SampleData/currents/ny_cg.nc')
        netcdf_file = os.path.join(here, r'SampleData/currents/ny_cg_top.nc')
        #topology_file = os.path.join(here, r'SampleData/currents/ny_cg_top.dat')
        self.gcm.text_read(grid_map_file)
        #self.gcm.export_topology(topology_file)
        self.gcm.save_netcdf(netcdf_file)

    def test_grid_map_tri(self):
        """
        Test a grid map - read and write out
        """
        #triangle grid
        grid_map_file = os.path.join(here, r'SampleData/currents/ChesBay.nc')
        netcdf_file = os.path.join(here, r'SampleData/currents/ChesBayTop.nc')
        #topology_file = os.path.join(here, r'SampleData/currents/chesbay_top.dat')
        self.gcm.text_read(grid_map_file)
        #self.gcm.export_topology(topology_file)
        self.gcm.save_netcdf(netcdf_file)
               
    
if __name__ == "__main__":
    """
    This makes it easy to use this file to debug the lib_gnome DLL 
    through Visual Studio
    """
    tgm = TestGridMap()
    tgm.test_grid_map_curv()
    tgm.test_grid_map_tri()
