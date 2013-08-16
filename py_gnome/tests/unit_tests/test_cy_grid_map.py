"""
unit tests cython wrapper

designed to be run with py.test
"""

import os
from gnome.cy_gnome.cy_grid_map import CyGridMap
from gnome.utilities.remote_data import get_datafile

import pytest

here = os.path.dirname(__file__)
cur_dir = os.path.join(here, 'sample_data', 'currents')

# def test_exceptions():
#     """
#     Test ValueError exception thrown if improper input arguments
#     """
#     with pytest.raises(ValueError):
#         cy_grid_map.CyGridMap()
#


@pytest.mark.slow
class TestGridMap():
    #gcm = cy_grid_map.CyGridMap()

    def test_grid_map_curv(self):
        """
        Test a grid map - read and write out
        """
		#curvilinear grid
	gcm1 = CyGridMap()
        grid_map_file = get_datafile( os.path.join(cur_dir, 'ny_cg.nc') )
        gcm1.text_read(grid_map_file)

	#topology_file = os.path.join(cur_dir, 'ny_cg_top.dat')
	#self.gcm.export_topology(topology_file)
        #self.gcm.save_netcdf(netcdf_file)

        netcdf_file = os.path.join(cur_dir, 'ny_cg_top.nc')
        gcm1.save_netcdf(netcdf_file)

    def test_grid_map_tri(self):
        """
        Test a grid map - read and write out
        """
        #triangle grid
        gcm2 = CyGridMap()
        grid_map_file = get_datafile( os.path.join(cur_dir, 'ChesBay.nc'))
        gcm2.text_read(grid_map_file)
        
        #topology_file = os.path.join( cur_dir, 'chesbay_top.dat')
        #self.gcm2.export_topology(topology_file)
        
        netcdf_file = os.path.join( cur_dir, 'ChesBayTop.nc')
        gcm2.save_netcdf( netcdf_file)


if __name__ == "__main__":
    """
    This makes it easy to use this file to debug the lib_gnome DLL
    through Visual Studio
    """
    tgm = TestGridMap()
    tgm.test_grid_map_curv()
    tgm.test_grid_map_tri()
