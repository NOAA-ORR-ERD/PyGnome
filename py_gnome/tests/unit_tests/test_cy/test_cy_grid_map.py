"""
unit tests cython wrapper

designed to be run with py.test
"""





import os

import pytest

from gnome.cy_gnome.cy_grid_map import CyGridMap

from ..conftest import testdata


@pytest.mark.slow
class TestGridMap(object):

    # gcm = cy_grid_map.CyGridMap()

    def test_grid_map_curv(self, dump_folder):
        """
        Test a grid map - read and write out
        """

        # curvilinear grid

        gcm1 = CyGridMap()
        grid_map_file = testdata['GridMap']['curr']
        gcm1.text_read(grid_map_file)

        # topology_file = os.path.join(cur_dir, 'ny_cg_top.dat')
        # self.gcm.export_topology(topology_file)
        # self.gcm.save_netcdf(netcdf_file)

        netcdf_file = os.path.join(dump_folder, 'ny_cg_top.nc')
        gcm1.save_netcdf(netcdf_file)

    def test_grid_map_tri(self, dump_folder):
        """
        Test a grid map - read and write out
        """

        # triangle grid

        gcm2 = CyGridMap()
        grid_map_file = testdata['c_GridCurrentMover']['curr_tri']
        gcm2.text_read(grid_map_file)

        # topology_file = os.path.join( cur_dir, 'chesbay_top.dat')
        # self.gcm2.export_topology(topology_file)

        netcdf_file = os.path.join(dump_folder, 'ChesBayTop.nc')
        gcm2.save_netcdf(netcdf_file)

    def test_grid_map_cats(self, dump_folder):
        """
        Test a grid map - read and write out
        """

        # triangle grid

        gcm3 = CyGridMap()
        grid_map_file = testdata['GridMap']['BigCombinedwMap']
        gcm3.text_read(grid_map_file)

        topology_file = os.path.join(dump_folder, 'BigCombinedTop.dat')
        gcm3.export_topology(topology_file)

        # netcdf_file = os.path.join(cur_dir, 'BigCombinedTop.nc')
        # gcm3.save_netcdf(netcdf_file)


if __name__ == '__main__':
    tgm = TestGridMap()
    tgm.test_grid_map_curv()
    tgm.test_grid_map_tri()
