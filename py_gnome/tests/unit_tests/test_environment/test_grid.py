import os
import pytest
import netCDF4 as nc
from gnome.environment.gridded_objects_base import Grid, Grid_U, Grid_S
from gnome.utilities.remote_data import get_datafile
import pprint as pp

@pytest.fixture()
def sg_data():
    base_dir = os.path.dirname(__file__)
    s_data = os.path.join(base_dir, 'sample_data')
    filename = os.path.join(s_data, 'currents')
    filename = get_datafile(os.path.join(filename, 'tbofs_example.nc'))
    return filename, nc.Dataset(filename)

@pytest.fixture()
def sg_topology():
    return {'node_lon': 'lonc',
            'node_lat': 'latc'}

@pytest.fixture()
def sg():
    return Grid.from_netCDF(sg_data()[0], sg_data()[1], grid_topology=sg_topology())

@pytest.fixture()
def ug_data():
    base_dir = os.path.dirname(__file__)
    s_data = os.path.join(base_dir, 'sample_data')
    filename = os.path.join(s_data, 'currents')
    filename = get_datafile(os.path.join(filename, 'ChesBay.nc'))
    return filename, nc.Dataset(filename)

@pytest.fixture()
def ug_topology():
    pass

@pytest.fixture()
def ug():
    return Grid.from_netCDF(ug_data()[0], ug_data()[1], grid_topology=ug_topology())

class TestPyGrid_S:
    def test_construction(self, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg = Grid_S.from_netCDF(filename, dataset, grid_topology=grid_topology)
        assert sg.filename == filename

        sg2 = Grid_S.from_netCDF(filename)
        assert sg2.filename == filename

        sg3 = Grid.from_netCDF(filename, dataset, grid_topology=grid_topology)
        sg4 = Grid.from_netCDF(filename)
        print sg3.shape
        print sg4.shape
        assert sg == sg3
        assert sg2 == sg4

    def test_serialize(self, sg, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg2 = Grid_S.from_netCDF(filename, dataset, grid_topology=grid_topology)
#         pytest.set_trace()
        print sg.serialize()['filename']
        print sg2.serialize()['filename']
        assert sg.serialize()['filename'] == sg2.serialize()['filename']

    def test_deserialize(self, sg, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg2 = Grid_S.from_netCDF(filename, dataset, grid_topology=grid_topology)
        d_sg = Grid_S.new_from_dict(sg.serialize())

        pp.pprint(sg.serialize())
        pp.pprint(d_sg.serialize())

        assert sg.name == d_sg.name
#         fn1 = 'C:\\Users\\jay.hennen\\Documents\\Code\\pygnome\\py_gnome\\scripts\\script_TAP\\arctic_avg2_0001_gnome.nc'
#         fn2 = 'C:\\Users\\jay.hennen\\Documents\\Code\\pygnome\\py_gnome\\scripts\\script_columbia_river\\COOPSu_CREOFS24.nc'
#         sg = PyGrid.from_netCDF(fn1)
#         ug = PyGrid.from_netCDF(fn2)
#         sg.save(".\\testzip.zip", name="testjson.json")
#         ug.save_as_netcdf("./testug.nc")
#     #     sg.save_as_netcdf("./testug.nc")
#         k = PyGrid_U.from_netCDF("./testug.nc")
#         k2 = PyGrid_U.from_netCDF(fn2)
#         c1 = PyGrid_S.from_netCDF(fn1)
#         ug4 = PyGrid.from_netCDF("./testug.nc")
#         sg2 = PyGrid.from_netCDF("./testsg.nc")
#         ug2 = PyGrid.from_netCDF(fn2)
#
#         ug3 = PyGrid.new_from_dict(ug.serialize(json_='save'))

class TestPyGrid_U:
    def test_construction(self, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology
        ug = Grid_U.from_netCDF(filename, dataset, grid_topology=grid_topology)
#         assert ug.filename == filename
#         assert isinstance(ug.node_lon, nc.Variable)
#         assert ug.node_lon.name == 'lonc'

        ug2 = Grid_U.from_netCDF(filename)
        assert ug2.filename == filename
#         assert isinstance(ug2.node_lon, nc.Variable)
#         assert ug2.node_lon.name == 'lon'

        ug3 = Grid.from_netCDF(filename, dataset, grid_topology=grid_topology)
        ug4 = Grid.from_netCDF(filename)
        print ug3.shape
        print ug4.shape
        assert ug == ug3
        assert ug2 == ug4

    def test_serialize(self, ug, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology
        ug2 = Grid_U.from_netCDF(filename, dataset, grid_topology=grid_topology)
        assert ug.serialize()['filename'] == ug2.serialize()['filename']

    def test_deserialize(self, ug, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology
        ug2 = Grid_U.from_netCDF(filename, dataset, grid_topology=grid_topology)
        d_ug = Grid_U.new_from_dict(ug.serialize())

        pp.pprint(ug.serialize())
        pp.pprint(d_ug.serialize())

        assert ug.name == d_ug.name
