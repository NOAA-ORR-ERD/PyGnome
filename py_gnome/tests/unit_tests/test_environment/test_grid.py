



import os

import pytest
import netCDF4 as nc

from gnome.environment.gridded_objects_base import PyGrid, Grid_U, Grid_S
from gnome.utilities.remote_data import get_datafile

import pprint as pp


@pytest.fixture()
def sg_data():
    base_dir = os.path.dirname(__file__)
    filename = get_datafile(os.path.join(base_dir,
                                         'sample_data',
                                         'currents',
                                         'tbofs_example.nc'))

    return filename, nc.Dataset(filename)


@pytest.fixture()
def sg_topology():
    return {'node_lon': 'lonc',
            'node_lat': 'latc'}


@pytest.fixture()
def sg(sg_data, sg_topology):
    return PyGrid.from_netCDF(filename=sg_data[0], dataset=sg_data[1],
                              grid_topology=sg_topology)


@pytest.fixture()
def ug_data():
    base_dir = os.path.dirname(__file__)
    filename = get_datafile(os.path.join(base_dir,
                                         'sample_data',
                                         'currents',
                                         'ChesBay.nc'))

    return filename, nc.Dataset(filename)


@pytest.fixture()
def ug_topology():
    pass


@pytest.fixture()
def ug(ug_data, ug_topology):
    return PyGrid.from_netCDF(filename=ug_data[0], dataset=ug_data[1],
                              grid_topology=ug_topology)


class TestPyGrid_S(object):
    def test_construction(self, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg = Grid_S.from_netCDF(filename, dataset,
                                grid_topology=grid_topology)
        assert sg.filename == filename
        assert sg.grid_topology == grid_topology

        sg2 = Grid_S.from_netCDF(filename)
        assert sg2.filename == filename

        sg3 = PyGrid.from_netCDF(filename, dataset,
                                 grid_topology=grid_topology)
        print(sg3.shape)
        assert sg == sg3

        sg4 = PyGrid.from_netCDF(filename)
        print(sg4.shape)
        assert sg2 == sg4

    def test_serialize(self, sg, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology

        sg2 = Grid_S.from_netCDF(filename, dataset,
                                 grid_topology=grid_topology)

        print(sg.serialize()['filename'])
        print(sg2.serialize()['filename'])
        assert sg.serialize()['filename'] == sg2.serialize()['filename']

    def test_deserialize(self, sg):
        d_sg = Grid_S.deserialize(sg.serialize())

        pp.pprint(sg.serialize())
        pp.pprint(d_sg.serialize())
        assert sg == d_sg


class TestPyGrid_U(object):
    def test_construction(self, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology

        ug = Grid_U.from_netCDF(filename, dataset, grid_topology=grid_topology)
        # assert ug.filename == filename
        # assert isinstance(ug.node_lon, nc.Variable)
        # assert ug.node_lon.name == 'lonc'

        ug2 = Grid_U.from_netCDF(filename)
        assert ug2.filename == filename
        assert ug2.grid_topology == grid_topology
        # assert isinstance(ug2.node_lon, nc.Variable)
        # assert ug2.node_lon.name == 'lon'

        ug3 = PyGrid.from_netCDF(filename, dataset,
                                 grid_topology=grid_topology)
        ug4 = PyGrid.from_netCDF(filename)
        print(ug3.shape)
        print(ug4.shape)
        assert ug == ug3
        assert ug2 == ug4

    def test_serialize(self, ug, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology

        ug2 = Grid_U.from_netCDF(filename, dataset,
                                 grid_topology=grid_topology)
        assert ug.serialize()['filename'] == ug2.serialize()['filename']

    def test_deserialize(self, ug, ug_data, ug_topology):
        d_ug = Grid_U.deserialize(ug.serialize())

        pp.pprint(ug.serialize())
        pp.pprint(d_ug.serialize())

        assert ug == d_ug
