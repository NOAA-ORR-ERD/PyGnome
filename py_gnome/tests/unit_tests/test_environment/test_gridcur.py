"""
tests for code that reads teh old gridcur format
"""

from pathlib import Path
from datetime import datetime

import numpy as np

import pytest

from gnome.environment.gridcur import read_file, write_gridcur

test_data_dir = Path(__file__).parent / "sample_data"

def make_gridcur_on_nodes(filename):
    """
    This makes a gridcur file, with a quarter circle of currents

    Data on the nodes

    (off the coast of Alabama)

    """


def make_gridcur(filename, location="cells"):
    """
    this makes a gridcur file, with a quarter circle of currents

    Data in the cells

    (off the coast of Alabama)
    """
    lon = np.linspace(29.0, 30.0, 11)
    lat = np.linspace(-88.0, -86.0, 21)

    times = [datetime(2020, 7, 14, 12, 0),
             datetime(2020, 7, 14, 13, 0),
             datetime(2020, 7, 14, 14, 0),
             ]
    vel = 0.5
    units = "m/s"
    data_type = "currents"
    data_u = []
    data_v = []
    for i in range(len(times)):
        v = vel + (i * 0.2)
        if location == "cells":
            U = np.zeros((len(lon) - 1, len(lat) - 1), dtype=np.float32)
        elif location == "nodes":
            U = np.zeros((len(lon), len(lat)), dtype=np.float32)
        else:
            raise ValueError('location must be "cells" or "nodes"')
        V = U.copy()
        for row in range(U.shape[0]):
            for col in range(U.shape[1]):
                theta = np.arctan2(row, col)
                U[row, col] = v * np.cos(theta)
                V[row, col] = v * np.sin(theta)
        data_u.append(U)
        data_v.append(V)

    write_gridcur(filename, data_type, units, times, lon, lat, data_u, data_v)


# create a test gridcur file:
CELL_EXAMPLE = test_data_dir / "example_gridcur_on_cells.cur"
make_gridcur(CELL_EXAMPLE)

NODE_EXAMPLE = test_data_dir / "example_gridcur_on_nodes.cur"
make_gridcur(NODE_EXAMPLE, "nodes")



def test_read_single_cell_center():
    data_type, units, times, lon, lat, data_u, data_v = read_file(
        test_data_dir / "grid_cur_1x1.cur")

    assert data_type == 'currents'
    assert units == 'KNOTS'
    assert times == [datetime(2002, 1, 30, 1, 0)]  # 30 1 2002 1 0
    assert np.array_equal(lon, [12, 15])
    assert np.array_equal(lat, [44, 46])

    print(data_u)
    print(data_v)

    assert len(data_u) == 1
    assert len(data_v) == 1
    assert np.array_equal(data_u[0], np.array([[0.092388]]))
    assert np.array_equal(data_v[0], np.array([[-.0382683]]))


def test_read_single_cell_nodes():
    data_type, units, times, lon, lat, data_u, data_v = read_file(
        test_data_dir / "grid_cur_2x2.cur")
    assert data_type == 'currents'
    assert units == 'm/s'
    assert times == [datetime(2012, 1, 30, 1, 15)]  # 30 1 2002 1 0
    assert np.array_equal(lon, [-70, -67])
    assert np.array_equal(lat, [45, 47])

    print(data_u)
    print(data_v)

    assert len(data_u) == 1
    assert len(data_v) == 1
    assert np.array_equal(data_u[0],
                          np.array([
                              [0.092388, 0.092388],
                              [0.092388, 0.092388],
                          ]))
    assert np.array_equal(
        data_v[0], np.array([
            [-.0382683, -.0382683],
            [-.0382683, -.0382683],
        ]))


def test_read_cells_multiple_times():
    data_type, units, times, lon, lat, data_u, data_v = read_file(
        test_data_dir / CELL_EXAMPLE)

    assert data_type == 'currents'
    assert units == 'm/s'
    assert times == [datetime(2020, 7, 14, 12, 0),
                     datetime(2020, 7, 14, 13, 0),
                     datetime(2020, 7, 14, 14, 0),
                     ]

    assert np.array_equal(lon, np.linspace(29.0, 30.0, 11))
    assert np.array_equal(lat, np.linspace(-88.0, -86.0, 21))

    assert len(data_u) == 3
    assert len(data_v) == 3
    for U, V in zip(data_u, data_v):
        U.shape == (10, 20)
        V.shape == (10, 20)
        # assert np.array_equal(data_u[0], np.array([[0.092388]]))
        # assert np.array_equal(data_v[0], np.array([[-.0382683]]))



