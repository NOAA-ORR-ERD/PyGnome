"""
tests for code that reads teh old gridcur format
"""

from pathlib import Path
from datetime import datetime

import numpy as np

import pytest

from ..conftest import sample_sc_release

from gnome.basic_types import oil_status, status_code_type  # .in_water
from gnome.environment import gridcur

from gnome.movers import CurrentMover
from gnome.spills import grid_spill

import gnome.scripting as gs

test_data_dir = Path(__file__).parent / "sample_data"
test_output_dir = Path(__file__).parent / "sample_output"


# NOTE: results are stored in git
#       this couuld be used to update example files
def make_gridcur(filename, location="cells"):
    """
    this makes a gridcur file, with a quarter circle of currents

    Data in the cells

    (off the coast of Alabama)
    """
    lat = np.linspace(29.0, 30.0, 11)
    lon = np.linspace(-88.0, -86.0, 21)

    times = [datetime(2020, 7, 14, 12, 0),
             datetime(2020, 7, 14, 18, 0),
             datetime(2020, 7, 15, 0, 0),
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

    gridcur.write_gridcur(filename, data_type, units, times, lon, lat, data_u, data_v)


# create a test gridcur file:
CELL_EXAMPLE = test_data_dir / "example_gridcur_on_cells.cur"
# make_gridcur(CELL_EXAMPLE)

NODE_EXAMPLE = test_data_dir / "example_gridcur_on_nodes.cur"
# make_gridcur(NODE_EXAMPLE, "nodes")


def test_read_single_cell_center():
    data_type, units, times, lon, lat, data_u, data_v = gridcur.read_file(
        test_data_dir / "grid_cur_1x1.cur")

    assert data_type == 'currents'
    assert units == 'KNOTS'
    assert times == [datetime(2002, 1, 30, 1, 0)]  # 30 1 2002 1 0
    assert np.array_equal(lon, [12, 15])
    assert np.array_equal(lat, [44, 46])

    # print(data_u)
    # print(data_v)

    assert len(data_u) == 1
    assert len(data_v) == 1
    assert np.array_equal(data_u[0], np.array([[0.092388]]))
    assert np.array_equal(data_v[0], np.array([[-.0382683]]))


def test_read_single_cell_nodes():
    data_type, units, times, lon, lat, data_u, data_v = gridcur.read_file(
        test_data_dir / "grid_cur_2x2.cur")
    assert data_type == 'currents'
    assert units == 'm/s'
    assert times == [datetime(2012, 1, 30, 1, 15)]  # 30 1 2002 1 0
    assert np.array_equal(lon, [-70, -67])
    assert np.array_equal(lat, [45, 47])

    # print(data_u)
    # print(data_v)

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
    data_type, units, times, lon, lat, data_u, data_v = gridcur.read_file(
        test_data_dir / CELL_EXAMPLE)

    assert data_type == 'currents'
    assert units == 'm/s'
    assert times == [datetime(2020, 7, 14, 12, 0),
                     datetime(2020, 7, 14, 18, 0),
                     datetime(2020, 7, 15, 0, 0),
                     ]

    assert np.array_equal(lon, np.linspace(-88.0, -86.0, 21))
    assert np.array_equal(lat, np.linspace(29.0, 30.0, 11))

    assert len(data_u) == 3
    assert len(data_v) == 3
    for U, V in zip(data_u, data_v):
        U.shape == (20, 10)
        V.shape == (20, 10)
    # A few values, just to be sure, but ...
    assert data_u[0][0, 0] == 0.5
    assert data_v[0][0, 0] == 0.0

    assert data_u[-1][19, 9] == 0.385278
    assert data_v[-1][19, 9] == 0.813364

def test_read_nodes_multiple_times():
    data_type, units, times, lon, lat, data_u, data_v = gridcur.read_file(
        test_data_dir / NODE_EXAMPLE)

    assert data_type == 'currents'
    assert units == 'm/s'
    assert times == [datetime(2020, 7, 14, 12, 0),
                     datetime(2020, 7, 14, 18, 0),
                     datetime(2020, 7, 15, 0, 0),
                     ]

    assert np.array_equal(lon, np.linspace(-88.0, -86.0, 21))
    assert np.array_equal(lat, np.linspace(29.0, 30.0, 11))

    assert len(data_u) == 3
    assert len(data_v) == 3
    for U, V in zip(data_u, data_v):
        U.shape == (21, 11)
        V.shape == (21, 11)
    # A few values, just to be sure, but ...
    assert data_u[0][0, 0] == 0.5
    assert data_v[0][0, 0] == 0.0

    assert data_u[-1][20, 10] == 0.402492
    assert data_v[-1][20, 10] == 0.804984


def test_GridR_node():
    # NOTE: The value-on-the-nodes version is the only one supported
    cur = gridcur.from_gridcur(filename=test_data_dir / NODE_EXAMPLE)

    print(cur.grid.nodes)

    # print(cur)
    points = np.array(((-88.0, 29.0, 0.0), ))
    # points = np.array(((29.0, -88.0, 0.0), ))
    times = (datetime(2020, 7, 14, 12))
    result = cur.at(points, times)

    print(f"{result=}")

    assert np.allclose(result, np.array([[0.5, 0.0, 0.0]]))


def test_make_mover_from_gridcur():
    """
    make a mover from a gridcur
    """
    current = gridcur.from_gridcur(filename=test_data_dir / NODE_EXAMPLE)

    mover = CurrentMover(current=current)

    assert mover.data_start == datetime(2020, 7, 14, 12)
    assert mover.data_stop == datetime(2020, 7, 15, 0)


def test_mover_get_move():
    current = gridcur.from_gridcur(filename=test_data_dir / NODE_EXAMPLE)
    mover = CurrentMover(current=current)

    # create a minimal spill container
    model_time_datetime = datetime(2020, 7, 14, 12)
    time_step = gs.minutes(30).total_seconds()
    initial_positions = np.array([(-88.0, 29.0, 0.0),
                                  (-87.0, 29.5, 0.0),  # near middle of grid
                                  (-89.0, 27.5, 0.0),  # outside the grid
                                  ])
    status_codes = np.array([oil_status.in_water,  # this is the default
                             oil_status.in_water,
                             oil_status.in_water],
                             dtype=status_code_type)
    num_le = 3
    pSpill = sample_sc_release(num_le, (0.,0.,0), model_time_datetime)
    pSpill['status_codes'] = status_codes
    pSpill['positions'] = initial_positions
    deltas = mover.get_move(pSpill, time_step, model_time_datetime)

    print("deltas are:", deltas)

    # not much of test, but at least its doing something
    assert not np.array_equal(deltas[0], [0.0, 0.0, 0.0])
    assert not np.array_equal(deltas[1], [0.0, 0.0, 0.0])

    # the last one should be zeros -- out of bounds
    assert np.array_equal(deltas[2], [0.0, 0.0, 0.0])


def test_cell_not_supported():
    with pytest.raises(NotImplementedError):
        current = gridcur.from_gridcur(filename=test_data_dir / CELL_EXAMPLE)


def test_in_model():

    current = gridcur.from_gridcur(filename=test_data_dir / NODE_EXAMPLE)
    mover = CurrentMover(current=current)

    start_time = "2020-07-14T12:00"
    model = gs.Model(time_step=gs.hours(1),
                     start_time=start_time,
                     duration=gs.hours(12),
                     uncertain=False)
    model.movers += mover

    spill = grid_spill(bounds=((-88.0, 29.0),
                               (-86.0, 30.0),
                               ),
                       resolution=20,
                       release_time=start_time,
                       )
    model.spills += spill
    renderer = gs.Renderer(output_dir=test_output_dir,
                           image_size=(800, 600),
                           viewport=((-88.0, 29.0),
                                     (-86.0, 30.0),
                                     ),
                           )
    model.outputters += renderer

    model.full_run()

