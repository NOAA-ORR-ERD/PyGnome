"""
tests for code that reads teh old gridcur format
"""



from pathlib import Path
from datetime import datetime

import numpy as np

import pytest

from gnome.environment.environment_objects import (GridCurrent,
                                                   FileGridCurrent,
                                                   )

import gnome.scripting as gs

test_data_dir = Path(__file__).parent / "sample_data"
test_output_dir = Path(__file__).parent / "sample_output"


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

    GridcurCurrent.write_gridcur(filename, data_type, units, times, lon, lat, data_u, data_v)


# create a test gridcur file:
CELL_EXAMPLE = test_data_dir / "example_gridcur_on_cells.cur"
# make_gridcur(CELL_EXAMPLE)

NODE_EXAMPLE = test_data_dir / "example_gridcur_on_nodes.cur"
# make_gridcur(NODE_EXAMPLE, "nodes")


def test_nonexistant_filename():
    with pytest.raises(ValueError):
        current = FileGridCurrent("non_existant_filename.cur")


def test_nonexistant_filename_nc():
    with pytest.raises(ValueError):
        current = FileGridCurrent("non_existant_filename.nc")


def test_gridcur_in_model():

    current = FileGridCurrent(test_data_dir / NODE_EXAMPLE)
    mover = gs.CurrentMover(current=current)

    start_time = "2020-07-14T12:00"
    model = gs.Model(time_step=gs.hours(1),
                     start_time=start_time,
                     duration=gs.hours(12),
                     uncertain=False)
    model.movers += mover

    spill = gs.grid_spill(bounds=((-88.0, 29.0),
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


def test_gridcur_serialize():
    """
    Can we persist one of these? and remake it from the persisted
    location?
    """
    filename = str(test_data_dir / NODE_EXAMPLE)
    current = FileGridCurrent(filename,
                              extrapolation_is_allowed=True,
                              )

    print("About to serialze")
    print(f"{current.filename}")
    serial = current.serialize()

    print(f"{serial}")

    current2 = FileGridCurrent.deserialize(serial)

    # really should test this better, but at least it didn't barf
    assert current2.extrapolation_is_allowed
    assert current2.filename == filename

@pytest.mark.skip
def test_netcdf_file():

    testfile = str(test_data_dir / 'tri_ring.nc')

    # create a GridCurrent
    current = FileGridCurrent(filename=testfile,
                              extrapolation_is_allowed=True,
                              )

    assert type(current) == FileGridCurrent
    assert isinstance(current, GridCurrent)

    # really should test more, but what?
    assert current.extrapolation_is_allowed

    assert current.units == "m/s"

    assert len(current.variables) == 3

@pytest.mark.skip
def test_netcdf_in_model():
    """
    the current object works with a model, and produces
    something in the rendered output

    correct? who knows, but it's running!
    """
    # Single timestep, so time doesn't matter.
    current = FileGridCurrent(str(test_data_dir / 'tri_ring.nc'))
    mover = gs.CurrentMover(current=current)

    start_time = "2020-07-14T12:00"
    model = gs.Model(time_step=gs.hours(1),
                     start_time=start_time,
                     duration=gs.hours(12),
                     uncertain=False)
    model.movers += mover

    # From the nodes of the netcdf file
    # In [8]: lat[:].min()
    # Out[8]: -0.9961946980917455

    # In [9]: lat[:].max()
    # Out[9]: 0.9961946980917455

    # In [10]: lon[:].min()
    # Out[10]: -0.9961946980917455

    # In [11]: lon[:].max()
    # Out[11]: 0.9961946980917455

    spill = gs.grid_spill(bounds=((-0.996, -0.996),
                                  (0.996, 0.996),
                                  ),
                          resolution=20,
                          release_time=start_time,
                          )
    model.spills += spill
    renderer = gs.Renderer(output_dir=test_output_dir / "netcdf",
                           image_size=(800, 600),
                           viewport=((-0.996, -0.996),
                                     (0.996, 0.996),
                                     ),
                           )
    model.outputters += renderer

    model.full_run()


