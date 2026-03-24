"""
tests for the grid_spill
"""
from pathlib import Path
import numpy as np

from datetime import datetime, timedelta

from gnome.spills.release import GridRelease
from gnome.spills.spill import grid_spill
from gnome.spill_container import SpillContainer
from gnome.basic_types import oil_status

import gnome.scripting as gs

SAMPLE_DATA = Path(__file__).parent.parent / "sample_data"
OUTPUT_PATH = Path(__file__).parent / "temp_output"

def test_compute_resolution():
    bounds = np.array(((-125.0, 48.0), (-124.6, 48.5)))

    resolution = GridRelease._compute_resolution(bounds, 100)

    print(resolution)

    # The total resolution should be within 1 of the provided value
    assert abs(np.sqrt(resolution[0] * resolution[1]) - 100) <= 1
    assert resolution[0] < resolution[1]


def test_grid_release_init():
    """
    fixme: test something more here? like actually have the release do its thing?

    Though that is tested elsewhere.
    """
    bounds = ((0, 10), (2, 12))
    release = GridRelease("2025-03-13 12:00",
                          bounds,
                          resolution = (3,3))

    assert np.all(release.centroid == (1, 11, 0))

def test_grid_release_single_resolution():
    """
    when you pass in a single value for resolution
    you should get a resolution that matches the aspect ratio
    in projected space
    """
    bounds = ((-125.0, 48.0), (-124.6, 48.5))

    release = GridRelease("2025-03-13 12:00",
                          bounds,
                          resolution=50)

    resolution = release.resolution
    # The total resolution should be within 1 of the provided value
    assert abs(np.sqrt(resolution[0] * resolution[1]) - 50) <= 1
    assert resolution[0] < resolution[1]

    assert np.all(release.centroid == (-124.8, 48.25, 0))

def test_grid_spill_mass():
    """
    make sure setting the mass works
    """
    # North part of Long Island Sound
    bounds = ((-73, 41.1), (-72.6, 41.3))
    timestep = gs.minutes(15)
    sp = grid_spill(bounds,
                    resolution=(20, 10),
                    release_time="2025-03-13 12:00",
                    # substance=None,
                    amount=300.,
                    units='kg',
                    # on=True,
                    # water=None,
                    # windage_range=None,
                    # windage_persist=None,
                    # name='Surface Grid Spill'
                    )

    assert sp.amount == 300.0

    sc = SpillContainer()
    sc.spills += sp
    sc.prepare_for_model_run(array_types=sp.array_types)
    sp.prepare_for_model_run(timestep.total_seconds())

    # call release at release time
    model_time = sp.release_time
    to_rel = sp.release_elements(sc, model_time, model_time + timestep)

    assert to_rel == 200

    mass = sc['mass']
    print(mass)
    assert np.allclose(mass.sum(), 300.0)


def test_grid_spill_release():
    # North part of Long Island Sound
    bounds = ((-73, 41.1), (-72.6, 41.3))
    timestep = gs.minutes(15)
    sp = grid_spill(bounds,
                    resolution=20,
                    release_time="2025-03-13 12:00",
                    # substance=None,
                    # amount=400.,
                    # units='kg',
                    # on=True,
                    # water=None,
                    # windage_range=None,
                    # windage_persist=None,
                    # name='Surface Grid Spill'
                    )

    sc = SpillContainer()
    sc.spills += sp
    sc.prepare_for_model_run(array_types=sp.array_types)
    sp.prepare_for_model_run(timestep.total_seconds())
    # call release before release time
    model_time = sp.release_time - 2 * timestep
    to_rel = sp.release_elements(sc, model_time, model_time + timestep)

    assert to_rel == 0

    # call release at release time
    model_time = sp.release_time
    to_rel = sp.release_elements(sc, model_time, model_time + timestep)

    assert to_rel == 400

    positions = sc['positions']
    assert np.all(positions.max(axis=0)[:2] == np.array(bounds).max(axis=0))


def test_grid_spill_model_mark_on_land():
    """
    tests that when a grid spill creates elements on land that they initially
    get marked as on land.

    spill start time is model start time.

    This is more of a test of the model, but what can you do?
    """
    # North part of Long Island Sound
    bounds = ((-73, 41.1), (-72.6, 41.3))
    timestep = gs.minutes(15)

    model = gs.Model(start_time="2026-03-06T12:00",
                     time_step=timestep,
                     duration=gs.hours(1),
                     )

    sp = grid_spill(bounds,
                    resolution=10,
                    release_time=model.start_time
                    )

    model.map = gs.MapFromBNA(SAMPLE_DATA / 'long_island_sound/LongIslandSoundMap.BNA')
    model.spills += sp

    model.movers += gs.constant_point_wind_mover(10, 45, units='m/s')

    model.outputters += gs.NetCDFOutput(OUTPUT_PATH / 'grid_results.nc')

    step = model.step()
    scs = model.get_spill_property('status_codes')
    # not sure exatly how many it should be!
    assert np.sum(scs == oil_status.on_land) > 0

    model.rewind()
    model.full_run()


def test_grid_spill_model_mark_on_land_delayed_start():
    """
    tests that when a grid spill creates elements on land that they initially
    get marked as on land.

    Spill start time is after model start time.

    This is more of a test of the model, but what can you do?
    """
    # North part of Long Island Sound
    bounds = ((-73, 41.1), (-72.6, 41.3))
    timestep = gs.minutes(15)

    model = gs.Model(start_time="2026-03-06T12:00",
                     time_step=timestep,
                     duration=gs.hours(1),
                     )

    sp = grid_spill(bounds,
                    resolution=10,
                    release_time=model.start_time + gs.minutes(30)
                    )

    model.map = gs.MapFromBNA(SAMPLE_DATA / 'long_island_sound/LongIslandSoundMap.BNA')
    model.spills += sp

    model.movers += gs.constant_point_wind_mover(10, 45, units='m/s')

    model.outputters += gs.NetCDFOutput(OUTPUT_PATH / 'grid_results_delayed.nc')

    for step in model:
        print(step)
        scs = model.get_spill_property('status_codes')
        if len(scs):  # if there are any, some should be on land
            assert np.sum(scs == oil_status.on_land) > 0


def test_grid_spill_model_longer_run():
    """
    tests that a grid spill creates elements when it should,
    and none afterward more.

    Spill start time is after model start time.

    This is an integration test with the model, but what can you do?
    """
    # North part of Long Island Sound
    bounds = ((-73, 41.1), (-72.6, 41.3))
    timestep = gs.minutes(15)

    model = gs.Model(start_time="2026-03-06T12:00",
                     time_step=timestep,
                     duration=gs.hours(1),
                     )

    sp = grid_spill(bounds, resolution=10,
                    release_time=model.start_time + gs.minutes(30))
    model.spills += sp
    for step in model:
        print(step)
        scs = model.get_spill_property('positions')
        print(f"total_elements:", len(scs))
        if model.model_time < sp.release_time:
            assert len(scs) == 0
        else:
            assert len(scs) == sp.release.num_elements
