"""
tests of how a continuous spill releases in an actual model run

See: https://gitlab.orr.noaa.gov/gnome/pygnome/-/issues/75

"""

import pytest

from datetime import datetime

from gnome.model import Model
from gnome.spills.spill import surface_point_line_spill
from gnome.outputters.memory_outputter import MemoryOutputter

import gnome.scripting as gs


START_TIME = datetime(2020, 1, 1, 0, 0)
TIME_STEP = gs.hours(1)
NUM_ELEMENTS = 100


@pytest.fixture
def empty_model():
    model = Model(
        time_step=TIME_STEP,
        start_time=START_TIME,
        duration=gs.hours(6),
        weathering_substeps=1,
        map=None,
        uncertain=False,
        cache_enabled=False,
        mode=None,
        make_default_refs=True,
        location=[],
        environment=[],
        outputters=[],
        movers=[],
        weatherers=[],
        spills=[],
        uncertain_spills=[],
        )

    model.outputters += MemoryOutputter()

    return model


@pytest.fixture
def inst_spill():
    spill = surface_point_line_spill(
        num_elements=NUM_ELEMENTS,
        start_position=(0, 0, 0),
        release_time=START_TIME,
        end_release_time=None,
        amount=100,
        units='kg',
        name='test spill',
        )

    return spill


@pytest.fixture
def cont_spill():
    spill = surface_point_line_spill(
        num_elements=NUM_ELEMENTS,
        start_position=(0, 0, 0),
        release_time=START_TIME,
        end_release_time=START_TIME + 4 * TIME_STEP,
        amount=100,
        units='kg',
        name='test spill',
        )

    return spill


def test_above_surface_warning(empty_model, inst_spill):
    """
    Spill above the surface
    """
    model = empty_model
    inst_spill.release.start_position[2]=-1
    model.spills += inst_spill
    with pytest.warns(UserWarning,match='Depth of spill is negative'):
        model.check_inputs()


def test_instantaneous(empty_model, inst_spill):
    """
    That shouldn't get broken!
    """
    model = empty_model
    model.spills += inst_spill
    model.full_run()
    results = model.outputters[0]

    data = results.data_buffer.certain

    # there should be 100 Elements at all times
    for ts in data:
        assert len(ts['mass']) == NUM_ELEMENTS


def test_instantaneous_early_start(empty_model, inst_spill):
    """
    That shouldn't get broken!
    """
    model = empty_model
    model.start_time = model.start_time - TIME_STEP
    model.spills += inst_spill
    model.full_run()
    results = model.outputters[0]

    data = results.data_buffer.certain

    # there should be 0 Elements at time zero
    assert len(data[0]['mass']) == 0
    # there should be 100 Elements at all other times
    for ts in data[1:]:
        assert len(ts['mass']) == NUM_ELEMENTS


def test_continuous_aligned_start(empty_model, cont_spill):
    """
    This behavior may change
    """
    model = empty_model
    model.spills += cont_spill
    model.full_run()
    results = model.outputters[0]

    data = results.data_buffer.certain

    # there should be 0 of the Elements at time zero
    # then go from there
    assert len(data[0]['mass']) == 0
    assert len(data[1]['mass']) == NUM_ELEMENTS / 4
    assert len(data[2]['mass']) == NUM_ELEMENTS / 4 * 2
    assert len(data[3]['mass']) == NUM_ELEMENTS / 4 * 3
    assert len(data[4]['mass']) == NUM_ELEMENTS


def test_continuous_unaligned_start(empty_model, cont_spill):
    """
    Model starts one timestep before the spill
    """
    model = empty_model
    model.start_time = model.start_time - TIME_STEP
    model.spills += cont_spill
    model.full_run()
    results = model.outputters[0]

    data = results.data_buffer.certain

    # there should be 100 Elements at all times
    # there should be 0 Elements at time zero
    assert len(data[0]['mass']) == 0
    assert len(data[1]['mass']) == 0
    assert len(data[2]['mass']) == NUM_ELEMENTS / 4 * 1
    assert len(data[3]['mass']) == NUM_ELEMENTS / 4 * 2
    assert len(data[4]['mass']) == NUM_ELEMENTS / 4 * 3
    assert len(data[5]['mass']) == NUM_ELEMENTS


def test_one_element_per_timestep():
    """
    Should get exactly one per timestep
    """
    model = Model(
        time_step=gs.hours(1),
        start_time='2022-12-10T00:00',
        duration=gs.hours(12),
    )
    model.outputters += MemoryOutputter()

    model.spills += surface_point_line_spill(
        num_elements=10,
        start_position=(0, 0, 0),
        release_time=model.start_time,
        end_release_time=model.start_time + 10 * gs.seconds(model.time_step),
        )
    model.full_run()
    results = model.outputters[0]

    data = results.data_buffer.certain

    # there should be 0 of the Elements at time zero
    # then one per timestep
    for i in range(11):
        assert len(data[i]['mass']) == i

    for i in range(11, 13):
        assert len(data[i]['mass']) == 10


def test_two_elements_per_timestep():
    """
    with the half-step release scheme, this should also work.
    """
    model = Model(
        time_step=gs.hours(1),
        start_time='2022-12-10T00:00',
        duration=gs.hours(12),
    )
    model.outputters += MemoryOutputter()

    model.spills += surface_point_line_spill(
        num_elements=20,
        start_position=(0, 0, 0),
        release_time=model.start_time,
        end_release_time=model.start_time + 10 * gs.seconds(model.time_step),
        )
    model.full_run()
    results = model.outputters[0]

    data = results.data_buffer.certain

    # there should be 0 of the Elements at time zero
    # then one per timestep
    for i in range(11):
        assert len(data[i]['mass']) == i*2

    # should max out
    for i in range(11, 13):
        assert len(data[i]['mass']) == 20


@pytest.mark.parametrize("num_per", [0.5, 1.0, 1.5, 2.0, 3.3])
def test_fractional_elements_per_timestep(num_per):
    """
    what happens with 1.5 per timestep?
    """
    model = Model(
        time_step=gs.hours(1),
        start_time='2022-12-10T00:00',
        duration=gs.hours(12),
    )
    model.outputters += MemoryOutputter()

    model.spills += surface_point_line_spill(
        num_elements= int(10 * num_per),
        start_position=(0, 0, 0),
        release_time=model.start_time,
        end_release_time=model.start_time + 10 * gs.seconds(model.time_step),
        )
    if num_per < 1:  # fewer than one per timestep is not allowed
        with pytest.raises(ValueError):
            model.full_run()
    else:
        model.full_run()
        results = model.outputters[0]

        data = results.data_buffer.certain

        # there should be 0 of the Elements at time zero
        # then one per timestep
        for i in range(11):
            print(i, len(data[i]['mass']))
            assert len(data[i]['mass']) <= i * num_per
            assert len(data[i]['mass']) >= i * num_per - 1

        # should max out
        for i in range(11, 13):
            assert len(data[i]['mass']) == 10 * num_per


