#!/usr/bin/env python

"""
test code for the tamoc module

so far, only dummy input, etc.
"""


from datetime import datetime
from datetime import timedelta
import numpy as np


from gnome.tamoc import tamoc


def init_spill():
    return tamoc.TamocSpill(release_time=datetime(2016, 8, 12, 12),
                            start_position=(-76.0, 28.0, 1000),
                            num_elements=10000,
                            end_release_time=datetime(2016, 12, 12, 12),
                            name='TAMOC plume',
                            TAMOC_interval=24,
                            on=True,)


def test_TamocDroplet():
    """
    Really jsut a way to run it

    This is a dummy object anyway -- won't always be needed
    """
    td = tamoc.TamocDroplet(radius=1e-5)

    assert td.radius == 1e-5
    # jsut mamking sure they exist
    assert td.mass_flux >= 0.0
    assert td.radius >= 0.0  # zero radius may be OK -- for dissolved?
    pos = td.position
    assert len(pos) == 3


def test_TamocSpill_init():
    """
    making sure it can be initialized
    """

    ts = tamoc.TamocSpill(release_time=datetime(2016, 8, 12, 12),
                          start_position=(28, -76, 2000),
                          num_elements=10000,
                          end_release_time=datetime(2016, 12, 12, 12),
                          name='TAMOC plume',
                          TAMOC_interval=24,
                          on=True,)

    assert ts.on


def test_fake_tamoc_results():
    """
    this is probably temporary, but useful for testing anyway

    not much tested here, but at least it runs.
    """

    results = tamoc.fake_tamoc_results(12)

    assert len(results) == 12
    assert np.isclose(sum([drop.mass_flux for drop in results]), 10.0)


def test_TamocSpill_run_tamoc():
    rt = datetime(2016, 8, 12, 12)
    ts = init_spill()

    drops = ts.run_tamoc(rt, 900)
    drops2 = ts.run_tamoc(rt + timedelta(hours=23), 900)
    assert drops is drops2
    drops3 = ts.run_tamoc(rt + timedelta(hours=25), 900)
    assert drops is not drops3
    drops4 = ts.run_tamoc(rt + timedelta(hours=25), 900)
    assert drops4 is drops3


def test_TamocSpill_num_elements_to_release():
    ts = init_spill()

    ts.end_release_time = ts.release_time + timedelta(hours=10)
    num_elem = ts.num_elements_to_release(ts.release_time, 3600)
    assert num_elem == 1000


def test_TamocSpill_set_newparticle_values():

    # release 1k particles over 1 hour, at an overall rate of 10kg/sec
    data_arrays = {}
    data_arrays['mass'] = np.zeros((1000))
    data_arrays['positions'] = np.zeros((1000, 3))
    data_arrays['init_mass'] = np.zeros((1000))

    ts = init_spill()
    ts.end_release_time = ts.release_time + timedelta(hours=10)
    num_elem = ts.num_elements_to_release(ts.release_time, 3600)
    ts.set_newparticle_values(num_elem, ts.release_time, 3600, data_arrays)
    assert abs(ts.amount_released - 36000) > 0.00001

    # eventually we will need to test several release scenarios
    data_arrays = {}
    data_arrays['mass'] = np.zeros((1000))
    data_arrays['positions'] = np.zeros((1000, 3))
    data_arrays['init_mass'] = np.zeros((1000))

    ts = init_spill()
    ts.tamoc_interval = 1
    ts.end_release_time = ts.release_time + timedelta(hours=10)
    num_elem = ts.num_elements_to_release(ts.release_time, 3600)
    ts.set_newparticle_values(num_elem, ts.release_time, 3600, data_arrays)
    assert abs(ts.amount_released - 36000) > 0.00001
#     assert data_arrays['mass'][0] == 36


if __name__ == '__main__':
    test_TamocSpill_run_tamoc()
    test_TamocSpill_num_elements_to_release()
    test_TamocSpill_set_newparticle_values()
