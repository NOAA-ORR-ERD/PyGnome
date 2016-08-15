#!/usr/bin/env python

"""
test code for the tamoc module

so far, only dummy input, etc.
"""


from datetime import datetime

import numpy as np

from gnome.tamoc import tamoc


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
                          end_release_time=datetime(2016, 2, 12, 12),
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

