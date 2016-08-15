#!/usr/bin/env python

"""
test code for the tamoc module

so far, only dummy input, etc.
"""


from datetime import datetime
from datetime import timedelta


from gnome.tamoc import tamoc


def test_TamocDroplet():
    """
    Really jsut a way to run it

    This is a dummy object anyway -- won't always be needed
    """
    td = tamoc.TamocDroplet(radius=1e-5)

    assert td.radius == 1e-5


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

def test_TamocSpill_run():
    rt = datetime(2016, 8, 12, 12)
    ts = tamoc.TamocSpill(release_time=rt,
                          start_position=(28, -76, 2000),
                          num_elements=10000,
                          end_release_time=datetime(2016, 12, 12, 12),
                          name='TAMOC plume',
                          TAMOC_interval=24,
                          on=True,)

    drops = ts.run_tamoc(rt, 900)
    drops2 = ts.run_tamoc(rt + timedelta(hours = 23), 900)
    assert  drops is drops2
    drops3 = ts.run_tamoc(rt + timedelta(hours = 25), 900)
    assert drops is not drops3
    drops4 = ts.run_tamoc(rt + timedelta(hours = 25), 900)
    assert drops4 is drops3

if __name__ == '__main__':
    test_TamocSpill_run()
