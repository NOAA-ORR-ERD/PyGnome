#!/usr/bin/env python

"""
Tests the Exception Handling and logging messages for being outside a valid time interval
"""

from datetime import datetime
import numpy as np

import pytest

from gnome.environment import wind_from_values
from gnome.movers import WindMover

# simple fake spill container
class SC(dict):
    uncertain = False
    num_released = 0

def test_wind_mover():
    """
    use a wind_mover, about as simple as it comes
    """

    # fake data arrays:
    num = 2
    pos_dt = np.dtype([('lon', np.float), ('lat', np.float), ('depth', np.float)])
    sc = SC({'positions': np.zeros((num,), dtype=pos_dt),
             'status_codes': np.zeros((num,), dtype=np.int16),
             'windages': np.zeros((num,)),
             })
#    delta = np.zeros_like(sc['positions'])

    wind = wind_from_values([(datetime(2016, 5, 10, 12, 0), 5, 45),
                             (datetime(2016, 5, 10, 12, 20), 6, 50),
                             (datetime(2016, 5, 10, 12, 40), 7, 55),
                             ])
    wm = WindMover(wind)
    # in time span, this should work:
    wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 12, 20))

    # before timespan -- this should fail
    with pytest.raises(RuntimeError):
        wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 11, 50))

    # after timespan -- this should fail
    with pytest.raises(RuntimeError):
        wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 12, 50))

    # # test the message:
    # try:
    #     wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 11, 50))
    # except RuntimeError as err:

    try:
        wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 11, 50))
    except RuntimeError as err:
        msg = err.args[0]
        assert "No available data" in msg
        assert "WindMover" in msg
        assert "2016-05-10 11:50:00" in msg
        assert "2016-05-10 12:00:00" in msg
        assert "2016-05-10 12:40:00" in msg
