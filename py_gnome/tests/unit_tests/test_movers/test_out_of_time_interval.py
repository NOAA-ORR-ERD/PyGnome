#!/usr/bin/env python

"""
Tests the Exception Handling and logging messages for being outside a valid
time interval
"""





from datetime import datetime
import numpy as np

import pytest

from gnome.environment import wind_from_values
from gnome.movers import PointWindMover


# simple fake spill container
class SC(dict):
    uncertain = False
    num_released = 0


def test_wind_mover():
    '''
        Use a wind_mover, about as simple as it comes
        - We are moving to a design where the environment objects contain the
          extrapolate flag instead of the movers.  This flag is off by default.
    '''

    # fake data arrays:
    num = 2

    pos_dt = np.dtype([('lon', np.float64),
                       ('lat', np.float64),
                       ('depth', np.float64)])

    sc = SC({'positions': np.zeros((num,), dtype=pos_dt),
             'status_codes': np.zeros((num,), dtype=np.int16),
             'windages': np.zeros((num,)),
             })

    wind = wind_from_values([(datetime(2016, 5, 10, 12, 0), 5, 45),
                             (datetime(2016, 5, 10, 12, 20), 6, 50),
                             (datetime(2016, 5, 10, 12, 40), 7, 55),
                             ])
    wm = PointWindMover(wind)

    # within the model's time span, this should work:
    wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 12, 20))

    # before time span -- this should fail
    with pytest.raises(RuntimeError):
        wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 11, 50))

    # after time span -- this should fail
    with pytest.raises(RuntimeError):
        wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 12, 50))

    # turn on extrapolation in the wind environment object
    wind.extrapolation_is_allowed = True

    # before timespan -- this should pass now
    wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 11, 50))

    # after timespan -- this should pass now
    wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 12, 50))

    # test the error message that we get when a RuntimeError is raised.
    with pytest.raises(RuntimeError):
        try:
            wind.extrapolation_is_allowed = False
            wm.prepare_for_model_step(sc, 600, datetime(2016, 5, 10, 11, 50))
        except RuntimeError as err:
            msg = err.args[0]
            assert "No available data" in msg
            assert "PointWindMover" in msg
            assert "2016-05-10 11:50:00" in msg
            assert "2016-05-10 12:00:00" in msg
            assert "2016-05-10 12:40:00" in msg
            raise









