#!/usr/bin/env python
'''
Test code for what happens when a model stops mid-run

e.g. outputters should close properly.

'''

from pathlib import Path
import numpy as np

import pytest


import gnome.scripting as gs

from gnome.environment import Wind
from gnome.model import Model

from gnome.spills.spill import Spill, point_line_spill

from gnome.outputters import Renderer, TrajectoryGeoJsonOutput

from gnome.exceptions import ReferencedObjectNotSet, GnomeRuntimeError

HERE = Path(__file__).parent

# def test_exceptions():
#     """
#     Test GnomeRuntimeError exception thrown if setup has errors
#     time_step < 0 with duration > 0
#     weathering on for backwards run (time_step < 0, duration < 0)

#     outputters shouldn't output in this case.

#     """
#     model = Model()
#     model.time_step = -900
#     with raises(GnomeRuntimeError):
#         model.check_inputs()

#     model.duration = -timedelta(days=1)
#     model.check_inputs()

#     wind = constant_wind(10, 270, units='knots')
#     water = Water()
#     model.weatherers += Evaporation(water, wind)
#     model.spills += Spill(Release(release_time=model.start_time), substance=test_oil)
#     with raises(GnomeRuntimeError):
#         model.check_inputs()


@pytest.fixture(scope='function')
def model():
    '''
    Utility to setup up a simple, but complete model for tests

    has a point wind and a single spill.
    '''
    start_time = "2025-03-26T9:00:00"

    model = Model(start_time = start_time,
                  time_step = gs.hours(1),
                  duration = gs.days(2),
                  )

    model.spills += gs.point_line_spill(
    num_elements=100,
    start_position=(),
    release_time=start_time,
    end_position=None,
    end_release_time=None,
    substance=None,
    amount=0,
    units='kg',
    water=None,
    on=True,
    windage_range=None,
    windage_persist=None,
    name='Point-Line Spill',
)

    # for weatherers and environment objects, make referenced to default
    # wind/water/waves
    model.set_make_default_refs(True)

    return model

# def test_run_out_of_data_full_run(model):
#     """
#     see what happens when the model runs out of data before the run is done
#     """

#     print(model.movers)

#     assert False




# def test_simple_run_backward_rewind():
#     '''
#     Pretty much all this tests is that the model will run
#     and the seed is set during first run, then set correctly
#     after it is rewound and run again
#     '''

#     start_time = datetime(2012, 9, 15, 12, 0)

#     model = Model(time_step = -900, duration = -timedelta(days=1))

#     model.map = GnomeMap()
#     a_mover = SimpleMover(velocity=(1., 2., 0.))

#     model.movers += a_mover
#     assert len(model.movers) == 1

#     spill = point_line_spill(num_elements=10,
#                                      start_position=(0., 0., 0.),
#                                      release_time=start_time)

#     model.spills += spill
#     assert len(model.spills) == 1

#     # model.add_spill(spill)

#     model.start_time = spill.release.release_time

#     # test iterator
#     for step in model:
#         print('just ran time step: %s' % model.current_time_step)
#         assert step['step_num'] == model.current_time_step

#     pos = np.copy(model.spills.LE('positions'))

#     # rewind and run again:
#     print('rewinding')
#     model.rewind()

#     # test iterator is repeatable
#     for step in model:
#         print('just ran time step: %s' % model.current_time_step)
#         assert step['step_num'] == model.current_time_step

#     assert np.all(model.spills.LE('positions') == pos)

