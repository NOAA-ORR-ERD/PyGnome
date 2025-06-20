"""
Some tests for running the model backwards
"""

import numpy as np

import gnome.scripting as gs  # just for the scripting specific things, like hours()
from gnome.model import Model
from gnome.environment.wind import constant_wind
from gnome.movers.py_wind_movers import WindMover
from gnome.environment import SteadyUniformCurrent
from gnome.movers import SimpleMover
from gnome.spills.spill import Spill, point_line_spill


def test_simple_run_backward_rewind():
    '''
    Pretty much all this tests is that the model will run
    and the seed is set during first run, then set correctly
    after it is rewound and run again
    '''

#    start_time = datetime(2012, 9, 15, 12, 0)
    start_time = "2012-9-15T12:00"

    #model = Model(time_step = -900, duration = -timedelta(days=1))
    model = Model(run_backwards = True)

    # use SteadyUniformCurrent ??
    a_mover = SimpleMover(velocity=(1., 2., 0.))

    model.movers += a_mover
    assert len(model.movers) == 1

    spill = point_line_spill(num_elements=10,
                                     start_position=(0., 0., 0.),
                                     release_time=start_time)

    model.spills += spill
    assert len(model.spills) == 1

    # model.add_spill(spill)

    model.start_time = spill.release.release_time

    # test iterator
    for step in model:
        print('just ran time step: %s' % model.current_time_step)
        assert step['step_num'] == model.current_time_step

    pos = np.copy(model.spills.LE('positions'))

    # rewind and run again:
    print('rewinding')
    model.rewind()

    # test iterator is repeatable
    for step in model:
        print('just ran time step: %s' % model.current_time_step)
        assert step['step_num'] == model.current_time_step

    assert np.all(model.spills.LE('positions') == pos)


def test_model_release_after_start_backwards():
    '''
    This runs the model backwards for a simple spill, that starts after the model starts

    Now tests that the elements get released at the right time!
    '''
    units = 'meter per second'
    start_time = "2013-2-22T12"  # model start at 12:00

    model = Model(time_step=gs.minutes(30),
                  start_time=start_time,
                  duration=gs.hours(3),
                  run_backwards = True)

    # add a spill that starts after the run begins.
    model.spills += point_line_spill(num_elements=5,
                                     start_position=(0, 0, 0),
                                     # should release at 11:00 -- e.g. second timestep
                                     release_time=(model.start_time - gs.hours(1)),
                                     )

    # and another that starts even later..

    model.spills += point_line_spill(num_elements=10,
                                     start_position=(0, 0, 0),
                                     # should release at 10:00 -- e.g. fourth timestep
                                     release_time=(model.start_time - gs.hours(2))
                                     )

    # Add a Wind mover -- so there's something going on ...
    wind = constant_wind(10, 45, units)
    model.movers += WindMover(wind)


    for step in model:
        step_num = step['step_num']
        step_time = step['step_time']
        num_LEs = len(model.get_spill_property('positions'))
        print(f'running step: {step_num}, {step_time}')
        print(f"{num_LEs=}")

        if step_num < 2:
            assert num_LEs == 0
        elif step_num < 4:
            assert num_LEs == 5
        else: num_LEs == 15

    assert False
