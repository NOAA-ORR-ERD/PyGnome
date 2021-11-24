#!/usr/bin/env python
"""
test code for the Outputter classes
"""

from datetime import timedelta

import pytest

from gnome.spills import surface_point_line_spill
from gnome.outputters import Outputter


@pytest.fixture(scope='function')
def model(sample_model):
    """
    Use fixture sample_model and add a few things to it for the
    test
    """
    model = sample_model['model']

    model.cache_enabled = True

    model.spills += surface_point_line_spill(
        num_elements=10,
        start_position=sample_model['release_start_pos'],
        release_time=model.start_time,
        end_release_time=model.start_time + model.duration,
        windage_persist=-1)

    return model


def test_rewind():
    """ test rewind resets data """
    o_put = Outputter(output_timestep=timedelta(minutes=30))
    o_put.rewind()

    assert o_put._model_start_time is None
    assert o_put._dt_since_lastoutput is None
    assert o_put._write_step


model_ts = timedelta(minutes=15)
# output_ts tuple defines
#  (output_timestep, num_model_steps, num_outputs)
# where last two integers specify the ratio of number of outputs produced for
# number of model steps
output_ts = [(model_ts, 1, 1),          # model_ts = output_ts
             (model_ts // 2, 1, 1),      # unlikely case, but test it!
             (model_ts * 3, 3, 1),      # model_ts thrice as fast as output
             (timedelta(seconds=(model_ts.seconds * 1.8)), 9, 5),
             (timedelta(seconds=(model_ts.seconds * 2.5)), 5, 2),
             ]

params = [(model_ts, item) for item in output_ts]
params.extend([(timedelta(hours=6), (timedelta(days=1), 4, 1))])


@pytest.mark.slow
@pytest.mark.parametrize(("model_ts", "output_ts"), params)
def test_output_timestep(model, model_ts, output_ts):
    """
    test the partial functionality implemented in base class
    For different output_timestep values as compared to model_timestep,
    this test ensures the internal _write_step flag is set correcting when
    outputting data. Since both the Renderer and NetCDFOutput call base method,
    using super - testing that _write_step toggles correctly should be
    sufficient
    """
    o_put = Outputter(model._cache, output_timestep=output_ts[0])
    model.duration = model_ts * output_ts[1]
    model.time_step = model_ts

    model.outputters += o_put
    factor = output_ts[0].total_seconds() / model_ts.total_seconds()

    # output timestep aligns w/ model timestep after these many steps
    match_after = output_ts[1]

    # rewind and make sure outputter resets values
    model.rewind()
    assert o_put._model_start_time is None
    assert o_put._dt_since_lastoutput is None
    assert o_put._write_step
    print(("\n\nmodel_ts: {0}, output_ts: {1}").format(model_ts.seconds,
                                                       output_ts[0].seconds))
    print(("num outputs: {0}, for model steps: {1}\n").format(output_ts[2],
                                                              match_after))
    while True:
        try:
            model.step()

            # write_output checks to see if it is zero or last time step and if
            # flags to output these steps are set. It changes _write_step
            # accordingly
            if model.current_time_step == 0:
                assert (o_put._write_step if o_put.output_zero_step
                        else not o_put._write_step)
            elif model.current_time_step == model.num_time_steps - 1:
                assert (o_put._write_step if o_put.output_last_step
                        else not o_put._write_step)
            else:
                # The check for writing output is as follows:
                #     frac_step = current_time_step % factor < 1.0
                #
                # In words,
                #  if frac_step == 0.0,
                #    output is written and
                #    dt_since_lastoutput == output_timestep
                #    so no fractional time and dt_since_lastoutput is resets 0
                #  if frac_step < 1.0,
                #    output is written and there is a fraction time
                #    dt_since_lastoutput > output_timestep at end of step
                #    so reset dt_since_lastoutput to remainder
                #  if frac_step >= 1.0,
                #    no output is written in this time step since not enough
                #    time has elapsed since last output was written, so
                #    dt_since_lastoutput < output_timestep
                #    keep accumulating dt_since_lastoutput - no reset
                #
                # NOTE: mod doesn't work correctly on floating points so
                # multiply by ten, operate on integers, divide by 10.0 and
                # round so check doesn't fail because of rounding issue
                frac_step = round(
                    (model.current_time_step * 10 % int(factor * 10)) / 10.0,
                    6)
                if frac_step < 1.0:
                    assert o_put._write_step
                else:
                    # it is greater than 1 so no output yet
                    assert not o_put._write_step

                if factor < 1.0:
                    # output every time step
                    assert o_put._dt_since_lastoutput == 0.0
                else:
                    remainder = frac_step * model.time_step
                    assert (o_put._dt_since_lastoutput == remainder)

                # check frac_step == 0 every match_after steps
                if model.current_time_step % match_after == 0:
                    assert frac_step == 0.0

                print(("step_num, _write_step, _dt_since_lastoutput:\t"
                       "{0}, {1}, {2}").format(model.current_time_step,
                                               o_put._write_step,
                                               o_put._dt_since_lastoutput))

        except StopIteration:
            break
