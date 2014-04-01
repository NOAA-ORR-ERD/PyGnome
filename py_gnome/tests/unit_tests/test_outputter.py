#!/usr/bin/env python
"""
test code for the Outputter classes
"""

from datetime import timedelta
import math

import pytest

from gnome.spill.elements import floating

from gnome.spill import point_line_release_spill
from gnome.outputters import Outputter


@pytest.fixture(scope='module')
def model(sample_model):
    """
    Use fixture sample_model and add a few things to it for the
    test
    """
    model = sample_model['model']

    model.cache_enabled = True

    model.spills += point_line_release_spill(num_elements=5,
                        start_position=sample_model['release_start_pos'],
                        release_time=model.start_time,
                        end_release_time=model.start_time + model.duration,
                        element_type=floating(windage_persist=-1))

    return model


def test_rewind():
    """ test rewind resets data """
    o_put = Outputter(output_timestep=timedelta(minutes=30))
    o_put.rewind()

    assert o_put._model_start_time is None
    assert o_put._next_output_time is None
    assert not o_put._write_step


output_ts = timedelta(minutes=30)
model_ts = [output_ts * 2, output_ts, output_ts // 2,
            timedelta(seconds=(output_ts.seconds / 3.5))]


@pytest.mark.parametrize(("output_ts", "model_ts"),
                         [(output_ts, item) for item in model_ts])
def test_output_timestep(model, output_ts, model_ts):
    """
    test the partial functionality implemented in base class
    For different output_timestep values as compared to model_timestep,
    this test ensures the internal _write_step flag is set correcting when
    outputting data. Since both the Renderer and NetCDFOutput call base method,
    using super - testing that _write_step toggles correctly should be
    sufficient
    """
    o_put = Outputter(model._cache, output_timestep=output_ts)
    model.duration = timedelta(hours=3)
    model.time_step = model_ts

    model.outputters += o_put
    factor = math.ceil(float(output_ts.seconds) / model_ts.seconds)

    # rewind and make sure outputter resets values
    model.rewind()
    assert o_put._model_start_time is None
    assert o_put._next_output_time is None
    assert not o_put._write_step

    # call each part of model.step separately so we can validate
    # how the outputter is setting its values, especially _write_step
    for step_num in range(-1, model.num_time_steps - 1):
        if step_num == -1:
            model.setup_model_run()

            assert o_put._model_start_time == model.start_time
            assert o_put._next_output_time == (model.model_time +
                                               o_put.output_timestep)
            assert not o_put._write_step
        else:
            model.setup_time_step()

            if o_put.output_timestep <= timedelta(seconds=model.time_step):
                assert o_put._write_step    # write every step
            else:
                if (step_num + 1) % factor == 0:
                    assert o_put._write_step

            # No need to call following since outputter doesn't use them, but
            # leave for completeness.
            model.move_elements()
            model.step_is_done()

        model.current_time_step += 1

        # no need to release elements or save to cache since we're not
        # interested in the actual data - just want to see if the step_num
        # should be written out or not
        #model._cache.save_timestep(model.current_time_step, model.spills)
        model.write_output()

        # each outputter, like the Renderer or NetCDFOutputter will update the
        # output timestep after completing the write_output method. Base class
        # not designed to be included as an outputter for the Model, as it only
        # implements partial functionality; need to manually update the output
        # timestep below.
        if o_put._write_step:
            # no update happens for last time step
            o_put._update_next_output_time(model.current_time_step,
                                           model.model_time)
            if model.current_time_step < model.num_time_steps - 1:
                assert o_put._next_output_time == (model.model_time +
                                                   o_put.output_timestep)

        # write_output checks to see if it is zero or last time step and if
        # flags to output these steps are set. It changes _write_step
        # accordingly
        if model.current_time_step == 0:
            assert (o_put._write_step if o_put.output_zero_step
                    else not o_put._write_step)
        elif model.current_time_step == model.num_time_steps - 1:
            assert (o_put._write_step if o_put.output_last_step
                    else not o_put._write_step)

        print 'Completed step: {0}'.format(model.current_time_step)
