'''
test object in environment module
'''

import pytest

from nucos import InvalidUnitError

from gnome.utilities.inf_datetime import InfDateTime
from gnome.utilities.time_utils import TZOffset
from gnome.environment import Environment, Water


def test_environment_init():
    env = Environment()
    sample_time = 60 * 60 * 24 * 365 * 30  # seconds

    assert env._ref_as == 'environment'

    assert isinstance(env.timezone_offset, TZOffset)

    with pytest.raises(NotImplementedError):
        _dstart = env.data_start

    with pytest.raises(NotImplementedError):
        env.data_start = sample_time

    with pytest.raises(NotImplementedError):
        _dstop = env.data_stop

    with pytest.raises(NotImplementedError):
        env.data_stop = sample_time

    # We want these base class methods available for use, so no implementation
    # exceptions.  But they don't actually do anything.
    assert env.prepare_for_model_run(sample_time) is None
    assert env.prepare_for_model_step(sample_time) is None

def test_environment_timezone_offset():
    env1 = Environment()
    assert env1.timezone_offset.offset == None
    env2 = Environment(timezone_offset=TZOffset(offset=5.0, title="Eastern"))
    assert env2.timezone_offset.offset == 5.0
    env2.timezone_offset = None
    assert env2.timezone_offset.offset == None
    env1.timezone_offset = TZOffset(offset=5.0, title="Eastern")
    assert env1.timezone_offset.offset == 5.0
