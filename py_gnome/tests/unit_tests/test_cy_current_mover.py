'''
Test base class functionality CyCurrentMover
Also test pickling of all current movers
'''
import pickle

import pytest

from gnome.cy_gnome.cy_current_mover import CyCurrentMover
from gnome.cy_gnome.cy_cats_mover import CyCatsMover


def test_init():
    c = CyCurrentMover()
    assert c.uncertain_duration == 172800
    assert c.uncertain_time_delay == 0
    assert c.up_cur_uncertain == 0.3
    assert c.down_cur_uncertain == -0.3
    assert c.right_cur_uncertain == 0.1
    assert c.left_cur_uncertain == -0.1


@pytest.mark.parametrize("prop", ['uncertain_duration',
                                  'uncertain_time_delay',
                                  'up_cur_uncertain',
                                  'down_cur_uncertain',
                                  'right_cur_uncertain',
                                  'left_cur_uncertain'])
def test_props(prop):
    c = CyCurrentMover()
    setattr(c, prop, 4)
    assert getattr(c, prop) == 4


'Test pickling all objects that derive from CyCurrentMover - parametrized test'
obj = [CyCurrentMover, CyCatsMover]


@pytest.mark.parametrize("obj", obj)
def test_pickle(obj):
    obj = obj()
    new_obj = pickle.loads(pickle.dumps(obj))
    # equality of obects is currently not tested since it isn't yet defined
    # only the repr
    # assert new_obj == obj will FAIL at present
    assert repr(new_obj) == repr(obj)
