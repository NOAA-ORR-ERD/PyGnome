'''
Test setting/getting properties defined in base class and pickle/unpickle
This is done for CyCurrentMover and all its children as parametrized tests
in this module
'''
import pickle

import pytest

from gnome.cy_gnome.cy_current_mover import CyCurrentMover
from gnome.cy_gnome.cy_cats_mover import CyCatsMover


'Test pickling all objects that derive from CyCurrentMover - parametrized test'
obj = [CyCurrentMover, CyCatsMover]


@pytest.mark.parametrize("obj", obj)
def test_init(obj, CyCurrentMover_props):
    props = CyCurrentMover_props
    c = obj()
    for prop in props:
        assert getattr(c, prop[0]) == prop[1]


@pytest.mark.parametrize("obj", obj)
def test_props(obj, CyCurrentMover_props):
    props = CyCurrentMover_props
    c = obj()
    for prop in props:
        setattr(c, prop[0], 4)
        assert getattr(c, prop[0]) == 4


@pytest.mark.parametrize("obj", obj)
def test_pickle(obj):
    obj = obj()
    new_obj = pickle.loads(pickle.dumps(obj))
    # equality of obects is currently not tested since it isn't yet defined
    # only the repr
    # assert new_obj == obj will FAIL at present
    assert repr(new_obj) == repr(obj)
