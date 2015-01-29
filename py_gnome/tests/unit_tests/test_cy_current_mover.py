'''
Test setting/getting properties defined in base class
This is done for CyCurrentMover and all its children as parametrized tests
in this module
'''
import pytest

from gnome.cy_gnome.cy_current_mover import CyCurrentMover
from gnome.cy_gnome.cy_cats_mover import CyCatsMover
from gnome.cy_gnome.cy_component_mover import CyComponentMover


'Test pickling all objects that derive from CyCurrentMover - parametrized test'
obj = [CyCurrentMover, CyCatsMover, CyComponentMover]


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
