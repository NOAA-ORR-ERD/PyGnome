"""
Test if this is how we want id property of
object that inherits from GnomeId to behave
"""

import pytest
import copy

from uuid import uuid1
from gnome.gnomeobject import GnomeId
from gnome import (environment,
                   movers,
                   outputters,
                   spill)


def test_exceptions():
    with pytest.raises(AttributeError):
        go = GnomeId()
        print '\n id exists: {0}'.format(go.id)  # calls getter, assigns an id
        go.id = uuid1()


def test_copy():
    go = GnomeId()
    go_c = copy.copy(go)
    assert go.id != go_c.id
    assert go is not go_c


def test_deepcopy():
    go = GnomeId()
    go_c = copy.deepcopy(go)
    assert go.id != go_c.id
    assert go is not go_c


'''
test 'name' is an input for all base classes
'''
base_class = [(environment.Environment, ()),
              (movers.Mover, ()),
              (outputters.Outputter, ()),
              (spill.Release, (10,)),
              (spill.Spill, (spill.Release(0),))
              ]


@pytest.mark.parametrize("b_class", base_class)
def test_set_name(b_class):
    name = "name_{0}".format(uuid1())
    class_ = b_class[0]
    inputs = b_class[1]
    obj = class_(*inputs, name=name)
    assert obj.name == name

    obj.name = obj.__class__.__name__
    assert obj.name == obj.__class__.__name__
