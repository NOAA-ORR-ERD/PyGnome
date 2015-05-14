"""
Test if this is how we want id property of
object that inherits from GnomeId to behave
"""
from datetime import datetime, timedelta
import pytest
import copy

from uuid import uuid1
from gnome.gnomeobject import GnomeId
from gnome import (environment,
                   movers,
                   outputters,
                   spill)
from gnome.environment import Waves, Wind
from gnome.weatherers import Evaporation, NaturalDispersion


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


t = datetime(2015, 1, 1, 12, 0)


@pytest.mark.parametrize("obj",
                         (Wind(timeseries=[(t, (0, 1)),
                                           (t + timedelta(10), (0, 2))],
                               units='m/s'),
                          Evaporation(),
                          NaturalDispersion()))
def test_base_validate(obj):
    '''
    base validate checks wind/water/waves objects are not None. Check these
    primarily for weatherers.
    '''
    out = obj.validate()
    print out
    assert len(out) > 0
