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
from gnome.model import Model
from gnome.environment import Waves, Wind, Water
from gnome.weatherers import Evaporation, NaturalDispersion
from gnome.exceptions import ReferencedObjectNotSet


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


@pytest.mark.parametrize(("obj", "make_default_refs", "objvalid"),
                         [(Wind(timeseries=[(t, (0, 1)),
                                            (t + timedelta(10), (0, 2))],
                                units='m/s'), False, True),
                          (Evaporation(), False, False),
                          (NaturalDispersion(), False, False),
                          (Evaporation(), True, True)])
def test_base_validate(obj, make_default_refs, objvalid):
    '''
    base validate checks wind/water/waves objects are not None. Check these
    primarily for weatherers.
    '''
    obj.make_default_refs = make_default_refs
    (out, isvalid) = obj.validate()
    print out
    print isvalid
    assert isvalid is objvalid
    assert len(out) > 0


def test_make_default_refs():
    '''
    ensure make_default_refs is a thread-safe operation
    once object is instantiated, object.make_default_refs is an attribute of
    instance
    '''
    model = Model()
    model1 = Model()
    wind = Wind(timeseries=[(t, (0, 1))], units='m/s')
    water = Water()

    waves = Waves(name='waves')
    waves1 = Waves(name='waves1', make_default_refs=False)
    model.environment += [wind,
                          water,
                          waves]
    model1.environment += waves1

    # waves should get auto hooked up/waves1 should not
    model.step()
    assert waves.wind is wind
    assert waves.water is water
    with pytest.raises(ReferencedObjectNotSet):
        model1.step()
