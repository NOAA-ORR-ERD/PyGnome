"""
Test if this is how we want id property of
object that inherits from GnomeId to behave
"""
from datetime import datetime, timedelta
import pytest
import copy

from uuid import uuid1
from gnome.gnomeobject import GnomeId, combine_signatures
from gnome import (environment,
                   movers,
                   outputters,
                   )
from gnome.spills.spill import Spill
from gnome.spills.release import Release
from gnome.model import Model
from gnome.environment import Waves, Wind, Water
from gnome.weatherers import Evaporation, NaturalDispersion
from gnome.exceptions import ReferencedObjectNotSet


def test_exceptions():
    with pytest.raises(AttributeError):
        go = GnomeId()
        print('\n id exists: {0}'.format(go.id))  # calls getter, assigns an id
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
              (Release, ()),
              (Spill, (Release(),))
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
    print(out)
    print(isvalid)
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

    assert waves1.wind is None
    assert waves1.water is None

class demo1(GnomeId):
    @combine_signatures
    def __init__(self, foo, bar=None, **kwargs):
        super().__init__(**kwargs)

    @combine_signatures
    def footest(self):
        return 'testfunc'

    @classmethod
    @combine_signatures
    def new_from_dict(cls, something, *args, options='foo'):
        '''new from dict with a different signature'''
        return super().new_from_dict(*args)

def test_signature_combination():
    #test class signature
    paramnames = [p.name for p in demo1.__signature__.parameters.values()]
    assert 'name' in paramnames

    #test footest signature. Should do nothing but assign a __signature__ (nothing to combine)
    paramnames = [p.name for p in demo1.footest.__signature__.parameters.values()]
    assert len(paramnames) == 1

    #test classmethod, and checks proper arg ordering
    paramnames = [p.name for p in demo1.new_from_dict.__signature__.parameters.values()]
    assert len(paramnames) == 4
    assert paramnames[0] == 'something'
    assert paramnames[1] == 'dict_'
    assert paramnames[2] == 'args'
    assert paramnames[3] == 'options'


