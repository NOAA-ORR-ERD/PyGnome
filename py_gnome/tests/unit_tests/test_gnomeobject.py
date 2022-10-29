"""
Test if this is how we want id property of
object that inherits from GnomeId to behave
"""
from datetime import datetime, timedelta
import pytest
import copy

from uuid import uuid1

import numpy as np

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

from gnome.persist import (
    Boolean, Float, Int, String, SchemaNode,  ObjTypeSchema, GeneralGnomeObjectSchema,
    TupleSchema, Range, NumpyArraySchema, drop)

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

## tests the __eq__ implimentation
# note: very hard to make this comprehensive, but it's something.

class ExampleSchema(ObjTypeSchema):

    a_float = SchemaNode(Float(), save=True, update=True)
    a_int = SchemaNode(Float(), save=True, update=True)
    a_bool = SchemaNode(Float(), save=True, update=True)
    a_str = SchemaNode(Float(), save=True, update=True)
    a_ndarray = SchemaNode(Float(), save=True, update=True)

    # There are more to do -- see extend_colander

class ExampleObject(GnomeId):
    _schema = ExampleSchema

    def __init__(self,
                 a_float=1.1,
                 a_int=12,
                 a_bool=True,
                 a_str='a string',
                 a_ndarray=np.ones((4,))
                 ):
        self.a_float = a_float
        self.a_int = a_int
        self.a_bool = a_bool
        self.a_str = a_str
        self.a_ndarray = a_ndarray

class DummySchema(ObjTypeSchema):

    a_float = SchemaNode(Float(), save=True, update=True)
    a_int = SchemaNode(Float(), save=True, update=True)
    # There are more to do -- see extend_colander


class DummyObject(GnomeId):
    _schema = ExampleSchema

    def __init__(self,
                 a_float=1.1,
                 a_int=12,
                 ):
        self.a_float = a_float
        self.a_int = a_int


def test_different_object_type():
    o1 = ExampleObject()
    o2 = DummyObject()

    assert not o1 == o2
    assert not o2 == o1

    # check diff
    diff = o1._diff(o2, fail_early=True)

    assert len(diff) == 1
    assert "Different type:" in diff[0]


def test_defaults():
    """
    they'd better be equal
    """
    o1 = ExampleObject()
    o2 = ExampleObject()

    assert o1 == o2
    assert o2 == o1


def test_float_different():
    """
    these should not match
    """
    o1 = ExampleObject(a_float=1.0)
    o2 = ExampleObject(a_float=2.0)

    assert not o1 == o2
    assert not o2 == o1

    # check diff
    diff = o1._diff(o2, fail_early=True)

    assert len(diff) == 1
    assert "Difference outside tolerance --" in diff[0]


def test_float_slightly_different():
    """
    they'd should be close enough
    """
    o1 = ExampleObject(a_float=1.0)
    o2 = ExampleObject(a_float=1.000000000001)

    print(o1._diff(o2))

    assert o1 == o2
    assert o2 == o1


def test_huge_float_slightly_different():
    """
    they'd should be close enough
    """
    o1 = ExampleObject(a_float=1.0e100)
    o2 = ExampleObject(a_float=1.000000000001e100)

    print(o1._diff(o2))

    assert o1 == o2
    assert o2 == o1


def test_int_different():
    o1 = ExampleObject(a_int=12345)
    o2 = ExampleObject(a_int=12346)

    assert not o2 == o1
    assert not o1 == o2


def test_bool_different():
    o1 = ExampleObject(a_bool=True)
    o2 = ExampleObject(a_bool=False)

    assert not o1 == o2
    assert not o2 == o1


## playing games with bool that probably don't matter ...
def test_bool_using_ints_same_true():
    """These are both Truthy -- but not equal
       Should that work? Probably not :-)
    """
    o1 = ExampleObject(a_bool=34)
    o2 = ExampleObject(a_bool=12)

    assert not o1 == o2
    assert not o2 == o1


def test_bool_using_ints_same_false():
    o1 = ExampleObject(a_bool=0)
    o2 = ExampleObject(a_bool=0)

    assert o1 == o2
    assert o2 == o1


def test_bool_using_ints_diff():
    o1 = ExampleObject(a_bool=34)
    o2 = ExampleObject(a_bool=0)

    assert not o1 == o2
    assert not o2 == o1


def test_str_different():
    o1 = ExampleObject(a_str='a string')
    o2 = ExampleObject(a_str='A string')

    assert not o1 == o2
    assert not o2 == o1

    # diff
    diff = o1._diff(o2, fail_early=True)

    assert len(diff) == 1
    assert "Values not equal --" in diff[0]


## now the array tests -- this is getting more complicated.
def test_arr_not_equal():
    o1 = ExampleObject(a_ndarray=np.array([1, 2, 3, 4, 5], dtype=np.float32))
    o2 = ExampleObject(a_ndarray=np.array([1, 2, 3, 4, 6], dtype=np.float32))

    assert not o1 == o2
    assert not o2 == o1


def test_arr_different_size():
    o1 = ExampleObject(a_ndarray=np.array([1, 2, 3, 4, 5], dtype=np.float32))
    o2 = ExampleObject(a_ndarray=np.array([1, 2, 3, 4, 6, 7], dtype=np.float32))

    assert not o1 == o2
    assert not o2 == o1

    # diff
    diff = o1._diff(o2, fail_early=True)

    assert len(diff) == 1
    assert "Arrays are not the same size --" in diff[0]


def test_arr_close_enough():
    o1 = ExampleObject(a_ndarray=np.array([1, 2, 3], dtype=np.float32))
    o2 = ExampleObject(a_ndarray=np.array([1, 2, 3.00001], dtype=np.float32))

    assert o1 == o2
    assert o2 == o1


def test_arr_not_close_enough():
    o1 = ExampleObject(a_ndarray=np.array([1, 2, 3], dtype=np.float32))
    o2 = ExampleObject(a_ndarray=np.array([0.99999, 2, 3.00001], dtype=np.float32))

    assert not o1 == o2
    assert not o2 == o1

    # diff
    diff = o1._diff(o2, fail_early=True)

    assert len(diff) == 1
    assert "Array values are not all close --" in diff[0]



def test_arr_tuple_close_enough():
    o1 = ExampleObject(a_ndarray=(1, 2, 3))
    o2 = ExampleObject(a_ndarray=np.array([1, 2, 3.00001], dtype=np.float32))

    assert o1 == o2
    assert o2 == o1


def test_arr_list_close_enough():
    o2 = ExampleObject(a_ndarray=[1, 2, 3])
    o1 = ExampleObject(a_ndarray=np.array([1, 2, 3.00001], dtype=np.float32))

    assert o1 == o2
    assert o2 == o1

    # diff test:
    diff = o1._diff(o2, fail_early=False)

    assert len(diff) == 0


def test_diff_multiple_differrences():
    o1 = ExampleObject(a_float=1.1,
                       a_int=12,
                       a_bool=True,
                       a_str='a string',
                       a_ndarray=np.ones((4, )))
    o2 = ExampleObject(a_float=3.1,
                       a_int=22,
                       a_bool=False,
                       a_str='another  string',
                       a_ndarray=np.ones((5, )))

    diffs = o1._diff(o2)

    messages = ['Difference outside tolerance -- a_float',
                'Values not equal -- a_int:',
                'Values not equal -- a_bool:',
                'Values not equal -- a_str:',
                'Arrays are not the same size --',
                ]

    print("diffs are:")
    for diff in diffs:
        print(diff)

    for msg in messages:
        for diff in diffs:
            if msg in diff:
                break
        else:
            raise AssertionError(f'{msg} is mising from diff')


# FIXME: there should be a test of weird arrays, like DatetimeValue2dArray

