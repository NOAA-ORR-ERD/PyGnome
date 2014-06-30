#!/usr/bin/env python

from pytest import raises, mark

from gnome.movers.simple_mover import SimpleMover
from gnome.movers import Mover, RandomMover

from gnome.utilities.orderedcollection import OrderedCollection


def s_id(val):
    'all IDs are stored as strings'
    try:
        return val.id
    except AttributeError:
        return str(id(val))


def test_getslice():
    # getting a slice returns a new list
    l_ = range(6)
    oc = OrderedCollection(l_)
    b = oc[:3]
    b[0] = 10
    assert b != l_
    assert l_[::2] == oc[::2]


def test_remake():
    'remakes internal lists without None enteries'
    oc = OrderedCollection(['p', 'q', 'ab', 'adsf', 'ss'])
    del oc[0]
    del oc[3]
    assert oc._elems[0] is None
    assert oc._elems[3] is None
    oc.remake()
    for ix, elem in enumerate(oc._elems):
        assert elem is not None
        assert oc._index[s_id(elem)] == ix


def test_remake_emptyoc():
    'empty OC'
    oc = OrderedCollection(dtype=int)
    oc.remake()


class TestOrderedCollection(object):

    def test_init(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        assert oc.dtype == int
        oc = OrderedCollection([1, 2, 3, 4, 5], int)
        assert oc.dtype == int
        oc = OrderedCollection(dtype=int)
        assert oc.dtype == int

        with raises(TypeError):

            # either a populated list or a dtype is required

            oc = OrderedCollection()

        with raises(TypeError):
            oc = OrderedCollection('not a list')

        with raises(TypeError):

            # either a populated list or a dtype is required

            oc = OrderedCollection([])

        with raises(TypeError):
            oc = OrderedCollection([1, 2, 3, 4, 5], float)

    def test_len(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        assert len(oc) == 5

    def test_iter(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        assert [i for i in oc] == [1, 2, 3, 4, 5]

    def test_contains(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        assert 5 in oc

    def test_not_contains(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        assert 10 not in oc

    def test_getitem(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        oc[s_id(3)]
        assert oc[s_id(3)] == 3
        with raises(KeyError):
            oc[s_id(6)]

    def test_getitem_byindex(self):
        oc = OrderedCollection(['x', 'a', 'p', 'd'])
        assert oc[1] == 'a'
        oc[s_id('a')] = 'b'
        assert oc[1] == 'b'
        del oc[1]
        assert oc[1] == 'p'

    def test_setitem_exceptions(self):
        'Use add to add an element'
        oc = OrderedCollection([1, 2, 3, 4, 5])
        with raises(KeyError):
            oc[s_id(6)] = 6

        with raises(IndexError):
            oc[5] = 6

    def test_setitem(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        oc[s_id(4)] = 7
        oc[0] = 0
        assert [i for i in oc] == [
            0,
            2,
            3,
            7,
            5,
            ]

    def test_delitem(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        with raises(KeyError):
            del oc[s_id(6)]
        del oc[s_id(4)]
        assert [i for i in oc] == [1, 2, 3, 5]

    def test_delitem_byindex(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        with raises(IndexError):
            del oc[5]
        del oc[3]
        assert [i for i in oc] == [1, 2, 3, 5]

    def test_iadd(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        oc += 6
        assert [i for i in oc] == [
            1,
            2,
            3,
            4,
            5,
            6,
            ]
        oc += [7, 8, 9]
        assert [i for i in oc] == [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            ]

    def test_add(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        oc.add(6)
        assert [i for i in oc] == [
            1,
            2,
            3,
            4,
            5,
            6,
            ]
        with raises(TypeError):
            oc.add('not an int')

    def test_remove(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        with raises(KeyError):
            oc.remove(s_id(6))
        oc.remove(s_id(4))
        assert [i for i in oc] == [1, 2, 3, 5]

        oc.remove(2)
        assert [i for i in oc] == [1, 2, 5]

    def test_replace(self):
        oc = OrderedCollection([1, 2, 3, 4, 5])
        oc.replace(s_id(4), 7)  # replace by object ID
        oc.replace(0, 0)    # replace by index
        assert [i for i in oc] == [
            0,
            2,
            3,
            7,
            5,
            ]
        assert oc[s_id(7)] == 7
        with raises(KeyError):
            # our key should also be gone after the delete
            oc[s_id(4)]
        with raises(TypeError):
            oc.replace(s_id(7), 'not an int')

    def test_index(self):
        'behaves like index for a list'
        oc = OrderedCollection([1, 2, 3, 4, 5])
        assert oc.index(3) == 2
        assert oc.index(s_id(3)) == 2
        oc[s_id(3)] = 6
        assert oc.index(6) == 2
        assert oc.index(s_id(6)) == 2
        del oc[s_id(6)]
        assert oc.index(4) == 2
        assert oc.index(s_id(4)) == 2

    def test_with_movers(self):
        mover_1 = SimpleMover(velocity=(1.0, -1.0, 0.0))
        mover_2 = SimpleMover(velocity=(1.0, -1.0, 0.0))
        mover_3 = SimpleMover(velocity=(1.0, -1.0, 0.0))
        mover_4 = SimpleMover(velocity=(1.0, -1.0, 0.0))

        # test our init, iter, get, and len methods

        mymovers = OrderedCollection([mover_1, mover_2], dtype=Mover)
        assert [m for m in mymovers] == [mover_1, mover_2]
        assert mymovers[mover_1.id] == mover_1
        assert len(mymovers) == 2

        # test our add methods

        mymovers = OrderedCollection(dtype=Mover)
        mymovers += mover_1
        mymovers += mover_2
        assert [m for m in mymovers] == [mover_1, mover_2]

        mymovers = OrderedCollection(dtype=Mover)
        mymovers += [mover_1, mover_2]
        assert [m for m in mymovers] == [mover_1, mover_2]

        # test our del method

        mymovers = OrderedCollection([mover_1, mover_2, mover_3],
                dtype=Mover)
        del mymovers[mover_2.id]
        assert [m for m in mymovers] == [mover_1, mover_3]

        # test our replace method

        mymovers = OrderedCollection([mover_1, mover_2, mover_3],
                dtype=Mover)
        mymovers[mover_2.id] = mover_4
        assert [m for m in mymovers] == [mover_1, mover_4, mover_3]
        assert mymovers[mover_4.id] == mover_4

    def test_eq(self):
        'Test comparison operator __eq__'

        assert OrderedCollection([1, 2, 3, 4, 5]) \
            == OrderedCollection([1, 2, 3, 4, 5])

    def test_ne(self):
        'Test comparison operator (not equal)'

        assert OrderedCollection([1, 2, 3, 4, 5]) \
            != OrderedCollection([2, 1, 3, 4, 5])
        assert OrderedCollection([1, 2, 3, 4, 5]) \
            != OrderedCollection([1, 2, 3, 4])
        assert OrderedCollection([1, 2, 3, 4, 5]) != [1, 2, 3, 4, 5]

    @mark.parametrize('json_', ['save', 'webapi'])
    def test_to_dict(self, json_):
        'added a to_dict() method - test this method'

        items = [SimpleMover(velocity=(i * 0.5, -1.0, 0.0)) for i in
                 range(2)]
        items.extend([RandomMover() for i in range(2)])
        mymovers = OrderedCollection(items, dtype=Mover)
        self._to_dict_assert(mymovers, items, json_)

    @mark.parametrize('json_', ['save', 'webapi'])
    def test_int_to_dict(self, json_):
        '''added a to_dict() method - test this method for int dtype.
        Tests the try, except is working correctly'''
        items = range(5)
        oc = OrderedCollection(items)
        self._to_dict_assert(oc, items, json_)

    def _to_dict_assert(self, oc, items, json_):
        toserial = oc.to_dict()
        print toserial
        for (i, mv) in enumerate(items):
            try:
                assert toserial[i]['obj_type'] == \
                    '{0}.{1}'.format(mv.__module__, mv.__class__.__name__)
            except AttributeError:
                assert toserial[i]['obj_type'] == mv.__class__.__name__

            assert toserial[i]['id'] == s_id(mv)


class ObjToAdd:
    'Define a helper class (mutable object) for use in TestCallbacks'
    def __init__(self):
        self.reset()

    def reset(self):
        self.add_callback = False
        self.rm_callback = False
        self.replace_callback = False


class TestCallbacks:

    to_add = [ObjToAdd(), ObjToAdd(), ObjToAdd()]

    def test_add_callback(self):
        '''
            test add callback is invoked after adding an object or
            list of objects
        '''

        # lets work with a mutable type

        oc = OrderedCollection(dtype=ObjToAdd)
        oc.register_callback(self._add_callback, events='add')

        # check everything if False initially

        self._reset_ObjToAdd_init_state()

        oc += self.to_add
        oc += ObjToAdd()

        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback

    def test_remove_callback(self):
        'test remove callback is invoked after removing an object'

        oc = OrderedCollection(dtype=ObjToAdd)  # lets work with a mutable type
        oc.register_callback(self._rm_callback, events='remove')
        oc.register_callback(self._add_callback, events='add')

        # check everything if False initially

        self._reset_ObjToAdd_init_state()

        oc += self.to_add

        del oc[s_id(self.to_add[0])]

        assert self.to_add[0].rm_callback
        assert self.to_add[0].add_callback
        assert not self.to_add[0].replace_callback

        self.to_add[0].reset()  # reset all to false
        oc += self.to_add[0]  # let's add this back in

        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback

    def test_replace_callback(self):
        'test replace callback is invoked after replacing an object'

        # lets work with a mutable type

        oc = OrderedCollection(dtype=ObjToAdd)
        oc.register_callback(self._replace_callback, events='replace')

        # check everything if False initially

        self._reset_ObjToAdd_init_state()

        oc += self.to_add
        rep = ObjToAdd()
        oc[s_id(self.to_add[0])] = rep

        for obj in oc:
            assert not obj.add_callback
            assert not obj.rm_callback
            if id(obj) == id(rep):
                assert obj.replace_callback
            else:
                assert not obj.replace_callback

    def test_add_replace_callback(self):
        'register one callback with multiple events (add, replace)'

        # lets work with a mutable type

        oc = OrderedCollection(dtype=ObjToAdd)
        oc.register_callback(self._add_callback, events=('add',
                             'replace'))

        # check everything if False initially

        self._reset_ObjToAdd_init_state()

        oc += self.to_add

        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback

        rep = ObjToAdd()
        oc[s_id(self.to_add[0])] = rep

        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback

    def _add_callback(self, obj_):
        obj_.add_callback = True

    def _rm_callback(self, obj_):
        obj_.rm_callback = True

    def _replace_callback(self, obj_):
        obj_.replace_callback = True

    def _reset_ObjToAdd_init_state(self):
        for obj in self.to_add:
            obj.reset()
