#!/usr/bin/env python

import pytest

import gnome
from gnome import movers, weather
from gnome.utilities.orderedcollection import *

class TestOrderedCollection(object):
    def test_init(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert oc.dtype == int
        oc = OrderedCollection([1,2,3,4,5], int)
        assert oc.dtype == int
        oc = OrderedCollection(dtype=int)
        assert oc.dtype == int

        with pytest.raises(TypeError):
            oc = OrderedCollection() # either a populated list or a dtype is required

        with pytest.raises(TypeError):
            oc = OrderedCollection('not a list')

        with pytest.raises(TypeError):
            oc = OrderedCollection([]) # either a populated list or a dtype is required

        with pytest.raises(TypeError):
            oc = OrderedCollection([1,2,3,4,5], float)

    def test_len(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert len(oc) == 5

    def test_iter(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert [i for i in oc] == [1,2,3,4,5]

    def test_contains(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert id(5) in oc

    def test_getitem(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert oc[id(3)] == 3
        with pytest.raises(KeyError):
            l__temp = oc[id(6)]

    def test_setitem(self):
        oc = OrderedCollection([1,2,3,4,5])
        oc[id(6)] = 6
        assert [i for i in oc] == [1,2,3,4,5,6]
        oc[id(4)] = 7
        assert [i for i in oc] == [1,2,3,7,5,6]

    def test_delitem(self):
        oc = OrderedCollection([1,2,3,4,5])
        with pytest.raises(KeyError):
            del oc[id(6)]
        del oc[id(4)]
        assert [i for i in oc] == [1,2,3,5]

    def test_iadd(self):
        oc = OrderedCollection([1,2,3,4,5])
        oc += 6
        assert [i for i in oc] == [1,2,3,4,5,6]
        oc += [7,8,9]
        assert [i for i in oc] == [1,2,3,4,5,6,7,8,9]

    def test_add(self):
        oc = OrderedCollection([1,2,3,4,5])
        oc.add(6)
        assert [i for i in oc] == [1,2,3,4,5,6]
        with pytest.raises(TypeError):
            oc.add('not an int')

    def test_remove(self):
        oc = OrderedCollection([1,2,3,4,5])
        with pytest.raises(KeyError):
            oc.remove(id(6))
        oc.remove(id(4))
        assert [i for i in oc] == [1,2,3,5]

    def test_replace(self):
        oc = OrderedCollection([1,2,3,4,5])
        oc.replace(id(6), 6)
        assert [i for i in oc] == [1,2,3,4,5,6]
        oc.replace(id(4), 7)
        assert [i for i in oc] == [1,2,3,7,5,6]
        assert oc[id(7)] == 7
        with pytest.raises(KeyError):
            # our key should also be gone after the delete
            l__temp = oc[id(4)]
        with pytest.raises(TypeError):
            oc.replace(id(7), 'not an int')

    def test_with_movers(self):
        mover_1 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
        mover_2 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
        mover_3 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))
        mover_4 = movers.simple_mover.SimpleMover(velocity=(1.0, -1.0, 0.0))

        # test our init, iter, get, and len methods
        mymovers = OrderedCollection([mover_1, mover_2], dtype=gnome.movers.Mover)
        assert [m for m in mymovers] == [mover_1, mover_2]
        assert mymovers[mover_1.id] ==  mover_1
        assert len(mymovers) == 2

        # test our add methods
        mymovers = OrderedCollection(dtype=gnome.movers.Mover)
        mymovers += mover_1
        mymovers += mover_2
        assert [m for m in mymovers] == [mover_1, mover_2]

        mymovers = OrderedCollection(dtype=gnome.movers.Mover)
        mymovers += [mover_1, mover_2]
        assert [m for m in mymovers] == [mover_1, mover_2]

        # test our del method
        mymovers = OrderedCollection([mover_1, mover_2, mover_3], dtype=gnome.movers.Mover)
        del mymovers[mover_2.id]
        assert [m for m in mymovers] == [mover_1, mover_3]

        # test our replace method
        mymovers = OrderedCollection([mover_1, mover_2, mover_3], dtype=gnome.movers.Mover)
        mymovers[mover_2.id] = mover_4
        assert [m for m in mymovers] == [mover_1, mover_4, mover_3]
        assert mymovers[mover_4.id] == mover_4




