#!/usr/bin/env python

import pytest

import gnome
from gnome import movers
from gnome.utilities.orderedcollection import OrderedCollection

class TestOrderedCollection(object):
    def test_init(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert oc.dtype == int
        oc = OrderedCollection([1,2,3,4,5], int)
        assert oc.dtype == int
        oc = OrderedCollection(dtype=int)
        assert oc.dtype == int

        with pytest.raises(TypeError): # IGNORE:E1101
            oc = OrderedCollection() # either a populated list or a dtype is required

        with pytest.raises(TypeError): # IGNORE:E1101
            oc = OrderedCollection('not a list')

        with pytest.raises(TypeError): # IGNORE:E1101
            oc = OrderedCollection([]) # either a populated list or a dtype is required

        with pytest.raises(TypeError): # IGNORE:E1101
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
        with pytest.raises(KeyError): # IGNORE:E1101
            oc[id(6)] # IGNORE:W0106

    def test_setitem(self):
        oc = OrderedCollection([1,2,3,4,5])
        oc[id(6)] = 6
        assert [i for i in oc] == [1,2,3,4,5,6]
        oc[id(4)] = 7
        assert [i for i in oc] == [1,2,3,7,5,6]

    def test_delitem(self):
        oc = OrderedCollection([1,2,3,4,5])
        with pytest.raises(KeyError): # IGNORE:E1101
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
        with pytest.raises(TypeError): # IGNORE:E1101
            oc.add('not an int')

    def test_remove(self):
        oc = OrderedCollection([1,2,3,4,5])
        with pytest.raises(KeyError): # IGNORE:E1101
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
        with pytest.raises(KeyError): # IGNORE:E1101
            # our key should also be gone after the delete
            oc[id(4)] # IGNORE:W0106
        with pytest.raises(TypeError): # IGNORE:E1101
            oc.replace(id(7), 'not an int')

    def test_index(self):
        oc = OrderedCollection([1,2,3,4,5])
        assert oc.index(id(3)) == 2
        oc[id(3)] = 6
        assert oc.index(id(6)) == 2
        del oc[id(6)]
        assert oc.index(id(4)) == 2

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


def test_to_dict():
    """
    added a to_dict() method - test this method
    """
    items = [movers.simple_mover.SimpleMover(velocity=(i*0.5, -1.0, 0.0)) for i in range(2)]
    items.extend([movers.RandomMover() for i in range(2)])
    mymovers = OrderedCollection(items, dtype=gnome.movers.Mover)
    dict_ = mymovers.to_dict()
    
    assert dict_['dtype'] == mymovers.dtype
    for id, mv in enumerate(items):
        assert dict_['id_list'][id][0] == "{0}.{1}".format( mv.__module__, mv.__class__.__name__)
        assert dict_['id_list'][id][1] == mv.id
     
""" Define a helper class (mutable object) for use in TestCallbacks """
class ObjToAdd():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.add_callback = False
        self.rm_callback = False
        self.replace_callback = False

class TestCallbacks():
    to_add = [ObjToAdd(), ObjToAdd(), ObjToAdd()]
    
    def test_add_callback(self):
        """ test add callback is invoked after adding an object or list of objects """
        oc = OrderedCollection(dtype=ObjToAdd) # lets work with a mutable type
        oc.register_callback(self._add_callback, events=('add'))
        
        self._reset_ObjToAdd_init_state()    # check everything if False initially
        
        oc += self.to_add
        oc += ObjToAdd()
        
        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback 
         
        
    def test_remove_callback(self):
        """ test remove callback is invoked after removing an object """
        oc = OrderedCollection(dtype=ObjToAdd) # lets work with a mutable type
        oc.register_callback(self._rm_callback, events='remove')
        oc.register_callback(self._add_callback, events=('add'))
        
        self._reset_ObjToAdd_init_state()    # check everything if False initially
        
        oc += self.to_add
            
        del oc[id(self.to_add[0])]
        
        assert self.to_add[0].rm_callback 
        assert self.to_add[0].add_callback
        assert not self.to_add[0].replace_callback
        
        self.to_add[0].reset()  # reset all to false
        oc += self.to_add[0]    # let's add this back in
        
        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback
            
    def test_replace_callback(self):
        """ test replace callback is invoked after replacing an object """
        oc = OrderedCollection(dtype=ObjToAdd) # lets work with a mutable type
        oc.register_callback(self._replace_callback, events=('replace'))
        
        self._reset_ObjToAdd_init_state()    # check everything if False initially
        
        oc += self.to_add
        rep = ObjToAdd()
        oc[id(self.to_add[0])] = rep
        
        for obj in oc:
            assert not obj.add_callback
            assert not obj.rm_callback
            if id(obj) == id(rep):
                assert obj.replace_callback
            else:
                assert not obj.replace_callback
                
    def test_add_replace_callback(self):
        """ register one callback with multiple events (add, replace) """
        oc = OrderedCollection(dtype=ObjToAdd) # lets work with a mutable type
        oc.register_callback(self._add_callback, events=('add','replace'))
        
        self._reset_ObjToAdd_init_state()    # check everything if False initially
        
        oc += self.to_add
        
        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback
        
        rep = ObjToAdd()
        oc[id(self.to_add[0])] = rep
        
        for obj in oc:
            assert obj.add_callback
            assert not obj.rm_callback
            assert not obj.replace_callback
        
        
    """ Helper Functions """    
    def _add_callback(self, obj_):
        obj_.add_callback = True
        
    def _rm_callback(self, obj_):
        obj_.rm_callback = True
        
    def _replace_callback(self, obj_):
        obj_.replace_callback = True
    
    def _reset_ObjToAdd_init_state(self):
        for obj in self.to_add:
            obj.reset()
    """ End helper objects/functions """
    
