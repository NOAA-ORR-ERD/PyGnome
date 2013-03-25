'''
Created on Mar 25, 2013

Unit Tests for classes in serializable module 
'''
import pytest
import copy

from gnome.utilities.serializable import State

state = State()
update = ['test0','test1', 'test2']
read = ['read']
create = ['create']
create.extend(update)

def test_state_exceptions():
    with pytest.raises(ValueError):
        state.add(update='test')
        
def test_state_add():
    state.add(read=read, update=update, create=create)
    assert read.sort() == state.read.sort()
    assert update.sort() == state.update.sort()
    assert create.sort() == state.create.sort()
    
def test_state_remove():
    state = copy.deepcopy(state)
    state.remove(update=update, read=read, create=create)
    assert state.read == []
    assert state.create == []
    assert state.update == []

def test_state_remove():
    state.remove(update=['not_exist'], read=['not_exist'], create=['not_exist'])
    assert state.read == read
    assert state.create == create
    assert state.update == update
