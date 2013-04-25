'''
Created on Mar 25, 2013

Unit Tests for classes in serializable module 
'''
import pytest
import copy

from gnome.utilities.serializable import State, Field

""" State object tests """
update = ['test0','test1', 'test2']
read = ['read']
create = ['create']
create.extend(update)

def test_state_exceptions():
    state = State()
    with pytest.raises(ValueError):
        state.add(update='test')    # Must be a list if it is a string
        
    with pytest.raises(ValueError):
        state.add(test=['test'])    # 'test' is not a keyword
        
    with pytest.raises(ValueError):
        state.add(update=['test'],read=['test'])    # read and read/write props must be disjoint
        
    s = State(update=['update'],read=['read'], create=['create'])    
    with pytest.raises(ValueError):
        s.get_names('test') # type_ outside what get_names expects
        
    with pytest.raises(ValueError):
        s.remove('xyz') # no Field object with this name
        
    with pytest.raises(ValueError):
        state.add(read=read, update=update, create=create)
        state.add(create=['read'])  # already exists
        
            
        
def test_state_add():
    state = State()
    state.add(read=read, update=update, create=create)
    all = []
    all.extend(create)
    all.extend(update)
    all.extend(read)
    all = list( set(all) )
    assert len(state.fields) == len(all)
    
    for field in state.fields:
        if field.name in update:
            assert field.update
        if field.name in read:
            assert field.read
        if field.name in create:
            assert field.create
            
        assert not field.isdatafile
            

def test_state_add_field():
    state = State()
    state.add_field(Field('test'))
    
    assert len(state.fields) == 1
    
    f = []
    f.append(Field('filename',create=True, isdatafile=True))
    f.append(Field('topology_file',create=True, isdatafile=True))
    state.add_field(f)
    assert len(state.fields) == 3
    
    for field in state.fields:
        assert field.name in ['test','filename','topology_file']
        
        if field.name == 'filename' or field.name == 'topology_file':
            assert field.isdatafile
            assert field.create
            assert not field.update
            assert not field.read
        else:
            assert not field.isdatafile
            assert not field.create
            assert not field.update
            assert not field.read
        
    with pytest.raises(ValueError):
        state.add_field(Field('test'))
        
    with pytest.raises(ValueError):
        state.add_field([Field('test1'), Field('test1')]) 
    
def test_state_remove():
    state = State(read=read, update=update, create=create)
    state.remove('test0')
    assert state.get_field_by_name('test0') == []
    
def test_state_remove_list():
    state = State(read=read, update=update, create=create)
    state.remove(['test0','create'])
    assert state.get_field_by_name('test0') == []
    assert state.get_field_by_name('create') == []


def test_state_get_field_by_name():
    state = State(read=read, update=update, create=create)
    field = state.get_field_by_name('test0')
    assert field.name == 'test0'
    assert field.create
    assert field.update
    assert not field.read
    
def test_state_get_field_by_name_list():
    state = State(read=read, update=update, create=create)
    names = ['test0','read','test1']
    fields= state.get_field_by_name(names)
    
    assert len(fields) == len(names)
    
    for field in fields:
        assert field.name in names
        if field.name in update:
            assert field.update
        if field.name in read:
            assert field.read
        if field.name in create:
            assert field.create
        names.remove(field.name)    # make sure there is only one field for each name

def test_get_field_by_attribute():
    state = State(read=read, update=update, create=create)
    state.add_field(Field('test',isdatafile=True))
    
    for field in state.get_field_by_attribute('read'):
        assert field.name in read
        
    for field in state.get_field_by_attribute('update'):
        assert field.name in update
        
    for field in state.get_field_by_attribute('create'):
        assert field.name in create

    for field in state.get_field_by_attribute('isdatafile'):
        assert field.name in ['test']
        
        
def test_state_get_names():
    state = State(read=read, update=update, create=create)
    assert state.get_names('read').sort() == read.sort()
    assert state.get_names('update').sort() == update.sort()
    assert state.get_names('create').sort() == create.sort()

""" Field object tests """
def test_field_eq():
    assert Field('test') == Field('test')
    assert Field('test') != Field('test',isdatafile=True)   # all fields must match for equality
    
def test_repr_str():
    repr(Field('test'))
    str(Field('test'))
    assert True
    
    