'''
Created on Mar 25, 2013

Unit Tests for classes in serializable module
'''
import pytest

from gnome.utilities.serializable import State, Field

''' State object tests '''
update = ['test0', 'test1', 'test2']
read = ['read']
create = ['create']
create.extend(update)

read.sort()
create.sort()
update.sort()


def test_init_exceptions():
    state = State()

    with pytest.raises(ValueError):
        state.add(create='c_test', update='c_update', read='c_read')

    with pytest.raises(TypeError):
        state.add(test=['test'])  # 'test' is not a keyword

    with pytest.raises(AttributeError):
        # read and read/write props must be disjoint
        state.add(update=['test'], read=['test'])

    s = State(update=['update'], read=['read'], create=['create'])

    with pytest.raises(AttributeError):
        # test is not a valid attribute of Field outside what get_names expects
        s.get_names('test')

    with pytest.raises(ValueError):
        # no Field object with this name
        s.remove('xyz')

    with pytest.raises(ValueError):
        state.add(read=read, update=update, create=create)
        state.add(create=['read'])  # already exists


def test_state_init_field():
    ''' test init if a single Field object is given as opposed to a list '''
    state = State(field=Field('field0', create=True))
    assert len(state.fields) == 1


def test_state_init():
    '''
    Test initialization of state
    '''
    state = State(read=read, update=update, create=create,
                  field=[Field('field0', read=True),
                         Field('field1', create=True)])

    all_fields = []
    all_fields.extend(create)
    all_fields.extend(update)
    all_fields.extend(read)
    all_fields = list(set(all_fields))
    assert len(state.fields) == len(all_fields) + 2


def test_state_add():
    """
    Tests add function of State. The add function has the same arguments as
    init function, so state can also be initialized as follows:

    >>> state = State(read=read, update=update, create=create)
    """
    state = State()
    state.add(read=read, update=update, create=create)

    all_fields = []
    all_fields.extend(create)
    all_fields.extend(update)
    all_fields.extend(read)
    all_fields = list(set(all_fields))
    assert len(state.fields) == len(all_fields)

    for field in state.fields:
        if field.name in update:
            assert field.update
        if field.name in read:
            assert field.read
        if field.name in create:
            assert field.create

        assert not field.isdatafile


def test_state_add_field():
    """
    Tests the add_field functionality to add a field to state object.
    This can also be a list of field objects.
    """
    state = State()
    state.add_field(Field('test'))

    assert len(state.fields) == 1

    f = []
    f.append(Field('filename', create=True, isdatafile=True))
    f.append(Field('topology_file', create=True, isdatafile=True))

    state.add_field(f)
    assert len(state.fields) == 3

    for field in state.fields:
        assert field.name in ['test', 'filename', 'topology_file']

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
    """
    tests the removal of a field by name
    """
    state = State(read=read, update=update, create=create)
    state.remove('test0')
    assert state.get_field_by_name('test0') == []


def test_state_remove_list():
    """
    tests removal of multiple fields by a list of names
    """
    state = State(read=read, update=update, create=create)
    state.remove(['test0', 'create'])
    assert state.get_field_by_name('test0') == []
    assert state.get_field_by_name('create') == []


def test_state_get_field_by_name():
    """
    This returns the field object stored in state.fields by name.
    Since this returns the actual object stored and not the copy.
    Manipulating this object will change the state of the object
    stored in state.fields list.
    """
    state = State(read=read, update=update, create=create)
    field = state.get_field_by_name('test0')
    assert field.name == 'test0'
    assert field.create
    assert field.update
    assert not field.read


def test_state_get_field_by_name_list():
    """
    tests a list of field objects can be obtained by get_field_by_name
    """
    state = State(read=read, update=update, create=create)
    names = ['test0', 'read', 'test1']

    fields = state.get_field_by_name(names)
    assert len(fields) == len(names)

    for field in fields:
        assert field.name in names
        if field.name in update:
            assert field.update
        if field.name in read:
            assert field.read
        if field.name in create:
            assert field.create

        # make sure there is only one field for each name
        names.remove(field.name)


def test_get_field_by_attribute():
    """
    tests the fields can also be obtained by the attributes that are
    set to True.
    get_field_by_attribute function
    """
    state = State(read=read, update=update, create=create)
    state.add_field(Field('test', isdatafile=True))

    for field in state.get_field_by_attribute('read'):
        assert field.name in read

    for field in state.get_field_by_attribute('update'):
        assert field.name in update

    for field in state.get_field_by_attribute('create'):
        assert field.name in create

    for field in state.get_field_by_attribute('isdatafile'):
        assert field.name in ['test']


def test_state_get_names():
    """
    tests get_names function to get the names based on attributes
    """
    state = State(read=read, update=update, create=create)
    r_ = state.get_names('read')
    u_ = state.get_names('update')
    c_ = state.get_names('create')
    r_.sort()
    u_.sort()
    c_.sort()

    assert r_ == read
    assert u_ == update
    assert c_ == create


def test_state_get_names_list():
    """
    tests get_names function to get the names based on attributes
    """
    state = State(read=read, update=update, create=create)
    check = []
    check.extend(read)
    check.extend(create)
    check.sort()

    l_ = state.get_names(['read', 'create'])
    l_.sort()
    assert l_ == check


def test_update():
    """ tests that field's are updated correctly """
    state = State(read=read, create=create, update=update)
    state.update(['read', 'test0'], read=False, update=True, isdatafile=True)

    for field in state.fields:
        if field.name not in ['read', 'test0']:
            if field.name in update:
                assert field.update
            if field.name in read:
                assert field.read
            if field.name in create:
                assert field.create
            assert not field.isdatafile
        else:
            assert not field.read
            assert field.update
            assert field.isdatafile


def test_update_exceptions():
    """
    test exceptions are raised if update and read are both True
    for a field
    """
    state = State(read=read, create=create, update=update)
    with pytest.raises(AttributeError):
        state.update(['read', 'test0'], update=True)

    with pytest.raises(AttributeError):
        state.update(['read', 'test0'], read=True, update=True)

    with pytest.raises(AttributeError):
        state.update(['read', 'test0'], read=True)


""" Field object tests """


def test_field_eq():
    """ tests equality of Field object """
    assert Field('test') == Field('test')

    # all fields must match for equality
    assert Field('test') != Field('test', isdatafile=True)


def test_repr_str():
    """ tests repr and str work """
    repr(Field('test'))
    str(Field('test'))
    assert True
