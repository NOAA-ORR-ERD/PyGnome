'''
Created on Mar 25, 2013

Unit Tests for classes in serializable module
'''

import pytest

from gnome.utilities.serializable import State, Field

update = ['test0', 'test1', 'test2']
read = ['read']
save = ['save']
save.extend(update)

read.sort()
save.sort()
update.sort()


def test_init_exceptions():
    _state = State()

    with pytest.raises(TypeError):
        _state.add(test=['test'])  # 'test' is not a keyword

    with pytest.raises(AttributeError):
        # read and read/write props must be disjoint
        _state.add(update=['test'], read=['test'])

    s = State(update=['update'], read=['read'], save=['save'])

    with pytest.raises(AttributeError):
        # test is not a valid attribute of Field outside what get_names expects
        s.get_names('test')

    with pytest.raises(ValueError):
        _state.add(read=read, update=update, save=save)
        _state.add(save=['read'])  # already exists


def test_state_init_field():
    ''' test init if a single Field object is given as opposed to a list '''

    _state = State(field=Field('field0', save=True))
    assert len(_state.fields) == 1


@pytest.mark.parametrize(("save_", "update_", "read_"),
                         [('c_test', 'c_update', 'c_read'),
                          (['c_test'], ['c_update'], ['c_read']),
                          (('c_test',), ('c_update',), ('c_read',)),
                          ]
                         )
def test_init(save_, update_, read_):
    'test various ways to initialize'
    _state = State()
    _state.add(save=save_, update=update_, read=read_)
    assert True


def test_state_init():
    '''
    Test initialization of _state
    '''
    _state = State(read=read, update=update, save=save,
                   field=(Field('field0', read=True),
                          Field('field1', save=True)))

    all_fields = []
    all_fields.extend(save)
    all_fields.extend(update)
    all_fields.extend(read)
    all_fields = list(set(all_fields))
    assert len(_state.fields) == len(all_fields) + 2


def test_state_add():
    """
    Tests add function of State. The add function has the same arguments as
    init function, so _state can also be initialized as follows:

    >>> _state = State(read=read, update=update, save=save)
    """
    _state = State()
    _state.add(read=read, update=update, save=save)

    all_fields = []
    all_fields.extend(save)
    all_fields.extend(update)
    all_fields.extend(read)

    all_fields = list(set(all_fields))
    assert len(_state.fields) == len(all_fields)

    for field in _state.fields:
        if field.name in update:
            assert field.update
        if field.name in read:
            assert field.read
        if field.name in save:
            assert field.save

        assert not field.isdatafile


def test_state_add_field():
    """
    Tests the add_field functionality to add a field to _state object.
    This can also be a list of field objects.
    """
    _state = State()
    _state += Field('test')

    assert len(_state.fields) == 1

    f = []
    f.append(Field('filename', save=True, isdatafile=True))
    f.append(Field('topology_file', save=True, isdatafile=True))

    _state += f
    assert len(_state.fields) == 3

    for field in _state.fields:
        assert field.name in ['test', 'filename', 'topology_file']

        if field.name == 'filename' or field.name == 'topology_file':
            assert field.isdatafile
            assert field.save
            assert not field.update
            assert not field.read
        else:
            assert not field.isdatafile
            assert not field.save
            assert not field.update
            assert not field.read

    with pytest.raises(ValueError):
        _state += Field('test')

    with pytest.raises(ValueError):
        _state += [Field('test1'), Field('test1')]


def test_state_remove():
    """
    tests the removal of a field by name
    """
    _state = State(read=read, update=update, save=save)

    #_state.remove('test0')
    _state -= 'test0'
    assert _state.get_field_by_name('test0') == []


def test_state_remove_list():
    """
    tests removal of multiple fields by a list of names
    """
    _state = State(read=read, update=update, save=save)
    _state.remove(['test0', 'save'])

    assert _state.get_field_by_name('test0') == []
    assert _state.get_field_by_name('save') == []


def test_state_get_field_by_name():
    """
    This returns the field object stored in _state.fields by name.
    Since this returns the actual object stored and not the copy.
    Manipulating this object will change the _state of the object
    stored in _state.fields list.
    """
    _state = State(read=read, update=update, save=save)
    field = _state.get_field_by_name('test0')

    assert field.name == 'test0'
    assert field.save
    assert field.update
    assert not field.read


def test_state_get_field_by_name_list():
    """
    tests a list of field objects can be obtained by get_field_by_name
    """
    _state = State(read=read, update=update, save=save)
    names = ['test0', 'read', 'test1']

    fields = _state.get_field_by_name(names)
    assert len(fields) == len(names)

    for field in fields:
        assert field.name in names
        if field.name in update:
            assert field.update
        if field.name in read:
            assert field.read
        if field.name in save:
            assert field.save

        # make sure there is only one field for each name
        names.remove(field.name)


def test_get_field_by_attribute():
    """
    tests the fields can also be obtained by the attributes that are
    set to True.
    get_field_by_attribute function
    """
    _state = State(read=read, update=update, save=save)
    _state.add_field(Field('test', isdatafile=True))

    for field in _state.get_field_by_attribute('read'):
        assert field.name in read

    for field in _state.get_field_by_attribute('update'):
        assert field.name in update

    for field in _state.get_field_by_attribute('save'):
        assert field.name in save

    for field in _state.get_field_by_attribute('isdatafile'):
        assert field.name in ['test']


def test_state_get_names():
    """
    tests get_names function to get the names based on attributes
    """
    _state = State(read=read, update=update, save=save)
    r_ = _state.get_names('read')
    r_.sort()

    u_ = _state.get_names('update')
    u_.sort()

    c_ = _state.get_names('save')
    c_.sort()

    assert r_ == read
    assert u_ == update
    assert c_ == save


def test_state_get_names_list():
    """
    tests get_names function to get the names based on attributes
    """
    _state = State(read=read, update=update, save=save)
    check = []
    check.extend(read)
    check.extend(save)
    check.sort()

    l_ = _state.get_names(['read', 'save'])
    l_.sort()
    assert l_ == check


def test_update():
    """
    tests that field's are updated correctly
    """
    _state = State(read=read, save=save, update=update)
    _state.update(['read', 'test0'],
                  read=False, update=True, isdatafile=True)

    for field in _state.fields:
        if field.name not in ['read', 'test0']:
            if field.name in update:
                assert field.update
            if field.name in read:
                assert field.read
            if field.name in save:
                assert field.save
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
    _state = State(read=read, save=save, update=update)
    with pytest.raises(AttributeError):
        _state.update(['read', 'test0'], update=True)

    with pytest.raises(AttributeError):
        _state.update(['read', 'test0'], read=True, update=True)

    with pytest.raises(AttributeError):
        _state.update(['read', 'test0'], read=True)


def test_field_eq():
    """
    tests equality of Field object
    """
    assert Field('test') == Field('test')

    # all fields must match for equality
    assert Field('test') != Field('test', isdatafile=True)


def test_repr_str():
    """
    tests repr and str work
    """
    repr(Field('test'))
    str(Field('test'))
    assert True
