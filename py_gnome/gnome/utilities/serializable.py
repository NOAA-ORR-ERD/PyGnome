'''
Created on Feb 15, 2013
'''
import copy
import inspect

import numpy as np

from gnome import GnomeId
from gnome.persist import Savable
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.persist import class_from_objtype
from gnome.persist.base_schema import CollectionItemsList


class Field(object):  # ,serializable.Serializable):
    '''
    Class containing information about the property to be serialized
    '''

    def __init__(self, name,
                 isdatafile=False,
                 update=False, save=False, read=False,
                 save_reference=False,
                 iscollection=False,
                 test_for_eq=True):
        """
        Constructor for the Field object.
        The Field object is used to describe the property of an object.
        For instance, if a property is required to re-create the object from
        a persisted _state, its 'save' attribute is True.
        If the property describes a data file that will need to be moved
        when persisting the model, isdatafile should be True.
        The gnome.persist.scenario module contains a Scenario class that loads
        and saves a model. It looks for these attributes to correctly save/load
        it.

        It sets all attributes to False by default.

        :param str name: Name of the property being described by this Field
            object
        :param bool isdatafile=False: Is the property a datafile that should be
            moved during persistence?
        :param bool update=False: Is the property update-able by the web app?
        :param bool save=False: Is the property required to re-create the
            object when loading from a save file?
        :param bool read=False: If property is not updateable, perhaps make it
            read only so web app has information about the object
        :param bool save_reference=False: bool with default value of False.
            if the property is object, you can either
            serialize the object and store it as a nested structure or just
            store a reference to the object. For instance, the WindMover
            contains a Wind object and a Weatherer could also contain the
            same wind object, in this case, the 'wind' property should be
            stored as a reference. The Model.load function is responsible for
            hooking up the correct Wind object to the WindMover, Weatherer etc

            .. note:: save_reference currently is only used when the field is
                stored with 'save' flag.

        :param bool iscollection=True: bool with default value of True
            If the property is a collection (list, tuple, ordered collection),
            we will need to special-case treat it.

        :param bool test_for_eq=True: bool with default value of True
            when checking equality (__eq__()) of two gnome
            objects that are serializable, look for equality of attributes
            corresponding with fields with 'save'=True and 'test_for_eq'=True
            For instance, if a gnome.model.Model() object is saved, then loaded
            back from save file location, the filename attributes of objects
            that read data from file will point to different location. The
            objects are still equal. To avoid this problem, we can customize
            whether to use a field when testing for equality or not.

        """
        self.name = name
        self.isdatafile = isdatafile
        self.save = save
        self.update = update
        self.read = read
        self.save_reference = save_reference
        self.iscollection = iscollection
        self.test_for_eq = test_for_eq

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False

        if self.__dict__ == other.__dict__:
            return True

        return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        'unambiguous object representation'
        attr_vals = ', '.join(['{0}={1}'.format(attr, val)
                               for attr, val in self.__dict__.iteritems()])
        return ('{0.__class__.__module__}.{0.__class__.__name__}'
                '({1})'.format(self, attr_vals))

    def __str__(self):
        attr_vals = ', '.join(['{0}={1}'.format(attr, val)
                               for attr, val in self.__dict__.iteritems()])
        return ('Field Object: '
                '{1}'.format(self, attr_vals))


class State(object):

    def __init__(self, save=None, update=None, read=None, field=None):
        """
        Object keeps the list of properties that are output by
        Serializable.to_dict() method.
        Each list is accompanied by a keyword as defined below

        Object can be initialized as follows:
        >>> s = State(update=['update_field'], field=[Field('field_name')]

        Args:
        :param save: A list of strings which are properties that are
                       required to create new object when JSON is read from
                       save file.
                       Only the create properties are saved to save file.
        :type save:  A list of str
        :param update: A list of strings which are properties that can be
                       updated, so read/write capable
        :type update:  list containing str
        :param read:   A list of strings which are properties that are for
                       info, so readonly. It is not required for creating
                       new object.
        :type read:    list containing str
        :param field:  A field object or a list of field objects that should
                       be added to the State for persistence.
        :type field:   Field object or list of Field objects.

        For 'update', 'read', 'save', a Field object is created for each
        property in the list

        .. note:: Copy will create a new State object but reference original
                  lists.  Deepcopy will create new State object and new lists
                  for the attributes.
        """
        self.fields = []
        self.add_field(field)
        self.add(save=save, update=update, read=read)

        # define valid attributes for Field object
        # Field object is in flux - more properties were added so make
        # _valid_field_attr dynamic
        test_obj = Field('test')
        self._valid_field_attr = test_obj.__dict__.keys()

    def __deepcopy__(self, memo):
        '''
        deep copy of _state object so makes a copy of the fields list
        '''
        new_ = self.__class__()
        if new_ != self:
            new_.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new_

    def __eq__(self, other):
        """
        check for equality
        """
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__dict__ == other.__dict__

    def __contains__(self, name):
        """
        given the name as a string, it tests whether the 'fields' attribute
        contains a 'field' with user specified name. The 'fields' attribute is
        a list of 'Field' objects corresponding with properties that are
        serialized
        """
        if name in [f.name for f in self.fields]:
            return True

        return False

    def __len__(self):
        return len(self.fields)

    def __iadd__(self, field):
        self.add_field(field)
        return self

    def __isub__(self, l_names):
        self.remove(l_names)
        return self

    def __delitem__(self, l_names):
        self.remove(l_names)

    def __getitem__(self, names):
        return self.get_field_by_name(names)

    def __iter__(self):
        for field in self.fields:
            yield field

    def add_field(self, l_field):
        """
        Adds a Field object or a list of Field objects to fields attribute

        Either use this to add a property to the _state object or use the
        'add' method to add a property.  add_field gives more control since
        the attributes other than 'save','update','read' can be set directly
        when defining the Field object.
        """
        if l_field is None:
            return

        if isinstance(l_field, Field):
            l_field = [l_field]

        names = [field_.name for field_ in l_field]
        state_fieldnames = self.get_names()

        for name in names:
            if names.count(name) > 1:
                raise ValueError('List of field objects contains multiple '
                                 'fields with same name: '
                                 '{0}'.format(names.count(name), name))

            if name in state_fieldnames:
                raise ValueError('A Field object with the name {0} already '
                                 'exists - cannot add another with the '
                                 'same name'.format(name))

        # everything looks good, add the field

        for field_ in l_field:
            self.fields.append(field_)

    def add(self, save=None, update=None, read=None):
        """
        Only checks to make sure 'read' and 'update' properties are disjoint.
        Also makes sure everything is a list.

        Arguments:
        :param save: A tuple of strings which are properties that are required
                     to create new object when JSON is read from save file.
                     Only the save properties are saved to save file
        :type save: A tuple of str

        :param update: A tuple of strings which are properties that can be
                       updated, so read/write capable
        :type update: Tuple containing str

        :param read: A tuple of strings which are properties that are for info,
                     so readonly. It is not required for creating new object.
        :type read: Tuple containing str

        For 'update', 'read', 'save', a Field object is created for each
        property in the tuple
        """
        save = self._check_inputs(save)
        update = self._check_inputs(update)
        read = self._check_inputs(read)

        if len(set(update).intersection(set(read))) > 0:
            raise AttributeError('update (read/write properties) and '
                                 'read (readonly props) lists lists '
                                 'must be disjoint')

        fields = []
        for item in update:
            f = Field(item, update=True)
            if item in save:
                f.save = True
                save.remove(item)
            fields.append(f)

        for item in save:
            f = Field(item, save=True)
            if item in read:
                f.read = True
                read.remove(item)
            fields.append(f)

        [fields.append(Field(item, read=True)) for item in read]

        self.add_field(fields)  # now add these to self.fields

    def _check_inputs(self, vals):
        'check inputs and convert to a list'
        if vals is not None:
            if isinstance(vals, basestring):
                vals = [vals]
            if not isinstance(vals, (list, tuple)):
                raise ValueError('inputs must be a list of strings')
            vals = list(set(vals))
        else:
            vals = []
        return vals

    def remove(self, l_names):
        """
        Analogous to add method, this removes Field objects associated with
        l_names from the list.
        l_names is a list containing the names (string) of properties.
        """
        if isinstance(l_names, basestring):
            l_names = l_names,

        for name in l_names:
            field = self.get_field_by_name(name)
            if field == []:
                return
            # do not raise error - if field doesn't exist, do nothing
            #    raise ValueError('Cannot remove {0} since self.fields '
            #                     'does not contain a field '
            #                     'with this name'.format(name))

            self.fields.remove(field)

    def update(self, l_names, **kwargs):
        """
        update the attributes of an existing field
        Kwargs are key,value pairs defining the _state of attributes.
        It must be one of the valid attributes of Field object
        (see Field object __dict__ for valid attributes)
        :param update:     True or False
        :param save:     True or False
        :param read:       True or False
        :param isdatafield:True or False

        Usage:
        >>> _state = State(read=['test'])
        >>> _state.update('test', read=False, update=True, save=True,
        ...               isdatafile=True)

        .. note:: An exception will be raised if both 'read' and 'update' are
                  True for a given field
        """
        for key in kwargs.keys():
            if key not in self._valid_field_attr:
                raise AttributeError('{0} is not a valid attribute '
                                     'of Field object. '
                                     'It cannot be updated.'.format(key))

        if 'read' in kwargs.keys() and 'update' in kwargs.keys():
            if kwargs.get('read') and kwargs.get('update'):
                raise AttributeError("The 'read' and 'update' attribute "
                                     "cannot both be True")

        l_field = self.get_field_by_name(l_names)
        if not isinstance(l_field, list):
            l_field = [l_field]

        # need to make sure both read and update are not True for a field

        read_ = kwargs.pop('read', None)
        update_ = kwargs.pop('update', None)
        for field in l_field:
            for (key, val) in kwargs.iteritems():
                setattr(field, key, val)

            if read_ is not None and update_ is not None:  # change them both
                setattr(field, 'read', read_)
                setattr(field, 'update', update_)
            elif read_ is not None:
                if field.update and read_:
                    raise AttributeError("The 'read' and 'update' attribute "
                                         "cannot both be True")
                setattr(field, 'read', read_)
            elif update_ is not None:
                if field.read and update_:
                    raise AttributeError("The 'read' and 'update' attribute "
                                         "cannot both be True")
                setattr(field, 'update', update_)

        read_ = None
        if 'read' in kwargs.keys():
            read_ = kwargs.pop('read')

        for field in l_field:
            if read_ is not None:
                setattr(field, 'read', read_)

            for (key, val) in kwargs.iteritems():
                if key == 'update' and val is True:
                    if getattr(field, 'read'):
                        raise AttributeError("The 'read' and 'update' "
                                             "attribute cannot both be True")

                setattr(field, key, val)

    def get_field_by_name(self, names):
        'get field object from list given a name or list of names'
        if isinstance(names, basestring):
            names = [names]

        out = [field for name in names
               for field in self.fields
               if name == field.name]

        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_field_by_attribute(self, attr):
        'returns a list of fields where attr is true'
        if attr not in self._valid_field_attr:
            raise AttributeError('{0} is not valid attribute. '
                                 'Field.__dict__ contains: '
                                 '{1}'.format(attr, self._valid_field_attr))

        out = [field for field in self.fields if getattr(field, attr)]
        return out

    def get_names(self, attr='all'):
        """
        Returns the property names in self.fields. Can return all field names,
        or fieldnames with an attribute equal to True.
        attr can also be a list:

        >>> _state = State(read=['t0'],save=['t0','t1'])
        >>> _state.get_names(['read','save'])    # returns 't0'
        >>> _state.get_names('save')     # returns ['t0', 't1']
        """
        if attr == 'all':
            return [field_.name for field_ in self.fields]

        names = []

        if isinstance(attr, list):
            for at in attr:
                names.extend([field_.name for field_ in self.fields
                              if getattr(field_, at)])
        else:
            if attr not in self._valid_field_attr:
                raise AttributeError('{0} is not valid attribute. '
                                     'Field.__dict__ contains: {1}'
                                     .format(attr, self._valid_field_attr))

            names.extend([field_.name for field_ in self.fields
                          if getattr(field_, attr)])

        return names


class Serializable(GnomeId, Savable):

    """
    contains the to_dict and update_from_dict method to output properties of
    object in a list.

    This class is intended as a mixin so to_dict and update_from_dict become
    part of the object and the object must define a _state attribute of type
    State().

    It mixes in the GnomeId class since all Serializable gnome objects will
    have an Id as well

    The default _state=State(save=['id']) is a static variable for this class
    It uses the same convention as State to obtain the lists, 'update' for
    updating  properties, 'read' for read-only properties and 'save' for a
    list of properties required to create new object.

    The default _state contains 'id' in the save list. This is because all
    objects in a Model need 'id' to create a new one.

    Similarly, 'obj_type' is required for all objects, this is so the scenario
    module knows which object to create when loading from file.
    A default implementation of obj_type_to_dict exists here.

    Used obj_type instead of type because type is a builtin in python and
    didn't want to use the same name. The obj_type contains the type of the
    object as a string.
    """

    _state = State(save=('obj_type', 'name'), read=('obj_type', 'id'),
                   update=('name',))

    @classmethod
    def _restore_attr_from_save(cls, new_obj, dict_):
        '''
        restore attributes from save files that are not set during init - broke
        out some functionality of new_from_dict so when child classes override
        it, they can make use of this as well
        '''
        # remove following since they are not attributes of object
        for key in ['obj_type', 'id']:
            if key in dict_:
                del dict_[key]

        # set remaining attributes to restore state of object when it was
        # persisted to save files (ie could be mid-run)
        for key in dict_.keys():
            if not hasattr(new_obj, key):
                raise AttributeError('{0} is not an attribute '
                                     'of {1}'.format(key, cls.__name__))
            try:
                setattr(new_obj, key, dict_[key])
            except AttributeError:
                print 'failed to set attribute {0}'.format(key)
                raise

    @classmethod
    def new_from_dict(cls, dict_):
        """
        creates a new object from dictionary

        This is base implementation and can be over-ridden by classes using
        this mixin
        """
        rqd = {}
        for parent in cls.mro():
            if inspect.ismethod(parent.__init__):
                kwargs = inspect.getargspec(parent.__init__)[0][1:]

                # pop kwargs for object creation into rqd dict
                rqd.update({key: dict_.pop(key) for key in kwargs
                            if key in dict_})

        # create object with required input arguments
        new_obj = cls(**rqd)

        if dict_.pop('json_') == 'save':
            cls._restore_attr_from_save(new_obj, dict_)
        else:
            # for webapi, ignore the readonly attributes and set only
            # attributes that are updatable. At present, the 'webapi' uses
            # new_from_dict to create a new object only. It does not restore
            # the state of a previously persisted object
            if dict_:
                new_obj.update_from_dict(dict_)

        msg = "constructed object {0}".format(new_obj.__class__.__name__)
        new_obj.logger.debug(new_obj._pid + msg)

        return new_obj

    def _attrlist(self, do=('update', 'read')):
        '''
        returns list of object attributes that need to be serialized. By
        default all fields that have 'update' == True are returned.
        By default it converts the 'update' list of the _state object to dict;
        however, do=('save',) or do=('update','read') will return the dict
        with the union of the corresponding lists.
        '''
        actions = {'update', 'save', 'read'}
        list_ = []

        for action in do:
            if action in actions:
                list_.extend(self._state.get_names(action))
            else:
                raise ValueError("input not understood. String must be one of "
                                 "following: 'update', 'save' or 'readonly'.")

        return list_

    def to_dict(self):
        """
        returns a dictionary containing the serialized representation of this
        object.

        For every field, if there is a method defined on the object such that
        the method name is `{field_name}_to_dict`, use the return value of that
        method as the field value.

        Note: any field in `list` that does not exist on the
        object and does not have a to_dict method will raise an AttributeError.

        NOTE: add the json_='webapi' key to be serialized so we know what the
        serialization is for
        """

        list_ = self._state.get_names('all')

        data = {}
        print "*******************"
        print "in to_dict"
        for key in list_:
            print "doing:", key

            value = self.attr_to_dict(key)
            if key == 'timeseries':
                print value
                print type(value)
                print value.dtype
            if hasattr(value, 'to_dict'):
                value = value.to_dict()  # recursive call

            if value is not None:
                # some issue in colander monkey patch and the Wind schema
                # if None values are not pruned - take them out for now
                # this also means the default values will not be applied
                # on serialized -- that's ok though since we don't define
                # defaults in colander
                data[key] = value
        print data['timeseries']
        return data

    def attr_to_dict(self, name):
        """
        refactor to_dict's functionality so child classes can convert a
        single attribute to_dict instead of doing a whole list of fields
        """
        to_dict_fn_name = '%s_to_dict' % name

        if hasattr(self, to_dict_fn_name):
            value = getattr(self, to_dict_fn_name)()
        else:
            value = getattr(self, name)

        return value

    def update_from_dict(self, data):
        """
        Update the attributes of this object using the dictionary ``data`` by
        looking up the value of each key in ``data``.
        The fields in self._state that have update=True are modified. The
        remaining keys in 'data' are ignored. The object's _state attribute
        defines what fields can be updated

        If an attribute has changed, then call 'update_attr' to update its
        value.

        :param data: dict containing state of object per the client
        :type data: dict
        :returns: True if something changed, False otherwise
        :rtype: bool

        .. note::

            Does not do updates on nested objects. If the attribute references
            another Serializable object, then the value is not a dict but
            rather the updated object. For instance, WindMover will receive:

              {..., 'wind': <Wind object>}

            as opposed to a nested dict of the 'wind' object. It is expected
            that the 'wind' object was updated by calling its own
            update_from_dict then added to this dict as the updated object.
        """
        list_ = self._state.get_names('update')
        updated = False

        for key in list_:
            if key not in data:
                continue

            if self.update_attr(key, data[key]):
                updated = True

        return updated

    def _attr_changed(self, current_value, received_value):
        '''
        Checks if an attribute passed back in a dict_ from client has changed.
        Returns True if changed, else False
        '''
        # first, we normalize our left and right args
        if (isinstance(current_value, np.ndarray) and
                isinstance(received_value, (list, tuple))):
            received_value = np.asarray(received_value)

        # For a nested object, check if it data contains a new object. If
        # object in data 'is' current_value then 'self' state has not
        # changed. Even if current_value == data[key], we still must update
        # it because it now references a new object
        if isinstance(current_value, Serializable):
            if current_value is not received_value:
                return True

        elif isinstance(current_value, OrderedCollection):
            if len(current_value) != len(received_value):
                return True
            for ix, item in enumerate(current_value):
                if item is not received_value[ix]:
                    return True
        else:
            try:
                if current_value != received_value:
                    return True
            except ValueError:
                # maybe an iterable - checking for
                # isinstance(current_value, collections.Iterable) fails for
                # string so just do a try/except
                if np.any(current_value != received_value):
                    return True

        return False

    def update_attr(self, name, value):
        '''
        update the attribute defined by 'name' with given 'value'

        If there is a method defined on the object such that the method name is
        `{name}_from_dict`, call that method with the field's data.

        If neither method exists, then set the field with the value from
        ``data`` directly on the object.

        :param name: name of attribute to be updated
        :type name: str
        :param value: the new value of the attribute
        :type value: depends on each attribute being updated
        '''
        current_value = getattr(self, name)
        if isinstance(current_value, OrderedCollection):
            return self._update_orderedcoll_attr(current_value, value)

        from_dict_fn_name = '%s_update_from_dict' % name
        if hasattr(self, from_dict_fn_name):
            return getattr(self, from_dict_fn_name)(value)

        if self._attr_changed(current_value, value):
            setattr(self, name, value)
            return True
        else:
            return False

    def _update_orderedcoll_attr(self, curr_oc, l_new_oc):
        '''
        update attribute of type OrderedCollection with items in list l_new_oc
        '''
        updated = False

        if len(l_new_oc) != len(curr_oc):
            updated = True
        elif any([x is not y for x, y in zip(curr_oc.values(), l_new_oc)]):
            updated = True

        if updated:
            curr_oc.clear()
            if l_new_oc:
                curr_oc += l_new_oc
        return updated

    def obj_type_to_dict(self):
        """
        returns object type to save in dict.
        This is base implementation and can be over-ridden
        """
        return '{0.__module__}.{0.__class__.__name__}'.format(self)

    def _check_type(self, other):
        'check basic type equality'
        if self is other:
            return True

        if type(self) == type(other):
            return True

        return False

    def __eq__(self, other):
        """
        .. function:: __eq__(other)

        Since this class is designed as a mixin with one objective being to
        save _state of the object, then recreate a new object with the same
        _state.

        Define a base implementation of __eq__ so an object before persistence
        can be compared with a new object created after it is persisted.
        It can be overridden by the class with which it is mixed.

        It looks at attributes defined in self._state and checks the plain
        python types match.

        It does an allclose() check for numpy arrays using default atol, rtol.

        :param other: object of the same type as self that is used for
                      comparison in obj1 == other

        NOTE: super is not used.
        """

        if not self._check_type(other):
            return False

        for name in self._state.get_names('save'):
            if not self._state[name].test_for_eq:
                continue

            if hasattr(self, name):
                self_attr = getattr(self, name)
                other_attr = getattr(other, name)
            else:
                # not an attribute, let attr_to_dict call appropriate function
                # and check the dicts are equal
                self_attr = self.attr_to_dict(name)
                other_attr = other.attr_to_dict(name)

            if not isinstance(self_attr, np.ndarray):
                if isinstance(self_attr, float):
                    if abs(self_attr - other_attr) > 1e-10:
                        return False
                elif self_attr != other_attr:
                    return False
            else:
                if not isinstance(self_attr, type(other_attr)):
                    return False

                if not np.allclose(self_attr, other_attr,
                                   rtol=1e-4, atol=1e-4):
                    return False

        return True

    def __ne__(self, other):
        return not self == other

    def to_serialize(self, json_='webapi'):
        '''
        Invoke to_dict() which converts all attributes defined in _state to
        dict.
        If json_='save', it subselects the Fields with save=True.
        If json_='webapi', it subselects Fields with (update=True, read=True)
        '''
        dict_ = self.to_dict()
        print "*******************"
        print "in to_serielize"
        if json_ == 'webapi':
            attrlist = self._attrlist()
        elif json_ == 'save':
            attrlist = self._attrlist(do=('save',))
        else:
            raise ValueError("desired json_ payload must be either for webapi "
                             "or for save files: ('webapi', 'save')")

        toserial = {}
        for key in attrlist:
            if key in dict_:
                # if attribute is None, then dict_ does not contain it
                if key == "timeseries":
                    print "processing", key
                if (hasattr(self, key) and
                        hasattr(getattr(self, key), 'to_serialize')):
                    # recursively call for nested objects
                    attr = getattr(self, key)
                    toserial[key] = attr.to_serialize(json_)
                else:
                    # not a nested object
                    toserial[key] = dict_[key]

        toserial['json_'] = json_
        print toserial['timeseries']
        return toserial

    def serialize(self, json_='webapi'):
        """
        Convert the dict returned by object's to_dict method to valid json
        format via colander schema

        It uses the modules_dict defined in gnome.persist to find the correct
        schema module.

        :param json_: tells object whether serialization is for web content
            or for generating a save file.
        :returns: json format of serialized data

        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        c_fields = self._state.get_field_by_attribute('iscollection')

        if json_ == 'webapi':
            serial = schema.serialize(toserial)
            # check for collections
            for field in c_fields:
                serial[field.name] = \
                    self.serialize_oc(getattr(self, field.name), json_)
        elif json_ == 'save':
            for field in c_fields:
                # add a node for each collection, then serialize
                schema.add(CollectionItemsList(name=field.name))

            serial = schema.serialize(toserial)

        return serial

    @classmethod
    def is_sparse(cls, json_):
        '''
            Sparse means that we have a previously created object,
            and we are receiving a reference to the object using just
            an obj_type and an id.  This is a valid payload that the
            Web Client will send.
        '''
        r_attr_list = cls._state.get_names('read')
        r_attr_list.append('json_')
        attr_list = [attr for attr in json_ if attr not in r_attr_list]

        return len(attr_list) == 0

    def serialize_oc(self, coll, json_='webapi'):
        '''
        Serialize Model attributes of type ordered collection or list - an
        iterable containing serializable objects
        '''
        json_out = []
        for item in coll:
            json_out.append(item.serialize(json_))
        return json_out

    @classmethod
    def deserialize(cls, json_):
        """
            classmethod takes json structure as input, deserializes it using a
            colander schema then invokes the new_from_dict method to create an
            instance of the object described by the json schema.

            We also need to accept sparse json objects, in which case we will
            not treat them, but just send them back.
        """
        if not cls.is_sparse(json_):
            # if there are collections in object, dynamically update schema
            schema = cls._schema()
            c_fields = cls._state.get_field_by_attribute('iscollection')

            if json_['json_'] == 'webapi':
                _to_dict = schema.deserialize(json_)
                for field in c_fields:
                    if field.name in json_:
                        _to_dict[field.name] = \
                            cls.deserialize_oc(json_[field.name])
            elif json_['json_'] == 'save':
                for field in c_fields:
                    if field.name in json_:
                        # add a node for each collection, then deserialize
                        schema.add(CollectionItemsList(name=field.name))

                _to_dict = schema.deserialize(json_)

            return _to_dict
        else:
            return json_

    @classmethod
    def deserialize_oc(cls, json_):
        '''
        check contents of orderered collections to figure out what schema to
        use.
        Basically, the json serialized ordered collection looks like a regular
        list.
        '''
        deserial = []
        for item in json_:
            d_item = cls._deserialize_nested_obj(item)
            deserial.append(d_item)

        return deserial

    @classmethod
    def _deserialize_nested_obj(cls, json_):
        if json_ is not None:
            py_class = class_from_objtype(json_['obj_type'])

            return py_class.deserialize(json_)
        else:
            return None

    def _collection_to_dict(self, coll):
        '''
        Method takes the instance of a collection containing serializable gnome
        objects and outputs a list of dicts, each with two fields:
            {obj_type: object type <module.class>,
            id: IDs of each object.}
        This is a general way to make a dict for a collection which is then
        serialized to a 'save' file
        '''
        items = []

        for obj in coll:
            try:
                obj_type = '{0.__module__}.{0.__class__.__name__}'.format(obj)
            except AttributeError:
                obj_type = '{0.__class__.__name__}'.format(obj)
            item = {'obj_type': obj_type, 'id': str(obj.id)}
            items.append(item)

        return items
