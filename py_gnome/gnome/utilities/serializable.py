'''
Created on Feb 15, 2013
'''

import copy
import os
import shutil

import numpy
np = numpy

import gnome
from gnome.persist import (
    modules_dict,
    environment_schema,     # used implicitly by eval()
    model_schema,           # used implicitly by eval()
    movers_schema,          # used implicitly by eval()
    weatherers_schema,      # used implicitly by eval()
    spills_schema,          # used implicitly by eval()
    map_schema,             # used implicitly by eval()
    outputters_schema       # used implicitly by eval()
    )


class Field(object):  # ,serializable.Serializable):
    '''
    Class containing information about the property to be serialized
    '''
    def __init__(self, name,
                 isdatafile=False,
                 update=False, create=False, read=False,
                 save_reference=False,
                 test_for_eq=True):
        """
        Constructor for the Field object.
        The Field object is used to describe the property of an object.
        For instance, if a property is required to re-create the object from
        a persisted _state, its 'create' attribute is True.
        If the property describes a data file that will need to be moved
        when persisting the model, isdatafile should be True.
        The gnome.persist.scenario module contains a Scenario class that loads
        and saves a model.
        It looks for these attributes to correctly save/load it.

        It sets all attributes to False by default.
        :param str name: Name of the property being described by this Field
            object
        :param bool isdatafile: Is the property a datafile that should be
            moved during persistence?
        :param bool update: Is the property update-able by the web app?
        :param bool create: Is the property required to re-create the object
            when loading from a save file?
        :param bool read: If property is not updateable, perhaps make it
            read only so web app has information about the object
        :param bool save_reference: bool with default value of False.
            if the property is object, you can either
            serialize the object and store it as a nested structure or just
            store a reference to the object. For instance, the WindMover
            contains a Wind object and a Weatherer could also contain the
            same wind object, in this case, the 'wind' property should be
            stored as a reference. The Model.load function is responsible for
            hooking up the correct Wind object to the WindMover, Weatherer etc

            NOTE: save_reference currently is only used when the field is
            stored with 'create' flag.
        :param bool test_for_eq: bool with default value of True
            when checking equality (__eq__()) of two gnome
            objects that are serializable, look for equality of attributes
            corresponding with fields with 'create'=True and 'test_for_eq'=True
            For instance, if a gnome.model.Model() object is saved, then loaded
            back from save file location, the filename attributes of objects
            that read data from file will point to different location. The
            objects are still equal. To avoid this problem, we can customize
            whether to use a field when testing for equality or not.
        """
        self.name = name
        self.isdatafile = isdatafile
        self.create = create
        self.update = update
        self.read = read
        self.save_reference = save_reference
        self.test_for_eq = test_for_eq

    def __eq__(self, other):
        'Check equality'
        if not isinstance(other, Field):
            return False

        if self.name == other.name and self.isdatafile \
            == other.isdatafile and self.create == other.create \
            and self.update == other.update and self.read == other.read:
            return True

    def __repr__(self):
        'unambiguous object representation'
        obj = 'Field( '
        for attr, val in self.__dict__.iteritems():
            obj += '{0}={1}, '.format(attr, val)
        obj = obj[:-2] + obj[-2:].replace(', ', ' )')

        return obj

    def __str__(self):
        info = 'Field object: '
        for attr, val in self.__dict__.iteritems():
            info += '{0}={1}, '.format(attr, val)
        info = info[:-2] + info[-2:].replace(', ', '')

        return info


class State(object):
    def __init__(self, create=[], update=[], read=[], field=[]):
        """
        Object keeps the list of properties that are output by
        Serializable.to_dict() method.
        Each list is accompanied by a keyword as defined below

        Object can be initialized as follows:
        >>> s = State(update=['update_field'], field=[Field('field_name')]

        Args:
        :param update: A list of strings which are properties that can be
                       updated, so read/write capable
        :type update:  list containing str
        :param read:   A list of strings which are properties that are for
                       info, so readonly. It is not required for creating
                       new object.
        :type read:    list containing str
        :param create: A list of strings which are properties that are
                       required to create new object when JSON is read from
                       save file.
                       Only the create properties are saved to save file.
        :type create:  A list of str
        :param field:  A field object or a list of field objects that should
                       be added to the State for persistence.
        :type field:   Field object or list of Field objects.

        For 'update', 'read', 'create', a Field object is create for each
        property in the list

        .. note:: Copy will create a new State object but reference original
                  lists.  Deepcopy will create new State object and new lists
                  for the attributes.
        """
        self.fields = []
        self.add_field(field)
        self.add(create=create, update=update, read=read)

        # define valid attributes for Field object

        test_obj = Field('test')
        self._valid_field_attr = test_obj.__dict__.keys()

    def __copy__(self):
        '''
        shallow copy of _state object so references original fields list
        '''
        new_ = type(self)()
        new_.__dict__.update(copy.copy(self.__dict__))
        return new_

    def __deepcopy__(self, memo):
        '''
        deep copy of _state object so makes a copy of the fields list
        '''
        new_ = type(self)()
        new_.__dict__.update(copy.deepcopy(self.__dict__))
        return new_

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

        Either use this to add a property to the _state object or use the 'add' method to add a property.
        add_field gives more control since the attributes other than 'create','update','read' can be set
        directly when defining the Field object.
        """
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

    def add(self, create=[], update=[], read=[]):
        """
        Only checks to make sure 'read' and 'update' properties are disjoint. Also makes sure everything is a list. 
        
        Args:
        :param update: a list of strings which are properties that can be updated, so read/write capable
        :type update:  list containing str
        :param read:   a list of strings which are properties that are for info, so readonly. It is not required for creating new object.
        :type read:    list containing str
        :param create: a list of strings which are properties that are required to create new object when 
                        JSON is read from save file
                       Only the create properties are saved to save file
        :type create:  a list of str
        :param field:  a field object or a list of field objects that should be added to the State for persistence
        :type field:   Field object or list of Field objects

        For 'update', 'read', 'create', a Field object is create for each
        property in the list
        """
        if not all([isinstance(vals, list) for vals in [create, update,
                   read]]):
            raise ValueError('inputs must be a list of strings')

        update_ = list(set(update))
        read_ = list(set(read))
        create_ = list(set(create))

        if len(set(update_).intersection(set(read_))) > 0:
            raise AttributeError('update (read/write properties) and read (readonly props) lists lists must be disjoint'
                                 )

        fields = []
        for item in update_:
            f = Field(item, update=True)
            if item in create_:
                f.create = True
                create_.remove(item)
            fields.append(f)

        for item in create_:
            f = Field(item, create=True)
            if item in read_:
                f.read = True
                read_.remove(item)
            fields.append(f)

        [fields.append(Field(item, read=True)) for item in read_]

        self.add_field(fields)  # now add these to self.fields

    def remove(self, l_names):
        """
        Analogous to add method, this removes Field objects associated with l_names from the list
        Provide a list containing the names (string) of properties to be removed
        """

        if isinstance(l_names, basestring):
            l_names = [l_names]

        for name in l_names:
            field = self.get_field_by_name(name)
            if field == []:
                return
            # do not raise error - if field doesn't exist, do nothing
            #    raise ValueError('Cannot remove {0} since self.fields does not'
            #        ' contain a field with this name'.format(name))

            self.fields.remove(field)

    def update(self, l_names, **kwargs):
        """
        update the attributes of an existing field
        Kwargs are key,value pairs defining the _state of attributes.
        It must be one of the valid attributes of Field object (see Field object __dict__ for valid attributes) 
        :param update:     True or False
        :param create:     True or False
        :param read:       True or False
        :param isdatafield:True or False
        
        Usage:
        >>> _state = State(read=['test'])
        >>> _state.update('test',read=False,update=True,create=True,isdatafile=True)
        
        .. note::An exception will be raised if both 'read' and 'update' are True for a given field
        """
        for key in kwargs.keys():
            if key not in self._valid_field_attr:
                raise AttributeError('{0} is not a valid attribute of Field object. It cannot be updated.'.format(key))

        if 'read' in kwargs.keys() and 'update' in kwargs.keys():
            if kwargs.get('read') and kwargs.get('update'):
                raise AttributeError("The 'read' attribute and 'update' attribute cannot both be True"
                        )

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
                    raise AttributeError("The 'read' attribute and 'update' attribute cannot both be True"
                            )
                setattr(field, 'read', read_)
            elif update_ is not None:
                if field.read and update_:
                    raise AttributeError("The 'read' attribute and 'update' attribute cannot both be True"
                            )
                setattr(field, 'update', update_)

        read_ = None
        if 'read' in kwargs.keys():
            read_ = kwargs.pop('read')

        for field in l_field:
            if read_ is not None:
                setattr(field, 'read', read_)

            for (key, val) in kwargs.iteritems():
                if key == 'update' and val == True:
                    if getattr(field, 'read'):
                        raise AttributeError("The 'read' attribute and 'update' attribute cannot both be True"
                                )

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
            raise AttributeError('{0} is not valid attribute. Field.__dict__ contains: {1}'.format(attr,
                                 self._valid_field_attr))

        out = [field for field in self.fields if getattr(field, attr)]
        return out

    def get_names(self, attr='all'):
        """ returns the property names in self.fields. Can return all field names, or fieldnames with 
        an attribute equal to True. attr can also be a list:
        
        >>> _state = State(read=['t0'],create=['t0','t1'])
        >>> _state.get_names(['read','create'])    # returns 't0'
        >>> _state.get_names('create')     # returns ['t0', 't1']
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
                                     'Field.__dict__ contains: '
                                     '{1}'.format(attr, self._valid_field_attr)
                                     )

            names.extend([field_.name for field_ in self.fields
                         if getattr(field_, attr)])

        return names


class Serializable(object):

    """
    contains the to_dict and from_dict method to output properties of object
    in a list.

    This class is intended as a mixin so to_dict and from_dict become part of
    the object and the object must define a _state attribute of type State().

    The default _state=State(create=['id']) is a static variable for this class
    It uses the same convention as State to obtain the lists, 'update' for
    updating  properties, 'read' for read-only properties and 'create' for a
    list of properties required to create new object.

    The default _state contains 'id' in the create list. This is because all
    objects in a Model need 'id' to create a new one.

    Similarly, 'obj_type' is required for all objects, this is so the scenario
    module knows which object to create when loading from file.
    A default implementation of obj_type_to_dict exists here.

    Used obj_type instead of type because type is a builtin in python and
    didn't want to use the same name. The obj_type contains the type of the
    object as a string.
    """

    _state = State(create=['obj_type'])

    # =========================================================================
    # @classmethod
    # def add_state(cls, **kwargs):
    #    """
    #    Each class that mixes-in Serializable will contain a _state attribute
    #    of type State.
    #    The _state should be a static member for each subclass. It is static
    #    because instances of the class will all have the same field names for
    #    the _state.
    #
    #    In addition, the _state of the child class extends the _state of the
    #    parent class.
    #
    #    As such, this classmethod is available and used by each subclass in
    #    __init__ to extend the definition of the parent class _state attribute
    #
    #    It recursively looks for '_state' attribute in base classes
    #    (cls.__bases__); gets the ('read','update','create') lists from each
    #    base class and adds;
    #    and creates a new State() object with its own lists and the lists of
    #    the parents
    #
    #    NOTE: removes duplicates (repeated fields) from list. The lists in
    #    State refer to attributes of the object. By default ['id'] in create
    #    list will end up duplicated if one of the base classes of cls already
    #    contained '_state' attribute
    #    """
    #    print "add_state"
    #    update = kwargs.pop('update',[])
    #    create = kwargs.pop('create',[])
    #    read   = kwargs.pop('read',[])
    #    for obj in cls.__bases__:
    #        if '_state' in obj.__dict__:
    #            update.extend( obj._state.get()['update'] )
    #            create.extend( obj._state.get()['create'] )
    #            read.extend( obj._state.get()['read'] )
    #
    #    update = list( set(update) )
    #    create = list( set(create) )
    #    read = list( set(read) )
    #    cls._state = State(update=update, create=create, read=read)
    # =========================================================================

    @classmethod
    def new_from_dict(cls, dict_):
        """
        creates a new object from dictionary

        This is base implementation and can be over-ridden by classes using mixin
        """
        # remove obj_type from dict since that is only used by scenario
        # module to load objects
        # In baseclass, cls() is used to get the obj_type
        if 'obj_type' in dict_:
            dict_.pop('obj_type')

        return cls(**dict_)

    def to_dict(self, do='update'):
        """
        returns a dictionary containing the serialized representation of this
        object.
        By default it converts the 'update' list of the _state object to dict;
        however, do='create' or do='read' will return the dict with the
        corresponding list.

        For every field, if there is a method defined on the object such that
        the method name is `{field_name}_to_dict`, use the return value of that
        method as the field value.

        Note: any field in `list` that does not exist on the
        object and does not have a to_dict method will raise an AttributeError.
        """

        if do == 'update':
            list_ = self._state.get_names('update')
        elif do == 'create':
            list_ = self._state.get_names('create')
        elif do == 'read':
            list_ = self._state.get_names('read')
        else:
            raise ValueError("input not understood. String must be one of following: 'update', 'create' or 'readonly'."
                             )

        data = {}
        for key in list_:
#==============================================================================
#             to_dict_fn_name = '%s_to_dict' % key
# 
#             if hasattr(self, to_dict_fn_name):
#                 value = getattr(self, to_dict_fn_name)()
#             else:
#                 value = getattr(self, key)
#==============================================================================
            value = self.attr_to_dict(key)
            if hasattr(value, 'to_dict'):
                value = value.to_dict(do)  # recursively call on contained objects

            if value is not None:  # no need to persist properties that are None!
                data[key] = value

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

    def from_dict(self, data):
        """
        modifies _state of the object using dictionary 'data'. 
        Only the self._state.update list contains properties that can me modified for existing object
        
        Set the _state of this object using the dictionary ``data`` by looking up
        the value of each key in ``data`` that is also in  `list_`. Input list_ 
        contains the object's attributes (or fields) updated with data

        For every field, the choice of how to set the field is as follows:

        If there is a method defined on the object such that the method name is
        `{field_name}_from_dict`, call that method with the field's data.

        If the field on the object has a ``from_dict`` method, then use that
        method instead.

        If neither method exists, then set the field with the value from
        ``data`` directly on the object.
        """
        # return self._from_dict(self._state.update, data)

        list_ = self._state.get_names('update')

        for key in list_:
            if not key in data:
                continue

            from_dict_fn_name = '%s_from_dict' % key
            value = data[key]

            if hasattr(self, from_dict_fn_name):
                getattr(self, from_dict_fn_name)(value)
            elif hasattr(getattr(self, key), 'from_dict'):
                getattr(self, key).from_dict(value)
            else:
                setattr(self, key, value)

                # =============================================================
                # try:
                #    setattr(self, key, value)
                # except AttributeError, err:
                #    print err.args
                #    print err.message
                #    print "==========="
                #    raise AttributeError("Failed to set {0}".format(key))
                # =============================================================
        # return self    # not required

    def obj_type_to_dict(self):
        """
        returns object type to save in dict.
        This is base implementation and can be over-ridden
        """
        return '{0.__module__}.{0.__class__.__name__}'.format(self)

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

        It does not compare numpy arrays - only the plain python types. If an
        object's state contains numpy arrays like Wind object, it must override
        this method and do the comparison in its own class. This is especially
        useful when a Model object is recreated from mid-run save and the
        SpillContainer's data_arrays are repopulated. The arrays may not be
        exact so SpillContainer does the equality check for numpy arrays but
        can still use this base class for testing equality for all other
        attributes. Helps in following the DRY principle.

        :param other: object of the same type as self that is used for
                      comparison in obj1 == other

        NOTE: This class does not have __init__ method and super is not used.
        """

        if self is other:
            return True

        if type(self) != type(other):
            return False

        if (self._state.get_field_by_attribute('create') !=
            other._state.get_field_by_attribute('create')):
            return False

        for name in self._state.get_names('create'):
            if not self._state[name].test_for_eq:
                continue

            if hasattr(self, name):
                self_attr = getattr(self, name)
                other_attr = getattr(other, name)
            else:
                '''not an attribute, let attr_to_dict call appropriate function
                and check the dicts are equal'''
                self_attr = self.attr_to_dict(name)
                other_attr = other.attr_to_dict(name)

            if not isinstance(self_attr, np.ndarray):
                if self_attr != other_attr:
                    return False

        return True

    def __ne__(self, other):
        """
        Checks if the object is not equal to another object (!=).
        Complementary operator to ==, it calls self == other and if that fails,
        it returns True since the two objects are not equal.
        """

        if self == other:
            return False
        else:
            return True

    def serialize(self, do='update'):
        """
        Convert the dict returned by object's to_dict method to valid json
        format via colander schema

        It uses the modules_dict defined in gnome.persist to find the correct
        schema module.

        1. adds 'obj_type' field to _state for 'create' attribute so it is
                contained in serialized data. todo: check if this is needed
        2. do serialization and return json

        :param do: tells object where serialization is for update or for
            creating a new object
        :returns: json format of serialized data

        Note: creating a new object versus 'update' or 'read' has a different
            set of fields for serialization so 'do' is required.
            todo: revisit this to see if it still makes sense to have different
            attributes for different operations like 'update', 'create', 'read'
        """
        #dict_ = self._dict_to_serialize(do)
        dict_ = self.to_dict(do)
        to_eval = ('{0}.{1}()'
                   .format(modules_dict[self.__class__.__module__],
                       self.__class__.__name__))
        schema = eval(to_eval)
        json_ = schema.serialize(dict_)

        return json_

    @classmethod
    def deserialize(cls, json_):
        """
        classmethod takes json structure as input, deserializes it using a
        colander schema then invokes the new_from_dict method to create an
        instance of the object described by the json schema
        """
        #gnome_mod, obj_name = json_['obj_type'].rsplit('.', 1)
        gnome_mod = cls.__module__
        obj_name = cls.__name__
        to_eval = ('{0}.{1}().deserialize(json_)'
                   .format(modules_dict[gnome_mod], obj_name))
        _to_dict = eval(to_eval)

        return _to_dict
