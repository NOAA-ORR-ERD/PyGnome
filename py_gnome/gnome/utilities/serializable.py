'''
Created on Feb 15, 2013
'''

import copy

import numpy
np = numpy

import gnome
from gnome import persist


class Field(object):  # ,serializable.Serializable):
    '''
    Class containing information about the property to be serialized
    '''
    def __init__(self, name,
                 isdatafile=False,
                 update=False, create=False, read=False):
        """
        Constructor for the Field object.
        The Field object is used to describe the property of an object.
        For instance, if a property is required to re-create the object from
        a persisted state, its 'create' attribute is True.
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
        """
        self.name = name
        self.isdatafile = isdatafile
        self.create = create
        self.update = update
        self.read = read

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
        return ('Field(name={0.name}, update={0.update}, create={0.create}, '
                'read={0.read}, isdatafile={0.isdatafile})'.format(self))

    def __str__(self):
        info = ('Field object: name={0.name}, update={0.update}, '
                'create={0.create}, read={0.read}, '
                'isdatafile={0.isdatafile}'.format(self))
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
        shallow copy of state object so references original fields list
        '''
        new_ = type(self)()
        new_.__dict__.update(copy.copy(self.__dict__))
        return new_

    def __deepcopy__(self, memo):
        '''
        deep copy of state object so makes a copy of the fields list
        '''
        new_ = type(self)()
        new_.__dict__.update(copy.deepcopy(self.__dict__))
        return new_

    def add_field(self, l_field):
        """
        Adds a Field object or a list of Field objects to fields attribute
        
        Either use this to add a property to the state object or use the 'add' method to add a property.
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
                raise ValueError('Cannot remove {0} since self.fields does not contain a field with this name'.format(name))

            self.fields.remove(field)

    def update(self, l_names, **kwargs):
        """
        update the attributes of an existing field
        Kwargs are key,value pairs defining the state of attributes.
        It must be one of the valid attributes of Field object (see Field object __dict__ for valid attributes) 
        :param update:     True or False
        :param create:     True or False
        :param read:       True or False
        :param isdatafield:True or False
        
        Usage:
        >>> state = State(read=['test'])
        >>> state.update('test',read=False,update=True,create=True,isdatafile=True)
        
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
        
        >>> state = State(read=['t0'],create=['t0','t1'])
        >>> state.get_names(['read','create'])    # returns 't0'
        >>> state.get_names('create')     # returns ['t0', 't1']
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


# ===============================================================================
# class State(object):
#    def __init__(self, **kwargs):
#        """
#        object keeps the list of properties that are output by Serializable.to_dict() method
#        Each list is accompanied by a keyword as defined below
#
#        'update' is list of properties that can be updated, so read/write capable
#        'read'   is list of properties that are for info, so readonly. This is not required for creating new element
#        'create' is list of properties that are required to create new object when JSON is read from save file
#                 The readonly properties are not saved in a file
#
#        NOTE: Since this object only contains lists, standard copy and deepcopy work fine.
#              copy will create a new State object but reference original lists
#              deepcopy will create new State object and new lists for the attributes
#        """
#        self.update = []
#        self.create = []
#        self.read = []
#        self._add_to_lists(**kwargs)
#
#
#    def add(self,**kwargs):
#        """
#        Only checks to make sure 'read' and 'update' properties are disjoint. Also makes sure everything is a list.
#
#        Takes the same keyword, value pairs as __init__ method:
#        'update' is list of properties that can be updated, so read/write capable
#        'read'   is list of properties that are for info, so readonly. This is not required for creating new element
#        'create' is list of properties that are required to create new object when JSON is read from save file
#                 The readonly properties are not saved in a file
#        add(update=['prop_name'] to add prop_name to list containing properties that can be updated
#        """
#        self._add_to_lists(**kwargs)
#
#    def remove(self,**kwargs):
#        """
#        Removes properties from the list. Provide a list containing the names of properties to be removed
#
#        Takes the same keyword, value pairs as __init__ method:
#        'update' is list of properties that can be updated, so read/write capable
#        'read'   is list of properties that are for info, so readonly. This is not required for creating new element
#        'create' is list of properties that are required to create new object when JSON is read from save file
#                 The readonly properties are not saved in a file
#        remove(update=['prop_name']) to remove prop_name from the list of properties that are updated ('update' list)
#        """
#        read_, update_, create_ = self._get_lists(**kwargs)
#        [self.read.remove(item) for item in read_ if item in self.read]
#        [self.update.remove(item) for item in update_ if item in self.update]
#        [self.create.remove(item) for item in create_ if item in self.create]
#
#
#    def _add_to_lists(self, **kwargs):
#        """
#        Make sure update list and read lists are disjoint
#        """
#        if not all([isinstance(vals,list) for vals in kwargs.values()]):
#            raise ValueError("inputs for State object must be a list of strings")
#
#        read_, update_, create_ = self._get_lists(**kwargs)
#
#        if len( set(update_).intersection(set(read_)) ) > 0:
#            raise ValueError('update (read/write properties) and read (readonly props) lists lists must be disjoint')
#
#        self.update.extend( update_ )  # unique elements
#        self.read.extend( read_)
#        self.create.extend( create_)
#
#    def _get_lists(self, **kwargs):
#        """
#        Internal method that just parses kwargs to get the update=[...], read=[...] and create=[...] lists
#        """
#        update_ = list( set( kwargs.pop('update',[])))
#        read_ = list( set( kwargs.pop('read',[])))
#        create_ = list( set(kwargs.pop('create',[])))
#
#        return read_, update_, create_
#
#
#    def get(self):
#        """
#        Returns a dict containing the 'update', 'read' and 'create' lists
#        """
#        return {'update':self.update,'read':self.read,'create':self.create}
# ===============================================================================

class Serializable(object):

    """
    contains the to_dict and from_dict method to output properties of object
    in a list.

    This class is intended as a mixin so to_dict and from_dict become part of
    the object and the object must define a state attribute of type State().

    The default state=State(create=['id']) is a static variable for this class
    It uses the same convention as State to obtain the lists, 'update' for
    updating  properties, 'read' for read-only properties and 'create' for a
    list of properties required to create new object.

    The default state contains 'id' in the create list. This is because all
    objects in a Model need 'id' to create a new one.

    Similarly, 'obj_type' is required for all objects, this is so the scenario
    module knows which object to create when loading from file.
    A default implementation of obj_type_to_dict exists here.

    Used obj_type instead of type because type is a builtin in python and
    didn't want to use the same name. The obj_type contains the type of the
    object as a string.
    """

    _state = State(create=['id'])

    # =========================================================================
    # @classmethod
    # def add_state(cls, **kwargs):
    #    """
    #    Each class that mixes-in Serializable will contain a state attribute
    #    of type State.
    #    The state should be a static member for each subclass. It is static
    #    because instances of the class will all have the same field names for
    #    the state.
    #
    #    In addition, the state of the child class extends the state of the
    #    parent class.
    #
    #    As such, this classmethod is available and used by each subclass in
    #    __init__ to extend the definition of the parent class state attribute
    #
    #    It recursively looks for 'state' attribute in base classes
    #    (cls.__bases__); gets the ('read','update','create') lists from each
    #    base class and adds;
    #    and creates a new State() object with its own lists and the lists of
    #    the parents
    #
    #    NOTE: removes duplicates (repeated fields) from list. The lists in
    #    State refer to attributes of the object. By default ['id'] in create
    #    list will end up duplicated if one of the base classes of cls already
    #    contained 'state' attribute
    #    """
    #    print "add_state"
    #    update = kwargs.pop('update',[])
    #    create = kwargs.pop('create',[])
    #    read   = kwargs.pop('read',[])
    #    for obj in cls.__bases__:
    #        if 'state' in obj.__dict__:
    #            update.extend( obj.state.get()['update'] )
    #            create.extend( obj.state.get()['create'] )
    #            read.extend( obj.state.get()['read'] )
    #
    #    update = list( set(update) )
    #    create = list( set(create) )
    #    read = list( set(read) )
    #    cls.state = State(update=update, create=create, read=read)
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
        By default it converts the 'update' list of the state object to dict;
        however, do='create' or do='read' will return the dict with the
        corresponding list.

        For every field, if there is a method defined on the object such that
        the method name is `{field_name}_to_dict`, use the return value of that
        method as the field value.

        Note: any field in `list` that does not exist on the
        object and does not have a to_dict method will raise an AttributeError.
        """

        if do == 'update':
            list_ = self.state.get_names('update')
        elif do == 'create':
            list_ = self.state.get_names('create')
        elif do == 'read':
            list_ = self.state.get_names('read')
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
        modifies state of the object using dictionary 'data'. 
        Only the self.state.update list contains properties that can me modified for existing object
        
        Set the state of this object using the dictionary ``data`` by looking up
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
        # return self._from_dict(self.state.update, data)

        list_ = self.state.get_names('update')

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

    #def obj_type_to_dict(self):
        """
        returns object type to save in dict.
        This is base implementation and can be over-ridden
        """
    #    return '{0.__module__}.{0.__class__.__name__}'.format(self)

    def __eq__(self, other):
        """
        .. function:: __eq__(other)

        Since this class is designed as a mixin with one objective being to
        save state of the object, then recreate a new object with the same
        state.

        Define a base implementation of __eq__ so an object before persistence
        can be compared with a new object created after it is persisted.
        It can be overridden by the class with which it is mixed.

        It calls to_dict(self.state.create) on both and checks the plain python
        types match.
        Since attributes defined in state.create maybe different from
        attributes defined in the object, to_dict is used

        It does not compare numpy arrays - only the plain python types

        :param other: object of the same type as self that is used for
                      comparison in obj1 == other

        NOTE: This class does not have __init__ method and super is not used.
        """

        if self is other:
            return True

        if type(self) != type(other):
            return False

        self_dict = self.to_dict('create')
        other_dict = other.to_dict('create')

        if len(self_dict) != len(other_dict):
            return False

        for val in self_dict:
            if not isinstance(self_dict[val], np.ndarray):
                if self_dict[val] != other_dict[val]:
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

    def serialize(self, dict_):
        to_eval = ('{0}.{1}().serialize(dict_)'
                   .format(persist.modules_dict[self.__class__.__module__],
                       self.__class__.__name__))
        json_ = eval(to_eval)
        return json_

