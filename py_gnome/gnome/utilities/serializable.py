'''
Created on Feb 15, 2013
'''
import copy

import numpy

class Field(object):#,serializable.Serializable):
    """
    Class containing information about the property to be serialized
    """
    _attr = ['isdatafile','update','create','read']
    def __init__(self, name, isdatafile=False, update=False, create=False, read=False):
        self.name = name
        self.isdatafile = isdatafile
        self.create = create
        self.update = update
        self.read = read
        
    def __eq__(self, other):
        """ Check equality """
        if not isinstance(other, Field):
            return False
        
        if self.name == other.name and self.isdatafile == other.isdatafile and self.create == other.create and\
        self.update == other.update and self.read == other.read:
            return True
    
    def __repr__(self):
        """ unambiguous object representation """
        return "Field(name={0.name},update={0.update},create={0.create},read={0.read},isdatafile={0.isdatafile})".format(self)
    
    def __repr__(self):
        """ unambiguous object representation """
        return "Field(name={0.name},update={0.update},create={0.create},read={0.read},isdatafile={0.isdatafile})".format(self)

    def __str__(self):
        info = "Field object: name={0.name},update={0.update},create={0.create},read={0.read},isdatafile={0.isdatafile}".format(self)
        return info

class State(object):
    def __init__(self, **kwargs):
        """
        object keeps the list of properties that are output by Serializable.to_dict() method
        Each list is accompanied by a keyword as defined below
        
        'update' a list of strings which are properties that can be updated, so read/write capable
        'read'   a list of strings which are properties that are for info, so readonly. 
                 This is not required for creating new object
        'create' a list of strings which are properties that are required to create new object when 
                 JSON is read from save file
                The readonly properties are not saved in a file
                 
        'field'  a field object or a list of field objects that should be added to the State for persistence
                 
        For 'update', 'read', 'create', a Field object is create for each property in the list
        
        NOTE: copy will create a new State object but reference original lists
              deepcopy will create new State object and new lists for the attributes
        """
        self._check_kwargs(**kwargs)
        self.fields = []
        
        field = kwargs.pop('field',[])
        self.add_field(field)
        
        create_ = kwargs.pop('create',[])
        update_ = kwargs.pop('update',[])
        read_ = kwargs.pop('read',[])
        self.add(create=create_,update=update_,read=read_)
    
    def _check_kwargs(self, **kwargs):
        unknown = [key for key in kwargs.keys() if key not in ['read','create','update']]
        if len(unknown) > 0:
            raise ValueError("Only accepts keywords 'read', 'create', 'update'")

    def __copy__(self):
        new_ = type(self)()
        new_.__dict__.update(copy.copy(self.__dict__))
        return new_
    
    def __deepcopy__(self, memo):
        new_ = type(self)()
        new_.__dict__.update(copy.deepcopy(self.__dict__))
        return new_
    
    def add_field(self, l_field):
        """ Adds a Field object or a list of Field objects to fields attribute """
        if isinstance(l_field, Field):
            l_field = [l_field]
        
        #for field_ in l_field:
        #    if l_field.count(field_) > 1:
        #        raise ValueError("List of field objects contains a field repeated {0} times. The 'name' is {1}.".format(l_field.count(field_), field_.name) )
                
        names = [field_.name for field_ in l_field]
        state_fieldnames = self.get_names()
        
        for name in names:
            if names.count(name) > 1:
                raise ValueError("List of field objects contains multiple fields with same name: {1}".format(names.count(name), name))
            
            if name in state_fieldnames:
                raise ValueError("A Field object with the name {0} already exists - cannot add another with same name".format(name) )
        
        # everything looks good, add the field
        for field_ in l_field:
            self.fields.append(field_)
    
    def add(self, **kwargs):
        """
        Only checks to make sure 'read' and 'update' properties are disjoint. Also makes sure everything is a list. 
        
        Takes the same keyword, value pairs as __init__ method:
        'update' is list of properties that can be updated, so read/write capable
        'read'   is list of properties that are for info, so readonly. This is not required for creating new element
        'create' is list of properties that are required to create new object when JSON is read from save file
                 The readonly properties are not saved in a file
        add(update=['prop_name'] to add prop_name to list containing properties that can be updated
        """
        if not all([isinstance(vals,list) for vals in kwargs.values()]):
            raise ValueError("inputs must be a list of strings")
        
        self._check_kwargs(**kwargs)
        
        update_ = list( set( kwargs.pop('update',[])))
        read_ = list( set( kwargs.pop('read',[])))
        create_ = list( set(kwargs.pop('create',[])))
        
        if len( set(update_).intersection(set(read_)) ) > 0:
            raise ValueError('update (read/write properties) and read (readonly props) lists lists must be disjoint')
        
        fields = []
        """ Add items in update_. If item is in create_, then set 'create' flag. Read and update are disjoint. """
        for item in update_:
            f = Field(item, update=True)
            if item in create_:
                f.create = True
                create_.remove(item)
            fields.append(f)
        
        """ Add items in create_. If item is in read_, then set 'read' flag. """    
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
        Analogous to add method, this removes properties from the list
        Provide a list containing the names (string) of properties to be removed
        """
        if isinstance(l_names, basestring):
            l_names = [l_names]
        
        for name in l_names:
            field = self.get_field_by_name(name)
            if field == []:
                raise ValueError("Cannot remove {0} since self.fields does not contain a field with this name".format(name))
            
            self.fields.remove(field)
            

    def get_field_by_name(self, names):
        """ get field object from list given 'name' """
        if isinstance(names,basestring):
            names = [names]
            
        out = [field for name in names for field in self.fields if name == field.name]
        if len(out) == 1:
            return out[0]
        else:
            return out
    
    def get_field_by_attribute(self, attr):
        """ returns a list of fields where attr is true """
        test_obj = Field('test')
        if attr not in test_obj.__dict__.keys():
            raise ValueError("{0} is not valid. Field.__dict__ contains: ".format(attr, test_obj.__dict__.keys()))
            
        out = [field for field in self.fields if getattr(field, attr)]
        return out
    
    def get_names(self, attr='all'):
        """ returns the property names in self.fields. Can return all field names, or fieldnames with 
        an attribute equal to true """
        if attr == 'all':
            return [field_.name for field_ in self.fields]
        elif attr == 'update':
            return [field_.name for field_ in self.fields if field_.update]
        elif attr == 'create':
            return [field_.name for field_ in self.fields if field_.create]
        elif attr == 'read':
            return [field_.name for field_ in self.fields if field_.read]
        elif attr == 'isdatafile':
            return [field_.name for field_ in self.fields if field_.isdatafile]
        else:
            raise ValueError("{0} is unknown. attr can only be one of: 'update', 'create', 'read'".format(attr) )

#===============================================================================
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
#===============================================================================


class Serializable(object):
    """
    contains the to_dict and from_dict method to output properties of object in a list.
    
    This class is intended as a mixin so to_dict and from_dict become part of the object
    and the object must define a state attribute of type State().
    
    The default state=State(create=['id']) is a static variable for this class
    It uses the same convention as State to obtain the lists, 'update' for updating 
    properties, 'read' for readonly properties and 'create' for a list of properties
    required to create new object.
    
    The default state contains 'id' in the create list. This is because all objects
    in a Model need 'id' to create a new one.
    
    Similary, 'obj_type' is required for all objects, this is so the scenario module
    knows which object to create when loading from file. A default implementation of obj_type_to_dict
    exists here.
    
    Used obj_type instead of type because type is a builtin in python and 
    didn't want to use the same name. The obj_type contains the type of the object as a string.
    """
    state = State(create=['id','obj_type'])
    #===========================================================================
    # @classmethod
    # def add_state(cls, **kwargs):
    #    """
    #    Each class that mixes-in Serializable will contain a state attribute of type State.
    #    The state should be a static member for each subclass. It is static because instances
    #    of the class will all have the same field names for the state. 
    #    
    #    In addition, the state of the child class extends the state of the parent class.
    #    
    #    As such, this classmethod is available and used by each subclass in __init__ 
    #    to extend the definition of the parent class state attribute
    #    
    #    It recursively looks for 'state' attribute in base classes (cls.__bases__);
    #    gets the ('read','update','create') lists from each base class and adds;
    #    and creates a new State() object with its own lists and the lists of the parents
    #    
    #    NOTE: removes duplicates (repeated fields) from list. The lists in State refer to 
    #    attributes of the object. By default ['id'] in create list will end up duplicated 
    #    if one of the base classes of cls already contained 'state' attribute
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
    #===========================================================================
        
    @classmethod
    def new_from_dict(cls, dict_):
        """
        creates a new object from dictionary
        
        This is base implementation and can be over-ridden by classes using mixin
        """
        return cls(**dict_)
    
    def to_dict(self,do='update'):
        """
        returns a dictionary containing the serialized representation of this
        object. By default it converts the 'update' list of the state object to dict;
        however, do='create' or do='read' will return the dict with the corresponding
        list.
        
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
            raise ValueError("input not understood. String must be one of following: 'update', 'create' or 'readonly'.")
        
        data = {}
        for key in list_:
            to_dict_fn_name = '%s_to_dict' % key

            if hasattr(self, to_dict_fn_name):
                value = getattr(self, to_dict_fn_name)()
            else:
                value = getattr(self, key)

            if hasattr(value, 'to_dict'):
                value = value.to_dict(do)   # recursively call on contained objects

            data[key] = value
        return data
    
        
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
        #return self._from_dict(self.state.update, data)
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
                #===============================================================
                # try:
                #    setattr(self, key, value)
                # except AttributeError, err:
                #    print err.args
                #    print err.message
                #    print "==========="
                #    raise AttributeError("Failed to set {0}".format(key))
                #===============================================================
        #return self    # not required
    
    def obj_type_to_dict(self):
        """returns object type to save in dict. This is base implementation and can be over-ridden"""
        return "{0.__module__}.{0.__class__.__name__}".format( self)
        
    def __eq__(self, other):
        """
        Since this class is designed as a mixin with one objective being to save state of the object,
        then recreate a new object with the same state.
        
        Define a base implementation of __eq__ so an object before persistence can be compared with a
        new object created after it is persisted. It can be overridden by the class with which it is mixed.
        
        It calls to_dict(self.state.create) on both and checks the plain python types match. Since attributes
        defined in state.create maybe different from attributes defined in the object, to_dict is used
        
        It does not compare numpy arrays - only the plain python types         
        
        NOTE: This class does not have __init__ method and super is not used.
        """
        if type(self) != type(other):
            return False
        
        self_dict = self.to_dict('create')
        other_dict= other.to_dict('create')
        
        for val in self_dict:
            if not isinstance( self_dict[val], numpy.ndarray):
                if self_dict[val] != other_dict[val]:
                    return False 
                
        return True
            