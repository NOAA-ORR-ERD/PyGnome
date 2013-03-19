'''
Created on Feb 15, 2013
'''

import numpy

class State(object):
    def __init__(self, **kwargs):
        """
        object keeps the list of properties that are output by Serializable.to_dict() method
        Each list is accompanied by a keyword as defined below
        
        'update' is list of properties that can be updated, so read/write capable
        'read'   is list of properties that are for info, so readonly. This is not required for creating new element
        'create' is list of properties that are required to create new object when JSON is read from save file
                 The readonly properties are not saved in a file
                 
        NOTE: Since this object only contains lists, standard copy and deepcopy work fine.
              copy will create a new State object but reference original lists
              deepcopy will create new State object and new lists for the attributes
        """
        self.update = []
        self.create = ['id']
        self.read = []
        self._add_to_lists(**kwargs)
        
        
    def add(self,**kwargs):
        """
        There is no check to make sure the lists of properties are disjoint.
        Takes the same keyword, value pairs as __init__ method:
        add(update=['prop_name'] to add prop_name to list containing properties that can be updated
        """
        self._add_to_lists(**kwargs)
        
    def _add_to_lists(self, **kwargs):
        """
        Make sure update list and read lists are disjoint
        """
        if not all([isinstance(vals,list) for vals in kwargs.values()]):
            raise ValueError("inputs for State object must be a list of strings")
        
        update_ = list( set( kwargs.pop('update',[])))
        read_ = list( set( kwargs.pop('read',[])))
        create_ = list( set(kwargs.pop('create',[])))
        
        if len( set(update_).intersection(set(read_)) ) > 0:
            raise ValueError('update (read/write properties) and read (readonly props) lists lists must be disjoint')
        
        self.update.extend( update_ )  # unique elements
        self.read.extend( read_)
        self.create.extend( create_)
        
        
    def get(self):
        """
        Returns a dict containing the 'update', 'read' and 'create' lists 
        """
        return {'update':self.update,'read':self.read,'create':self.create}


class Serializable(object):
    """
    contains the to_dict and from_dict method to output properties of object in a list.
    
    This class is intented as a mixin so to_dict and from_dict become part of the object
    and the object must define a state attribute of type State().
    
    The default state=State() is a static variable for this class
    It uses the same convention as State to obtain the lists, 'update' for updating 
    properties, 'read' for readonly properties and 'create' for a list of properties
    required to create new object.
    """
    state = State()
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
        new_obj = cls(**dict_)
        if dict_.get('id'):
            new_obj.id = dict_.get('id')  # let's assign this as well?

        return new_obj
    
    def to_dict(self,do='update'):
        """
        returns a dictionary containing the serialized representation of this
        object. By default it converts the 'update' list of the state object to dict;
        however, do='create' or do='read' will return the dict with the corresponding
        list.
        
        Under the hood, it calls _to_dict(self.state.update) if do='update'
        """
        if do == 'update':
            return self._to_dict(self.state.update)
        
        elif do == 'create':
            return self._to_dict(self.state.create)
        
        elif do == 'read':
            return self._to_dict(self.state.read)
        else:
            raise ValueError("input not understood. String must be one of following: 'update', 'create' or 'readonly'.")
        
    def from_dict(self, data):
        """
        modifies state of the object using dictionary 'data'. 
        Only the self.state.update list contains properties that can me modified for existing object
        
        Under the hood, it calls _from_dict(self.state.update, data)
        """
        return self._from_dict(self.state.update, data)
        
    def _to_dict(self, list_):
        """
        Return a dictionary containing the serialized representation of this
        object, using provided input list to look up fields on the object
        that the dictionary should contain.

        For every field, if there is a method defined on the object such that
        the method name is `{field_name}_to_dict`, use the return value of that
        method as the field value.

        Note: any field in `list` that does not exist on the
        object and does not have a to_dict method will raise an AttributeError.
        """
        data = {}

        for key in list_:
            to_dict_fn_name = '%s_to_dict' % key

            if hasattr(self, to_dict_fn_name):
                value = getattr(self, to_dict_fn_name)()
            else:
                value = getattr(self, key)

            if hasattr(value, 'to_dict'):
                value = value.to_dict()

            data[key] = value
        return data
    
    def _from_dict(self, list_, data):
        """
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
            