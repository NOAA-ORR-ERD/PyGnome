'''
Created on Feb 15, 2013
'''

class Serializable(object):
    """
    Designed as a mix-in for adding to Gnome objects
    if they should have to_dict and from_dict methods
    """
    serializable_state = ['id'] # store id of each object
    serializable_readwrite = [] # fields that are are both read/write capable
    serializable_readonly = []  # fields that are read only attributes of object

    @classmethod
    def new_from_dict(cls, dict):
        """
        creates a new object from dictionary
        
        This is base implementation and can be over-ridden by classes using mixin
        """
        new_obj = cls(**dict)
        if dict.get('id'):
            new_obj.id = dict.get('id')  # let's assign this as well?

        return new_obj
    
    def state_to_dict(self):
        """
        returns a dictionary containing the serialized representation of this
        object, using self.serializable_state as a list of fields to be stored in the
        dict. 
        
        Under the hood, it calls _to_dict(serializable_state)
        """
        return self._to_dict(self.serializable_state)
    
    def readonly_to_dict(self):
        """
        returns a dictionary containing the serialized representation of this
        object's read only attributes, using self.serializable_readonly as a list of fields 
        to be stored in the dict. 
        
        Under the hood, it calls _to_dict(serializable_readonly)
        
        Note: This returns a dict that contains readonly fields
        """
        return self._to_dict(self.serializable_readonly)
    
    def to_dict(self):
        """
        returns a dictionary containing the serialized representation of this
        object, using self.serializable_readwrite as a list of fields to be stored in the
        dict. 
        
        Under the hood, it calls _to_dict(serializable_readwrite)
        
        Note: This returns a dict that contains both read/write fields
        """
        return self._to_dict(self.serializable_readwrite)
        
    def from_dict(self, data):
        """
        sets state of the object using dictionary 'data'. The keys in 'data' must
        be a subset of the items in self.serializale_readwrite
        
        Under the hood, it calls _from_dict(serializable_readwrite, data)
        """
        self._from_dict(self.serializable_readwrite, data)
        
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
                    
        return self
