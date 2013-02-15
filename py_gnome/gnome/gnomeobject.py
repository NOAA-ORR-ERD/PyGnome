#!/usr/bin/env python

from uuid import uuid1,UUID
import copy

class GnomeObject(object):
    """
    The top-level base class in which all Gnome Objects are derived.

    Any global members or functionality will go in this class.
    """
    _id = None

    @property
    def id(self):
        """
        Override this method for more exotic forms of identification.

        :return: a unique ID returned by uuid.uuid1()
        """
        if not self._id:
            self.__create_new_id()
        return self._id
    
    @id.setter
    def id(self, value):
        """
        Define setter, but need to ask if this is what we want to do
        
        This can only be assigned if object does not yet have an id
        Do we want to on accept strings that are valid uuids?
        """
        # make sure string is valid UUID in string format
        if isinstance(value, str):
            UUID(value) # this will throw error if not a valid ID
            
        elif isinstance(value, UUID):
            value = str(value)
            
        if not self._id:
            self._id = value
        else:
            raise ValueError("UUID already assigned to object - cannot assign a new UUID")

    def __create_new_id(self):
        """
        Override this method for more exotic forms of identification.

        Used only for deep copy. 
        Used to make a new object which is a copy of the original.
        """
        self._id = str(uuid1())

    def __deepcopy__(self, memo=None):
        """
        the deepcopy implementation

        we need this, as we don't want the spill_nums copied, but do want everything else.

        got the method from:

        http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy() ends up using recursion
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.deepcopy(self.__dict__, memo)
        obj_copy.__create_new_id()
        return obj_copy

    def __copy__(self):
        """
        might as well have copy, too.
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = self.__dict__.copy()
        obj_copy.__create_new_id()
        return obj_copy
