#!/usr/bin/env python

from uuid import uuid1, UUID
import copy


class GnomeId(object):
    '''
    A class for assigning a unique ID for an object
    '''
    _id = None

    def __init__(self, id=None):
        '''
        Sets the ID of the object.

        :param id: (optional) input valid UUID to set
        '''
        if id is not None:
            if isinstance(id, UUID):
                self._id = str(id)
            elif isinstance(id, basestring):
                # check if our string is a valid uuid.  Throws an error if not
                UUID(id)
                self._id = id
            else:
                raise ValueError('id cannot be set. It is not a UUID object, '
                                 'nor a valid UUID in string format')
        else:
            self.__create_new_id()

    @property
    def id(self):
        '''
        Override this method for more exotic forms of identification.

        :return: a unique ID assigned during construction
        '''
        return self._id

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

        We need this, as we don't want the spill_nums copied,
        but do want everything else.

        got the method from:
            http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy()
        ends up using recursion
        """
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.deepcopy(self.__dict__, memo)
        obj_copy.__create_new_id()
        return obj_copy

    def __copy__(self):
        '''
        might as well have copy, too.
        '''
        obj_copy = object.__new__(type(self))
        obj_copy.__dict__ = copy.copy(self.__dict__)
        obj_copy.__create_new_id()

        return obj_copy
