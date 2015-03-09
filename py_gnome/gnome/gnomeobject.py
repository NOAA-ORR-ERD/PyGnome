#!/usr/bin/env python
import os
from uuid import uuid1, UUID
import copy
import logging


def init_obj_log(obj, setLevel=logging.INFO):
    '''
    convenience function for initializing a logger with an object
    the logging.getLogger() will always return the same logger so calling
    this multiple times for same object is valid.
    '''
    logger = logging.getLogger("{0.__class__.__module__}."
                               "{0.__class__.__name__}".format(obj))
    logger.propagate = True
    logger.setLevel = setLevel
    return logger


class AddLogger(object):
    '''
    Mixin for including a logger
    '''
    _log = None

    @property
    def logger(self):
        '''
        define attribute '_log'. If it doesn't exist, define it here.
        This is so we don't have to add it to all PyGnome classes - this
        property makes the logger available to each object.
        - default log_level is INFO
        '''
        if not self._log:
            self._log = init_obj_log(self)
        return self._log

    @property
    def _pid(self):
        '''
        returns os.getpid() as a string. Since we have multiple models, each
        running in its own process that is managed by multi_model_broadcast
        module, each debug log messages starts with os.getpid(). This function
        just returns a string that the gnome object can append to - don't
        want to keep typing this everywhere.
        '''
        return "{0} - ".format(os.getpid())


class GnomeId(AddLogger):
    '''
    A class for assigning a unique ID for an object
    '''
    _id = None

    @property
    def id(self):
        '''
        Override this method for more exotic forms of identification.

        :return: a unique ID assigned during construction
        '''
        if not self._id:
            self.__create_new_id()
        return self._id

    def __create_new_id(self):
        """
        Override this method for more exotic forms of identification.

        Used only for deep copy.
        Used to make a new object which is a copy of the original.
        """
        self._id = str(uuid1())

    def __deepcopy__(self, memo):
        """
        the deepcopy implementation

        We need this, as we don't want the id of spill object and logger
        object copied, but do want everything else.

        got the method from:
            http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy()
        ends up using recursion
        """
        obj_copy = object.__new__(type(self))

        if '_log' in self.__dict__:
            # just set the _log to None since it cannot be deepcopied
            # since logging.getLogger() is used to get the logger - can leave
            # this as None and the 'logger' property will automatically set
            # this the next time it is used
            self.__dict__['_log'] = None

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

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self == other

    @property
    def name(self):
        '''
        define as property in base class so all objects will have a name
        by default
        '''
        try:
            return self._name
        except AttributeError:
            self._name = self.__class__.__name__
            return self._name

    @name.setter
    def name(self, val):
        self._name = val
