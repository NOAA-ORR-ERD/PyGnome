#!/usr/bin/env python
import os
from uuid import uuid1, UUID
import copy
import logging

from gnome.exceptions import ReferencedObjectNotSet


def init_obj_log(obj, setLevel=logging.INFO):
    '''
    convenience function for initializing a logger with an object
    the logging.getLogger() will always return the same logger so calling
    this multiple times for same object is valid.

    By default adds a NullHandler() so we don't get errors if application
    using PyGnome hasn't configured a logging.
    '''
    logger = logging.getLogger("{0.__class__.__module__}."
                               "{0.__class__.__name__}".format(obj))
    logger.propagate = True
    logger.setLevel = setLevel
    logger.addHandler(logging.NullHandler())
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
        if self._log is None:
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
    make_default_refs = True

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

    def validate_refs(self, refs=['wind', 'water', 'waves']):
        '''
        level is the logging level to use for messages. Default is 'warning'
        but if called from prepare_for_model_run, we want to use error and
        raise exception.
        '''
        isvalid = True
        msgs = []
        prepend = self._warn_pre

        for attr in refs:
            if hasattr(self, attr) and getattr(self, attr) is None:
                msg = 'no {0} object defined'.format(attr)

                setattr(self.logger, 'warning', msg)
                msgs.append(prepend + msg)

                # if we get here, level is 'warning' or lower
                # if make_default_refs is True, object is valid since Model
                # will setup the missing references during run
                if not self.make_default_refs:
                    isvalid = False

        return (msgs, isvalid)

    def validate(self):
        '''
        All pygnome objects should be able to validate themselves. Many
        py_gnome objects reference other objects like wind, water, waves. These
        may not be defined when object is created so they can be None at
        construction time; however, they should reference valid objects
        when running model. If make_default_refs is True, then object isvalid
        because model will set these up at runtime. To raise exception
        for missing references at runtime, directly call
        validate_refs(level='error')

        'wind', 'water', 'waves' attributes also have special meaning. An
        object containing this attribute references the corresponding object.

        Logs warnings:

        :returns: a tuple of length two containing:
            (a list of messages that were logged, isvalid bool)
            If any references are missing and make_default_refs is False,
            object is not valid

        .. note: validate() only logs warnings since it designed to be used
            to validate before running model. To log these as errors during
            model run, invoke validate_refs() directly.
        '''
        (msgs, isvalid) = self.validate_refs()

        return (msgs, isvalid)

    @property
    def _warn_pre(self):
        '''
        standard text prepended to warning messages - not required for logging
        used by validate to prepend to message since it also returns a list
        of messages that were logged
        '''
        return 'warning: ' + self.__class__.__name__ + ': '
