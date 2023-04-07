#!/usr/bin/env python

import os
import copy
import logging
import glob
import json
import zipfile
import tempfile

from uuid import uuid1

import numpy as np
import colander

import gnome
from gnome.utilities.orderedcollection import OrderedCollection
from functools import reduce
from gnome.utilities.save_updater import extract_zipfile, update_savefile

log = logging.getLogger(__name__)

allowzip64 = False

SAVEFILE_VERSION = '5'


def class_from_objtype(obj_type):
    '''
    object type must be a string in the gnome namespace:
        gnome.xxx.xxx
    '''
    if len(obj_type.split('.')) == 1:
        return

    try:
        # call getattr recursively
        return reduce(getattr, obj_type.split('.')[1:], gnome)
    except AttributeError:
        log.warning("{0} is not part of gnome namespace".format(obj_type))
        raise


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

    def __init__(self, *args, **kwargs):
        # because old save files
        kwargs.pop('json_', None)
        try:
            super(AddLogger, self).__init__(**kwargs)
        # Due to super magic, the object __init__ does not always get called here
        #   but if it does, and there are kwargs, you get a TypeError
        #   trapping that allows a more meaningful message
        except TypeError:
            if kwargs:  # could it fail for some other reason? maybe???
                msg = ("{} are invalid keyword arguments for:\n"
                       "{}".format(list(kwargs.keys()), self.__class__))
                raise TypeError(msg)

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


class Refs(dict):
    '''
    Class to store and handle references during saving/loading.
    Provides some convenience functions
    '''
    def __setitem__(self, i, y):
        if i in self and self[i] is not y:
            raise ValueError('You must not set the same id twice!!')

        return dict.__setitem__(self, i, y)

    def gen_default_name(self, obj):
        '''
        Goes through the dict, finds all objects of obj.obj_type stored, and
        provides a unique name by appending length+1
        '''
        base_name = obj.obj_type.split('.')[-1]
        num_of_same_type = [v for v in self.values() if v.obj_type == obj.obj_type]

        return base_name + num_of_same_type + 1


def combine_signatures(cls_or_func):
    '''
    Decorator to tag a function or class that needs a full function signature.
    CAUTION: Do not use this decorator on functions that do not pass kwargs up to
    super, or do so after alteration. Doing so will definitely give you a wrong signature.
    Use with care, verify your work.

    This may not work when applied to classes if they get imported in a particular sequence.
    If this occurs try applying this decorator to the __init__ directly.
    '''
    if not hasattr(cls_or_func, '__signature_tag__'):
        cls_or_func.__signature_tag__ = 'combine'
    return cls_or_func

import inspect
def create_signatures(cls, dct):
    '''
    Looks through the class for tagged functions, then builds and assigns signatures.
    Uses the MRO to find ancestor functions and merges the signatures.
    '''
    worklist = []
    for k in dct.keys():
        f = getattr(cls, k)

        if hasattr(f, '__wrapped__'):
            try:
                if f.__wrapped__.__signature_tag__ == 'combine':
                    worklist.append(f.__wrapped__)
            except AttributeError:
                pass
        elif hasattr(f, '__func__'):
            try:
                if (f.__func__.__signature_tag__ == 'combine'):
                    worklist .append(f.__func__)
            except AttributeError:
                pass
        else:
            try:
                if (f.__signature_tag__ == 'combine'):
                    worklist.append(f)

            except AttributeError:
                pass

    if hasattr(cls, '__signature_tag__') and cls.__signature_tag__ == 'combine':
        worklist.append(cls)

    for item in worklist:
        #would love to clean up the tag, but this breaks for classmethods
        #del item.__signature_tag__
        t = None
        if item is cls:
            t = lambda c: c
        else:
            t = lambda c: getattr(c, item.__name__) if hasattr(c, item.__name__) else None

        pruned_func_mro = [i for i in map(t, cls.__mro__) if i is not None]
        sigs = [inspect.signature(e) for e in pruned_func_mro]

        paramlist = []
        #0-5 is the enum values of the parameter kind
        #https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
        for k in range(0,5):
            for sig in sigs:
                params = [s for s in sig.parameters.values()]
                for p in params:
                    if p.kind == k and p not in paramlist and p.name not in [prm.name for prm in paramlist]:
                        paramlist.append(p)
        #replace does not actually replace anything, it returns a new value
        item.__signature__ = sigs[0].replace(parameters=paramlist)

class GnomeObjMeta(type):
    def __new__(cls, name, parents, dct):
        if '_instance_count' not in dct:
            dct['_instance_count'] = 0

        newclass = super(GnomeObjMeta, cls).__new__(cls, name, parents, dct)
        create_signatures(newclass, newclass.__dict__)

        if hasattr(newclass.__init__, '__signature__'):
            newclass.__signature__ = newclass.__init__.__signature__
        return newclass



class GnomeId(AddLogger, metaclass=GnomeObjMeta):
    '''
    A class for assigning a unique ID for an object
    '''
    _id = None
    make_default_refs = True
    _name = None  # so that it will always exist

    # tolerances used for equality testing
    # these could be overridden in a subclass if desired
    RTOL = 1e-05
    # ATOL = 1e-08 # this is the default, but assumes values are of Order 1.
    ATOL = 1e-38 # this will only let tiny float32 values be takes as close to zero.

    def __init__(self, name=None, _appearance=None, *args, **kwargs):
        super(GnomeId, self).__init__(*args, **kwargs)
        self.__class__._instance_count += 1

        if name:
            self.name = name
        self._appearance = _appearance
        self.array_types = dict()

    @property
    def all_array_types(self):
        '''
        Returns all the array types required by this object

        If this object contains or is composed of other gnome objects
        (Spill->Substance->Initializers for example) then override this
        function to ensure all array types get presented at the top level.
        See ``Spill`` for an example
        '''
        return self.array_types.copy()


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

    @property
    def obj_type(self):
        try:
            obj_type = ('{0.__module__}.{0.__class__.__name__}'.format(self))
        except AttributeError:
            obj_type = '{0.__class__.__name__}'.format(self)

        return obj_type

    def __deepcopy__(self, memo):
        """
        The deepcopy implementation

        We need this, as we don't want the id of spill object and logger
        object copied, but do want everything else.

        Got the method from:

            http://stackoverflow.com/questions/3253439/python-copy-how-to-inherit-the-default-copying-behaviour

        Despite what that thread says for __copy__, the built-in deepcopy()
        ends up using recursion
        """
        obj_copy = object.__new__(type(self))

        if '_log' in self.__dict__:
            # just set the _log to None since it cannot be copied
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

    @property
    def name(self):
        '''
        define as property in base class so all objects will have a name
        by default
        '''
        if not hasattr(self, '_name') or self._name is None:
            return '{}_{}'.format(self.__class__.__name__.split('.')[-1],
                                        str(self.__class__._instance_count))
        else:
            return self._name

    @name.setter
    def name(self, val):
        self._name = val

    def gather_ref_as(self, src, refs):
        """
        Gathers refs from single or collection of GnomeId objects.
        :param src: GnomeId object or collection of GnomeId
        :param refs: dictionary of str->list of GnomeId
        :returns {'ref1': [list of GnomeId], 'ref2 : [list of GnomeId], ...}
        """
        if isinstance(src, GnomeId):
            src = [src,]
        for ob in src:
            if hasattr(ob, '_ref_as'):
                names = ob._ref_as
                if not isinstance(names, list):
                    names = [names,]
                for n in names:
                    if n in refs:
                        if ob not in refs[n]:
                            #only add if it doesn't already exist in the list
                            refs[n].append(ob)
                    else:
                        refs[n] = [ob,]

    def _attach_default_refs(self, ref_dict):
        '''
        !!!IMPORTANT!!!
        If this object requires default references (self._req_refs exists), this
        function will use the name of the references as keys into a reference
        dictionary to get a list of satisfactory references (objects that have
        obj._ref_as == self._req_refs). It will then attach the first object in
        the reference list to that attribute on this object.

        This behavior can be overridden if the object needs more specific
        attachment behavior than simply 'first in line'

        In addition, this function SHOULD BE EXTENDED if this object should
        provide default references to any contained child objects. When doing
        so, please be careful to respect already existing references. The
        reference attachment system should only act if the requested reference
        'is None' when the function is invoked. See Model._attach_default_refs()
        for an example.
        '''
        if not hasattr(self, '_req_refs') or not self.make_default_refs:
            return
        else:
            for refname in self._req_refs:
                reflist = ref_dict.get(refname, [])
                if len(reflist) > 0 and getattr(self, refname) is None:
                    setattr(self, refname, reflist[0])

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

                self.logger.warning(msg)
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
        when running in the model. If make_default_refs is True, then object
        is valid because the model will set these up at runtime. To raise
        an exception for missing references at runtime, directly call
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
        return self.validate_refs()

    @property
    def _warn_pre(self):
        '''
        standard text prepended to warning messages - not required for logging
        used by validate to prepend to message since it also returns a list
        of messages that were logged
        '''
        return 'warning: {}: '.format(self.__class__.__name__)

    @classmethod
    def new_from_dict(cls, dict_):
        """
        creates a new object from dictionary

        This is base implementation and can be over-ridden by classes using
        this mixin
        """
        read_only_attrs = cls._schema().get_nodes_by_attr('read_only')

        [dict_.pop(n, None) for n in read_only_attrs]

        new_obj = cls(**dict_)

        msg = "constructed object {0}".format(new_obj.__class__.__name__)
        new_obj.logger.debug(new_obj._pid + msg)

        return new_obj

    def to_dict(self, json_=None):
        """
        Returns a dictionary representation of this object. Uses the schema to
        determine which attributes are put into the dictionary. No extra
        processing is done to each attribute. They are presented as is.

        The ``json_`` parameter is ignored in this base class. 'save' is passed
        in when the schema is saving the object. This allows an override of
        this function to do any custom stuff necessary to prepare for saving.
        """
        data = {}

        list_ = self._schema().get_nodes_by_attr('all')

        for key in list_:
            data[key] = getattr(self, key)

        return data

    def update_from_dict(self, dict_, refs=None):
        if refs is None:
            refs = Refs()
            self._schema.register_refs(self._schema(), self, refs)

        updatable = self._schema().get_nodes_by_attr('update')
        attrs = copy.copy(dict_)
        updated = False

        for k in list(attrs):
            if k not in updatable:
                attrs.pop(k)

        for name in updatable:
            node = self._schema().get(name)

            if name in attrs:
                attrs[name] = self._schema.process_subnode(node,
                                                           self,
                                                           getattr(self, name),
                                                           name,
                                                           attrs,
                                                           attrs[name],
                                                           refs)

                if attrs[name] is colander.drop:
                    del attrs[name]

        # attrs may be out of order. However, we want to process the data in schema order (held in 'updatable')
        for k in updatable:
            if hasattr(self, k) and k in attrs:
                if not updated and self._attr_changed(getattr(self, k), attrs[k]):
                    updated = True

                try:
                    setattr(self, k, attrs[k])
                except AttributeError:
                    self.logger.error('Failed to set {} on {} to {}'
                                         .format(k, self, v))
                    raise
                attrs.pop(k)

        # process all remaining items in any order...can't wait to see where problems pop up in here
        for k, v in attrs.items():
            if hasattr(self, k):
                if not updated and self._attr_changed(getattr(self, k), v):
                    updated = True

                try:
                    setattr(self, k, v)
                except AttributeError:
                    self.logger.error('Failed to set {} on {} to {}'
                                         .format(k, self, v))
                    raise

        return updated

    def update(self, *args, **kwargs):  # name alias only, do not override!
        return self.update_from_dict(*args, **kwargs)

    def _attr_changed(self, current_value, received_value):
        '''
        Checks if an attribute passed back in a ``dict_`` from client has changed.
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
        if isinstance(current_value, GnomeId):
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
                if np.any(np.array(current_value) != np.array(received_value)):
                    return True

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

        Defines a base implementation of __eq__ so an object before persistence
        can be compared with a new object created after it is persisted.
        It can be overridden by the class with which it is mixed.

        It looks at attributes defined in self._state and checks that the values match

        It uses allclose() check for floats and numpy arrays, to avoid floating point
            tolerances: set to:   RTOL=1e-05, ATOL=1e-08

        :param other: another GnomeObject used for comparison in obj1 == other

        NOTE: super is not used.

        """
        # It turns out that this and _diff are using (or should be) the same
        # so I have added a fail_early flag to _diff, so it can be used here.

        return not bool(self._diff(other, fail_early=True))

        # Keeping this code here (for now, just in case)

        # # are these even the same type at all?
        # if not self._check_type(other):
        #     return False

        # schema = self._schema()

        # for name in schema.get_nodes_by_attr('all'):
        #     subnode = schema.get(name)
        #     # skip it if it's marked to not be included
        #     if ((hasattr(subnode, 'test_equal') and
        #          not subnode.test_equal) or
        #             subnode.name == 'id'):
        #         continue

        #     self_attr = getattr(self, name)
        #     other_attr = getattr(other, name)

        #     if (isinstance(self_attr, np.ndarray)
        #         or isinstance(other_attr, np.ndarray)):
        #         # process as an array
        #         try:
        #             if not np.allclose(self_attr, other_attr,
        #                                rtol=self.RTOL, atol=self.ATOL):
        #                 return False
        #         except TypeError:
        #             # compound types (such as old Timeseries) will break
        #             return np.all(self_attr == other_attr)

        #     elif isinstance(self_attr, float) or isinstance(other_attr, float):
        #         # process as a float
        #         # using np.allclose rather than math.isclose for consistency
        #         if not np.allclose(self_attr, other_attr, rtol=self.RTOL, atol=self.ATOL):
        #             return False
        #     else: # generic object
        #         if self_attr != other_attr:
        #             return False
        # # if nothing returned, then they are equal
        # return True

    def _diff(self, other, fail_early=False):
        """
        Returns a list of differences between this GnomeObject and another GnomeObject

        :param other: other object to compare to.

        :param fail_early=False: If true, it will return on the first error

        """
        # NOTE: if you find a case where this breaks, please add a test to
        #       test_gnome_object before fixing it.

        # Fixme: perhaps serializing both, and then comparing the serialized version
        #        would make it more consistent and less tricky with types.

        diffs = []

        # are these even the same type at all?
        if not self._check_type(other):
            diffs.append(f'Different type: self={self.__class__}, other={other.__class__}')
            if fail_early:
                return diffs

        # same type -- check the schema nodes (attributes)
        schema = self._schema()

        for name in schema.get_nodes_by_attr('all'):
            subnode = schema.get(name)
            # don't check the ones that are expected to be different
            if ((hasattr(subnode, 'test_equal') and
                 not subnode.test_equal) or
                    subnode.name == 'id'):
                continue

            self_attr = getattr(self, name)
            other_attr = getattr(other, name)

            if (isinstance(self_attr, np.ndarray)
                or isinstance(other_attr, np.ndarray)):
                # process as an array
                if np.asarray(self_attr).size != np.asarray(other_attr).size:
                    diffs.append(f'Arrays are not the same size -- {name!r}: '
                                 f'self={self_attr!r}, other={other_attr!r}')
                    if fail_early:
                        return diffs
                    continue
                try:
                    if not np.allclose(self_attr, other_attr,
                                       rtol=self.RTOL, atol=self.ATOL):
                        diffs.append(f'Array values are not all close -- {name!r}: '
                                     f'self={self_attr!r}, other={other_attr!r}')
                        if fail_early:
                            return diffs
                except TypeError:
                    # compound types (such as old Timeseries) will break
                    # note: not tests for this!
                    if not np.all(self_attr == other_attr):
                        diffs.append(f'Array values are not equal -- {name!r}: '
                                     f'self={self_attr!r}, other={other_attr!r}')
                        if fail_early:
                            return diffs

            elif isinstance(self_attr, float) or isinstance(other_attr, float):
                # process as a float
                # using np.allclose rather than math.isclose for consistency
                if not np.allclose(self_attr, other_attr,
                                   rtol=self.RTOL, atol=self.ATOL):
                    diffs.append(f"Difference outside tolerance -- {name}: "
                                 f"self={self_attr!r}, other= {other_attr!r}")
                    if fail_early:
                        return diffs
            else:  # generic object -- just check equality
                if self_attr != other_attr:
                    diffs.append(f"Values not equal -- {name}: "
                                 f"self={self_attr!r}, other={other_attr!r}")
                    if fail_early:
                        return diffs

        # All attributes processed: return all the diffs
        return diffs

    def __ne__(self, other):
        return not self == other

    def serialize(self, options={}):
        """
        Returns a json serialization of this object ("webapi" mode only)
        """
        if 'raw_paths' not in options:
            options['raw_paths'] = True

        schema = self.__class__._schema()
        serial = schema.serialize(self, options=options)

        return serial

    @classmethod
    def deserialize(cls, json_, refs=None):
        """
            classmethod takes json structure as input, deserializes it using a
            colander schema then invokes the new_from_dict method to create an
            instance of the object described by the json schema.

            We also need to accept sparse json objects, in which case we will
            not treat them, but just send them back.
        """
        if refs is None:
            refs = Refs()

        return cls._schema().deserialize(json_, refs=refs)

    def save(self, saveloc='.', refs=None, overwrite=True):
        """
        Save object state as json to user specified saveloc

        :param saveloc: A directory, file path, open zipfile.ZipFile, or None.
                        If a directory, it will place the zip file there,
                        overwriting if specified.

                        If a file path, it will write the file there
                        as follows:

                        If the file does not exist, it will create the zip
                        archive there. If the saveloc is a zip file or
                        ``zipfile.Zipfile`` object and overwrite is False,
                        it will append there. Otherwise, it will overwrite
                        the file if allowed.

                        If set to None, this function will instead return an
                        open ``zipfile.Zipfile`` object linked to a temporary
                        file.  The zip file will be named [object.name].zip
                        if a directory is specified

        :param refs: dictionary of references to objects
        :param overwrite: If True, overwrites the file at the saveloc

        :returns (obj_json, saveloc, refs): ``obj_json`` is the json that is
                                            written to this object's file
                                            in the zipfile.  For example if
                                            saving a Model named Model1,
                                            obj_json will contain the contents
                                            of the Model1.json in the save
                                            file.

                                            ``saveloc`` will be the string
                                            path passed in EXCEPT if None was
                                            passed in. In this case, it will
                                            be an open ``zipfile.ZipFile``
                                            based on a temporary file.

                                            ``refs`` will be a dict containing
                                            all the objects that were saved
                                            in the save file, keyed by object
                                            id. It will also contain the
                                            reference to the object that
                                            called ``.save`` itself.
        """
        # convert save location to a string
        # fixme: we should probably use pathlib in the rest of this instead,
        #        but this should work for now.
        if saveloc is not None:
            saveloc = os.fspath(saveloc)


        zipfile_ = None

        if saveloc is None:
            # Only provide an open zipfile object.  When this is closed or
            # the object loses context, the temporary file will automatically
            # delete itself.
            # Should be useful for testing without having to deal with cleanup.
            # fixme: I'm not sure it's getting deleted -- we should make sure
            #        And why not use a StringIO object instead, and keep
            #        it totally in memory?
            zipfile_ = zipfile.ZipFile(tempfile.SpooledTemporaryFile(mode='w+b'),
                                       'a',
                                       compression=zipfile.ZIP_DEFLATED,
                                       allowZip64=allowzip64)

        elif os.path.isdir(saveloc):
            n = gnome.persist.base_schema.sanitize_string(self.name)
            saveloc = os.path.join(saveloc, n + '.gnome')

            if os.path.exists(saveloc):
                if not overwrite:
                    raise ValueError('{} already exists and overwrite is False'
                                     .format(saveloc))

            zipfile_ = zipfile.ZipFile(saveloc, 'w',
                                       compression=zipfile.ZIP_DEFLATED,
                                       allowZip64=allowzip64)
        else:
            # saveloc is file path
            if not (saveloc.endswith(".zip") or saveloc.endswith(".gnome")):
                saveloc = saveloc + ".gnome"
            if not overwrite:
                if zipfile.is_zipfile(saveloc):
                    zipfile_ = zipfile.ZipFile(saveloc, 'a',
                                               compression=zipfile.ZIP_DEFLATED,
                                               allowZip64=allowzip64)
                else:
                    raise ValueError('{} already exists and overwrite is False'
                                     .format(saveloc))
            else:
                zipfile_ = zipfile.ZipFile(saveloc, 'w',
                                           compression=zipfile.ZIP_DEFLATED,
                                           allowZip64=allowzip64)
        if refs is None:
            refs = Refs()

        obj_json = self._schema()._save(self, zipfile_=zipfile_, refs=refs)

        zipfile_.writestr('version.txt', SAVEFILE_VERSION)

        if saveloc is None:
            log.info('Returning open zipfile in memory')
            return (obj_json, zipfile_, refs)
        else:
            zipfile_.close()
            return (obj_json, saveloc, refs)

    @classmethod
    def load(cls, saveloc='.', filename=None, refs=None):
        '''
        Load an instance of this class from an archive or folder

        :param saveloc: Can be an open zipfile.ZipFile archive, a folder, or a
                        filename. If it is an open zipfile or folder, it must
                        contain a .json file that describes an instance of this
                        object type. If 'filename' is not specified, it will
                        load the first instance of this object discovered.
                        If a filename, it must be a zip archive or a json file
                        describing an object of this type.

        :param filename: If saveloc is an open zipfile or folder,
                         this indicates the name of the file to be loaded.
                         If saveloc is a filename, this parameter is ignored.

        :param refs: A dictionary of id -> object instances that will be used
                     to complete references, if available.
        '''

        fp = json_ = None

        if not refs:
            refs = Refs()

        if isinstance(saveloc, str):
            if os.path.isdir(saveloc):
                #run the savefile update system
                update_savefile(saveloc)

                if filename:
                    fn = os.path.join(saveloc, filename)

                    with open(fn) as fp:
                        json_ = json.load(fp)

                    return cls._schema().load(json_, saveloc=saveloc,
                                              refs=refs)
                else:
                    search = os.path.join(saveloc, '*.json')

                    for fn in glob.glob(search):
                        with open(fn) as fp:
                            json_ = json.load(fp)
                            if 'obj_type' in json_:
                                if class_from_objtype(json_['obj_type']) is cls:
                                    return cls._schema().load(json_,
                                                              saveloc=saveloc,
                                                              refs=refs)

                    raise ValueError('No .json file containing a {} '
                                     'found in folder {}'
                                     .format(cls.__name__, saveloc))
            elif zipfile.is_zipfile(saveloc):
                # saveloc is a zip archive
                # extract to a temporary file and retry load
                tempdir = tempfile.mkdtemp()
                extract_zipfile(saveloc, tempdir)
                return cls.load(saveloc=tempdir, filename=filename, refs=refs)
            else:
                # saveloc is .json file
                with open(saveloc, 'r') as fp:
                    json_ = json.load(fp)

                if 'obj_type' in json:
                    if class_from_objtype(json_['obj_type']) is not cls:
                        raise ValueError("{} does not contain a {}"
                                         .format(saveloc, cls.__name__))
                    else:
                        folder = os.path.dirname(saveloc)

                        return cls._schema().load(json_, saveloc=folder,
                                                  refs=refs)
        elif isinstance(saveloc, zipfile.ZipFile):
            # saveloc is an already open zip archive
            # extract to a temporary directory and retry load
            tempdir = tempfile.mkdtemp()
            extract_zipfile(saveloc, tempdir)
            return cls.load(saveloc=tempdir, filename=filename, refs=refs)
        else:
            raise ValueError('saveloc was not a string path '
                             'or an open zipfile.ZipFile object')
