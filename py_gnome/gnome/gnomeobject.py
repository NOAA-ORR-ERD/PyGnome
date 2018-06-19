#!/usr/bin/env python
import os
from uuid import uuid1, UUID
import copy
import logging
import numpy as np
import zipfile
import json
import glob
import tempfile
import gnome
import six
import colander
from gnome.utilities.orderedcollection import OrderedCollection

log = logging.getLogger(__name__)

allowzip64=False


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
        super(AddLogger, self).__init__(*args, **kwargs)

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
        num_of_same_type = filter(lambda v: v.obj_type == obj.obj_type, self.values())
        return base_name + num_of_same_type+1


class GnomeObjMeta(type):
    def __new__(cls, name, parents, dct):
        if '_instance_count' not in dct:
            dct['_instance_count'] = 0
        return super(GnomeObjMeta, cls).__new__(cls, name, parents, dct)


class GnomeId(AddLogger):
    '''
    A class for assigning a unique ID for an object
    '''
    __metaclass__ = GnomeObjMeta
    _id = None
    make_default_refs = True

    def __init__(self, name=None, *args, **kwargs):
        super(GnomeId, self).__init__(*args, **kwargs)
        self.__class__._instance_count += 1
        if name:
            self.name = name

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
            obj_type = ('{0.__module__}.{0.__class__.__name__}'
                        .format(self))
        except AttributeError:
            obj_type = '{0.__class__.__name__}'.format(self)
        return obj_type

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

    @property
    def name(self):
        '''
        define as property in base class so all objects will have a name
        by default
        '''
        try:
            return self._name
        except AttributeError:
            self._name = self.__class__.__name__.split('.')[-1] + '_' + str(self.__class__._instance_count)
            return self._name

    @name.setter
    def name(self, val):
        self._name = val

    def _attach_default_refs(self, ref_dict):
        '''
        If provided a dictionary of references this function will validate it
        against the _req_refs specified by the class, and if a match is found
        and the instance's reference is None, it will set it to the instance
        from ref_dict
        '''
        if not hasattr(self, '_req_refs'):
            return
        else:
            for var in ref_dict.keys():
                if getattr(self, var) is None:
                    setattr(self, var, ref_dict[var])

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

    @classmethod
    def new_from_dict(cls, dict_):
        """
        creates a new object from dictionary

        This is base implementation and can be over-ridden by classes using
        this mixin
        """
        read_only_attrs = cls._schema().get_nodes_by_attr('read_only')
        map(lambda n: dict_.pop(n, None), read_only_attrs)
        new_obj = cls(**dict_)

        msg = "constructed object {0}".format(new_obj.__class__.__name__)
        new_obj.logger.debug(new_obj._pid + msg)

        return new_obj

    def to_dict(self, json_=None):
        """
        Returns a dictionary representation of this object. Uses the schema to
        determine which attributes are put into the dictionary. No extra
        processing is done to each attribute. They are presented as is.

        The json_ parameter is ignored in this base class. 'save' is passed
        in when the schema is saving the object. This allows an override of this
        function to do any custom stuff necessary to prepare for saving.
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
        for k in attrs.keys():
            if k not in updatable:
                attrs.pop(k)

        for name in updatable:
            node = self._schema().get(name)
            if name in attrs:
                attrs[name] = self._schema.process_subnode(node, self, getattr(self, name), name, attrs, attrs[name], refs)
                if attrs[name] is colander.drop:
                    del attrs[name]

        for k, v in attrs.items():
            if hasattr(self, k):
                if not updated and self._attr_changed(getattr(self, k), v):
                    updated = True
                try:
                    setattr(self, k, v)
                except AttributeError:
                    raise AttributeError('Failed to set {0} on {1} to {2}'.format(k, self, v))
        return updated

    update = update_from_dict

    def _attr_changed(self, current_value, received_value):
        '''
        Checks if an attribute passed back in a dict_ from client has changed.
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
                if np.any(current_value != received_value):
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

        Define a base implementation of __eq__ so an object before persistence
        can be compared with a new object created after it is persisted.
        It can be overridden by the class with which it is mixed.

        It looks at attributes defined in self._state and checks the plain
        python types match.

        It does an allclose() check for numpy arrays using default atol, rtol.

        :param other: object of the same type as self that is used for
                      comparison in obj1 == other

        NOTE: super is not used.
        """
        #print('comparing {0} and {1}'.format(self, other))

        if not self._check_type(other):
            return False

        schema = self._schema()

        for name in schema.get_nodes_by_attr('all'):
            subnode = schema.get(name)
            if ((hasattr(subnode, 'test_equal') and
                not subnode.test_equal) or
                subnode.name == 'id'):
                continue

            self_attr = getattr(self, name)
            other_attr = getattr(other, name)

            if not isinstance(self_attr, np.ndarray):
                if isinstance(self_attr, float):
                    if abs(self_attr - other_attr) > 1e-10:
                        return False
                elif self_attr != other_attr:
                    return False
            else:
                if not isinstance(self_attr, type(other_attr)):
                    return False
                try:
                    if not np.allclose(self_attr, other_attr,
                                       rtol=1e-4, atol=1e-4):
                        return False
                except TypeError: #compound types (such as old Timeseries) will break
                    return all(self_attr == other_attr)

        return True

    def _diff(self, other):
        'returns the differences between this object and another of the same type'
        diffs = []
        if not self._check_type(other):
            diffs.append('Different type: self={0}, other={1}'.format(self.__class__, other.__class__))

        schema = self._schema()

        for name in schema.get_nodes_by_attr('all'):
            subnode = schema.get(name)
            if ((hasattr(subnode, 'test_equal') and
                not subnode.test_equal) or
                subnode.name == 'id'):
                continue

            self_attr = getattr(self, name)
            other_attr = getattr(other, name)

            if not isinstance(self_attr, np.ndarray):
                if isinstance(self_attr, float):
                    if abs(self_attr - other_attr) > 1e-10:
                        diffs.append('Outside tolerance {0}: self={1}, other={2}'.format(name, self_attr, other_attr))
                elif self_attr != other_attr:
                    diffs.append('!= {0}: self={1}, other={2}'.format(name, self_attr, other_attr))
            else:
                if not isinstance(self_attr, type(other_attr)):
                    diffs.append('Different Types {0}: self={1}, other={2}'.format(name, self_attr, other_attr))

                if not np.allclose(self_attr, other_attr,
                                   rtol=1e-4, atol=1e-4):
                    diffs.append('Not allclose {0}: self={1}, other={2}'.format(name, self_attr, other_attr))

        return diffs

    def __ne__(self, other):
        return not self == other

    def serialize(self):
        """
        Returns a json serialization of this object ("webapi" mode only)
        """
        schema = self.__class__._schema()
        serial = schema.serialize(self)
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
        save object state as json to user specified saveloc

        :param saveloc: A directory, file path, open zipfile.ZipFile, or None
        .
        If a directory, it will place the zip file there, overwriting if specified.
        If a file path, it will write the file there as follows: If the file
        does not exist, it will create the zip archive there. If the
        saveloc is a zip file or zipfile.Zipfile object and overwrite is False, it
        will append there. Otherwise, it will overwrite the file if allowed.
        If set to None, this function will instead return an open
        zipfile.Zipfile object linked to a temporary file.

        The zip file will be named [object.name].zip if a directory is specified
        :param refs: dictionary of references to objects
        :param overwrite: If True, overwrites the file at the saveloc

        :returns: obj_json, saveloc, and refs
        obj_json is the json that is written to this object's file in the zipfile
        For example if saving a Model named Model1, obj_json will contain the contents
        of the Model1.json in the save file

        saveloc will be the string path passed in EXCEPT if None was passed in. In
        this case, it will be an open zipfile.ZipFile based on a temporary
        file.

        refs will be a dict containing all the objects that were saved in the
        save file, keyed by object id. It will also contain the reference to
        the object that called .save itself.

        """
        zipfile_=None
        if saveloc is None:
            #only provide an open zipfile object. When this is closed or the object
            #loses context, the temporary file will automatically delete itself
            #should be useful for testing without having to deal with cleanup.
            zipfile_ = zipfile.ZipFile(tempfile.SpooledTemporaryFile('w+b'), 'a',
                             compression=zipfile.ZIP_DEFLATED,
                             allowZip64=allowzip64)
        elif os.path.isdir(saveloc):
            saveloc = os.path.join(saveloc, self.name + '.zip')
            if os.path.exists(saveloc):
                if not overwrite:
                    raise ValueError('{0} already exists and overwrite is False'.format(saveloc))
            else:
                zipfile_ = zipfile.ZipFile(saveloc, 'w',
                                               compression=zipfile.ZIP_DEFLATED,
                                               allowZip64=allowzip64)
        else:
            #saveloc is file path
            if not overwrite:
                if zipfile.is_zipfile(saveloc):
                    zipfile_ = zipfile.ZipFile(saveloc,'a',
                                            compression=zipfile.ZIP_DEFLATED,
                                            allowZip64=allowzip64)
                else:
                    raise ValueError('{0} already exists and overwrite is False'.format(saveloc))
            else:
                zipfile_ = zipfile.ZipFile(saveloc,'w',
                                           compression=zipfile.ZIP_DEFLATED,
                                           allowZip64=allowzip64)
        if refs is None:
            refs = Refs()
        obj_json = self._schema()._save(self,
                                        zipfile_=zipfile_,
                                        refs=refs)
        if saveloc is None:
            log.info('Returning open zipfile in memory')
            return (obj_json, zipfile_, refs)
            zipfile_.close()
        else:
            zipfile_.close()
            return (obj_json, saveloc, refs)


    @classmethod
    def load(cls, saveloc='.', filename=None, refs=None):
        '''
        Load an instance of this class from an archive or folder

        :param saveloc: Can be an open zipfile.ZipFile archive, a folder, or a
        filename. If it is an open zipfile or folder, it must contain a .json
        file that describes an instance of this object type. If 'filename' is
        not specified, it will load the first instance of this object discovered.
        If a filename, it must be a zip archive or a json file describing an object
        of this type.
        :param filename: If saveloc is an open zipfile or folder, this indicates
        the name of the file to be loaded. If saveloc is a filename, this
        parameter is ignored.
        :param refs: A dictionary of id -> object instances that will be used to
        complete references, if available.
        '''
        fp = json_ =None
        if not refs:
            refs = Refs()
        if isinstance(saveloc, six.string_types):
            if os.path.isdir(saveloc):
                if filename:
                    fn = os.path.join(saveloc, filename)
                    json_ = json.load(fn)
                    return cls._schema().load(json_, saveloc=saveloc, refs=refs)
                else:
                    search = os.path.join(saveloc, '*.json')
                    for fn in glob.glob(search):
                        json_ = json.load(fn)
                        if 'obj_type' in json_:
                            if class_from_objtype(json_['obj_type']) is cls:
                                return cls._schema().load(json_, saveloc=saveloc, refs=refs)
                    raise ValueError('No .json file containing a {0} found in folder {1}'.format(cls.__name__, saveloc))
            elif zipfile.is_zipfile(saveloc):
                #saveloc is a zip archive
                #get json from the file to start the process
                saveloc = zipfile.ZipFile(saveloc, 'r')
                if filename:
                    fp = saveloc.open(filename, 'rU')
                    json_ = json.load(fp, parse_float=True, parse_int=True)
                    return cls._schema().load(json_, saveloc=saveloc, refs=refs)
                else:
                    #no filename, so search archive
                    for fn in saveloc.namelist():
                        if fn.endswith('.json'):
                            fp = saveloc.open(fn, 'rU')
                            json_ = json.load(fp)
                            if 'obj_type' in json_:
                                if class_from_objtype(json_['obj_type']) is cls:
                                    return cls._schema().load(json_, saveloc=saveloc, refs=refs)
                    raise ValueError('No .json file containing a {0} found in archive {1}'.format(cls.__name__, saveloc))
            else:
                #saveloc is .json file
                fp = open(saveloc, 'r')

                json_ = json.load(fp)
                if 'obj_type' in json:
                    if class_from_objtype(json_['obj_type']) is not cls:
                        raise ValueError("{1} does not contain a {0}".format(cls.__name__, saveloc))
                    else:
                        dir = os.path.dirname(saveloc)
                        return cls._schema().load(json_, saveloc=dir, refs=refs)
        elif isinstance(saveloc, zipfile.ZipFile):
            if filename:
                fp = saveloc.open(filename, 'rU')
                json_ = json.load(fp)
                return cls._schema().load(json_, saveloc=saveloc, refs=refs)
            else:
                #no filename, so search archive
                for fn in saveloc.namelist():
                    if fn.endswith('.json'):
                        fp = saveloc.open(fn, 'r')
                        json_ = json.load(fp)
                        if 'obj_type' in json_:
                            if class_from_objtype(json_['obj_type']) is cls:
                                return cls._schema().load(json_, saveloc=saveloc, refs=refs)
                raise ValueError('No .json file containing a {0} found in archive {1}'.format(cls.__name__, saveloc))
        else:
            raise ValueError('saveloc was not a string path or an open zipfile.ZipFile object')
        pass