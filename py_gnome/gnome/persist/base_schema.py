import datetime
import zipfile
import logging
from collections import abc
import os
import json
import tempfile
import geojson
import re

from colander import (SchemaNode, deferred, drop, required, Invalid, UnsupportedFields,
                      SequenceSchema, TupleSchema, MappingSchema, Mapping,
                      String, Float, Int, SchemaType, Sequence, Tuple, Positional, null)

from .extend_colander import NumpyFixedLenSchema

from gnome.gnomeobject import Refs, class_from_objtype
from gnome.persist.extend_colander import OrderedCollectionType
from gnome.utilities.geometry.polygons import PolygonSet

log = logging.getLogger(__name__)


@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of
                                    datetime.datetime.now to when it is called
                                    in Schema
    """
    return datetime.datetime.now().replace(microsecond=0)

class ObjType(SchemaType):

    def __init__(self, unknown='ignore'):
        self.unknown = unknown

    def _set_unknown(self, value):
        if not value in ('ignore', 'raise', 'preserve'):
            raise ValueError(
                'unknown attribute must be one of "ignore", "raise", '
                'or "preserve"')
        self._unknown = value

    def _get_unknown(self):
        return self._unknown

    unknown = property(_get_unknown, _set_unknown)

    def cstruct_children(self, node, cstruct):
        if cstruct is None:
            value = {}
        else:
            value = self._validate(node, cstruct)

        children = []

        for subnode in node.children:
            name = subnode.name
            subval = value.get(name, required)

            if subval is required:
                subval = subnode.serialize(null)

            children.append(subval)

        return children

    def _impl(self, node, value, callback):
        error = None
        result = {}

        for num, subnode in enumerate(node.children):
            name = subnode.name
            subval = None

            if self.unknown == 'ignore':
                subval = value.get(name, None)
            else:
                subval = value.pop(name, None)
            if subval is None: #deserialization
                if subnode.missing == drop:
                    continue
                sub_result = subnode.missing
            elif subval is null: #serialization
                if subnode.default == drop:
                    continue
                sub_result = subnode.default if subnode.default is not null else None
            else:
                sub_result = callback(subnode,subval)
#             try:
#             except Invalid as e:
#                 raise e
#                 if error is None:
#                     error = Invalid(node)
#                 error.add(e, num)
#             else:
            #if sub_result is None:
            #    continue
            result[name] = sub_result

        if self.unknown == 'raise':
            if value:
                raise UnsupportedFields(node, value,
                                        msg=_('Unrecognized keys in mapping: '
                                              '"${val}"',
                                              mapping={'val': value}))

        elif self.unknown == 'preserve':
            result.update(value)

        if error is not None:
            raise error

        return result

    def _ser(self, node, value, options=None):
        dict_ = None
        try:
            if hasattr(value, 'to_dict'):
                dict_ = value.to_dict('webapi')
            else:
                raise TypeError('Object does not have a to_dict function')
        except Exception as e:
            raise e
        return dict_

    def serialize(self, node, appstruct, options=None):
        def callback(subnode, subappstruct):
            if (isinstance(subnode.typ, (Sequence, OrderedCollectionType)) and
                    isinstance(subnode.children[0], ObjTypeSchema)):
                scalar = (hasattr(subnode.typ, 'accept_scalar') and
                          subnode.typ.accept_scalar)

                return subnode.typ._impl(subnode, subappstruct, callback,
                                         scalar)
            else:
                try:
                    return subnode.serialize(subappstruct, options=options)
                except TypeError as e:
                    if 'unexpected keyword argument' in str(e):
                        return subnode.serialize(subappstruct)
                    else:
                        raise e

        value = self._ser(node, appstruct, options=options)

        return self._impl(node, value, callback)

    def _deser(self, node, value, refs):
        # value in this case would be 
#         try:
            if value is None:
                return None
            if isinstance(value, dict) and 'obj_type' in value:
                id_ = value.get('id', None)
                if id_ not in refs or id_ is None:
                    # object does not exist in refs already
                    obj_type = class_from_objtype(value['obj_type'])
                    obj = obj_type.new_from_dict(value)
                    log.info('Created new {0} object {1}'.format(obj.obj_type, obj.name))
                    node.register_refs(node, obj, refs)
                    if id_ is None:
                        id_ = obj.id
                    refs[id_] = obj
                return refs[id_]
            else:
                raise TypeError('{0} is not dictionary, or does not have an obj_type'.format(value))
#         except Exception as e:
#             raise e
#             raise Invalid(node, '{0}" does not implement GnomeObj functionality: {1}'.format(value, e))

    def deserialize(self, node, cstruct, refs):
        if cstruct is None:
            return None

        def callback(subnode, subcstruct):
            #This callback implements the branching recursion in deserialization
            if subnode.typ.__class__ is ObjType:
                #All GNOME object types
                return subnode.deserialize(subcstruct, refs)
            else:
                # This needs to become more flexible! It needs to detect
                # subclasses of Sequence!
                if (isinstance(subnode.typ, (Sequence, OrderedCollectionType)) and
                        isinstance(subnode.children[0], ObjTypeSchema)):
                    # To deal with iterable schemas that do not have
                    # a deserialize with refs function
                    # Lists
                    scalar = (hasattr(subnode.typ, 'accept_scalar') and
                              subnode.typ.accept_scalar)

                    return subnode.typ._impl(subnode, subcstruct, callback,
                                             scalar)
                elif (subnode.schema_type is Tuple):
                    #Tuples
                    return subnode.typ._impl(subnode, subcstruct, callback)
                else:
                    #Strings, dates, numbers, etc
                    return subnode.deserialize(subcstruct)

        if 'id' in cstruct and cstruct['id'] in refs:
            #future appstruct already exists in refs
            #however cstruct may hold updates that should be applied to object in refs
            #In this case the object should be 'update_from_dict' and the result
            #be the newly updated object
            id_ = cstruct['id']
            try:
                log.info(refs[id_].name)
            except Exception as e:
                log.warning('Could not log object name: ' + e.msg)
            updated = refs[id_].update_from_dict(cstruct)
            if updated:
                log.info('Updated object {0} from json'.format(refs[id_].name))
            return refs[id_]
        else:
            #
            result = self._impl(node, cstruct, callback)

        return self._deser(node, result, refs)

    def flatten(self, node, appstruct, prefix='', listitem=False):
        result = {}
        if listitem:
            selfprefix = prefix
        else:
            if node.name:
                selfprefix = '{}{}.'.format(prefix, node.name)
            else:
                selfprefix = prefix

        for subnode in node.children:
            name = subnode.name
            substruct = appstruct.get(name, None)

            result.update(subnode.typ.flatten(subnode, substruct,
                                              prefix=selfprefix))
        return result

    def unflatten(self, node, paths, fstruct):
        return super(ObjType, self).unflatten(node, paths, fstruct)

    def set_value(self, node, appstruct, path, value):
        if '.' in path:
            next_name, rest = path.split('.', 1)
            next_node = node[next_name]
            next_appstruct = appstruct[next_name]

            appstruct[next_name] = next_node.typ.set_value(next_node,
                                                           next_appstruct,
                                                           rest, value)
        else:
            appstruct[path] = value
        return appstruct

    def get_value(self, node, appstruct, path):
        if '.' in path:
            name, rest = path.split('.', 1)
            next_node = node[name]

            return next_node.typ.get_value(next_node, appstruct[name], rest)
        return appstruct[path]

    def _prepare_save(self, node, raw_object, saveloc, refs):
        #Gets the json for the object, as if this were being serialized
        if hasattr(raw_object, 'to_dict'):
            # Passing the 'save' in case a class wants to do some special stuff on
            # saving specifically.
            dict_ = raw_object.to_dict('save')
            for k in dict_.keys():
                if dict_[k] is None:
                    dict_[k] = null
            return dict_
        else:
            raise TypeError('Object does not have a to_dict function')

    def _save(self, node, json_, zipfile_, refs):
        if json_['id'] in refs:
            fname = refs[json_['id']]
        else:
            fname = str(json_['name']) + '.json'
            fname = gen_unique_filename(fname, zipfile_)

            refs[json_['id']] = fname
        # strips out any entries that do not need saving. They're still in refs,
        # but that shouldn't do any harm.
        savable_attrs = node.get_nodes_by_attr('save')
        for k in list(json_.keys()):
            subnode = node.get(k)

            # Need to exclude lists from this culling,
            # unless explicitly set save=false
            #t1 = not isinstance(subnode, (SequenceSchema, TupleSchema))
            t2 = hasattr(subnode, 'save') and subnode.save is False
            t3 = k not in savable_attrs
            #if (t1 and t2 and t3 and k in json_):
            if (t2 and t3 and k in json_):
                json_.pop(k)

        #replace all save_reference json with just the json filename containing said object
        refd_names = node.get_nodes_by_attr('save_reference')
        for n in refd_names:
            if n in json_: #if a 'save_reference' attr is None, it will hit this
                if json_[n] is None:
                    continue
                elif isinstance(json_[n], list):
                    #this is a SequenceSchema or TupleSchema tagged with save_reference
                    for i, subjson in enumerate(json_[n]):
                        refname = subjson['name'] + '.json'
                        if subjson['id'] in refs:
                            #this name has already been added somewhere else,
                            #and MAY have a different name if there was a naming
                            #conflict
                            refname = refs[subjson['id']]
                        json_[n][i] = refname
                else:
                    # single reference
                    refname = json_[n]['name'] + '.json'

                    if json_[n]['id'] in refs:
                        # this name has already been added somewhere else,
                        # and MAY have a different name if there was a
                        # naming conflict
                        refname = refs[json_[n]['id']]

                    json_[n] = refname

        # Put supporting files into the zipfile and edit their paths
        # in the json
        datafiles = [d for d in node.get_nodes_by_attr('isdatafile')
                     if d in json_]
        for d in datafiles:
            if json_[d] is None:
                continue
            elif isinstance(json_[d], str):
                json_[d] = self._process_supporting_file(json_[d], zipfile_)
            elif isinstance(json_[d], abc.Iterable):
                # List, tuple, etc
                for i, filename in enumerate(json_[d]):
                    json_[d][i] = self._process_supporting_file(filename,
                                                                zipfile_)

        # Finally, write the json itself to the zipfile, and return the json
        if fname not in zipfile_.namelist():
            zipfile_.writestr(fname, json.dumps(json_, indent=True))
        return json_

    def save(self, node, appstruct, zipfile_, refs):
        def callback(subnode, subappstruct):
            if not hasattr(subnode, '_save'):
                # This happens when it goes into non-gnome object attributes
                # (Strings, Numbers, etc)
                if (isinstance(subnode.typ, (Sequence, OrderedCollectionType)) and
                        isinstance(subnode.children[0], ObjTypeSchema)):
                    # To be able to continue saving inside iterables,
                    # whose schema does not contain a save function,
                    # call the subnode typ _impl with this function as callback
                    # Doing that will execute this function against each item
                    # in the iterable, and should continue the chain if it
                    # contains further iterables.
                    scalar = (hasattr(subnode.typ, 'accept_scalar') and
                              subnode.typ.accept_scalar)
                    return subnode.typ._impl(subnode, subappstruct, callback,
                                             scalar)
                elif (subnode.schema_type is Tuple):
                    return subnode.typ._impl(subnode, subappstruct, callback)
                else:
                    #Not an iterable containing Gnome objects, so simply return
                    #the serialization of this non-GNOME object.
                    return subnode.serialize(subappstruct)
            else:
                # This is the path for Gnome objects
                if subnode.save is True:
                    return subnode._save(subappstruct,
                                         zipfile_=zipfile_,
                                         refs=refs)
                else:
                    logging.info('skipped {0}'.format(subnode))

        #gets the dictionary representation of the object, 'save' is passed
        dict_ = self._prepare_save(node, appstruct, zipfile_, refs)
        #Recursively serializes each node, producing the object's json
        preprocessed_json = self._impl(node, dict_, callback)
        #Processes references, adds supporting files to the zipfile, and writes
        #the json to the zip, and returns the json of the object.
        return self._save(node, preprocessed_json, zipfile_, refs)

    def _process_supporting_file(self, raw_path, zipfile_):
        '''
        raw_path is the filename stored on the object
        zipfile is an open zipfile.Zipfile in append mode
        returns the name of the file in the archive
        '''
        d_fname = os.path.split(raw_path)[1]
        # add datafile to zip archive

        if d_fname not in zipfile_.namelist():
            zipfile_.write(raw_path, d_fname)

        return d_fname

    def load(self, node, cstruct, saveloc, refs):
        def callback(subnode, subcstruct):
            if not hasattr(subnode, 'load'):
                # This is the path for non-gnome attributes
                if (isinstance(subnode.typ, (Sequence, OrderedCollectionType)) and
                        isinstance(subnode.children[0], ObjTypeSchema)):
                    # To deal with iterable schemas that do not have a
                    # load function
                    scalar = (hasattr(subnode.typ, 'accept_scalar') and
                              subnode.typ.accept_scalar)

                    return subnode.typ._impl(subnode, subcstruct, callback,
                                             scalar)
                elif (subnode.schema_type is Tuple):
                    return subnode.typ._impl(subnode, subcstruct, callback)
                else:
                    return subnode.deserialize(subcstruct)
            else:
                #this is the path for Gnome attributes
                return subnode.load(subcstruct, saveloc=saveloc, refs=refs)

        #takes the obj_json with references and replaces the references
        #with the un-hydrated obj_json from each file.
        hydrated_json = self.hydrate_json(node, cstruct, saveloc, refs)
        #Recursively loads each node. After this, the hydrated json is
        #an object dict
        dict_ = self._impl(node, hydrated_json, callback)
        #instantiates the object exactly as deserialize would do
        return self._deser(node, dict_, refs)

    def hydrate_json(self, node, cstruct, saveloc, refs):
        # Get all the save_reference attributes and load the files
        refd_attrs = [r for r in node.get_nodes_by_attr('save_reference')
                      if r in cstruct]

        for r in refd_attrs:
            if isinstance(cstruct[r], list):
                # Need to turn this into a list of unhydrated object json
                for i, fn in enumerate(cstruct[r]):
                    if isinstance(fn, dict):  # old-style ref
                        if 'id' in fn:
                            fn = fn['id']
                            cstruct[r][i] = (self
                                             ._load_json_from_file(fn, saveloc)
                                             )
                            log.info('Loaded json from {0}'.format(fn))
                            cstruct[r][i]['id'] = fn
                        else:
                            # old-style obj-in-list (spill.initializers)
                            # do nothing
                            pass
                    else:
                        cstruct[r][i] = self._load_json_from_file(fn, saveloc)
                        log.info('Loaded json from {0}'.format(fn))
                        cstruct[r][i]['id'] = fn
            else:
                fn = cstruct[r]

                if isinstance(fn, dict):
                    # compatibility
                    # this case occurs if loading an old save file where an
                    # attribute that is now save_reference=True (such as
                    # model.map) is not a filename reference, but already
                    # a json structure
                    cstruct[r]['id'] = (cstruct
                                        .get('name',
                                             '{}.{}'
                                             .format(cstruct.get('json_'), r)))
                elif fn is not None:
                    cstruct[r] = self._load_json_from_file(fn, saveloc)
                    log.info('Loaded json from {0}'.format(fn))

                    cstruct[r]['id'] = fn

            # since object id is not saved, but we need a means to ID
            # this object during deserialization, append filename as
            # object id. It's removed later

        # extract any isdatafile attributes to a temporary directory and amend
        # the entry in json with this path.
        tmpdir = None
        datafiles = [d for d in node.get_nodes_by_attr('isdatafile')
                     if d in cstruct]

        if len(datafiles) > 0:
            tmpdir = tempfile.mkdtemp()

        for d in datafiles:
            if isinstance(cstruct[d], str):
                cstruct[d] = self._load_supporting_file(cstruct[d],
                                                        saveloc, tmpdir)
                log.info('Extracted file {0}'.format(cstruct[d]))
            elif isinstance(cstruct[d], abc.Iterable):
                # List, tuple, etc
                for i, filename in enumerate(cstruct[d]):
                    cstruct[d][i] = self._load_supporting_file(filename,
                                                               saveloc, tmpdir)

        return cstruct

    def _load_supporting_file(self, filename, saveloc, tmpdir):
        '''
        filename is the name of the file in the zip
        saveloc can be a folder or open zipfile.ZipFile object
        if saveloc is a folder and the filename exists inside,
        this does not return an altered name, nor does it extract to the temporary directory.
        An altered filename is returned if it cannot find the filename directly
        or if saveloc is an open zipfile
        in a temporary directory
        '''
        if filename is None:
            return
        if isinstance(saveloc, zipfile.ZipFile):
            dirname = os.path.dirname(saveloc.fp.name)

            # Keep an eye on this. It may cause previously existing files
            # to be recognized incorrectly as what's in the zip since it's a
            # simple existence check
            if not os.path.exists(os.path.join(dirname, filename)):
                saveloc.extract(filename, dirname)
                return os.path.join(dirname, filename)
            else:
                return os.path.join(dirname, filename)
        elif os.path.exists(os.path.join(saveloc, filename)):
            return os.path.join(saveloc, filename)
        elif os.path.exists(filename):
            return filename
        else:
            return filename

    def _load_json_from_file(self, fname, saveloc):
        '''
        filename is the name of the file in the zip
        saveloc can be a folder or open zipfile.ZipFile object
        '''
        if fname is None:
            return
        fp = None
        if isinstance(saveloc, zipfile.ZipFile):
            fp = saveloc.open(fname)
        else:
            fname = os.path.join(saveloc, fname)
            fp = open(fname)
        return json.load(fp)


class ObjTypeSchema(MappingSchema):
    schema_type = ObjType

    '''
    These are the default schema settings for GnomeId objects.
    Put them in your schema declaration if an override is desired.
    '''

    save=True               #Attr will appear in save files
    update=True             #Attr is updatable through .update
    read_only=False         #Attr is read only. This supersedes update
    test_equal=True         #Attr will be ignored in == tests
    isdatafile = False      #Attr references filenames to be added to save files
    save_reference = True   #Attr is a link to another GnomeId and should be
                            #saved as a reference in save files
    default = null          #(Colander) Set default = drop to skip this attribute
                            #if it is None during serialization
    missing = drop            #(Colander) If not present in cstruct, ignore. This
                            #attribute is NOT required for correct deserialization
                            #set missing=required for all attributes that ARE
                            #required for object init.

    #These are the defaults automatically applied to children of this node
    #if not defined already
    _colander_defaults = {'save':save,
                          'update':update,
                          'read_only':read_only,
                          'test_equal':test_equal,
                          'isdatafile':isdatafile,
                          'missing':missing,
                          'default':default
                          }

    # Attr will appear in save files
    save = True

    # Attr is updatable through .update
    update = True

    # Attr is read only. This supersedes update
    read_only = False

    # Attr will be ignored in == tests
    test_equal = True

    # Attr references filenames to be added to save files
    isdatafile = False

    # Attr is a link to another GnomeId and should be saved as a reference
    # in save files
    save_reference = True

    # (Colander) Set default = drop to skip this attribute if it is None
    # during serialization
    default = null

    # (Colander) If not present in cstruct, ignore. This attribute is NOT
    # required for correct deserialization.  Set missing=required for all
    # attributes that ARE required for object init.
    missing = drop

    # These are the defaults automatically applied to children of this node
    # if not defined already
    _colander_defaults = {'save': save,
                          'update': update,
                          'read_only': read_only,
                          'test_equal': test_equal,
                          'isdatafile': isdatafile,
                          'missing': missing,
                          'default': default}

    # defines the obj_type which is stored by all gnome objects when persisting
    # to save files
    obj_type = SchemaNode(String(), missing=required, read_only=True)

    # It also optionally stores the 'id' if present
    # Note: The idea is that the save file is something to be used externally.
    #       Ideally, one could craft a save file for import into PyGnome
    #       without needing an unreasonable amount of knowledge of its
    #       internal implementations.
    #       So the json structures that we produce for a save file
    #       should not rely on any unique object identifiers
    id = SchemaNode(String(), save=True, read_only=True)
    name = SchemaNode(String(), test_equal=False)
    # _appearance is to allow the client to persist visualization information.

    def __init__(self, *args, **kwargs):
        super(ObjTypeSchema, self).__init__(*args, **kwargs)
        for c in self.children:
            for k, v in self._colander_defaults.items():
                if not hasattr(c, k):
                    setattr(c, k, v)
                elif hasattr(c, k) and hasattr(c.__class__, k) and getattr(c, k) is getattr(c.__class__, k):
                    #things like missing, which by default are 'required' on the class
                    #if overridden, it will be on the instance instead
                    setattr(c, k, v)
            if c.read_only and c.update:
                c.update=False

    def serialize(self, appstruct=None, options={}):

        return self.typ.serialize(self, appstruct, options=options)

    def deserialize(self, cstruct=None, refs=None):
        if refs is None:
            refs = Refs()
        appstruct = self.typ.deserialize(self, cstruct, refs=refs)
        return appstruct

    def _save(self, obj, zipfile_=None, refs=None):
        '''
        Saves the object passed in to a zip file. Note that ths name of this
        function is '_save' to allow the attribute 'save' to be used to specify
        if this SchemaNode represents an attribute that should be saved.

        :param obj: Gnome object to be saved
        :param zipfile_: an open zipfile.Zipfile object, in append mode
        :param name: unless specified, uses name of obj
        :param refs: references dict

        :returns Processed json representation of the object.
        When this returns, zipfile_ should be a complete zip savefile of obj,
        and refs will be a dictionary of all GNOME objects keyed by id.
        '''
        if obj is None:
            raise ValueError(self.__class__.__name__ + ': Cannot save a None')
        if obj._schema is not self.__class__:
            raise TypeError('A {0} cannot save a {1}'.format(self.__class__.__name__, obj.__class__.__name__))

        if zipfile is None:
            raise ValueError('Must pass an open zipfile.Zipfile in append mode to zipfile_')
        if refs is None:
            refs = Refs()

        obj_json = self.typ.save(self, obj, zipfile_, refs)
        return obj_json

    def load(self, obj_json, saveloc=None, refs=None):
        if obj_json is None:
            raise ValueError(self.__class__.__name__ + ': Cannot load a None')
        cls = class_from_objtype(obj_json['obj_type'])
        if cls._schema is not self.__class__:
            raise TypeError('A {0} cannot save a {1}'.format(self.__class__.__name__, cls.__name__))
        if zipfile is None:
            raise ValueError('Must pass an open zipfile.Zipfile in append mode to saveloc')

        obj = self.typ.load(self, obj_json, saveloc, refs)
        return obj

    def get_nodes_by_attr(self, attr):
        '''
        Returns a list of child node names that have the specified attr set on them
        This replaces the State and Field mechanisms from the old serialization paradigm
        Now such attributes are on the schema directly

        If attr is 'all' it just returns a list of all child node names
        '''
        if attr == 'all':
            return [n.name for n in self.children]
        else:
            names = [n.name for n in [c for c in self.children if hasattr(c, attr) and getattr(c, attr)]]
            #sequences need to be taken into account. If present they will
            #considered to always have 'save' and 'update' as true, read as false,
            return names

    @staticmethod
    def register_refs(node, subappstruct, refs):
        if ((node.schema_type is Sequence or
             node.schema_type is OrderedCollectionType) and
             isinstance(node.children[0], ObjTypeSchema)):
            [subitem._schema.register_refs(subitem._schema(), subitem, refs) for subitem in subappstruct]
        if not isinstance(node, ObjTypeSchema) or subappstruct is None:
            return
        if subappstruct.id not in refs:
            refs[subappstruct.id] = subappstruct
        names = node.get_nodes_by_attr('all')
        for n in names:
            subappstruct._schema.register_refs(subappstruct._schema().get(n), getattr(subappstruct, n), refs)

    @staticmethod
    def process_subnode(subnode, appstruct, subappstruct, subname, cstruct, subcstruct, refs):
        '''
        The recursive function that a schema uses to process it's child attributes.
        returns the value of what subappstruct should become, based on the incoming subcstruct
        '''
        if subnode.schema_type is ObjType:
            if subcstruct is None:
                return None

            r = refs.get(subcstruct.get('id', None), None)
            if r is None:
                # object doesn't exist, so attempt to instantiate through
                # deserialize, or use existing object if found in refs
                r = subnode.deserialize(subcstruct, refs=refs)
                refs[r.id] = r

                return r
            else:
                # existing object, so update using subdict (dict_[name])
                # and set dict_[name] to object
                r.update_from_dict(subcstruct, refs=refs)

                return r
        elif (subnode.schema_type is Sequence and
              isinstance(subnode.children[0], ObjTypeSchema)):
            #subnode is a list of Gnome objects
            if subappstruct is None:
                subappstruct = []
            else:
                del subappstruct[:]

            for subitem in subcstruct:
                subappstruct.append(ObjTypeSchema.process_subnode(subnode.children[0], appstruct, subappstruct, subname, cstruct, subitem, refs))
            return subappstruct
        elif (subnode.schema_type is OrderedCollectionType and
                    isinstance(subnode.children[0], ObjTypeSchema)):
            if subappstruct is not None:
                subappstruct.clear()
            else:
                raise ValueError('process_subnode(): '
                                 'why is orderedcollection subappstruct None????')

            for subitem in subcstruct:
                subappstruct.add(ObjTypeSchema.process_subnode(subnode.children[0],appstruct, subappstruct, subname, cstruct, subitem, refs))
            return subappstruct
        else:
            return subnode.deserialize(subcstruct)


class GeneralGnomeObjectSchema(ObjTypeSchema):
    '''
    The purpose of this schema is to be a placeholder in situations where you
    need to specify that an attribute may be one of many different types.

    For example, a CurrentMover's .current may be a GridCurrent, an
    IceAwareGridCurrent, a TimeseriesCurrent, etc. Alternatively, you may
    be composing an attribute from several types of Gnome object
    '''

    def __init__(self, acceptable_schemas=None, **kwargs):
        if acceptable_schemas is None:
            raise ValueError('Must provide a list of at least one '
                             'valid schema')

        self.acceptable_schemas = acceptable_schemas
        super(GeneralGnomeObjectSchema, self).__init__(**kwargs)

    def validate_input_schema(self, obj_or_json):
        '''
        Takes an object or json dict and determines if it can be represented by
        this schema instance. Returns an instance of the schema of the object,
        or raises an error if the object cannot be used with this schema.
        '''
        if type(obj_or_json) is dict:
            json_ = obj_or_json
            obj_type = class_from_objtype(json_['obj_type'])
            schema = obj_type._schema
        else:
            obj = obj_or_json
            obj_type = obj.__class__
            schema = obj.__class__._schema

        for s in self.acceptable_schemas:
            if schema is s or issubclass(schema, s):
                return schema()

        raise TypeError('This type of object {} is not supported. '
                        'Schema: {}'
                        .format(obj_type, schema))

    def serialize(self, appstruct=None, options=None):
        substitute_schema = self.validate_input_schema(appstruct)
        return substitute_schema.typ.serialize(substitute_schema, appstruct, options=options)

    def deserialize(self, cstruct=None, refs=None):
        if refs is None:
            refs = Refs()
        substitute_schema = self.validate_input_schema(cstruct)
        return substitute_schema.typ.deserialize(substitute_schema, cstruct, refs=refs)

    def _save(self, obj, zipfile_=None, refs=None):
        if refs is None:
            refs = Refs()
        substitute_schema = self.validate_input_schema(obj)
        return substitute_schema.typ.save(substitute_schema, obj, zipfile_=zipfile_, refs=refs)

    def load(self, obj_json, saveloc=None, refs=None):
        if refs is None:
            refs = Refs()
        substitute_schema = self.validate_input_schema(obj_json)
        return substitute_schema.typ.load(substitute_schema, obj_json, saveloc=saveloc, refs=refs)


class CollectionItemMap(MappingSchema):
    '''
    This stores the obj_type and obj_index
    '''
    obj_type = SchemaNode(String())
    id = SchemaNode(String(), missing=drop)


class CollectionItemsList(SequenceSchema):
    item = CollectionItemMap()


class LongLat(TupleSchema):
    'Only contains 2D (long, lat) positions'
    long = SchemaNode(Float())
    lat = SchemaNode(Float())


class LongLatBounds(SequenceSchema):
    'Used to define bounds on a map'
    bounds = LongLat()


Polygon = LongLatBounds

class FeatureCollectionSchema(MappingSchema):
    '''
    geojson.FeatureCollection -> String via __geo_interface__ attribute
    '''
    def serialize(self, appstruct):
        assert isinstance(appstruct, geojson.FeatureCollection)
        return appstruct.__geo_interface__
    def deserialize(self, cstruct):
        if isinstance(cstruct, str):
            return geojson.loads(cstruct)
        else:
            return geojson.loads(geojson.dumps(cstruct))
        

class FeatureSchema(MappingSchema):
    '''
    geojson.Feature -> String via __geo_interface__ attribute
    '''
    def serialize(self, appstruct):
        assert isinstance(appstruct, geojson.Feature)
        return appstruct.__geo_interface__
    def deserialize(self, cstruct):
        assert isinstance(cstruct, str)
        return geojson.loads(cstruct)


class PolygonSetSchema(SequenceSchema):
    polygonset = Polygon()

    def serialize(self, appstruct):
        appstruct = [poly.tolist() for poly in appstruct]
        return super(PolygonSetSchema, self).serialize(appstruct)

    def deserialize(self, cstruct):
        if cstruct is None:
            return None
        else:
            appstruct = super(PolygonSetSchema, self).deserialize(cstruct)
            if len(appstruct) == 0:
                # empty --should be None
                return None
            # fixme: is there any need to for a PolygonSet here?
            #        a list of lists would work fine.
            #        a PolygonSet is created in the spillable_area.setter anyway.
            ps = PolygonSet()
            for poly in appstruct:
                ps.append(poly)
            return ps


class WorldPoint(LongLat):
    'Used to define reference points. 3D positions (long,lat,z)'
    z = SchemaNode(Float(), default=0.0)


class WorldPointNumpy(NumpyFixedLenSchema):
    '''
    Define same schema as WorldPoint; however, the base class
    NumpyFixedLenSchema serializes/deserializes it from/to a numpy array
    '''
    long = SchemaNode(Float())
    lat = SchemaNode(Float())
    z = SchemaNode(Float())


class ImageSize(TupleSchema):
    'Only contains 2D (long, lat) positions'
    width = SchemaNode(Int())
    height = SchemaNode(Int())


# The following two functions have analogues in the webgnomeAPI
# fixme: the webgnomeAPI has access to py_gnome -- shouldn't we
#        use the same implementation in both?
def get_file_name_ext(filename_in):
    # in case a path was passed as the name
    base_name = os.path.basename(filename_in)
    file_name, extension = os.path.splitext(base_name)

    return file_name, extension


def sanitize_string(s):
    # basic HTML string sanitization
    return re.sub(r'[/\\<>:"|?*$]', '_', s)


def gen_unique_filename(filename_in, zipfile_):
    """
    add a trailing number to the file name in case the user accidentally
    uploads multiple files with the same name for different objects.
    also sanitizes out illegal characters
    """
    filename_in = sanitize_string(filename_in)
    existing_files = set(zipfile_.namelist())
    file_name, extension = get_file_name_ext(filename_in)
    fmtstring = file_name + '{0}' + extension
    new_fn = fmtstring.format('')

    for i in range(255):
        if new_fn not in existing_files:
            return new_fn
        else:
            new_fn = fmtstring.format(' (' + str(i) + ')')

    raise ValueError('Same file name used too many times')


