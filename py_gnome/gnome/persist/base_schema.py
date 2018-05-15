import pdb
import datetime
import zipfile
import six
import io
import collections
import os
import json

from colander import (SchemaNode, deferred, drop, required, Invalid, UnsupportedFields,
                      SequenceSchema, TupleSchema, MappingSchema,
                      String, Float, Int, SchemaType, Sequence, Tuple, Positional)

from extend_colander import NumpyFixedLenSchema

from gnome.persist.save_load import class_from_objtype
from gnome.gnomeobject import GnomeId


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
                subval = subnode.serialize(None)
            children.append(subval)
        return children

    def _impl(self, node, value, callback):
        error = None
        result = {}

        for num, subnode in enumerate(node.children):
            name = subnode.name
            subval = value.get(name, None)
            if subval is None and subnode.missing is drop:
                continue
            try:
                sub_result = callback(subnode,subval)
            except Invalid as e:
                if error is None:
                    error = Invalid(node)
                error.add(e, num)
            else:
                if sub_result is None:
                    continue
                result[name] = sub_result

        if self.unknown == 'raise':
            if value:
                raise UnsupportedFields(
                    node, value,
                    msg=_('Unrecognized keys in mapping: "${val}"',
                          mapping={'val': value}))

        elif self.unknown == 'preserve':
            result.update(value)

        if error is not None:
            raise error

        return result

    def _ser(self, node, value):
        try:
            if hasattr(value, 'to_dict'):
                return value.to_dict('webapi')
            else:
                raise TypeError('Object does not have a to_dict function')
        except Exception as e:
            raise e
            raise Invalid(node, '{0}" does not implement GnomeObj functionality: {1}'.format(value, e))

    def serialize(self, node, appstruct):
#         if appstruct is None:
#             appstruct = None

        def callback(subnode, subappstruct):
            return subnode.serialize(subappstruct)

        value = self._ser(node, appstruct)
        return self._impl(node, value, callback)

    def _deser(self, node, value, refs):
#         try:
            if type(value) is dict and 'obj_type' in value:
                id_ = value['id']
                if value['id'] not in refs:
                    obj_type = class_from_objtype(value['obj_type'])
                    refs[id_] = obj_type.new_from_dict(value)
                return refs[id_]
            else:
                raise TypeError('Object is not dictionary, or does not have an obj_type')
#         except Exception as e:
#             raise e
#             raise Invalid(node, '{0}" does not implement GnomeObj functionality: {1}'.format(value, e))

    def deserialize(self, node, cstruct, refs):
        if cstruct is None:
            return None

        def callback(subnode, subcstruct):
            if subnode.typ.__class__ is self.__class__:
                return subnode.deserialize(subcstruct, refs)
            else:
                if (subnode.schema_type is Sequence):
                    #To deal with iterable schemas that do not have a deserialize with refs function
                    scalar = hasattr(subnode, 'accept_scalar') and subnode.accept_scalar
                    return subnode.typ._impl(subnode, subcstruct, callback, scalar)
                elif (subnode.schema_type is Tuple):
                    return subnode.typ._impl(subnode, subcstruct, callback)
                else:
                    return subnode.deserialize(subcstruct)

        result = self._impl(node, cstruct, callback)
        return self._deser(node, result, refs)

    def flatten(self, node, appstruct, prefix='', listitem=False):
        result = {}
        if listitem:
            selfprefix = prefix
        else:
            if node.name:
                selfprefix = '%s%s.' % (prefix, node.name)
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
            appstruct[next_name] = next_node.typ.set_value(
                next_node, next_appstruct, rest, value)
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
        obj_json = None
        if hasattr(raw_object, 'to_dict'):
            #Passing the 'save' in case a class wants to do some special stuff on
            #saving specifically.
            obj_json = raw_object.to_dict('save')
        else:
            raise TypeError('Object does not have a to_dict function')
        #also adds the object to refs by id
        refs[raw_object.id] = raw_object

        #note you cannot immediately strip out attributes that wont get saved here
        #because they are still needed by _impl
        return obj_json

    def _save(self, node, json_, zipfile_, refs):
        #strips out any entries that do not need saving. They're still in refs,
        #but that shouldn't do any harm.
        savable_attrs = node.get_nodes_by_attr('save')
        for k in json_.keys():
            subnode = node.get(k)
            #Need to exclude lists from this culling, unless explicitly set save=false
            t1 = not isinstance(subnode, SequenceSchema) and not isinstance(subnode, TupleSchema)
            t2 = hasattr(subnode, 'save') and subnode.save == False
            t3 = k not in savable_attrs
            if (t1 and t2 and t3):
                json_.pop(k)

        #replace all save_reference json with just the json filename containing said object
        refd_names = node.get_nodes_by_attr('save_reference')
        for n in refd_names:
            if isinstance(json_[n], list):
                #this is a SequenceSchema or TupleSchema tagged with save_reference
                for i, subjson in enumerate(json_[n]):
                    json_[n][i] = subjson['name'] + '.json'
            else:
                #single reference
                json_[n] = json_[n]['name'] + '.json'

        #Put supporting files into the zipfile and edit their paths in the json
        datafiles = node.get_nodes_by_attr('is_datafile')
        for d in datafiles:
            if isinstance(d, six.string_types):
                json_[d] = self._process_supporting_file(d, zipfile_)
            elif isinstance(d, collections.Iterable):
                #List, tuple, etc
                for i, filename in enumerate(d):
                    json_[d][i] = self._process_supporting_file(filename, zipfile_)

        #Finally, write the json itself to the zipfile, and return the json
        if json_['name'] + '.json' not in zipfile_.namelist():
            zipfile_.writestr(str(json_['name']) + '.json', json.dumps(json_, indent=True))
        return json_

    def save(self, node, appstruct, zipfile_, refs):
        def callback(subnode, subappstruct):
            if not hasattr(subnode, '_save'):
                #This happens when it goes into non-gnome object attributes (Strings, Numbers, etc)
                if (subnode.schema_type is Sequence):# and \
#                     issubclass(subnode.children[0].__class__, ObjTypeSchema):
                    #To be able to continue saving inside iterables, whose schema does not contain a
                    #save function, call the subnode typ _impl with this function as callback
                    #Doing that will execute this function against each item in the iterable,
                    #and should continue the chain if it contains further iterables.
                    scalar = hasattr(subnode, 'accept_scalar') and subnode.accept_scalar
                    return subnode.typ._impl(subnode, subappstruct, callback, scalar)
                elif (subnode.schema_type is Tuple):
                    return subnode.typ._impl(subnode, subappstruct, callback)
                else:
                    #Not an iterable containing Gnome objects, so simply return
                    #the serialization of this non-GNOME object.
                    return subnode.serialize(subappstruct)
            else:
                #This is the path for Gnome objects
                return subnode._save(subappstruct,
                                    zipfile_=zipfile_,
                                    refs=refs)

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
                #This is the path for non-gnome attributes
                if (subnode.schema_type is Sequence):
                    #To deal with iterable schemas that do not have a load function
                    scalar = hasattr(subnode, 'accept_scalar') and subnode.accept_scalar
                    return subnode.typ._impl(subnode, subcstruct, callback, scalar)
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
        #Get all the save_reference attributes and load the files
        refd_attrs = node.get_nodes_by_attr('save_reference')
        for r in refd_attrs:
            if isinstance(cstruct[r], list):
                #Need to turn this into a list of unhydrated object json
                for i, fn in enumerate(cstruct[r]):
                    cstruct[r][i] = self._load_json_from_file(fn, saveloc)
                    cstruct[r][i]['id'] = fn
            else:
                fn = cstruct[r]
                cstruct[r] = self._load_json_from_file(fn, saveloc)
                cstruct[r]['id'] = fn
            #since object id is not saved, but we need a means to ID this object
            #during deserialization, append filename as object id. It's removed
            #later
            print cstruct[r]
        return cstruct

    def _load_json_from_file(self, fname, saveloc):
        '''
        filename is the name of the file in the zip
        saveloc can be a folder or open zipfile.ZipFile object
        '''
        fp = None
        if zipfile.is_zipfile(saveloc):
            fp = saveloc.open(fname, 'rU')
        else:
            fname = os.path.join(saveloc, fname)
            fp = saveloc.open(fp, 'rU')
        return json.load(fp, parse_float=True, parse_int=True)

class ObjTypeSchema(MappingSchema):
    schema_type = ObjType
    '''
    defines the obj_type which is stored by all gnome objects when persisting
    to save files
    It also optionally stores the 'id' if present
    '''
    id = SchemaNode(String(), missing=drop, read=True)
    obj_type = SchemaNode(String(), missing=drop, save=True, read=True)
    name = SchemaNode(String(), missing=drop, save=True, update=True)


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
        if not issubclass(obj.__class__, GnomeId) and not isinstance(obj, GnomeId):
            raise TypeError('This schema cannot save {0}, a non-GNOME object'.format(obj.__class__))
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
            names = [n.name for n in filter(lambda c: hasattr(c, attr) and getattr(c, attr), self.children)]
            #sequences need to be taken into account. If present they will
            #considered to always have 'save' and 'update' as true, read as false,
            return names


class GeneralGnomeObjectSchema(ObjTypeSchema):
    '''
    The purpose of this schema is to be a placeholder in situations where you
    need to specify that an attribute may be one of many different types.

    For example, a PyCurrentMover's .current may be a GridCurrent, an
    IceAwareGridCurrent, a TimeseriesCurrent, etc. Alternatively, you may
    be composing an attribute from several types of Gnome object
    '''

    def __init__(self, acceptable_schemas=None, **kwargs):
        if not acceptable_schemas:
            raise ValueError('Must provide a list of at least one valid schema')
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
            if obj_type._schema in self.acceptable_schemas:
                return obj_type._schema()
            else:
                raise TypeError('This type of json is not supported')
        else:
            obj = obj_or_json
            schema = obj.__class__._schema
            if schema in self.acceptable_schemas:
                return schema()
            else:
                raise TypeError('This type of object is not supported')

    def serialize(self, appstruct=None):
        substitute_schema = self.validate_input_schema(appstruct)
        return substitute_schema.typ.serialize(substitute_schema, appstruct)

    def deserialize(self, cstruct=None, refs=None):
        substitute_schema = self.validate_input_schema(cstruct)
        return substitute_schema.typ.deserialize(substitute_schema, cstruct, refs=refs)

    def _save(self, obj, zipfile_=None, refs=None):
        substitute_schema = self.validate_input_schema(obj)
        return substitute_schema.typ.save(substitute_schema, obj, zipfile_, refs)

    def load(self, obj_json, zipfile_=None, refs=None):
        substitute_schema = self.validate_input_schema(obj_json)
        return substitute_schema.typ.load(substitute_schema, obj_json, zipfile_, refs)


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


class PolygonSet(SequenceSchema):
    polygonset = Polygon()


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
