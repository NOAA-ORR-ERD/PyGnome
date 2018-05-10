import datetime
import zipfile
import six
import io

from colander import (SchemaNode, deferred, drop, required, Invalid, UnsupportedFields,
                      SequenceSchema, TupleSchema, MappingSchema,
                      String, Float, Int, Mapping, Sequence, Tuple)

from extend_colander import NumpyFixedLenSchema

from gnome.persist.save_load import class_from_objtype


@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of
                                    datetime.datetime.now to when it is called
                                    in Schema
    """
    return datetime.datetime.now().replace(microsecond=0)


class ObjType(Mapping):

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

    def _deser(self, node, value):
        try:
            if type(value) is dict and 'obj_type' in value:
                obj_type = class_from_objtype(value['obj_type'])
                return obj_type.new_from_dict(value)
            else:
                raise TypeError('Object is not dictionary, or does not have an obj_type')
        except Exception as e:
            raise e
            raise Invalid(node, '{0}" does not implement GnomeObj functionality: {1}'.format(value, e))

    def deserialize(self, node, cstruct):
        if cstruct is None:
            return None

        def callback(subnode, subcstruct):
            return subnode.deserialize(subcstruct)

        result = self._impl(node, cstruct, callback)
        return self._deser(node, result)

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

    def _prepare_save(self, node, value, saveloc, refs):
        #Gets the json for the object, as if this were being serialized
        obj_json = None
        if hasattr(value, 'to_dict'):
            obj_json = value.to_dict()
        else:
            raise TypeError('Object does not have a to_dict function')
        #also adds the object to refs by id
        refs[value['id']] = value
        return obj_json

    def _save(self, node, value, saveloc, refs):
        #strips out any entries that do not need saving
        savable_attrs = node.get_nodes_by_attr('save')
        for k in value.keys():
            if k not in savable_attrs:
                value.pop(k)
        #replace all save_reference objects with the json filename containing said object
        refd_names = node.get_nodes_by_attr('save_reference')
        for n in refd_names:
            value[n] = value[n]['name'] + '.json'
        #and lastly, put the json and any supporting files into the 


    def save(self, node, appstruct, saveloc, refs):
        def callback(subnode, subappstruct):
            if not getattr(subnode, '_save'):
                #This happens when it goes into non-gnome object attributes (Strings, Numbers, etc)
                if subnode.schema_type is Sequence or subnode.schema_type is Tuple:
                    #To be able to continue saving inside iterables, whose schema does not contain a
                    #save function, call the subnode typ _impl with this function as callback
                    #Doing that will execute this function against each item in the iterable,
                    #and continue the chain if it contains further iterables.
                    return subnode.typ._impl(subnode, subappstruct, callback)
                else:
                    return subnode.serialize(subappstruct)
            else:
                return subnode.save(subappstruct, saveloc, refs)

        value = self._prepare_save(node, appstruct, saveloc, refs)
        return self._impl(node, value, callback)


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

    def _save(self, obj, saveloc=None, name=None, refs=None):
        if obj is None:
            raise ValueError(self.__class__.__name__ + ': Cannot save a None')
        if obj._schema is not self.__class__:
            raise TypeError('A {0} cannot save a {1}'.format(self.__class__.__name__, obj.__class__.__name__))

        if refs is None:
            refs = Refs()
        if name is None:
            if obj.name is None:
                name = refs.gen_default_name(obj)
            else:
                name = obj.name
        name += '.json'

        obj_json = zip_file = None

        if isinstance(saveloc, six.string_types):
            #This is the top level save call, so create the zipfile buffer
            zip_file = zipfile.ZipFile(io.BytesIO(), 'a')

        obj_json, zip_file = self.typ.save(self, obj, zip_file, refs)

        if saveloc is not None:
            pass

        return (obj_json, zip_file, refs)

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
            return [n.name for n in filter(lambda c: hasattr(c, attr) and getattr(c, attr), self.children)]


class Refs(dict):
    '''
    Class to store and handle references during saving/loading.
    Provides some convenience functions
    '''
    def gen_default_name(self, obj):
        '''
        Goes through the dict, finds all objects of obj.obj_type stored, and
        provides a unique name by appending length+1
        '''
        base_name = obj.obj_type.split('.')[-1]
        num_of_same_type = filter(lambda v: v.obj_type == obj.obj_type, self.values())
        return base_name + num_of_same_type+1


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
