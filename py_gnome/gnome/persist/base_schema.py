import datetime

from colander import (SchemaNode, deferred, drop,
                      SequenceSchema, TupleSchema, MappingSchema,
                      String, Float, Int)


@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of
                                    datetime.datetime.now to when it is called
                                    in Schema
    """
    return datetime.datetime.now()


class ObjType(MappingSchema):
    '''
    defines the obj_type which is stored by all gnome objects when persisting
    to save files
    It also optionally stores the 'id' if present
    '''
    id = SchemaNode(String(), missing=drop)
    obj_type = SchemaNode(String(), missing=drop)
    name = SchemaNode(String(), missing=drop)
    json_ = SchemaNode(String())    # either 'webapi' or 'save'


class OrderedCollectionItemMap(MappingSchema):
    '''
    This stores the obj_type and obj_index. An alternative schema for
    orderedcollection just stores the IDs (UUIDs) of objects in list
    '''
    obj_type = SchemaNode(String())
    json_file = SchemaNode(String(), missing=drop)
    id = SchemaNode(String(), missing=drop)


class OrderedCollectionItemsList(SequenceSchema):
    item = OrderedCollectionItemMap()


class LongLat(TupleSchema):
    'Only contains 2D (long, lat) positions'
    long = SchemaNode(Float())
    lat = SchemaNode(Float())


class LongLatBounds(SequenceSchema):
    'Used to define bounds on a map'
    bounds = LongLat()


class WorldPoint(LongLat):
    'Used to define reference points. 3D positions (long,lat,z)'
    z = SchemaNode(Float(), default=0.0)


class ImageSize(TupleSchema):
    'Only contains 2D (long, lat) positions'
    width = SchemaNode(Int())
    height = SchemaNode(Int())
