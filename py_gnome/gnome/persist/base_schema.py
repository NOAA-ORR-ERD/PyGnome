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


class Id(MappingSchema):
    '''
    Any need to ensure it is valid UUID?
    '''
    id = SchemaNode(String(), missing=drop)
    obj_type = SchemaNode(String(), missing=drop)


class OrderedCollectionIdListItem(TupleSchema):
    obj_type = SchemaNode(String())
    obj_idx = SchemaNode(String())


class OrderedCollectionIdList(SequenceSchema):
    items = OrderedCollectionIdListItem()


class OrderedCollection(MappingSchema):
    dtype = SchemaNode(String(), missing=drop)
    items = OrderedCollectionIdList(missing=drop)


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


