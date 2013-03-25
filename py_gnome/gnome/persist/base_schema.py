import datetime

from colander import (
    MappingSchema,
    SchemaNode,
    String,
    deferred,
    SequenceSchema,
    TupleSchema,
    Float,
    drop
)

@deferred
def now(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of datetime.datetime.now to when it is called in Schema
    """
    return datetime.datetime.now()


class Id(MappingSchema):
    """
    any need to ensure it is valid UUID?
    """
    id = SchemaNode(String(), missing=drop)
    obj_type = SchemaNode(String(), missing=drop)

class OrderedCollectionIdListItem(TupleSchema):
    obj_type = SchemaNode(String() )  
    obj_id = SchemaNode(String() )

class OrderedCollectionIdList(SequenceSchema):
    id_list = OrderedCollectionIdListItem() 
    
class OrderedCollection(MappingSchema):
    dtype = SchemaNode( String(), missing=drop)
    id_list = OrderedCollectionIdList(missing=drop)
    
class LongLat(TupleSchema):
    """only contains 2D (long, lat) positions"""
    long = SchemaNode( Float() )
    lat = SchemaNode( Float() )
    
class WorldPoint(LongLat):
    """used to define reference points. 3D positions (long,lat,z)"""
    z = SchemaNode( Float(), default=0.0)    