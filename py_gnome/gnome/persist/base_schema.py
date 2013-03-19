import datetime

from colander import (
    MappingSchema,
    SchemaNode,
    String,
    deferred,
    SequenceSchema,
    TupleSchema
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
    id = SchemaNode(String() )

class OrderedCollectionIdListItem(TupleSchema):
    obj_type = SchemaNode(String() )  
    obj_id = SchemaNode(String() )

class OrderedCollectionIdList(SequenceSchema):
    id_list = OrderedCollectionIdListItem() 
    
class OrderedCollection(MappingSchema):
    dtype = SchemaNode( String() )
    id_list = OrderedCollectionIdList(missing=None)
    