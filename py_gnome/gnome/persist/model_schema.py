'''
Created on Mar 4, 2013
'''
from datetime import timedelta

from colander import (
    SchemaNode,
    MappingSchema,
    Bool,
    Float,
    Range,
    TupleSchema,
    Int,
    String,
    SequenceSchema,
    deferred
    )

import gnome
from gnome.persist import validators, extend_colander, base_schema

@deferred
def model_uncertain(node, kw):
    """
    Used by TimeseriesValueSchema - assume it defers the calculation of datetime.datetime.now to when it is called in Schema
    """
    print node
    print kw
    uncertain_spills = None
    #base_schema.OrderedCollection()
    return uncertain_spills

class ArrayTypeShape(TupleSchema):
     len_ = SchemaNode(Int())

class ArrayType(MappingSchema):
     shape = ArrayTypeShape()
     dtype = SchemaNode( String() )
     #initial_value = SchemaNode( String() )    # Figure out what this is - tuple?
    
class AllArrayTypes(SequenceSchema):
    name = SchemaNode( String() )
    value = ArrayType()
    
    
class SpillContainerPair(MappingSchema):
    certain_spills = base_schema.OrderedCollection()
    #uncertain_spills= SchemaNode( model_uncertain )
    #print "model_uncertain: {0}".format(model_uncertain)
    #if uncertain:
    #    uncertain_spills = base_schema.OrderedCollection()
    

class UpdateModel(MappingSchema):
    time_step = SchemaNode( Float()) 
    start_time= SchemaNode(extend_colander.LocalDateTime(), validator=validators.convertable_to_seconds)
    duration = SchemaNode(extend_colander.TimeDelta() )   # put a constraint for max duration?
    movers = base_schema.OrderedCollection()
    environment = base_schema.OrderedCollection()
    uncertain = SchemaNode( Bool() )
    spills = SpillContainerPair(uncertain=False)
    
class CreateModel(base_schema.Id, UpdateModel):
    """
    Likely to be used when validating the state of the object read in from a save file.
    
    This is a union of the properties in UpdateWind and Id
    """
    pass
