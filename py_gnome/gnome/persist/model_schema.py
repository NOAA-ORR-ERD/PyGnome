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
    )

import gnome
from gnome.persist import validators, extend_colander, base_schema

class UpdateModel(MappingSchema):
    time_step = SchemaNode( Float()) 
    start_time= SchemaNode(extend_colander.LocalDateTime(), validator=validators.convertable_to_seconds)
    duration = SchemaNode(extend_colander.TimeDelta() )   # put a constraint for max duration?
    movers = base_schema.OrderedCollection()
    environment = base_schema.OrderedCollection()
    uncertain = SchemaNode( Bool() )
    
class CreateModel(base_schema.Id, UpdateModel):
    """
    Likely to be used when validating the state of the object read in from a save file.
    
    This is a union of the properties in UpdateWind and Id
    """
    pass