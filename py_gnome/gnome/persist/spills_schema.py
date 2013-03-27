'''
Created on Mar 18, 2013

Schema containing Spills
'''

from colander import (
    SchemaNode,
    MappingSchema,
    Bool,
    Float,
    Int,
    String,
    TupleSchema,
    drop,
    Range,
    SequenceSchema
    )

import gnome
from gnome.persist.validators import convertable_to_seconds
from gnome.persist.base_schema import Id, WorldPoint
from gnome.persist.extend_colander import LocalDateTime

""" Unsure if ArrayTypes will be serialized - leave here for now. Not Used """
class ArrayTypeShape(TupleSchema):
     len_ = SchemaNode(Int())

class ArrayType(MappingSchema):
     shape = ArrayTypeShape()
     dtype = SchemaNode( String() )
     #initial_value = SchemaNode( String() )    # Figure out what this is - tuple?
    
class AllArrayTypes(SequenceSchema):
    name = SchemaNode( String() )
    value = ArrayType()

""" End - unused ArrayTypes schema """
    
class Windage(TupleSchema):
    min_windage = SchemaNode( Float(), validator=Range(0.01, 0.04) )
    max_windage = SchemaNode( Float(), validator=Range(0.01, 0.04) )

class Spill(MappingSchema):
    """Base schema class from which spills that are serialized derive"""
    on = SchemaNode(Bool(), default=True, missing=True)
    num_elements = SchemaNode( Int() )

class SurfaceReleaseSpill(Id, Spill):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    _create= ['', '', '','']
    start_position = WorldPoint()
    release_time = SchemaNode(LocalDateTime(), validator=convertable_to_seconds)
    end_position = WorldPoint(missing=None)
    end_release_time = SchemaNode(LocalDateTime(), missing=None, validator=convertable_to_seconds)
    windage_range = Windage()
    windage_persist = SchemaNode( Float() )
    
    # following will be used when restoring a saved scenario that is partially run
    num_released = SchemaNode(Int(), missing=drop)
    not_called_yet = SchemaNode( Bool(), missing=drop)
    prev_release_pos = WorldPoint( missing=drop)    # Not sure how this will work w/ WebGnome
    delta_pos = WorldPoint(missing=drop)