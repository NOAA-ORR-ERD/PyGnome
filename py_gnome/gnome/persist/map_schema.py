'''
Created on Mar 28, 2013

Schemas for map and output_map used by model

Currently associated with classes in the map.py module and canvas_map.py module
'''
from colander import (
    SequenceSchema,           
    SchemaNode,
    MappingSchema,
    TupleSchema,
    String,
    Float,
    drop,
    Int,
    )

import gnome
from gnome.persist import base_schema

"""
Schemas for gnome map classes (map.py module) used for computation
"""
class LongLatBounds(SequenceSchema):
    """ used to define bounds on a map """
    bounds = base_schema.LongLat()

class GnomeMap(base_schema.Id, MappingSchema):
    map_bounds = LongLatBounds()
    spillable_area = LongLatBounds()
    
class MapFromBNA(GnomeMap):    
    filename = SchemaNode(String() )
    refloat_halflife = SchemaNode( Float() )

"""
Schemas for gnome output_map classes (map_canvas.py module) used for display
"""
class ImageSize(TupleSchema):
    """only contains 2D (long, lat) positions"""
    width = SchemaNode( Int() )
    height = SchemaNode( Int() )
    
class MapCanvasFromBNA(base_schema.Id, MappingSchema):
    viewport = LongLatBounds()  # not sure if bounding box needs defintion separate from LongLatBounds
    
    # following are only used when creating objects, not updating - so missing=drop
    filename = SchemaNode(String(), missing=drop)
    projection_type = SchemaNode(String(), missing=drop) 
    image_size= ImageSize(missing=drop)
    