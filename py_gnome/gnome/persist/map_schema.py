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
class GnomeMap(base_schema.Id, MappingSchema):
    map_bounds = base_schema.LongLatBounds()
    spillable_area = base_schema.LongLatBounds(missing=drop)
    
class MapFromBNA(GnomeMap):    
    filename = SchemaNode(String() )
    refloat_halflife = SchemaNode( Float() )

"""
Schemas for gnome output_map classes (map_canvas.py module) used for display

Now deprecated. Added the notion of outputters to the model. The Renderer object
is used instead of MapCanvasFromBNA for making images. It has has more methods.
"""   
class MapCanvasFromBNA(base_schema.Id, MappingSchema):
    viewport = base_schema.LongLatBounds()  # not sure if bounding box needs defintion separate from LongLatBounds
    
    # following are only used when creating objects, not updating - so missing=drop
    filename = SchemaNode(String(), missing=drop)
    projection_class = SchemaNode(String(), missing=drop) 
    image_size= base_schema.ImageSize(missing=drop)
    