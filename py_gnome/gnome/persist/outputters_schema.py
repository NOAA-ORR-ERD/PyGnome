'''
Outputters schema
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
Schemas for Outputters
"""   
class Renderer(base_schema.Id, MappingSchema):
    viewport = base_schema.LongLatBounds()  # not sure if bounding box needs defintion separate from LongLatBounds
    
    # following are only used when creating objects, not updating - so missing=drop
    filename = SchemaNode(String(), missing=drop)
    projection_class = SchemaNode(String(), missing=drop) 
    image_size= base_schema.ImageSize(missing=drop)
    images_dir= SchemaNode( String() )
