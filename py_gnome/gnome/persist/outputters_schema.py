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
    Bool,
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

class NetCDFOutput(base_schema.Id, MappingSchema):
    netcdf_filename = SchemaNode(String(), missing=drop)
    all_data = SchemaNode(Bool(), missing=drop)
    format = SchemaNode(String(), missing=drop)
    compress = SchemaNode(Bool(), missing=drop)
    _start_idx = SchemaNode(Int(), missing=drop)
    _middle_of_run = SchemaNode(Bool(), missing=drop)