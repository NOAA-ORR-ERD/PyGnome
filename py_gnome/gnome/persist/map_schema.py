'''
Created on Mar 28, 2013

Schemas associated with classes in the map.py module
'''
from colander import (
    SequenceSchema,           
    SchemaNode,
    MappingSchema,
    String,
    Float,
    drop,
    )

import gnome
from gnome.persist import base_schema

# Should have a schema for different Map classes
class MapPolygon(SequenceSchema):
    bounds = base_schema.LongLat()

class GnomeMap(base_schema.Id, MappingSchema):
    map_bounds = MapPolygon()
    spillable_area = MapPolygon()
    
class MapFromBNA(GnomeMap):    
    filename = SchemaNode(String() )
    refloat_halflife = SchemaNode( Float() )
