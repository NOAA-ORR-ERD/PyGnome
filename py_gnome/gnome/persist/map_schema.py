'''
Created on Mar 28, 2013

Schemas for map and output_map used by model

Currently associated with classes in the map.py module and canvas_map.py module
'''

from colander import SequenceSchema, SchemaNode, MappingSchema, \
    TupleSchema, String, Float, drop, Int

import gnome
from gnome.persist import base_schema


class GnomeMap(base_schema.Id, MappingSchema):

    map_bounds = base_schema.LongLatBounds()
    spillable_area = base_schema.LongLatBounds(missing=drop)


class MapFromBNA(GnomeMap):

    filename = SchemaNode(String())
    refloat_halflife = SchemaNode(Float())


