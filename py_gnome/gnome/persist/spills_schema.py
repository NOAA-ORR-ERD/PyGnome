'''
Created on Mar 18, 2013

Schema containing Spills
'''

from colander import (SchemaNode, drop,
                      SequenceSchema, TupleSchema, MappingSchema,
                      Bool, Float, Int, String, Range)

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.base_schema import Id, WorldPoint
from gnome.persist.extend_colander import LocalDateTime
from gnome.persist.elements_schema import ElementType


class ArrayTypeShape(TupleSchema):
    len_ = SchemaNode(Int())


class ArrayType(MappingSchema):
    shape = ArrayTypeShape()
    dtype = SchemaNode(String())


    # initial_value = SchemaNode(String())  # Figure out what this is - tuple?


class AllArrayTypes(SequenceSchema):
    name = SchemaNode(String())
    value = ArrayType()


class Release(Id):
    'Base class for Release schemas'
    num_elements = SchemaNode(Int(), default=1000)

    # used to create a new Release object if model is persisted mid-run
    num_released = SchemaNode(Int())
    start_time_invalid = SchemaNode(Bool())
    name = 'release'


class PointLineRelease(Release):
    '''
    Contains properties required by UpdateWindMover and CreateWindMover
    TODO: also need a way to persist list of element_types
    '''
    start_position = WorldPoint()
    release_time = SchemaNode(LocalDateTime(),
                              validator=convertible_to_seconds)
    end_position = WorldPoint(missing=drop)
    end_release_time = SchemaNode(LocalDateTime(), missing=drop,
                                  validator=convertible_to_seconds)

    # following will be used when restoring a saved scenario that is
    # partially run
    num_released = SchemaNode(Int(), missing=drop)
    start_time_invalid = SchemaNode(Bool(), missing=drop)

    # Not sure how this will work w/ WebGnome
    prev_release_pos = WorldPoint(missing=drop)
    description = 'PointLineRelease object schema'


class Spill(Id):
    'Spill class schema'
    on = SchemaNode(Bool(), default=True, missing=True,
        description='on/off status of spill')
