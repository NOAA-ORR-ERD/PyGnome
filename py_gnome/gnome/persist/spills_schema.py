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

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.base_schema import Id, WorldPoint
from gnome.persist.extend_colander import LocalDateTime


""" Unsure if ArrayTypes will be serialized - leave here for now. Not Used """


class ArrayTypeShape(TupleSchema):
    len_ = SchemaNode(Int())


class ArrayType(MappingSchema):
    shape = ArrayTypeShape()
    dtype = SchemaNode(String())
    #initial_value = SchemaNode(String())  # Figure out what this is - tuple?


class AllArrayTypes(SequenceSchema):
    name = SchemaNode(String())
    value = ArrayType()

""" End - unused ArrayTypes schema """


class Windage(TupleSchema):
    min_windage = SchemaNode(Float(), validator=Range(0, 1.0), default=0.01)
    max_windage = SchemaNode(Float(), validator=Range(0, 1.0), default=0.04)


class Spill(MappingSchema):
    """Base schema class from which spills that are serialized derive"""
    on = SchemaNode(Bool(), default=True, missing=True)
    num_elements = SchemaNode(Int(), default=1000)


class PointSourceSurfaceRelease(Id, Spill):
    """
    Contains properties required by UpdateWindMover and CreateWindMover
    """
    start_position = WorldPoint()
    release_time = SchemaNode(LocalDateTime(),
                              validator=convertible_to_seconds)
    end_position = WorldPoint(missing=drop)
    end_release_time = SchemaNode(LocalDateTime(), missing=drop,
                                  validator=convertible_to_seconds)
    windage_range = Windage(default=(0.01, 0.04))
    windage_persist = SchemaNode(Float(), default=900)

    # following will be used when restoring a saved scenario that is
    # partially run
    num_released = SchemaNode(Int(), missing=drop)
    not_called_yet = SchemaNode(Bool(), missing=drop)

    # Not sure how this will work w/ WebGnome
    prev_release_pos = WorldPoint(missing=drop)

    delta_pos = WorldPoint(missing=drop)
