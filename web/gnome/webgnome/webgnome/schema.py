import datetime

from gnome.persist import (
    environment_schema,
    movers_schema,
    model_schema,
    validators,
)
from gnome.persist.extend_colander import LocalDateTime
from colander import (
    MappingSchema,
    SchemaNode,
    Bool,
    Int,
    Float,
    Range,
    String,
    SequenceSchema,
    OneOf,
    TupleSchema,
    deferred,
    drop,
)


@deferred
def now(node, kw):
    return datetime.datetime.now()


class WindMoverSchema(movers_schema.WindMover):
    uncertain_angle_scale_units = SchemaNode(String(), default='rad',
                                             missing='rad',
                                             validator=OneOf(['rad', 'deg']))


class PositionSchema(TupleSchema):
    start_position_x = SchemaNode(Float())
    start_position_y = SchemaNode(Float())
    start_position_z = SchemaNode(Float())


class WindageRangeSchema(TupleSchema):
    windage_min = SchemaNode(Float())
    windage_max = SchemaNode(Float())


class SurfaceReleaseSpillSchema(MappingSchema):
    default_name = 'Surface Release Spill'
    name = SchemaNode(String(), default=default_name, missing=default_name)
    id = SchemaNode(String(), missing=drop)
    num_elements = SchemaNode(Int(), default=0, validator=validators.positive)
    release_time = SchemaNode(LocalDateTime(default_tzinfo=None), default=now,
                              missing=now,
                              validator=validators.convertible_to_seconds)
    end_release_time = SchemaNode(LocalDateTime(default_tzinfo=None),
                                  default=now, missing=drop,
                                  validator=validators.convertible_to_seconds)
    start_position = PositionSchema(default=(0, 0, 0))
    end_position = PositionSchema(default=(0, 0, 0), missing=drop)
    windage_range = WindageRangeSchema(default=(0.01, 0.04))
    windage_persist = SchemaNode(Float(), default=900, missing=900)
    is_active = SchemaNode(Bool(), default=True)


class SurfaceReleaseSpillsSchema(SequenceSchema):
    spill = SurfaceReleaseSpillSchema()


class WindMoversSchema(SequenceSchema):
    mover = WindMoverSchema()


class RandomMoversSchema(SequenceSchema):
    mover = movers_schema.RandomMover()


class MapSchema(model_schema.Map):
    default_name = 'Map'
    name = SchemaNode(String(), default=default_name, missing=default_name)
    relative_path = SchemaNode(String(), default=None, missing=drop)


# Input values GOODS expects for the `resolution` field on a custom map form.
custom_map_resolutions = [
    'c',  # course
    'l',  # low
    'i',  # intermediate
    'h',  # high
    'f'   # full
]


class CustomMapSchema(MappingSchema):
    name = SchemaNode(String(), default="Map")
    north_lat = SchemaNode(Float())
    west_lon = SchemaNode(Float())
    east_lon = SchemaNode(Float())
    south_lat = SchemaNode(Float())
    cross_dateline = SchemaNode(Bool(), missing=False, default=False)
    resolution = SchemaNode(String(), validator=OneOf(custom_map_resolutions),
                            default='i', missing='i')
    refloat_halflife = SchemaNode(Float(), default=1)


class WindsSchema(SequenceSchema):
    wind = environment_schema.Wind()


class ModelSchema(MappingSchema):
    id = SchemaNode(String(), missing=drop)
    start_time = SchemaNode(LocalDateTime(), default=now,
                            validator=validators.convertible_to_seconds)
    duration_days = SchemaNode(Int(), default=1, validator=Range(min=0))
    duration_hours = SchemaNode(Int(), default=0, validator=Range(min=0))
    uncertain = SchemaNode(Bool(), default=False)
    time_step = SchemaNode(Float(), default=0.1)
    surface_release_spills = SurfaceReleaseSpillsSchema(
        default=[], missing=drop)
    wind_movers = WindMoversSchema(default=[], missing=drop)
    random_movers = RandomMoversSchema(default=[], missing=drop)
    winds = WindsSchema(default=[], missing=drop)
    map = MapSchema(missing=drop)


class LocationFileSchema(MappingSchema):
    name = SchemaNode(String())
    latitude = SchemaNode(Float())
    longitude = SchemaNode(Float())
    model_data = ModelSchema()

