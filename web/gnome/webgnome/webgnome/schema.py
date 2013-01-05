import gnome.basic_types
import datetime
import numpy

from colander import (
    MappingSchema,
    SchemaNode,
    Bool,
    Int,
    Float,
    Range,
    DateTime,
    String,
    SequenceSchema,
    OneOf,
    Invalid,
    Sequence,
    TupleSchema,
    deferred
)

from webgnome import util


def get_direction_degree(direction):
    """
    Convert user input for direction into degree.
    """
    if direction.isalpha():
        return util.DirectionConverter.get_degree(direction)
    else:
        return direction


def get_timeseries_ndarray(timeseries):
    num_timeseries = len(timeseries)
    timeseries = numpy.zeros((num_timeseries,),
                             dtype=gnome.basic_types.datetime_value_2d)

    for idx, wind_value in enumerate(timeseries):
        direction = get_direction_degree(wind_value['direction'])
        timeseries['time'][idx] = wind_value['datetime']
        timeseries['value'][idx] = (wind_value['speed'], direction)


@deferred
def now(node, kw):
    return datetime.datetime.now()


def _validate_degrees_true(node, direction):
    if 0 > direction > 360:
        raise Invalid(
            node, 'Direction in degrees true must be between 0 and 360.')


def _validate_cardinal_direction(node, direction):
    if not util.DirectionConverter.is_cardinal_direction(direction):
        raise Invalid(
            node, 'A cardinal directions must be one of: %s' % ', '.join(
                util.DirectionConverter.DIRECTIONS))


def validate_direction(node, value):
    try:
        _validate_degrees_true(node, float(value))
    except ValueError:
        _validate_cardinal_direction(node, value.upper())


class WindValueSchema(MappingSchema):
    datetime = SchemaNode(DateTime(default_tzinfo=None), default=now)
    speed = SchemaNode(Float(), default=0, validator=Range(min=0))
    direction = SchemaNode(Float(), default=0)


class DatetimeValue2dArray(Sequence):
    """
    A subclass of :class:`colander.Sequence` that converts itself to a numpy
    array using :class:`gnome.basic_types.datetime_value_2d` as the data type.
    """

    def deserialize(self, *args, **kwargs):
        items = super(DatetimeValue2dArray, self).deserialize(*args, **kwargs)
        num_timeseries = len(items)
        timeseries = numpy.zeros((num_timeseries,),
                                 dtype=gnome.basic_types.datetime_value_2d)

        for idx, value in enumerate(items):
            direction = value['direction']
            datetime = value['datetime']
            timeseries['time'][idx] = datetime
            timeseries['value'][idx] = (value['speed'], direction)

        return timeseries


class DatetimeValue2dArraySchema(SequenceSchema):
    schema_type = DatetimeValue2dArray


class WindTimeSeriesSchema(DatetimeValue2dArraySchema):
    value = WindValueSchema()


class WindSchema(MappingSchema):
    timeseries = WindTimeSeriesSchema(default=[])
    units = SchemaNode(String(), validator=OneOf(util.velocity_unit_values),
                       default='m/s')


class WindMoverSchema(MappingSchema):
    default_name = 'Wind Mover'
    wind = WindSchema()
    is_active = SchemaNode(Bool(), default=True)
    name = SchemaNode(String(), default=default_name, missing=default_name)
    uncertain_duration = SchemaNode(Float(), default=3, validator=Range(min=0))
    uncertain_time_delay = SchemaNode(Float(), default=0, validator=Range(min=0))
    uncertain_speed_scale = SchemaNode(Float(), default=2, validator=Range(min=0))
    uncertain_angle_scale = SchemaNode(Float(), default=0.4, validator=Range(min=0))
    uncertain_angle_scale_units = SchemaNode(String(), default='rad', missing='rad',
                                             validator=OneOf(['rad', 'deg']))


class PositionSchema(TupleSchema):
    start_position_x = SchemaNode(Float())
    start_position_y = SchemaNode(Float())
    start_position_z = SchemaNode(Float())


class WindageSchema(TupleSchema):
    windage_min = SchemaNode(Float())
    windage_max = SchemaNode(Float())


class PointReleaseSpillSchema(MappingSchema):
    default_name = 'Point Release Spill'
    num_LEs = SchemaNode(Int(), default=0)
    release_time = SchemaNode(DateTime(default_tzinfo=None), default=now)
    start_position = PositionSchema(default=(0, 0, 0))
    windage = WindageSchema(default=(0.01, 0.04))
    persist = SchemaNode(Float(), default=900)
    uncertain = SchemaNode(Bool(), default=False)
    is_active = SchemaNode(Bool(), default=True)
    name = SchemaNode(String(), default=default_name, missing=default_name)


class PointReleaseSpillsSchema(SequenceSchema):
    spill = PointReleaseSpillSchema()


class WindMoversSchema(SequenceSchema):
    mover = WindMoverSchema()


class ModelSettingsSchema(MappingSchema):
    start_time = SchemaNode(DateTime(default_tzinfo=None), default=now)
    duration_days = SchemaNode(Int(), default=1, validator=Range(min=0))
    duration_hours = SchemaNode(Int(),default=0, validator=Range(min=0))
    uncertain = SchemaNode(Bool(), default=False)
    time_step = SchemaNode(Float(), default=0.1)


class ModelSchema(ModelSettingsSchema):
    point_release_spills = PointReleaseSpillsSchema(default=[])
    wind_movers = WindMoversSchema(default=[])
