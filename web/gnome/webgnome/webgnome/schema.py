import gnome.basic_types
import datetime
import numpy
import time

from webgnome import util
util.monkey_patch_colander()

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
    deferred,
    null,
    drop,
    Tuple
)

from webgnome.model_manager import WebWind


def get_direction_degree(direction):
    """
    Convert user input for direction into degrees.
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


def positive(node, value):
    if value <= 0:
        raise Invalid(node, 'Value must be greater than zero.')


def zero_or_greater(node, value):
    if value < 0:
        raise Invalid(node, 'Value must be zero or greater.')


def convertable_to_seconds(node, value):
    try:
        time.mktime(list(value.timetuple()))
    except (OverflowError, ValueError) as e:
        raise Invalid(node, 'Invalid date.')


def degrees_true(node, direction):
    if 0 > direction > 360:
        raise Invalid(
            node, 'Direction in degrees true must be between 0 and 360.')


def cardinal_direction(node, direction):
    if not util.DirectionConverter.is_cardinal_direction(direction):
        raise Invalid(
            node, 'A cardinal directions must be one of: %s' % ', '.join(
                util.DirectionConverter.DIRECTIONS))


def no_duplicates(node, values):
    """
    Reject ``values`` if it contains duplicates.
    """
    try:
        unique = numpy.unique(values)
    except AttributeError:
        return

    num_dups = len(values) - len(unique)

    if num_dups:
        raise Invalid(
            node, 'Duplicates are not allowed. Found %s duplicates.' % num_dups)


def valid_direction(node, value):
    """
    Unused.
    """
    try:
        degrees_true(node, float(value))
    except ValueError:
        cardinal_direction(node, value.upper())


class LocalDateTime(DateTime):
    def __init__(self, *args, **kwargs):
        kwargs['default_tzinfo'] = kwargs.get('default_tzinfo', None)
        super(LocalDateTime, self).__init__(*args, **kwargs)

    def strip_timezone(self, _datetime):
        if _datetime and isinstance(_datetime, datetime.datetime)\
                or isinstance(_datetime, datetime.date):
            _datetime = _datetime.replace(tzinfo=None)
        return _datetime

    def serialize(self, node, appstruct):
        appstruct = self.strip_timezone(appstruct)
        return super(LocalDateTime, self).serialize(node, appstruct)

    def deserialize(self, node, cstruct):
        dt = super(LocalDateTime, self).deserialize(node, cstruct)
        return self.strip_timezone(dt)


class DefaultTuple(Tuple):
    """
    A Tuple subclass that provides defaults from child nodes.

    Required because Tuple returns `colander.null` by default when ``appstruct``
    is not provided, instead of creating a Tuple of default values.
    """
    def serialize(self, node, appstruct):
        items = super(DefaultTuple, self).serialize(node, appstruct)

        if items is null and node.children:
            items = tuple([field.default for field in node.children])

        return items


class DefaultTupleSchema(TupleSchema):
    schema_type = DefaultTuple


class TimeseriesValueSchema(DefaultTupleSchema):
    datetime = SchemaNode(LocalDateTime(default_tzinfo=None), default=now,
                          validator=convertable_to_seconds)
    speed = SchemaNode(Float(), default=0, validator=zero_or_greater)
    # TODO: Validate string and float or just float?
    direction = SchemaNode(Float(), default=0, validator=degrees_true)


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
            timeseries['time'][idx] = value[0]
            timeseries['value'][idx] = (value[1], value[2])

        return timeseries


class DatetimeValue2dArraySchema(SequenceSchema):
    schema_type = DatetimeValue2dArray


class WindTimeSeriesSchema(DatetimeValue2dArraySchema):
    value = TimeseriesValueSchema(default=(datetime.datetime.now(), 0, 0))


class WindSchema(MappingSchema):
    source_id = SchemaNode(String(), default=None, missing=None)
    source_type = SchemaNode(String(), default='manual', missing='manual',
                             validator=OneOf([source[0] for source in
                                              WebWind.source_types]))
    description = SchemaNode(String(), default=None, missing=None)
    timeseries = WindTimeSeriesSchema(default=[], validator=no_duplicates)
    units = SchemaNode(String(), validator=OneOf(util.velocity_unit_values),
                       default='m/s')
    latitude = SchemaNode(Float(), default=None, missing=None)
    longitude = SchemaNode(Float(), default=None, missing=None)
    updated_at = SchemaNode(LocalDateTime(), default=None, missing=None, )


class BaseMoverSchema(MappingSchema):
    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), default=None, missing=None,
                              validator=convertable_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), default=None, missing=None,
                             validator=convertable_to_seconds)


class WindMoverSchema(BaseMoverSchema):
    default_name = 'Wind Mover'
    wind = WindSchema()
    name = SchemaNode(String(), default=default_name, missing=default_name)
    uncertain_duration = SchemaNode(Float(), default=3, validator=Range(min=0))
    uncertain_time_delay = SchemaNode(Float(), default=0, validator=Range(min=0))
    uncertain_speed_scale = SchemaNode(Float(), default=2, validator=Range(min=0))
    uncertain_angle_scale = SchemaNode(Float(), default=0.4, validator=Range(min=0))
    uncertain_angle_scale_units = SchemaNode(String(), default='rad', missing='rad',
                                             validator=OneOf(['rad', 'deg']))


class RandomMoverSchema(BaseMoverSchema):
    default_name = 'Random Mover'
    name = SchemaNode(String(), default=default_name, missing=default_name)
    diffusion_coef = SchemaNode(Float(), default=100000, missing=100000)


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
    num_elements = SchemaNode(Int(), default=0, validator=positive)
    release_time = SchemaNode(LocalDateTime(default_tzinfo=None), default=now,
                              missing=now, validator=convertable_to_seconds)
    end_release_time = SchemaNode(LocalDateTime(default_tzinfo=None),
                                  default=now, missing=drop,
                                  validator=convertable_to_seconds)
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
    mover = RandomMoverSchema()


class MapBoundarySchema(TupleSchema):
    x = SchemaNode(Float())
    y = SchemaNode(Float())


class MapBoundsSchema(SequenceSchema):
    """
    Bounds of a map.

    TODO: A sequence schema accepts a variable-length list. Should this be
    a Tuple instead and accept only four items?
    """
    boundary = MapBoundarySchema()


default_map_bounds = ((-360, 90),
                      ( 360, 90),
                      ( 360, -90),
                      (-360, -90))


class MapSchema(MappingSchema):
    name = SchemaNode(String(), default="Map")
    filename = SchemaNode(String(), default=None, missing=None)
    refloat_halflife = SchemaNode(Float(), default=1)
    map_bounds = MapBoundsSchema(default=default_map_bounds,
                                 missing=default_map_bounds)
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


class ModelSchema(MappingSchema):
    start_time = SchemaNode(LocalDateTime(), default=now,
                            validator=convertable_to_seconds)
    duration_days = SchemaNode(Int(), default=1, validator=Range(min=0))
    duration_hours = SchemaNode(Int(),default=0, validator=Range(min=0))
    uncertain = SchemaNode(Bool(), default=False)
    time_step = SchemaNode(Float(), default=0.1)
    surface_release_spills = SurfaceReleaseSpillsSchema(default=[], missing=drop)
    wind_movers = WindMoversSchema(default=[], missing=drop)
    random_movers = RandomMoversSchema(default=[], missing=drop)
    map = MapSchema(missing=drop)


class LocationFileSchema(MappingSchema):
    name = SchemaNode(String())
    latitude = SchemaNode(Float())
    longitude = SchemaNode(Float())
    model_data = ModelSchema()

