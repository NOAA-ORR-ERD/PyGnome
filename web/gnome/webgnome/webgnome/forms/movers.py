import gnome.basic_types
import gnome.movers
import numpy

from wtforms import (
    SelectField,
    IntegerField,
    FloatField,
    BooleanField,
    ValidationError,
    FieldList,
    FormField
)

from wtforms.validators import Required, NumberRange, Optional

from webgnome.model_manager import WindMoverProxy
from webgnome import util
from base import AutoIdForm, DateTimeForm
from object_form import ObjectForm


class WindForm(DateTimeForm):
    """
    A specific value in a :class:`gnome.movers.WindMover` time series.
    Note that this class inherits the ``date``, ``hour`` and ``minute`` fields.
    """
    DIRECTION_DEGREES = 'Degrees true'

    SPEED_KNOTS = 'knots'
    SPEED_METERS = 'meters'
    SPEED_MILES = 'miles'

    SPEED_CHOICES = (
        (SPEED_KNOTS, 'Knots'),
        (SPEED_METERS, 'Meters / sec'),
        (SPEED_MILES, 'Miles / hour')
        )

    speed = IntegerField('Speed', default=0, validators=[NumberRange(min=1)])
    speed_type = SelectField(
        choices=SPEED_CHOICES,
        validators=[Required()],
        default=SPEED_KNOTS
    )

    direction = SelectField(
        'Wind direction is from', default='S',
        choices=[(d, d) for d in
                 [DIRECTION_DEGREES] + util.DirectionConverter.DIRECTIONS],
        validators=[Required()])

    direction_degrees = IntegerField(
        validators=[Optional(), NumberRange(min=0, max=360)])

    def get_direction_degree(self):
        """
        Convert user input for direction into degree.
        """
        if self.direction.data == self.DIRECTION_DEGREES:
            return self.direction_degrees.data
        else:
            return util.DirectionConverter.get_degree(self.direction.data)


class WindMoverForm(ObjectForm):
    """
    A form class representing a :class:`gnome.mover.WindMover` object.
    """
    wrapped_class = WindMoverProxy

    SCALE_RADIANS = 'rad'
    SCALE_DEGREES = 'deg'

    SCALE_CHOICES = (
        (SCALE_RADIANS, 'rad'),
        (SCALE_DEGREES, 'deg')
    )

    timeseries = FieldList(FormField(WindForm), min_entries=1)

    is_active = BooleanField('Active', default=True)
    uncertain_speed_scale = IntegerField('Speed Scale', default=2,
                               validators=[NumberRange(min=0)])
    uncertain_angle_scale = FloatField('Total Angle Scale', default=0.4,
                                   validators=[NumberRange(min=0)])
    uncertain_angle_scale_type = SelectField(
        default=SCALE_RADIANS,
        choices=SCALE_CHOICES,
        validators=[Required()]
    )

    uncertain_time_delay = IntegerField('Start Time', default=0,
        validators=[NumberRange(min=0)])
    uncertain_duration = IntegerField('Duration', default=3,
        validators=[NumberRange(min=0)])

    auto_increment_time_by = IntegerField('Auto-increment time by', default=6)

    def __init__(self, *args, **kwargs):
        """
        Always include an extra field in ``timeseries`` for use as the "Add"
        form. Do this by taking the length of timeseries values passed in from
        an ``obj`` argument and adding one to it.
        """
        obj = kwargs.get('obj', None)
        if obj and hasattr(obj, 'timeseries') and obj.timeseries:
            self.timeseries.kwargs['min_entries'] = len(obj.timeseries) + 1

        super(WindMoverForm, self).__init__(*args, **kwargs)

    def get_timeseries_ndarray(self):
        num_timeseries = len(self.timeseries)
        timeseries = numpy.zeros((num_timeseries,),
            dtype=gnome.basic_types.datetime_r_theta)

        for idx, time_form in enumerate(self.timeseries):
            direction = time_form.get_direction_degree()
            datetime = time_form.get_datetime()
            timeseries['time'][idx] = datetime
            timeseries['value'][idx] = (time_form.speed.data, direction,)

        return timeseries
    
    def create_mover(self):
         return gnome.movers.WindMover(
            is_active=self.is_active.data,
            uncertain_angle_scale=self.uncertain_angle_scale.data,
            uncertain_speed_scale=self.uncertain_speed_scale.data,
            uncertain_duration=self.uncertain_duration.data,
            timeseries=self.get_timeseries_ndarray())


class AddMoverForm(AutoIdForm):
    """
    The initial form used in a multi-step process for adding a mover to the
    user's running model. This step asks the user to choose the type of mover
    to add.
    """
    mover_type = SelectField('Type', choices=(
        (WindMoverForm.get_id(), 'Winds'),
    ))


class DeleteMoverForm(AutoIdForm):
    """
    Delete mover with ``mover_id``. Validates that a mover with that ID exists
    in ``self.model``.

    This is a hidden form submitted via AJAX by the JavaScript client.
    """
    mover_id = IntegerField()

    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(DeleteMoverForm, self).__init__(*args, **kwargs)

    def mover_id_validate(self, field):
        mover_id = field.data

        if mover_id is None or self.model.has_mover_with_id(mover_id) is False:
            raise ValidationError('Mover with that ID does not exist')


