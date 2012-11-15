import copy
import gnome.movers
import math

from wtforms import (
    SelectField,
    IntegerField,
    FloatField,
    BooleanField,
    TextField,
    ValidationError
)

from wtforms.validators import Required, NumberRange, Optional

from base import AutoIdForm, DateTimeForm
from object_form import ObjectForm


class WindMoverForm(ObjectForm, DateTimeForm):
    """
    A form class representing a :class:`gnome.mover.WindMover` object.

    This form is used for both "variable" and "constant" wind movers, the
    difference being the number of time series values entered.
    """
    wrapped_class = gnome.movers.WindMover

    DIRECTION_CUSTOM = 'Custom'

    DIRECTIONS = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW"
    ]

    SPEED_KNOTS = 'knots'
    SPEED_METERS = 'meters'
    SPEED_MILES = 'miles'

    SPEED_CHOICES = (
        (SPEED_KNOTS, 'Knots'),
        (SPEED_METERS, 'Meters / sec'),
        (SPEED_MILES, 'Miles / hour')
    )

    SCALE_RADIANS = 'rad'
    SCALE_DEGREES = 'deg'

    SCALE_CHOICES = (
        (SCALE_RADIANS, 'rad'),
        (SCALE_DEGREES, 'deg')
    )

    auto_increment_time_by = IntegerField('Auto-increment time by')
    speed = IntegerField('Speed', default=0, validators=[NumberRange(min=0)])
    speed_type = SelectField(
        choices=SPEED_CHOICES,
        validators=[Required()]
    )

    direction = SelectField(
        'Wind direction is from', default='S',
        choices=[(d, d) for d in ['Degrees true'] + DIRECTIONS],
        validators=[Required()])

    direction_degrees = IntegerField(
        validators=[Optional(), NumberRange(min=0,  max=360)])

    is_active = BooleanField('Active', default=True)
    start_time = IntegerField('Start Time', default=0, validators=[NumberRange(min=0)])
    duration = IntegerField('Duration', default=3, validators=[NumberRange(min=0)])
    speed_scale = IntegerField('Speed Scale', default=2,
                               validators=[NumberRange(min=0)])
    total_angle_scale = FloatField('Total Angle Scale', default=0.4,
                                   validators=[NumberRange(min=0)])
    total_angle_scale_type = SelectField(
        default=SCALE_RADIANS,
        choices=SCALE_CHOICES,
        validators=[Required()]
    )

    def get_direction_degree(self):
        """
        Convert user input for direction into degree.
        """
        if self.direction.data == self.DIRECTION_CUSTOM:
            return self.direction_custom.data
        elif self.direction.data in self.DIRECTIONS:
            idx = self.DIRECTIONS.index(self.direction.data)
            return (360.0 / 16) * idx


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


