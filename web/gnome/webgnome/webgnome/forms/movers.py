import gnome.movers

from wtforms import (
    SelectField,
    IntegerField,
    FloatField,
    BooleanField,
    TextField,
    ValidationError
)

from wtforms.validators import Required, NumberRange

from base import AutoIdForm, DateTimeForm
from object_form import ObjectForm


class WindMoverForm(ObjectForm, DateTimeForm):
    """
    A form class representing a :class:`gnome.mover.WindMover` object.

    This form is used for both "variable" and "constant" wind movers, the
    difference being the number of time series values entered.
    """
    wrapped_class = gnome.movers.WindMover

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
    direction = TextField(
        'Wind direction is from',
        default='S',
        validators=[Required()]
    )

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


