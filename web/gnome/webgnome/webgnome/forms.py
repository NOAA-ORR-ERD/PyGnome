"""
forms.py: Forms for the webgnome package.
"""

from wtforms import (
    Form,
    SelectField,
    DateTimeField,
    IntegerField,
    FloatField,
    BooleanField,
    TextField
)


from wtforms.validators import (
    Required,
    NumberRange
)


class AddMoverForm(Form):
    MOVER_CONSTANT_WIND = 'constant_wind'
    MOVER_VARIABLE_WIND = 'variable_wind'

    mover_type = SelectField('Type', choices=(
        (MOVER_CONSTANT_WIND, 'Winds - Constant'),
        (MOVER_VARIABLE_WIND, 'Winds - Variable')
    ))


class WindMoverForm(Form):
    """
    A form class containing fields common to `VariableWindMoverForm` and
    `ConstantWindMoverForm`
    """
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

    speed = IntegerField('Speed', default=0, validators=[NumberRange(min=0)])
    speed_type = SelectField(
        choices=SPEED_CHOICES,
        validators=[Required()]
    )
    direction = TextField(
        default='S',
        validators=[Required()]
    )

    is_active = BooleanField('Active', default=True, validators=[Required()])
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


class ConstantWindMoverForm(WindMoverForm):
    pass


class VariableWindMoverForm(WindMoverForm):
    time = DateTimeField(validators=[Required()])
    auto_increment_time_by = IntegerField('Auto-increment time by')

