"""
forms.py: Forms for the webgnome package.
"""

from wtforms import (
    Form,
    SelectField,
    DateTimeField,
    IntegerField,
    FloatField,
    BooleanField
)


from wtforms.validators import (
    Required,
)


class AddMoverForm(Form):
    MOVER_CONSTANT_WIND = 'constant_wind'
    MOVER_VARIABLE_WIND = 'variable_wind'

    mover_type = SelectField('Type', choices=(
        (MOVER_CONSTANT_WIND, 'Winds - Constant'),
        (MOVER_VARIABLE_WIND, 'Winds - Variable')
    ))


class WindFieldsMixin(object):
    """
    A mixin of fields common to `VariableWindMoverForm` and
    `ConstantWindMoverForm`
    """
    SPEED_KNOTS = 'knots'
    SPEED_METERS = 'meters'
    SPEED_MILES = 'miles'

    SCALE_RADIANS = 'rad'
    SCALE_DEGREES = 'deg'

    speed = IntegerField('Speed', validators=[Required()])
    speed_type = SelectField(choices=(
        (SPEED_KNOTS, 'Knots'),
        (SPEED_METERS, 'Meters / sec'),
        (SPEED_MILES, 'Miles / hour')
    ))

    active = BooleanField('Active'),
    start_time = IntegerField('Start Time')
    duration = IntegerField('Duration')
    speed_scale = IntegerField('Speed Scale')
    total_angle_scale = FloatField('Total Angle Scale')
    total_angle_scale_type = SelectField(choices=(
        (SCALE_RADIANS, 'rad'),
        (SCALE_DEGREES, 'deg')
    ))


class ConstantWindMoverForm(Form, WindFieldsMixin):
    pass


class VariableWindMoverForm(Form, WindFieldsMixin):
    time = DateTimeField(validators=[Required()])
    auto_increment_time_by = IntegerField('Auto-increment time by')

