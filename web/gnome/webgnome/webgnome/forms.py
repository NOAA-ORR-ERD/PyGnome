"""
forms.py: Forms for the webgnome package.
"""
import datetime

from wtforms import (
    Form,
    SelectField,
    DateTimeField,
    IntegerField,
    FloatField,
    BooleanField,
    TextField,
    HiddenField
)

from wtforms.widgets import TextInput

from wtforms.validators import (
    Required,
    NumberRange
)


class DatePickerWidget(TextInput):
    """
    A widget that assigns itself a CSS class needed to render a jQuery UI
    date picker.
    """
    def __call__(self, field, **kwargs):
        kwargs['class']  = 'date'
        return super(DatePickerWidget, self).__call__(field, **kwargs)


class LeadingZeroNumberWidget(TextInput):
    """
    A widget that will pad single-digit numbers with leading zeroes.

    To use, create a subclass and provide the class variable `cast_to`, which
    should be a callable used to cast the field's value, e.g.:

            class IntegerLeadingZeroWidget(LeadingZeroNumberWidget):
                cast_to = int

    The result for, e.g. an `IntegerLeadingZeroWidget`:

            User input: 1
            Result of calling the widget: 01

    This widget is used primarily to adjust user input so that it may be fed
    directly into a `datetime.datetime` or `datetime.date` constructor for, e.g.,
    an hour or minute value.
    """
    def cast_to(self, number):
        raise NotImplementedError

    def cast(self, number):
        """
        Try to use the class's `cast_to` value to cast `number`.

        Return the casted value if it worked; otherwise return None.
        """
        try:
            number = self.cast_to(number)
        except TypeError:
            # Not a suitable value.
            return None
        return number

    def __call__(self, field, **kwargs):
        if 'value' not in kwargs:
            kwargs['value'] = field._value()

        # The value must be cast to a data type than works with "%02d" first.
        safe_value = self.cast(kwargs['value'])
        if safe_value:
            kwargs['value'] = "%02d" % safe_value

        return super(LeadingZeroNumberWidget, self).__call__(field, **kwargs)



class LeadingZeroFloatWidget(LeadingZeroNumberWidget):
    cast_to = float


class LeadingZeroIntegerWidget(LeadingZeroNumberWidget):
    cast_to = int


MOVER_CONSTANT_WIND = 'constant_wind'
MOVER_VARIABLE_WIND = 'variable_wind'


class AddMoverForm(Form):
    """
    The initial form used in a multi-step process for adding a mover to the
    user's running model. This step asks the user to choose the type of mover
    to add.
    """
    mover_type = SelectField('Type', choices=(
        (MOVER_CONSTANT_WIND, 'Winds - Constant'),
        (MOVER_VARIABLE_WIND, 'Winds - Variable')
    ))


class WindMoverForm(Form):
    """
    A form base class containing fields common to `VariableWindMoverForm` and
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


class DateTimeForm(Form):
    """
    A form base class that has a `date` field and two fields to choose hour
    and minute. Taken together, the values can be passed into a
    `datetime.datetime` constructor.
    """
    date = DateTimeField('Date', widget=DatePickerWidget(),
                     format="%m/%d/%Y",
                     validators=[Required()],
                     default=datetime.date.today())
    hour = IntegerField(widget=LeadingZeroIntegerWidget(),
                        validators=[NumberRange(min=0, max=24)],
                        default=lambda: datetime.datetime.now().hour)
    minute = IntegerField(widget=LeadingZeroIntegerWidget(),
                          validators=[NumberRange(min=0, max=60)],
                          default=lambda: datetime.datetime.now().minute)

    def get_datetime(self):
        """
        Return a `datetime.datetime` value set to time `self.hour`:`self.minute`
        if the user specified those values, else 00:00.
        """
        date = self.date.data

        return datetime.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=self.hour.data,
            minute=self.minute.data)


class ConstantWindMoverForm(WindMoverForm):
    """
    A form for adding a constant wind mover to the user's running model.
    """
    type = HiddenField(default=MOVER_CONSTANT_WIND)


class VariableWindMoverForm(WindMoverForm, DateTimeForm):
    """
    A form for adding a variable wind mover to the user's running model.
    """
    type = HiddenField(default=MOVER_VARIABLE_WIND)
    auto_increment_time_by = IntegerField('Auto-increment time by')


class RunModelUntilForm(DateTimeForm):
    """
    A form for submitting a step that the user wishes to run his or her model
    until.
    """
    pass


class ModelSettingsForm(DateTimeForm):
    """
    A form for adding and editing model-related settings.
    """
    duration_days = IntegerField(default=1, validators=[NumberRange(min=0)])
    duration_hours = IntegerField(default=0, validators=[NumberRange(min=0)])
    include_minimum_regret = BooleanField(
        label="Include the Minimum Regret (Uncertainty) Solution",
        default=False)
    show_currents = BooleanField(label="Show Currents", default=False)
    computation_time_step = FloatField("Computation Time Step:", default=0.1)
    prevent_land_jumping = BooleanField("Prevent Land Jumping", default=True)
    run_backwards = BooleanField("Run Backwards", default=False)
