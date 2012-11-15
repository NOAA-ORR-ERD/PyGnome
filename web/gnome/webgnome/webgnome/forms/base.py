import datetime

from wtforms.widgets import TextInput
from wtforms.validators import (
    Required,
    NumberRange
)
from wtforms import (
    Form,
    DateTimeField,
    IntegerField
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


class DateTimeForm(Form):
    """
    A form base class that has a `date` field and two fields to choose hour
    and minute. Taken together, the values can be passed into a
    `datetime.datetime` constructor.
    """
    DATE_FORMAT = "%m/%d/%Y"

    date = DateTimeField('Date', widget=DatePickerWidget(),
                     format=DATE_FORMAT,
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


class AutoIdForm(Form):
    """
    A form that can generate an ID for itself.

    The form makes an ``id`` property available that can be used within
    templates to automatically generate a form ID for the form instance.

    The class method ``get_id`` can be used outside of a template context to
    look up a form ID given an ``obj``.
    """
    def __init__(self, *args, **kwargs):
        """
        If an object instance is passed to the form, save a reference to it.
        """
        self.instance = kwargs.get('obj', None)
        super(AutoIdForm, self).__init__(*args, **kwargs)

    @property
    def id(self):
        """
        Return an HTML ID for this form using ``self.instance``. This property
        is intended as a helper when using form instances within templates.
        """
        return self.get_id(self.instance)

    @classmethod
    def get_id(cls, obj=None):
        """
        Get an ID for this form that combines the form's class name and ``obj``.
        """
        object_id = ''

        if obj:
            if hasattr(obj, 'id') and obj.id:
                object_id = obj.id
            else:
                object_id = id(obj)

            object_id = '_%s' % object_id

        return '%s%s' % (cls.__name__, object_id)
