"""Shared validators for FormEncode.

FormEncode (http://python.org/pypi/formencode) is the default form validator in
Pylons, and is widely used outside it; e.g., as a validator for ToscaWidgets.
"""

import formencode as fe
import formencode.validators as v
_ = v._

class Boolean(v.StringBoolean):
    """A validator for checkbox fields."""

    if_missing = False


class Int(v.Int):
    """An integer validator with a more meaningful message."""

    messages = {
        "integer": _("Please enter a numeric value"),
        }


class SelectInt(v.FancyValidator):
    """A combination of the Int and OneOf validators with a custom message.
    
    Usage:
        myselect = SelectInt(range(1, 5))  # Arg is a list of ints.
        # Legal Python values: 1, 2, 3, 4
        # Legal HTML values: "1", "2", "3", "4"

    Useful for select lists where the values are numbers.
    """

    __unpackargs__ = ("list",)
    not_empty = True

    def _to_python(self, value, state):
        try:
            return int(value)
        except ValueError:
            self._invalid(value, state)

    _from_python = _to_python

    def validate_python(self, value, state):
        if value not in self.list:
            self._invalid(value, state)

    def _invalid(self, value, state):
        message = "please choose an item from the list"
        raise v.Invalid(message, value, state)


class HasCriteria(v.FancyValidator):
    """A validator for multi-field search forms.
    Ensure that at least one of the search fields is filled in.

    Usage in a FormEncode schema:
        chained_validators = [HasCriteria(["field1", "field2"])]

    Attributes:

    ``fields`` (first positional arg): a list of regular search fields;
    i.e., not falling into any of the specialized categories below.

    ``fields_with_zero_allowed`` (second positional arg): a list of 
    search fields where numeric 0 is a valid search term.

    The validator ensures that at least one of the fields has a 
    non-empty value (i.e., not "", 0, None, False, [], etc).  However,
    0 is allowed in fields listed in ``fields_with_zero_allowed``.
    Missing keys are treated as empty values.

    The error message will be placed in a key "form".  The form should
    contain an '<if:error name="form" />' placeholder to display the message.

    This class assumes each field is independent; i.e., it may be filled in
    regardless of the other fields. If there are dependencies between fields
    (i.e., if multiple fields together must have a certain combination of
    values), these can be checked in a subclass.
    """
    __unpackargs__ = ["fields", "fields_with_zero_allowed"]

    def validate_python(self, value, state):
        for field in self.fields:
            val = value.get(field)
            if val:
                return
        for field in self.fields_with_zero_allowed:
            val = value.get(field)
            if val or val == 0:
                return
        errmsg = "You must fill in at least one search criterion."
        raise fe.Invalid({"form": errmsg}, value, state)
