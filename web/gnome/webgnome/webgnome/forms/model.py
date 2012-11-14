from wtforms import (
    IntegerField,
    FloatField,
    BooleanField,
)

from wtforms.validators import NumberRange

from base import AutoIdForm, DateTimeForm


class RunModelUntilForm(DateTimeForm):
    """
    A form for submitting a step that the user wishes to run his or her model
    until.

    TODO: This form should validate that the given date and time value are
    a valid time step in the model.
    """
    pass


class ModelSettingsForm(AutoIdForm, DateTimeForm):
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