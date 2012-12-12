import gnome.spill

from wtforms import IntegerField, FloatField, BooleanField, SelectField, StringField
from wtforms import ValidationError
from wtforms.validators import Optional, Required

from webgnome.model_manager import WebPointReleaseSpill

from base import AutoIdForm, DateTimeForm


class PointReleaseSpillForm(DateTimeForm, AutoIdForm):
    """
    A form wrapping gnome.spill.PointReleaseSpill.
    """
    start_position_x = FloatField()
    start_position_y = FloatField()
    start_position_z = FloatField(validators=[Optional()])
    windage_min = FloatField(default=0.01)
    windage_max = FloatField(default=0.04)
    windage_persist = FloatField(default=900)
    is_uncertain = BooleanField(default=False)
    is_active = BooleanField(default=True)
    name = StringField(default='Point Release Spill', validators=[Required()])

    def get_start_position(self):
        return (self.start_position_x.data, self.start_position_y.data,
                self.start_position_z.data)

    def get_windage_range(self):
        return self.windage_min.data, self.windage_max.data

    def create(self):
        """
        Create a new :class:`WebWindMover` using data from this form.
        """
        return WebPointReleaseSpill(
            is_active=self.is_active.data,
            name=self.name.data,
            start_position=self.get_start_position(),
            windage_range=self.get_windage_range(),
            windage_persist=self.windage_persist.data,
            is_uncertain=self.is_uncertain.data
        )

    def update(self, spill):
        """
        Update ``spill`` using data from this form.
        """
        spill._name = self.name.data
        spill.is_active = self.is_active.data
        spill.start_position = self.get_start_position()
        spill.windage_range = self.get_windage_range()
        spill.windage_persist = self.windage_persist.data
        spill.is_uncertain = self.is_uncertain.data

        return spill


class AddSpillForm(AutoIdForm):
    """
    The initial form used in a multi-step process for adding a spill to the
    user's running model. This step asks the user to choose the type of spill
    to add.
    """
    mover_type = SelectField(
        'Type',
        choices=(
            (PointReleaseSpillForm.get_id(), 'Point Release Spill'),
        )
    )


class DeleteSpillForm(AutoIdForm):
    """
    Delete spill with ``spill_id``. Validates that a spill with that ID exists
    in ``self.model``.

    This is a hidden form submitted via AJAX by the JavaScript client.
    """
    spill_id = IntegerField()

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model', None)
        super(DeleteSpillForm, self).__init__(*args, **kwargs)

    def spill_id_validate(self, field):
        spill_id = field.data

        if spill_id is None or self.model.has_spill_with_id(spill_id) is False:
            raise ValidationError('Spill with that ID does not exist')


spill_form_classes = {
    WebPointReleaseSpill: PointReleaseSpillForm
}

