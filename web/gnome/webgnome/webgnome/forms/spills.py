import gnome.spill

from wtforms import IntegerField, FloatField, BooleanField
from wtforms import ValidationError
from wtforms.validators import Optional

from webgnome.model_manager import WebPointReleaseSpill

from base import AutoIdForm, DateTimeForm


class PointReleaseSpillForm(DateTimeForm, AutoIdForm):
    """
    A form wrapping gnome.spill.PointReleaseSpill.
    """
    start_position_x = FloatField()
    start_position_y = FloatField()
    start_position_z = FloatField(validators=Optional())
    windage_min = FloatField(default=0.01)
    windage_max = FloatField(default=0.04)
    windage_persist = FloatField(default=900)
    uncertain = BooleanField(default=False)
    is_active = BooleanField(default=True)


    def get_start_position(self):
        return (self.start_position_x, self.start_position_y,
                self.start_position_z)


class DeleteSpillForm(AutoIdForm):
    """
    Delete spill with ``spill_id``. Validates that a spill with that ID exists
    in ``self.model``.

    This is a hidden form submitted via AJAX by the JavaScript client.
    """
    spill_id = IntegerField()

    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(DeleteSpillForm, self).__init__(*args, **kwargs)

    def spill_id_validate(self, field):
        spill_id = field.data

        if spill_id is None or self.model.has_spill_with_id(spill_id) is False:
            raise ValidationError('Spill with that ID does not exist')


spill_form_classes = {
    WebPointReleaseSpill: PointReleaseSpillForm
}

