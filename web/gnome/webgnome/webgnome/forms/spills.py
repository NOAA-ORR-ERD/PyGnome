import gnome.spill

from wtforms import IntegerField
from wtforms import ValidationError

from base import AutoIdForm
from object_form import ObjectForm


class SpillForm(ObjectForm):
    """
    A form wrapping gnome.spill.PointReleaseSpill.

    TODO: What fields do we need?
    """
    wrapped_class = gnome.spill.PointReleaseSpill


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

