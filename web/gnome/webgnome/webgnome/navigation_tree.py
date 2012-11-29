from collections import OrderedDict

from webgnome.forms.model import ModelSettingsForm
from webgnome.forms.movers import AddMoverForm, DeleteMoverForm
from webgnome.forms.object_form import get_object_form_cls
from webgnome.forms.spills import DeleteSpillForm


class NavigationTree(object):
    """
    An class that renders a JSON representation of a ``gnome.model.Model``
    used to initialize a navigation tree widget in the JavaScript app.
    """
    def __init__(self, request, model):
        self.request = request
        self.model = model

    def _get_model_settings(self):
        """
        Return a dict of values containing each model setting that the client
        should be able to read and change.
        """
        settings_attrs = [
            'start_time',
            'duration'
        ]

        settings = OrderedDict()

        for attr in settings_attrs:
            if hasattr(self.model, attr):
                settings[attr] = getattr(self.model, attr)

        return settings

    def _get_value_title(self, name, value, max_chars=8):
        """
        Return a title string that combines ``name`` and ``value``, with value
        shortened if it is longer than ``max_chars``.
        """
        name = name.replace('_', ' ').title()
        value = (str(value)).title()
        value = value if len(value) <= max_chars else '%s ...' % value[:max_chars]
        return '%s: %s' % (name, value)

    def render(self):
        """
        Return an ordered list of tree elements for ``self.model``, suitable
        for JSON serialization.

        Nodes are given a ``form_id`` value that points to a form rendered in
        the client. The client uses this value to display a form for the item
        when appropriate (i.e., when the user clicks on an "Add" or "Edit"
        button).
        """
        settings = {
            'title': 'Model Settings',
            'key': ModelSettingsForm.get_id(),
            'form_id': ModelSettingsForm.get_id(),
            'children': []
        }

        movers = {
            'title': 'Movers',
            'key': AddMoverForm.get_id(),
            'form_id': AddMoverForm.get_id(),
            'children': []
        }

        # XXX: Hard-coded form ID. FormView class does not exist yet.
        spills = {
            'title': 'Spills',
            'key': 'add_spill',
            'form_id': 'add_spill',
            'children': []
        }

        for name, value in self._get_model_settings().items():
            settings['children'].append({
                # All settings use the model update form.
                'key': ModelSettingsForm.get_id(),
                'form_id': ModelSettingsForm.get_id(),
                'title': self._get_value_title(name, value),
            })

        # XXX: Hard-coded form ID. FormView class does not exist yet.
        settings['children'].append({
            'key': 'model_map',
            'form_id': 'model_map',
            'title': 'Map: None'
        })

        for mover in self.model.movers:
            form = get_object_form_cls(mover)

            if not form:
                continue

            _id = form.get_id(mover)

            movers['children'].append({
                'key': _id,
                'form_id': _id,
                'delete_form_id': DeleteMoverForm.get_id(mover),
                'object_id': mover.id,
                'title': str(mover)
            })

        for spill in self.model.spills:
            spills['children'].append({
                'key': get_object_form_cls(spill).get_id(spill),
                'form_id': get_object_form_cls(spill).get_id(spill),
                'delete_form_id': DeleteSpillForm.get_id(spill),
                'object_id': spill.id,
                'title': str(spill),
            })

        return [settings, movers, spills]
