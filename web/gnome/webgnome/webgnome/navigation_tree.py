from collections import OrderedDict

from webgnome.form_view import FormView


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

    def _get_route_name_for_obj(self, obj):
        """
        Return a form view route for ``obj``.
        """
        return FormView.get_route_for_object(obj)

    def render(self):
        """
        Return an ordered list of tree elements for ``self.model``, suitable
        for JSON serialization.

        Nodes are given a ``form_id`` value that corresponds to two things:

            - In the client, ``form_id`` represents an HTLM form
            - On the server, ``form_id`` corresponds to a route name for the
              view that handles GET and POST requests for the form

        This value is used by the client to detect which form is appropriate to
        display for user actions on an item in the tree.

        The reason that ``form_id`` also corresponds to a route name, and not
        just a form ID, is slightly dubious from a design perspective. This is
        so the JavaScript app can create URLs for delete requests. Constructing
        delete URLs is based on the convention that for a given node, the delete
        URL will be a string of the format:

            "/``node.data.form_id``.``node.data.id``"

        Where ``node.data.id`` is the ID of the object to delete.
        """
        settings = {
            'title': 'Model Settings',
            'form_id': 'model_settings',
            'children': []
        }

        movers = {
            'title': 'Movers',
            'form_id': 'add_mover',
            'children': []
        }

        spills = {
            'title': 'Spills',
            'form_id': 'add_spill',
            'children': []
        }

        for name, value in self._get_model_settings().items():
            settings['children'].append({
                'form_id': 'model_settings',
                'title': self._get_value_title(name, value),
            })

        # TODO: Handle a real map.
        settings['children'].append({
            'form_id': 'model_map',
            'title': 'Map: None'
        })

        for mover in self.model.movers:
            movers['children'].append({
                'form_id': self._get_route_name_for_obj(mover),
                'id': mover.id,
                'title': str(mover)
            })

        for spill in self.model.spills:
            spills['children'].append({
                'form_id': self._get_route_name_for_obj(spill),
                'id': spill.id,
                'title': str(spill),
            })

        return [settings, movers, spills]
