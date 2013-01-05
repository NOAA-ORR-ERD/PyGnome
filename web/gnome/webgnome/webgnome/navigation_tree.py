class NavigationTree(object):
    """
    A class that renders a JSON representation of a ``gnome.model.Model``
    used to initialize a navigation tree widget in the JavaScript app.
    """
    def __init__(self, model):
        self.model = model

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
        data = self.model.to_dict()

        movers = {
            'title': 'Movers',
            'form_id': 'add_mover',
            'children': []
        }

        for mover in data.pop('wind_movers', []):
            movers['children'].append({
                'object_id': mover['id'],
                'form_id': 'edit_wind_mover',
                'title': mover['name']
            })

        spills = {
            'title': 'Spills',
            'form_id': 'add_spill',
            'children': []
        }

        for spill in data.pop('point_release_spills', []):
            spills['children'].append({
                'object_id': spill['id'],
                'form_id': 'edit_point_release_spill',
                'title': spill['name']
            })

        settings = {
            'title': 'Model Settings',
            'form_id': 'model_settings',
            'children': []
        }

        for name, value in data.items():
            settings['children'].append({
                'form_id': 'model_settings',
                'title': self._get_value_title(name, value),
            })

        return [settings, movers, spills]
