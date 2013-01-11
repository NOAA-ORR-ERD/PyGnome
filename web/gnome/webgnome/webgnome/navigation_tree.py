class NavigationTree(object):
    """
    A class that renders a JSON representation of a ``gnome.model.Model``
    used to initialize a navigation tree widget in the JavaScript app.
    """
    def __init__(self, model):
        self.model = model

    def _get_value_title(self, name, value, max_chars=50):
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
            mover_item = {
                'object_id': mover['id'],
                'form_id': 'edit_wind_mover',
                'title': mover['name'],
                 'children': []
            }

            for name, value in mover.items():
                mover_item['children'].append({
                    'object_id': mover['id'],
                    'form_id': 'edit_wind_mover',
                    'title': self._get_value_title(name, value)
                })

            movers['children'].append(mover_item)

        spills = {
            'title': 'Spills',
            'form_id': 'add_spill',
            'children': []
        }

        for spill in data.pop('point_release_spills', []):
            spill_item = {
                'object_id': spill['id'],
                'form_id': 'edit_point_release_spill',
                'title': spill['name'],
                'children': []
            }

            for name, value in spill.items():
                spill_item['children'].append({
                    'object_id': spill['id'],
                    'form_id': 'edit_point_release_spill',
                    'title': self._get_value_title(name, value)
                })

            spills['children'].append(spill_item)

        settings = {
            'title': 'Model Settings',
            'form_id': 'model_settings',
            'children': []
        }

        map_data = data.pop('map')
        map_form_id = 'edit_map' if map_data else 'add_map'


        settings['children'].append({
            'form_id': map_form_id,
            'object_id': map_data['id'] if map_data else None,
            'title': 'Map: %s' % (map_data['name'] if map_data else 'None')
        })

        for name, value in data.items():
            settings['children'].append({
                'form_id': 'model_settings',
                'title': self._get_value_title(name, value),
            })


        return [settings, movers, spills]
