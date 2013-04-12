class NavigationTree(object):
    """
    A class that renders a JSON representation of a ``gnome.model.Model``
    used to initialize a navigation tree widget in the JavaScript app.
    """
    def __init__(self, model):
        self.model = model

    def _get_value_title(self, name, value, max_chars=200):
        """
        Return a title string that combines ``name`` and ``value``, with value
        shortened if it is longer than ``max_chars``.
        """
        name = name.replace('_', ' ').title()
        value = (str(value)).title()
        value = value if len(value) <= max_chars else '%s ...' % value[:max_chars]
        return '%s: %s' % (name, value)

    def _render_children(self, nodes, form_id):
        """
        Render a list of dictionaries ``nodes`` as child nodes of a root node.

        Returns a list dictionaries each of which has a 'form_id', 'title', and
        'children' field, the last of which will include any key, value pairs
        in the dict.

        If a 'name' field exists in a node, it is used as the title of the
        rendered child.

        If an 'id' field exists in a node, it is used as the 'object_id' field
        of the rendered child.
        """
        children = []
        for node in nodes:
            node_id = node.pop('id') if 'id' in node else None

            item = {
                'form_id': form_id,
                'title': node.pop('name', 'Item'),
                'children': []
            }

            if node_id:
                item['object_id'] = node_id

            for name, value in node.items():
                if name == 'id' or name == 'obj_type':
                    continue

                sub_item = {
                    'form_id': form_id,
                    'title': self._get_value_title(name, value)
                }

                if node_id:
                    sub_item['object_id'] = node_id

                item['children'].append(sub_item)

            children.append(item)
        return children

    def _render_root_node(self, title, form_id):
        return {
            'title': title,
            'form_id': form_id,
            'children': []
        }

    def render(self):
        data = self.model.to_dict()
        environment = self._render_root_node('Environment', 'add-environment')
        movers = self._render_root_node('Movers', 'add-mover')
        spills = self._render_root_node('Spills', 'add-spill')
        settings = self._render_root_node('Model Settings', 'model-settings')

        environment['children'].extend(
            self._render_children(data.pop('winds', []), form_id='edit-wind'))

        movers['children'].extend(
            self._render_children(data.pop('wind_movers', []),
                                  form_id='edit-wind-mover'))

        movers['children'].extend(
            self._render_children(data.pop('random_movers', []),
                                  form_id='edit-random-mover'))

        movers['children'].extend(
            self._render_children(data.pop('cats_movers', []),
                                  form_id='edit-cats-mover'))

        spills['children'].extend(
            self._render_children(data.pop('surface_release_spills', []),
                                  form_id='edit-surface-release-spill'))

         # Add the map manually as the first model setting
        map_data = data.pop('map', None)
        map_form_id = 'edit-map' if map_data else 'add-map'

        settings['children'].append({
            'form_id': map_form_id,
            'object_id': map_data['id'] if map_data else None,
            'title': 'Map: %s' % (map_data['name'] if map_data else 'None')
        })

        model_items = [
            dict(name=self._get_value_title(key, value), id=self.model.id)
            for key, value in data.items() if key != 'id']

        settings['children'].extend(
            self._render_children(model_items, form_id='model-settings'))

        return [settings, environment, movers, spills]
