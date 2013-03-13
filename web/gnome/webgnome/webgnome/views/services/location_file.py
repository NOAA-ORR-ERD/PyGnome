import json
import os

from cornice.resource import resource, view
from webgnome.views.services.base import BaseResource
from webgnome import util, schema


@resource(path='/model/{model_id}/location_file/{location}',
          renderer='gnome_json',
          description='Get configuration for a location file in JSON format. '
                      'Post to create a new location file from model JSON.')
class LocationFile(BaseResource):

    @view(validators=util.valid_location_file)
    def get(self):
        model_schema = schema.ModelSchema().bind()
        data = model_schema.deserialize(
            self.request.validated['location_file_model_data'])
        return model_schema.serialize(data)

    @view(validators=[util.valid_model_id, util.valid_new_location_file],
          schema=schema.LocationFileSchema)
    def post(self):
        data = self.request.validated
        location_dir = data.pop('location_dir')
        model = data.pop('model')
        model_data = data.pop('model_data')
        _map = model_data.get('map', None)

        # Create a directory skeleton for the location file.
        util.create_location_file(location_dir, **data)

        # If a map was specified, move it into the new location file directory.
        if _map and _map['filename']:
            filename = _map['filename']
            old = os.path.join(model.base_dir, filename)
            new = os.path.join(data['location_dir'], filename)
            os.rename(old, new)

        with open(os.path.join(location_dir, 'location.json')) as f:
            f.write(json.dumps(model_data, default=util.json_encoder))

        return {
            'location': data['name'],
        }


@resource(path='/model/{model_id}/location_file/{location}/wizard',
          renderer='gnome_json',
          description='Get location file wizard HTML. Post user options for '
                      'the wizard to apply them to the model.')
class LocationFileWizard(BaseResource):
    @view(validators=[util.valid_location_file,
                      util.valid_location_file_wizard])
    def get(self):
        return {
            'html': self.request.validated['wizard_html']
        }

    @view(validators=[util.valid_model_id, util.valid_location_file,
                      util.valid_location_file_wizard])
    def put(self):
        self.request.validated['wizard_handler'](
            self.request.validated['model'], self.request.json_body)

        return {
            'location': self.request.matchdict['location']
        }
