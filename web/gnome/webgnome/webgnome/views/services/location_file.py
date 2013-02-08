import json
import os

from cornice.resource import resource, view
from webgnome.schema import ModelSchema
from webgnome.views.services.base import BaseResource
from webgnome import util


@resource(path='/model/{model_id}/location_file/{location}',
          renderer='gnome_json',
          description='Add a location file to the current model')
class LocationFile(BaseResource):

    @view(validators=util.valid_location_file)
    def post(self):
        data = self.request.validated
        model = data['model']
        raw_data = open(
            os.path.join(data['location_dir'], 'location.json')).read()
        location_data = json.loads(raw_data)
        validated = ModelSchema().bind().deserialize(location_data)
        _map = validated.get('map', None)

        if _map and _map['filename']:
            _map['filename'] = '%s/%s' % (
                data['location_dir'], _map['filename'])

        model.from_dict(validated)

        return model.to_dict()
