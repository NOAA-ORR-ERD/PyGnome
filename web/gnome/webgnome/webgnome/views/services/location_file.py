import json
import os

from cornice.resource import resource, view
from pyramid.renderers import render
from webgnome.schema import ModelSchema
from webgnome.views.services.base import BaseResource
from webgnome import util


@resource(path='/model/{model_id}/location_file/{location}',
          renderer='gnome_json',
          description='Add a location file to the current model')
class LocationFile(BaseResource):

    @view(validators=[util.valid_model_id, util.valid_location_file])
    def post(self):
        data = self.request.validated
        model = data['model']
        raw_data = open(data['location_file']).read()
        location_data = json.loads(raw_data)
        validated = ModelSchema().bind().deserialize(location_data)
        _map = validated.get('map', None)

        if _map and _map['filename']:
            _map['filename'] = os.path.join(data['location_dir'],
                                            _map['filename'])

        model.from_dict(validated)

        location_handlers = self.settings.get('location_handlers', {})
        handler = location_handlers.get(
            self.request.matchdict['location'], None)

        if handler and hasattr(handler, '__call__'):
            handler(data)

        return model.to_dict()


@resource(path='/location_file/{location}/wizard',
          renderer='gnome_json',
          description='Get HTML for a location file wizard.')
class LocationFileWizard(BaseResource):

    @view(validators=util.valid_location_file)
    def get(self):
        location = self.request.matchdict['location']
        html = render('webgnome:location_files/%s/wizard.mak' % location, {})

        return {
            'html': html
        }
