import requests

from bs4 import BeautifulSoup
from cornice.resource import resource, view
from pyramid.httpexceptions import HTTPServerError
from tempfile import NamedTemporaryFile

from webgnome import util
from webgnome.schema import MapSchema, CustomMapSchema
from webgnome.views.services.base import BaseResource


@resource(path='/model/{model_id}/map',
          renderer='gnome_json', description='A map.')
class Map(BaseResource):

    @view(validators=util.valid_map)
    def get(self):
        """
        Return a JSON representation of the current map.
        """
        model = self.request.validated.pop('model')
        return model.map.to_dict()

    @view(validators=util.map_filename_exists, schema=MapSchema)
    def post(self):
        """
        Add a map to the current model.
        """
        data = self.request.validated
        model = data.pop('model')
        # Ignore the map bounds on setting -- this is readonly.
        data.pop('map_bounds')
        filename = data.pop('filename')
        model.add_bna_map(filename, data)
        return model.map.to_dict()

    @view(validators=util.valid_model_id)
    def delete(self):
        self.request.validated['model'].remove_map()


@resource(path='/model/{model_id}/custom_map',
          renderer='gnome_json', description='A custom map from GOODS data.')
class CustomMap(BaseResource):

    def get_form_errors(self, content):
        if content.find('<html>') == -1:
            return

        soup = BeautifulSoup(content)
        errors = soup.find_all('span', class_='error-message')

        if errors:
            for e in errors:
                self.request.errors.add('body', 'map', e.text)
                self.request.errors.status = 500

        return errors

    @view(validators=util.valid_map)
    def get(self):
        """
        Return a JSON representation of the current map.
        """
        model = self.request.validated.pop('model')
        return model.map.to_dict()

    @view(validators=util.valid_model_id, schema=CustomMapSchema)
    def post(self):
        """
        Get a custom map file from GOODS for the given coordinates and add it
        to the current model.
        """
        data = self.request.validated
        url = self.settings['goods.custom_map_url']
        model = data.pop('model')
        goods_data = {
            'NorthLat': data.pop('north_lat'),
            'WestLon': data.pop('west_lon'),
            'EastLon': data.pop('east_lon'),
            'SouthLat': data.pop('south_lat'),
            'resolution': data.pop('resolution'),
        }

        cross_dateline = data.pop('cross_dateline')
        if cross_dateline:
            goods_data['xDateline'] = True

        resp = requests.post(url, goods_data)

        if not resp.status_code == 200:
            raise HTTPServerError('Could not contact GOODS.')

        self.get_form_errors(resp.content)

        if self.request.errors:
            return

        f = NamedTemporaryFile(dir=model.base_dir, delete=False)
        f.write(resp.content)
        f.close()
        model.add_bna_map(f.name, data)

        return model.map.to_dict()

