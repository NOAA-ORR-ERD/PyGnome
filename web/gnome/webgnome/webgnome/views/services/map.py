import logging
import os
import requests

from bs4 import BeautifulSoup
from cornice.resource import resource, view
from pyramid.httpexceptions import HTTPServerError
from tempfile import NamedTemporaryFile

from webgnome import util
from webgnome.schema import MapSchema, CustomMapSchema
from webgnome.views.services.base import BaseResource


logger = logging.getLogger(__name__)


class MapResource(BaseResource):
    """
    A base class for Map services.
    """
    def get_map_data(self, model):
        """
        A helper that returns a dict of data for the model's current map, with
        the background image changed into a URL so the client can use it to
        request the background image.
        """
        map_data = model.map.to_dict()

        if model.background_image:
            map_data['background_image_url'] = util.get_model_image_url(
                self.request, model, model.background_image)

        return map_data


@resource(path='/model/{model_id}/map', renderer='gnome_json',
          description="The user's current map.")
class Map(MapResource):

    @view(validators=util.valid_map)
    def get(self):
        """
        Return a JSON representation of the current map.
        """
        model = self.request.validated.pop('model')
        return self.get_map_data(model)

    @view(validators=util.valid_uploaded_file, schema=MapSchema)
    def post(self):
        """
        Add a map to the current model.

        The 'filename' field must refer to a BNA file that already exists on
        the server. The file will be *deleted* after it is used to construct
        a map.
        """
        data = self.request.validated
        model = data.pop('model')
        # Ignore the map bounds on setting -- this is readonly.
        data.pop('map_bounds')
        filename = data.pop('filename')
        model.add_bna_map(filename, data)
        try:
            os.remove(filename)
        except OSError as e:
            logger.error('Could not remove file: %s. Error was: %s' % (
                filename, e))
        return self.get_map_data(model)

    @view(validators=util.valid_map, schema=MapSchema)
    def put(self):
        """
        Update an existing map.
        """
        data = self.request.validated
        model = data.pop('model')

        # Ignore readonly values.
        data.pop('map_bounds')
        data.pop('filename')

        model.map.from_dict(data)
        return model.map.to_dict()

    @view(validators=util.valid_model_id)
    def delete(self):
        self.request.validated['model'].remove_map()


@resource(path='/model/{model_id}/custom_map', renderer='gnome_json',
          description='A custom map created from GOODS data.')
class CustomGoodsMap(BaseResource):

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
        return self.get_map(model)

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


@resource(path='/model/{model_id}/file_upload', renderer='gnome_json',
          description='A file upload.')
class FileUpload(BaseResource):

    @view(validators=util.valid_model_id)
    def post(self):
        """
        Upload a file.
        """
        input_file = self.request.POST['filename'].file
        temp_file = NamedTemporaryFile(dir=self.settings.upload_dir,
                                       delete=False)
        try:
            temp_file.write(input_file.read())
            temp_file.close()
        except OSError as e:
            logger.error('Could not write file: %s. Error was: %s' % (
                temp_file.name, e))
            raise HTTPServerError
        else:
            return {
                'filename': temp_file.name.split(os.path.sep)[-1]
            }
