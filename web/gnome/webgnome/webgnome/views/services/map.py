import os
import gnome.map
import gnome.utilities.map_canvas

from cornice.resource import resource, view
from hazpy.file_tools import haz_files

from webgnome import util
from webgnome.schema import MapSchema
from webgnome.model_manager import WebMapFromBNA
from webgnome.views.services.base import BaseResource


@resource(path='/model/{model_id}/map',
          renderer='gnome_json', description='A map.')
class Map(BaseResource):

    @view(validators=util.valid_map)
    def get(self):
        """
        Return a JSON representation of WindMover matching the ``id`` matchdict
        value.
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
        filename = data.pop('filename')
        model.add_bna_map(filename, data)
        return model.map.to_dict()
