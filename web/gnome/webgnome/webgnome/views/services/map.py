import os
import gnome.map
import gnome.utilities.map_canvas

from cornice.resource import resource, view
from hazpy.file_tools import haz_files

from webgnome import util
from webgnome.schema import MapSchema
from webgnome.model_manager import WebMapFromBNA
from webgnome.views.services.base import BaseResource


@resource(path='/model/{model_id:\d+}/map',
          renderer='gnome_json', description='A map.')
class Map(BaseResource):

    @view(validators=util.valid_map)
    def get(self):
        """
        Return a JSON representation of WindMover matching the ``id`` matchdict
        value.
        """
        model = self.request.validated.pop('model')
        map = model.map

        return {
            'filename': map.filename,
            'name': map.name,
            'refloat_halflife': map.refloat_halflife
        }

    @view(validators=util.valid_model_id, schema=MapSchema)
    def post(self):
        """
        Add a map to the current model.
        """
        model = self.request.validated.pop('model')
        map_file = os.path.join(
            self.settings['project_root'],
            'webgnome', 'data', self.validated['filename'])

        # Create the land-water map
        model.map = WebMapFromBNA(
            map_file, name=self.validated['name'],
            refloat_halflife=self.validated['refloat_halflife'])

        # TODO: Should size be user-configurable?
        canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
        polygons = haz_files.ReadBNA(map_file, "PolygonSet")
        canvas.set_land(polygons)
        model.output_map = canvas

        return {
            'filename': self.validated['filename'],
            'name': self.validated['name'],
            'refloat_halflife': self.validated['refloat_halflife']
        }
