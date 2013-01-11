from cornice.resource import resource, view

from webgnome import util
from webgnome.model_manager import WebPointReleaseSpill
from webgnome.schema import PointReleaseSpillSchema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id:\d+}/spill/point_release',
          path='/model/{model_id:\d+}/spill/point_release/{id:\d+}',
          renderer='gnome_json', description='A point release spill.')
class PointReleaseSpill(BaseResource):

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing PointReleaseSpills.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_spills=True)

        return model_data['point_release_spills']

    @view(validators=util.valid_model_id, schema=PointReleaseSpillSchema)
    def collection_post(self):
        """
        Create a PointReleaseSpill from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        spill = WebPointReleaseSpill(**data)
        model.spills.add(spill)

        return {
            'success': True,
            'id': spill.id
        }

    @view(validators=util.valid_spill_id)
    def get(self):
        """
        Return a JSON representation of the PointReleaseSpill matching the
        ``id`` matchdict value.
        """
        spill = self.request.validated['model'].spills.get(self.id)
        return spill.to_dict()

    @view(validators=util.valid_spill_id, schema=PointReleaseSpillSchema)
    def put(self):
        """
        Update an existing PointReleaseSpill from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        spill = model.spills.get(self.id)
        spill.from_dict(data)

        return {
            'success': True,
            'id': spill.id
        }

    @view(validators=util.valid_spill_id)
    def delete(self):
        """
        Delete a PointReleaseSpill.
        """
        self.request.validated['model'].spills.remove(self.id)
        message = util.make_message('success', 'Deleted point release spill.')

        return {
            'success': True,
            'mover_id': self.id,
            'message': message
        }

