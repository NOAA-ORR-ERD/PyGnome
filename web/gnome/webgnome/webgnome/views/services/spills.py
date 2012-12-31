from cornice.resource import resource, view

from webgnome import util
from webgnome.model_manager import WebPointReleaseSpill
from webgnome.schema import PointReleaseSpillSchema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/spill/point_release',
          path='/spill/point_release/{id:\d+}',
          renderer='gnome_json', description='A point release spill.')
class PointRelease(BaseResource):

    @property
    def data(self):
        return self.request.validated

    @view(validators=util.valid_model_id, schema=PointReleaseSpillSchema)
    def collection_post(self):
        """
        Create a PointReleaseSpill from a JSON representation.
        """
        spill = WebPointReleaseSpill(**self.data)
        self.model.add_spill(spill)

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
        spill = self.model.get_spill(self.id)
        return spill.to_dict()

    @view(validators=util.valid_spill_id, schema=PointReleaseSpillSchema)
    def put(self):
        """
        Update an existing PointReleaseSpill from a JSON representation.
        """
        spill = self.model.get_spill(self.id)
        spill.from_dict(self.data)

        return {
            'success': True,
            'id': spill.id
        }

    @view(validators=util.valid_spill_id)
    def delete(self):
        """
        Delete a PointReleaseSpill.
        """
        self.model.remove_spill(self.id)
        message = util.make_message('success', 'Deleted point release spill.')

        return {
            'success': True,
            'mover_id': self.id,
            'message': message
        }

