from cornice.resource import resource, view

from webgnome import util
from webgnome.model_manager import WebSurfaceReleaseSpill
from webgnome import schema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id}/spill/surface_release',
          path='/model/{model_id}/spill/surface_release/{id}',
          renderer='gnome_json', description='A surface release spill.')
class SurfaceReleaseSpill(BaseResource):

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing SurfaceReleaseSpills.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_spills=True)

        return schema.SurfaceReleaseSpillsSchema(
            model_data['surface_release_spills'])

    @view(validators=util.valid_model_id,
          schema=schema.SurfaceReleaseSpillSchema)
    def collection_post(self):
        """
        Create a SurfaceReleaseSpill from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        spill = WebSurfaceReleaseSpill(**data)
        model.spills.add(spill)

        return schema.SurfaceReleaseSpillSchema().bind().serialize(
            spill.to_dict(do='create'))

    @view(validators=util.valid_spill_id)
    def get(self):
        """
        Return a JSON representation of the SurfaceReleaseSpill matching the
        ``id`` matchdict value.
        """
        spill = self.request.validated['model'].spills[self.id]

        return schema.SurfaceReleaseSpillSchema().bind().serialize(
            spill.to_dict())

    @view(validators=util.valid_spill_id,
          schema=schema.SurfaceReleaseSpillSchema)
    def put(self):
        """
        Update an existing SurfaceReleaseSpill from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        spill = model.spills[self.id]
        spill.from_dict(data)
        model.rewind()

        return schema.SurfaceReleaseSpillSchema().bind().serialize(
            spill.to_dict())

    @view(validators=util.valid_spill_id)
    def delete(self):
        """
        Delete a SurfaceReleaseSpill.
        """
        model = self.request.validated['model']
        model.spills.remove(self.id)
        model.rewind()

