from cornice.resource import resource, view

from webgnome import util
from webgnome.model_manager import WebPointSourceRelease
from webgnome import schema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id}/spill/surface_release',
          path='/model/{model_id}/spill/surface_release/{id}',
          renderer='gnome_json', description='A surface release spill.')
class PointSourceRelease(BaseResource):

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing PointSourceReleases.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_spills=True)

        return schema.PointSourceReleasesSchema(
            model_data['surface_release_spills'])

    @view(validators=util.valid_model_id,
          schema=schema.PointSourceReleaseSchema)
    def collection_post(self):
        """
        Create a PointSourceRelease from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')

        if 'end_release_time' not in data:
            data['end_release_time'] = data['release_time']

        spill = WebPointSourceRelease(**data)
        model.spills.add(spill)

        return schema.PointSourceReleaseSchema().bind().serialize(
            spill.to_dict(do='create'))

    @view(validators=util.valid_spill_id)
    def get(self):
        """
        Return a JSON representation of the PointSourceRelease matching the
        ``id`` matchdict value.
        """
        spill = self.request.validated['model'].spills[self.id]

        return schema.PointSourceReleaseSchema().bind().serialize(
            spill.to_dict('create'))

    @view(validators=util.valid_spill_id,
          schema=schema.PointSourceReleaseSchema)
    def put(self):
        """
        Update an existing PointSourceRelease from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        spill = model.spills[self.id]
        spill.from_dict(data)

        # XXX: The model will set ``end_position`` to the start position if
        # end position is None in __init__, but not afterward. Is there a
        # better way to keep these in sync? Perhaps we set ``end_position`` to
        # None here and have an attribute setter that sets it to start_position
        # if it's None. Then we could just call self.end_position = end_position
        # in __init__, too, and it would do the right thing.
        if 'end_position' not in data:
            spill.end_position = spill.start_position

        if 'end_release_time' not in data:
            spill.end_release_time = spill.release_time

        model.rewind()

        return schema.PointSourceReleaseSchema().bind().serialize(
            spill.to_dict('create'))

    @view(validators=util.valid_spill_id)
    def delete(self):
        """
        Delete a PointSourceRelease.
        """
        model = self.request.validated['model']
        model.spills.remove(self.id)
        model.rewind()

