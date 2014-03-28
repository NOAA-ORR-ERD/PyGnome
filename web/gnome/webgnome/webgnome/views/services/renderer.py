from cornice.resource import resource, view

from gnome.outputters import RendererSchema

from webgnome import util
from webgnome.views.services.base import BaseResource


@resource(path='/model/{model_id}/renderer', renderer='gnome_json',
          description="The user's current renderer.")
class Renderer(BaseResource):
    @view(validators=util.valid_renderer)
    def get(self):
        """
        Return a JSON representation of the current renderer.
        """
        return RendererSchema().bind().serialize(
            self.request.validated['renderer'].to_dict(do='create'))

    @view(validators=util.valid_renderer, schema=RendererSchema)
    def put(self):
        """
        Update an existing renderer.
        """
        renderer = self.request.validated['renderer']
        renderer.from_dict(self.request.validated)
        self.request.validated['model'].mark_changed()

        return RendererSchema().bind().serialize(
            renderer.to_dict(do='create'))

    @view(validators=util.valid_renderer)
    def delete(self):
        self.request.validated['model'].remove_renderer()
