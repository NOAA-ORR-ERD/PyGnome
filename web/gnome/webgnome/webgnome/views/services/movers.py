from cornice.resource import resource, view
from pyramid.httpexceptions import HTTPBadRequest

from webgnome import util
from webgnome.model_manager import WebWindMover, WebRandomMover
from webgnome.schema import WindMoverSchema, RandomMoverSchema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id}/mover/wind',
          path='/model/{model_id}/mover/wind/{id}',
          renderer='gnome_json', description='A wind mover.')
class WindMover(BaseResource):
    optional_fields = ['active_start', 'active_stop']

    @view(validators=util.valid_model_id, schema=WindMoverSchema)
    def collection_post(self):
        """
        Create a WindMover from a JSON representation.
        """
        data = self.prepare(self.request.validated)
        model = data.pop('model')
        mover = WebWindMover(**data)
        model.movers.add(mover)

        return mover.to_dict()

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing WindMovers.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_movers=True)

        return model_data['wind_movers']

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of WindMover matching the ``id`` matchdict
        value.
        """
        model = self.request.validated.pop('model')
        return model.movers.get(self.id).to_dict()

    @view(validators=util.valid_mover_id, schema=WindMoverSchema)
    def put(self):
        """
        Update an existing WindMover from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = model.movers.get(self.id).from_dict(data)

        return mover.to_dict()

    @view(validators=util.valid_mover_id)
    def delete(self):
        """
        Delete a WindMover.
        """
        self.request.validated['model'].movers.remove(self.id)


@resource(collection_path='/model/{model_id}/mover/random',
          path='/model/{model_id}/mover/random/{id}',
          renderer='gnome_json', description='A random mover.')
class RandomMover(BaseResource):
    optional_fields = ['active_start', 'active_stop']

    @view(validators=util.valid_model_id, schema=RandomMoverSchema)
    def collection_post(self):
        """
        Create a RandomMover from a JSON representation.
        """
        data = self.prepare(self.request.validated)
        model = data.pop('model')
        mover = WebRandomMover(**data)
        model.movers.add(mover)

        return mover.to_dict()

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing RandomMovers.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_movers=True)

        return model_data['wind_movers']

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of RandomMover matching the ``id``
        matchdict value.
        """
        model = self.request.validated.pop('model')
        return model.movers.get(self.id).to_dict()

    @view(validators=util.valid_mover_id, schema=RandomMoverSchema)
    def put(self):
        """
        Update an existing RandomMover from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = model.movers.get(self.id).from_dict(data)

        return mover.to_dict()

    @view(validators=util.valid_mover_id)
    def delete(self):
        """
        Delete a RandomMover.
        """
        self.request.validated['model'].movers.remove(self.id)       
