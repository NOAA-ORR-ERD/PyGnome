from cornice.resource import resource, view

from webgnome import util
from webgnome.model_manager import WebWindMover, WebRandomMover, WebWind
from webgnome import schema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id}/mover/wind',
          path='/model/{model_id}/mover/wind/{id}',
          renderer='gnome_json', description='A wind mover.')
class WindMover(BaseResource):
    optional_fields = ['active_start', 'active_stop']

    @view(validators=util.valid_model_id, schema=schema.WindMoverSchema)
    def collection_post(self):
        """
        Create a WindMover from a JSON representation.
        """
        data = self.prepare(self.request.validated)
        model = data.pop('model')
        wind_data = data.pop('wind')
        data['wind'] = WebWind(**wind_data)
        mover = WebWindMover(**data)
        model.movers.add(mover)
        mover_data = mover.to_dict(do='create')
        mover_data['wind'] = wind_data

        return schema.WindMoverSchema().bind().serialize(mover_data)

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing WindMovers.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_movers=True)

        return schema.WindMoversSchema().bind().serialize(
            model_data['wind_movers'])

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of WindMover matching the ``id`` matchdict
        value.
        """
        model = self.request.validated.pop('model')
        mover = model.movers.get(self.id)

        return schema.WindMoverSchema().bind().serialize(mover.to_dict())

    @view(validators=util.valid_mover_id, schema=schema.WindMoverSchema)
    def put(self):
        """
        Update an existing WindMover from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = model.movers.get(self.id)
        mover.from_dict(data)
        mover_data = mover.to_dict(do='create')

        return schema.WindMoverSchema().bind().serialize(mover_data)

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

    @view(validators=util.valid_model_id, schema=schema.RandomMoverSchema)
    def collection_post(self):
        """
        Create a RandomMover from a JSON representation.
        """
        data = self.prepare(self.request.validated)
        model = data.pop('model')
        mover = WebRandomMover(**data)
        model.movers.add(mover)

        return schema.RandomMoverSchema().bind().serialize(
            mover.to_dict(do='create'))

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing RandomMovers.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_movers=True)

        return schema.RandomMoversSchema().bind().serialize(
            model_data['random_movers'])

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of RandomMover matching the ``id``
        matchdict value.
        """
        model = self.request.validated.pop('model')
        return model.movers.get(self.id).to_dict()

    @view(validators=util.valid_mover_id, schema=schema.RandomMoverSchema)
    def put(self):
        """
        Update an existing RandomMover from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = model.movers.get(self.id)
        mover.from_dict(data)

        return schema.RandomMoverSchema().bind().serialize(mover.to_dict())

    @view(validators=util.valid_mover_id)
    def delete(self):
        """
        Delete a RandomMover.
        """
        self.request.validated['model'].movers.remove(self.id)       
