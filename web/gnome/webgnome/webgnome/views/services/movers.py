
from cornice.resource import resource, view

from gnome.movers.random_movers import RandomMoverSchema

from webgnome import util
from webgnome import schema
from webgnome.model_manager import WebWindMover, WebRandomMover
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id}/mover/wind',
          path='/model/{model_id}/mover/wind/{id}',
          renderer='gnome_json', description='A wind mover.')
class WindMover(BaseResource):
    @view(validators=[util.valid_model_id, util.valid_wind_id],
          schema=schema.WindMoverSchema)
    def collection_post(self):
        """
        Create a :class:`model_manager.WebWebWindMover` from a JSON
        representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = WebWindMover(**data)
        mover.wind = self.request.validated['wind']
        model.movers.add(mover)

        mover_data = mover.to_dict(do='create')
        return schema.WindMoverSchema().bind().serialize(mover_data)

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of JSON representations of the target model's
        :class:`model_manager.WebWindMover`s.
        """
        data = self.request.validated
        model = data.pop('model')
        model_data = model.to_dict(include_movers=True)

        return schema.WindMoverSchema().bind().serialize(
            model_data['wind_movers'])

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of the :class:`model_manager.WebWindMover`
        whose ID matches the ``id`` matchdict value.
        """
        model = self.request.validated.pop('model')
        mover = model.movers.get(self.id)
        mover_data = mover.to_dict('create')

        return schema.WindMoverSchema().bind().serialize(mover_data)

    @view(validators=[util.valid_mover_id, util.valid_wind_id],
          schema=schema.WindMoverSchema)
    def put(self):
        """
        Update an existing :class:`model_manager.WebWindMover` from a JSON
        representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = model.movers.get(self.id)
        wind = data.pop('wind')
        mover.from_dict(data)
        mover.wind = wind
        mover_data = mover.to_dict(do='create')

        return schema.WindMoverSchema().bind().serialize(mover_data)

    @view(validators=util.valid_mover_id)
    def delete(self):
        """
        Delete a :class:`model_manager.WebWindMover`.
        """
        self.request.validated['model'].movers.remove(self.id)


@resource(collection_path='/model/{model_id}/mover/random',
          path='/model/{model_id}/mover/random/{id}',
          renderer='gnome_json', description='A random mover.')
class RandomMover(BaseResource):
    @view(validators=util.valid_model_id,
          schema=schema.RandomMoverSchema)
    def collection_post(self):
        """
        Create a :class:`model_manager.WebRandomMover` from a JSON
        representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = WebRandomMover(**data)
        model.movers.add(mover)

        mover_data = mover.to_dict(do='create')
        return schema.RandomMoverSchema().bind().serialize(mover_data)

    @view(validators=util.valid_model_id)
    def collection_get(self):
        """
        Return a list of existing :class:`model_manager.WebRandomMover`.
        """
        data = self.request.validated
        model = data.pop('model')

        model_data = model.to_dict(include_movers=True)
        return schema.RandomMoverSchema().bind().serialize(
            model_data['random_movers'])

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of :class:`model_manager.WebRandomMover`
        matching the ``id`` matchdict value.
        """
        model = self.request.validated.pop('model')
        mover = model.movers.get(self.id)
        mover_data = mover.to_dict('create')

        return schema.RandomMoverSchema().bind().serialize(mover_data)

    @view(validators=util.valid_mover_id, schema=schema.RandomMoverSchema)
    def put(self):
        """
        Update an existing :class:`model_manager.WebRandomMover` from a JSON
        representation.
        """
        data = self.request.validated
        model = data.pop('model')
        mover = model.movers.get(self.id)
        mover.from_dict(data)

        return RandomMoverSchema().bind().serialize(mover.to_dict())

    @view(validators=util.valid_mover_id)
    def delete(self):
        """
        Delete a :class:`model_manager.WebRandomMover`.
        """
        self.request.validated['model'].movers.remove(self.id)
