from cornice.resource import resource, view
from webgnome import util
from webgnome import schema
from webgnome.model_manager import WebWind
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model/{model_id}/environment/wind',
          path='/model/{model_id}/environment/wind/{id}',
          renderer='gnome_json', description='A wind mover.')
class Wind(BaseResource):

    @view(validators=util.valid_model_id, schema=schema.WindSchema)
    def collection_post(self):
        """
        Create a :class:`model_manager.WebWind` from a JSON representation.
        """
        data = self.request.validated
        model = data.pop('model')
        wind = WebWind(**data)
        model.environment.add(wind)
        wind_data = wind.to_dict()

        return schema.WindSchema().bind().serialize(wind_data)

    @view(validators=util.valid_environment_id)
    def get(self):
        """
        Return a JSON representation of the :class:`model_manager.WebWind`
        whose ID matches the ``id`` matchdict value.
        """
        model = self.request.validated.pop('model')
        wind = model.environment.get(self.id)
        wind_data = wind.to_dict()

        return schema.WindSchema().bind().serialize(wind_data)

    @view(validators=util.valid_environment_id,
          schema=schema.WindSchema)
    def put(self):
        """
        Update an existing :class:`model_manager.WebWind` from a JSON
        representation.
        """
        data = self.request.validated
        model = data.pop('model')
        wind = model.environment.get(self.id)
        wind.from_dict(data)
        wind_data = wind.to_dict(do='create')

        return schema.WindSchema().bind().serialize(wind_data)

    @view(validators=util.valid_environment_id)
    def delete(self):
        self.request.validated['model'].environment.remove(self.id)
