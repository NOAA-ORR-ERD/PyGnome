from cornice.resource import resource, view
from gnome.weather import Wind

from webgnome import util
from webgnome.model_manager import WebWindMover
from webgnome.schema import WindMoverSchema
from webgnome.views.services.base import BaseResource


@resource(collection_path='/mover/wind', path='/mover/wind/{id:\d+}',
          renderer='gnome_json', description='A wind mover.')
class WindMover(BaseResource):

    @property
    def data(self):
        """
        Return ``self.request.validated`` after modifying it for use as input
        to a WindMover constructor.

        NOTE again that this modified ``self.request.validated`` rather than
        copying it as the dict may contain a lot of data.
        """
        data = self.request.validated

        if 'wind' in data and data['wind']:
            wind_data = data.pop('wind')
            wind = Wind(units=wind_data['units'],
                        timeseries=wind_data['timeseries'])
            data['wind'] = wind

        return data

    @view(validators=util.valid_model_id, schema=WindMoverSchema)
    def collection_post(self):
        """
        Create a WindMover from a JSON representation.
        """
        mover = WebWindMover(**self.data)
        self.model.add_mover(mover)

        return {
            'success': True,
            'id': mover.id
        }

    @view(validators=util.valid_mover_id)
    def get(self):
        """
        Return a JSON representation of WindMover matching the ``id`` matchdict
        value.
        """
        mover = self.model.get_mover(self.id)
        return mover.to_dict()

    @view(validators=util.valid_mover_id, schema=WindMoverSchema)
    def put(self):
        """
        Update an existing WindMover from a JSON representation.
        """
        mover = self.model.get_mover(self.id)
        mover.from_dict(self.data)

        return {
            'success': True,
            'id': mover.id
        }

    @view(validators=util.valid_mover_id)
    def delete(self):
        """
        Delete a WindMover.
        """
        self.model.remove_mover(self.id)
        message = util.make_message('success', 'Deleted wind mover.')

        return {
            'success': True,
            'mover_id': self.id,
            'message': message
        }
