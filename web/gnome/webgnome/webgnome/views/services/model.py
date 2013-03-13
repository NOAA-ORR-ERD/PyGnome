import datetime
import os

from cornice.resource import resource, view
from pyramid.httpexceptions import HTTPNotFound

from webgnome import util, schema
from webgnome.navigation_tree import NavigationTree
from webgnome.views.services.base import BaseResource


@resource(collection_path='/model', path='/model/{model_id}',
          renderer='gnome_json',
          description='Create a new model or delete the current model.')
class Model(BaseResource):

    @view(validators=util.valid_model_id)
    def get(self):
        """
        Return a JSON tree representation of the entire model, including movers
        and spills.
        """
        model_data = self.request.validated['model'].to_dict()

        return schema.ModelSchema().bind().serialize(model_data)

    @view(schema=schema.ModelSchema, validators=util.valid_model_id)
    def put(self):
        """
        Update settings for the current model.
        """
        model = self.request.validated['model']
        model.from_dict(self.request.validated)

        return schema.ModelSchema().bind().serialize(model.to_dict())

    @view(validators=util.valid_model_id)
    def delete(self):
        """
        Delete the current model.
        """
        self.settings.Model.delete(self.request.matchdict['model_id'])

    @view()
    def collection_post(self):
        """
        Create a new model with default settings.
        """
        model = self.settings.Model.create()
        self.request.session[self.settings['model_session_key']] = model.id

        return schema.ModelSchema().bind().serialize(model.to_dict())


@resource(path='/model/{model_id}/tree', renderer='gnome_json',
          description='A Dynatree JSON representation of the current model.')
class ModelTree(BaseResource):

    @view(validators=util.valid_model_id)
    def get(self):
        """
        Return a JSON representation of the current state of the model, to be
        used to create a tree view of the model in the JavaScript application.
        """
        return NavigationTree(self.request.validated['model']).render()


@resource(path='/model/{model_id}/runner', renderer='gnome_json',
          description='Run the current model.')
class GnomeRunner(BaseResource):
    def _get_timestamps(self):
        """
        Get the expected timestamps for a model run.

        TODO: Move into ``gnome.model.Model``?
        """
        timestamps = []
        model = self.request.validated['model']

        # XXX: Why is _num_time_steps a float? Is this ok?
        for step_num in range(int(model._num_time_steps) + 1):
            if step_num == 0:
                dt = model.start_time
            else:
                delta = datetime.timedelta(
                    seconds=step_num * model.time_step)
                dt = model.start_time + delta
            timestamps.append(dt)

        return timestamps

    def _get_next_step(self):
        """
        Generate the next step of the model run and return a dict of metadata
        describing the step, including a URL to an image of particles.
        """
        step = None
        model = self.request.validated['model']
        if not os.path.exists(model.data_dir):
            util.mkdir_p(model.data_dir)

        try:
            curr_step, file_path, timestamp = model.next_image(model.data_dir)
            filename = file_path.split(os.path.sep)[-1]
            image_url = self.request.static_url(
                'webgnome:static/%s/%s/%s/%s' % (
                    self.settings['model_images_url_path'],
                    model.id, model.runtime, filename))

            step = {
                'id': curr_step,
                'url': image_url,
                'timestamp': timestamp
            }
        except StopIteration:
            pass

        return step

    @view(validators=util.valid_model_id)
    def post(self):
        """
        Start a run of the user's current model and return a JSON object
        containing the first time step.
        """
        model = self.request.validated['model']
        data = {}

        # TODO: Some of this should probably be in a model method.
        timestamps = self._get_timestamps()
        model.timestamps = timestamps
        model.runtime = util.get_runtime()
        model.rewind()
        model.time_steps = []

        # Make the directory for the current model run if it doesn't exist.
        util.mkdir_p(model.data_dir)

        first_step = self._get_next_step()
        model.time_steps.append(first_step)

        data['expected_time_steps'] = timestamps
        data['time_step'] = first_step

        return data

    @view(validators=util.valid_model_id)
    def get(self):
        """
        Get the next step in the model run.
        """
        step = self._get_next_step()
        model = self.request.validated['model']
        data = {}

        if not step:
            raise HTTPNotFound

        # The model rewound itself. Reset web-specific caches
        # and send the list of expected timestamps.
        if step['id'] == 0:
            model.time_steps = []
            model.timestamps = self._get_timestamps()
            model.runtime = util.get_runtime()
            data['expected_time_steps'] = self._get_timestamps()

        self.request.validated['model'].time_steps.append(step)
        data['time_step'] = step

        return data


@resource(path='/model/from_location_file/{location}', renderer='gnome_json',
          description='Create a new model from a location file.')
class ModelFromLocationFile(BaseResource):

    @view(validators=util.valid_location_file)
    def post(self):
        """
        Create a new model using settings from a location file.
        """
        model = self.settings.Model.create()
        self.request.session[self.settings['model_session_key']] = model.id
        model_schema = schema.ModelSchema().bind()

        model_data = model_schema.deserialize(
            self.request.validated['location_file_model_data'])
        model.from_dict(model_data)
        data = model.to_dict()

        return model_schema.serialize(data)
