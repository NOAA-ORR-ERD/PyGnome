import os
import logging

from cornice.resource import resource, view
from gnome.utilities.cache import CacheError
from pyramid.httpexceptions import HTTPNotFound

from webgnome import util, schema
from webgnome.navigation_tree import NavigationTree
from webgnome.views.services.base import BaseResource


log = logging.getLogger(__name__)
valid_map = util.make_map_validator(400)


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


def get_web_step_data(request, step_data, model, images_url):
    filename = step_data['image_filename'].split(os.path.sep)[-1]
    image_url = request.static_url(
        'webgnome:static/%s/%s/%s/%s' % (
            images_url, model.id,
            util.get_filename_safe_time(model.changed_at), filename))

    return {
        'id': step_data['step_num'],
        'url': image_url,
        'timestamp': step_data['time_stamp']
    }


@resource(path='/model/{model_id}/step_generator', renderer='gnome_json',
          description='Run the current model.')
class StepGenerator(BaseResource):
    def _get_next_step(self):
        """
        Generate the next step of the model run and return a dict of metadata
        describing the step, including a URL to an image of particles for the
        step.
        """
        model = self.request.validated['model']

        try:
            step_data = model.step()
        except StopIteration:
            return

        return get_web_step_data(self.request, step_data, model,
                                 self.settings['model_images_url_path'])

    @view(validators=[util.valid_model_id, valid_map])
    def post(self):
        """
        Start a run of the user's current model and return a JSON object
        containing the first time step.
        """
        model = self.request.validated['model']
        data = {}
        model.rewind()

        first_step = self._get_next_step()
        model.time_steps.append(first_step)

        data['expected_time_steps'] = model.timestamps
        data['time_step'] = first_step

        return data

    @view(validators=[util.valid_model_id, valid_map])
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
        #
        # TODO: When does this happen? Is it still used? Should we prepare the
        # renderer with `self.renderer.prepare_for_model_run()` here or has that
        # already happened elsewhere?
        if step['id'] == 0:
            model.rewind()
            data['expected_time_steps'] = model.timestamps

        self.request.validated['model'].time_steps.append(step)
        data['time_step'] = step

        return data


@resource(path='/model/{model_id}/step/{id}', renderer='gnome_json',
          description='A single step in a model. run.')
class Step(BaseResource):
    """
    Return data about a step of the model run that has already been generated.
    """
    @view(validators=util.valid_step_id)
    def get(self):
        model = self.request.validated['model']
        step_data = self.request.validated['step_data']

        return get_web_step_data(self.request, step_data, model,
                                 self.settings['model_images_url_path'])


@resource(path='/model/from_location_file/{location}', renderer='gnome_json',
          description='Create a new model from a location file.')
class ModelFromLocationFile(BaseResource):
    """
    Creates a new :class:`webgnome.model_manager.WebModel` from a location file
    with the name given by the `location` parameter.
    """
    def apply_location_file_to_model(self, model):
        self.request.session[self.settings['model_session_key']] = model.id
        model_schema = schema.ModelSchema().bind()

        model_data = model_schema.deserialize(
            self.request.validated['location_file_model_data'])
        model.from_dict(model_data)
        data = model.to_dict()

        return model_schema.serialize(data)

    @view(validators=util.valid_location_file)
    def post(self):
        """
        Create a new model using settings from a location file.
        """
        model = self.settings.Model.create()
        return self.apply_location_file_to_model(model)


@resource(path='/model/{model_id}/from_location_file/{location}',
          renderer='gnome_json',
          description='Apply a location file to an existing model.')
class ExistingModelFromLocationFile(ModelFromLocationFile):
    """
    Apply settings from a location file with the name given by the `location`
    parameter to the user's current model.

    NOTE: This service is not used by the JavaScript application at this time.
    """
    @view(validators=[util.valid_model_id, util.valid_location_file])
    def post(self):
        model = self.request.validated['model']
        return self.apply_location_file_to_model(model)
