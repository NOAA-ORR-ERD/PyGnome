import json
import datetime
import gnome.model

from pyramid.renderers import render
from pyramid.view import view_config

from ..forms import (
    RunModelUntilForm,
    ModelSettingsForm,
    AddMoverForm,
    ConstantWindMoverForm,
    VariableWindMoverForm
)

from webgnome.form_view import FormView
from webgnome.navigation_tree import NavigationTree
from webgnome.util import json_require_model, make_message, json_encoder


class ModelFormView(FormView):
    """
    Form routes for :class:`gnome.model.Model`.
    """
    wrapped_class = gnome.model.Model

    CREATE = 'create_model'
    RUN_UNTIL = 'run_model_until'
    SETTINGS = 'model_settings'

    @view_config(route_name=CREATE, renderer='gnome_json')
    def create_model(self):
        """
        Create a new model for the user. Delete the user's current model if one
        exists.
        """
        settings = self.request.registry.settings
        model_id = self.request.session.get(settings.model_session_key, None)
        confirm = self.request.POST.get('confirm_new', None)

        if model_id and confirm:
            settings.Model.delete(model_id)
            model = settings.Model.create()
            model_id = model.id
            self.request.session[settings.model_session_key] = model.id
            message = make_message('success', 'Created a new model.')
        else:
            message = make_message('error', 'Could not create a new model. '
                                             'Invalid data was received.')

        return {
            'model_id': model_id,
            'message': message
        }

    @view_config(route_name=RUN_UNTIL, renderer='gnome_json')
    @json_require_model
    def run_model_until(self, model):
        """
        Render a :class:`webgnome.forms.RunModelUntilForm` for the user's
        current model on GET and validate form input on POST.
        """
        form = RunModelUntilForm(self.request.POST)
        data = {}

        if self.request.method == 'POST' and form.validate():
            date = form.get_datetime()
            model.set_run_until(date)
            return {'run_until': date, 'form_html': None}

        context = {
            'form': form,
            'action_url': self.request.route_url(self.RUN_UNTIL)
        }

        data['form_html'] = render(
            'webgnome:templates/forms/run_model_until.mak', context)

        return data

    @view_config(route_name='model_settings', renderer='gnome_json')
    @json_require_model
    def model_settings(self, model):
        form = ModelSettingsForm(self.request.POST)

        if self.request.method == 'POST' and form.validate():
            return {
                'form_html': None
            }

        context = {
            'form': form,
            'action_url': self.request.route_url(self.SETTINGS)
        }

        return {
            'form_html': render(
                'webgnome:templates/forms/model_settings.mak', context)
        }


@view_config(route_name='show_model', renderer='model.mak')
def show_model(request):
    """
    Show the current user's model.

    Get an existing :class:`gnome.model.Model` using the ``model_id`` field
    in the user's session or create a new one.

    If ``model_id`` was found in the user's session but the model did not
    exist, warn the user and suggest that they reload from a save file.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    model, created = settings.Model.get_or_create(model_id)
    data = {}

    if created:
        request.session[settings.model_session_key] = model.id
        if model_id:
            data['warning'] = 'The model you were working on is no longer ' \
                              'available. We created a new one for you.'
    data['model'] = model

    # TODO: Remove this after we decide on where to put the drop-down menu.
    data['show_menu_above_map'] = 'map_menu' in request.GET

    form_urls = {
        'run_until': request.route_url('run_model_until'),
        'settings': request.route_url('model_settings'),
        'constant_wind_mover': request.route_url('add_constant_wind_mover'),
        'variable_wind_mover': request.route_url('add_variable_wind_mover')
    }

    data['run_model_until_form'] = RunModelUntilForm()
    data['run_model_until_form_url'] = form_urls['run_until']

    data['settings_form'] = ModelSettingsForm()
    data['settings_form_url'] = form_urls['settings']

    data['add_mover_form'] = AddMoverForm()

    data['constant_wind_form'] = ConstantWindMoverForm()
    data['constant_wind_form_url'] = form_urls['constant_wind_mover']

    data['variable_wind_form'] = VariableWindMoverForm()
    data['variable_wind_form_url'] = form_urls['variable_wind_mover']

    data['form_urls'] = json.dumps(form_urls)

    if model.time_steps:
        data['generated_time_steps_json'] = json.dumps(model.time_steps,
                                                       default=json_encoder)
        data['expected_time_steps_json'] = json.dumps(model.timestamps,
                                                      default=json_encoder)

    return data


@view_config(route_name='run_model', renderer='gnome_json')
@json_require_model
def run_model(request, model):
    """
    Start a run of the user's current model and return a JSON object
    containing the first time step.
    """
    # TODO: Accept this value from the user as a setting and require it to run.
    two_weeks_ago = datetime.datetime.now() - datetime.timedelta(weeks=4)
    model.start_time = two_weeks_ago
    data = {}

    try:
        data['running'] = model.run()
        data['expected_time_steps'] = model.timestamps
    except RuntimeError:
        # TODO: Use an application-specific exception.
        data['running'] = False
        data['message'] = make_message('error', 'Model failed to run.')

    return data


@view_config(route_name='get_next_step', renderer='gnome_json')
@json_require_model
def get_next_step(request, model):
    """
    Generate the next step of a model run and return the result.
    """
    return {
        'time_step': model.get_next_step()
    }


@view_config(route_name='get_tree', renderer='gnome_json')
@json_require_model
def get_tree(request, model):
    """
    Return a JSON representation of the current state of the model, to be used
    to create a tree view of the model in the JavaScript application.
    """
    return NavigationTree(request, model).render()
