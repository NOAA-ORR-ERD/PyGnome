import json
import datetime

from collections import OrderedDict
from pyramid.renderers import render
from pyramid.view import view_config

from ..forms import (
    RunModelUntilForm,
    ModelSettingsForm,
    AddMoverForm,
    ConstantWindMoverForm,
    VariableWindMoverForm
)

from ..util import json_require_model, make_message, json_encoder


@view_config(route_name='show_model', renderer='model.mak')
def show_model(request):
    """
    Show the current user's model.

    Get an existing `py_gnome.model.Model` using the `model_id` field in the
    user's session or create a new one.

    If `model_id` was found in the user's session but the model did not exist,
    warn the user and suggest that they reload from a save file.
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


@view_config(route_name='create_model', renderer='gnome_json')
def create_model(request):
    """
    Create a new model for the user. Delete the user's current model if one exists.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    confirm = request.POST.get('confirm_new', None)

    if model_id and confirm:
        settings.Model.delete(model_id)
        model = settings.Model.create()
        model_id = model.id
        request.session[settings.model_session_key] = model.id
        message = make_message('success', 'Created a new model.')
    else:
        message = make_message('error', 'Could not create a new model. '
                                         'Invalid data was received.')

    return {
        'model_id': model_id,
        'message': message
    }


@view_config(route_name='run_model', renderer='gnome_json')
@json_require_model
def run_model(request, model):
    """
    Starts a run of the user's current model and return a JSON object
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


@view_config(route_name='run_model_until', renderer='gnome_json')
@json_require_model
def run_model_until(request, model):
    """
    An AJAX form view that renders a `RunModelUntilForm` and validates its input.
    """
    form = RunModelUntilForm(request.POST)
    data = {}

    if request.method == 'POST' and form.validate():
        date = form.get_datetime()
        model.set_run_until(date)
        return {'run_until': date, 'form_html': None}

    context = {
        'form': form,
        'action_url': request.route_url('run_model_until')
    }

    data['form_html'] = render(
        'webgnome:templates/forms/run_model_until.mak', context)

    return data


@view_config(route_name='get_next_step', renderer='gnome_json')
@json_require_model
def get_next_step(request, model):
    return {
        'time_step': model.get_next_step()
    }


@view_config(route_name='model_settings', renderer='gnome_json')
@json_require_model
def model_settings(request, model):
    form = ModelSettingsForm(request.POST)

    if request.method == 'POST' and form.validate():
        return {
            'form_html': None
        }

    context = {
        'form': form,
        'action_url': request.route_url('model_settings')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/model_settings.mak', context)
    }


def _get_model_settings(model):
    """
    Return a dict of values containing each setting in `model` that the client
    should be able to read and change.
    """
    settings_attrs = [
        'start_time',
        'duration'
    ]

    settings = OrderedDict()

    for attr in settings_attrs:
        if hasattr(model, attr):
            settings[attr] = getattr(model, attr)

    return settings


@view_config(route_name='get_tree', renderer='gnome_json')
@json_require_model
def get_tree(request, model):
    """
    Return a JSON representation of the current state of the model, to be used
    to create a tree view of the model in the JavaScript application.
    """
    settings = {'title': 'Model Settings', 'type': 'settings', 'children': []}
    movers = {'title': 'Movers', 'type': 'add_mover', 'children': []}
    spills = {'title': 'Spills', 'type': 'add_spill', 'children': []}

    def get_value_title(name, value, max_chars=8):
        """
        Return a title string that combines `name` and `value`, with value
        shortened if it is longer than `max_chars`.
        """
        name = name.replace('_', ' ').title()
        value = (str(value)).title()
        value = value if len(value) <= max_chars else '%s ...' % value[:max_chars]
        return '%s: %s' % (name, value)

    for name, value in _get_model_settings(model).items():
        settings['children'].append({
            'type': 'settings',
            'title': get_value_title(name, value),
        })

    # If we had a map, we would set its ID value here, whatever that value
    # ends up being.
    settings['children'].append({
        'type': 'map',
        'title': 'Map: None'
    })

    for mover in model.movers:
        movers['children'].append({
            'type': mover.name,
            'id': mover.id,
            'title': str(mover)
        })

    for spill in model.spills:
        spills['children'].append({
            'type': spill.name,
            'id': spill.id,
            'title': get_value_title('ID', id),
        })

    return [settings, movers, spills]
