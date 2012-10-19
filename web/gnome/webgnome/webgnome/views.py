import json
import datetime

from pyramid.httpexceptions import HTTPFound, HTTPNotFound
from pyramid.renderers import render
from pyramid.view import view_config

from forms import (
    AddMoverForm,
    VariableWindMoverForm,
    ConstantWindMoverForm,
    RunModelUntilForm,
    MOVER_VARIABLE_WIND,
    MOVER_CONSTANT_WIND
)

from util import json_require_model, json_encoder


def _make_message(type, text):
    """
    Create a "message" dictionary suitable to be returned in a JSON response.
    """
    return dict(mesage=dict(type=type, text=text))


@view_config(route_name='show_model', renderer='model.mak')
def show_model(request):
    """
    Show the current user's model.

    Get or create an existing `py_gnome.model.Model` using the `model_id`
    field in the user's session.

    If `model_id` was found in the user's session but the model did not exist,
    warn the user and suggest that they reload from a save file.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    model, created = settings.Model.get_or_create(model_id)
    data = {}

    if created:
        request.session[settings.model_session_key] = model.id

        # A model with ID `model_id` did not exist, so we created a new one.
        if model_id:
            data['warning'] = 'The model you were working on is no longer ' \
                              'available. We created a new one for you.'
    data['model'] = model
    data['show_menu_above_map'] = 'map_menu' in request.GET

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
    request.session[settings.model_session_key] = model.id
    message = _make_message('success', 'Created a new model.')

    return {
        'model_id': model.id,
        'message': message
    }


@view_config(route_name='run_model', renderer='gnome_json')
@json_require_model
def run_model(request, model):
    """
    Run the user's current model and return a JSON object containing the result
    of the run.

    A `startAtTimeStep` POST value will attempt to generate (or return the
    cached) time steps until `startAtTimeStep` and set the model's `current_step`
    to `startAtTimeStep`, so that when the user asks for the next step, the
    model returns `startAtTimeStep` + 1.

    A `runUntilTimeStep` POST value will set the `run_until_step` value in the
    model and raise a `StopIteration` exception during the call to
    `model.get_next_step()` at that step.
    """
    # TODO: Accept this value from the user as a setting and require it to run.
    # TODO: Parse POST values.
    two_weeks_ago = datetime.datetime.now() - datetime.timedelta(weeks=4)
    model.start_time = two_weeks_ago
    data = {}

    try:
        data['running'] = model.run()
        data['expected_time_steps'] = model.timestamps
    except RuntimeError:
        # TODO: Use an application-specific exception.
        data['running'] = False
        data['message'] = _make_message('error', 'Model failed to run.')

    return data


@view_config(route_name='run_model_until', renderer='gnome_json')
@json_require_model
def run_model_until(request, model):
    """
    An AJAX form view that renders `RunModelUntilForm` and validates its input.
    """
    form = RunModelUntilForm(request.POST)
    data = {}

    if request.method == 'POST' and form.validate():
        model.set_run_until(form.run_until)
        return {'run_until': form.run_until.data}

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


@view_config(route_name='edit_constant_wind_mover', renderer='gnome_json')
@json_require_model
def edit_constant_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(mover_id)
    opts = {'obj': mover} if mover else {}
    form = ConstantWindMoverForm(request.POST or None, **opts)

    if request.method == 'POST' and form.validate():
        if model.has_mover_with_id(mover_id):
            model.update_mover(mover_id, form.data)
            message = _make_message('success',
                                    'Updated constant wind mover successfully.')
        else:
            mover_id = model.add_mover(form.data)
            message = _make_message('warning',
                                    'The specified mover did not exist. Added '
                                    'a new constant wind mover to the model.')
        return {
            'id': mover_id,
            'message': message
        }

    html = render('webgnome:templates/forms/constant_wind_mover.mak', {
        'form': form,
        'action_url': request.route_url(
            'edit_constant_wind_mover', id=mover_id)
    })

    return {'form_html': html}


@view_config(route_name='edit_variable_wind_mover', renderer='gnome_json')
@json_require_model
def edit_variable_wind_mover(request, model):
    mover_id = request.matchdict['id']
    mover = model.get_mover(mover_id)
    opts = {'obj': mover} if mover else {}
    form = VariableWindMoverForm(request.POST or None, **opts)

    if request.method == 'POST' and form.validate():
        if model.has_mover_with_id(mover_id):
            model.update_mover(mover_id, form.data)
            message = _make_message('success',
                                    'Updated variable wind mover successfully.')
        else:
            mover_id = model.add_mover(form.data)
            message = _make_message('warning',
                                    'The specified mover did not exist. Added '
                                    'a new variable wind mover to the model.')
        return {
            'id': mover_id,
            'message': message
        }

    html = render('webgnome:templates/forms/variable_wind_mover.mak', {
        'form': form,
        'action_url': request.route_url('edit_variable_wind_mover', id=mover_id)
    })

    return {'form_html': html}


@view_config(route_name='add_constant_wind_mover', renderer='gnome_json')
@json_require_model
def add_constant_wind_mover(request, model):
    form = ConstantWindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        return {
            'id': model.add_mover(form.data),
            'type': 'mover',
            'message': _make_message(
                'success', 'Added a variable wind mover to the model.')
        }

    html = render('webgnome:templates/forms/constant_wind_mover.mak', {
        'form': form,
        'action_url': request.route_url('add_constant_wind_mover')
    })

    return {'form_html': html}


@view_config(route_name='add_variable_wind_mover', renderer='gnome_json')
@json_require_model
def add_variable_wind_mover(request, model):
    form = VariableWindMoverForm(request.POST)

    if request.method == 'POST' and form.validate():
        return {
            'id': model.add_mover(form.data),
            'type': 'mover',
            'message': _make_message(
                'success', 'Added a variable wind mover to the model.')
        }

    context = {
        'form': form,
        'action_url': request.route_url('add_variable_wind_mover')
    }

    return {
        'form_html': render(
            'webgnome:templates/forms/variable_wind_mover.mak', context)
    }


@view_config(route_name='add_mover', renderer='gnome_json')
@json_require_model
def add_mover(request, model, type=None):
    form = AddMoverForm(request.POST)
    data = {}

    mover_routes = {
        MOVER_VARIABLE_WIND: 'add_variable_wind_mover',
        MOVER_CONSTANT_WIND: 'add_constant_wind_mover'
    }

    if request.method == 'POST' and form.validate():
        route = mover_routes.get(form.mover_type.data)
        return HTTPFound(request.route_url(route))

    context = {
        'form': form,
        'action_url': request.route_url('add_mover')
    }

    data['form_html'] = render(
        'webgnome:templates/forms/add_mover_form.mak', context)

    return data


@view_config(route_name='delete_mover', renderer='gnome_json', request_method='POST')
@json_require_model
def delete_mover(request, model):
    mover_id = request.POST.get('mover_id', None)

    if mover_id is None or model.has_mover_with_id(mover_id) is False:
        raise HTTPNotFound

    model.delete_mover(mover_id)

    return {
        'message': _make_message('success', 'Mover deleted.')
    }


@view_config(route_name='get_tree', renderer='gnome_json')
@json_require_model
def get_tree(request, model):
    settings = {'title': 'Model Settings', 'key': 'setting', 'children': []}
    movers = {'title': 'Movers', 'key': 'mover', 'children': []}
    spills = {'title': 'Spills', 'key': 'spill', 'children': []}

    def get_value_title(name, value, max_chars=8):
        """
        Return a title string that uses `name` and `value`, with value shortened
        if it's longer than `max_chars`.
        """
        value = str(value)
        value = value if len(value) <= max_chars else '%s ...' % value[:max_chars]
        return '%s: %s' % (name, value)

    for setting in model.get_settings():
        settings['children'].append({
            'key': setting.name,
            'title': get_value_title(setting.name, setting.value),
            'type': 'setting'
        })

    map = model.get_map()

    if map:
        settings['children'].append({
            'key': 'map',
            'title': get_value_title('Map', map.name),
            'type': 'setting'
        })

    for id, mover in model.get_movers().items():
        movers['children'].append({
            'key': id,
            'title': model.get_mover_title(mover),
            'type': mover.type
        })

    for id, spill in model.get_spills().items():
        spills['children'].append({
            'key': id,
            'title': get_value_title('ID', id),
            'type': spill.type
        })

    return [settings, movers, spills]


