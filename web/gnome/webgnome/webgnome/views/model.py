import json
import datetime

from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.forms import (
    RunModelUntilForm,
    ModelSettingsForm,
    AddMoverForm,
    WindMoverForm,
    DeleteMoverForm
)

from webgnome import util
from webgnome.navigation_tree import NavigationTree
from webgnome.views import movers


@view_config(route_name='model_forms', renderer='model_forms.mak')
def model_forms(request, model):
    """
    A partial view that renders all of the add and edit forms for ``model``,
    including settings, movers and spills.
    """
    data = {
        'run_model_until_form': RunModelUntilForm(),
        'run_model_until_form_url': request.route_url('run_model_until'),
        'settings_form': ModelSettingsForm(),
        'settings_form_url': request.route_url('model_settings'),
        'add_mover_form': AddMoverForm(),
        'add_mover_form_id': 'add_mover',
        'wind_mover_form': WindMoverForm(),
        'wind_mover_form_url': request.route_url('create_wind_mover'),
        'mover_update_forms': [],
        'mover_delete_forms': []
    }

    # The template will render a delete and edit form for each mover instance.
    for mover in model.movers:
        delete_form = DeleteMoverForm(model, obj=mover)
        delete_url = request.route_url('delete_mover')
        data['mover_delete_forms'].append(
            (delete_url, delete_form))

        update_route = movers.form_routes.get(mover.__class__, None)
        update_form = util.get_object_form(mover)

        if update_route and update_form:
            update_url = request.route_url(update_route, mover.id)
            data['mover_update_forms'].append(
                (update_url, update_form(obj=mover)))

        # TODO: Spill forms.

    return data


@view_config(route_name='show_model', renderer='model.mak')
def show_model(request):
    """
    The entry-point for the web application. Load all forms and data
    needed to show a model.

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
    data['model_form_html'] = render(
        'model_forms.mak', model_forms(request, model), request)

     # TODO: Remove this after we decide on where to put the drop-down menu.
    data['show_menu_above_map'] = 'map_menu' in request.GET

    # These values are needed to initialize the JavaScript app.
    data['add_mover_form_id'] = 'add_mover'
    data['run_model_until_form_url'] = request.route_url('run_model_until')

    if model.time_steps:
        data['generated_time_steps_json'] = json.dumps(model.time_steps,
                                                       default=util.json_encoder)
        data['expected_time_steps_json'] = json.dumps(model.timestamps,
                                                      default=util.json_encoder)

    return data


@view_config(route_name='create_model', renderer='gnome_json')
def create_model(request):
    """
    Create a new model for the user. Delete the user's current model if one
    exists.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    confirm = request.POST.get('confirm_new', None)

    if model_id and confirm:
        settings.Model.delete(model_id)
        model = settings.Model.create()
        model_id = model.id
        request.session[settings.model_session_key] = model.id
        message = util.make_message('success', 'Created a new model.')
    else:
        message = util.make_message('error', 'Could not create a new model. '
                                             'Invalid data was received.')

    return {
        'model_id': model_id,
        'message': message
    }


@view_config(route_name='model_settings', renderer='gnome_json')
@util.json_require_model
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


@view_config(route_name='run_model', renderer='gnome_json')
@util.json_require_model
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
        data['message'] = util.make_message('error', 'Model failed to run.')

    return data


@view_config(route_name='run_model_until', renderer='gnome_json')
@util.json_require_model
def run_model_until(request, model):
    """
    Render a :class:`webgnome.forms.RunModelUntilForm` for the user's
    current model on GET and validate form input on POST.
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
@util.json_require_model
def get_next_step(request, model):
    """
    Generate the next step of a model run and return the result.
    """
    return {
        'time_step': model.get_next_step()
    }


@view_config(route_name='get_tree', renderer='gnome_json')
@util.json_require_model
def get_tree(request, model):
    """
    Return a JSON representation of the current state of the model, to be used
    to create a tree view of the model in the JavaScript application.
    """
    return NavigationTree(request, model).render()



