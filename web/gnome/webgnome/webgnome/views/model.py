import json
import datetime
import gnome.model
import gnome.movers

from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.forms import (
    RunModelUntilForm,
    ModelSettingsForm,
    AddMoverForm,
    WindMoverForm
)

from webgnome.form_view import FormViewBase
from webgnome.navigation_tree import NavigationTree
from webgnome.util import json_require_model, make_message, json_encoder

from mover_forms import MoverFormViews


@view_config(route_name='model_forms', renderer='model_forms.mak')
def model_forms(request, model):
    """
    A partial that renders all of the add and edit forms for ``model``,
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
        delete_form = MoverFormViews.forms[MoverFormViews.DELETE_ROUTE]
        delete_form_id = FormViewBase.get_form_id(mover, 'delete')
        delete_url = request.route_url(MoverFormViews.DELETE_ROUTE)
        data['mover_delete_forms'].append(
            (delete_url, delete_form(model, obj=mover), delete_form_id))

        form_view = FormViewBase.get_form_view(mover)
        update_form_id = FormViewBase.get_form_id(mover)
        update_form = form_view.forms.get(form_view.UPDATE_ROUTE, None)
        update_url = request.route_url(form_view.UPDATE_ROUTE, id=mover.id)
        data['mover_update_forms'].append(
            (update_url, update_form(obj=mover), update_form_id))

        # TODO: Edit forms for existing spills.

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



