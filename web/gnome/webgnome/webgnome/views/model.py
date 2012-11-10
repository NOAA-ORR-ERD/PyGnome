import json
import datetime
import gnome.map
import gnome.utilities.map_canvas
import shutil
import os

from gnome.utilities.file_tools import haz_files

from pyramid.httpexceptions import HTTPNotFound
from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.forms.movers import (
    AddMoverForm,
    WindMoverForm,
    DeleteMoverForm
)

from webgnome.forms.model import RunModelUntilForm, ModelSettingsForm
from webgnome.forms import object_form
from webgnome.navigation_tree import NavigationTree
from webgnome import util
from webgnome.views import movers


@view_config(route_name='model_forms', renderer='gnome_json')
@util.json_require_model
def model_forms(request, model):
    """
    A partial view that renders all of the add and edit forms for ``model``,
    including settings, movers and spills.
    """
    context = {
        'run_model_until_form': RunModelUntilForm(),
        'run_model_until_form_url': request.route_url('run_model_until'),
        'settings_form': ModelSettingsForm(),
        'settings_form_url': request.route_url('model_settings'),
        'add_mover_form': AddMoverForm(),
        'wind_mover_form': WindMoverForm(),
        'wind_mover_form_url': request.route_url('create_wind_mover'),
        'form_view_container_id': 'modal-container',
        'mover_update_forms': [],
        'mover_delete_forms': []
    }

    # The template will render a delete and edit form for each mover instance.
    for mover in model.movers:
        delete_form = DeleteMoverForm(model, obj=mover)
        delete_url = request.route_url('delete_mover')
        context['mover_delete_forms'].append(
            (delete_url, delete_form))

        routes = movers.form_routes.get(mover.__class__, None)

        if not routes:
            continue

        update_route = routes.get('update', None)
        update_form = object_form.get_object_form(mover)

        if update_route and update_form:
            update_url = request.route_url(update_route, id=mover.id)
            context['mover_update_forms'].append(
                (update_url, update_form(obj=mover)))

        # TODO: Spill forms.

    return {
        'html': render('model_forms.mak', context, request)
    }


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
    data['model_form_html'] = model_forms(request)['html']

     # TODO: Remove this after we decide on where to put the drop-down menu.
    data['show_menu_above_map'] = 'map_menu' in request.GET

    # These values are needed to initialize the JavaScript app.
    data['add_mover_form_id'] = AddMoverForm.get_id()
    data['model_forms_url'] = request.route_url('model_forms')
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

    if confirm:
        if model_id:
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
    data = {}

    if request.method == 'POST' and form.validate():
        date = form.date.data

        model.time_step = form.computation_time_step.data

        model.start_time = datetime.datetime(
            day=date.day, month=date.month, year=date.year,
            hour=form.hour.data, minute=form.minute.data,
            second=0)

        model.duration = datetime.timedelta(
                days=form.duration_days.data, hours=form.duration_hours.data)

        model.uncertain = form.include_minimum_regret.data

        # TODO: show_currents, prevent_land_jumping, run_backwards

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


def _get_timestamps(model):
    """
    TODO: Move into ``gnome.model.Model``?
    """
    timestamps = []

    # XXX: Why is _num_time_steps a float? Is this ok?
    for step_num in range(int(model._num_time_steps) + 1):
        if step_num == 0:
            dt = model.start_time
        else:
            delta = datetime.timedelta(seconds=step_num * model.time_step)
            dt = model.start_time + delta
        timestamps.append(dt)

    return timestamps


def _get_time_step(request, model):
    step = None
    images_dir = os.path.join(
        request.registry.settings['model_images_dir'], str(model.id))

    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    try:
        curr_step, file_path, timestamp = model.next_image(images_dir)
        filename = file_path.split(os.path.sep)[-1]
        image_url = request.static_url(
                'webgnome:static/%s/%s/%s' % (
                    request.registry.settings['model_images_url_path'],
                    model.id,
                    filename))
        step = {
            'id': curr_step,
            'url': image_url,
            'timestamp': timestamp
        }
    except StopIteration:
        pass

    return step


@view_config(route_name='run_model', renderer='gnome_json')
@util.json_require_model
def run_model(request, model):
    """
    Start a run of the user's current model and return a JSON object
    containing the first time step.
    """
    data = {}

    # TODO: This should probably be on the model.
    timestamps = _get_timestamps(model)
    data['expected_time_steps'] = timestamps

    # TODO: Set separately in map configuration view
    map_file = os.path.join(
        request.registry.settings['project_root'],
        'sample_data', 'MapBounds_Island.bna')

    # the land-water map
    model.map = gnome.map.MapFromBNA(
        map_file, refloat_halflife=6 * 3600)

    canvas = gnome.utilities.map_canvas.MapCanvas((400, 300))
    polygons = haz_files.ReadBNA(map_file, "PolygonSet")
    canvas.set_land(polygons)
    model.output_map = canvas

    data['background_image'] = request.static_url(
        'webgnome:static/%s/%s/%s' % (
        request.registry.settings['model_images_url_path'],
        model.id,
        'background_map.png'))

    first_step = _get_time_step(request, model)

    if not first_step:
        return {}

    data['step'] = first_step

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
    step = _get_time_step(request, model)

    if not step:
        raise HTTPNotFound

    return {
        'time_step': step
    }


@view_config(route_name='get_tree', renderer='gnome_json')
@util.json_require_model
def get_tree(request, model):
    """
    Return a JSON representation of the current state of the model, to be used
    to create a tree view of the model in the JavaScript application.
    """
    return NavigationTree(request, model).render()



