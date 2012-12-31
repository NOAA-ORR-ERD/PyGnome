import json

from pyramid.renderers import render
from pyramid.view import view_config

from webgnome.model_manager import WebWindMover, WebPointReleaseSpill
from webgnome import schema
from webgnome import util


form_templates = {
    WebWindMover: (schema.WindMoverSchema, 'forms/wind_mover.mak'),
    WebPointReleaseSpill: (schema.PointReleaseSpillSchema,
                           'forms/point_release_spill.mak')
}

default_wind_value = schema.WindValueSchema().bind().serialize()


def render_obj_forms(request, context_fn, objs):
    forms = []

    for obj in objs:
        schema, template = form_templates.get(obj.__class__, None)

        if template and schema:
            appstruct = schema().serialize(obj.to_dict())
            context = context_fn(obj, appstruct)
            forms.append(render(template, context, request))

    return '\n'.join(forms) if forms else ''


def get_spill_context(spill, appstruct):
    return {'spill': appstruct, 'spill_id': spill.id}


def get_mover_context(mover, appstruct):
    context = {'mover': appstruct, 'mover_id': mover.id}

    if mover.__class__ is WebWindMover:
        context['default_wind'] = default_wind_value

    return context


@view_config(route_name='model_forms', renderer='gnome_json')
@util.require_model
def model_forms(request, model):
    """
    A partial view that renders all forms for ``model``,
    including settings, movers and spills.
    """
    context = {
        'form_view_container_id': 'modal-container',
        'model': schema.ModelSettingsSchema().bind().serialize(model.to_dict()),
        'default_wind_mover': schema.WindMoverSchema().bind().serialize(),
        'default_point_release_spill': schema.PointReleaseSpillSchema().bind().serialize(),
        'default_wind': default_wind_value
    }

    html = render('model_forms.mak', context, request)
    spill_forms = render_obj_forms(request, get_spill_context, model.spills)
    mover_forms = render_obj_forms(request, get_mover_context, model.movers)

    return {
        'html': '%s %s %s' % (html, spill_forms, mover_forms)
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

    data['map_bounds'] = []
    if model.map and model.map.map_bounds.any():
        data['map_bounds'] = model.map.map_bounds.tolist()

    data['model'] = model
    data['model_form_html'] = model_forms(request)['html']
    data['add_mover_form_id'] = 'add_mover'
    data['add_spill_form_id'] = 'add_spill'
    data['model_forms_url'] = request.route_url('model_forms')

    if model.time_steps:
        data['background_image_url'] = _get_model_image_url(
            request, model, 'background_map.png')
        data['generated_time_steps_json'] = json.dumps(
            model.time_steps, default=util.json_encoder)
        data['expected_time_steps_json'] = json.dumps(
            model.timestamps, default=util.json_encoder)

    return data

