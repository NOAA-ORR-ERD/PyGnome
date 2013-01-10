from pyramid.view import view_config

from webgnome import schema
from webgnome import util


@view_config(route_name='show_model', renderer='model.mak')
def show_model(request):
    """
    The entry-point for the web application.

    Get an existing :class:`gnome.model.Model` using the ``model_id`` field
    in the user's session or create a new one.

    If ``model_id`` was found in the user's session but the model did not
    exist, warn the user and suggest that they reload from a save file.

    Render all forms, JSON and HTML needed to load the JavaScript app on the
    model page.
    """
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    model_images_dir = request.registry.settings['model_images_dir']
    model, created = settings.Model.get_or_create(
        model_id, model_images_dir=model_images_dir)
    model_dict = model.to_dict()
    point_release_spills = model_dict.pop('point_release_spills')
    wind_movers = model_dict.pop('wind_movers')
    model_settings = util.SchemaForm(schema.ModelSettingsSchema, model_dict)
    default_wind_mover = util.SchemaForm(schema.WindMoverSchema)
    default_wind = util.SchemaForm(schema.WindSchema)
    default_wind_value = util.SchemaForm(schema.WindValueSchema)
    default_point_release_spill = util.SchemaForm(schema.PointReleaseSpillSchema)

    data = {
       'form_view_container_id': 'modal-container',
        'model': model_settings,
        'default_wind_mover': default_wind_mover,
        'default_point_release_spill': default_point_release_spill,
        'default_wind': default_wind,
        'default_wind_value': default_wind_value
    }

    if created:
        request.session[settings.model_session_key] = model.id
        if model_id:
            data['warning'] = 'The model you were working on is no longer ' \
                              'available. We created a new one for you.'

    data['map_bounds'] = []
    if model.map and model.map.map_bounds.any():
        data['map_bounds'] = model.map.map_bounds.tolist()

    data['model_id'] = model.id
    data['current_time_step'] = model.current_time_step
    data['point_release_spills'] = util.to_json(point_release_spills)
    data['wind_movers'] = util.to_json(wind_movers)
    data['model_settings'] = util.to_json(model_dict)
    data['model_forms_url'] = request.route_url('model_forms')

    if model.background_image:
        data['background_image_url'] = util.get_model_image_url(
            request, model, 'background_map.png')

    if model.time_steps:
        data['generated_time_steps_json'] = util.to_json(model.time_steps)
        data['expected_time_steps_json'] = util.to_json(model.timestamps)

    return data

