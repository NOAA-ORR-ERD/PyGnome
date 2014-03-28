import hammer
import json

from pyramid.view import view_config

from gnome.outputters import RendererSchema

from webgnome import schema
from webgnome import util


def _default_schema_json(schema_cls):
    """
    Return a JSON object that contains default values for ``schema_cls``.
    """
    return json.dumps(schema_cls().bind().serialize(),
                      default=util.json_encoder)


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
    # A flag that when true means the user just deleted their prior model.
    deleted = request.cookies.get('model_deleted', False)
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    model, created = settings.Model.get_or_create(model_id)
    model_schema = schema.ModelSchema().bind()
    model_data = model_schema.serialize(model.to_dict())
    model_json_schema = json.dumps(
        hammer.to_json_schema(model_schema, draft_version=3))
    surface_release_spills = model_data.pop('surface_release_spills')
    wind_movers = model_data.pop('wind_movers')
    winds = model_data.pop('winds')
    random_movers = model_data.pop('random_movers')
    map_data = model_data.get('map', None)
    renderer = None

    if model.renderer:
        renderer = RendererSchema().bind().serialize(
            model.renderer.to_dict(do='create'))

        if map_data:
            map_data['background_image_url'] = util.get_model_image_url(
                request, model, model.renderer.background_map_name)

    # JavaScript model default values, used in "Add [object]" types of forms.
    default_wind_mover = _default_schema_json(schema.WindMoverSchema)
    default_wind = _default_schema_json(schema.WindSchema)
    default_random_mover = _default_schema_json(schema.RandomMoverSchema)
    default_surface_release_spill = _default_schema_json(
        schema.PointSourceReleaseSchema)
    default_map = _default_schema_json(schema.MapSchema)
    default_custom_map = _default_schema_json(schema.CustomMapSchema)

    data = {
        'model_id': model.id,
        'created': created,
        'map_is_loaded': True if model.map else False,
        'current_time_step': model.current_time_step,
        'json_schema': model_json_schema,
        'renderer_data': util.to_json(renderer),

        # Default values for forms that use them.
        'default_wind_mover': default_wind_mover,
        'default_surface_release_spill': default_surface_release_spill,
        'default_wind': default_wind,
        'default_random_mover': default_random_mover,
        'default_map': default_map,
        'default_custom_map': default_custom_map,

        # JSON data to bootstrap the JS application.
        'map_data': util.to_json(map_data),
        'surface_release_spills': util.to_json(surface_release_spills),
        'wind_movers': util.to_json(wind_movers),
        'random_movers': util.to_json(random_movers),
        'winds': util.to_json(winds),
        'model_settings': util.to_json(model_data),
        'location_files': sorted(
            request.registry.settings.location_file_data.values(),
            key=lambda location_file: location_file['name']),
        'location_file_json': json.dumps(
            request.registry.settings.location_file_data.values())
    }

    if created:
        request.session[settings.model_session_key] = model.id

        if model_id and not deleted:
            data['warning'] = 'The model you were working on is no longer ' \
                              'available. We created a new one for you.'

    if model.time_steps:
        data['generated_time_steps_json'] = util.to_json(model.time_steps)
        data['expected_time_steps_json'] = util.to_json(model.timestamps)

    return data
