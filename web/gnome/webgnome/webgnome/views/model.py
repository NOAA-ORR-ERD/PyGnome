import datetime
import json
import os
import gnome
import gnome.basic_types
import gnome.utilities.map_canvas
import numpy

from hazpy.file_tools import haz_files
from pyramid.view import view_config
from webgnome import schema
from webgnome import util
from webgnome.model_manager import (
    WebMapFromBNA,
    WebRandomMover,
    WebSurfaceReleaseSpill,
    WebWindMover,
    WebWind
)


def _default_schema_json(schema_cls):
    """
    Return a JSON object that contains default values for ``schema_cls``.
    """
    return json.dumps(schema_cls().bind().serialize(), default=util.json_encoder)


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
    deleted = request.cookies.get('model_deleted', False)
    settings = request.registry.settings
    model_id = request.session.get(settings.model_session_key, None)
    model_data_dir = request.registry.settings['model_data_dir']
    model, created = settings.Model.get_or_create(
        model_id, model_data_dir=model_data_dir)
    model_data = model.to_dict()
    surface_release_spills = model_data.pop('surface_release_spills')
    wind_movers = model_data.pop('wind_movers')
    random_movers = model_data.pop('random_movers')
    model_settings = util.SchemaForm(schema.ModelSchema, model_data)
    map_data = model_data['map']

    # JSON defaults for initializing JavaScript models
    default_wind_mover = _default_schema_json(schema.WindMoverSchema)
    default_wind_timeseries_value = _default_schema_json(schema.TimeseriesValueSchema)
    default_random_mover = _default_schema_json(schema.RandomMoverSchema)
    default_surface_release_spill = _default_schema_json(
        schema.SurfaceReleaseSpillSchema)
    default_map = _default_schema_json(schema.MapSchema)
    default_custom_map = _default_schema_json(schema.CustomMapSchema)

    if map_data and model.background_image:
        map_data['background_image_url'] = util.get_model_image_url(
            request, model, model.background_image)

    data = {
        'model': model_settings,
        'model_id': model.id,
        'created': created,
        'map_bounds': [],
        'map_is_loaded': False,
        'current_time_step': model.current_time_step,

        # Default values for forms that use them.
        'default_wind_mover': default_wind_mover,
        'default_surface_release_spill': default_surface_release_spill,
        'default_wind_timeseries_value': default_wind_timeseries_value,
        'default_random_mover': default_random_mover,
        'default_map': default_map,
        'default_custom_map': default_custom_map,
        'location_files': json.dumps(
            request.registry.settings.location_file_data),

        # JSON data to bootstrap the JS application.
        'map_data': util.to_json(map_data),
        'surface_release_spills': util.to_json(surface_release_spills),
        'wind_movers': util.to_json(wind_movers),
        'random_movers': util.to_json(random_movers),
        'model_settings': util.to_json(model_data)
    }

    if created:
        request.session[settings.model_session_key] = model.id

        if model_id and not deleted:
            data['warning'] = 'The model you were working on is no longer ' \
                              'available. We created a new one for you.'

    if model.map:
        data['map_is_loaded'] = True

        if model.map.map_bounds.any():
            data['map_bounds'] = model.map.map_bounds.tolist()

    if model.time_steps:
        data['generated_time_steps_json'] = util.to_json(model.time_steps)
        data['expected_time_steps_json'] = util.to_json(model.timestamps)

    return data


@view_config(route_name='long_island_manual', renderer='gnome_json')
@util.require_model
def configure_long_island(request, model):
    """
    Configure the user's current model with the parameters of the Long Island
    script.

    XXX: Deprecated, but still useful to generate the Long Island save-file
    for testing. At least until we have a better way to create that file ...
    """
    spill = WebSurfaceReleaseSpill(
        name="Long Island Spill",
        num_elements=1000,
        start_position=(-72.419992, 41.202120, 0.0),
        release_time=model.start_time)

    model.spills.add(spill)

    start_time = model.start_time

    r_mover = WebRandomMover(diffusion_coef=500000)
    model.movers.add(r_mover)

    series = numpy.zeros((5,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, (30, 50) )
    series[1] = (start_time + datetime.timedelta(hours=18), (30, 50))
    series[2] = (start_time + datetime.timedelta(hours=30), (20, 25))
    series[3] = (start_time + datetime.timedelta(hours=42), (25, 10))
    series[4] = (start_time + datetime.timedelta(hours=54), (25, 180))

    wind = WebWind(units='mps', timeseries=series)
    w_mover = WebWindMover(wind=wind, is_constant=False)
    model.movers.add(w_mover)

    map_file = os.path.join(
        request.registry.settings['project_root'],
        'sample_data',
        'LongIslandSoundMap.BNA')

    # the land-water map
    model.map = WebMapFromBNA(
        map_file, refloat_halflife=6 * 3600, name="Long Island Sound")

    model.uncertain = False

    canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
    polygons = haz_files.ReadBNA(map_file, "PolygonSet")
    canvas.set_land(polygons)
    model.output_map = canvas

    # Save the background image.
    model.output_map.draw_background()
    model.output_map.save_background(model.background_image)

    return {
        'success': True
    }
