import datetime
import os
import gnome
import gnome.basic_types
import gnome.utilities.map_canvas
from gnome.weather import Wind
from hazpy.file_tools import haz_files
import numpy

from pyramid.view import view_config

from webgnome import schema, WebPointReleaseSpill, WebWindMover
from webgnome import util
from webgnome.model_manager import WebMapFromBNA


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
    model_data = model.to_dict()
    point_release_spills = model_data.pop('point_release_spills')
    wind_movers = model_data.pop('wind_movers')
    map_data = model_data.pop('map')
    model_settings = util.SchemaForm(schema.ModelSettingsSchema, model_data)
    map_settings = util.SchemaForm(schema.MapSchema, map_data)
    default_wind_mover = util.SchemaForm(schema.WindMoverSchema)
    default_wind = util.SchemaForm(schema.WindSchema)
    default_wind_value = util.SchemaForm(schema.WindValueSchema)
    default_point_release_spill = util.SchemaForm(schema.PointReleaseSpillSchema)

    data = {
        'model': model_settings,
        'model_id': model.id,
        '_map': map_settings,
        'map_bounds': [],
        'current_time_step': model.current_time_step,

        # Default values for forms that use them.
        'default_wind_mover': default_wind_mover,
        'default_point_release_spill': default_point_release_spill,
        'default_wind': default_wind,
        'default_wind_value': default_wind_value,

        # JSON data to bootstrap the JS application.
        'map_data': util.to_json(map_data),
        'point_release_spills': util.to_json(point_release_spills),
        'wind_movers': util.to_json(wind_movers),
        'model_settings': util.to_json(model_data)
    }

    if created:
        request.session[settings.model_session_key] = model.id
        if model_id:
            data['warning'] = 'The model you were working on is no longer ' \
                              'available. We created a new one for you.'

    if model.map and model.map.map_bounds.any():
        data['map_bounds'] = model.map.map_bounds.tolist()

    if model.background_image:
        data['background_image_url'] = util.get_model_image_url(
            request, model, 'background_map.png')

    if model.time_steps:
        data['generated_time_steps_json'] = util.to_json(model.time_steps)
        data['expected_time_steps_json'] = util.to_json(model.timestamps)

    return data


@view_config(route_name='long_island', renderer='gnome_json')
@util.require_model
def configure_long_island(request, model):
    """
    Configure the user's current model with the parameters of the Long Island
    script.
    """
    spill = WebPointReleaseSpill(
        name="Long Island Spill",
        num_LEs=1000,
        start_position=(-72.419992, 41.202120, 0.0),
        release_time=model.start_time)

    model.add_spill(spill)

    start_time = model.start_time

    r_mover = gnome.movers.RandomMover(diffusion_coef=500000)
    model.add_mover(r_mover)

    series = numpy.zeros((5,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, (30, 50) )
    series[1] = (start_time + datetime.timedelta(hours=18), (30, 50))
    series[2] = (start_time + datetime.timedelta(hours=30), (20, 25))
    series[3] = (start_time + datetime.timedelta(hours=42), (25, 10))
    series[4] = (start_time + datetime.timedelta(hours=54), (25, 180))

    wind = Wind(units='mps', timeseries=series)
    w_mover = WebWindMover(wind=wind, is_constant=False)
    model.add_mover(w_mover)

    map_file = os.path.join(
        request.registry.settings['project_root'],
        'sample_data',
        'LongIslandSoundMap.BNA')

    # the land-water map
    model.map = WebMapFromBNA(
        map_file, refloat_halflife=6 * 3600, name="Long Island Sound")

    canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
    polygons = haz_files.ReadBNA(map_file, "PolygonSet")
    canvas.set_land(polygons)
    model.output_map = canvas

    return {
        'success': True
    }
