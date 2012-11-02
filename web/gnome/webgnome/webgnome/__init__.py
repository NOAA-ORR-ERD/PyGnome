import datetime
import uuid

from pyramid.config import Configurator
from pyramid.renderers import JSON
from pyramid.session import UnencryptedCookieSessionFactoryConfig

from model_manager import ModelManager
from util import json_date_adapter


# TODO: Replace with Beaker.
session_factory = UnencryptedCookieSessionFactoryConfig('ibjas45u3$@#$++slkjf__22134bbb')

gnome_json = JSON(adapters=(
    (datetime.datetime, json_date_adapter),
    (datetime.date, json_date_adapter),
    (uuid.UUID, lambda obj, request: str(obj))
))


def main(global_config, **settings):
    """
    This function returns a Pyramid WSGI application.
    """
    settings['Model'] = ModelManager()
    config = Configurator(settings=settings, session_factory=session_factory)
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_renderer('gnome_json', gnome_json)

    config.add_route('show_model', '/')
    config.add_route('create_model', '/model/create')
    config.add_route('get_time_steps', '/model/time_steps')
    config.add_route('get_next_step', '/model/next_step')
    config.add_route('get_tree', '/tree')
    config.add_route('run_model', '/model/run')
    config.add_route('run_model_until', '/model/run_until')
    config.add_route('model_settings', '/model/settings')
    config.add_route('model_map', '/model/map')

    config.add_route('delete_mover', '/model/mover/delete')
    config.add_route('add_wind_mover', '/model/mover/wind')
    config.add_route('edit_wind_mover', '/model/mover/wind/{id}')
    config.add_route('add_constant_wind_mover', '/model/mover/constant_wind')
    config.add_route('add_variable_wind_mover', '/model/mover/variable_wind')
    config.add_route('edit_constant_wind_mover', '/model/mover/constant_wind/{id}')
    config.add_route('edit_variable_wind_mover', '/model/mover/variable_wind/{id}')

    config.scan()
    return config.make_wsgi_app()
