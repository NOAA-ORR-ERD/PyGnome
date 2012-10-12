import datetime
import uuid

from pyramid.config import Configurator
from pyramid.renderers import JSON
from pyramid.session import UnencryptedCookieSessionFactoryConfig

from mock_model import ModelManager
from util import json_date_adapter


# TODO: Replace with Beaker.
session_factory = UnencryptedCookieSessionFactoryConfig('ibjas45u3$@#$++slkjf__22134bbb')

gnome_json = JSON(adapters=(
    (datetime.datetime, json_date_adapter),
    (datetime.date, json_date_adapter),
    (uuid.UUID, lambda obj, request: str(obj))
))


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    settings['running_models'] = ModelManager()
    config = Configurator(settings=settings, session_factory=session_factory)
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('get_tree', '/tree')
    config.add_route('show_model', '/')
    config.add_route('run_model', '/model/run')
    config.add_route('add_mover', '/model/mover/add')
    config.add_route('delete_mover', '/model/mover/delete')
    config.add_route('add_constant_wind_mover', '/model/mover/constant_wind/add')
    config.add_route('add_variable_wind_mover', '/model/mover/variable_wind/add')
    config.add_route('edit_constant_wind_mover', '/model/mover/constant_wind/edit/{id}')
    config.add_route('edit_variable_wind_mover', '/model/mover/variable_wind/edit/{id}')
    config.add_renderer('gnome_json', gnome_json)
    config.scan()
    return config.make_wsgi_app()
