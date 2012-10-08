import datetime

from pyramid.config import Configurator
from pyramid.renderers import JSON
from pyramid.session import UnencryptedCookieSessionFactoryConfig

from util import json_date_adapter


# TODO: Replace with Beaker.
session_factory = UnencryptedCookieSessionFactoryConfig('ibjas45u3$@#$++slkjf__22134bbb')

gnome_json = JSON(adapters=(
    (datetime.datetime, json_date_adapter),
    (datetime.date, json_date_adapter)
))


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    config = Configurator(settings=settings, session_factory=session_factory)
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('show_model', '/')
    config.add_route('run_model', '/model/run')
    config.add_renderer('gnome_json', gnome_json)
    config.scan()
    return config.make_wsgi_app()
