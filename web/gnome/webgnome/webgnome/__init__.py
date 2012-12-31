import os

from pyramid.config import Configurator
from pyramid.events import BeforeRender
from pyramid.session import UnencryptedCookieSessionFactoryConfig

from webgnome import helpers
from webgnome.model_manager import ModelManager, WebWindMover, WebPointReleaseSpill
from webgnome.util import json_date_adapter, gnome_json


# TODO: Replace with Beaker.
session_factory = UnencryptedCookieSessionFactoryConfig('ibjas45u3$@#$++slkjf__22134bbb')


def add_renderer_globals(event):
    """
    Add an ``h`` variable accessible inside of Mako templates that makes
    WebHelper tag generation functions available.
    """
    event['h'] = helpers


def main(global_config, **settings):
    """
    This function returns a Pyramid WSGI application.
    """
    settings['Model'] = ModelManager()

    settings['package_root'] = os.path.abspath(os.path.dirname(__file__))
    settings['project_root'] = os.path.dirname(settings['package_root'])
    settings['model_images_url_path'] = 'img/%s' % settings['model_images_dir']
    settings['model_images_dir'] = os.path.join(
        settings['package_root'], 'static', 'img', settings['model_images_dir'])

    # Create the output directory if it does not exist.
    if not os.path.isdir(settings['model_images_dir']):
        os.mkdir(settings['model_images_dir'])

    config = Configurator(settings=settings, session_factory=session_factory)
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_renderer('gnome_json', gnome_json)

    config.add_route('show_model', '/')
    config.add_route('model_forms', 'model/forms')
    config.add_route('create_model', '/model/create')
    config.add_route('get_time_steps', '/model/time_steps')
    config.add_route('run_model', '/model/run')
    config.add_route('run_model_until', '/model/run_until')
    config.add_route('model_map', '/model/map')

    config.include("cornice")

    config.add_subscriber(add_renderer_globals, BeforeRender)

    config.scan()

    return config.make_wsgi_app()
