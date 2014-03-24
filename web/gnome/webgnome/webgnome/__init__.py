import os

from pyramid.config import Configurator
from pyramid.events import BeforeRender
from pyramid.session import UnencryptedCookieSessionFactoryConfig

from webgnome import helpers
from webgnome.model_manager import (
    ModelManager,
    WebWindMover,
    WebPointSourceRelease
)
from webgnome.util import json_date_adapter, gnome_json


# TODO: Replace with Beaker.
session_factory = UnencryptedCookieSessionFactoryConfig(
    'ibjas45u3$@#$++slkjf__22134bbb')


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
    settings['package_root'] = os.path.abspath(os.path.dirname(__file__))
    settings['project_root'] = os.path.dirname(settings['package_root'])
    settings['location_file_dir'] = os.path.join(settings['package_root'],
                                                 'location_files')
    settings['data_dir'] = os.path.join(settings['package_root'], 'data')
    settings['model_images_url_path'] = settings['model_data_dir']
    settings['model_data_dir'] = os.path.join(settings['package_root'],
                                              'static',
                                              settings['model_data_dir'])

    settings['Model'] = ModelManager(data_dir=settings['model_data_dir'],
                                     package_root=settings['package_root'])

    mako_dirs = settings['mako.directories']
    settings['mako.directories'] = mako_dirs if type(
        mako_dirs) is list else [mako_dirs]

    # Create the output directory if it does not exist.
    if not os.path.isdir(settings['model_data_dir']):
        os.mkdir(settings['model_data_dir'])

    config = Configurator(settings=settings, session_factory=session_factory)

    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_renderer('gnome_json', gnome_json)

    config.add_route('show_model', '/')
    config.add_route('long_island', '/long_island')

    config.include("cornice")

    config.add_subscriber(add_renderer_globals, BeforeRender)

    config.scan()

    return config.make_wsgi_app()
