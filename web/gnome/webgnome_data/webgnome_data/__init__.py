"""Main entry point
"""
from pyramid.config import Configurator

from webgnome.util import json_date_adapter, gnome_json


def main(global_config, **settings):
    config = Configurator(settings=settings)

    config.scan("webgnome_data.views")

    return config.make_wsgi_app()
