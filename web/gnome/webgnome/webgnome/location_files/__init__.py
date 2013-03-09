import json
import logging
import os

from mako.exceptions import TopLevelLookupException
from mako.template import Template
from pyramid.exceptions import ConfigurationError
from pyramid.mako_templating import PkgResourceTemplateLookup
from webgnome import util, helpers


logger = logging.getLogger(__file__)


def get_location_file_wizard(location):
    """
    Render a wizard file for ``location`` and return the rendered HTML.
    """
    wizard_file = 'webgnome:location_files/%s/wizard.mak' % location
    lookup = PkgResourceTemplateLookup(
        directories=[wizard_file, 'webgnome:templates'])

    try:
        template = lookup.get_template(wizard_file)
        html = template.render(h=helpers)
    except (IOError, TopLevelLookupException):
        html = None

    return html


def get_location_file_config(location_file_dir, location):
    """
    Load the `config.json` file for ``location``, looking in
    ``location_file_dir`` for ``location``.

    Return the parsed JSON contents of the file if it exists and is valid JSON,
    or None.
    """
    config = os.path.join(location_file_dir, location, 'config.json')

    with open(config) as f:
        try:
            data = json.loads(f.read())
        except TypeError:
            logger.error(
                'TypeError reading conf.json file for location: %s' % config)
            return None

    data['filename'] = location
    html = get_location_file_wizard(location)

    if html:
        data['wizard_html'] = html

    return data


def get_location_file_handler(import_path, location):
    """
    Look in ``import_path`` for a ``location`` module and attempt to import
    and return a ``handle_input`` callable from it. Return None if the module
    wasn't found or the callable did not exist (or was not a callable).
    """
    try:
        return __import__('%s.wizard' % import_path,
                          fromlist=['handle_input']).handle_input
    except (ImportError, AttributeError):
        logger.debug('Did not find a wizard input handler for location '
                     'file: %s.' % location)
        return None


def includeme(config):
    """
    Configure location files.

     - Add each location file directory as a static files view.

     - Add configuration data for the location file (if provided in
       config.json) to the 'location_file_data' setting

     - Load any location-specific Pyramid configuration found in each location
       file's __init__ module.

     - Add a `handle_input` callable found in any location file's `wizard`
       module to the 'location_file_handlers' setting.
    """
    settings = config.get_settings()
    settings['location_handlers'] = {}
    settings['location_file_data'] = {}
    ignored_location_files = settings['ignored_location_files'].split(',')
    location_files = util.get_location_files(settings.location_file_dir,
                                             ignored_location_files)

    for location in location_files:
        data = get_location_file_config(settings.location_file_dir, location)

        if not data:
            continue

        settings['location_file_data'][location] = data
        logger.info('Loaded location file: %s' % location)

        location_import_path = 'webgnome.location_files.%s' % location
        config.add_static_view('static/location_file/%s' % location,
                               'webgnome.location_files:%s' % location,
                               cache_max_age=3600)
        try:
            config.include(location_import_path)
        except ConfigurationError:
            logger.debug('No extra configuration for location file: %s.'
                         % location)

        handler = get_location_file_handler(location_import_path, location)

        if handler:
            settings.location_handlers[location] = handler
