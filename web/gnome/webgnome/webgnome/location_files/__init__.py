import logging
import os

from pyramid.exceptions import ConfigurationError
from webgnome import util


logger = logging.getLogger(__file__)


def includeme(config):
    """
    Configure location files.

     - Add each location file directory as a Mako template directory.

     - Add each location file directory as a static files view.

     - Each location file may provide location-specific Pyramid configuration
       changes via the __init__ module in the location file directory.

     - Any location file with a wizard.py module that has a `handle_input`
       function will have that function added to the Registry in the
       `location_file_handlers` dictionary, using the location name as the key.
    """
    settings = config.get_settings()
    location_files = util.get_location_files(settings.location_file_dir,
                                             settings.ignored_location_files)

    for location in location_files:
        location_import_path = 'webgnome.location_files.%s' % location
        config.add_static_view('static/location/%s' % location,
                               'webgnome.location_files:%s' % location,
                               cache_max_age=3600)
        try:
            config.include(location_import_path)
        except ConfigurationError:
            logger.debug('No extra configuration for location file: %s.'
                         % location)
        try:
            handler = __import__('%s.wizard' % location_import_path,
                                 fromlist=['handle_input']).handle_input
        except (ImportError, AttributeError):
            logger.debug('Did not find a wizard input handler for location '
                        'file: %s.' % location)
            continue
        else:
            settings.location_handlers[location] = handler

