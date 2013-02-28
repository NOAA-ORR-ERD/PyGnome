import logging
import os

from pyramid.exceptions import ConfigurationError
from webgnome import util


logger = logging.getLogger(__file__)


def includeme(config):
    """
    Configure location files.

     - Add each location file directory as a Mako template directory.

     - Each location file may provide location-specific Pyramid configuration
       changes via the __init__ module in the location file directory.

     - Any location file with a wizard.py module that has a `handle_input`
       function will have that function added to the Registry in the
       `location_file_handlers` dictionary, using the location name as the key.
    """
    location_files_dir = os.path.dirname(__file__)
    location_files = util.dirnames(location_files_dir)

    for location in location_files:
        location_import_path = 'webgnome.location_files.%s' % location
        settings = config.get_settings()
        settings['mako.directories'].append(
            'webgnome.location_files:%s' % location)

        print settings['mako.directories']

        try:
            config.include(location_import_path)
        except ConfigurationError as e:
            logger.debug('Did not find extra configuration for location file: '
                        '%s. Caught ConfigurationError: %s)' % (location, e))
        try:
            handler = __import__('%s.wizard' % location_import_path,
                                 fromlist=['handle_input']).handle_input
        except ImportError as e:
            logger.debug('Did not find a wizard input handler for location '
                        'file: %s. Caught ImportError: %s' % (location, e))
            continue
        else:
            settings.location_handlers[location] = handler

