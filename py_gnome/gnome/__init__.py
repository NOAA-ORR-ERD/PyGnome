"""
__init__.py for the gnome package

"""
import logging.config
import json


def initialize_log(config, logfile=None):
    '''
    helper function to initialize a log - done by the application using PyGnome
    config can be a file containing json or it can be a Python dict
    '''
    if isinstance(config, basestring):
        config = json.load(open(config, 'r'))

    if logfile is not None:
        config['handlers']['file']['filename'] = logfile

    logging.config.dictConfig(config)


__version__ = '0.1.1'
# a few imports so that the basic stuff is there

from gnomeobject import GnomeId, init_obj_log
from . import map
from . import spill
from . import spill_container
from . import movers
from . import environment
from . import model
from . import outputters

__all__ = [GnomeId,
           map,
           spill,
           spill_container,
           movers,
           environment,
           model,
           outputters,
           initialize_log]
