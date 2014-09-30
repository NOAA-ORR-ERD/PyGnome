"""
__init__.py for the gnome package

"""
import logging.config
import json


def initialize_log(config, filename=None):
    log_config = json.load(open(config, 'r'))
    if filename is not None:
        log_config['handlers']['file']['filename'] = filename

    logging.config.dictConfig(log_config)


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
