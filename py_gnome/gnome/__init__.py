"""
__init__.py for the gnome package

"""
from itertools import chain

import logging.config
import json

import unit_conversion as uc


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


def _valid_units(unit_name):
    'convenience function to get all valid units accepted by unit_conversion'
    _valid_units = uc.GetUnitNames(unit_name)
    _valid_units.extend(chain(*[val[1] for val in
                                uc.ConvertDataUnits[unit_name].values()]))
    return tuple(_valid_units)


__version__ = '0.1.1'
# a few imports so that the basic stuff is there

from gnomeobject import GnomeId, init_obj_log, AddLogger
from . import map
from . import spill
from . import spill_container
from . import movers
from . import environment
from . import model
from . import outputters
from . import multi_model_broadcast

__all__ = [GnomeId,
           map,
           spill,
           spill_container,
           movers,
           environment,
           model,
           outputters,
           initialize_log,
           AddLogger,
           multi_model_broadcast]
