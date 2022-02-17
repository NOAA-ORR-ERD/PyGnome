"""
__init__.py for the gnome package

import various names, and provides:

initialize_console_log(level='debug')

  set up the logger to dump to console.


"""

from itertools import chain

import sys

import logging
import json
import warnings

import importlib

import nucos as uc

# just so it will be in the namespace.
from .gnomeobject import GnomeId, AddLogger
# from gnomeobject import init_obj_log

__version__ = '1.1.2'


# a few imports so that the basic stuff is there

def check_dependency_versions():
    """
    Checks the versions of the following libraries:

    These are checked, as they are maintained by NOAA ERD, so may be installed
    from source, rather than managed by conda, etc.
        gridded
        oillibrary
        unit_conversion
        py_gd
        adios_db
    If the version is not at least as current as what's defined here
    a warning is displayed
    """
    libs = [('gridded', '0.3.0', ''),
            ('nucos', '3.0.0', ''),
            ('py_gd', '0.1.7', ''),
            ('adios_db', '1.0.0', 'Only required to use the ADIOS Database '
                                  'JSON format for oil data.')
            ]

    for name, version, note in libs:
        # import the lib:
        try:
            module = importlib.import_module(name)
        except ImportError:
            msg = ("ERROR: The {} package, version >= {} "
                   "needs to be installed: {}".format(name, version, note))
            warnings.warn(msg)
        else:
            ver = ".".join(module.__version__.split(".")[:3])
            if ver < version:
                msg = ('Version {0} of {1} package is required, '
                       'but actual version in module is {2}:'
                       '{3}'
                       .format(version, name, module.__version__, note))
                warnings.warn(msg)


def initialize_log(config, logfile=None):
    '''
    helper function to initialize a log - done by the application using PyGnome
    config can be a file containing json or it can be a Python dict

    :param config: logging configuration as a json file or config dict
                   it needs to be in the dict config format used by
                   ``logging.dictConfig``:
                   https://docs.python.org/2/library/logging.config.html#logging-config-dictschema

    :param logfile=None: optional name of file to log to

    '''
    if isinstance(config, str):
        config = json.load(open(config, 'r'))

    if logfile is not None:
        config['handlers']['file']['filename'] = logfile

    logging.config.dictConfig(config)


def initialize_console_log(level='debug'):
    '''
    Initializes the logger to simply log everything to the console (stdout)

    Likely what you want for scripting use

    :param level='debug': the level you want your log to show. options are,
                          in order of importance: "debug", "info", "warning",
                          "error", "critical"

    You will only get the logging messages at or above the level you set.

    '''
    levels = {"debug": logging.DEBUG,
              "info": logging.INFO,
              "warning": logging.WARNING,
              "error": logging.ERROR,
              "critical": logging.CRITICAL,
              }
    level = levels[level.lower()]
    format_str = '%(levelname)s - %(module)8s - line:%(lineno)d - %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=level,
                        format=format_str,
                        )


def _valid_units(unit_name):
    # fixme: I think there is something built in to nucos for this
    #        or there should be
    'convenience function to get all valid units accepted by unit_conversion'
    _valid_units = list(uc.GetUnitNames(unit_name))
    _valid_units.extend(chain(*[val[1] for val in
                                uc.ConvertDataUnits[unit_name].values()]))
    return tuple(_valid_units)


# we have a sort of chicken-egg situation here.  The above functions need
# to be defined before we can import these modules.
check_dependency_versions()

from . import (environment,
               model,
               # multi_model_broadcast,
               spills,
               movers,
               outputters)

from .maps import map
