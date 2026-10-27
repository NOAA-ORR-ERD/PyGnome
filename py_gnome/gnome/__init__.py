"""
__init__.py for the gnome package

import various names, and provides:

initialize_console_log(level='debug')

set up the logger to dump to console.
"""

# lots of import-just-to-have-it-there in this
# ruff: noqa: F401

import json
import logging
import os
import pathlib
import sys
import warnings

import nucos

from ._version import __version__

# just so it will be in the namespace.
from .gnomeobject import AddLogger, GnomeId

# set up to show DeprecationWarnings that come from PyGNOME
warnings.filterwarnings("default",
                        category=DeprecationWarning,
                        module="gnome.*")

if os.name == 'nt':
    # Get site-packages path from the current Python environment
    import site
    site_packages = next((p for p in site.getsitepackages() if 'site-packages' in p), None)

    if site_packages:
        dll_path = pathlib.Path(site_packages) / 'bin'
        if dll_path.exists():
            os.add_dll_directory(dll_path)

#
# A few imports so that the basic stuff is there
#

#########
# This was helpful way back in the day -- no longer needed with pixi environments, etc.
#########
# def check_dependency_versions():
#     """
#     Checks the versions of the following libraries:

#     These are checked, as they are maintained by NOAA ERD, so may be installed
#     from source, rather than managed by conda, etc.

#     ::

#         gridded
#         oillibrary
#         nucos
#         py_gd
#         adios_db

#     If the version is not at least as current as what's defined here
#     a warning is displayed
#     """
#     def ver_check(required, installed):
#         required = tuple(int(part) for part in required.split(".")[:3])
#         try:
#             installed = tuple(int(part) for part in installed.split(".")[:3])
#         except ValueError: # something is odd -- dev version, or ??
#             return False
#         if installed < required:
#             return False
#         else:
#             return True

#     libs = [('gridded', '0.8.2', ''),
#             ('nucos', '3.4.1', ''),
#             ('py_gd', '2.3.3', ''),
#             ('adios_db', '1.2.7', 'Only required to use the ADIOS Database '
#                                   'JSON format for oil data.')
#             ]

#     for name, version, note in libs:
#         # import the lib:
#         try:
#             module = importlib.import_module(name)
#         except ImportError:
#             msg = ("ERROR: The {} package, version >= {} "
#                    "needs to be installed: {}".format(name, version, note))
#             warnings.warn(msg)
#         else:
#             if not ver_check(version, module.__version__):
#                 msg = ('Version {0} of {1} package is required, '
#                        'but actual version in module is {2}:'
#                        '{3}'
#                        .format(version, name, module.__version__, note))
#                 warnings.warn(msg)


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
        with open(config, 'r', encoding="utf-8") as confile:
            config = json.load(confile)

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

initialize_console_log(level='warning')

def _valid_units(unit_type):
    """
    return all the units for a given unit type

    :param unit_type: unit type, e.g. "Mass" or "Temperature"
    :type unit_type: str

    NOTE: this is just a wrapper for nucos.get_supported_names
    """
    ## fixme: why is this in the gnome module __init__"
    ##        but maybe we should jsut call nucos directly anyway.
    return nucos.get_supported_names(unit_type)
    # 'convenience function to get all valid units accepted by nucos'
    # _valid_units = list(uc.GetUnitNames(unit_type))
    # _valid_units.extend(chain(*[val[1] for val in
    #                             uc.ConvertDataUnits[unit_type].values()]))
    # return tuple(_valid_units)


# we have a sort of chicken-egg situation here.  The above functions need
# to be defined before we can import these modules.
# FIXME: they should be defined in a utilities module or something
# to avoid this
# check_dependency_versions()

from . import (
    environment,
    model,
    movers,
    outputters,
    # multi_model_broadcast,
    spills,
)

# redundant import to make ruff happy
from .maps import map as map
