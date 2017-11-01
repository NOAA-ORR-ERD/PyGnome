"""
__init__.py for the gnome package

"""
from itertools import chain

import sys
import os
import logging
import json
import warnings
import pkg_resources
import importlib

import unit_conversion as uc

from gnomeobject import GnomeId, AddLogger
# from gnomeobject import init_obj_log

# using a PEP 404 compliant version name
__version__ = '0.6.0'


# a few imports so that the basic stuff is there


def check_dependency_versions():
    '''
    Checks the versions of the following libraries:
        gridded
        oillibrary
        unit_conversion
    If the version is not at least as current as what's in the conda_requirements file,
    a warning is displayed
    '''
    def get_version(package):
        package = package.lower()
        return next((p.version for p in pkg_resources.working_set if p.project_name.lower() == package), "No match")
    libs = ['gridded', 'oil-library', 'unit-conversion']
    condafiledir = os.path.relpath(__file__).split(__file__.split('\\')[-3])[0]
    condafile = os.path.join(condafiledir, 'conda_requirements.txt')
    with open(condafile, 'r') as conda_reqs:
        for line in conda_reqs.readlines():
            for libname in libs:
                if libname in line:
                    criteria = None
                    cmp_str = None
                    if '>' in line:
                        criteria, cmp_str = (lambda a, b: a >= b, '>=') if '=' in line else (lambda a, b: a > b, '>')
                    elif '<' in line:
                        criteria, cmp_str = (lambda a, b: a <= b, '<=') if '=' in line else (lambda a, b: a < b, '<')
                    else:
                        criteria, cmp_str = (lambda a, b: a == b, '==')
                    reqd_ver = line.split('=')[-1].strip()
                    inst_ver = get_version(libname)
                    module_ver = importlib.import_module(libname.replace('-','_')).__version__
                    if not criteria(inst_ver, reqd_ver):
                        if criteria(module_ver, reqd_ver):
                            w = 'Version {0} of {1} package is reported, but actual version in module is {2}'.format(inst_ver, libname, module_ver)
                            warnings.warn(w)
                        else:
                            w = 'Version {0} of {1} package is installed in environment, {2}{3} required'.format(inst_ver, libname, cmp_str, reqd_ver)
                            warnings.warn(w)


def initialize_log(config, logfile=None):
    '''
    helper function to initialize a log - done by the application using PyGnome
    config can be a file containing json or it can be a Python dict

    :param config: logging configuration as a json file or config dict
                   it needs to be in the dict config format used by ``logging.dictConfig``:
                   https://docs.python.org/2/library/logging.config.html#logging-config-dictschema

    :param logfile=None: optional name of file to log to

    '''
    if isinstance(config, basestring):
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
    'convenience function to get all valid units accepted by unit_conversion'
    _valid_units = uc.GetUnitNames(unit_name)
    _valid_units.extend(chain(*[val[1] for val in
                                uc.ConvertDataUnits[unit_name].values()]))
    return tuple(_valid_units)


# we have a sort of chicken-egg situation here.  The above functions need
# to be defined before we can import these modules.
check_dependency_versions()
from . import (map,
                environment,
                model,
#                 multi_model_broadcast,
                spill_container,
                spill,
                movers,
                outputters
)
