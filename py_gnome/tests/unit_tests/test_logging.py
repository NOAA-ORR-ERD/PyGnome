'''
test logging functionality
'''
import json
import os
import logging
import copy
from datetime import timedelta

import pytest

from gnome import initialize_log
from gnome.model import Model
from gnome.environment import constant_wind, Water
from gnome.weatherers import Evaporation
from gnome.spill import point_line_release_spill
from gnome.spill.elements import floating_weathering


here = os.path.dirname(__file__)
sample_conf = os.path.join(here, 'sample_conf.json')
logfile = os.path.join(here, 'logfile.log')


config_dict = {
        "version": 1,   # way to track schema versions
        "disbale_existing_loggers": True,
        "root":
        {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "formatters":
        {
          "simple":
          {
              "format": "format=%(asctime)s - %(name)s - %(levelname)s - %(message)s",
              "datefmt": "%Y-%m-%d %H:%M:%S"
          },
          "brief":
          {
              "format": "%(levelname)-8s: %(name)-28s: %(message)s",
              "datefmt": "%Y-%m-%d %H:%M:%S"
          },
          "precise":
          {
              "format": "%(asctime)s %(name)-28s %(levelname)-8s %(message)s",
              "datefmt": "%Y-%m-%d %H:%M:%S"
          }
        },
        "handlers":
        {
          "console":
          {
              "class": "logging.StreamHandler",
              "level": "WARNING",
              "formatter": "brief"
          },
          "file":
          {
              "class": "logging.handlers.RotatingFileHandler",
              "formatter": "precise",
              "filename": "default.log",
              "maxBytes": 1000000,
              "backupCount": 3
          }
        }
    }


@pytest.fixture(scope='function')
def config_file():
    with open(sample_conf, 'w') as outfile:
        json.dump(config_dict, outfile, indent=True)

    return sample_conf


def rm_logfile():
    try:
        os.remove(logfile)
    except:
        pass
    assert not os.path.exists(logfile)


def test_conf_from_dict():
    rm_logfile()
    initialize_log(config_dict, logfile=logfile)
    logging.info('Logfile initialized from dict')
    assert os.path.exists(logfile)


def test_conf_from_file(config_file):
    rm_logfile()
    l = initialize_log(config_file, logfile=logfile)
    logging.info('Logfile initialized from file')


def test_full_run_logging():
    rm_logfile()
    c_dict = copy.deepcopy(config_dict)
    #c_dict['root']['handlers'].remove('file')
    #del c_dict['handlers']['file']
    et = floating_weathering(substance=u'ALAMO')
    initialize_log(c_dict, logfile)
    model = Model()
    model.spills += point_line_release_spill(1000,
                                             (0, 0, 0),
                                             model.start_time,
                                             end_release_time=model.start_time + timedelta(days=1),
                                             element_type=et,
                                             amount=200,
                                             units='m^3')

    #==========================================================================
    # s2 = point_line_release_spill(10,
    #                               (0, 0, 0),
    #                               model.start_time,
    #                               end_release_time=model.start_time + timedelta(days=1),
    #                               amount=1000,
    #                               element_type=et,
    #                               units='kg',
    #                               name='s2')
    #==========================================================================
    model.environment += Water()
    model.environment += constant_wind(1., 0.)
    model.weatherers += Evaporation(model.environment[-2],
                                    model.environment[-1])
    for spill in model.spills:
        spill.set('num_released', 10)

    model.full_run()
