'''
test logging functionality
'''
import json
import os
import logging
import copy

import pytest

from gnome import initialize_log
from gnome.model import Model
from gnome.spill import point_line_release_spill


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


def test_console_only_logging():
    rm_logfile()
    c_dict = copy.deepcopy(config_dict)
    #c_dict['root']['handlers'].remove('file')
    #del c_dict['handlers']['file']
    initialize_log(c_dict, logfile)
    model = Model()
    model.spills += point_line_release_spill(10,
                                             (0, 0, 0),
                                             model.start_time,
                                             amount=1000,
                                             units='kg')

    s2 = point_line_release_spill(10,
                                  (0, 0, 0),
                                  model.start_time,
                                  amount=1000,
                                  units='kg',
                                  name='s2')
    for spill in model.spills:
        spill.set('num_released', 10)
    #model.full_run()
