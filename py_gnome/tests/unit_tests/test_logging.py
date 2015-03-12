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
from conftest import test_oil


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
def config_file(dump):
    'create a file from which to load configuration'
    sample_conf = os.path.join(dump, 'sample_conf.json')
    with open(sample_conf, 'w') as outfile:
        json.dump(config_dict, outfile, indent=True)

    return sample_conf


@pytest.fixture(scope="module")
def logfile(dump):
    return os.path.join(dump, 'logfile.log')


class TestLog():
    @pytest.fixture(autouse=True)
    def rm_logfile(self, logfile):
        try:
            os.remove(logfile)
        except:
            pass
        assert not os.path.exists(logfile)

    def test_conf_from_dict(self, logfile):
        initialize_log(config_dict, logfile=logfile)
        logging.info('Logfile initialized from dict')
        assert os.path.exists(logfile)

    def test_conf_from_file(self, config_file, logfile):
        initialize_log(config_file, logfile=logfile)
        logging.info('Logfile initialized from file')

    def test_full_run_logging(self, logfile):
        c_dict = copy.deepcopy(config_dict)
        c_dict["root"]["level"] = "DEBUG"
        initialize_log(c_dict, logfile)
        #model = Model(time_step=timedelta(minutes=15))
        model = Model(uncertain=True)
        model.spills += point_line_release_spill(100,
                                                 (0, 0, 0),
                                                 model.start_time,
                                                 end_release_time=model.start_time + timedelta(days=1),
                                                 amount=200,
                                                 units='m^3',
                                                 substance=test_oil)
        model.water = Water()
        model.environment += constant_wind(1., 0.)
        model.weatherers += Evaporation(model.water,
                                        model.environment[-1])
        model.full_run()

        # log warning - visually check logged warnging to ensure message logged
        # after deepcopy
        for spill in model.spills:
            spill.set('num_released', 10)
