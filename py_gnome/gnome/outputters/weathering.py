'''
Weathering Outputter
'''
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import copy
import os
from glob import glob
import json
import random

import numpy as np
from geojson import Point, Feature, FeatureCollection, dump
from colander import SchemaNode, String, drop

from gnome.utilities.serializable import Serializable, Field

from .outputter import Outputter, BaseSchema

from gnome.basic_types import oil_status
from gnome.utilities.time_utils import date_to_sec


class WeatheringOutputSchema(BaseSchema):
    output_dir = SchemaNode(String(), missing=drop)


class WeatheringOutput(Outputter, Serializable):
    '''
    class that outputs GNOME weathering results.
    The output is the aggregation of properties for all LEs (aka Mass Balance)
    for a particular time step.
    There are a number of different things we would like to graph:
    - Evaporation
    - Dissolution
    - Dissipation
    - Biodegradation
    - ???

    However at this time we will simply try to implement an outputter for the
    halflife Weatherer.
    Following is the output format.

        {
        "type": "WeatheringGraphs",
        "half_life": {"properties": {"mass_components": <Component values>,
                                     "mass": <total Mass value>,
                                     }
                      },
            ...
        }

    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state += [Field('output_dir', update=True, save=True)]
    _schema = WeatheringOutputSchema

    def __init__(self,
                 output_dir=None,   # default is to not output to file
                 **kwargs):
        '''
        :param str output_dir='./': output directory for geojson files

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.output_dir = output_dir
        self.units = {'default': 'kg',
                      'avg_density': 'kg/m^3'}
        super(WeatheringOutput, self).__init__(**kwargs)

    def _add_fake_uncertainty(self, nom_dict):
        '''
        add uncertainty to 'amount_released', then compute fraction of total
        of weathered quantities
        '''
        runs = {}
        for ix in range(27):
            runs[str(ix)] = {}
            sum_mass = 0.0
            for key, val in nom_dict.iteritems():
                runs[str(ix)][key] = val * random.uniform(0.9, 1.2)
                if key not in ('avg_density', 'amount_released', 'floating'):
                    sum_mass += runs[str(ix)][key]

            runs[str(ix)]['floating'] = (runs[str(ix)]['amount_released'] -
                                         sum_mass)

        return runs

    def write_output(self, step_num, islast_step=False):
        super(WeatheringOutput, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        # return a dict - json of the weathering_data data
        for sc in self.cache.load_timestep(step_num).items():
            # Not capturing 'uncertain' info yet
            # dict_ = {'uncertain': sc.uncertain}
            dict_ = {}
            dict_['nominal'] = {}

            for key, val in sc.weathering_data.iteritems():
                dict_['nominal'][key] = val

            # add fake uncertainty
            dict_.update(self._add_fake_uncertainty(dict_['nominal']))

            output_info = {'step_num': step_num,
                           'time_stamp': sc.current_time_stamp.isoformat()}
            output_info.update(dict_)

            if self.output_dir:
                output_filename = self.output_to_file(output_info, step_num)
                output_info.update({'output_filename': output_filename})

        return output_info

    def output_to_file(self, json_content, step_num):
        file_format = 'weathering_data_{0:06d}.json'
        filename = os.path.join(self.output_dir,
                                file_format.format(step_num))

        with open(filename, 'w') as outfile:
            dump(json_content, outfile, indent=True)

        return filename

    def clean_output_files(self):
        if self.output_dir:
            files = glob(os.path.join(self.output_dir, 'weathering_data_*.json'))
            for f in files:
                os.remove(f)

    def rewind(self):
        'remove previously written files'
        super(WeatheringOutput, self).rewind()
        self.clean_output_files()

