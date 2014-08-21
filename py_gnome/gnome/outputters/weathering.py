'''
Weathering Outputter
'''
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import copy
import os
from glob import glob

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

    def __init__(self, output_dir='./',
                 **kwargs):
        '''
        :param str output_dir='./': output directory for geojson files

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.output_dir = output_dir

        super(WeatheringOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        super(WeatheringOutput, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        features = []
        print 'WeatheringOutput.write_output(): our spill container',
        print self.cache.load_timestep(step_num).items()
        for sc in self.cache.load_timestep(step_num).items():

            time = date_to_sec(sc.current_time_stamp)
            position = self._dataarray_p_types(sc['positions'])
            status = self._dataarray_p_types(sc['status_codes'])
            p_id = self._dataarray_p_types(sc['id'])

            all_nums = np.unique(sc['spill_num'])
            id_len = len(self.sc_pair.spill_by_index(0).id)
            spill_id = np.chararray(len(p_id,), itemsize=id_len)

            # NOTE: spill_num are not renumbered if a spill is deleted;
            # HOWEVER, if a spill is deleted, a callback in the model should
            # shrink the OrderedCollection and everything should get renumbered
            for num in all_nums:
                if not sc.uncertain:
                    spill_id[sc['spill_num'] == num] = \
                        self.sc_pair.spill_by_index(num).id
                else:
                    spill_id[sc['spill_num'] == num] = \
                        self.sc_pair.spill_by_index(num, True).id

            sc_type = 'forecast'
            if sc.uncertain:
                sc_type = 'uncertain'

            spill_id = self._dataarray_p_types(spill_id)

            for ix, pos in enumerate(position):
                st_code = oil_status._attr[oil_status._int.index(status[ix])]
                feature = Feature(geometry=Point(pos[:2]),
                                id=p_id[ix],
                                properties={'depth': pos[2],
                                    'step_num': step_num,
                                    'spill_type': sc_type,
                                    'spill_id': spill_id[ix],
                                    'current_time': time,
                                    'status_code': st_code})

                features.append(feature)

        geojson = FeatureCollection(features)
        output_filename = self.output_filename(geojson, step_num)

        # decided geojson should only be output to file
        # read data from file and send it to web client
        output_info = {'step_num': step_num,
                       #'geojson': geojson,
                       'time_stamp': sc.current_time_stamp.isoformat(),
                       'output_filename': output_filename}

        return output_info

    def output_filename(self, json_content, step_num):
        file_format = 'geojson_{0:06d}.geojson'
        filename = os.path.join(self.output_dir,
                                file_format.format(step_num))

        with open(filename, 'w') as outfile:
            dump(json_content, outfile, indent=True)

        return filename

    def clean_output_files(self):
        files = glob(os.path.join(self.output_dir, 'geojson_*.geojson'))
        for f in files:
            os.remove(f)

    def rewind(self):
        'remove previously written files'
        super(WeatheringOutput, self).rewind()
        self.clean_output_files()

    def _dataarray_p_types(self, data_array):
        '''
        return array as list with appropriate python dtype
        This is partly to make sure the dtype of the list elements is a python
        data type else geojson fails
        '''
        p_type = type(np.asscalar(data_array.dtype.type(0)))

        if p_type is long:
            'geojson expects int - it fails for a long'
            p_type = int

        return data_array.astype(p_type).tolist()
