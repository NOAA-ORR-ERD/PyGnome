'''
GeoJson outputter
Does not contain a schema for persistence yet
'''
import copy
import os
from glob import glob

import numpy as np
from geojson import Point, Feature, FeatureCollection, dump
from colander import SchemaNode, String, drop, Int, Bool

from gnome.utilities.serializable import Serializable, Field

from .outputter import Outputter, BaseSchema

from gnome.basic_types import oil_status
from gnome import array_types
from gnome.utilities.time_utils import date_to_sec


class GeoJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''
    round_data = SchemaNode(Bool(), missing=drop)
    round_to = SchemaNode(Int(), missing=drop)
    output_dir = SchemaNode(String(), missing=drop)


class GeoJson(Outputter, Serializable):
    '''
    class that outputs GNOME results in a geojson format. The output is a
    collection of Features. Each Feature contains a Point object with
    associated properties. Following is the format for a particle - the
    data in <> are the results for each element.

    ::

        {
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        <LONGITUDE>,
                        <LATITUDE>
                    ]
                },
                "type": "Feature",
                "id": <PARTICLE_ID>,
                "properties": {::
                    "current_time": <TIME IN SEC SINCE EPOCH>,
                    "status_code": <>,
                    "spill_id": <UUID OF SPILL OBJECT THAT RELEASED PARTICLE>,
                    "depth": <DEPTH>,
                    "spill_type": <FORECAST OR UNCERTAIN>,
                    "step_num": <OUTPUT ASSOCIATED WITH THIS STEP NUMBER>
                }
            },
            ...
        }

    '''

    outputfile_format = 'geojson_%05i.geojson'
    outputfile_glob = 'geojson_*.geojson'
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state += [Field('round_data', update=True, save=True),
               Field('round_to', update=True, save=True),
               Field('output_dir', update=True, save=True)]

    def __init__(self,
        round_data=True,
        round_to=4,
        output_dir='./',
        **kwargs):
        '''
        :param bool round_data=True: if True, then round the numpy arrays
            containing float to number of digits specified by 'round_to'.
            Default is True
        :param int round_to=4: round float arrays to these number of digits.
            Default is 4.
        :param str output_dir='./': output directory for geojson files

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.round_data = round_data
        self.round_to = round_to
        self.output_dir = output_dir
        super(GeoJson, self).__init__(**kwargs)

    def prepare_for_model_run(self, model_start_time, spills, **kwargs):
        '''
        geo_json outputter also requires spills to be passed in - this is
        because it needs to match the 'spill_num' from the data array to the
        spill object's ID. The keyword, spills is the SpillContainerPair object
        '''
        self.sc_pair = spills
        super(GeoJson, self).prepare_for_model_run(model_start_time, **kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(GeoJson, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        features = []
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
        output_filename = os.path.join(self.output_dir,
                                    self.outputfile_format % step_num)
        with open(output_filename, 'w') as outfile:
            dump(geojson, outfile, indent=True)

        # decided geojson should only be output to file
        # read data from file and send it to web client
        output_info = {'step_num': step_num,
                       #'geojson': geojson,
                       'time_stamp': sc.current_time_stamp,
                       'output_filename': output_filename}

        return output_info

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

        if p_type is float and self.round_data:
            data = data_array.round(self.round_to).astype(p_type).tolist()
        else:
            data = data_array.astype(p_type).tolist()
        return data

    def rewind(self):
        'remove previously written files'
        super(GeoJson, self).rewind()
        files = glob(os.path.join(self.output_dir, self.outputfile_glob))
        for file_ in files:
            os.remove(file_)
