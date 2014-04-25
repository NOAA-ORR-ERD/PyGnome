'''
GeoJson outputter
Does not contain a schema for persistence yet
'''
import copy
import os

import numpy as np
from geojson import Point, Feature, FeatureCollection, dump

from gnome.utilities.serializable import Serializable

from .outputter import Outputter

from gnome.basic_types import oil_status
from gnome import array_types as at
from gnome.utilities.time_utils import date_to_sec


class GeoJson(Outputter, Serializable):
    '''
    class that outputs GNOME results in a geojson format. The output is a
    collection of Features. Each Feature contains a Point object with
    associated properties. Following is the format for a particle - the
    data in <> are the results for each element.
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
            "properties": {
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
    _state = copy.deepcopy(Outputter._state)

    def __init__(self,
        round_data=True,
        roundto=4,
        output_dir='./',
        **kwargs):
        '''
        :param bool round_data=True: if True, then round the numpy arrays
            containing float to number of digits specified by 'roundto'.
            Default is True
        :param int roundto=4: round float arrays to these number of digits.
            Default is 4.
        :param str output_dir='./': output directory for geojson files

        use super to pass optional **kwargs to base class __init__ method
        '''
        self.round_data = round_data
        self.roundto = roundto
        self.output_dir = output_dir
        super(GeoJson, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(GeoJson, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        features = []
        for sc in self.cache.load_timestep(step_num).items():
            sc_type = 'forecast'
            if sc.uncertain:
                sc_type = 'uncertain'

            time = date_to_sec(sc.current_time_stamp)
            position = self._dataarray_p_types(sc, 'positions')
            status = self._dataarray_p_types(sc, 'status_codes')
            p_id = self._dataarray_p_types(sc, 'id')
            for ix, pos in enumerate(position):
                st_code = oil_status._attr[oil_status._int == status]
                feature = Feature(geometry=Point(pos[:2]),
                                id=p_id[ix],
                                properties={'depth': pos[2],
                                    'step_num': step_num,
                                    'spill_type': sc_type,
                                    'spill_id': sc['spill_id'][ix].tostring(),
                                    'current_time': time,
                                    'status_code': st_code})

                features.append(feature)

        geojson = FeatureCollection(features)
        output_filename = os.path.join(self.output_dir,
                                    self.outputfile_format % step_num)
        with open(output_filename, 'w') as outfile:
            dump(geojson, outfile, indent=True)

        output_info = {'step_num': step_num,
                       'geojson': geojson,
                       'time_stamp': sc.current_time_stamp,
                       'output_filename': output_filename}

        return output_info

    def _dataarray_p_types(self, sc, name):
        '''
        return array as list with appropriate python dtype
        This is partly to make sure the dtype of the list elements is a python
        data type else geojson fails
        '''
        try:
            # assume dtype for the array_type is a numpy dtype
            p_type = type(np.asscalar(getattr(at, name).dtype(0)))
        except AttributeError:
            # the dtype for the array_type is already a python type
            p_type = type(getattr(at, name).dtype(0))

        if p_type is long:
            'geojson expects int - it fails for a long'
            p_type = int

        if p_type is float and self.round_data:
            data = sc[name].round(self.roundto).astype(p_type).tolist()
        else:
            data = sc[name].astype(p_type).tolist()
        return data
