'''
GeoJson outputter
Does not contain a schema for persistence yet
'''
import copy
import os
from glob import glob

import numpy as np
from geojson import Feature, FeatureCollection, dump, MultiPoint, Point
from colander import SchemaNode, String, drop, Int, Bool

from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.serializable import Serializable, Field

from .outputter import Outputter, BaseSchema


class TrajectoryGeoJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''
    round_data = SchemaNode(Bool(), missing=drop)
    round_to = SchemaNode(Int(), missing=drop)
    output_dir = SchemaNode(String(), missing=drop)


class TrajectoryGeoJsonOutput(Outputter, Serializable):
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
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state += [Field('round_data', update=True, save=True),
               Field('round_to', update=True, save=True),
               Field('output_dir', update=True, save=True)]
    _schema = TrajectoryGeoJsonSchema

    def __init__(self, round_data=True, round_to=4, output_dir=None,
                 **kwargs):
        '''
        :param bool round_data=True: if True, then round the numpy arrays
            containing float to number of digits specified by 'round_to'.
            Default is True
        :param int round_to=4: round float arrays to these number of digits.
            Default is 4.
        :param str output_dir=None: output directory for geojson files. Default
            is None since data is returned in dict for webapi. For using
            write_output_post_run(), this must be set

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.round_data = round_data
        self.round_to = round_to
        self.output_dir = output_dir

        super(TrajectoryGeoJsonOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(TrajectoryGeoJsonOutput, self).write_output(step_num,
                                                          islast_step)

        if not self._write_step:
            return None

        # one feature per element client; replaced with multipoint
        # because client performance is much more stable with one
        # feature per step rather than (n) features per step.
        features = []
        for sc in self.cache.load_timestep(step_num).items():
            sc_type = 'uncertain' if sc.uncertain else 'forecast'

            # only display lat/long for now
            lat_long = self._dataarray_p_types(sc['positions'][:, :2])
            feature = Feature(geometry=MultiPoint(lat_long),
                              id="1",
                              properties={'sc_type': sc_type})
            if sc.uncertain:
                features.insert(0, feature)
            else:
                features.append(feature)

        geojson = FeatureCollection(features)
        # default geojson should not output data to file
        # read data from file and send it to web client
        output_info = {'step_num': step_num,
                       'time_stamp': sc.current_time_stamp.isoformat(),
                       'feature_collection': geojson
                       }

        if self.output_dir:
            output_filename = self.output_to_file(geojson, step_num)
            output_info.update({'output_filename': output_filename})

        return output_info

    def output_to_file(self, json_content, step_num):
        file_format = 'geojson_{0:06d}.geojson'
        filename = os.path.join(self.output_dir,
                                file_format.format(step_num))

        with open(filename, 'w') as outfile:
            dump(json_content, outfile, indent=True)

        return filename

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
        super(TrajectoryGeoJsonOutput, self).rewind()
        self.clean_output_files()

    def clean_output_files(self):
        if self.output_dir:
            files = glob(os.path.join(self.output_dir, 'geojson_*.geojson'))
            for f in files:
                os.remove(f)


class CurrentGridGeoJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class CurrentGridGeoJsonOutput(Outputter, Serializable):
    '''
    class that outputs GNOME current velocity results in a geojson format.
    The output is a collection of Features. Each Feature contains a Point
    object with associated properties.
    Following is the output format - the data in <> are the results
    for each element.

    ::
    {
     "time_stamp": <TIME IN ISO FORMAT>,
     "step_num": <OUTPUT ASSOCIATED WITH THIS STEP NUMBER>,
     "feature_collection": {"type": "FeatureCollection",
                            "features": [{"type": "Feature",
                                          "id": <PARTICLE_ID>,
                                          "properties": {"velocity": [u, v]
                                                         },
                                          "geometry": {"type": "Point",
                                                       "coordinates": [<LONGITUDE>,
                                                                       <LATITUDE>]
                                                       },
                                          },
                                          ...
                                          ],
                            },
     }
    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _schema = CurrentGridGeoJsonSchema

    def __init__(self, current_mover, **kwargs):
        '''
        :param bool round_data=True: if True, then round the numpy arrays
            containing float to number of digits specified by 'round_to'.
            Default is True
        :param int round_to=4: round float arrays to these number of digits.
            Default is 4.
        :param str output_dir=None: output directory for geojson files. Default
            is None since data is returned in dict for webapi. For using
            write_output_post_run(), this must be set

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        super(CurrentGridGeoJsonOutput, self).__init__(**kwargs)

        self.current_mover = current_mover

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(CurrentGridGeoJsonOutput, self).write_output(step_num,
                                                           islast_step)

        if not self._write_step:
            return None

        # one feature per element client; replaced with multipoint
        # because client performance is much more stable with one
        # feature per step rather than (n) features per step.
        features = []

        for sc in self.cache.load_timestep(step_num).items():
            pass

        model_time = date_to_sec(sc.current_time_stamp)
        velocities = self.current_mover.get_scaled_velocities(model_time)
        centers = self.current_mover.mover._get_center_points()

        for v, c in zip(velocities, centers):
            feature = Feature(geometry=Point(c),
                              id="1",
                              properties={'velocity': v})
            features.append(feature)

        geojson = FeatureCollection(features)
        # default geojson should not output data to file
        # read data from file and send it to web client
        output_info = {'step_num': step_num,
                       'time_stamp': sc.current_time_stamp.isoformat(),
                       'feature_collection': geojson
                       }

        return output_info

    def rewind(self):
        'remove previously written files'
        super(CurrentGridGeoJsonOutput, self).rewind()






















