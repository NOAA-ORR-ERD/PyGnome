'''
GeoJson outputter
Does not contain a schema for persistence yet
'''
import copy
import os
from glob import glob
from collections import Iterable, defaultdict

import numpy as np

from geojson import (Feature, FeatureCollection, dump,
                     Point, MultiPoint, MultiPolygon)

from colander import SchemaNode, String, drop, Int, Bool

from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.serializable import Serializable, Field

from gnome.persist import class_from_objtype

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
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state += [Field('round_data', update=True, save=True),
               Field('round_to', update=True, save=True),
               Field('output_dir', update=True, save=True)]
    _schema = TrajectoryGeoJsonSchema

    def __init__(self,
                 round_data=True,
                 round_to=4,
                 output_dir=None,
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

        super(TrajectoryGeoJsonOutput, self).__init__(output_dir=output_dir,
                                                      **kwargs)

    def prepare_for_model_run(self, *args, **kwargs):
        """
        prepares the outputter for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define **kwargs and pass into base class

        In this case, it cleans out previous written data files

        If you want to keep them, a new output_dir should be set
        """

        super(TrajectoryGeoJsonOutput, self).prepare_for_model_run(*args,
                                                                   **kwargs)
        self.clean_output_files()

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(TrajectoryGeoJsonOutput, self).write_output(step_num,
                                                          islast_step)

        if not self._write_step:
            return None

        # one feature per element client; replaced with multipoint
        # because client performance is much more stable with one
        # feature per step rather than (n) features per step.features = []
        features = []
        for sc in self.cache.load_timestep(step_num).items():
            time = date_to_sec(sc.current_time_stamp)
            position = self._dataarray_p_types(sc['positions'])
            status = self._dataarray_p_types(sc['status_codes'])
            sc_type = 'uncertain' if sc.uncertain else 'forecast'

            # break elements into multipoint features based on their
            # status code
            #   evaporated : 10
            #   in_water : 2
            #   not_released : 0
            #   off_maps : 7
            #   on_land : 3
            #   to_be_removed : 12
            points = {}
            for ix, pos in enumerate(position):
                st_code = status[ix]

                if st_code not in points:
                    points[st_code] = []

                points[st_code].append(pos[:2])

            for k in points:
                feature = Feature(geometry=MultiPoint(points[k]), id="1",
                                  properties={'sc_type': sc_type,
                                              'status_code': k,
                                              })

                if sc.uncertain:
                    features.insert(0, feature)
                else:
                    features.append(feature)

        geojson = FeatureCollection(features)
        # default geojson should not output data to file
        # read data from file and send it to web client
        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
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

    # def rewind(self):
    #     'remove previously written files'
    #     super(TrajectoryGeoJsonOutput, self).rewind()
    #     self.clean_output_files()

    def clean_output_files(self):
        print "in clean_output_files"
        if self.output_dir:
            files = glob(os.path.join(self.output_dir, 'geojson_*.geojson'))
            print "files are:"
            print files
            for f in files:
                os.remove(f)


class CurrentGeoJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class CurrentGeoJsonOutput(Outputter, Serializable):
    '''
    Class that outputs GNOME current velocity results for each current mover
    in a geojson format.  The output is a collection of Features.
    Each Feature contains a Point object with associated properties.
    Following is the output format - the data in <> are the results
    for each element.
    ::

        {
         "time_stamp": <TIME IN ISO FORMAT>,
         "step_num": <OUTPUT ASSOCIATED WITH THIS STEP NUMBER>,
         "feature_collections": {<mover_id>: {"type": "FeatureCollection",
                                              "features": [{"type": "Feature",
                                                            "id": <PARTICLE_ID>,
                                                            "properties": {"velocity": [u, v]
                                                                           },
                                                            "geometry": {"type": "Point",
                                                                         "coordinates": [<LONG>, <LAT>]
                                                                         },
                                                        },
                                                        ...
                                                       ],
                                          },
                             ...
                             }
        }

    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state.add_field(Field('current_movers', save=True, update=True,
                           iscollection=True))

    _schema = CurrentGeoJsonSchema

    def __init__(self, current_movers, **kwargs):
        '''
        :param list current_movers: A list or collection of current grid mover
                                    objects.

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.current_movers = current_movers

        super(CurrentGeoJsonOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(CurrentGeoJsonOutput, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            model_time = date_to_sec(sc.current_time_stamp)
            iso_time = sc.current_time_stamp.isoformat()

        geojson = {}
        for cm in self.current_movers:
            centers = cm.get_center_points()

            velocities = cm.get_scaled_velocities(model_time)
            velocities = self.get_rounded_velocities(velocities)

            velocity_dict = defaultdict(list)
            for k, v in zip(velocities, centers):
                k = tuple(k)
                v = list(v)
                velocity_dict[k].append(v)

            features = []
            for v, cps in velocity_dict.items():
                feature = Feature(geometry=MultiPoint(cps),
                                  id="1",
                                  properties={'velocity': v})
                features.append(feature)

            geojson[cm.id] = FeatureCollection(features)

        # default geojson should not output data to file
        # read data from file and send it to web client
        output_info = {'time_stamp': iso_time,
                       'feature_collections': geojson
                       }

        return output_info

    def get_rounded_velocities(self, velocities):
        return np.vstack((velocities['u'].round(decimals=2),
                          velocities['v'].round(decimals=2))).T

    def get_unique_velocities(self, velocities):
        '''
        In order to make numpy perform this function fast, we will use a
        contiguous structured array using a view of a void type that
        joins the whole row into a single item.
        '''
        dtype = np.dtype((np.void,
                          velocities.dtype.itemsize * velocities.shape[1]))
        voidtype_array = np.ascontiguousarray(velocities).view(dtype)

        _, idx = np.unique(voidtype_array, return_index=True)

        return velocities[idx]

    def get_matching_velocities(self, velocities, v):
        return np.where((velocities == v).all(axis=1))

    def rewind(self):
        'remove previously written files'
        super(CurrentGeoJsonOutput, self).rewind()

    def current_movers_to_dict(self):
        '''
        a dict containing 'obj_type' and 'id' for each object in
        list/collection
        '''
        return self._collection_to_dict(self.current_movers)


class IceGeoJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class IceGeoJsonOutput(Outputter):
    '''
    Class that outputs GNOME ice velocity results for each ice mover
    in a geojson format.  The output is a collection of Features.
    Each Feature contains a Point object with associated properties.
    Following is the output format - the data in <> are the results
    for each element.
    ::

    {
     "time_stamp": <TIME IN ISO FORMAT>,
     "step_num": <OUTPUT ASSOCIATED WITH THIS STEP NUMBER>,
     "feature_collections": {<mover_id>: {"type": "FeatureCollection",
                                          "features": [{"type": "Feature",
                                                        "id": <PARTICLE_ID>,
                                                        "properties": {"ice_fraction": <FRACTION>,
                                                                       "ice_thickness": <METERS>,
                                                                       "water_velocity": [u, v],
                                                                       "ice_velocity": [u, v]
                                                                       },
                                                        "geometry": {"type": "Point",
                                                                     "coordinates": [<LONG>, <LAT>]
                                                                     },
                                                        },
                                                        ...
                                                       ],
                                          },
                             ...
                             }
    }
    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state.add_field(Field('ice_movers',
                           save=True, update=True, iscollection=True))

    _schema = IceGeoJsonSchema

    def __init__(self, ice_movers, **kwargs):
        '''
            :param ice_movers: ice_movers associated with this outputter.
            :type ice_movers: An ice_mover object or sequence of ice_mover
                              objects.

            Use super to pass optional \*\*kwargs to base class __init__ method
        '''
        if (isinstance(ice_movers, Iterable) and
                not isinstance(ice_movers, str)):
            self.ice_movers = ice_movers
        elif ice_movers is not None:
            self.ice_movers = (ice_movers,)
        else:
            self.ice_movers = tuple()

        super(IceGeoJsonOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(IceGeoJsonOutput, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            pass

        model_time = date_to_sec(sc.current_time_stamp)

        geojson = {}
        for mover in self.ice_movers:
            grid_data = mover.get_grid_data()
            ice_coverage, ice_thickness = mover.get_ice_fields(model_time)

            geojson[mover.id] = []
            geojson[mover.id].append(self.get_coverage_fc(ice_coverage,
                                                          grid_data))
            geojson[mover.id].append(self.get_thickness_fc(ice_thickness,
                                                           grid_data))

        # default geojson should not output data to file
        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'feature_collections': geojson
                       }

        return output_info

    def get_coverage_fc(self, coverage, triangles):
        return self.get_grouped_fc_from_1d_array(coverage, triangles,
                                                 'coverage',
                                                 decimals=2)

    def get_thickness_fc(self, thickness, triangles):
        return self.get_grouped_fc_from_1d_array(thickness, triangles,
                                                 'thickness',
                                                 decimals=1)

    def get_grouped_fc_from_1d_array(self, values, triangles,
                                     property_name, decimals):
        rounded = values.round(decimals=decimals)
        unique = np.unique(rounded)

        features = []
        for u in unique:
            matching = np.where(rounded == u)
            matching_triangles = (triangles[matching])

            dtype = matching_triangles.dtype.descr
            shape = matching_triangles.shape + (len(dtype),)

            coordinates = (matching_triangles.view(dtype='<f8')
                           .reshape(shape).tolist())

            prop_fmt = '{{:.{}f}}'.format(decimals)
            properties = {'{}'.format(property_name): prop_fmt.format(u)}

            feature = Feature(id="1",
                              properties=properties,
                              geometry=MultiPolygon(coordinates=coordinates
                                                    ))
            features.append(feature)

        return FeatureCollection(features)

    def get_rounded_ice_values(self, coverage, thickness):
        return np.vstack((coverage.round(decimals=2),
                          thickness.round(decimals=1))).T

    def get_unique_ice_values(self, ice_values):
        '''
        In order to make numpy perform this function fast, we will use a
        contiguous structured array using a view of a void type that
        joins the whole row into a single item.
        '''
        dtype = np.dtype((np.void,
                          ice_values.dtype.itemsize * ice_values.shape[1]))
        voidtype_array = np.ascontiguousarray(ice_values).view(dtype)

        _, idx = np.unique(voidtype_array, return_index=True)

        return ice_values[idx]

    def get_matching_ice_values(self, ice_values, v):
        return np.where((ice_values == v).all(axis=1))

    def rewind(self):
        'remove previously written files'
        super(IceGeoJsonOutput, self).rewind()

    def ice_movers_to_dict(self):
        '''
        a dict containing 'obj_type' and 'id' for each object in
        list/collection
        '''
        return self._collection_to_dict(self.ice_movers)

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for current mover
        """
        schema = cls._schema()
        _to_dict = schema.deserialize(json_)

        if 'ice_movers' in json_:
            _to_dict['ice_movers'] = []
            for i, cm in enumerate(json_['ice_movers']):
                cm_cls = class_from_objtype(cm['obj_type'])
                cm_dict = cm_cls.deserialize(json_['ice_movers'][i])

                _to_dict['ice_movers'].append(cm_dict)

        return _to_dict


class IceRawJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class IceRawJsonOutput(Outputter):
    '''
    Class that outputs GNOME ice property results for each ice mover
    in a raw JSON format.  The output contains a dict keyed by mover id.
    Each value item in the dict contains a list of feature data.
    Each feature item contains the ice fraction and thickness, and polyonal
    coordinate data.
    Following is the output format.

    ::

        {
         "time_stamp": <TIME IN ISO FORMAT>,
         "step_num": <OUTPUT ASSOCIATED WITH THIS STEP NUMBER>,
         "feature_collections": {<mover_id>: [[<ICE_CONCENTRATION>,
                                               <ICE_THICKNESS>,
                                               [[<LONG>, <LAT>], ...],
                                               ],
                                              ...
                                              ],
                                 ...
                                 }
        }
    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state.add_field(Field('ice_movers',
                           save=True, update=True, iscollection=True))

    _schema = IceRawJsonSchema

    def __init__(self, ice_movers, **kwargs):
        '''
            :param ice_movers: ice_movers associated with this outputter.
            :type ice_movers: An ice_mover object or sequence of ice_mover
                              objects.

            Use super to pass optional \*\*kwargs to base class __init__ method
        '''
        if (isinstance(ice_movers, Iterable) and
                not isinstance(ice_movers, str)):
            self.ice_movers = ice_movers
        elif ice_movers is not None:
            self.ice_movers = (ice_movers,)
        else:
            self.ice_movers = tuple()

        super(IceRawJsonOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(IceRawJsonOutput, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            pass

        model_time = date_to_sec(sc.current_time_stamp)

        raw_json = {}
        for mover in self.ice_movers:
            grid_data = mover.get_grid_data().tolist()
            ice_coverage, ice_thickness = mover.get_ice_fields(model_time)

            raw_json[mover.id] = []
            for c, th, grid in zip(ice_coverage, ice_thickness, grid_data):
                raw_json[mover.id].append([c, th, grid])

        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'feature_collections': raw_json
                       }

        return output_info

    def rewind(self):
        'remove previously written files'
        super(IceRawJsonOutput, self).rewind()

    def ice_movers_to_dict(self):
        '''
        a dict containing 'obj_type' and 'id' for each object in
        list/collection
        '''
        return self._collection_to_dict(self.ice_movers)

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for current mover
        """
        schema = cls._schema()
        _to_dict = schema.deserialize(json_)

        if 'ice_movers' in json_:
            _to_dict['ice_movers'] = []
            for i, cm in enumerate(json_['ice_movers']):
                cm_cls = class_from_objtype(cm['obj_type'])
                cm_dict = cm_cls.deserialize(json_['ice_movers'][i])

                _to_dict['ice_movers'].append(cm_dict)

        return _to_dict
