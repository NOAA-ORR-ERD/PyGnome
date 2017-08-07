'''
JSON outputter
Does not contain a schema for persistence yet
'''
import copy
from collections import Iterable

import numpy as np

from gnome.utilities.time_utils import date_to_sec
from gnome.utilities.serializable import Serializable, Field

from gnome.movers import PyMover
from gnome.persist import class_from_objtype

from .outputter import Outputter, BaseSchema


class CurrentJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class CurrentJsonOutput(Outputter, Serializable):
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

    _schema = CurrentJsonSchema

    def __init__(self, current_movers, **kwargs):
        '''
        :param list current_movers: A list or collection of current grid mover
                                    objects.

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.current_movers = current_movers

        super(CurrentJsonOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(CurrentJsonOutput, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            model_time = date_to_sec(sc.current_time_stamp)

        json_ = {}

        for cm in self.current_movers:
            is_pymover = isinstance(cm, PyMover)

            if is_pymover:
                model_time = sc.current_time_stamp

            velocities = cm.get_scaled_velocities(model_time)

            if is_pymover:
                velocities = velocities[:, 0:2].round(decimals=2)
            else:
                velocities = self.get_rounded_velocities(velocities)

            x = velocities[:, 0]
            y = velocities[:, 1]

            direction = np.arctan2(y, x) - np.pi/2
            magnitude = np.sqrt(x**2 + y**2)

            direction = np.round(direction, 2)
            magnitude = np.round(magnitude, 2)

            json_[cm.id] = {'magnitude': magnitude.tolist(),
                            'direction': direction.tolist()}

        return json_

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
        super(CurrentJsonOutput, self).rewind()

    def current_movers_to_dict(self):
        '''
        a dict containing 'obj_type' and 'id' for each object in
        list/collection
        '''
        return self._collection_to_dict(self.current_movers)


class IceJsonSchema(BaseSchema):
    '''
    Nothing is required for initialization
    '''


class IceJsonOutput(Outputter):
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
    _state.add_field(Field('ice_movers', save=True, update=True,
                           iscollection=True))

    _schema = IceJsonSchema

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

        super(IceJsonOutput, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(IceJsonOutput, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            pass

        model_time = date_to_sec(sc.current_time_stamp)

        raw_json = {}

        for mover in self.ice_movers:
            ice_coverage, ice_thickness = mover.get_ice_fields(model_time)

            raw_json[mover.id] = {"thickness": [],
                                  "concentration": []}

            raw_json[mover.id]["thickness"] = ice_thickness.tolist()
            raw_json[mover.id]["concentration"] = ice_coverage.tolist()

        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'data': raw_json}

        return output_info

    def rewind(self):
        'remove previously written files'
        super(IceJsonOutput, self).rewind()

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
