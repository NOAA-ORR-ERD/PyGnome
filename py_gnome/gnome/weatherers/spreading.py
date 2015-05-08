'''
objects used to model the spreading of oil
Include the Langmuir process here as well
'''
import copy

import numpy as np

from gnome.utilities.serializable import Serializable, Field
from gnome.environment import constant_wind, WindSchema
from gnome.constants import gravity
from .core import Weatherer

from gnome.persist.base_schema import ObjType


class Langmuir(Weatherer, Serializable):
    '''
    Easiest to define this as an environmental process
    WeatheringData uses it to set the fractional_coverage for
    '''
    _schema = ObjType

    _state = copy.deepcopy(Weatherer._state)
    _state += [Field('wind', update=True, save=True, save_reference=True)]

    def __init__(self,
                 wind=None,
                 **kwargs):
        '''
        initialize wind to (0, 0) if it is None
        '''
        super(Langmuir, self).__init__(**kwargs)
        self.array_types.update(('area', ))

        if wind is None:
            self.wind = constant_wind(0, 0)
        else:
            self.wind = wind

    def get_value(self, model_time, rel_buoy, thickness):
        '''
        return fractional coverage for a blob of oil with inputs;
        relative_buoyancy, and thickness

        Assumes the thickness is the minimum oil thickness associated with
        max area achievable by Fay Spreading

        Frac coverage bounds are constants. If computed frac_coverge is outside
        the bounds of (0.1, or 1.0), then limit it to:
            0.1 <= frac_cov <= 1.0
        '''
        v_max = self.wind.get_value(model_time)[0] * 0.005
        cr_k = \
            (v_max**2 * 4 * np.pi**2/(thickness * rel_buoy * gravity))**(1./3)
        frac_cov = 1./cr_k

        # if rel_buoy is an array, then frac_cov will be an array
        if not isinstance(frac_cov, np.ndarray):
            frac_cov = np.asarray([frac_cov], np.float64)

        frac_cov[frac_cov < 0.1] = 0.1
        frac_cov[frac_cov > 1.0] = 1.0

        if isinstance(cr_k, np.ndarray):
            return frac_cov
        else:
            # must be a scalar
            return frac_cov[0]

    def _wind_speed_bound(self, rel_buoy, thickness):
        '''
        return min/max wind speed for given rel_buoy, thickness such that
        Langmuir effect is within bounds:
            0.1 <= frac_coverage <= 1.0
        '''
        v_min = np.sqrt(1.0 * thickness * rel_buoy * gravity /
                        (4 * np.pi**2))/0.005
        v_max = np.sqrt((1./0.1)**3 * thickness * rel_buoy * gravity /
                        (4 * np.pi**2))/0.005
        return (v_min, v_max)

    def weather_elements(self):
        '''
        set the 'area' array based on the Langmuir process
        '''
        pass

    def serialize(self, json_='webapi'):
        """
        Since 'wind' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to WindMover schema
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema(name=self.__class__.__name__)
        if json_ == 'webapi':
            # add wind schema
            schema.add(WindSchema(name='wind'))

        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema(name=cls.__name__)
        if 'wind' in json_:
            schema.add(WindSchema(name='wind'))
        _to_dict = schema.deserialize(json_)

        return _to_dict
