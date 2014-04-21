'''
GeoJson outputter
'''
import copy

import numpy as np
from geojson import MultiPoint, Feature, FeatureCollection

from gnome.utilities.serializable import Serializable

from .outputter import Outputter

from gnome.basic_types import oil_status
from gnome import array_types as at


class GeoJson(Outputter):
    def __init__(self, round_data=True, roundto=4, **kwargs):
        '''
        :param bool round_data=True: if True, then round the numpy arrays
            containing float to number of digits specified by 'roundto'.
            Default is True
        :param int roundto=4: round float arrays to these number of digits.
            Default is 4.

        use super to pass optional **kwargs to base class __init__ method
        '''
        self.round_data = round_data
        self.roundto = roundto
        super(GeoJson, self).__init__(**kwargs)

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(GeoJson, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        arrays = ['positions', 'id', 'spill_num', 'status_codes']
        features = []
        for sc in self.cache.load_timestep(step_num).items():
            time_stamp = sc.current_time_stamp
            sc_type = 'forecast'
            if sc.uncertain:
                sc_type = 'uncertain'

            common = {'spill': sc_type, 'step_num': step_num}
            for name in arrays:
                (data, props) = self._prepare_dataarray(sc, name)
                props.update(common)
                # todo: should probably also save timestamp with each feature
                feature = Feature(geometry=MultiPoint(data),
                                       id=name,
                                       properties=copy.deepcopy(props))
                features.append(feature)

        output_info = {'step_num': step_num,
                       'geojson': FeatureCollection(features),
                       'time_stamp': time_stamp}

        return output_info

    def _prepare_dataarray(self, sc, name):
        '''
        return array as list with appropriate dtype for geojson and additional
        properties to be added
        '''
        s_flags = sorted(zip(oil_status._int, oil_status._attr))
        add_props = {}
        if name == 'status_codes':
            add_props = {'flag_meanings': s_flags}

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
        return (data, add_props)
