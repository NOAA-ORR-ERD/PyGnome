'''
JSON outputter
Does not contain a schema for persistence yet
'''

import copy
import numpy as np
import json
import geopandas as gpd
import shapely
import time
from collections.abc import Iterable
from colander import Boolean, SchemaNode, SequenceSchema, String, drop

from gnome.utilities.time_utils import date_to_sec

from gnome.movers import PyMover

from .outputter import Outputter, BaseOutputterSchema
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.movers.c_current_movers import CatsMoverSchema,\
    ComponentMoverSchema, c_GridCurrentMoverSchema, CurrentCycleMoverSchema,\
    IceMoverSchema
from gnome.movers.c_wind_movers import PointWindMoverSchema
from gnome.utilities.hull import calculate_hull, calculate_contours


class SpillJsonSchema(BaseOutputterSchema):
    _additional_data = SequenceSchema(
        SchemaNode(String()), missing=drop, save=True, update=True
    )
    include_uncertain_bounds = SchemaNode(
        Boolean(), missing=False, save=True, update=True
    )


class SpillJsonOutput(Outputter):
    '''
    Class that outputs data on GNOME particles.
    Following is the format for a particle - the
    data in <> are the results for each element.
    ::

        {
            "certain": {
                "length":<LENGTH>
                "longitude": []
                "latitude": []
                "status_code": []
                "mass": []
                "spill_num":[]
            }
            "uncertain":{
                "length":<LENGTH>
                "longitude": []
                "latitude": []
                "status_code": []
                "mass": []
                "spill_num":[]
            }
            "uncertain_bounds":{
                "length":<LENGTH>
                "bounds": []
                "spill_num":[]
            }
            "step_num": <STEP_NUM>
            "timestamp": <TIMESTAMP>
        }
    '''
    _schema = SpillJsonSchema

    def __init__(self, _additional_data=None, include_uncertain_bounds=False, **kwargs):
        '''
        :param list current_movers: A list or collection of current grid mover
                                    objects.

        use super to pass optional kwargs to base class __init__ method
        '''
        self._additional_data =_additional_data if _additional_data else []
        self.include_uncertain_bounds = include_uncertain_bounds
        super(SpillJsonOutput, self).__init__(**kwargs)

        # Build some mappings for styling
        self.default_unit_map = {'Mass':{'column': 'mass',
                                         'unit':'kilograms'},
                                 'Surface Concentration':{'column': 'surf_conc',
                                                          'unit':'kg/m^2'},
                                 'Age': {'column': 'age',
                                         'unit': 'seconds'},
                                 'Viscosity': {'column': 'viscosity',
                                               'unit': 'm^2/s'}
                                 }

    def prepare_for_model_run(self,
                              model_start_time,
                              spills,
                              uncertain = False,
                              **kwargs):
        """ Setup before we run the model. """
        if not self.on:
            return
        super(SpillJsonOutput, self).prepare_for_model_run(model_start_time,
                                                           spills,
                                                           **kwargs)
        self.model_start_time = model_start_time
        self.spills = spills

        self.cutoff_struct = self.generate_cutoff_struct()
        self.uncertain = uncertain

    # Generate a cutoff stuct for contours if needed
    def generate_cutoff_struct(self):
        cutoff_struct = {}
        for spill_idx, spill in enumerate(self.spills):
            if spill._appearance and spill._appearance.include_certain_contours and spill._appearance.colormap:
                appearance = spill._appearance.to_dict()
                # self.logger.debug(f'appearance: {appearance}')
                colormap = spill._appearance.colormap.to_dict()
                requested_display_param = appearance['data']
                requested_display_unit = appearance['units']
                unit_map = {}
                if requested_display_param in self.default_unit_map:
                    unit_map = self.default_unit_map[requested_display_param]
                else:
                    raise ValueError(f'Style requested is not supported!!! {requested_display_param}')
                data_column = unit_map['column']
                # Looks like the domains are always in the base units...
                number_scale_domain = [val for val in colormap['numberScaleDomain']]
                cutoff_array = [val for val in colormap['colorScaleDomain']]
                cutoff_array.append(number_scale_domain[-1])
                cutoffs = []
                for idx, val in enumerate(cutoff_array):
                    colorblocklabel = colormap['colorBlockLabels'][idx]
                    color = colormap['colorScaleRange'][idx]
                    cutoffs.append({'cutoff': val,
                                    'cutoff_id': idx,
                                    'color': color,
                                    'label': colorblocklabel})
                cutoff_struct[spill_idx] = {'param': data_column,
                                            'cutoffs': cutoffs}
        return cutoff_struct

    def write_output(self, step_num, islast_step=False):
        'dump data in geojson format'
        super(SpillJsonOutput, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        # one feature per element client; replaced with multipoint
        # because client performance is much more stable with one
        # feature per step rather than (n) features per step.features = []
        certain_scs = []
        certain_contours_scs = []
        uncertain_scs = []
        uncertain_bounds_scs = []
        self.cutoff_struct = self.generate_cutoff_struct()

        sp = self.cache.load_timestep(step_num).items()
        for sc in sp:
            position = sc['positions']
            longitude = np.around(position[:, 0], 5).tolist()
            latitude = np.around(position[:, 1], 5).tolist()
            status = sc['status_codes'].tolist()
            mass = np.around(sc['mass'], 4).tolist()
            spill_num = sc['spill_num'].tolist()
            # break elements into multipoint features based on their
            # status code
            #   evaporated : 10
            #   in_water : 2
            #   not_released : 0
            #   off_maps : 7
            #   on_land : 3
            #   to_be_removed : 12

            out = {"longitude": longitude,
                   "latitude": latitude,
                   "status": status,
                   "mass": mass,
                   "spill_num": spill_num,
                   "length": len(longitude)
                   }

            if self._additional_data and len(self._additional_data) > 0:
                for d in self._additional_data:
                    if d == 'viscosity' or d == 'surface_concentration':
                        out[d] = np.around(sc[d], 8).tolist()
                    else:
                        out[d] = np.around(sc[d], 4).tolist()

            if sc.uncertain:
                uncertain_scs.append(out)
                if self.include_uncertain_bounds:
                    # Calculate the uncertain bounds
                    separate_by_spill = True
                    hull_ratio = 0.8
                    hull_allow_holes = False
                    start = time.time()
                    hull = calculate_hull(sp, separate_by_spill=separate_by_spill,
                                          ratio=hull_ratio, union_results=True,
                                          allow_holes=hull_allow_holes)
                    end = time.time()
                    # self.logger.debug(f'calculate_hull: {end-start}')
                    json_geoms = []
                    spill_num = []
                    for h in hull['hulls']:
                        json_geoms.append(shapely.geometry.mapping(h))
                    if len(hull['spill_num']):
                        if hull['spill_num'][0]:
                            spill_num = list(map(int, hull['spill_num']))
                        else:
                            spill_num = hull['spill_num']
                    boundary_out = {"bounds": json_geoms, #hull['hulls'],
                                    "spill_num": spill_num,
                                    "length": len(hull['hulls'])
                                    }
                    uncertain_bounds_scs.append(boundary_out)
            else:
                certain_scs.append(out)
                # Calculate the contours
                hull_ratio = 0.8
                hull_allow_holes = False
                start = time.time()
                # self.logger.debug(f'cutoff_struct: {self.cutoff_struct}')
                contours = calculate_contours(sc, cutoff_struct=self.cutoff_struct,
                                              ratio=hull_ratio,
                                              allow_holes=hull_allow_holes)
                end = time.time()
                # self.logger.debug(f'calculate_contours: {end-start}')
                json_geoms = []
                spill_num = []
                # Unique spill ids
                spill_ids = list(set([item['spill_num'] for item in contours]))
                contour_list = []
                for id in spill_ids:
                    # Make our list to convert to json
                    schema = {'geometry': [item['contour']
                                           for item in contours if item['spill_num'] == id],
                              'spill_num': [item['spill_num']
                                            for item in contours if item['spill_num'] == id],
                              'color': [item['color']
                                        for item in contours if item['spill_num'] == id],
                              'label': [item['label']
                                        for item in contours if item['spill_num'] == id]}
                    data_frame = gpd.GeoDataFrame(schema, crs='epsg:4326',
                                                  geometry='geometry')
                    json_geoms = json.loads(data_frame.to_json())
                    json_geoms['spill_num'] = id
                    contour_list.append(json_geoms)
                certain_contours_out = {"contours": contour_list,
                                        "length": len(contour_list)
                                        }
                certain_contours_scs.append(certain_contours_out)

        # default geojson should not output data to file
        # read data from file and send it to web client
        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'step_num': step_num,
                       'certain': certain_scs,
                       'certain_contours': certain_contours_scs,
                       'uncertain': uncertain_scs,
                       'uncertain_bounds': uncertain_bounds_scs}

        if self.output_dir:
            output_info['output_filename'] = self.output_to_file(certain_scs,
                                                                 step_num)
            self.output_to_file(uncertain_scs, step_num)

        return output_info


class CurrentJsonSchema(BaseOutputterSchema):
    current_movers = SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[CatsMoverSchema,
                                ComponentMoverSchema,
                                c_GridCurrentMoverSchema,
                                CurrentCycleMoverSchema,
                                PointWindMoverSchema]
        ),
        save=True, update=True, save_reference=True
    )
    '''
    Nothing is required for initialization
    '''


class CurrentJsonOutput(Outputter):
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
    _schema = CurrentJsonSchema

    def __init__(self, current_movers, **kwargs):
        '''
        :param list current_movers: A list or collection of current grid mover
                                    objects.

        use super to pass optional kwargs to base class __init__ method
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

    # # if all it does is call super, you don't need it!
    # def rewind(self):
    #     'remove previously written files'
    #     super(CurrentJsonOutput, self).rewind()

    def current_movers_to_dict(self):
        '''
        a dict containing 'obj_type' and 'id' for each object in
        list/collection
        '''
        return self._collection_to_dict(self.current_movers)


class IceJsonSchema(BaseOutputterSchema):
    ice_movers =  SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[IceMoverSchema]
        ),
        save=True, update=True, save_reference=True
    )


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
    _schema = IceJsonSchema

    def __init__(self, ice_movers, **kwargs):
        '''
            :param ice_movers: ice_movers associated with this outputter.
            :type ice_movers: An ice_mover object or sequence of ice_mover
                              objects.

            Use super to pass optional kwargs to base class __init__ method
        '''
        if (isinstance(ice_movers, Iterable) and
                not isinstance(ice_movers, str)):
            self.ice_movers = ice_movers
        elif ice_movers is not None:
            self.ice_movers = (ice_movers,)
        else:
            self.ice_movers = tuple()

        super(IceJsonOutput, self).__init__(**kwargs)

    def clean_output_files(self):
        """
        this outputter doesn't write any files

        but this method needs to be here
        """
        pass

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



