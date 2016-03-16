'''
Movers using currents and tides as forcing functions
'''
import os
from os.path import basename
import copy

import numpy as np

from colander import (SchemaNode, Bool, String, Float, drop)

from gnome.persist.base_schema import ObjType, WorldPoint

from gnome.movers import CyMover, ProcessSchema
from gnome import environment
from gnome.utilities import serializable

from gnome import basic_types
from gnome.cy_gnome.cy_cats_mover import CyCatsMover
from gnome.cy_gnome.cy_gridcurrent_mover import CyGridCurrentMover
from gnome.cy_gnome.cy_ice_mover import CyIceMover
from gnome.cy_gnome.cy_currentcycle_mover import CyCurrentCycleMover
from gnome.cy_gnome.cy_shio_time import CyShioTime
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_component_mover import CyComponentMover


class CurrentMoversBaseSchema(ObjType, ProcessSchema):
    uncertain_duration = SchemaNode(Float(), missing=drop)
    uncertain_time_delay = SchemaNode(Float(), missing=drop)


class CurrentMoversBase(CyMover):
    _state = copy.deepcopy(CyMover._state)
    _state.add(update=['uncertain_duration', 'uncertain_time_delay'],
               save=['uncertain_duration', 'uncertain_time_delay'])

    _ref_as = 'current_movers'

    def __init__(self,
                 uncertain_duration=24,
                 uncertain_time_delay=0,
                 **kwargs):
        '''
        set common properties
        children should define self.mover, then we can set common properties
        '''
        self.uncertain_duration = uncertain_duration
        self.uncertain_time_delay = uncertain_time_delay
        super(CurrentMoversBase, self).__init__(**kwargs)

    uncertain_duration = property(lambda self:
                                  self.mover.uncertain_duration / 3600.,
                                  lambda self, val:
                                  setattr(self.mover, 'uncertain_duration',
                                          val * 3600.))

    uncertain_time_delay = property(lambda self:
                                    self.mover.uncertain_time_delay / 3600.,
                                    lambda self, val:
                                    setattr(self.mover, 'uncertain_time_delay',
                                            val * 3600.))

    def get_triangles(self):
        """
            Invokes the GetToplogyHdl method of TimeGridVel_c object.
            Cross-references point data to get triangle coordinates.
        """
        triangle_data = self.mover._get_triangle_data()
        points = self.get_points()

        dtype = triangle_data[0].dtype.descr
        unstructured_type = dtype[0][1]
        unstructured = (triangle_data.view(dtype=unstructured_type)
                        .reshape(-1, len(dtype))[:, :3])

        return points[unstructured]

    def get_cells(self):
        """
            Invokes the GetCellDataHdl method of TimeGridVel_c object.
            Cross-references point data to get cell coordinates.
        """
        cell_data = self.mover._get_cell_data()
        points = self.get_points()

        dtype = cell_data[0].dtype.descr
        unstructured_type = dtype[0][1]
        unstructured = (cell_data.view(dtype=unstructured_type)
                        .reshape(-1, len(dtype))[:, 1:])

        return points[unstructured]

    def get_triangle_center_points(self):
        '''
            Right now the cython mover only gets the triangular center points.
        '''
        return self.mover._get_center_points().view(dtype='<f8').reshape(-1, 2)

    def get_cell_center_points(self):
        '''
        Right now the cython mover only gets the triangular center points,
        so we need to calculate centers based on the cells themselves.

        Cells will have the format (tl, tr, bl, br)
        We need to get the rectangular centers
        Center will be: (tl + ((br - tl) / 2.))
        '''
        return self.mover._get_center_points().view(dtype='<f8').reshape(-1, 2)

#         cells = self.get_cells()
#         raw_cells = cells.view(dtype='<f8').reshape(-1, 4, 2)
#         centers = (raw_cells[:, 0, :] +
#                    #(raw_cells[:, 3, :] - raw_cells[:, 0, :]) / 2.)
#                    (raw_cells[:, 2, :] - raw_cells[:, 0, :]) / 2.)
#
#         return centers

    def get_points(self):
        points = (self.mover._get_points()
                  .astype([('long', '<f8'), ('lat', '<f8')]))
        points['long'] /= 10 ** 6
        points['lat'] /= 10 ** 6

        return points


class CatsMoverSchema(CurrentMoversBaseSchema):
    '''static schema for CatsMover'''
    filename = SchemaNode(String(), missing=drop)
    scale = SchemaNode(Bool(), missing=drop)
    scale_refpoint = WorldPoint(missing=drop)
    scale_value = SchemaNode(Float(), missing=drop)

    # the following six could be shared with grid_current
    # in a currents base class
    down_cur_uncertain = SchemaNode(Float(), missing=drop)
    up_cur_uncertain = SchemaNode(Float(), missing=drop)
    right_cur_uncertain = SchemaNode(Float(), missing=drop)
    left_cur_uncertain = SchemaNode(Float(), missing=drop)
    uncertain_eddy_diffusion = SchemaNode(Float(), missing=drop)
    uncertain_eddy_v0 = SchemaNode(Float(), missing=drop)


class CatsMover(CurrentMoversBase, serializable.Serializable):

    _state = copy.deepcopy(CurrentMoversBase._state)

    _update = ['scale', 'scale_refpoint', 'scale_value',
               'up_cur_uncertain', 'down_cur_uncertain',
               'right_cur_uncertain', 'left_cur_uncertain',
               'uncertain_eddy_diffusion', 'uncertain_eddy_v0']
    _create = []
    _create.extend(_update)
    _state.add(update=_update, save=_create)
    _state.add_field([serializable.Field('filename',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('tide',
                                         save=True, update=True,
                                         save_reference=True)])
    _schema = CatsMoverSchema

    def __init__(self, filename, tide=None, uncertain_duration=48,
                 **kwargs):
        """
        Uses super to invoke base class __init__ method.

        :param filename: file containing currents patterns for Cats

        Optional parameters (kwargs).
        Defaults are defined by CyCatsMover object.

        :param tide: a gnome.environment.Tide object to be attached to
                     CatsMover
        :param scale: a boolean to indicate whether to scale value at
                      reference point or not
        :param scale_value: value used for scaling at reference point
        :param scale_refpoint: reference location (long, lat, z). The scaling
                               applied to all data is determined by scaling
                               the raw value at this location.

        :param uncertain_duration: how often does a given uncertain element
                                   gets reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param up_cur_uncertain: Scale for uncertainty along the flow
        :param down_cur_uncertain: Scale for uncertainty along the flow
        :param right_cur_uncertain: Scale for uncertainty across the flow
        :param left_cur_uncertain: Scale for uncertainty across the flow
        :param uncertain_eddy_diffusion: Diffusion coefficient for
                                         eddy diffusion. Default is 0.
        :param uncertain_eddy_v0: Default is .1 (Check that this is still used)
        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """
        if not os.path.exists(filename):
            raise ValueError('Path for Cats filename does not exist: {0}'
                             .format(filename))

        self._filename = filename

        # check if this is stored with cy_cats_mover?
        self.mover = CyCatsMover()
        self.mover.text_read(filename)
        self.name = os.path.split(filename)[1]

        self._tide = None
        if tide is not None:
            self.tide = tide

        self.scale = kwargs.pop('scale', self.mover.scale_type)
        self.scale_value = kwargs.get('scale_value',
                                      self.mover.scale_value)

        self.up_cur_uncertain = kwargs.pop('up_cur_uncertain', .3)
        self.down_cur_uncertain = kwargs.pop('down_cur_uncertain', -.3)
        self.right_cur_uncertain = kwargs.pop('right_cur_uncertain', .1)
        self.left_cur_uncertain = kwargs.pop('left_cur_uncertain', -.1)
        self.uncertain_eddy_diffusion = kwargs.pop('uncertain_eddy_diffusion',
                                                   0)
        self.uncertain_eddy_v0 = kwargs.pop('uncertain_eddy_v0', .1)
        # TODO: no need to check for None since properties that are None
        # are not persisted

        if 'scale_refpoint' in kwargs:
            self.scale_refpoint = kwargs.pop('scale_refpoint')
            self.mover.compute_velocity_scale()

        if (self.scale and
            self.scale_value != 0.0 and
                self.scale_refpoint is None):
            raise TypeError("Provide a reference point in 'scale_refpoint'.")

        super(CatsMover, self).__init__(uncertain_duration, **kwargs)

    def __repr__(self):
        return 'CatsMover(filename={0})'.format(self.filename)

    # Properties
    filename = property(lambda self: basename(self._filename),
                        lambda self, val: setattr(self, '_filename', val))

    scale = property(lambda self: bool(self.mover.scale_type),
                     lambda self, val: setattr(self.mover,
                                               'scale_type',
                                               int(val)))

    scale_value = property(lambda self: self.mover.scale_value,
                           lambda self, val: setattr(self.mover,
                                                     'scale_value',
                                                     val))

    up_cur_uncertain = property(lambda self: self.mover.up_cur_uncertain,
                                lambda self, val: setattr(self.mover,
                                                          'up_cur_uncertain',
                                                          val))

    down_cur_uncertain = property(lambda self: self.mover.down_cur_uncertain,
                                  lambda self, val:
                                  setattr(self.mover, 'down_cur_uncertain',
                                          val))

    right_cur_uncertain = property(lambda self: self.mover.right_cur_uncertain,
                                   lambda self, val:
                                   setattr(self.mover, 'right_cur_uncertain',
                                           val))

    left_cur_uncertain = property(lambda self: self.mover.left_cur_uncertain,
                                  lambda self, val:
                                  setattr(self.mover, 'left_cur_uncertain',
                                          val))

    uncertain_eddy_diffusion = property(lambda self:
                                        self.mover.uncertain_eddy_diffusion,
                                        lambda self, val:
                                        setattr(self.mover,
                                                'uncertain_eddy_diffusion',
                                                val))

    uncertain_eddy_v0 = property(lambda self: self.mover.uncertain_eddy_v0,
                                 lambda self, val: setattr(self.mover,
                                                           'uncertain_eddy_v0',
                                                           val))

    @property
    def ref_scale(self):
        return self.mover.ref_scale

    @property
    def scale_refpoint(self):
        return self.mover.ref_point

    @scale_refpoint.setter
    def scale_refpoint(self, val):
        '''
        Must be a tuple of length 2 or 3: (long, lat, z). If only (long, lat)
        is given, the set z = 0
        '''
        if len(val) == 2:
            self.mover.ref_point = (val[0], val[1], 0.)
        else:
            self.mover.ref_point = val

        self.mover.compute_velocity_scale()

    @property
    def tide(self):
        return self._tide

    @tide.setter
    def tide(self, tide_obj):
        if not isinstance(tide_obj, environment.Tide):
            raise TypeError('tide must be of type environment.Tide')

        if isinstance(tide_obj.cy_obj, CyShioTime):
            self.mover.set_shio(tide_obj.cy_obj)
        elif isinstance(tide_obj.cy_obj, CyOSSMTime):
            self.mover.set_ossm(tide_obj.cy_obj)
        else:
            raise TypeError('Tide.cy_obj attribute must be either '
                            'CyOSSMTime or CyShioTime type for CatsMover.')

        self._tide = tide_obj

    def get_grid_data(self):
        """
            Invokes the GetToplogyHdl method of TriGridVel_c object
        """
        # we are assuming cats are always triangle grids,
        # but may want to extend
        return self.get_triangles()

    def get_center_points(self):
        return self.get_triangle_center_points()

    def get_scaled_velocities(self, model_time):
        """
        Get file values scaled to ref pt value, with tide applied (if any)
        """
        velocities = self.mover._get_velocity_handle()
        ref_scale = self.ref_scale  # this needs to be computed, needs a time

        if self._tide is not None:
            time_value = self._tide.cy_obj.get_time_value(model_time)
            tide = time_value[0][0]
        else:
            tide = 1

        velocities['u'] *= ref_scale * tide
        velocities['v'] *= ref_scale * tide

        return velocities

    def serialize(self, json_='webapi'):
        """
        Since 'wind' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to WindMover schema
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        if json_ == 'save':
            toserial['filename'] = self._filename

        if 'tide' in toserial:
            schema.add(environment.TideSchema(name='tide'))

        return schema.serialize(toserial)

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()

            if 'tide' in json_:
                schema.add(environment.TideSchema())

            return schema.deserialize(json_)
        else:
            return json_


class GridCurrentMoverSchema(CurrentMoversBaseSchema):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)
    current_scale = SchemaNode(Float(), missing=drop)
    uncertain_along = SchemaNode(Float(), missing=drop)
    uncertain_cross = SchemaNode(Float(), missing=drop)
    extrapolate = SchemaNode(Bool(), missing=drop)
    time_offset = SchemaNode(Float(), missing=drop)


class GridCurrentMover(CurrentMoversBase, serializable.Serializable):

    _update = ['uncertain_cross', 'uncertain_along', 'current_scale']
    _save = ['uncertain_cross', 'uncertain_along', 'current_scale']
    _state = copy.deepcopy(CurrentMoversBase._state)

    _state.add(update=_update, save=_save)
    _state.add_field([serializable.Field('filename',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('topology_file',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False)])
    _schema = GridCurrentMoverSchema

    def __init__(self, filename,
                 topology_file=None,
                 extrapolate=False,
                 time_offset=0,
                 current_scale=1,
                 uncertain_along=0.5,
                 uncertain_across=0.25,
                 num_method=basic_types.numerical_methods.euler,
                 **kwargs):
        """
        Initialize a GridCurrentMover

        :param filename: absolute or relative path to the data file:
                         could be netcdf or filelist
        :param topology_file=None: absolute or relative path to topology file.
                                   If not given, the GridCurrentMover will
                                   compute the topology from the data file.
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param current_scale: Value to scale current data
        :param uncertain_duration: how often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param extrapolate: Allow current data to be extrapolated
                            before and after file data
        :param time_offset: Time zone shift if data is in GMT
        :param num_method: Numerical method for calculating movement delta.
                           Default Euler
                           option: Runga-Kutta 4 (RK4)

        uses super, super(GridCurrentMover,self).__init__(\*\*kwargs)
        """

        # if child is calling, the self.mover is set by child - do not reset
        if type(self) == GridCurrentMover:
            self.mover = CyGridCurrentMover()

        if not os.path.exists(filename):
            raise ValueError('Path for current file does not exist: {0}'
                             .format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        # check if this is stored with cy_gridcurrent_mover?
        self.filename = filename
        self.name = os.path.split(filename)[1]

        # check if this is stored with cy_gridcurrent_mover?
        self.topology_file = topology_file
        self.current_scale = current_scale
        self.uncertain_along = uncertain_along
        self.uncertain_across = uncertain_across
        self.mover.text_read(filename, topology_file)
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset * 3600.)
        self.num_method = num_method

        super(GridCurrentMover, self).__init__(**kwargs)

        if self.topology_file is None:
            self.topology_file = filename + '.dat'
            self.export_topology(self.topology_file)

    def __repr__(self):
        return ('GridCurrentMover('
                'uncertain_duration={0.uncertain_duration},'
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'uncertain_cross={0.uncertain_cross}, '
                'uncertain_along={0.uncertain_along}, '
                'active_start={1.active_start}, '
                'active_stop={1.active_stop}, '
                'on={1.on})'.format(self.mover, self))

    def __str__(self):
        return ('GridCurrentMover - current _state.\n'
                '  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_cross={0.uncertain_cross}\n'
                '  uncertain_along={0.uncertain_along}\n'
                '  active_start time={1.active_start}\n'
                '  active_stop time={1.active_stop}\n'
                '  current on/off status={1.on}'
                .format(self.mover, self))

    # Define properties using lambda functions: uses lambda function, which are
    # accessible via fget/fset as follows:
    uncertain_cross = property(lambda self: self.mover.uncertain_cross,
                               lambda self, val: setattr(self.mover,
                                                         'uncertain_cross',
                                                         val))

    uncertain_along = property(lambda self: self.mover.uncertain_along,
                               lambda self, val: setattr(self.mover,
                                                         'uncertain_along',
                                                         val))

    current_scale = property(lambda self: self.mover.current_scale,
                             lambda self, val: setattr(self.mover,
                                                       'current_scale',
                                                       val))

    extrapolate = property(lambda self: self.mover.extrapolate,
                           lambda self, val: setattr(self.mover,
                                                     'extrapolate',
                                                     val))

    time_offset = property(lambda self: self.mover.time_offset / 3600.,
                           lambda self, val: setattr(self.mover,
                                                     'time_offset',
                                                     val * 3600.))
    num_method = property(lambda self: self.mover.num_method,
                          lambda self, val: setattr(self.mover,
                                                    'num_method',
                                                    val))

    def get_grid_data(self):
        """
            The main function for getting grid data from the mover
        """
        if self.mover._is_triangle_grid():
            return self.get_triangles()
        else:
            return self.get_cells()

    def get_center_points(self):
        if self.mover._is_triangle_grid():
            if self.mover._is_data_on_cells():
                return self.get_triangle_center_points()
            else:
                return self.get_points()
        else:
            return self.get_cell_center_points()

    def get_scaled_velocities(self, time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        # will need to update this for regular grids
        if self.mover._is_triangle_grid():
            if self.mover._is_data_on_cells():
                num_cells = num_tri
            else:
                num_vertices = self.mover.get_num_points()
                num_cells = num_vertices
        else:
            num_cells = num_tri / 2
        vels = np.zeros(num_cells, dtype=basic_types.velocity_rec)

        self.mover.get_scaled_velocities(time, vels)

        return vels

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where
                                   topology file will be written.
        """
        if topology_file is None:
            raise ValueError('Topology file path required: {0}'
                             .format(topology_file))

        self.mover.export_topology(topology_file)

    def extrapolate_in_time(self, extrapolate):
        """
        :param extrapolate=false: allow current data to be extrapolated
                                  before and after file data.
        """
        self.mover.extrapolate_in_time(extrapolate)

    def offset_time(self, time_offset):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset
                              (hours).
        """
        self.mover.offset_time(time_offset * 3600.)

    def get_offset_time(self):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset
                              (hours).
        """
        return (self.mover.get_offset_time()) / 3600.

    def get_num_method(self):
        return self.mover.num_method


class IceMoverSchema(CurrentMoversBaseSchema):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)
    current_scale = SchemaNode(Float(), missing=drop)
    uncertain_along = SchemaNode(Float(), missing=drop)
    uncertain_cross = SchemaNode(Float(), missing=drop)
    extrapolate = SchemaNode(Bool(), missing=drop)


class IceMover(CurrentMoversBase, serializable.Serializable):

    _update = ['uncertain_cross', 'uncertain_along',
               'current_scale', 'extrapolate']
    _save = ['uncertain_cross', 'uncertain_along',
             'current_scale', 'extrapolate']
    _state = copy.deepcopy(CurrentMoversBase._state)

    _state.add(update=_update, save=_save)
    _state.add_field([serializable.Field('filename',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('topology_file',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False)])
    _schema = IceMoverSchema

    def __init__(self, filename,
                 topology_file=None,
                 extrapolate=False,
                 time_offset=0,
                 **kwargs):
        """
        Initialize an IceMover

        :param filename: absolute or relative path to the data file:
                         could be netcdf or filelist
        :param topology_file=None: absolute or relative path to topology file.
                                   If not given, the IceMover will
                                   compute the topology from the data file.
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param current_scale: Value to scale current data
        :param uncertain_duration: how often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param extrapolate: Allow current data to be extrapolated
                            before and after file data
        :param time_offset: Time zone shift if data is in GMT

        uses super, super(IceMover,self).__init__(\*\*kwargs)
        """

        # NOTE: will need to add uncertainty parameters and other dialog fields
        #       use super with kwargs to invoke base class __init__

        # if child is calling, the self.mover is set by child - do not reset
        if type(self) == IceMover:
            self.mover = CyIceMover()

        if not os.path.exists(filename):
            raise ValueError('Path for current file does not exist: {0}'
                             .format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        # check if this is stored with cy_ice_mover?
        self.filename = filename
        self.name = os.path.split(filename)[1]

        # check if this is stored with cy_ice_mover?
        self.topology_file = topology_file

        self.mover.text_read(filename, topology_file)
        self.extrapolate = extrapolate
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset * 3600.)

        super(IceMover, self).__init__(**kwargs)

    def __repr__(self):
        return ('IceMover('
                'uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'uncertain_cross={0.uncertain_cross}, '
                'uncertain_along={0.uncertain_along}, '
                'active_start={1.active_start}, '
                'active_stop={1.active_stop}, '
                'on={1.on})'.format(self.mover, self))

    def __str__(self):
        return ('IceMover - current _state.\n'
                '  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_cross={0.uncertain_cross}\n'
                '  uncertain_along={0.uncertain_along}\n'
                '  active_start time={1.active_start}\n'
                '  active_stop time={1.active_stop}\n'
                '  current on/off status={1.on}'
                .format(self.mover, self))

    # Define properties using lambda functions: uses lambda function, which are
    # accessible via fget/fset as follows:
    uncertain_cross = property(lambda self: self.mover.uncertain_cross,
                               lambda self, val: setattr(self.mover,
                                                         'uncertain_cross',
                                                         val))

    uncertain_along = property(lambda self: self.mover.uncertain_along,
                               lambda self, val: setattr(self.mover,
                                                         'uncertain_along',
                                                         val))

    current_scale = property(lambda self: self.mover.current_scale,
                             lambda self, val: setattr(self.mover,
                                                       'current_scale',
                                                       val))

    extrapolate = property(lambda self: self.mover.extrapolate,
                           lambda self, val: setattr(self.mover,
                                                     'extrapolate',
                                                     val))

    time_offset = property(lambda self: self.mover.time_offset / 3600.,
                           lambda self, val: setattr(self.mover,
                                                     'time_offset',
                                                     val * 3600.))

    def get_grid_data(self):
        if self.mover._is_triangle_grid():
            return self.get_triangles()
        else:
            return self.get_cells()

    def get_grid_bounding_rect(self, grid_data=None):
        if grid_data is None:
            grid_data = self.get_grid_data()

        dtype = grid_data.dtype.descr
        unstructured_type = dtype[0][1]

        longs = (grid_data
                 .view(dtype=unstructured_type)
                 .reshape(-1, len(dtype))[:, 0])
        lats = (grid_data
                .view(dtype=unstructured_type)
                .reshape(-1, len(dtype))[:, 1])

        left_top = (longs.min(), lats.max())
        right_top = (longs.max(), lats.max())
        right_bottom = (longs.max(), lats.min())
        left_bottom = (longs.min(), lats.min())

        return [left_top, right_top,
                right_bottom, left_bottom]

    def get_center_points(self):
        if self.mover._is_triangle_grid():
            return self.get_triangle_center_points()
        else:
            return self.get_cell_center_points()

    def get_scaled_velocities(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        if self.mover._is_triangle_grid():
            num_cells = num_tri
        else:
            num_cells = num_tri / 2
        vels = np.zeros(num_cells, dtype=basic_types.velocity_rec)
        self.mover.get_scaled_velocities(model_time, vels)

        return vels

    def get_ice_velocities(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        vels = np.zeros(num_tri, dtype=basic_types.velocity_rec)
        self.mover.get_ice_velocities(model_time, vels)

        return vels

    def get_movement_velocities(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        vels = np.zeros(num_tri, dtype=basic_types.velocity_rec)
        self.mover.get_movement_velocities(model_time, vels)

        return vels

    def get_ice_fields(self, model_time):
        """
        :param model_time=0:
        """
        num_tri = self.mover.get_num_triangles()
        num_cells = num_tri / 2
        frac_coverage = np.zeros(num_cells, dtype=np.float64)
        thickness = np.zeros(num_cells, dtype=np.float64)
        self.mover.get_ice_fields(model_time, frac_coverage, thickness)

        return frac_coverage, thickness

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where
                                   topology file will be written.
        """
        if topology_file is None:
            raise ValueError('Topology file path required: {0}'
                             .format(topology_file))

        self.mover.export_topology(topology_file)

    def extrapolate_in_time(self, extrapolate):
        """
        :param extrapolate=false: allow current data to be extrapolated
                                  before and after file data.
        """
        self.mover.extrapolate_in_time(extrapolate)
        self.extrapolate = extrapolate

    def offset_time(self, time_offset):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset
                              (hours).
        """
        self.mover.offset_time(time_offset * 3600.)

    def get_offset_time(self):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset
                              (hours).
        """
        return (self.mover.get_offset_time()) / 3600.


class CurrentCycleMoverSchema(ObjType, ProcessSchema):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)
    current_scale = SchemaNode(Float(), default=1, missing=drop)
    uncertain_duration = SchemaNode(Float(), default=24, missing=drop)
    uncertain_time_delay = SchemaNode(Float(), default=0, missing=drop)
    uncertain_along = SchemaNode(Float(), default=.5, missing=drop)
    uncertain_cross = SchemaNode(Float(), default=.25, missing=drop)


class CurrentCycleMover(GridCurrentMover, serializable.Serializable):
    _state = copy.deepcopy(GridCurrentMover._state)
    _state.add_field([serializable.Field('tide',
                                         save=True, update=True,
                                         save_reference=True)])
    _schema = CurrentCycleMoverSchema

    def __init__(self,
                 filename,
                 topology_file=None,
                 **kwargs):
        """
        Initialize a CurrentCycleMover

        :param filename: Absolute or relative path to the data file:
                         could be netcdf or filelist
        :param topology_file=None: Absolute or relative path to topology file.
                                   If not given, the GridCurrentMover will
                                   compute the topology from the data file.
        :param tide: A gnome.environment.Tide object to be attached to
                     CatsMover
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param current_scale: Value to scale current data
        :param uncertain_duration: How often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param extrapolate: Allow current data to be extrapolated
                            before and after file data
        :param time_offset: Time zone shift if data is in GMT

        uses super: super(CurrentCycleMover,self).__init__(**kwargs)
        """

        # NOTE: will need to add uncertainty parameters
        #       and other dialog fields.
        #       use super with kwargs to invoke base class __init__
        self.mover = CyCurrentCycleMover()

        tide = kwargs.pop('tide', None)
        self._tide = None

        if tide is not None:
            self.tide = tide

        super(CurrentCycleMover, self).__init__(filename=filename,
                                                topology_file=topology_file,
                                                **kwargs)

    def __repr__(self):
        return ('GridCurrentMover(uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'uncertain_cross={0.uncertain_cross}, '
                'uncertain_along={0.uncertain_along}, '
                'active_start={1.active_start}, '
                'active_stop={1.active_stop}, '
                'on={1.on})'
                .format(self.mover, self))

    def __str__(self):
        return ('GridCurrentMover - current _state.\n'
                '  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_cross={0.uncertain_cross}\n'
                '  uncertain_along={0.uncertain_along}'
                '  active_start time={1.active_start}'
                '  active_stop time={1.active_stop}'
                '  current on/off status={1.on}'
                .format(self.mover, self))

    @property
    def tide(self):
        return self._tide

    @tide.setter
    def tide(self, tide_obj):
        if not isinstance(tide_obj, environment.Tide):
            raise TypeError('tide must be of type environment.Tide')

        if isinstance(tide_obj.cy_obj, CyShioTime):
            self.mover.set_shio(tide_obj.cy_obj)
        elif isinstance(tide_obj.cy_obj, CyOSSMTime):
            self.mover.set_ossm(tide_obj.cy_obj)
        else:
            raise TypeError('Tide.cy_obj attribute must be either '
                            'CyOSSMTime or CyShioTime type '
                            'for CurrentCycleMover.')

        self._tide = tide_obj

    def serialize(self, json_='webapi'):
        """
        Since 'tide' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to
        CurrentCycleMover schema
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()

        if json_ == 'webapi' and 'tide' in toserial:
            schema.add(environment.TideSchema(name='tide'))

        return schema.serialize(toserial)

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for tide object
        """
        schema = cls._schema()

        if 'tide' in json_:
            schema.add(environment.TideSchema())

        return schema.deserialize(json_)


class ComponentMoverSchema(ObjType, ProcessSchema):
    '''static schema for ComponentMover'''
    filename1 = SchemaNode(String(), missing=drop)
    filename2 = SchemaNode(String(), missing=drop)
    # scale = SchemaNode(Bool())
    # ref_point = WorldPoint(missing=drop)
    scale_refpoint = WorldPoint(missing=drop)
    # scale_value = SchemaNode(Float())


class ComponentMover(CyMover, serializable.Serializable):

    _state = copy.deepcopy(CyMover._state)

    _update = ['scale_refpoint',
               'pat1_angle', 'pat1_speed', 'pat1_speed_units',
               'pat1_scale_to_value',
               'pat2_angle', 'pat2_speed', 'pat2_speed_units',
               'pat2_scale_to_value', 'scale_by']
    _create = []
    _create.extend(_update)
    _state.add(update=_update, save=_create)
    _state.add_field([serializable.Field('filename1',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('filename2',
                                         save=True, read=True, isdatafile=True,
                                         test_for_eq=False),
                      serializable.Field('wind',
                                         save=True, update=True,
                                         save_reference=True)])
    _schema = ComponentMoverSchema

    def __init__(self, filename1, filename2=None, wind=None,
                 **kwargs):
        """
        Uses super to invoke base class __init__ method.

        :param filename: file containing currents for first Cats pattern

        Optional parameters (kwargs).
        Defaults are defined by CyCatsMover object.

        :param filename: file containing currents for second Cats pattern

        :param wind: A gnome.environment.Wind object to be used to drive the
                     CatsMovers.  Will want a warning that mover will
                     not be active without a wind
        :param scale: A boolean to indicate whether to scale value
                      at reference point or not
        :param scale_value: Value used for scaling at reference point
        :param scale_refpoint: Reference location (long, lat, z).
                               The scaling applied to all data is determined
                               by scaling the raw value at this location.

        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """

        if not os.path.exists(filename1):
            raise ValueError('Path for Cats filename1 does not exist: {0}'
                             .format(filename1))

        if filename2 is not None:
            if not os.path.exists(filename2):
                raise ValueError('Path for Cats filename2 does not exist: {0}'
                                 .format(filename2))

        self.filename1 = filename1
        self.filename2 = filename2

        self.mover = CyComponentMover()
        self.mover.text_read(filename1, filename2)

        self._wind = None
        if wind is not None:
            self.wind = wind

        # self.scale = kwargs.pop('scale', self.mover.scale_type)
        # self.scale_value = kwargs.get('scale_value',
        #                               self.mover.scale_value)

        # TODO: no need to check for None since properties that are None
        #       are not persisted

        # I think this is required...
        if 'scale_refpoint' in kwargs:
            self.scale_refpoint = kwargs.pop('scale_refpoint')

#         if self.scale and self.scale_value != 0.0 \
#             and self.scale_refpoint is None:
#             raise TypeError("Provide a reference point in 'scale_refpoint'."
#                             )

        super(ComponentMover, self).__init__(**kwargs)

    def __repr__(self):
        """
        unambiguous representation of object
        """
        return 'ComponentMover(filename={0})'.format(self.filename1)

    # Properties

    # scale_type = property(lambda self: bool(self.mover.scale_type),
    #                       lambda self, val: setattr(self.mover, 'scale_type',
    #                                                 int(val)))

    # scale_by = property(lambda self: bool(self.mover.scale_by),
    #                     lambda self, val: setattr(self.mover, 'scale_by',
    #                                               int(val)))

    pat1_angle = property(lambda self: self.mover.pat1_angle,
                          lambda self, val: setattr(self.mover, 'pat1_angle',
                                                    val))

    pat1_speed = property(lambda self: self.mover.pat1_speed,
                          lambda self, val: setattr(self.mover, 'pat1_speed',
                                                    val))

    pat1_speed_units = property(lambda self: self.mover.pat1_speed_units,
                                lambda self, val: setattr(self.mover,
                                                          'pat1_speed_units',
                                                          val))

    pat1_scale_to_value = property(lambda self: self.mover.pat1_scale_to_value,
                                   lambda self, val:
                                   setattr(self.mover, 'pat1_scale_to_value',
                                           val))

    pat2_angle = property(lambda self: self.mover.pat2_angle,
                          lambda self, val: setattr(self.mover, 'pat2_angle',
                                                    val))

    pat2_speed = property(lambda self: self.mover.pat2_speed,
                          lambda self, val: setattr(self.mover, 'pat2_speed',
                                                    val))

    pat2_speed_units = property(lambda self: self.mover.pat2_speed_units,
                                lambda self, val: setattr(self.mover,
                                                          'pat2_speed_units',
                                                          val))

    pat2_scale_to_value = property(lambda self: self.mover.pat2_scale_to_value,
                                   lambda self, val:
                                   setattr(self.mover, 'pat2_scale_to_value',
                                           val))

    scale_by = property(lambda self: self.mover.scale_by,
                        lambda self, val: setattr(self.mover, 'scale_by', val))

    extrapolate = property(lambda self: self.mover.extrapolate,
                           lambda self, val: setattr(self.mover,
                                                     'extrapolate',
                                                     val))

    use_averaged_winds = property(lambda self: self.mover.use_averaged_winds,
                                  lambda self, val: setattr(self.mover,
                                                            'use_averaged_winds',
                                                            val))

    wind_power_factor = property(lambda self: self.mover.wind_power_factor,
                                 lambda self, val: setattr(self.mover,
                                                           'wind_power_factor',
                                                           val))

    past_hours_to_average = property(lambda self: self.mover.past_hours_to_average,
                                     lambda self, val: setattr(self.mover,
                                                               'past_hours_to_average',
                                                               val))

    scale_factor_averaged_winds = property(lambda self: self.mover.scale_factor_averaged_winds,
                                           lambda self, val: setattr(self.mover,
                                                                     'scale_factor_averaged_winds',
                                                                     val))

    use_original_scale_factor = property(lambda self: self.mover.use_original_scale_factor,
                                         lambda self, val: setattr(self.mover,
                                                                   'use_original_scale_factor',
                                                                   val))

    @property
    def scale_refpoint(self):
        return self.mover.ref_point

    @scale_refpoint.setter
    def scale_refpoint(self, val):
        '''
        Must be a tuple of length 2 or 3: (long, lat, z). If only (long, lat)
        is given, the set z = 0
        '''
        if len(val) == 2:
            self.mover.ref_point = (val[0], val[1], 0.)
        else:
            self.mover.ref_point = val

    @property
    def wind(self):
        return self._wind

    @wind.setter
    def wind(self, wind_obj):
        if not isinstance(wind_obj, environment.Wind):
            raise TypeError('wind must be of type environment.Wind')

        self.mover.set_ossm(wind_obj.ossm)
        self._wind = wind_obj

    def serialize(self, json_='webapi'):
        """
        Since 'wind' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to WindMover schema
        """
        dict_ = self.to_serialize(json_)
        schema = self.__class__._schema()

        if json_ == 'webapi' and 'wind' in dict_:
            schema.add(environment.WindSchema(name='wind'))

        return schema.serialize(dict_)

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema()

        if 'wind' in json_:
            # for 'webapi', there will be nested Wind structure
            # for 'save' option, there should be no nested 'wind'. It is
            # removed, loaded and added back after deserialization
            schema.add(environment.WindSchema())
        _to_dict = schema.deserialize(json_)

        return _to_dict
