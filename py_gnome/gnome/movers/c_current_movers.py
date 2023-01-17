'''
Movers using currents and tides as forcing functions
'''

import os

import numpy as np

from colander import (SchemaNode, Bool, String, Float, Int, drop)

from gnome import basic_types

from gnome.cy_gnome.cy_shio_time import CyShioTime
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_cats_mover import CyCatsMover
from gnome.cy_gnome.cy_gridcurrent_mover import CyGridCurrentMover
from gnome.cy_gnome.cy_ice_mover import CyIceMover
from gnome.cy_gnome.cy_currentcycle_mover import CyCurrentCycleMover
from gnome.cy_gnome.cy_component_mover import CyComponentMover

from gnome.utilities.time_utils import sec_to_datetime
from gnome.utilities.inf_datetime import InfTime, MinusInfTime

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime

from gnome.environment import Tide, TideSchema, Wind, WindSchema
from gnome.movers import CyMover, ProcessSchema

from gnome.persist.base_schema import WorldPoint
from gnome.persist.extend_colander import FilenameSchema


class CurrentMoversBaseSchema(ProcessSchema):
    uncertain_duration = SchemaNode(Float())
    uncertain_time_delay = SchemaNode(Float())
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class CurrentMoversBase(CyMover):
    _ref_as = 'c_current_movers'

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

    def get_points(self):
        points = (self.mover._get_points()
                  .astype([('long', '<f8'), ('lat', '<f8')]))
        points['long'] /= 10 ** 6
        points['lat'] /= 10 ** 6

        return points

    def get_bounds(self):
        '''
            Right now the cython mover only gets the triangular center points.
        '''
        bounds = self.mover._get_bounds()
        current_bounds = ((bounds["loLong"] / 1e6, bounds["loLat"] / 1e6), (bounds["hiLong"] / 1e6, bounds["hiLat"] / 1e6))
        return current_bounds


class CatsMoverSchema(CurrentMoversBaseSchema):
    '''static schema for CatsMover'''
    filename = FilenameSchema(
        save=True, isdatafile=True, test_equal=False, update=False
    )
    scale = SchemaNode(
        Bool(), missing=drop, save=True, update=True
    )
    scale_refpoint = WorldPoint(
        missing=drop, save=True, update=True
    )
    scale_value = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )

    # the following six could be shared with grid_current
    # in a currents base class
    down_cur_uncertain = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    up_cur_uncertain = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    right_cur_uncertain = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    left_cur_uncertain = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_eddy_diffusion = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_eddy_v0 = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    tide = TideSchema(
        missing=drop, save=True, update=True, save_reference=True
    )
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class CatsMover(CurrentMoversBase):

    _schema = CatsMoverSchema

    def __init__(self,
                 filename=None,
                 tide=None,
                 uncertain_duration=48,
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

        f = open(filename)
        header = f.readline()
        f.close()
        header.strip()
        fields = header.split(' ')
        if fields[0]!='DAG':
            raise ValueError('File has incorrect header line for Cats format: {0}'
                             .format(header))

        self._filename = filename

        # check if this is stored with cy_cats_mover?
        self.mover = CyCatsMover()
        self.mover.text_read(filename)
        if 'name' not in kwargs:
            kwargs['name'] = os.path.split(filename)[1]

        self.up_cur_uncertain = kwargs.pop('up_cur_uncertain', .3)
        self.down_cur_uncertain = kwargs.pop('down_cur_uncertain', -.3)
        self.right_cur_uncertain = kwargs.pop('right_cur_uncertain', .1)
        self.left_cur_uncertain = kwargs.pop('left_cur_uncertain', -.1)
        self.uncertain_eddy_diffusion = kwargs.pop('uncertain_eddy_diffusion',
                                                   0)
        self.uncertain_eddy_v0 = kwargs.pop('uncertain_eddy_v0', .1)

        self.scale = kwargs.pop('scale', self.mover.scale_type)
        self.scale_value = kwargs.pop('scale_value',
                                      self.mover.scale_value)
        # TODO: no need to check for None since properties that are None
        # are not persisted

        if 'scale_refpoint' in kwargs:
            self.scale_refpoint = kwargs.pop('scale_refpoint')
            #self.mover.compute_velocity_scale()

        super(CatsMover, self).__init__(uncertain_duration=uncertain_duration,
                                        **kwargs)

        self._tide = None
        if tide is not None:
            self.tide = tide

        if (self.scale and
            self.scale_value != 0.0 and
                self.scale_refpoint is None):
            raise TypeError("Provide a reference point in 'scale_refpoint'.")

        self.mover.compute_velocity_scale()

    def __repr__(self):
        return 'CatsMover(filename={0})'.format(self.filename)

    # Properties
    filename = property(lambda self: self._filename,
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
        if val is None:
            return
        if len(val) == 2:
            self.mover.ref_point = (val[0], val[1], 0.)
        else:
            self.mover.ref_point = val

    @property
    def tide(self):
        return self._tide

    @tide.setter
    def tide(self, tide_obj):
        if tide_obj is None:
            self._tide = tide_obj
            self.mover.unset_tide()
            return
        if not isinstance(tide_obj, Tide):
            raise TypeError('tide must be of type environment.Tide')

        if isinstance(tide_obj.cy_obj, CyShioTime):
            self.mover.set_shio(tide_obj.cy_obj)
        elif isinstance(tide_obj.cy_obj, CyOSSMTime):
            self.mover.set_ossm(tide_obj.cy_obj)
        else:
            raise TypeError('Tide.cy_obj attribute must be either '
                            'CyOSSMTime or CyShioTime type for CatsMover.')

        self._tide = tide_obj

    @property
    def data_start(self):
        if self.tide is not None:
            return self.tide.data_start
        else:
            return MinusInfTime()

    @property
    def data_stop(self):
        if self.tide is not None:
            return self.tide.data_stop
        else:
            return InfTime()

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
        self.mover.compute_velocity_scale()  # make sure ref_scale is up to date
        ref_scale = self.ref_scale

        if self._tide is not None:
            time_value, _err = self._tide.cy_obj.get_time_value(model_time)
            tide = time_value[0][0]
        else:
            tide = 1

        velocities['u'] *= ref_scale * tide
        velocities['v'] *= ref_scale * tide

        return velocities


class c_GridCurrentMoverSchema(CurrentMoversBaseSchema):
    filename = FilenameSchema(
        missing=drop, save=True, update=False, isdatafile=True, test_equal=False
    )
    topology_file = FilenameSchema(
        missing=drop, save=True, update=False, isdatafile=True, test_equal=False
    )
    current_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_along = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_cross = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    extrapolate = SchemaNode(
        Bool(), missing=drop, save=True, update=True
    )
    time_offset = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    is_data_on_cells = SchemaNode(
        Bool(), missing=drop, read_only=True
    )
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class c_GridCurrentMover(CurrentMoversBase):

    _schema = c_GridCurrentMoverSchema

    def __init__(self, filename,
                 topology_file=None,
                 extrapolate=False,
                 time_offset=0,
                 current_scale=1,
                 uncertain_along=0.5,
                 uncertain_across=0.25,
                 uncertain_cross=0.25,
                 num_method='Euler',
                 **kwargs):
        """
        Initialize a c_GridCurrentMover

        :param filename: absolute or relative path to the data file:
                         could be netcdf or filelist
        :param topology_file=None: absolute or relative path to topology file.
                                   If not given, the c_GridCurrentMover will
                                   compute the topology from the data file.

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

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

        uses super, ``super(c_GridCurrentMover,self).__init__(**kwargs)``
        """
        # if child is calling, the self.mover is set by child - do not reset
        if type(self) == c_GridCurrentMover:
            self.mover = CyGridCurrentMover()

        if not os.path.exists(filename):
            raise ValueError('Path for current file does not exist: {0}'
                             .format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'
                                 .format(topology_file))

        super(c_GridCurrentMover, self).__init__(**kwargs)

        # check if this is stored with cy_gridcurrent_mover?
        self.filename = filename
        self.name = os.path.split(filename)[1]

        # check if this is stored with cy_gridcurrent_mover?
        self.topology_file = topology_file
        self.current_scale = current_scale
        self.uncertain_along = uncertain_along
        self.uncertain_across = uncertain_across
        self.uncertain_cross = uncertain_cross

        self.mover.text_read(filename, topology_file)
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset * 3600.)

        self.num_method = num_method

        if self.topology_file is None:
            # this causes an error saving for currents that don't have topology
            #self.topology_file = filename + '.dat'
            #self.export_topology(self.topology_file)
            temp_topology_file = filename + '.dat'
            self.export_topology(temp_topology_file)
            if os.path.exists(temp_topology_file):
                self.topology_file = temp_topology_file

    def __repr__(self):
        return ('c_GridCurrentMover('
                'uncertain_duration={0.uncertain_duration},'
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'uncertain_cross={0.uncertain_cross}, '
                'uncertain_along={0.uncertain_along}, '
                'active_range={1.active_range}, '
                'on={1.on})'
                .format(self.mover, self))

    def __str__(self):
        return ('c_GridCurrentMover - current _state.\n'
                '  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_cross={0.uncertain_cross}\n'
                '  uncertain_along={0.uncertain_along}\n'
                '  active_range time={1.active_range}\n'
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

    @property
    def data_start(self):
        return sec_to_datetime(self.mover.get_start_time())

    @property
    def data_stop(self):
        return sec_to_datetime(self.mover.get_end_time())

    @property
    def num_method(self):
        return self._num_method

    @num_method.setter
    def num_method(self, val):
        self.mover.num_method = val
        self._num_method = val

    @property
    def is_data_on_cells(self):
        return self.mover._is_data_on_cells()

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
        :param time=0: model time in integer seconds
        """
        num_tri = self.mover.get_num_triangles()

        # will need to update this for regular grids
        if self.mover._is_triangle_grid():
            if self.mover._is_data_on_cells():
                num_cells = num_tri
            else:
                num_vertices = self.mover.get_num_points()
                num_cells = num_vertices
        elif self.mover._is_regular_grid():
            num_cells = self.mover.get_num_points()
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
        return self.mover.get_offset_time() / 3600.

    def get_start_time(self):
        """
        :this will be the real_data_start time (seconds).
        """
        return self.mover.get_start_time()

    def get_end_time(self):
        """
        :this will be the real_data_stop time (seconds).
        """
        return self.mover.get_end_time()

    def get_num_method(self):
        return self.mover.num_method


class IceMoverSchema(CurrentMoversBaseSchema):
    filename = FilenameSchema(
        missing=drop, save=True, isdatafile=True, test_equal=False, update=False
    )
    topology_file = FilenameSchema(
        missing=drop, save=True, isdatafile=True, test_equal=False, update=False
    )
    current_scale = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_along = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    uncertain_cross = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    extrapolate = SchemaNode(
        Bool(), missing=drop, save=True, update=True
    )


class IceMover(CurrentMoversBase):

    _schema = IceMoverSchema

    def __init__(self,
                 filename=None,
                 topology_file=None,
                 current_scale=1,
                 uncertain_along=0.5,
                 uncertain_cross=0.25,
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

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

        :param current_scale: Value to scale current data
        :param uncertain_duration: how often does a given uncertain element
                                   get reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param extrapolate: Allow current data to be extrapolated
                            before and after file data
        :param time_offset: Time zone shift if data is in GMT

        uses super, ``super(IceMover,self).__init__(**kwargs)``
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
        self.uncertain_along = uncertain_along
        self.uncertain_cross = uncertain_cross
        self.current_scale = current_scale

        super(IceMover, self).__init__(**kwargs)

    def __repr__(self):
        return ('IceMover('
                'uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'uncertain_cross={0.uncertain_cross}, '
                'uncertain_along={0.uncertain_along}, '
                'active_range={1.active_range}, '
                'on={1.on})'
                .format(self.mover, self))

    def __str__(self):
        return ('IceMover - current _state.\n'
                '  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_cross={0.uncertain_cross}\n'
                '  uncertain_along={0.uncertain_along}\n'
                '  active_range time={1.active_range}\n'
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

    def get_grid_bounding_box(self, grid_data=None, box_to_merge=None):
        '''
            Return a bounding box surrounding the grid data.

            :param grid_data: The point data of our grid.
            :type grid_data: A sequence of 3-tuples or 4-tuples containing
                             (long, lat) pairs.

            :param box_to_merge: A bounding box to surround in combination
                                 with our grid data.  This allows us to pad
                                 the bounding box that we generate.
            :type box_to_merge: A bounding box (extent) of the form:
                                ((left, bottom),
                                 (right, top))
        '''
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

        left, right = longs.min(), longs.max()
        bottom, top = lats.min(), lats.max()

        if (box_to_merge is not None and
                len(box_to_merge) == 2 and
                [len(p) for p in box_to_merge] == [2, 2]):
            if left > box_to_merge[0][0]:
                left = box_to_merge[0][0]

            if right < box_to_merge[1][0]:
                right = box_to_merge[1][0]

            if bottom > box_to_merge[0][1]:
                bottom = box_to_merge[0][1]

            if top < box_to_merge[1][1]:
                top = box_to_merge[1][1]

        return ((left, bottom), (right, top))

    def get_center_points(self):
        if self.mover._is_triangle_grid():
            return self.get_triangle_center_points()
        else:
            return self.get_cell_center_points()

    def get_scaled_velocities(self, model_time):
        """
        :param model_time=0: datetime in integer seconds
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
        num_cells = num_tri // 2

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


class CurrentCycleMoverSchema(c_GridCurrentMoverSchema):
    tide = TideSchema(
        missing=drop, save=True, update=True, save_reference=True
    )
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class CurrentCycleMover(c_GridCurrentMover):
    _schema = CurrentCycleMoverSchema

    _ref_as = 'current_cycle_mover'

    _req_refs = {'tide': Tide}

    def __init__(self,
                 filename=None,
                 topology_file=None,
                 tide=None,
                 **kwargs):
        """
        Initialize a CurrentCycleMover

        :param filename: Absolute or relative path to the data file:
                         could be netcdf or filelist
        :param topology_file=None: Absolute or relative path to topology file.
                                   If not given, the c_GridCurrentMover will
                                   compute the topology from the data file.
        :param tide: A gnome.environment.Tide object to be attached to
                     CatsMover

        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes

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

        self._tide = None
        if tide:
            self.tide = tide

        super(CurrentCycleMover, self).__init__(filename=filename,
                                                topology_file=topology_file,
                                                **kwargs)

    def __repr__(self):
        return ('CurrentCycletMover(uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'uncertain_cross={0.uncertain_cross}, '
                'uncertain_along={0.uncertain_along}, '
                'active_range={1.active_range}, '
                'on={1.on})'
                .format(self.mover, self))

    def __str__(self):
        return ('CurrentCycleMover - current _state.\n'
                '  uncertain_duration={0.uncertain_duration}\n'
                '  uncertain_time_delay={0.uncertain_time_delay}\n'
                '  uncertain_cross={0.uncertain_cross}\n'
                '  uncertain_along={0.uncertain_along}'
                '  active_range time={1.active_range}'
                '  current on/off status={1.on}'
                .format(self.mover, self))

    @property
    def tide(self):
        return self._tide

    @tide.setter
    def tide(self, tide_obj):
        if not isinstance(tide_obj, Tide):
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

    @property
    def data_start(self):
        if self.tide is not None:
            return self.tide.data_start
        else:
            return MinusInfTime()

    @property
    def data_stop(self):
        if self.tide is not None:
            return self.tide.data_stop
        else:
            return InfTime()

    @property
    def is_data_on_cells(self):
        return None

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
        :param time=0: datetime in integer seconds
        """
        num_tri = self.mover.get_num_triangles()

        # will need to update this for regular grids
        if self.mover._is_triangle_grid():
            if self.mover._is_data_on_cells():
                num_cells = num_tri
            else:
                num_vertices = self.mover.get_num_points()
                num_cells = num_vertices
        elif self.mover._is_regular_grid():
            num_cells = self.mover.get_num_points()
        else:
            num_cells = num_tri / 2

        vels = np.zeros(num_cells, dtype=basic_types.velocity_rec)

        self.mover.get_scaled_velocities(time, vels)

        return vels


class ComponentMoverSchema(ProcessSchema):
    '''static schema for ComponentMover'''
    filename1 = SchemaNode(
        String(), missing=drop,
        save=True, update=True, isdatafile=True, test_equal=False
    )
    filename2 = SchemaNode(
        String(), missing=drop,
        save=True, update=True, isdatafile=True, test_equal=False
    )
    scale_refpoint = WorldPoint(
        missing=drop, save=True, update=True
    )
    pat1_angle = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    pat1_speed = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    pat1_speed_units = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    pat1_scale_to_value = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    pat2_angle = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    pat2_speed = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    pat2_speed_units = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    pat2_scale_to_value = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    scale_by = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    wind = WindSchema(
        missing=drop, save=True, update=True, save_reference=True
    )
    data_start = SchemaNode(LocalDateTime(), read_only=True,
                            validator=convertible_to_seconds)
    data_stop = SchemaNode(LocalDateTime(), read_only=True,
                           validator=convertible_to_seconds)


class ComponentMover(CurrentMoversBase):
    _schema = ComponentMoverSchema

    _ref_as = 'component_mover'

    _req_refs = {'wind': Wind}

    def __init__(self,
                 filename1=None,
                 filename2=None,
                 wind=None,
                 scale_refpoint=None,
                 pat1_angle=0,
                 pat1_speed=10,
                 pat1_speed_units=2,
                 pat1_scale_to_value=0.1,
                 pat2_angle=90,
                 pat2_scale_to_value=0.1,
                 pat2_speed=10,
                 pat2_speed_units=2,
                 scale_by=0,
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

        if filename1 and not os.path.exists(filename1):
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
        super(ComponentMover, self).__init__(**kwargs)


        self._wind = None
        if wind is not None:
            self.wind = wind
        self.scale_by = scale_by
        self.scale_refpoint = scale_refpoint
        self.pat1_angle = pat1_angle
        self.pat1_speed = pat1_speed
        self.pat1_speed_units = pat1_speed_units
        self.pat1_scale_to_value = pat1_scale_to_value
        self.pat2_angle = pat2_angle
        self.pat2_scale_to_value = pat2_scale_to_value
        self.pat2_speed = pat2_speed
        self.pat2_speed_units = pat2_speed_units

    def __repr__(self):
        """
        unambiguous representation of object
        """
        return 'ComponentMover(filename={0})'.format(self.filename1)

    # Properties
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
                                  lambda self, val:
                                  setattr(self.mover, 'use_averaged_winds',
                                          val))

    wind_power_factor = property(lambda self: self.mover.wind_power_factor,
                                 lambda self, val: setattr(self.mover,
                                                           'wind_power_factor',
                                                           val))

    past_hours_to_average = property(lambda self: (self.mover
                                                   .past_hours_to_average),
                                     lambda self, val:
                                     setattr(self.mover,
                                             'past_hours_to_average', val))

    scale_factor_averaged_winds = property(lambda self: self.mover.scale_factor_averaged_winds,
                                           lambda self, val: setattr(self.mover,
                                                                     'scale_factor_averaged_winds',
                                                                     val))

    use_original_scale_factor = property(lambda self: self.mover.use_original_scale_factor,
                                         lambda self, val: setattr(self.mover,
                                                                   'use_original_scale_factor',
                                                                   val))

    @property
    def data_start(self):
        if self.wind is not None:
            return self.wind.data_start
        else:
            return MinusInfTime()

    @property
    def data_stop(self):
        if self.wind is not None:
            return self.wind.data_stop
        else:
            return InfTime()

    @property
    def scale_refpoint(self):
        return self.mover.ref_point

    @scale_refpoint.setter
    def scale_refpoint(self, val):
        '''
        Must be a tuple of length 2 or 3: (long, lat, z). If only (long, lat)
        is given, the set z = 0
        '''
        if val is None:
            return
        if len(val) == 2:
            self.mover.ref_point = (val[0], val[1], 0.)
        else:
            self.mover.ref_point = val

    @property
    def wind(self):
        return self._wind

    @wind.setter
    def wind(self, wind_obj):
        if not isinstance(wind_obj, Wind):
            self._wind = None
            return

        self.mover.set_ossm(wind_obj.ossm)
        self._wind = wind_obj

    def get_grid_data(self):
        """
            Invokes the GetToplogyHdl method of TriGridVel_c object
        """
        return self.get_triangles()

    def get_center_points(self):
        return self.get_triangle_center_points()

    def get_optimize_values(self, model_time):
        optimize_pat1 = self.mover.get_optimize_value(model_time, 1)
        optimize_pat2 = self.mover.get_optimize_value(model_time, 2)
        return optimize_pat1, optimize_pat2

    def get_scaled_velocities(self, model_time):
        """
        Get file values scaled to optimized
        check if pat2 exists
        """
        vels_pat1 = self.mover._get_velocity_handle(1)
        vels_pat2 = 0
        if self.filename2 is not None:
            vels_pat2 = self.mover._get_velocity_handle(2)

        optimize_pat1, optimize_pat2 = self.get_optimize_values(model_time)

        vels_pat1['u'] *= optimize_pat1
        vels_pat1['v'] *= optimize_pat1

        if optimize_pat2 != 0:
            vels_pat2['u'] *= optimize_pat2
            vels_pat2['v'] *= optimize_pat2
            vels_pat1['u'] += vels_pat2['u']
            vels_pat1['v'] += vels_pat2['v']

        return vels_pat1
