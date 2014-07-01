'''
Movers using currents and tides as forcing functions
'''

import os
import copy

from colander import (SchemaNode, Bool, String, Float, drop)

from gnome.persist.base_schema import ObjType, WorldPoint, LongLat

from gnome.movers import CyMover, MoverSchema
from gnome import environment
from gnome.utilities import serializable
from gnome.cy_gnome import cy_cats_mover, cy_shio_time, cy_ossm_time, \
    cy_gridcurrent_mover, cy_component_mover, cy_currentcycle_mover


class CatsMoverSchema(ObjType, MoverSchema):
    '''static schema for CatsMover'''
    filename = SchemaNode(String(), missing=drop)
    scale = SchemaNode(Bool())
    scale_refpoint = WorldPoint(missing=drop)
    scale_value = SchemaNode(Float())
    #the following six could be shared with grid_current in a currents base class
    uncertain_duration = SchemaNode(Float(), missing=drop)
    uncertain_time_delay = SchemaNode(Float(), missing=drop)
    down_cur_uncertain = SchemaNode(Float(), missing=drop)
    up_cur_uncertain = SchemaNode(Float(), missing=drop)
    right_cur_uncertain = SchemaNode(Float(), missing=drop)
    left_cur_uncertain = SchemaNode(Float(), missing=drop)
    uncertain_eddy_diffusion = SchemaNode(Float(), missing=drop)
    uncertain_eddy_v0 = SchemaNode(Float(), missing=drop)


class CatsMover(CyMover, serializable.Serializable):

    _state = copy.deepcopy(CyMover._state)

    _update = ['scale', 'scale_refpoint', 'scale_value',
                      'uncertain_duration', 'uncertain_time_delay',
                      'up_cur_uncertain', 'down_cur_uncertain',
                      'right_cur_uncertain', 'left_cur_uncertain',
                      'uncertain_eddy_diffusion', 'uncertain_eddy_v0']
    _create = []
    _create.extend(_update)
    _state.add(update=_update, save=_create)
    _state.add_field([serializable.Field('filename', save=True,
                                read=True, isdatafile=True, test_for_eq=False),
                      serializable.Field('tide', save=True,
                                update=True, save_reference=True)])
    _schema = CatsMoverSchema

    def __init__(
        self,
        filename,
        tide=None,
        **kwargs
        ):
        """
        Uses super to invoke base class __init__ method. 
        
        :param filename: file containing currents patterns for Cats 
        
        Optional parameters (kwargs). Defaults are defined by CyCatsMover object.
        
        :param tide: a gnome.environment.Tide object to be attached to CatsMover
        :param scale: a boolean to indicate whether to scale value at reference point or not
        :param scale_value: value used for scaling at reference point
        :param scale_refpoint: reference location (long, lat, z). The scaling applied to all data is determined by scaling the 
                               raw value at this location.
        
        :param uncertain_duration: how often does a given uncertain element gets reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param up_cur_uncertain: Scale for uncertainty along the flow
        :param down_cur_uncertain: Scale for uncertainty along the flow
        :param right_cur_uncertain: Scale for uncertainty across the flow
        :param left_cur_uncertain: Scale for uncertainty across the flow
        :param uncertain_eddy_diffusion: Diffusion coefficient for eddy diffusion. Default is 0.
        :param uncertain_eddy_v0: Default is .1 (Check that this is still used)
        Remaining kwargs are passed onto Mover's __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """

        if not os.path.exists(filename):
            raise ValueError('Path for Cats filename does not exist: {0}'.format(filename))

        self.filename = filename  # check if this is stored with cy_cats_mover?
        self.mover = cy_cats_mover.CyCatsMover()
        self.mover.text_read(filename)
        self.name = os.path.split(filename)[1]

        self._tide = None
        if tide is not None:
            self.tide = tide

        self.scale = kwargs.pop('scale', self.mover.scale_type)
        self.scale_value = kwargs.get('scale_value',
                self.mover.scale_value)

        self.uncertain_start_time = kwargs.pop('uncertain_duration',48)
        self.uncertain_time_delay = kwargs.pop('uncertain_time_delay', 0)
        self.up_cur_uncertain = kwargs.pop('up_cur_uncertain', .3)
        self.down_cur_uncertain = kwargs.pop('down_cur_uncertain', -.3)
        self.right_cur_uncertain = kwargs.pop('right_cur_uncertain', .1)
        self.left_cur_uncertain = kwargs.pop('left_cur_uncertain', -.1)
        self.uncertain_eddy_diffusion = kwargs.pop('uncertain_eddy_diffusion', 0)
        self.uncertain_eddy_v0 = kwargs.pop('uncertain_eddy_v0', .1)
        # todo: no need to check for None since properties that are None are not persisted

        if 'scale_refpoint' in kwargs:
            self.scale_refpoint = kwargs.pop('scale_refpoint')

        if self.scale and self.scale_value != 0.0 \
            and self.scale_refpoint is None:
            raise TypeError("Provide a reference point in 'scale_refpoint'."
                            )

        super(CatsMover, self).__init__(**kwargs)

    def __repr__(self):
        """
        unambiguous representation of object
        """

        info = 'CatsMover(filename={0})'.format(self.filename)
        return info

    # Properties

    scale = property(lambda self: bool(self.mover.scale_type),
                     lambda self, val: setattr(self.mover,
                            'scale_type', int(val)))
    scale_refpoint = property(lambda self: self.mover.ref_point,
                              lambda self, val: setattr(self.mover,
                              'ref_point', val))

    scale_value = property(lambda self: self.mover.scale_value,
                           lambda self, val: setattr(self.mover,
                           'scale_value', val))

#     @property
#     def uncertain_duration(self):
#         return self._seconds_to_hours(self.mover.uncertain_duration)
# 
#     @uncertain_duration.setter
#     def uncertain_duration(self, val):
#         self.mover.uncertain_duration = self._hours_to_seconds(val)
# 
#     @property
#     def uncertain_time_delay(self):
#         return self._seconds_to_hours(self.mover.uncertain_time_delay)
# 
#     @uncertain_time_delay.setter
#     def uncertain_time_delay(self, val):
#         self.mover.uncertain_time_delay = self._hours_to_seconds(val)
# 
    uncertain_duration = property(lambda self: \
                                  self.mover.uncertain_duration/3600.,
                                  lambda self, val: setattr(self.mover,
                                  'uncertain_duration', val*3600.))

    uncertain_time_delay = property(lambda self: \
                                    self.mover.uncertain_time_delay/3600.,
                                    lambda self, val: \
                                    setattr(self.mover,
                                    'uncertain_time_delay', val*3600.))

    up_cur_uncertain = property(lambda self: \
            self.mover.up_cur_uncertain, lambda self, val: \
            setattr(self.mover, 'up_cur_uncertain', val))

    down_cur_uncertain = property(lambda self: \
            self.mover.down_cur_uncertain, lambda self, val: \
            setattr(self.mover, 'down_cur_uncertain', val))

    right_cur_uncertain = property(lambda self: \
            self.mover.right_cur_uncertain, lambda self, val: \
            setattr(self.mover, 'right_cur_uncertain', val))

    left_cur_uncertain = property(lambda self: \
            self.mover.left_cur_uncertain, lambda self, val: \
            setattr(self.mover, 'left_cur_uncertain', val))

    uncertain_eddy_diffusion = property(lambda self: \
            self.mover.uncertain_eddy_diffusion, lambda self, val: \
            setattr(self.mover, 'uncertain_eddy_diffusion', val))

    uncertain_eddy_v0 = property(lambda self: \
            self.mover.uncertain_eddy_v0, lambda self, val: \
            setattr(self.mover, 'uncertain_eddy_v0', val))

    @property
    def tide(self):
        return self._tide

    @tide.setter
    def tide(self, tide_obj):
        if not isinstance(tide_obj, environment.Tide):
            raise TypeError('tide must be of type environment.Tide')

        if isinstance(tide_obj.cy_obj, cy_shio_time.CyShioTime):
            self.mover.set_shio(tide_obj.cy_obj)
        elif isinstance(tide_obj.cy_obj, cy_ossm_time.CyOSSMTime):
            self.mover.set_ossm(tide_obj.cy_obj)
        else:
            raise TypeError('Tide.cy_obj attribute must be either CyOSSMTime or CyShioTime type for CatsMover.'
                            )

        self._tide = tide_obj

    def serialize(self, json_='webapi'):
        """
        Since 'wind' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to WindMover schema
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        if json_ == 'webapi' and 'tide' in toserial:
            schema.add(environment.TideSchema(name='tide'))

        return schema.serialize(toserial)

    @classmethod
    def deserialize(cls, json_):
        """
        append correct schema for wind object
        """
        schema = cls._schema()
        if 'tide' in json_:
            schema.add(environment.TideSchema())
        _to_dict = schema.deserialize(json_)

        return _to_dict


class GridCurrentMoverSchema(ObjType, MoverSchema):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)
    current_scale = SchemaNode(Float(), default=1)
    uncertain_duration = SchemaNode(Float(), default=24)
    uncertain_time_delay = SchemaNode(Float(), default=0)
    uncertain_along = SchemaNode(Float(), default=.5)
    uncertain_cross = SchemaNode(Float(), default=.25)


class GridCurrentMover(CyMover, serializable.Serializable):

    _update = ['uncertain_duration', 'uncertain_time_delay',
               'uncertain_cross', 'uncertain_along', 'current_scale']
    _save = ['uncertain_duration', 'uncertain_time_delay',
               'uncertain_cross', 'uncertain_along', 'current_scale']
    _state = copy.deepcopy(CyMover._state)

    _state.add(update=_update,save=_save)
    _state.add_field([serializable.Field('filename', save=True,
                    read=True, isdatafile=True, test_for_eq=False),
                    serializable.Field('topology_file', save=True,
                    read=True, isdatafile=True, test_for_eq=False)])
    _schema = GridCurrentMoverSchema

    def __init__(
        self,
        filename,
        topology_file=None,
        extrapolate=False,
        time_offset=0,
#         current_scale=1,
#         uncertain_duration=timedelta(hours=24),
#         uncertain_time_delay=timedelta(hours=0),
#         uncertain_along=0.5,
#         uncertain_cross=.25,
        **kwargs
        ):
        """
        Initialize a GridCurrentMover

        :param filename: absolute or relative path to the data file: could be netcdf or filelist
        :param topology_file=None: absolute or relative path to topology file. If not given, the
                                   GridCurrentMover will copmute the topology from the data file.
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param current_scale: Value to scale current data
        :param uncertain_duration: how often does a given uncertain element gets reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param extrapolate: Allow current data to be extrapolated before and after file data
        :param time_offset: Time zone shift if data is in GMT 

        uses super, super(GridCurrentMover,self).__init__(\*\*kwargs)
        """

        # # NOTE: will need to add uncertainty parameters and other dialog fields
        # #       use super with kwargs to invoke base class __init__

        if not os.path.exists(filename):
            raise ValueError('Path for current file does not exist: {0}'.format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'.format(topology_file))

        self.filename = filename  # check if this is stored with cy_gridcurrent_mover?
        self.topology_file = topology_file  # check if this is stored with cy_gridcurrent_mover?
        #self.mover = cy_gridcurrent_mover.CyGridCurrentMover()
        self.mover = \
        cy_gridcurrent_mover.CyGridCurrentMover(current_scale=kwargs.pop('current_scale', 1),
             uncertain_duration=3600.*kwargs.pop('uncertain_duration', 24),
             uncertain_time_delay=3600.*kwargs.pop('uncertain_time_delay', 0),
             uncertain_along=kwargs.pop('uncertain_along', 0.5),
             uncertain_cross=kwargs.pop('uncertain_cross', 0.25))
        self.mover.text_read(filename, topology_file)
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset*3600.)

        super(GridCurrentMover, self).__init__(**kwargs)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """

        info = \
            'GridCurrentMover( uncertain_duration={0.uncertain_duration},' \
            + 'uncertain_time_delay={0.uncertain_time_delay}, '\
            + 'uncertain_cross={0.uncertain_cross}, ' \
            + 'uncertain_along={0.uncertain_along}, '\
            + 'active_start={1.active_start}, active_stop={1.active_stop}, '\
            + 'on={1.on})'
        return info.format(self.mover, self)

    def __str__(self):
        info = 'GridCurrentMover - current _state.\n' \
            + '  uncertain_duration={0.uncertain_duration}\n' \
            + '  uncertain_time_delay={0.uncertain_time_delay}\n' \
            + '  uncertain_cross={0.uncertain_cross}\n' \
            + '  uncertain_along={0.uncertain_along}' \
            + '  active_start time={1.active_start}' \
            + '  active_stop time={1.active_stop}' \
            + '  current on/off status={1.on}'
        return info.format(self.mover, self)

    # Define properties using lambda functions: uses lambda function, which are
    #accessible via fget/fset as follows:
    uncertain_duration = property(lambda self: \
                                  self.mover.uncertain_duration/3600.,
                                  lambda self, val: setattr(self.mover,
                                  'uncertain_duration', val*3600.))

    uncertain_time_delay = property(lambda self: \
                                    self.mover.uncertain_time_delay/3600.,
                                    lambda self, val: \
                                    setattr(self.mover,
                                    'uncertain_time_delay', val*3600.))

    uncertain_cross = property(lambda self: \
            self.mover.uncertain_cross, lambda self, val: \
            setattr(self.mover, 'uncertain_cross', val))

    uncertain_along = property(lambda self: \
            self.mover.uncertain_along, lambda self, val: \
            setattr(self.mover, 'uncertain_along', val))

    current_scale = property(lambda self: \
            self.mover.current_scale, lambda self, val: \
            setattr(self.mover, 'current_scale', val))

    extrapolate = property(lambda self: \
            self.mover.extrapolate, lambda self, val: \
            setattr(self.mover, 'extrapolate', val))

    time_offset = property(lambda self: \
            self.mover.time_offset/3600., lambda self, val: \
            setattr(self.mover, 'time_offset', val*3600.))

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where topology file will be written.
        """

        if topology_file is None:
            raise ValueError('Topology file path required: {0}'.format(topology_file))

        self.mover.export_topology(topology_file)

    def extrapolate_in_time(self, extrapolate):
        """
        :param extrapolate=false: allow current data to be extrapolated before and after file data.
        """

        self.mover.extrapolate_in_time(extrapolate)

    def offset_time(self, time_offset):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset (hours).
        """

        self.mover.offset_time(time_offset*3600.)

    def get_offset_time(self):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset (hours).
        """

        off_set_time = self.mover.get_offset_time()/3600.
        return (self.mover.get_offset_time())/3600.


class CurrentCycleMoverSchema(ObjType, MoverSchema):
    filename = SchemaNode(String(), missing=drop)
    topology_file = SchemaNode(String(), missing=drop)
    current_scale = SchemaNode(Float(), default=1)
    uncertain_duration = SchemaNode(Float(), default=24)
    uncertain_time_delay = SchemaNode(Float(), default=0)
    uncertain_along = SchemaNode(Float(), default=.5)
    uncertain_cross = SchemaNode(Float(), default=.25)


class CurrentCycleMover(CyMover, serializable.Serializable):

    _update = ['uncertain_duration', 'uncertain_time_delay',
               'uncertain_cross', 'uncertain_along', 'current_scale']
    _save = ['uncertain_duration', 'uncertain_time_delay',
               'uncertain_cross', 'uncertain_along', 'current_scale']
    _state = copy.deepcopy(CyMover._state)

    _state.add(update=_update,save=_save)
    _state.add_field([serializable.Field('filename', save=True,
                    read=True, isdatafile=True, test_for_eq=False),
                    serializable.Field('topology_file', save=True,
                    read=True, isdatafile=True, test_for_eq=False),
                      serializable.Field('tide', save=True,
                                update=True, save_reference=True)])
    _schema = CurrentCycleMoverSchema

    def __init__(
        self,
        filename,
        topology_file=None,
        extrapolate=False,
        time_offset=0,
        tide=None,
#         current_scale=1,
#         uncertain_duration=timedelta(hours=24),
#         uncertain_time_delay=timedelta(hours=0),
#         uncertain_along=0.5,
#         uncertain_cross=.25,
        **kwargs
        ):
        """
        Initialize a CurrentCycleMover

        :param filename: absolute or relative path to the data file: could be netcdf or filelist
        :param topology_file=None: absolute or relative path to topology file. If not given, the
                                   GridCurrentMover will copmute the topology from the data file.
        :param tide: a gnome.environment.Tide object to be attached to CatsMover
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param current_scale: Value to scale current data
        :param uncertain_duration: how often does a given uncertain element gets reset
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param uncertain_cross: Scale for uncertainty perpendicular to the flow
        :param uncertain_along: Scale for uncertainty parallel to the flow
        :param extrapolate: Allow current data to be extrapolated before and after file data
        :param time_offset: Time zone shift if data is in GMT 

        uses super: super(CurrentCycleMover,self).__init__(**kwargs)
        """

        # # NOTE: will need to add uncertainty parameters and other dialog fields
        # #       use super with kwargs to invoke base class __init__

        if not os.path.exists(filename):
            raise ValueError('Path for current file does not exist: {0}'.format(filename))

        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError('Path for Topology file does not exist: {0}'.format(topology_file))

        self.filename = filename  # check if this is stored with cy_currentcycle_mover?
        self.topology_file = topology_file  # check if this is stored with cy_currentcycle_mover?
        #self.mover = cy_currentcycle_mover.CyCurrentCycleMover()
        self.mover = \
        cy_currentcycle_mover.CyCurrentCycleMover(current_scale=kwargs.pop('current_scale', 1),
             uncertain_duration=3600.*kwargs.pop('uncertain_duration', 24),
             uncertain_time_delay=3600.*kwargs.pop('uncertain_time_delay', 0),
             uncertain_along=kwargs.pop('uncertain_along', 0.5),
             uncertain_cross=kwargs.pop('uncertain_cross', 0.25))
        self.mover.text_read(filename, topology_file)
        self.mover.extrapolate_in_time(extrapolate)
        self.mover.offset_time(time_offset*3600.)

        self._tide = None
        if tide is not None:
            self.tide = tide
            #print "self._tide"
            #print self._tide
            #print self.tide

        super(CurrentCycleMover, self).__init__(**kwargs)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """

        info = \
            'GridCurrentMover( uncertain_duration={0.uncertain_duration},' \
            + 'uncertain_time_delay={0.uncertain_time_delay}, '\
            + 'uncertain_cross={0.uncertain_cross}, ' \
            + 'uncertain_along={0.uncertain_along}, '\
            + 'active_start={1.active_start}, active_stop={1.active_stop}, '\
            + 'on={1.on})'
        return info.format(self.mover, self)

    def __str__(self):
        info = 'GridCurrentMover - current _state.\n' \
            + '  uncertain_duration={0.uncertain_duration}\n' \
            + '  uncertain_time_delay={0.uncertain_time_delay}\n' \
            + '  uncertain_cross={0.uncertain_cross}\n' \
            + '  uncertain_along={0.uncertain_along}' \
            + '  active_start time={1.active_start}' \
            + '  active_stop time={1.active_stop}' \
            + '  current on/off status={1.on}'
        return info.format(self.mover, self)

    # Define properties using lambda functions: uses lambda function, which are
    #accessible via fget/fset as follows:
    uncertain_duration = property(lambda self: \
                                  self.mover.uncertain_duration/3600.,
                                  lambda self, val: setattr(self.mover,
                                  'uncertain_duration', val*3600.))

    uncertain_time_delay = property(lambda self: \
                                    self.mover.uncertain_time_delay/3600.,
                                    lambda self, val: \
                                    setattr(self.mover,
                                    'uncertain_time_delay', val*3600.))

    uncertain_cross = property(lambda self: \
            self.mover.uncertain_cross, lambda self, val: \
            setattr(self.mover, 'uncertain_cross', val))

    uncertain_along = property(lambda self: \
            self.mover.uncertain_along, lambda self, val: \
            setattr(self.mover, 'uncertain_along', val))

    current_scale = property(lambda self: \
            self.mover.current_scale, lambda self, val: \
            setattr(self.mover, 'current_scale', val))

    extrapolate = property(lambda self: \
            self.mover.extrapolate, lambda self, val: \
            setattr(self.mover, 'extrapolate', val))

    time_offset = property(lambda self: \
            self.mover.time_offset/3600., lambda self, val: \
            setattr(self.mover, 'time_offset', val*3600.))

    @property
    def tide(self):
        return self._tide

    @tide.setter
    def tide(self, tide_obj):
        if not isinstance(tide_obj, environment.Tide):
            raise TypeError('tide must be of type environment.Tide')

        if isinstance(tide_obj.cy_obj, cy_shio_time.CyShioTime):
            self.mover.set_shio(tide_obj.cy_obj)
        elif isinstance(tide_obj.cy_obj, cy_ossm_time.CyOSSMTime):
            self.mover.set_ossm(tide_obj.cy_obj)
        else:
            raise TypeError('Tide.cy_obj attribute must be either CyOSSMTime or CyShioTime type for CurrentCycleMover.'
                            )

        self._tide = tide_obj

    def serialize(self, json_='webapi'):
        """
        Since 'tide' property is saved as a reference when used in save file
        and 'save' option, need to add appropriate node to CurrentCycleMover schema
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
        _to_dict = schema.deserialize(json_)

        return _to_dict

    def export_topology(self, topology_file):
        """
        :param topology_file=None: absolute or relative path where topology file will be written.
        """

        if topology_file is None:
            raise ValueError('Topology file path required: {0}'.format(topology_file))

        self.mover.export_topology(topology_file)

    def extrapolate_in_time(self, extrapolate):
        """
        :param extrapolate=false: allow current data to be extrapolated before and after file data.
        """

        self.mover.extrapolate_in_time(extrapolate)

    def offset_time(self, time_offset):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset (hours).
        """

        self.mover.offset_time(time_offset*3600.)

    def get_offset_time(self):
        """
        :param offset_time=0: allow data to be in GMT with a time zone offset (hours).
        """

        off_set_time = self.mover.get_offset_time()/3600.
        return (self.mover.get_offset_time())/3600.


class ComponentMoverSchema(ObjType, MoverSchema):
    '''static schema for ComponentMover'''
    filename1 = SchemaNode(String(), missing=drop)
    filename2 = SchemaNode(String(), missing=drop)
    #scale = SchemaNode(Bool())
    #ref_point = WorldPoint(missing=drop)
    ref_point = LongLat(missing=drop)
    #scale_value = SchemaNode(Float())


class ComponentMover(CyMover, serializable.Serializable):

    _state = copy.deepcopy(CyMover._state)

    _update = [ 'ref_point', 'pat1_angle', 'pat1_speed', 'pat1_speed_units', 'pat1_scale_to_value',
                    'pat2_angle', 'pat2_speed', 'pat2_speed_units', 'pat2_scale_to_value']
    _create = []
    _create.extend(_update)
    _state.add(update=_update, save=_create)
    _state.add_field([serializable.Field('filename1', save=True,
                    read=True, isdatafile=True, test_for_eq=False),
                      serializable.Field('filename2', save=True,
                    read=True, isdatafile=True, test_for_eq=False),
                      serializable.Field('wind', save=True,
                                update=True, save_reference=True)])
    _schema = ComponentMoverSchema

    def __init__(
        self,
        filename1,
        filename2=None,
        wind=None,
        **kwargs
        ):
        """
        Uses super to invoke base class __init__ method. 
        
        :param filename: file containing currents for first Cats pattern
        
        Optional parameters (kwargs). Defaults are defined by CyCatsMover object.
        
        :param filename: file containing currents for second Cats pattern
        
        :param wind: a gnome.environment.Wind object to be used to drive the CatsMovers
        will want a warning that mover will not be active without a wind
        :param scale: a boolean to indicate whether to scale value at reference point or not
        :param scale_value: value used for scaling at reference point
        :param scale_refpoint: reference location (long, lat, z). The scaling applied to all data is determined by scaling the 
                               raw value at this location.
        
        Remaining kwargs are passed onto Mover's __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """

        if not os.path.exists(filename1):
            raise ValueError('Path for Cats filename1 does not exist: {0}'.format(filename1))

        if filename2 is not None:
            if not os.path.exists(filename2):
                raise ValueError('Path for Cats filename2 does not exist: {0}'.format(filename2))

        self.filename1 = filename1  
        self.filename2 = filename2  

        self.mover = cy_component_mover.CyComponentMover()	# pass in parameters
        self.mover.text_read(filename1, filename2)

        self._wind = None
        if wind is not None:
            self.wind = wind

        #self.scale = kwargs.pop('scale', self.mover.scale_type)
        #self.scale_value = kwargs.get('scale_value',
                #self.mover.scale_value)

        # todo: no need to check for None since properties that are None are not persisted

	
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

        info = 'ComponentMover(filename={0})'.format(self.filename1)
        return info

    # Properties

#     scale_type = property(lambda self: bool(self.mover.scale_type),
#                      lambda self, val: setattr(self.mover, 'scale_type'
#                      , int(val)))
                     
#     scale_by = property(lambda self: bool(self.mover.scale_by),
#                      lambda self, val: setattr(self.mover, 'scale_by'
#                      , int(val)))

    ref_point = property(lambda self: self.mover.ref_point,
                              lambda self, val: setattr(self.mover,
                              'ref_point', val))

    pat1_angle = property(lambda self: self.mover.pat1_angle,
                           lambda self, val: setattr(self.mover,
                           'pat1_angle', val))

    pat1_speed = property(lambda self: self.mover.pat1_speed,
                           lambda self, val: setattr(self.mover,
                           'pat1_speed', val))

    pat1_speed_units = property(lambda self: self.mover.pat1_speed_units,
                           lambda self, val: setattr(self.mover,
                           'pat1_speed_units', val))

    pat1_scale_to_value = property(lambda self: self.mover.pat1_scale_to_value,
                           lambda self, val: setattr(self.mover,
                           'pat1_scale_to_value', val))

    pat2_angle = property(lambda self: self.mover.pat2_angle,
                           lambda self, val: setattr(self.mover,
                           'pat2_angle', val))

    pat2_speed = property(lambda self: self.mover.pat2_speed,
                           lambda self, val: setattr(self.mover,
                           'pat2_speed', val))

    pat2_speed_units = property(lambda self: self.mover.pat2_speed_units,
                           lambda self, val: setattr(self.mover,
                           'pat2_speed_units', val))

    pat2_scale_to_value = property(lambda self: self.mover.pat2_scale_to_value,
                           lambda self, val: setattr(self.mover,
                           'pat2_scale_to_value', val))

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

