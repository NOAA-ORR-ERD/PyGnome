import copy

from colander import (SchemaNode, Float)

from gnome.persist.base_schema import ObjType
from gnome.utilities import serializable
from gnome.movers import CyMover, MoverSchema
from gnome.cy_gnome.cy_rise_velocity_mover import CyRiseVelocityMover
from gnome.array_types import rise_vel
from gnome.basic_types import world_point, world_point_type


class RiseVelocityMoverSchema(ObjType, MoverSchema):
    water_density = SchemaNode(Float())
    water_viscosity = SchemaNode(Float())


class RiseVelocityMover(CyMover, serializable.Serializable):

    """
    This mover class inherits from CyMover and contains CyRiseVelocityMover

    The real work is done by CyRiseVelocityMover.
    CyMover sets everything up that is common to all movers.
    """

    _state = copy.deepcopy(CyMover._state)
    #_state.add(update=['water_density'], save=['water_density'])
    #_state.add(update=['water_viscosity'], save=['water_viscosity'])
    _schema = RiseVelocityMoverSchema

    def __init__(
        self,
       # water_density=1020,
       # water_viscosity=1.e-6,
        **kwargs
        ):
        """
        Uses super to invoke base class __init__ method.

        Optional parameters (kwargs) used to initialize CyRiseVelocityMover

        :param water_density: Default is 1020 kg/m3
        :param water_viscosity: Default is 1.e-6

        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """

       # self.mover = CyRiseVelocityMover(water_density, water_viscosity)
        self.mover = CyRiseVelocityMover()
        super(RiseVelocityMover, self).__init__(**kwargs)
        self.array_types.update({'rise_vel': rise_vel})

#     @property
#     def water_density(self):
#         return self.mover.water_density
# 
#     @property
#     def water_viscosity(self):
#         return self.mover.water_viscosity
# 
#     @water_density.setter
#     def water_density(self, value):
#         self.mover.water_density = value
# 
#     @water_viscosity.setter
#     def water_viscosity(self, value):
#         self.mover.water_viscosity = value

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """

        return ('RiseVelocityMover(active_start={0}, active_stop={1},'
                ' on={2})').format(self.active_start, self.active_stop, self.on)

    def get_move(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
        """
        Override base class functionality because mover has a different
        get_move signature

        :param sc: an instance of the gnome.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time
            object
        """

        self.prepare_data_for_get_move(sc, model_time_datetime)

        if self.active and len(self.positions) > 0:
            self.mover.get_move(self.model_time,
                time_step,
                self.positions,
                self.delta,
                sc['rise_vel'],
                self.status_codes,
                self.spill_type,
                )

        return self.delta.view(dtype=world_point_type).reshape((-1,
                len(world_point)))
