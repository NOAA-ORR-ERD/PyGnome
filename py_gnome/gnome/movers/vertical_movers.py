



import copy

from colander import (SchemaNode, Float)

from gnome.basic_types import world_point, world_point_type
from gnome.array_types import gat
from gnome.cy_gnome.cy_rise_velocity_mover import CyRiseVelocityMover

from gnome.movers import CyMover, ProcessSchema
from gnome.persist.base_schema import ObjTypeSchema


class RiseVelocityMoverSchema(ProcessSchema):
    pass
    #water_density = SchemaNode(Float(), save=True, update=True)
    #water_viscosity = SchemaNode(Float(), save=True, update=True)


class RiseVelocityMover(CyMover):
    """
    This mover class inherits from CyMover and contains CyRiseVelocityMover

    The real work is done by CyRiseVelocityMover.
    CyMover sets everything up that is common to all movers.
    """
    _schema = RiseVelocityMoverSchema

    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method.

        Optional parameters (kwargs) used to initialize CyRiseVelocityMover

        :param water_density: Default is 1020 kg/m3
        :param water_viscosity: Default is 1.e-6

        Remaining kwargs are passed onto Mover's __init__ using super.
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRiseVelocityMover()

        super(RiseVelocityMover, self).__init__(**kwargs)

        self.array_types['rise_vel'] = gat('rise_vel')

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        return ('RiseVelocityMover(active_range={0}, on={1})'
                .format(self.active_range, self.on))

    def get_move(self, sc, time_step, model_time_datetime):
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
                                self.spill_type)

        return (self.delta.view(dtype=world_point_type)
                .reshape((-1, len(world_point))))


class TamocRiseVelocityMover(RiseVelocityMover):
    """
    The only thing this adds (so far)

    are droplet_diameter and density array types
    """
    def __init__(self, *args, **kwargs):
        super(TamocRiseVelocityMover, self).__init__(*args, **kwargs)

        self.array_types.update({'density': gat('density'),
                                 'droplet_diameter': gat('droplet_diameter')})
