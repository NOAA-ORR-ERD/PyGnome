import copy

from gnome.utilities import serializable
from gnome.movers import CyMover
                
class RiseVelocityMover(CyMover, serializable.Serializable):
    """
    This mover class inherits from CyMover and contains CyRiseVelocityMover

    The real work is done by CyRiseVelocityMover.
    CyMover sets everything up that is common to all movers.
    """
    state = copy.deepcopy(CyMover.state)
    state.add(update=['water_density'], create=['water_density'])
    state.add(update=['water_viscosity'], create=['water_viscosity'])
    
    def __init__(self, **kwargs):
        """
        Uses super to invoke base class __init__ method. 
        
        Optional parameters (kwargs)
        :param water_density: Default is 1020 kg/m3
        :param water_viscosity: Default is 1.e-6
        
        Remaining kwargs are passed onto Mover's __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyRiseVelocityMover(water_density=kwargs.pop('water_density',1020),water_viscosity=kwargs.pop('water_viscosity',.000001))
        super(RiseVelocityMover,self).__init__(**kwargs)

    @property
    def water_density(self):
        return self.mover.water_density
    @property
    def water_viscosity(self):
        return self.mover.water_viscosity
    @water_density.setter
    def water_density(self, value):
        self.mover.water_density = value
    @water_viscosity.setter
    def water_viscosity(self, value):
        self.mover.water_viscosity = value

    def __repr__(self):
        """
        .. todo:: 
            We probably want to include more information.
        """
        return "RiseVelocityMover(water_density=%s,water_viscosity=%s,active_start=%s, active_stop=%s, on=%s)" % (self.water_density,self.water_viscosity,self.active_start, self.active_stop, self.on)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Override base class functionality because mover has a different get_move signature
        
        :param sc: an instance of the gnome.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """
        self.prepare_data_for_get_move(sc, model_time_datetime)
        
        if self.active and len(self.positions) > 0: 
            self.mover.get_move(  self.model_time,
                                  time_step, 
                                  self.positions,
                                  self.delta,
                                  sc['rise_velocity'],
                                  sc['density'],
                                  sc['droplet_size'],
                                  self.status_codes,
                                  self.spill_type,
                                  0)    # only ever 1 spill_container so this is always 0!
            
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))

