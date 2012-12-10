import numpy as np

from gnome.utilities import time_utils, transforms
from gnome import basic_types
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_random_mover import CyRandomMover

from gnome.helpers import convert

class Mover(object):
    """
    Base class from which all Python movers can inherit
    
    It defines the interface for a Python mover. The model expects the methods defined here. 
    The get_move(...) method needs to be implemented by the derived class.  
    """
    def __init__(self, is_active=True, **kwargs):
        """
        During init, it defaults is_active = True
        """
        self._is_active = True
        super(Mover,self).__init__(**kwargs)
        
    @property
    def id(self):
        """
        Return an ID value for this mover.

        This method uses Python's builtin `id()` function to identify the
        object. Override it for more exotic forms of identification.

        :return: the integer ID returned by id() for this object
        """
        return id(self)
    
    # Methods for is_active property definition
    def get_is_active(self):
       return self._is_active
    
    def set_is_active(self, value):
       self._is_active = value
    
    is_active = property(get_is_active, set_is_active)
    
    def datetime_to_seconds(self, model_time):
        """
        put the time conversion call here - incase we decide to change it, it only updates here
        """
        return time_utils.date_to_sec(model_time)
        
    def prepare_for_model_run(self):
        """
        override method in derived class if the mover needs to perform any actions at beginning of model run 
        """
        pass

    def prepare_for_model_step(self, model_time_datetime, time_step,
                               uncertain_spills_count=0, uncertain_spills_size=None):
        """
        default implementation of prepare_for_model_step(...)
        
        It checks the inputs for uncertainty spills are valid. 
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)
        if uncertain_spills_count < 0:
            raise ValueError("The uncertain_spills_count cannot be less than 0")
        elif uncertain_spills_count > 0:
            """" preparing for a certainty spill """
            if uncertain_spills_size is None:
                raise ValueError("uncertain_spills_size cannot be None if uncertain_spills_count is greater than 0")
            
            if len(uncertain_spills_size) != uncertain_spills_count:
                raise ValueError("uncertain_spills_size needs an entry for each of the uncertain spills")
            
            
    def get_move(self, spill, time_step, model_time, uncertain_spill_number=0):
        """
        Not implemented in base class. 
        
        Each class that derives from Mover object must implement it's own get_move
        """
        raise NotImplementedError("Each mover that derives from Mover base class must implement get_move(...)")
    
    def model_step_is_done(self):
        """
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc in here in subclassed movers.
        """
        pass 


class CyMover(Mover):
    """
    Base class for python wrappers around cython movers
    All movers (CyWindMover, CyRandomMover) will need to extract info from spill object.
    Put this code and any other code common to all movers here
    
    Base class assumes any derived class will instantiate a 'mover' object that
    has methods like: prepare_for_model_run, prepare_for_model_step,
    """
    def __init__(self, is_active=True, **kwargs):
        super(CyMover,self).__init__(is_active=is_active,**kwargs)
        
    def prepare_for_model_run(self):
        """
        calls the underlying cython mover's prepare_for_model_run() 
        """
        self.mover.prepare_for_model_run()

    def prepare_for_model_step(self, model_time_datetime, time_step,
                               uncertain_spills_count=0, uncertain_spills_size=None):
        """
        default implementation of prepare_for_model_step(...)
        
        Checks the inputs for uncertainty spills are valid. Then it invokes the 
        mover's prepare_for_model_step. The 'mover' is a member of this class, but it is
        instantiated by the derived class. 
        """
        super(CyMover,self).prepare_for_model_step(model_time_datetime, time_step, uncertain_spills_count, uncertain_spills_size)
        #Mover.prepare_for_model_step(self, model_time_datetime, time_step, uncertain_spills_count, uncertain_spills_size)
        self.mover.prepare_for_model_step(self.model_time, time_step, uncertain_spills_count, uncertain_spills_size)

    def prepare_data_for_get_move(self, spill, model_time_datetime):
        """
        organizes the spill object into inputs for calling with Cython wrapper's get_move(...)
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param model_time_datetime: model time as a date time object
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)
        
        # Get the data:
        try:
            self.positions      = spill['positions']
            self.status_codes   = spill['status_codes']
            
            if spill.is_uncertain:
                self.spill_type = basic_types.spill_type.uncertainty
            else:
                self.spill_type = basic_types.spill_type.forecast
            
            # make sure prepare_for_model_step was setup correctly for an uncertainty spill
            if self.spill_type is basic_types.spill_type.forecast:
                if spill.is_uncertain: 
                    raise ValueError("prepare_for_model_step was not called for a uncertainty spill. Model is not prepared for this spill and move")
            
            
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        # Array is not the same size, change view and reshape
        self.positions = self.positions.view(dtype=basic_types.world_point)
        self.positions = np.reshape(self.positions, (len(self.positions),))
        self.delta = np.zeros((len(self.positions)), dtype=basic_types.world_point)

    def model_step_is_done(self):
        """
        This method gets called by the model when afer everything else is done
        in a time step put any code need for clean-up, etc in here in
        subclassed movers.
        """
        self.mover.model_step_is_done()
    
class WindMover(CyMover):
    """
    WindMover is a Python wrapper around the Cython wind_mover module.
    This inherits from CyMover and self.mover references CyWindMover 
    
    The real work is done by the CyWindMover object.

    CyMover sets everything up that is common to all movers.
    """
    def __init__(self, wind, is_active=True,
                 uncertain_duration=10800, uncertain_time_delay=0, 
                 uncertain_speed_scale=2., uncertain_angle_scale=0.4):
        """
        Initializes a wind mover object. It requires a Wind object who's attributes define the wind conditions
        
        :param wind: wind object
        """
        self.wind = wind
        self.mover = CyWindMover(uncertain_duration=uncertain_duration, 
                                 uncertain_time_delay=uncertain_time_delay, 
                                 uncertain_speed_scale=uncertain_speed_scale,  
                                 uncertain_angle_scale=uncertain_angle_scale)
        self.mover.set_ossm(self.wind.ossm)
        super(WindMover,self).__init__(is_active=is_active)
        
    def __repr__(self):
        """
        Return a string representation of this `WindMover`.

        TODO: We probably want to include more information.
        """
        return "WindMover( wind=<wind_object>, uncertain_duration= %s, uncertain_time_delay=%s, uncertain_speed_scale=%s, uncertain_angle_scale=%s)" \
               % (self.uncertain_duration, self.uncertain_time_delay, \
                  self.uncertain_speed_scale, self.uncertain_angle_scale)

    def __str__(self):
        """
        Return string representation of this object
        """
        info = "WindMover - current state. See 'wind' object for wind conditions: \n" + \
               "  uncertain_duration={0.uncertain_duration}\n  uncertain_time_delay={0.uncertain_time_delay}\n  uncertain_speed_scale={0.uncertain_speed_scale}\n  uncertain_angle_scale={0.uncertain_angle_scale}".format(self.mover)
                
        return info

    # Define properties using lambda functions: uses lambda function, which are accessible via fget/fset as follows:
    #     WindMover.<property_name>.fget(<instance_name>)
    #     WindMover.<property_name>.fset(<instance_name>, <value>)
    uncertain_duration = property( lambda self: self.mover.uncertain_duration,
                                   lambda self, val: setattr(self.mover,'uncertain_duration', val))
    
    uncertain_time_delay = property( lambda self: self.mover.uncertain_time_delay,
                                     lambda self, val: setattr(self.mover,'uncertain_time_delay', val))
    
    uncertain_speed_scale = property( lambda self: self.mover.uncertain_speed_scale,
                                      lambda self, val: setattr(self.mover,'uncertain_speed_scale', val))
    
    uncertain_angle_scale = property( lambda self: self.mover.uncertain_angle_scale,
                                      lambda self, val: setattr(self.mover,'uncertain_angle_scale', val))

    def _get_timeseries(self):
        """
        private method - returns the timeseries used internally by the C++ WindMover_c object.
        This should be the same as the timeseries stored in the self.wind object
        
        get function for timeseries property
        """
        dtv = self.wind.get_timeseries(data_format=basic_types.data_format.wind_uv)
        tv  = convert.to_time_value_pair(dtv, basic_types.data_format.wind_uv)
        val = self.mover.get_time_value(tv['time'])
        tv['value']['u'] = val['u']
        tv['value']['v'] = val['v']
        
        return convert.to_datetime_value_2d( tv, basic_types.data_format.wind_uv)
    
    timeseries = property(_get_timeseries)
        
    def get_move(self, spill, time_step, model_time_datetime, uncertain_spill_number=0):
        """
        :param spill: spill object
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as date time object
        :param uncertain_spill_number: starting from 0 for the 1st uncertain spill, it is the order in which the uncertain spill is added
        """
        self.prepare_data_for_get_move(spill, model_time_datetime)
        try:
            windage = spill['windages']
        except:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        self.mover.get_move(  self.model_time,
                              time_step, 
                              self.positions,
                              self.delta,
                              windage,
                              self.status_codes,
                              self.spill_type,
                              uncertain_spill_number)
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))


class RandomMover(CyMover):
    """
    Python wrapper around the Cython CyRandomMover module.
    
    The real work is done by the cython object.

    CyMover sets everything up that is common to all movers.
    """
    def __init__(self, diffusion_coef=100000, is_active=True):
        """
        Initialize
        """
        self.mover = CyRandomMover(diffusion_coef=diffusion_coef)
        super(RandomMover,self).__init__(is_active=is_active)
    
    @property
    def diffusion_coef(self):
        return self.mover.diffusion_coef
    
    @diffusion_coef.setter
    def diffusion_coef(self, value):
        self.mover.diffusion_coef = value
        
    def __repr__(self):
        """
        Return a string representation of this `WindMover`.

        TODO: We probably want to include more information.
        """
        return 'Random Mover'

    def prepare_for_model_step(self, model_time_datetime, time_step, uncertain_spills_count=0, uncertain_spills_size=None):
       """
       Random mover does not use uncertainty for anything during prepare_for_model_step(...)
       
       This method does not call super().prepare_for_model_step() .. it over-rides the base class method.
       It does however call the Mover.prepare_for_model_step() base class method
       
       """
       Mover.prepare_for_model_step(self, model_time_datetime, time_step, uncertain_spills_count, uncertain_spills_size)
       model_time = super(RandomMover,self).datetime_to_seconds(model_time_datetime)
       self.mover.prepare_for_model_step(model_time, time_step)
    
    def get_move(self, spill, time_step, model_time_datetime, uncertain_spill_number=0):
        """
        :param spill: spill object
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as date time object
        :param uncertain_spill_number: starting from 0 for the 1st uncertain spill, it is the order in which the uncertain spill is added
        """
        self.prepare_data_for_get_move(spill, model_time_datetime)
        
        self.mover.get_move(  self.model_time,
                              time_step, 
                              self.positions,
                              self.delta,
                              self.status_codes,
                              self.spill_type,
                              uncertain_spill_number)
        #return self.delta
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))
        
