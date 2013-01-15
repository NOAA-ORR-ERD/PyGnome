import numpy as np

from gnome.utilities import time_utils, transforms, convert
from gnome import basic_types, GnomeObject
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_random_mover import CyRandomMover

from datetime import datetime, timedelta
from time import gmtime

class Mover(GnomeObject):
    """
    Base class from which all Python movers can inherit
    :param is_active_start: datetime when the mover should be active
    :param is_active_stop: datetime after which the mover should be inactive
    It defines the interface for a Python mover. The model expects the methods defined here. 
    The get_move(...) method needs to be implemented by the derived class.  
    """
    def __init__(self, 
                 is_active_start= datetime( *gmtime(0)[:7] ), 
                 is_active_stop = datetime.max):   # default min + max values for timespan
        """
        During init, it defaults is_active = True
        """
        self._is_active = True  # initialize to True, though this is set in prepare_for_model_step for each step
        self.on = True          # turn the mover on / off for the run
        if is_active_stop <= is_active_start:
            raise ValueError("is_active_start should be a python datetime object strictly smaller than is_active_stop")
        
        self.is_active_start = is_active_start
        self.is_active_stop  = is_active_stop

    # Methods for is_active property definition
    is_active = property(lambda self: self._is_active)

    def datetime_to_seconds(self, model_time):
        """
        Put the time conversion call here - in case we decide to change it, it only updates here
        """
        return time_utils.date_to_sec(model_time)

    def prepare_for_model_run(self):
        """
        Override this method if a derived mover class needs to perform any actions prior to a model run 
        """
        pass

    def prepare_for_model_step(self, model_time, time_step,
                               uncertain_spills_count=0, uncertain_spills_size=None):
        """
        sets is_active flag based on time_span. If 
            model_time + time_step > is_active_start and model_time + time_span < is_active_stop
        then set flag to true.
        
        :param model_time: datetime object giving current model_time
        :param time_step: time step in seconds
        
        NOTE: method signature will change after one_spill is merged - update docs after merge 
        """
        if (model_time + timedelta(seconds=time_step) > self.is_active_start) and \
        (model_time + timedelta(seconds=time_step) <= self.is_active_stop):
            self._is_active = True
        else:
            self._is_active = False


    def get_move(self, spill, time_step, model_time, uncertain_spill_number=0):
        """
        Not implemented in base class.  Each class derived from Mover object must implement it's own get_move

        .. todo::
            maybe we should elaborate on exactly what this function does.
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
    Base class for python wrappers around cython movers.

    All cython movers (CyWindMover, CyRandomMover) are instantiated by a derived class,
    and then contained by this class in the member 'movers'.  They will need to extract
    info from spill object.

    We assumes any derived class will instantiate a 'mover' object that
    has methods like: prepare_for_model_run, prepare_for_model_step,
    """
    def __init__(self, is_active_start= datetime( *gmtime(0)[:7] ), is_active_stop = datetime.max):
        super(CyMover,self).__init__(is_active_start, is_active_stop)

    def prepare_for_model_run(self):
        """
        Calls the contained cython mover's prepare_for_model_run() 
        """
        self.mover.prepare_for_model_run()

    def prepare_for_model_step(self, model_time, time_step,
                               uncertain_spills_count=0, uncertain_spills_size=None):
        """
        Default implementation of prepare_for_model_step(...)
         - Sets the mover's is_active flag if time is within specified timespan
         - Invokes the cython mover's prepare_for_model_step
         
        :param model_time: datetime object with current model time
        :param time_step: time step in seconds
        
        NOTE: Remaining inputs will change after one_spill merge .. TODO: Fix this
        """
        super(CyMover,self).prepare_for_model_step(model_time, time_step, uncertain_spills_count, uncertain_spills_size)
        if self.is_active and self.on:
            self.mover.prepare_for_model_step( self.datetime_to_seconds(model_time), time_step, uncertain_spills_count, uncertain_spills_size)

    def prepare_data_for_get_move(self, spill, model_time_datetime):
        """
        organizes the spill object into inputs for calling with Cython wrapper's get_move(...)

        :param spill: an instance of the gnome.spill.Spill class
        :param model_time_datetime: current time of the model as a date time object
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)

        # Get the data:
        try:
            self.positions      = spill['positions']
            self.status_codes   = spill['status_codes']
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n" + err.message)

        if spill.is_uncertain:
            self.spill_type = basic_types.spill_type.uncertainty
        else:
            self.spill_type = basic_types.spill_type.forecast

        # Array is not the same size, change view and reshape
        self.positions = self.positions.view(dtype=basic_types.world_point).reshape( (len(self.positions),) )
        self.delta = np.zeros((len(self.positions)), dtype=basic_types.world_point)

    def model_step_is_done(self):
        """
        This method gets called by the model after everything else is done
        in a time step, and is intended to perform any necessary clean-up operations.
        Subclassed movers can override this method, but should probably call the super()
        method, as the contained mover most likely needs cleanup.
        """
        if self.is_active and self.on:
            self.mover.model_step_is_done()

class WindMover(CyMover):
    """
    Python wrapper around the Cython wind_mover module.
    This class inherits from CyMover and contains CyWindMover 

    The real work is done by the CyWindMover object.  CyMover sets everything up that is common to all movers.
    """
    def __init__(self, wind, 
                 is_active_start= datetime( *gmtime(0)[:7] ), 
                 is_active_stop = datetime.max,
                 uncertain_duration=10800, uncertain_time_delay=0, 
                 uncertain_speed_scale=2., uncertain_angle_scale=0.4):
        """
        :param wind: wind object
        :param is_active: active flag
        :param uncertain_duration:     Used by the cython wind mover.
        :param uncertain_time_delay:   Used by the cython wind mover.
        :param uncertain_speed_scale:  Used by the cython wind mover.
        :param uncertain_angle_scale:  Used by the cython wind mover.
        """
        self.wind = wind
        self.mover = CyWindMover(uncertain_duration=uncertain_duration, 
                                 uncertain_time_delay=uncertain_time_delay, 
                                 uncertain_speed_scale=uncertain_speed_scale,  
                                 uncertain_angle_scale=uncertain_angle_scale)
        self.mover.set_ossm(self.wind.ossm)
        super(WindMover,self).__init__(is_active_start, is_active_stop)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        info="WindMover( wind=<wind_object>, uncertain_duration={0.uncertain_duration}," +\
        "uncertain_time_delay={0.uncertain_time_delay}, uncertain_speed_scale={0.uncertain_speed_scale}," + \
        "uncertain_angle_scale={0.uncertain_angle_scale}, is_active_start={1.is_active_start}, is_active_stop={1.is_active_stop}, on={1.on})" \
        
        return info.format(self.mover, self)
               

    def __str__(self):
        info = "WindMover - current state. See 'wind' object for wind conditions:\n" + \
               "  uncertain_duration={0.uncertain_duration}\n" + \
               "  uncertain_time_delay={0.uncertain_time_delay}\n" + \
               "  uncertain_speed_scale={0.uncertain_speed_scale}\n" + \
               "  uncertain_angle_scale={0.uncertain_angle_scale}" + \
               "  is_active_start time={1.is_active_start}" + \
               "  is_active_stop time={1.is_active_stop}" + \
               "  current on/off status={1.on}" 
        return info.format(self.mover, self)

    # Define properties using lambda functions: uses lambda function, which are accessible via fget/fset as follows:
    uncertain_duration = property( lambda self: self.mover.uncertain_duration,
                                   lambda self, val: setattr(self.mover,'uncertain_duration', val))

    uncertain_time_delay = property( lambda self: self.mover.uncertain_time_delay,
                                     lambda self, val: setattr(self.mover,'uncertain_time_delay', val))

    uncertain_speed_scale = property( lambda self: self.mover.uncertain_speed_scale,
                                      lambda self, val: setattr(self.mover,'uncertain_speed_scale', val))

    uncertain_angle_scale = property( lambda self: self.mover.uncertain_angle_scale,
                                      lambda self, val: setattr(self.mover,'uncertain_angle_scale', val))


    def get_move(self, spill, time_step, model_time_datetime, uncertain_spill_number=0):
        """
        :param spill: an instance of the gnome.spill.Spill class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        :param uncertain_spill_number: starting from 0 for the 1st uncertain spill, it is the order in which the uncertain spill is added
        """
        self.prepare_data_for_get_move(spill, model_time_datetime)
        
        if self.is_active and self.on: 
            try:
                windage = spill['windages']
            except KeyError, e:
                raise KeyError("The spill does not have the required data arrays\n" + e.message)
    
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
    This mover class inherits from CyMover and contains CyRandomMover

    The real work is done by CyRandomMover.
    CyMover sets everything up that is common to all movers.
    """
    def __init__(self, diffusion_coef=100000, 
                 is_active_start= datetime( *gmtime(0)[:7] ), 
                 is_active_stop = datetime.max):
        self.mover = CyRandomMover(diffusion_coef=diffusion_coef)
        super(RandomMover,self).__init__(is_active_start, is_active_stop)

    @property
    def diffusion_coef(self):
        return self.mover.diffusion_coef
    @diffusion_coef.setter
    def diffusion_coef(self, value):
        self.mover.diffusion_coef = value

    def __repr__(self):
        """
        .. todo:: 
            We probably want to include more information.
        """
        return "RandomMover(diffusion_coef=%s,is_active_start=%s, is_active_stop=%s, on=%s)" % (self.diffusion_coef,self.is_active_start, self.is_active_stop, self.on)

    def get_move(self, spill, time_step, model_time_datetime, uncertain_spill_number=0):
        """
        :param spill: spill object
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        :param uncertain_spill_number: starting from 0 for the 1st uncertain spill, it is the order in which the uncertain spill is added
        """
        self.prepare_data_for_get_move(spill, model_time_datetime)

        if self.is_active and self.on: 
            self.mover.get_move(  self.model_time,
                                  time_step, 
                                  self.positions,
                                  self.delta,
                                  self.status_codes,
                                  self.spill_type,
                                  uncertain_spill_number)
            
        #return self.delta
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))


class WeatheringMover(Mover):
    """
    Python Weathering mover

    """
    def __init__(self, wind, 
                 is_active_start= datetime( *gmtime(0)[:7] ), 
                 is_active_stop = datetime.max,
                 uncertain_duration=10800, uncertain_time_delay=0,
                 uncertain_speed_scale=2., uncertain_angle_scale=0.4):
        """
        :param wind: wind object
        :param is_active: active flag
        :param uncertain_duration:     Used by the cython wind mover.  We may still need these.
        :param uncertain_time_delay:   Used by the cython wind mover.  We may still need these.
        :param uncertain_speed_scale:  Used by the cython wind mover.  We may still need these.
        :param uncertain_angle_scale:  Used by the cython wind mover.  We may still need these.
        """
        self.wind = wind
        self.uncertain_duration=uncertain_duration
        self.uncertain_time_delay=uncertain_time_delay
        self.uncertain_speed_scale=uncertain_speed_scale
        self.uncertain_angle_scale=uncertain_angle_scale

        super(WeatheringMover,self).__init__(is_active_start, is_active_stop)

    def __repr__(self):
        return "WeatheringMover( wind=<wind_object>, uncertain_duration= %s, uncertain_time_delay=%s, uncertain_speed_scale=%s, uncertain_angle_scale=%s)" \
               % (self.uncertain_duration, self.uncertain_time_delay, \
                  self.uncertain_speed_scale, self.uncertain_angle_scale)

    def validate_spill(self, spill):
        try:
            self.positions = spill['positions']
            # reshape to our needs
            self.positions = self.positions.view(dtype=basic_types.world_point).reshape( (len(self.positions),) )
            self.status_codes   = spill['status_codes']
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n" + err.message)
        # create an array of position deltas
        self.delta = np.zeros((len(self.positions)), dtype=basic_types.world_point)

        if spill.is_uncertain:
            self.spill_type = basic_types.spill_type.uncertainty
        else:
            self.spill_type = basic_types.spill_type.forecast

    def prepare_for_model_step(self, model_time_datetime, time_step, uncertain_spills_count=0, uncertain_spills_size=None):
       """
       Right now this method just calls its super() method.
       """
       super(WeatheringMover,self).prepare_for_model_step(model_time_datetime, time_step,
                                                          uncertain_spills_count, uncertain_spills_size)

    def get_move(self, spill, time_step, model_time_datetime, uncertain_spill_number=0):
        """
        :param spill: spill object
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        :param uncertain_spill_number: starting from 0 for the 1st uncertain spill, it is the order in which the uncertain spill is added
        """

        # validate our spill object
        self.validate_spill(spill)

        self.model_time = self.datetime_to_seconds(model_time_datetime)
        self.prepare_data_for_get_move(spill, model_time_datetime)

        if self.is_active and self.on: 
            self.mover.get_move(  self.model_time,
                                  time_step,
                                  self.positions,
                                  self.delta,
                                  self.status_codes,
                                  self.spill_type,
                                  uncertain_spill_number)
        #return self.delta
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))

