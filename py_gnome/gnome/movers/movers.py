import numpy as np

from gnome.utilities import time_utils, transforms, convert
from gnome import basic_types, GnomeObject
from gnome.cy_gnome.cy_wind_mover import CyWindMover     #@UnresolvedImport IGNORE:E0611
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime       #@UnresolvedImport @UnusedImport IGNORE:W0611
from gnome.cy_gnome.cy_random_mover import CyRandomMover #@UnresolvedImport IGNORE:E0611
from gnome.cy_gnome import cy_cats_mover, cy_shio_time
from gnome.cy_gnome import cy_gridcurrent_mover
from gnome import environment
from gnome.utilities import rand    # not to confuse with python random module

import os

from datetime import datetime, timedelta
from time import gmtime

class Mover(GnomeObject):
    """
    Base class from which all Python movers can inherit
    :param active_start: datetime when the mover should be active
    :param active_stop: datetime after which the mover should be inactive
    It defines the interface for a Python mover. The model expects the methods defined here. 
    The get_move(...) method needs to be implemented by the derived class.  
    """
    def __init__(self, 
                 active_start= datetime( *gmtime(0)[:6] ), 
                 active_stop = datetime(2038,1,18,0,0,0)):   # default min + max values for timespan
        """
        During init, it defaults active = True
        """
        self._active = True  # initialize to True, though this is set in prepare_for_model_step for each step
        self.on = True          # turn the mover on / off for the run
        if active_stop <= active_start:
            raise ValueError("active_start should be a python datetime object strictly smaller than active_stop")
        
        self.active_start = active_start
        self.active_stop  = active_stop

    # Methods for active property definition
    @property
    def active(self):
        return self._active

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

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        sets active flag based on time_span. If 
            model_time + time_step > active_start and model_time + time_span < active_stop
        then set flag to true.
        
        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        
        """
        if (model_time_datetime + timedelta(seconds=time_step) > self.active_start) and \
        (model_time_datetime + timedelta(seconds=time_step) <= self.active_stop):
            self._active = True
        else:
            self._active = False
            

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size (number_elements X 3)
        and dtype = gnome.basic_types.world_point_type
         
        Not implemented in base class
        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
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
    def __init__(self, active_start= datetime( *gmtime(0)[:6] ), active_stop = datetime(2038,1,18,0,0,0)):
        super(CyMover,self).__init__(active_start, active_stop)
        
        # initialize variables for readability, tough self.mover = None produces errors, so that is not init here
        self.model_time = 0 
        self.positions = np.zeros((0,3), dtype=basic_types.world_point_type)
        self.delta = np.zeros((0,3), dtype=basic_types.world_point_type)
        self.status_codes = np.zeros((0,1), dtype=basic_types.status_code_type)
        self.spill_type = 0    # either a 1, or 2 depending on whether spill is certain or not

    def prepare_for_model_run(self):
        """
        Calls the contained cython mover's prepare_for_model_run() 
        """
        self.mover.prepare_for_model_run()

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Default implementation of prepare_for_model_step(...)
         - Sets the mover's active flag if time is within specified timespan (done in base class Mover)
         - Invokes the cython mover's prepare_for_model_step
         
        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """
        super(CyMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)
        if self.active and self.on:
            uncertain_spill_count = 0
            uncertain_spill_size = np.array( (0,) ) # only useful if spill.uncertain
            if sc.uncertain:
                uncertain_spill_count = 1
                uncertain_spill_size = np.array( (sc.num_elements,) )
            
            self.mover.prepare_for_model_step( self.datetime_to_seconds(model_time_datetime), time_step, uncertain_spill_count, uncertain_spill_size)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Base implementation of Cython wrapped C++ movers
        Override for things like the WindMover since it has a different implementation
        
        :param sc: spill_container.SpillContainer object
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """
        self.prepare_data_for_get_move(sc, model_time_datetime)
        
        # only call get_move if mover is active, it is on and there are LEs that have been released
        if self.active and self.on and len(self.positions) > 0:
            self.mover.get_move(  self.model_time,
                                  time_step, 
                                  self.positions,
                                  self.delta,
                                  self.status_codes,
                                  self.spill_type,
                                  0)    # only ever 1 spill_container so this is always 0!
            
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))
    
    def prepare_data_for_get_move(self, sc, model_time_datetime):
        """
        organizes the spill object into inputs for calling with Cython wrapper's get_move(...)

        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param model_time_datetime: current time of the model as a date time object
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)

        # Get the data:
        try:
            self.positions      = sc['positions']
            self.status_codes   = sc['status_codes']
        except KeyError, err:
            raise ValueError("The spill container does not have the required data arrays\n" + err.message)

        if sc.uncertain:
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
        if self.active and self.on:
            self.mover.model_step_is_done()

class WindMover(CyMover):
    """
    Python wrapper around the Cython wind_mover module.
    This class inherits from CyMover and contains CyWindMover 

    The real work is done by the CyWindMover object.  CyMover sets everything up that is common to all movers.
    """
    # One wind mover with a 20knot wind should (on average) produce the same results as a two wind movers
    # with a 10know wind each. 
    # This requires the windage is only set once for each timestep irrespective of how many wind movers are active during that time
    # Another way to state this, is the get_move operation is linear. This is why the following class level attributes are defined.
    _windage_is_set = False         # class scope, independent of instances of WindMover  
    _uspill_windage_is_set = False  # need to set uncertainty spill windage as well
    def __init__(self, wind, 
                 active_start= datetime( *gmtime(0)[:6] ), 
                 active_stop = datetime(2038,1,18,0,0,0),
                 uncertain_duration=10800,
                 uncertain_time_delay=0, 
                 uncertain_speed_scale=2.,
                 uncertain_angle_scale=0.4):
        """
        :param wind: wind object  -- provides the wind time series for the mover
        :param active_start: datetime object for when the mover starts being active.
        :param active_start: datetime object for when the mover stops being active.
        :param uncertain_duration:  (seconds) how often does a given uncertian windage get re-set
        :param uncertain_time_delay:   when does the uncertainly kick in.
        :param uncertain_speed_scale:  Scale for how uncertain the wind speed is
        :param uncertain_angle_scale:  Scale for how uncertain the wind direction is
        """
        self.wind = wind
        self.mover = CyWindMover(uncertain_duration=uncertain_duration, 
                                 uncertain_time_delay=uncertain_time_delay, 
                                 uncertain_speed_scale=uncertain_speed_scale,  
                                 uncertain_angle_scale=uncertain_angle_scale)
        self.mover.set_ossm(self.wind.ossm)
        super(WindMover,self).__init__(active_start, active_stop)

    def __repr__(self):
        """
        .. todo::
            We probably want to include more information.
        """
        info="WindMover( wind=<wind_object>, uncertain_duration={0.uncertain_duration}," +\
        "uncertain_time_delay={0.uncertain_time_delay}, uncertain_speed_scale={0.uncertain_speed_scale}," + \
        "uncertain_angle_scale={0.uncertain_angle_scale}, active_start={1.active_start}, active_stop={1.active_stop}, on={1.on})" \
        
        return info.format(self.mover, self)
               

    def __str__(self):
        info = "WindMover - current state. See 'wind' object for wind conditions:\n" + \
               "  uncertain_duration={0.uncertain_duration}\n" + \
               "  uncertain_time_delay={0.uncertain_time_delay}\n" + \
               "  uncertain_speed_scale={0.uncertain_speed_scale}\n" + \
               "  uncertain_angle_scale={0.uncertain_angle_scale}" + \
               "  active_start time={1.active_start}" + \
               "  active_stop time={1.active_stop}" + \
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

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Call base class method and also update windage for this timestep
         
        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """
        super(WindMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)
        
        # if no particles released, then no need for windage
        if len(sc['positions']) == 0:
            return
        
        if (not WindMover._windage_is_set and not sc.uncertain) or (not WindMover._uspill_windage_is_set and sc.uncertain):
            for spill in sc.spills:
                ix = sc['spill_num'] == sc.spills.index(spill.id)   # matching indices
                sc['windages'][ix] = rand.random_with_persistance(spill.windage_range[0],
                                                                  spill.windage_range[1],
                                                                  spill.windage_persist,
                                                                  time_step,
                                                                  array_len=len(ix) )
            if sc.uncertain:
                WindMover._uspill_windage_is_set = True
            else:
                WindMover._windage_is_set = True
        
    
    def get_move(self, sc, time_step, model_time_datetime):
        """
        Override base class functionality because mover has a different get_move signature
        
        :param sc: an instance of the gnome.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """
        self.prepare_data_for_get_move(sc, model_time_datetime)
        
        if self.active and self.on and len(self.positions) > 0: 
            self.mover.get_move(  self.model_time,
                                  time_step, 
                                  self.positions,
                                  self.delta,
                                  sc['windages'],
                                  self.status_codes,
                                  self.spill_type,
                                  0)    # only ever 1 spill_container so this is always 0!
            
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))

    def model_step_is_done(self):
        """
        Set _windage_is_set flag back to False
        """
        if WindMover._windage_is_set:
            WindMover._windage_is_set = False
        if WindMover._uspill_windage_is_set:
            WindMover._uspill_windage_is_set = False
        super(WindMover,self).model_step_is_done() 

def wind_mover_from_file(filename, **kwargs):
    """
    Creates a wind mover from a wind time-series file (OSM long wind format)

    :param filename: The full path to the data file
    :param **kwargs: All keyword arguments are passed on to the WindMover constuctor

    :returns mover: returns a wind mover, built from the file
    """
    w = environment.Wind(file=filename,
                     ts_format=basic_types.ts_format.magnitude_direction)
    ts = w.get_timeseries(ts_format=basic_types.ts_format.magnitude_direction)
    wm = WindMover(w, **kwargs)

    return wm


class RandomMover(CyMover):
    """
    This mover class inherits from CyMover and contains CyRandomMover

    The real work is done by CyRandomMover.
    CyMover sets everything up that is common to all movers.
    """
    def __init__(self, diffusion_coef=100000, 
                 active_start= datetime( *gmtime(0)[:6] ), 
                 active_stop = datetime(2038,1,18,0,0,0)):
        self.mover = CyRandomMover(diffusion_coef=diffusion_coef)
        super(RandomMover,self).__init__(active_start, active_stop)

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
        return "RandomMover(diffusion_coef=%s,active_start=%s, active_stop=%s, on=%s)" % (self.diffusion_coef,self.active_start, self.active_stop, self.on)


class CatsMover(CyMover):
    
    def __init__(self, curr_file, shio_file=None, shio_yeardata_file=None,
                 active_start= datetime( *gmtime(0)[:6] ), 
                 active_stop = datetime(2038,1,18,0,0,0)):
        """
        
        """
        if not os.path.exists(curr_file):
            raise ValueError("Path for Cats file does not exist: {0}".format(curr_file))
        
        self.curr_file = curr_file  # check if this is stored with cy_cats_mover?
        self.mover = cy_cats_mover.CyCatsMover()
        self.mover.read_topology(curr_file)
        
        if shio_file is not None:
            if not os.path.exists(shio_file):
                raise ValueError("Path for Shio file does not exist: {0}".format(shio_file))
            else:
                self.shio = cy_shio_time.CyShioTime(shio_file)   # not sure if this should be managed externally?
                self.mover.set_shio(self.shio)
                #self.shio.set_shio_yeardata_path(shio_yeardata_file)
            if shio_yeardata_file is not None:
                if not os.path.exists(shio_yeardata_file):
                    raise ValueError("Path for Shio Year Data does not exist: {0}".format(shio_yeardata_file))
                else:
                    self.shio.set_shio_yeardata_path(shio_yeardata_file)
            else:
                raise ValueError("Shio data requires path for Shio Year Data: {0}".format(shio_yeardata_file))
        
        super(CatsMover,self).__init__(active_start, active_stop)
        
    def __repr__(self):
        """
        unambiguous representation of object
        """
        info = "CatsMover(curr_file={0},shio_file={1})".format(self.curr_mover, self.shio.filename)
        return info
     
    # Properties
    scale = property( lambda self: bool(self.mover.scale_type),
                      lambda self, val: setattr(self.mover,'scale_type', int(val)) ) 
    scale_refpoint = property( lambda self: self.mover.ref_point, 
                               lambda self,val: setattr(self.mover, 'ref_point', val) )
     
    scale_value = property( lambda self: self.mover.scale_value, 
                            lambda self,val: setattr(self.mover, 'scale_value', val) )
        
        
class WeatheringMover(Mover):
    """
    Python Weathering mover

    """
    def __init__(self, wind, 
                 active_start= datetime( *gmtime(0)[:6] ), 
                 active_stop = datetime(2038,1,18,0,0,0),
                 uncertain_duration=10800, uncertain_time_delay=0,
                 uncertain_speed_scale=2., uncertain_angle_scale=0.4):
        """
        :param wind: wind object
        :param active: active flag
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

        super(WeatheringMover,self).__init__(active_start, active_stop)

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

        if spill.uncertain:
            self.spill_type = basic_types.spill_type.uncertainty
        else:
            self.spill_type = basic_types.spill_type.forecast

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Right now this method just calls its super() method.
        """
        super(WeatheringMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        :param spill: spill object
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """

        # validate our spill object
        self.validate_spill(sc)

        self.model_time = self.datetime_to_seconds(model_time_datetime)
        #self.prepare_data_for_get_move(sc, model_time_datetime)

        if self.active and self.on and len(self.positions) > 0: 
            #self.mover.get_move(  self.model_time,
            #                      time_step,
            #                      self.positions,
            #                      self.delta,
            #                      self.status_codes,
            #                      self.spill_type,
            #                      0)    # only ever 1 spill_container so this is always 0!
            pass

        #return self.delta
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))

class GridCurrentMover(CyMover):
    
    def __init__(self, curr_file, topology_file=None,
                 active_start= datetime( *gmtime(0)[:6] ), 
                 active_stop = datetime(2038,1,18,0,0,0)):
        """
        
        """
        if not os.path.exists(curr_file):
            raise ValueError("Path for current file does not exist: {0}".format(curr_file))
        
        if topology_file is not None:
            if not os.path.exists(topology_file):
                raise ValueError("Path for Topology file does not exist: {0}".format(topology_file))

        self.curr_file = curr_file  # check if this is stored with cy_gridcurrent_mover?
        self.mover = cy_gridcurrent_mover.CyGridCurrentMover()
        self.mover.text_read(curr_file,topology_file)
        
        super(GridCurrentMover,self).__init__(active_start, active_stop)
        
#     def __repr__(self):
#         """
#         not sure what to do here
#         unambiguous representation of object
#         """
#         info = "GridCurrentMover(curr_file={0},topology_file={1})".format(self.curr_mover, self.curr_mover)
#         return info
#      
#     # Properties
# Will eventually need some properties, depending on what user gets to set
#     scale_value = property( lambda self: self.mover.scale_value, 
#                             lambda self,val: setattr(self.mover, 'scale_value', val) )
#         
        

