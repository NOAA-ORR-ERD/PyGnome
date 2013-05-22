import os
import copy
from datetime import datetime

import numpy as np

from gnome.utilities import time_utils, transforms, convert, serializable
from gnome import basic_types, GnomeId
from gnome.cy_gnome.cy_wind_mover import CyWindMover     #@UnresolvedImport IGNORE:E0611
#===============================================================================
# from gnome.cy_gnome.cy_random_mover import CyRandomMover #@UnresolvedImport IGNORE:E0611
# from gnome.cy_gnome.cy_random_vertical_mover import CyRandomVerticalMover #@UnresolvedImport IGNORE:E0611
# from gnome.cy_gnome import cy_cats_mover, cy_shio_time, cy_ossm_time
# from gnome.cy_gnome import cy_gridcurrent_mover
# from gnome.cy_gnome import cy_gridwind_mover
#===============================================================================
from gnome import environment
from gnome.utilities import rand, inf_datetime    # not to confuse with python random module

class Mover(object):
    """
    Base class from which all Python movers can inherit
    
    It defines the interface for a Python mover. The model expects the methods defined here. 
    The get_move(...) method needs to be implemented by the derived class.  
    """
    state = copy.deepcopy(serializable.Serializable.state)
    state.add(update=['on','active_start','active_stop'],
              create=['on','active_start','active_stop'],
              read=['active'] )
    
    def __init__(self, **kwargs):   # default min + max values for timespan
        """
        Initialize default Mover parameters
        
        All parameters are optional (kwargs)
        
        :param on: boolean as to whether the object is on or not. Default is on
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param id: Unique Id identifying the newly created mover (a UUID as a string). 
                   This is used when loading an object from a persisted model
        """
        self._active = True  # initialize to True, though this is set in prepare_for_model_step for each step
        self.on = kwargs.pop('on',True)  # turn the mover on / off for the run
        active_start = kwargs.pop('active_start', inf_datetime.InfDateTime('-inf'))
        active_stop  = kwargs.pop('active_stop', inf_datetime.InfDateTime('inf'))
        
        if active_stop <= active_start:
            raise ValueError("active_start should be a python datetime object strictly smaller than active_stop")
        
        self.active_start = active_start
        self.active_stop = active_stop
        self._gnome_id = GnomeId(id=kwargs.pop('id',None))

    # Methods for active property definition
    @property
    def active(self):
        return self._active

    id = property( lambda self: self._gnome_id.id )

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
        sets active flag based on time_span and on flag. If 
            model_time > active_start and model_time < active_stop then set flag to true.
        
        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        
        """
        #=======================================================================
        # if (model_time_datetime >= self.active_start) and \
        # (model_time_datetime < self.active_stop) and self.on:
        #=======================================================================
        if (self.active_start <= model_time_datetime ) and \
        (self.active_stop > model_time_datetime) and self.on:
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
    def __init__(self, **kwargs):
        """
        Base class for python wrappers around cython movers. 
        Uses super(CyMover,self).__init__(\*\*kwargs) to call Mover class __init__ method
    
        All cython movers (CyWindMover, CyRandomMover) are instantiated by a derived class,
        and then contained by this class in the member 'movers'.  They will need to extract
        info from spill object.
    
        We assumes any derived class will instantiate a 'mover' object that
        has methods like: prepare_for_model_run, prepare_for_model_step,
        
        All kwargs passed on to super class
        """
        super(CyMover,self).__init__(**kwargs)
        
        # initialize variables here for readability, though self.mover = None produces errors, so that is not initialized here
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
        
        Uses super to invoke Mover class prepare_for_model_step and does a couple more things specific to CyMover.
        """
        super(CyMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)
        if self.active:
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
        if self.active and len(self.positions) > 0:
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
        Subclassed movers can override this method.
        """
        if self.active:
            self.mover.model_step_is_done()

class WindMover(CyMover, serializable.Serializable):
    """
    Python wrapper around the Cython wind_mover module.
    This class inherits from CyMover and contains CyWindMover 

    The real work is done by the CyWindMover object.  CyMover sets everything up that is common to all movers.
    """
    # One wind mover with a 20knot wind should (on average) produce the same results as a two wind movers
    # with a 10know wind each. 
    # This requires the windage is only set once for each timestep irrespective of how many wind movers are active during that time
    # Another way to state this, is the get_move operation is linear. This is why the following class level attributes are defined.
    #_windage_is_set = False         # class scope, independent of instances of WindMover  
    #_uspill_windage_is_set = False  # need to set uncertainty spill windage as well
    
    _update = ['uncertain_duration','uncertain_time_delay','uncertain_speed_scale','uncertain_angle_scale']
    _create = ['wind_id']
    _read = ['wind_id']
    _create.extend(_update)
    
    state = copy.deepcopy(CyMover.state)
    state.add(read=_read, update=_update, create=_create)
    
    @classmethod
    def new_from_dict(cls, dict_):
        """
        define in WindMover and check wind_id matches wind
        
        invokes: super(WindMover,cls).new_from_dict(dict\_)
        """
        wind_id = dict_.pop('wind_id')
        if dict_.get('wind').id != wind_id:
            raise ValueError("id of wind object does not match the wind_id parameter")
        
        return super(WindMover,cls).new_from_dict(dict_)
    
    def wind_id_to_dict(self):
        """
        used only for storing state so no wind_id_from_dict is defined. This
        is not a read/write attribute. Only defined for serializable_state
        """
        return self.wind.id
    
    def from_dict(self, dict_):
        """
        For updating the object from dictionary
        
        'wind' object is not part of the state since it is not serialized/deserialized;
        however, user can still update the wind attribute with a new Wind object. That must
        be poped out of the dict() here, then call super to process the standard dict\_
        """
        self.wind = dict_.pop('wind',self.wind)
            
        super(WindMover, self).from_dict(dict_)
    
    def __init__(self, wind, **kwargs):
        """
        Uses super to call CyMover base class __init__
        
        :param wind: wind object  -- provides the wind time series for the mover
        
        Optional parameters (kwargs):
        :param uncertain_duration:  (seconds) how often does a given uncertian windage get re-set
        :param uncertain_time_delay:   when does the uncertainly kick in.
        :param uncertain_speed_scale:  Scale for how uncertain the wind speed is
        :param uncertain_angle_scale:  Scale for how uncertain the wind direction is
        
        Remaining kwargs are passed onto Mover's __init__ using super. 
        See Mover documentation for remaining valid kwargs.
        """
        self.mover = CyWindMover(uncertain_duration=kwargs.pop('uncertain_duration',10800), 
                                 uncertain_time_delay=kwargs.pop('uncertain_time_delay',0), 
                                 uncertain_speed_scale=kwargs.pop('uncertain_speed_scale',2.),  
                                 uncertain_angle_scale=kwargs.pop('uncertain_angle_scale',0.4) )
        #self.mover.set_ossm(self.wind.ossm)
        self.wind = wind    
        super(WindMover,self).__init__(**kwargs)

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
    
    @property
    def wind(self):
        return self._wind
    
    @wind.setter
    def wind(self, value):
        if not isinstance(value, environment.Wind):
            raise TypeError("wind must be of type environment.Wind")
        else:
            self._wind = value
            self.mover.set_ossm(self.wind.ossm) # update reference to underlying cython object
    
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
        Call base class method using super
        Also updates windage for this timestep
         
        :param sc: an instance of the gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current time of the model as a date time object
        """
        super(WindMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)
        
        # if no particles released, then no need for windage
        if len(sc['positions']) == 0:
            return
        
        #if (not WindMover._windage_is_set and not sc.uncertain) or (not WindMover._uspill_windage_is_set and sc.uncertain):
        for spill in sc.spills:
            spill_mask = sc.get_spill_mask(spill)
            if np.any(spill_mask):
                sc['windages'][spill_mask] = rand.random_with_persistance(spill.windage_range[0],
                                                                  spill.windage_range[1],
                                                                  spill.windage_persist,
                                                                  time_step,
                                                                  array_len=np.count_nonzero(spill_mask))
            #===================================================================
            # if sc.uncertain:
            #    WindMover._uspill_windage_is_set = True
            # else:
            #    WindMover._windage_is_set = True
            #===================================================================
        
    
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
                                  sc['windages'],
                                  self.status_codes,
                                  self.spill_type,
                                  0)    # only ever 1 spill_container so this is always 0!
            
        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))

    #===========================================================================
    # def model_step_is_done(self):
    #    """
    #    Set _windage_is_set flag back to False
    #    use super to invoke base class model_step_is_done
    #    """
    #    if WindMover._windage_is_set:
    #        WindMover._windage_is_set = False
    #    if WindMover._uspill_windage_is_set:
    #        WindMover._uspill_windage_is_set = False
    #    super(WindMover,self).model_step_is_done() 
    #===========================================================================

def wind_mover_from_file(filename, **kwargs):
    """
    Creates a wind mover from a wind time-series file (OSM long wind format)

    :param filename: The full path to the data file
    :param **kwargs: All keyword arguments are passed on to the WindMover constuctor

    :returns mover: returns a wind mover, built from the file
    """
    w = environment.Wind(filename=filename,
                     ts_format=basic_types.ts_format.magnitude_direction)
    ts = w.get_timeseries(format=basic_types.ts_format.magnitude_direction)
    wm = WindMover(w, **kwargs)

    return wm

def constant_wind_mover(speed, dir, units='m/s'):
    """
    utility function to create a mover with a constant wind

    :param speed: wind speed
    :param dir:   wind direction in degrees true
                  (direction from, following the meteorological convention)
    :param units='m/s': the units that the input wind speed is in.
                        options: 'm/s', 'knot', 'mph', others...


    :returns WindMover: returns a gnome.movers.WindMover object all set up.
    """
    series = np.zeros((1,), dtype=basic_types.datetime_value_2d)
    # note: if there is ony one entry, the time is arbitrary
    series[0] = (datetime.now(), ( speed, dir) )
    wind = environment.Wind(timeseries=series, units=units)
    w_mover = WindMover(wind)
    return w_mover


#===============================================================================
# class RandomMover(CyMover, serializable.Serializable):
#    """
#    This mover class inherits from CyMover and contains CyRandomMover
# 
#    The real work is done by CyRandomMover.
#    CyMover sets everything up that is common to all movers.
#    """
#    state = copy.deepcopy(CyMover.state)
#    state.add(update=['diffusion_coef'], create=['diffusion_coef'])
#    
#    def __init__(self, **kwargs):
#        """
#        Uses super to invoke base class __init__ method. 
#        
#        Optional parameters (kwargs)
#        :param diffusion_coef: Diffusion coefficient for random diffusion. Default is 100,000 cm2/sec
#        
#        Remaining kwargs are passed onto Mover's __init__ using super. 
#        See Mover documentation for remaining valid kwargs.
#        """
#        self.mover = CyRandomMover(diffusion_coef=kwargs.pop('diffusion_coef',100000))
#        super(RandomMover,self).__init__(**kwargs)
# 
#    @property
#    def diffusion_coef(self):
#        return self.mover.diffusion_coef
#    @diffusion_coef.setter
#    def diffusion_coef(self, value):
#        self.mover.diffusion_coef = value
# 
#    def __repr__(self):
#        """
#        .. todo:: 
#            We probably want to include more information.
#        """
#        return "RandomMover(diffusion_coef=%s,active_start=%s, active_stop=%s, on=%s)" % (self.diffusion_coef,self.active_start, self.active_stop, self.on)
# 
# 
# class RandomVerticalMover(CyMover, serializable.Serializable):
#    """
#    This mover class inherits from CyMover and contains CyRandomVerticalMover
# 
#    The real work is done by CyRandomVerticalMover.
#    CyMover sets everything up that is common to all movers.
#    """
#    state = copy.deepcopy(CyMover.state)
#    state.add(update=['vertical_diffusion_coef'], create=['vertical_diffusion_coef'])
#    
#    def __init__(self, **kwargs):
#        """
#        Uses super to invoke base class __init__ method. 
#        
#        Optional parameters (kwargs)
#        :param vertical_diffusion_coef: Vertical diffusion coefficient for random diffusion. Default is 5 cm2/s
#        
#        Remaining kwargs are passed onto Mover's __init__ using super. 
#        See Mover documentation for remaining valid kwargs.
#        """
#        self.mover = CyRandomVerticalMover(diffusion_coef=kwargs.pop('vertical_diffusion_coef',5))
#        super(RandomVerticalMover,self).__init__(**kwargs)
# 
#    @property
#    def vertical_diffusion_coef(self):
#        return self.mover.vertical_diffusion_coef
#    @vertical_diffusion_coef.setter
#    def vertical_diffusion_coef(self, value):
#        self.mover.vertical_diffusion_coef = value
# 
#    def __repr__(self):
#        """
#        .. todo:: 
#            We probably want to include more information.
#        """
#        return "RandomVerticalMover(vertical_diffusion_coef=%s,active_start=%s, active_stop=%s, on=%s)" % (self.vertical_diffusion_coef,self.active_start, self.active_stop, self.on)
# 
# 
# class CatsMover(CyMover, serializable.Serializable):
#    
#    state = copy.deepcopy(CyMover.state)
#    
#    _update = ['scale','scale_refpoint','scale_value']
#    _create = ['tide_id']
#    _create.extend(_update)
#    state.add(update=_update, create=_create, read=['tide_id'])
#    state.add_field(serializable.Field('filename',create=True,read=True,isdatafile=True))
#    
#    @classmethod
#    def new_from_dict(cls, dict_):
#        """
#        define in WindMover and check wind_id matches wind
#        
#        invokes: super(WindMover,cls).new_from_dict(dict_)
#        """
#        if 'tide' in dict_ and 'tide' is not None:
#            if dict_.get('tide').id != dict_.pop('tide_id'):
#                raise ValueError("id of tide object does not match the tide_id parameter")
#                
#        return super(CatsMover,cls).new_from_dict(dict_)
#    
#    
#    def __init__(self, 
#                 filename, 
#                 **kwargs):
#        """
#        Uses super to invoke base class __init__ method. 
#        
#        :param filename: file containing currents patterns for Cats 
#        
#        Optional parameters (kwargs)
#        :param tide: a gnome.environment.Tide object to be attached to CatsMover
#        :param scale: a boolean to indicate whether to scale value at reference point or not
#        :param scale_value: value used for scaling at reference point
#        :param scale_refpoint: reference location (long, lat, z). The scaling applied to all data is determined by scaling the 
#                               raw value at this location.
#        
#        Remaining kwargs are passed onto Mover's __init__ using super. 
#        See Mover documentation for remaining valid kwargs.
#        """
#        if not os.path.exists(filename):
#            raise ValueError("Path for Cats filename does not exist: {0}".format(filename))
#        
#        self.filename = filename  # check if this is stored with cy_cats_mover?
#        self.mover = cy_cats_mover.CyCatsMover()
#        self.mover.text_read(filename)
#        
#        self._tide = None   
#        if kwargs.get('tide') is not None:
#            self.tide = kwargs.pop('tide')
#        
#        self.scale = kwargs.pop('scale', self.mover.scale_type)
#        self.scale_value = kwargs.get('scale_value', self.mover.scale_value)
#        self.scale_refpoint = kwargs.pop('scale_refpoint', self.mover.ref_point)
#        
#        super(CatsMover,self).__init__(**kwargs)
#        
#    def __repr__(self):
#        """
#        unambiguous representation of object
#        """
#        info = "CatsMover(filename={0})".format(self.filename)
#        return info
#     
#    # Properties
#    scale = property( lambda self: bool(self.mover.scale_type),
#                      lambda self, val: setattr(self.mover,'scale_type', int(val)) ) 
#    scale_refpoint = property( lambda self: self.mover.ref_point, 
#                               lambda self,val: setattr(self.mover, 'ref_point', val) )
#     
#    scale_value = property( lambda self: self.mover.scale_value, 
#                            lambda self,val: setattr(self.mover, 'scale_value', val) )
#    
#    def tide_id_to_dict(self):
#        if self.tide is None:
#            return None
#        else:
#            return self.tide.id    
#    
#    @property
#    def tide(self):
#        return self._tide
#    
#    @tide.setter
#    def tide(self, tide_obj):
#        if not isinstance(tide_obj, environment.Tide):
#            raise TypeError("tide must be of type environment.Tide")
#        
#        if isinstance(tide_obj.cy_obj, cy_shio_time.CyShioTime):
#            self.mover.set_shio(tide_obj.cy_obj)
#        elif isinstance(tide_obj.cy_obj, cy_ossm_time.CyOSSMTime):
#            self.mover.set_ossm(tide_obj.cy_obj)
#        else:
#            raise TypeError("Tide.cy_obj attribute must be either CyOSSMTime or CyShioTime type for CatsMover.")
#        
#        self._tide = tide_obj
# 
# 
#    def from_dict(self, dict_):
#        """
#        For updating the object from dictionary
#        
#        'tide' object is not part of the state since it is not serialized/deserialized;
#        however, user can still update the tide attribute with a new Tide object. That must
#        be poped out of the dict here, then call super to process the standard dict_
#        """
#        if 'tide' in dict_ and dict_.get('tide') is not None:
#            self.tide = dict_.pop('tide')
#            
#        super(CatsMover, self).from_dict(dict_)
#        
# class WeatheringMover(Mover):
#    """
#    Python Weathering mover
# 
#    """
#    def __init__(self, wind, 
#                 uncertain_duration=10800, uncertain_time_delay=0,
#                 uncertain_speed_scale=2., uncertain_angle_scale=0.4, **kwargs):
#        """
#        :param wind: wind object
#        :param active: active flag
#        :param uncertain_duration:     Used by the cython wind mover.  We may still need these.
#        :param uncertain_time_delay:   Used by the cython wind mover.  We may still need these.
#        :param uncertain_speed_scale:  Used by the cython wind mover.  We may still need these.
#        :param uncertain_angle_scale:  Used by the cython wind mover.  We may still need these.
#        """
#        self.wind = wind
#        self.uncertain_duration=uncertain_duration
#        self.uncertain_time_delay=uncertain_time_delay
#        self.uncertain_speed_scale=uncertain_speed_scale
#        self.uncertain_angle_scale=uncertain_angle_scale
# 
#        super(WeatheringMover,self).__init__(**kwargs)
# 
#    def __repr__(self):
#        return "WeatheringMover( wind=<wind_object>, uncertain_duration= %s, uncertain_time_delay=%s, uncertain_speed_scale=%s, uncertain_angle_scale=%s)" \
#               % (self.uncertain_duration, self.uncertain_time_delay, \
#                  self.uncertain_speed_scale, self.uncertain_angle_scale)
# 
#    def validate_spill(self, spill):
#        try:
#            self.positions = spill['positions']
#            # reshape to our needs
#            self.positions = self.positions.view(dtype=basic_types.world_point).reshape( (len(self.positions),) )
#            self.status_codes   = spill['status_codes']
#        except KeyError, err:
#            raise ValueError("The spill does not have the required data arrays\n" + err.message)
#        # create an array of position deltas
#        self.delta = np.zeros((len(self.positions)), dtype=basic_types.world_point)
# 
#        if spill.uncertain:
#            self.spill_type = basic_types.spill_type.uncertainty
#        else:
#            self.spill_type = basic_types.spill_type.forecast
# 
#    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
#        """
#        Right now this method just calls its super() method.
#        """
#        super(WeatheringMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)
# 
#    def get_move(self, sc, time_step, model_time_datetime):
#        """
#        :param spill: spill object
#        :param time_step: time step in seconds
#        :param model_time_datetime: current time of the model as a date time object
#        """
# 
#        # validate our spill object
#        self.validate_spill(sc)
# 
#        self.model_time = self.datetime_to_seconds(model_time_datetime)
#        #self.prepare_data_for_get_move(sc, model_time_datetime)
# 
#        if self.active and len(self.positions) > 0: 
#            #self.mover.get_move(  self.model_time,
#            #                      time_step,
#            #                      self.positions,
#            #                      self.delta,
#            #                      self.status_codes,
#            #                      self.spill_type,
#            #                      0)    # only ever 1 spill_container so this is always 0!
#            pass
# 
#        #return self.delta
#        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))
# 
# class GridCurrentMover(CyMover, serializable.Serializable):
#    
#    state = copy.deepcopy(CyMover.state)
#    
#    state.add_field( [serializable.Field('filename',create=True,read=True,isdatafile=True),
#                      serializable.Field('topology_file',create=True,read=True,isdatafile=True)])
#    
#    def __init__(self, filename, topology_file=None, **kwargs):
#        """
#        will need to add uncertainty parameters and other dialog fields
#        use super with kwargs to invoke base class __init__
#        """
#        if not os.path.exists(filename):
#            raise ValueError("Path for current file does not exist: {0}".format(filename))
#        
#        if topology_file is not None:
#            if not os.path.exists(topology_file):
#                raise ValueError("Path for Topology file does not exist: {0}".format(topology_file))
# 
#        self.filename = filename  # check if this is stored with cy_gridcurrent_mover?
#        self.topology_file = topology_file  # check if this is stored with cy_gridcurrent_mover?
#        self.mover = cy_gridcurrent_mover.CyGridCurrentMover()
#        self.mover.text_read(filename,topology_file)
#        
#        super(GridCurrentMover,self).__init__(**kwargs)
#        
# #     def __repr__(self):
# #         """
# #         not sure what to do here
# #         unambiguous representation of object
# #         """
# #         info = "GridCurrentMover(filename={0},topology_file={1})".format(self.curr_mover, self.curr_mover)
# #         return info
# #      
# #     # Properties
# # Will eventually need some properties, depending on what user gets to set
# #     scale_value = property( lambda self: self.mover.scale_value, 
# #                             lambda self,val: setattr(self.mover, 'scale_value', val) )
# #         
#        
# 
# class GridWindMover(CyMover, serializable.Serializable):
#    
#    _update = ['uncertain_duration','uncertain_time_delay','uncertain_speed_scale','uncertain_angle_scale']
#    #_create = ['wind_file', 'topology_file']
#    #_read = ['wind_file', 'topology_file']
#    #_create.extend(_update)
#    
#    state = copy.deepcopy(CyMover.state)
#    state.add(update=_update, create=_update)
#    state.add_field( [serializable.Field('wind_file',create=True,read=True,isdatafile=True),
#                      serializable.Field('topology_file',create=True,read=True,isdatafile=True)])
#        
#    def __init__(self, wind_file, topology_file=None, 
#                 uncertain_duration=10800,
#                 uncertain_time_delay=0, 
#                 uncertain_speed_scale=2.,
#                 uncertain_angle_scale=0.4, **kwargs):
#        """
#        :param active_start: datetime object for when the mover starts being active.
#        :param active_start: datetime object for when the mover stops being active.
#        :param uncertain_duration:  (seconds) how often does a given uncertian windage get re-set
#        :param uncertain_time_delay:   when does the uncertainly kick in.
#        :param uncertain_speed_scale:  Scale for how uncertain the wind speed is
#        :param uncertain_angle_scale:  Scale for how uncertain the wind direction is
#        
#        uses super: super(GridWindMover,self).__init__(**kwargs)
#        """
# 
#        """
#        .. todo::
#            We will need the uncertainty parameters.
#            The c++ mover derives from windmover. Here the windmover code is repeated
#        """
#        if not os.path.exists(wind_file):
#            raise ValueError("Path for wind file does not exist: {0}".format(wind_file))
#        
#        if topology_file is not None:
#            if not os.path.exists(topology_file):
#                raise ValueError("Path for Topology file does not exist: {0}".format(topology_file))
# 
#        self.wind_file = wind_file  # check if this is stored with cy_gridwind_mover?
#        self.topology_file = topology_file  # check if this is stored with cy_gridwind_mover?
#        self.mover = cy_gridwind_mover.CyGridWindMover()
#        self.mover.text_read(wind_file,topology_file)
#        
# #         self.mover = CyGridWindMover(uncertain_duration=uncertain_duration, 
# #                                  uncertain_time_delay=uncertain_time_delay, 
# #                                  uncertain_speed_scale=uncertain_speed_scale,  
# #                                  uncertain_angle_scale=uncertain_angle_scale)
#        super(GridWindMover,self).__init__(**kwargs)
# 
#    def __repr__(self):
#        """
#        .. todo::
#            We probably want to include more information.
#        """
#        info="GridWindMover( uncertain_duration={0.uncertain_duration}," +\
#        "uncertain_time_delay={0.uncertain_time_delay}, uncertain_speed_scale={0.uncertain_speed_scale}," + \
#        "uncertain_angle_scale={0.uncertain_angle_scale}, active_start={1.active_start}, active_stop={1.active_stop}, on={1.on})" \
#        
#        return info.format(self.mover, self)
#               
# 
#    def __str__(self):
#        info = "GridWindMover - current state.\n" + \
#               "  uncertain_duration={0.uncertain_duration}\n" + \
#               "  uncertain_time_delay={0.uncertain_time_delay}\n" + \
#               "  uncertain_speed_scale={0.uncertain_speed_scale}\n" + \
#               "  uncertain_angle_scale={0.uncertain_angle_scale}" + \
#               "  active_start time={1.active_start}" + \
#               "  active_stop time={1.active_stop}" + \
#               "  current on/off status={1.on}" 
#        return info.format(self.mover, self)
# 
#    # Define properties using lambda functions: uses lambda function, which are accessible via fget/fset as follows:
#    uncertain_duration = property( lambda self: self.mover.uncertain_duration,
#                                   lambda self, val: setattr(self.mover,'uncertain_duration', val))
# 
#    uncertain_time_delay = property( lambda self: self.mover.uncertain_time_delay,
#                                     lambda self, val: setattr(self.mover,'uncertain_time_delay', val))
# 
#    uncertain_speed_scale = property( lambda self: self.mover.uncertain_speed_scale,
#                                      lambda self, val: setattr(self.mover,'uncertain_speed_scale', val))
# 
#    uncertain_angle_scale = property( lambda self: self.mover.uncertain_angle_scale,
#                                      lambda self, val: setattr(self.mover,'uncertain_angle_scale', val))
# 
#    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
#        """
#        Call base class method using super and also update windage for this timestep
#         
#        :param sc: an instance of the gnome.spill_container.SpillContainer class
#        :param time_step: time step in seconds
#        :param model_time_datetime: current time of the model as a date time object
#        """
#        super(GridWindMover,self).prepare_for_model_step(sc, time_step, model_time_datetime)
#        
#        # if no particles released, then no need for windage
#        if len(sc['positions']) == 0:
#            return
#        
#        for spill in sc.spills:
#            spill_mask = sc.get_spill_mask(spill)
#            if np.any(spill_mask):
#                sc['windages'][spill_mask] = rand.random_with_persistance(spill.windage_range[0],
#                                                                  spill.windage_range[1],
#                                                                  spill.windage_persist,
#                                                                  time_step,
#                                                                  array_len=np.count_nonzero(spill_mask))
#    
#    
#    def get_move(self, sc, time_step, model_time_datetime):
#        """
#        Override base class functionality because mover has a different get_move signature
#        
#        :param sc: an instance of the gnome.SpillContainer class
#        :param time_step: time step in seconds
#        :param model_time_datetime: current time of the model as a date time object
#        """
#        self.prepare_data_for_get_move(sc, model_time_datetime)
#        
#        if self.active and len(self.positions) > 0: 
#            self.mover.get_move(  self.model_time,
#                                  time_step, 
#                                  self.positions,
#                                  self.delta,
#                                  sc['windages'],
#                                  self.status_codes,
#                                  self.spill_type,
#                                  0)    # only ever 1 spill_container so this is always 0!
#            
#        return self.delta.view(dtype=basic_types.world_point_type).reshape((-1,len(basic_types.world_point)))
#===============================================================================
