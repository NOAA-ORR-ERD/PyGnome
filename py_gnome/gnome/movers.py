import numpy as np

from gnome.utilities import time_utils, transforms
from gnome import basic_types
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_random_mover import CyRandomMover

class Mover(object):
    """
    Base class from which all Python movers can inherit
    
    It defines the interface for a Python mover. The model expects the methods defined here. 
    The get_move(...) method needs to be implemented by the derived class.  
    """
    def __init__(self, **kwargs):
        """
        During init, it defaults is_active = True
        """
        self.is_active = True
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
        super(CyMover,self).__init__(**kwargs)
        self.is_active = is_active
        
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
    def __init__(self, 
                 timeseries=None, 
                 data_format=basic_types.data_format.magnitude_direction,
                 file=None, 
                 uncertain_duration=10800, is_active=True,
                 uncertain_time_delay=0, uncertain_speed_scale=2, uncertain_angle_scale=0.4):
        """
        Initializes a wind mover object. It requires a numpy array containing 
        gnome.basic_types.time_value_pair which defines the wind velocity
        
        :param timeseries: (Required) numpy array containing time_value_pair
        :type timeseries: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param file: path to a long wind file from which to read wind data
        :param uncertain_duraton=10800: only used in case of uncertain wind. Default is 3 hours
        :param is_active: flag defines whether mover is active or not
        :param uncertain_time_delay=0: wait this long after model_start_time to turn on uncertainty
        :param uncertain_speed_scale=2: used in uncertainty computation
        :param uncertain_angle_scale=0.4: used in uncertainty computation
        """
        if( timeseries == None and file == None):
            raise ValueError("Either provide timeseries or a valid long file")
        
        if( timeseries != None):
            try:
                if( timeseries.dtype is not basic_types.datetime_value_2d):
                    # Should this be 'is' or '==' - both work in this case. There is only one instance of basic_types.time_value_pair 
                    raise ValueError("timeseries must be a numpy array containing basic_types.datetime_r_theta dtype")
            
            except AttributeError as err:
                raise AttributeError("timeseries is not a numpy array. " + err.message)
            
            # convert datetime_value_2d to time_value_pair
            if data_format == basic_types.data_format.magnitude_direction:
                time_value_pair = self._datetime_r_theta_to_time_value(timeseries)
            elif data_format == basic_types.data_format.wind_uv:
                time_value_pair = self._datetime_uv_to_time_value(timeseries)
                
            self.ossm = CyOSSMTime(timeseries=time_value_pair) # this has same scope as CyWindMover object
            
        else:
            self.ossm = CyOSSMTime(path=file,file_contains=data_format)
        
        self.mover = CyWindMover(uncertain_duration=uncertain_duration, 
                                 uncertain_time_delay=uncertain_time_delay, 
                                 uncertain_speed_scale=uncertain_speed_scale,  
                                 uncertain_angle_scale=uncertain_angle_scale)
        self.mover.set_ossm(self.ossm)
        super(WindMover,self).__init__(is_active=is_active)

    @property
    def timeseries(self):
        datetimeval = self._time_value_to_datetime_r_theta(self.ossm.timeseries)
        return datetimeval
    
    @timeseries.setter
    def timeseries(self, datetime_value_2d):
        timeval = self._datetime_r_theta_to_time_value(datetime_value_2d)
        self.ossm.timeseries = timeval
        
    @property
    def uncertain_duration(self):
        return self.mover.uncertain_duration
    
    @uncertain_duration.setter
    def uncertain_duration(self, value):
        self.mover.uncertain_duration = value
    
    @property
    def uncertain_time_delay(self):
        return self.mover.uncertain_time_delay
    
    @uncertain_time_delay.setter
    def uncertain_time_delay(self, value):
        self.mover.uncertain_time_delay = value
        
    @property 
    def uncertain_speed_scale(self):
        return self.mover.uncertain_speed_scale
    
    @uncertain_speed_scale.setter
    def uncertain_speed_scale(self, value):
        self.mover.uncertain_speed_scale = value
        
    @property
    def uncertain_angle_scale(self):
        return self.mover.uncertain_angle_scale
    
    @uncertain_angle_scale.setter
    def uncertain_angle_scale(self, value):
        self.mover.uncertain_angle_scale = value

    def _datetime_uv_to_time_value(self, datetime_value_2d):
        """
        converts the datetime_value_2d array to a time_value_pair array
        """
        timeval = np.zeros((len(datetime_value_2d),), dtype=basic_types.time_value_pair)
        timeval['time'] = time_utils.date_to_sec(datetime_value_2d['time'])
        timeval['value']['u'] = datetime_value_2d['value'][:,0]
        timeval['value']['v'] = datetime_value_2d['value'][:,1]
        return timeval

    def _datetime_r_theta_to_time_value(self, datetime_r_theta):
        """
        convert the datetime_value_2d array to a time_value_pair array
        """
        timeval = np.zeros((len(datetime_r_theta),), dtype=basic_types.time_value_pair)
        timeval['time'] = time_utils.date_to_sec(datetime_r_theta['time'])
        uv = transforms.r_theta_to_uv_wind(datetime_r_theta['value'])
        timeval['value']['u'] = uv[:,0]
        timeval['value']['v'] = uv[:,1]
        return timeval
        
    def _time_value_to_datetime_r_theta(self, time_value_pair):
        """
        convert the time_value_pair array to a datetime_value_2d array
        """
        datetimeval = np.zeros((len(time_value_pair),), dtype=basic_types.datetime_value_2d)
        datetimeval['time'] = time_utils.sec_to_date(time_value_pair['time'])
        
        # remove following three after reworking the time_value_pair array
        uv = np.zeros((len(time_value_pair),2), dtype=np.double)
        uv[:,0] = time_value_pair['value']['u']
        uv[:,1] = time_value_pair['value']['v']
        
        datetimeval['value'] = transforms.uv_to_r_theta_wind(uv)
        return datetimeval
    
    def get_time_value(self, datetime):
        time_sec = self.datetime_to_seconds(datetime)
        return self.ossm.get_time_value(time_sec).view(dtype=np.double).reshape(-1,len(basic_types.velocity_rec))

    def __repr__(self):
        """
        Return a string representation of this `WindMover`.

        TODO: We probably want to include more information.
        """
        return 'Wind Mover'

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
        
