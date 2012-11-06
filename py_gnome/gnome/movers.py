import numpy as np

from gnome.utilities import time_utils
from gnome import basic_types
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime
from gnome.cy_gnome.cy_random_mover import CyRandomMover

class PyMover(object):
    """
    Base class for python wrappers around cython movers
    All movers (CyWindMover, RandomMover) will need to extract info from spill object.
    Put this code and any other code common to all movers here
    
    Base class assumes any derived class will instantiate a 'mover' object that
    has methods like: prepare_for_model_run, prepare_for_model_step,
    """
    def __init__(self, is_active=True):
        self.is_active = is_active  # all movers need this flag

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
        self.model_time = self.datetime_to_seconds(model_time_datetime)
        if uncertain_spills_count < 0:
            raise ValueError("The uncertain_spills_count cannot be less than 0")
        elif uncertain_spills_count == 0:
            """" preparing for a certainty spill """
            self.spill_type = basic_types.spill_type.forecast
        else:
            self.spill_type = basic_types.spill_type.uncertainty
            
            if uncertain_spills_size is None:
                raise ValueError("uncertain_spills_size is invalid for the input uncertain_spills_count")
            
            if len(uncertain_spills_size) != uncertain_spills_count:
                raise ValueError("uncertain_spills_size needs an entry for each of the uncertain spills")
        
        # TODO: would be nice to save self.uncertain_spills_size for each step and check the spills.num_LEs match
        # when get_move is called. 
        self.mover.prepare_for_model_step(self.model_time, time_step, uncertain_spills_count, uncertain_spills_size)

    def prepare_data_for_get_move(self, spill, model_time_datetime):
        """
        organizes the spill object into inputs for get_move(...)
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param model_time_datetime: model time as a date time object
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)
        
        # Get the data:
        try:
            self.positions      = spill['positions']
            self.status_codes   = spill['status_codes']
            
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


class WindMover(PyMover):
    """
    WindMover is a Python wrapper around the Cython wind_mover module.
    This inherits CyWindMover as well as PyMover.
    
    The real work is done by the CyWindMover object.

    PyMover sets everything up that is common to all movers.
    """
    def __init__(self, timeseries=None, file=None, uncertain_duration=10800, is_active=True,
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
                if( timeseries.dtype is not basic_types.time_value_pair):
                    # Should this be 'is' or '==' - both work in this case. There is only one instance of basic_types.time_value_pair 
                    raise ValueError("timeseries must be a numpy array containing basic_types.time_value_pair dtype")
            
            except AttributeError as err:
                raise AttributeError("timeseries is not a numpy array. " + err.message)
            
            self.ossm = CyOSSMTime(timeseries=timeseries) # this has same scope as CyWindMover object
            
        else:
            self.ossm = CyOSSMTime(path=file)
        
        if len(timeseries) == 1:
            self.constant_wind = True
        else:
            self.constant_wind = False
        
        self.mover = CyWindMover(uncertain_duration=uncertain_duration, 
                                 uncertain_time_delay=uncertain_time_delay, 
                                 uncertain_speed_scale=uncertain_speed_scale,  
                                 uncertain_angle_scale=uncertain_angle_scale)
        self.mover.set_ossm(self.ossm)
        PyMover.__init__(self, is_active=is_active)

    @property
    def is_constant(self):
        return self.constant_wind

    @property
    def timeseries(self):
        return self.ossm.timeseries
    
    @timeseries.setter
    def timeseries(self, value):
       self.ossm.timeseries = value

    def get_time_value(self, datetime):
        time_sec = self.datetime_to_seconds(datetime)
        return self.ossm.get_time_value(time_sec)

    def __repr__(self):
        """
        Return a string representation of this `WindMover`.

        TODO: We probably want to include more information.
        """
        return 'Wind Mover'

    def get_move(self, spill, time_step, model_time_datetime):
        """
        TODO: Currently, spill_id is 0. This needs to be updated!
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
                              0)
        return self.delta


class RandomMover(PyMover):
    """
    Python wrapper around the Cython CyRandomMover module.
    
    The real work is done by the cython object.

    PyMover sets everything up that is common to all movers.
    """
    def __init__(self, diffusion_coef=100000, is_active=True):
        """
        Initialize
        """
        self.mover = CyRandomMover(diffusion_coef=diffusion_coef)
    
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
        This mover only needs to know if spill is uncertain or not; however, it has the same signature as movers that require
        this info. Perhaps it is best to put the uncertain flag in the prepare_for_model_step
        """
        model_time = PyMover.datetime_to_seconds(self, model_time_datetime)
        if uncertain_spills_count != 0:
            self.spill_type = basic_types.spill_type.uncertain
            self.mover.prepare_for_model_step(model_time, time_step, uncertain=True)
        else:
            self.spill_type = basic_types.spill_type.forecast
            self.mover.prepare_for_model_step(model_time, time_step, uncertain=False)
    
    def get_move(self, spill, time_step, model_time_datetime):
        """
        TODO: Currently, spill_id is 0. This needs to be updated!
        """
        self.prepare_data_for_get_move(spill, model_time_datetime)
        
        self.mover.get_move(  self.model_time,
                              time_step, 
                              self.positions,
                              self.delta,
                              self.status_codes,
                              self.spill_type,
                              0)
        return self.delta
    