"""
Base class for python wrappers around cython movers
All movers (CyWindMover, RandomMover) will need to extract info from spill object.
Put this code and any other code common to all movers here
"""
from gnome import basic_types
import numpy as np

class PyMovers():
    def organize_spill(self, spill):
        """
        organizes the spill object into inputs for get_move(...)
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param time_step: time_step in seconds
        :param model_time: current model time as a datetime instance
        """
        # Get the data:
        try:
            self.positions      = spill['positions']
            self.status_codes   = spill['status_codes']
            
            # pick the correct enum type consistent with C++, based on spill certainty/uncertainty
            if spill.is_uncertain:
                self.spill_type = basic_types.spill_type.uncertainty
            else:
                self.spill_type = basic_types.spill_type.forecast
            
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        # Array is not the same size, change view and reshape
        self.positions = self.positions.view(dtype=basic_types.world_point)
        self.positions = np.reshape(self.positions, (len(self.positions),))
        
        self.delta = np.zeros((len(self.positions)), dtype=basic_types.world_point)
        
"""
Wind_mover.py

Python wrapper around the Cython wind_mover module
This inherits CyWindMover as well as PyMovers.

But PyMovers really just has some functions to capture
common functionality of all Movers. Should we inherit from PyMovers
or just declare it and use it locally in methods like get_move(..)
"""
from gnome.cy_gnome.cy_wind_mover import CyWindMover
from gnome.cy_gnome.cy_ossm_time import CyOSSMTime

class WindMover(CyWindMover, PyMovers):
    """
    WindMover class
    
    the real work is done by the CyWindMover object from which CyWindMover derives
    
    PyMovers sets everything up that is common to all movers
    """
    def __init__(self, wind_vel=None, wind_file=None, wind_duration=10800):
        """
        Should this object take as input an CyOSSMTime object or constant wind velocity.
        If so, then something outside (model?) maintains the CyOSSMTime object
        
        :param wind_vel: numpy array containing time_value_pair
        :type wind_vel: numpy.ndarray[basic_types.time_value_pair, ndim=1]
        :param wind_file: path to a long wind file from which to read wind data
        :param wind_duraton: only used in case of variable wind. Default is 3 hours
        """
        if( wind_vel == None and wind_file == None):
            raise ValueError("Either provide wind_vel or a valid long wind_file")
        
        if( wind_vel != None):
            try:
                if( wind_vel.dtype is not basic_types.time_value_pair):
                    # Should this be 'is' or '==' - I believe both work in this case
                    raise ValueError("wind_vel must be a numpy array containing basic_types.time_value_pair dtype")
            
            except AttributeError as err:
                raise AttributeError("wind_vel is not a numpy array. " + err.message)
            
            self.ossm = CyOSSMTime(timeseries=wind_vel) # this has same scope as CyWindMover object
            
        else:
            self.ossm = CyOSSMTime(path=wind_file)
            
        CyWindMover.__init__(self, wind_duration=wind_duration)
        CyWindMover.set_ossm(self, self.ossm)
        
    
    def get_move(self, spill, time_step, model_time_seconds):
        PyMovers.organize_spill(self, spill)
        
        try:
            windage = spill['windages']
        except:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        CyWindMover.get_move( self,
                              model_time_seconds,
                              time_step, 
                              self.positions,
                              self.delta,
                              windage,
                              self.status_codes,
                              self.spill_type,
                              0)
        return self.delta
