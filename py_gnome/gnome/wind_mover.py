#!/usr/bin/env python

"""
Wind_mover.py

Python wrapper around the Cython wind_mover module

"""

## make sure the main lib is imported
import gnome.cy_basic_types

from gnome.cy_wind_mover import Cy_wind_mover

##fixme: should this use delegation, rather than subclassing?
class WindMover(Cy_wind_mover):
    """
    WindMover class
    
    the real work is delegated to the cython class
    
    but this sets everything up
    """
    def __init__(self, wind_vel=):
        """
        not much here, but there should be!
        """
        Cy_wind_mover.__init__(self)
        
        # keep the cython version around
        self_get_move = Cy_wind_mover.get_move
            
    def get_move(self, spill, time_step, model_time):
        """
        moves the particles defined in the spill object
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param time_step: time_step in seconds
        :param model_time: current model time as a datetime instance
        
        In this case, it uses the:
            positions
            status_code
        data arrays.
        
        """
        
        # Get the data:
        try:
            positions      = spill['positions']
            status_codes   = spill['status_codes']
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        model_time_seconds = basic_types.dt_to_epoch(model_time)        
        positions = position.view(dtype = basic_types.world_point)
        delta = np.zeros_like(positions)
                
        ## should we need these???
        windage_array = np.ones((spill.num_LEs,), dtype=basic_types.mover_type)
        dispersion_array = np.zeros((spill.num_LEs,), dtype=np.short) ## what should be the dtype here???

        ## we certainly shouldn't need these!!
        time_vals = np.empty((1,), dtype=basic_types.time_value_pair)
        # Straight south wind... 10 meters per second
        time_vals['value']['u'] =  0  # meters per second?
        time_vals['value']['v'] = 10 # 

        # initialize uncerainty array:
        # fixme: this needs to get stored with the Mover -- keyed to a particular spill.
        uncertain_ra = np.empty((spill.num_LEs,), dtype=basic_types.wind_uncertain_rec)	# one uncertain rec per le
        for x in range(0, N):
            theta = random()*2*pi
            uncertain_ra[x]['randCos'] = cos(theta)
            uncertain_ra[x]['randSin'] = sin(theta)

        ## or these:
        ## dump for now
        breaking_wave = 10 #?? 
        mix_layer_depth = 10 #??

        # call the Cython version 
        # (delta is changed in place)
        self._get_move(self,
                       model_time,
                       time_step,
                       positions,
                       delta,
                       windage_array,
                       dispersion_array, # don't need
                       breaking_wave, # don't need
                       mix_layer, # don't need
                       time_vals): # don't need
        return delta

        
        
    
    
    