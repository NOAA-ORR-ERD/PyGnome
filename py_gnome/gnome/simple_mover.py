#!/usr/bin/env python

"""
simple_mover.py

This is a an example mover class -- not really useful, but about as simple as
can be, for testing and demonstration purposes

"""

import numpy as np

from gnome import basic_types

## this allows for this to be changed in the future.
from gnome.utilities.projections import FlatEarthProjection as proj


class simple_mover(object):
    """
    simple_mover
    
    a really simple mover -- moves all LEs a constant speed and direction
    
    (not all that different than a constant wind mover, now that I think about it)    
    """

    def __init__(self, velocity):
        """
        simple_mover (velocity)

        create a simple_mover instance

        :param velocity: a (u, v, w) triple -- in meters per second

        """
        self.velocity = np.asarray( velocity,
                                    dtype = basic_types.mover_type, # use this, to be compatible with whatever we are using for location
                                    ).reshape((3,))
        
        
    def get_move(self, spill, time_step, model_time=None):
        """
        moves the particles defined in the spill object
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param time_step: time_step in seconds
        
        In this case, it uses the:
            positions
            status_code
        data arrays.
        
        :returns delta: Nx3 numpy array of movement -- in (long, lat, meters) units
        
        """
        
        # Get the data:
        try:
            positions      = spill['positions']
            status_codes   = spill['status_codes']
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        # which ones should we move?
        in_water_mask =  (status_codes == basic_types.oil_status.status_in_water)
                
        # compute the move
        delta = np.zeros((in_water_mask.sum(), 3), dtype = basic_types.mover_type)

        delta[:] = self.velocity * time_step

        print "delta:", delta

        # scale for projection
        # fixme -- move this to a utility function???
        print "positions", positions
        latitudes = positions[in_water_mask, 1]
        print "latitudes:", latitudes
        print "delta:", delta
        delta = proj.meters_to_latlon(delta[:,:2], latitudes) # just the lat-lon...
        print "delta:", delta
        
        return delta
        

        
