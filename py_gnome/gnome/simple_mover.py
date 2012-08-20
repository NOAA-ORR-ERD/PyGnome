#!/usr/bin/env python

"""
simple_mover.py

This is a an example mover class -- not really useful, but about as simple as
can be, for testing and demonstration purposes

"""

import numpy as np

from gnome import basic_types

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

        :param velocity: a (u, v) pair -- in meters per second

        """
        self.velocity = np.asarray( velocity,
                                    dtype = basic_types.mover_type, # use this, to be compatible with whatber we are using for location
                                    ).reshape((2,))
        
        
    def get_move(self, spill, time_step):
        """
        moves the particles defined in the spill object
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param time_step: time_step in seconds
        
        In this case, it uses the:
            positions
            next_positions
            status_code
        data arrays.
        
        """
        
        # Get the data:
        try:
            positions = spill['positions']
            next_positions = spill['next_positions']
            status_codes = spill['status_codes']
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        # which ones should we move?
        in_water_mask = status_codes == basic_types.status_in_water
        
        # compute the move
        delta = np.array((in_water_mask.sum(),), dtype = basic_types.mover_type)
        delta[:] = self.velocity * time_step

        # scale for projection
        # fixme -- move this to a utility function???
        latitudes = positions[in_water_mask, 1]
        scale = np.deg2rad(lattitudes)
        delta[:,0] *= scale

        next_positions[in_water_mask] += delta
        

        