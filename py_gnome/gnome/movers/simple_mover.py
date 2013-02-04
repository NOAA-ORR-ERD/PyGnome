#!/usr/bin/env python

"""
simple_mover.py

This is a an example mover class -- not really useful, but about as simple as
can be, for testing and demonstration purposes

"""

import numpy as np
from numpy import random

from gnome import basic_types
from gnome.movers import Mover

## this allows for this to be changed in the future.
from gnome.utilities.projections import FlatEarthProjection as proj
from datetime import datetime
from time import gmtime

class SimpleMover(Mover):
    """
    simple_mover
    
    a really simple mover -- moves all LEs a constant speed and direction
    
    (not all that different than a constant wind mover, now that I think about it)    
    """
    def __init__(self, velocity, uncertainty_scale=0.5,
                 active_start= datetime( *gmtime(0)[:7] ), 
                 active_stop = datetime.max):
        """
        simple_mover (velocity)

        create a simple_mover instance

        :param velocity: a (u, v, w) triple -- in meters per second

        """
        self.velocity = np.asarray( velocity,
                                    dtype = basic_types.mover_type, # use this, to be compatible with whatever we are using for location
                                    ).reshape((3,))
        self.uncertainty_scale = uncertainty_scale
        super(SimpleMover,self).__init__(active_start, active_stop)
    def __repr__(self):
        return 'SimpleMover(<%s>)' % (self.id)

    def get_move(self, spill, time_step, model_time, uncertain_spill_number=0):
        """
        moves the particles defined in the spill object
        
        :param spill: spill is an instance of the gnome.spill.Spill class
        :param time_step: time_step in seconds
        :param model_time: current model time as a datetime object
        In this case, it uses the:
            positions
            status_code
        data arrays.
        :param uncertain_spill_number: starting from 0 for the 1st uncertain spill, it is the order in which the uncertain spill is added
        
        :returns delta: Nx3 numpy array of movement -- in (long, lat, meters) units
        
        """
        # Get the data:
        try:
            positions      = spill['positions']
            status_codes   = spill['status_codes']
        except KeyError, err:
            raise ValueError("The spill does not have the required data arrays\n"+err.message)
        
        # which ones should we move?
        in_water_mask =  (status_codes == basic_types.oil_status.in_water)
                
        # compute the move
        delta = np.zeros_like(positions)
        
        if self.active and self.on:
            delta[in_water_mask] = self.velocity * time_step
            # add some random stuff if uncertainty is on
            if spill.is_uncertain:
                num = sum(in_water_mask)
                scale = self.uncertainty_scale * self.velocity * time_step
                delta[in_water_mask,0] += random.uniform( -scale[0], scale[0], num )
                delta[in_water_mask,1] += random.uniform( -scale[1], scale[1], num )
                delta[in_water_mask,2] += random.uniform( -scale[2], scale[2], num )
        # scale for projection
        delta = proj.meters_to_lonlat(delta, positions) # just the lat-lon...
        
        return delta
