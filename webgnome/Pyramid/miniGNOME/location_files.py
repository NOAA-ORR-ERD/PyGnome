#!/usr/bin/env python

"""
Classes for Location files

These should hold all the information about how GNOME is set up for a given location

"""

import math

import gnome
import gnome.model
import datetime

class LocationFile:
    """
    The base class for a Location File
    
    There isn't much here, but it will serve as documentation for the interface
    """
    ## set of defaults -- thes can/should be overridden in subclasses

    run_duration = (0, 24*3600) # in seconds 
    time_step = 10*60 # 10 minutes in seconds 

    map_image_size = (800, 500)
    map_file_name = "LongIslandSoundMap.bna"
    map_refloat_halflife = 300

    def __init__(self):
        """
        Initializing the location file should start up the model, and load
        the basics
        
        all location files with have a map, for instance -- some may (will) have other stuff to load up
        
        """
        
        self.model = gnome.model.Model()
        self.model.startTime = datetime.datetime.now() ##fixme -- does the model need a start time at this point?
        self.model.set_run_duration( *self.run_duration ) 
        self.model.set_timestep(self.time_step) # 10 minutes in seconds 
        
        self.model.add_map(self.map_image_size,
                           self.map_file_name,
                           self.map_refloat_halflife)
                
        self.set_movers()
                                  
    def set_default_movers(self):
        """
        Sets up the default movers for the location
        
        only a simple diffusion (random) mover here
         
         This is expected to be overridden by subclasses
        """
        self.model.add_random_mover(1e5)
        
    def get_map_image(self):
        return self.model.get_map_image()
    
    def run_and_draw(self, path_to_pngs):
        # do a full run of GNOME
        images = self.model.full_run(path_to_pngs)
        
        # rest the model so it's ready for the next run.
        self.model.reset_steps()

 
    
class LongIslandSound( LocationFile ):
    run_duration = (0, 24*3600) # in seconds 
    time_step = 10*60 # 10 minutes in seconds 

    map_image_size = (800, 500)
    map_file_name = "LongIslandSoundMap.bna"
    
    def __init__(self):
        LocationFile.__init__(self)
    
        self.wind_mover = self.model.add_wind_mover(self, (0.0, 0.0)) # setting a zero constant wind.
        
    
    def replace_constant_wind_mover(self, speed, direction):
        """
        replace the wind mover with a new one
        
        only constant mover for now
        """
        u = speed * cos(direction/180. * math.pi)
        v = speed * sin(direction/ 180. * math.pi)
        self.wind_mover..mover.fConstantValue.u = u
        self.wind_mover..mover.fConstantValue.v = v

if __name__ == "__main__":
    """
    a simple test case, to see how it works and how to use it.
    """

    location = LongIslandSound()
    
    
