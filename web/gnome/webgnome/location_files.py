#!/usr/bin/env python

"""
Classes for Location files

These should hold all the information about how GNOME is set up for a given location

"""

import os
import math

import gnome
import gnome.model
import gnome.c_gnome
import datetime

class LocationFile:
    """
    The base class for a Location File
    
    There isn't much here, but it will serve as documentation for the interface
    """
    ## set of defaults -- the can/should be overridden in subclasses

    run_duration = (0, 24*3600) # in seconds 
    time_step = 15*60 # 15 minutes in seconds 

    map_image_size = (800, 500)
    map_file_name = "LongIslandSoundMap.bna"
    map_refloat_halflife = 300

    def __init__(self, model_start_time, model_stop_time, timestep):
        """
        Initializing the location file should start up the model, and load
        the basics
        
        all location files with have a map, for instance -- some may (will) have other stuff to load up
        
        """
        
        self.model = gnome.model.Model()
        self.model.set_run_duration(model_start_time, model_stop_time)
        self.model.set_timestep(timestep)        
        self.model.add_map(self.map_image_size,
                           self.map_file_name,
                           self.map_refloat_halflife)
                
        self.set_default_movers()
                                  
    def set_default_movers(self):
        """
        Sets up the default movers for the location
        
        only a simple diffusion (random) mover here
         
         This is expected to be overridden by subclasses
        """
        self.model.add_random_mover(100000)
    
    def set_wind_mover(self, vector):
        self.model.add_wind_mover(vector)

    def get_map_image(self):
        return self.model.get_map_image()
    
    def run(self, output_dir):
        """
        runs the model formthestart to the end
        
        output_dir is the path you want the png files written to
        
        returns a list of png files -- one for each time step
        """
        out_files = []
 
        while True:
            print "running another step:", self.model.time_step
            filename = self.model.step(output_dir)
            if filename is False:
                break
            out_files.append(filename)
        return out_files
 
    def reset(self):
        """
        resets the model to start_time,and removes all spills
        
        """
        self.model.reset()
        self.model.spills = []

    def set_spill(self, num_particles, start_time, location):
        self.model.set_spill(num_particles,
                             self.windage,
                             (start_time, start_time),
                             (location, location),
                             )

class LowerMississippiRiver( LocationFile ):

    run_duration = (0, 24*3600) # in seconds 
    time_step = 10*60 # 10 minutes in seconds 

    map_image_size = (800, 500)
    map_file_name = "./LMiss.bna"
    topology_file = "./LMiss.CUR"

    # some spill defaults
    windage = 0.03
    num_particles = 1000
    scale_type = 1  # constant scaling.
    scale_value = 1 # leaving alone for now.

    def __init__(self, model_start_time, model_stop_time, timestep, scale_value=None):
        if scale_value:
            self.scale_value = scale_value
        LocationFile.__init__(self, model_start_time, model_stop_time, timestep)
    
    def set_default_movers(self):
        LocationFile.set_default_movers(self)
        self.model.add_cats_mover(self.topology_file, self.scale_type, (-89.699944, 29.494558), self.scale_value)

class LongIslandSound( LocationFile ):

    run_duration = (0, 24*3600) # in seconds 
    time_step = 10*60 # 10 minutes in seconds 

    map_image_size = (800, 500)
    map_file_name = "LongIslandSoundMap.bna"
    topology_file = "./tidesWAC.CUR"
    shio_file = "./CLISShio.txt"
    
    # some spill defaults
    windage = 0.03
    num_particles = 1000
    scale_type = 1  # constant scaling.
    scale_value = 1 # leaving alone for now.
    
    def __init__(self, model_start_time, model_stop_time, timestep, scale_value=None):
        if scale_value:
            self.scale_value = scale_value
        LocationFile.__init__(self, model_start_time, model_stop_time, timestep)
    
    def set_default_movers(self):
        LocationFile.set_default_movers(self)
        self.model.add_cats_mover(self.topology_file, self.scale_type, self.shio_file, self.scale_value) # value needs to be changed here.

if __name__ == "__main__":
    """
    a simple test case, to see how it works and how to use it.
    """

    location = LongIslandSound('12/11/2012 06:55:00', '12/12/2012 06:55:00', 900)
    location.set_spill(start_time='12/11/2012 06:55:00',
                       location = (-72.419882,41.202120))
    png_files = location.run(os.path.join('.', 'temp1'))
    print png_files
    
    
    
