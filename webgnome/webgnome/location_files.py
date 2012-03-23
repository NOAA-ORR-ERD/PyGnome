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
    ## set of defaults -- thes can/should be overridden in subclasses

    run_duration = (0, 24*3600) # in seconds 
    time_step = 15*60 # 15 minutes in seconds 

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
        
        model_start_time = '2011-11-12 06:55:00'
        model_stop_time = '2011-12-12 06:59:00'

        self.model.set_run_duration(model_start_time, model_stop_time)
        self.model.set_timestep(900)

        self.model.set_timestep(self.time_step) # 10 minutes in seconds 
        
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
        
    def get_map_image(self):
        return self.model.get_map_image()
    
    def run(self, output_dir):
        """
        runs the model formthestart to the end
        
        output_dir is the path you want the png files written to
        
        returns a list of png files -- one for each time step
        """
        print "About to run:"
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

    def replace_constant_wind_mover(self, speed, direction):
        """
        Replace the wind mover with a new one
        
        Only the constant wind mover for now
        
        Speed is the wind speed in m/s
        
        Direction is the direction the wind is blowing from (degrees true)

        """
        u = speed * -math.sin(direction/180. * math.pi)
        v = speed * -math.cos(direction/180. * math.pi)
        
        new_mover = gnome.c_gnome.wind_mover( (u,v) )
        self.wind_mover = self.model.replace_mover(self.wind_mover, new_mover)

    def set_spill(self, start_time, location):
        start_time = 0.0 ##fixme: only good 'till we get times right!
        print location
        self.model.set_spill(self.num_particles,
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

    def __init__(self, scale_value):
        LocationFile.__init__(self)
        self.cats_mover = self.add_cats_mover(topology_file, cats_scale_type, (-89.699944, 29.494558), scale_value) # value needs to be changed here.
        self.wind_mover = self.model.add_wind_mover((0.0, 0.0)) # setting a zero constant wind.
        
    
    def reset_constant_wind_mover(self, speed, direction):
        ## this should be here -- but can't do this now
        """
        reset the values of the wind mover with a new one
        
        only constant mover for now
        
        speed is the wind speed in m/s
        direction is the direction the wind is blowing from (degrees true)
        """
        u = speed * -math.sin(direction/180. * math.pi)
        v = speed * -math.cos(direction/180. * math.pi)
        
        ## is this right??
        self.wind_mover.mover.fConstantValue.u = u
        self.wind_mover.mover.fConstantValue.v = v


class LongIslandSound( LocationFile ):

    run_duration = (0, 24*3600) # in seconds 
    time_step = 10*60 # 10 minutes in seconds 

    map_image_size = (800, 500)
    map_file_name = "LongIslandSoundMap.bna"
    
    # some spill defaults
    windage = 0.03
    num_particles = 1000

    def __init__(self):
        LocationFile.__init__(self)
    
        self.wind_mover = self.model.add_wind_mover((0.0, 0.0)) # setting a zero constant wind.
        
    
    def reset_constant_wind_mover(self, speed, direction):
        ## this should be here -- but can't do this now
        """
        reset the values of the wind mover with a new one
        
        only constant mover for now
        
        speed is the wind speed in m/s
        direction is the direction the wind is blowing from (degrees true)
        """
        u = speed * -math.sin(direction/180. * math.pi)
        v = speed * -math.cos(direction/180. * math.pi)
        
        ## is this right??
        self.wind_mover.mover.fConstantValue.u = u
        self.wind_mover.mover.fConstantValue.v = v


    

if __name__ == "__main__":
    """
    a simple test case, to see how it works and how to use it.
    """

    location = LongIslandSound()
    location.reset()
    location.replace_constant_wind_mover(speed = 5.0, direction=45.0)
    location.set_spill(start_time='2011-11-12 06:55:00',
                       location = (-89.699944, 29.494558))
    png_files = location.run(os.path.join('.', 'temp1'))
    
    ## run a new version
    location.reset()
    location.set_spill('2011-11-12 06:55:00',
                       location = (-89.699944, 29.494558))
    location.replace_constant_wind_mover(speed = 5.0, direction=270.0)
    png_files = location.run(os.path.join('.', 'temp2'))
    print png_files
    
    
    
    
    
    