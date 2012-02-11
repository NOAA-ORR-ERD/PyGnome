#!/usr/bin/env python

"""
Classes for Location files

These should hold all the information about how GNOME is set up for a given location

"""

#fixme: imports will change as we re-structure the gnome python package.

import gnome

class LocationFile:
    """
    The base class for a Location File
    
    There isn't much here, but it will serve as documentation for the interface
    """
    ## set of defaults -- thes can/should be overridden in subclasses

    run_duration = (0, 24*3600 # in seconds -- but what units should it be?)

    def __init__(self):
        """
        Inititalizing the location file should start up the model, and load
        the basics
        
        all location files with have a map, for instance -- some may (will) have other stuff to load up
        
        """
        
        self.model = model.Model
        self.model.startTime = datetime.datetime.now() ##fixme -- does the model need a start time at this point?
        self.model.set_run_duration( *self.run_duration) 0, 24*3600 # in seconds -- but what units should it be?
        self.model.set_timestep(600) # 10 minutes in seconds 
        
        self.model.set_map()
        
        self.set_movers()
                          
    def set_map(self):
        """
        This sets a simple map with only a bound -- 
          it is expected to be overridden by a real location subclass
        """
        #self.model.set_map( gnome.map.SimpleMap(bounds = (min_lat=40.0,
        #                                                  max_lat=45.0,
        #                                                  min_lon=-135.0
        #                                                  max_lon=-130.0),
        #                                        ) )
        self.set_up_mapcanvas()
        
    def set_movers(self):
        """
        Sets up the default movers for the location
        
        only a simple diffusion (random) mover here
         
         This is expected to be overridden by subclasses
        """
       
        self.model.add_random_mover(1e5)
        
    
    def set_up_map_canvas(self):
        """
        This initializes the map canvas: the models map is expected to be set up now.
        """
        self.map_canvas = map_canvas.MapCanvas(size = (800, 500) )
        self.draw_land(self, [], BB=self.map.boundingbox) # no land in this case, but that initializes it.
        
    
    def get_map_image(self):
        
        return self.map_canvas.image
    
class LongIslandSound( LocationFile):
    mapfile = "LongIslandSoundMap.BNA"
    map_size = (800, 500)
    pass  


if __name__ == "__main__":
    """
    a simple test case, to see how it works and how to use it.
    """

    location = LocationFile()
    
    
