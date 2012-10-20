#!/usr/bin/env python

import os
from datetime import datetime, timedelta

import gnome

from gnome import greenwich

from gnome.utilities.time_utils import round_time

class Model(object):
    
    """ 
    PyGNOME Model Class
    
    """
    def __init__(self):
        """ 
        Initializes model attributes. 
        """
        self.uncertain_on = False # sets whether uncertainty is on or not.

        self._start_time = round_time(datetime.now(), 3600) # default to now, rounded to the nearest hour
        self._duration = timedelta(days=2) # fixme: should round to multiple of time_step?

        ## Various run-time parameters for output
        self.output_types = [] # default to no output type -- there could be: "image", "netcdf", etc)
        self.images_dir = '.'

        self.reset() # initializes everything to nothing.
        
    def reset(self):
        """
        Resets model to defaults -- Caution -- clears all movers, etc.
        
        """
        self.output_map = None
        self.map = None
        self.movers = []
        self.spills = []

        self._current_time_step = -1

        
        self.time_step = timedelta(minutes=15).total_seconds()

        
        self.uncertain = False
        
    def rewind(self):
        """
        resets the model to the beginning (start_time)
        """
        self._current_time_step = -1 # start at -1 -- it get incremented first.
        for spill in self.spills:
            spill.reset()
        ## fixme: do the movers need re-setting? -- or wait for prepare_for_model_run?

    ### Assorted properties
    @property
    def start_time(self):
        return self._start_time
    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time
        self.rewind()
    
    @property
    def time_step(self):
        return self._time_step
    @time_step.setter
    def time_step(self, time_step):
        """
        sets the time step, and rewinds the model

        :param time_step: the timestep as a timedelta object or integer seconds.

        """
        try: 
            self._time_step = time_step.total_seconds()
        except AttributeError: # not a timedelta object...
            self._time_step = int(time_step)
        self._num_time_steps = self._duration.total_seconds() // self._time_step
        self.rewind()
    
    @property
    def current_time_step(self):
        return self._current_time_step

    @property
    def duration(self):
        return self._duration
    @duration.setter
    def duration(self, duration):
        if duration < self._duration: # only need to rewind if shorter than it was...
            self.rewind()
        self._duration = duration
        self._num_time_steps = self._duration.total_seconds() // self.time_step
        
    @property
    def map(self):
        return self._map
    @map.setter
    def map(self, map):
        ## we'll want to do more here, probably
        self._map = map

    def add_mover(self, mover):
        """
        add a new mover to the model -- at the end of the stack
        """
        self.movers.append(mover)

    def remove_mover(self, mover):
        """
        remove the passed-in mover from the mover list
        """
        self.movers.remove(mover)

    def replace_mover(self, old_mover, new_mover):
        """
        replace a given mover with a new one
        """
        i = self.movers.index(old_mover)
        self.movers[i] = new_mover
        return new_mover 
   
    def add_spill(self, spill):
        #fixme: where should we check if a spill is in a valic location on the map?
        """
        add a spill to the model
        
        :param spill: an instance of the gnome.Spill class
        
        """
        self.spills.append(spill)

    def setup_time_step(self):
        """
        sets up everything for the current time_step:
        
        releases elements, refloats, prepares the movers, etc.
        """
        self.model_time = self._start_time + timedelta(seconds=self._current_time_step*self.time_step)
        for spill in self.spills:
            spill.prepare_for_model_step(self.model_time, self.time_step, self.uncertain_on)
            self.map.refloat_elements(spill)
        for mover in self.movers:
            #loop through each spill and pass it in.
            mover.prepare_for_model_step(self.model_time, self.time_step, self.uncertain_on)

    def move_elements(self):
        """ 
        Moves elements: loops through all the movers. and moves the elements
            -- sets new_position array for each spill
            -- calls the beaching code to beach the elements that need beaching.
            -- sets the new position
        """

        # Set up next_position
        for spill in self.spills:
            spill['next_positions'][:] = spill['positions']

        for mover in self.movers:
            for spill in self.spills:
                delta = mover.get_move(spill, self.time_step, self.model_time)
                spill['next_positions'] += delta
                # print "in move loop"
                # print "pos:", spill['positions']
                # print "next_pos:", spill['next_positions']
        for spill in self.spills:
            self.map.beach_elements(spill)
            # print "in map loop"
            # print "pos:", spill['positions']
            # print "next_pos:", spill['next_positions']

        # the final move to the new positions
        for spill in self.spills:
            spill['positions'][:] = spill['next_positions']

    def write_output(self):
        """
        write the output of the current time step to whatever output
        methods have been selected
        """
        for output_method in self.output_types:
            if output_method == "image":
                self.write_image()
            else:
                raise ValueError("%s output type not supported"%output_method)
    
    def write_image(self):
        ##fixme: put this in an "Output" class?
        """
        render the map image, according to current parameters
        """
        if self.output_map is None:
            raise ValueError("You must have an ouput map to use the image output")
        if self.current_time_step == 0:
            self.output_map.draw_background()
            self.output_map.save_background(os.path.join(self.images_dir, "background_map.png"))

        filename = os.path.join(self.images_dir, 'foreground_%05i.png'%self._current_time_step)

        self.output_map.create_foreground_image()
        for spill in self.spills:
            #print "drawing elements"
            #print spill['positions']
            self.output_map.draw_elements(spill)
        self.output_map.save_foreground(filename)

        return filename

    def step(self):
        """
        Steps the model forward in time. Needs testing for hindcasting.
                
        """
                
        if self._current_time_step >= self._num_time_steps:
            return False
        self._current_time_step += 1
        self.setup_time_step()
        self.move_elements()

        return True
    
    def __iter__(self):
        """
        for compatibility with Python's iterator protocol
        
        resets the model and returns itself so it can be iterated over. 
        """
        self.rewind()
        return self

    def next(self):
        """
        (This method here to satisfy Python's iterator and generator protocols)

        Compute the next model step

        Return the step number
        """
        
        if not self.step():
            raise StopIteration
        return self.current_time_step

                
    def next_image(self):
        """
        compute the next model step, render an image, and return info about the
        step rendered
        
        """
        if not self.step():
            raise StopIteration
        filename = self.write_image()
        return (self.current_time_step, filename, "a timestamp")

    def full_run_and_output(self):
        """
        Do a full run of the model, outputting whatever has been set.
        """
        raise NotImplmentedError
        

        