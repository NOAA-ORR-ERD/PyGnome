#!/usr/bin/env python
import os
from datetime import datetime, timedelta
from collections import OrderedDict

import gnome

from gnome.utilities.time_utils import round_time
import numpy as np
import copy

class Model(object):
    
    """ 
    PyGNOME Model Class
    
    """
    def __init__(self):
        """ 
        Initializes model attributes. 
        """
        
        self._uncertain = False # sets whether uncertainty is on or not.
        self.reset() # initializes everything to defaults/nothing

    def reset(self):
        """
        Resets model to defaults -- Caution -- clears all movers, spills, etc.
        
        """

        self.output_map = None
        self.map = None
        self._wind = OrderedDict()  #list of wind objects
        self._movers = OrderedDict()
        self._spill_container = gnome.spill_container.SpillContainer()
        self._uncertain_spill_container = None
        
        self._start_time = round_time(datetime.now(), 3600) # default to now, rounded to the nearest hour
        self._duration = timedelta(days=2) # fixme: should round to multiple of time_step?
        self.time_step = timedelta(minutes=15).total_seconds()

        self.is_uncertain = False
        self.rewind()
        
    def rewind(self):
        """
        resets the model to the beginning (start_time)
        """
        self.current_time_step = -1 # start at -1
        self.model_time = self._start_time
        self._spill_container.reset()
        if self._uncertain:
            self._uncertain_spill_container = self._spill_container.copy(uncertain=True)
        else:
            self._uncertain_spill_container = None
        ## fixme: do the movers need re-setting? -- or wait for prepare_for_model_run?

    ### Assorted properties
    @property
    def is_uncertain(self):
        return self._uncertain
    @is_uncertain.setter
    def is_uncertain(self, uncertain_value):
        """
        only if uncertainty switch is toggled, then restart model
        """
        if self._uncertain != uncertain_value:
            self._uncertain = uncertain_value
            self.rewind()   
    
    @property
    def id(self):
        """
        Return an ID value for this model.

        :return: an integer ID value for this model
        """
        return id(self)

    @property
    def wind(self):
        """
        Return a list of wind objects added to the model, in order of insertion
        
        :return: a list of wind objects
        """
        return self._wind.values()

    @property
    def movers(self):
        """
        Return a list of the movers added to this model, in order of insertion.

        :return: a list of movers
        """
        return self._movers.values()

    @property
    def spills(self):
        """
        Return a list of the spills added to this model, in order of insertion.

        :return: a list of spills
        """
        return self._spill_container.spills

#    ## uncertainspills mirror the regular ones... 
#    @property
#    def uncertain_spills(self):
#        return self.uncertain_spill_container.spills
        
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
        except AttributeError: # not a timedelta object -- assume it's in seconds.
            self._time_step = int(time_step)
        self._num_time_steps = self._duration.total_seconds() // self._time_step
        self.rewind()
    
    @property
    def current_time_step(self):
        return self._current_time_step
    @current_time_step.setter
    def current_time_step(self, step):
        self.model_time = self._start_time + timedelta(seconds=step*self.time_step)
        self._current_time_step = step

    @property
    def duration(self):
        return self._duration
    @duration.setter
    def duration(self, duration):
        if duration < self._duration: # only need to rewind if shorter than it was...
            ## fixme: actually, only need to rewide is current model time is byond new time...
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

    def get_mover(self, mover_id):
        """
        Return a :class:`gnome.movers.Mover` in the ``self._movers`` dict with
        the key ``mover_id`` if one exists.
        """
        return self._movers.get(mover_id, None)

    def add_mover(self, mover):
        """
        add a new mover to the model -- at the end of the stack

        :param mover: an instance of one of the gnome.movers classes
        """
        self._movers[mover.id] = mover
        return mover.id

    def remove_mover(self, mover_id):
        """
        remove the passed-in mover from the mover list
        """
        if mover_id in self._movers:
            del self._movers[mover_id]

    # def replace_mover(self, mover_id, new_mover):
    #     """
    #     replace a given mover with a new one

    #     this is probably broken -- ids won't match!
    #     """
    #     self._movers[mover_id] = new_mover
    #     return new_mover

    def get_spill(self, spill_id):
        """
        Return a :class:`gnome.spill.Spill` in the ``self._spills`` dict with
        the key ``spill_id`` if one exists.
        """
        return self._spill_container.get_spill(spill_id)

    def add_spill(self, spill):
        """
        add a spill to the model

        :param spill: an instance of one of the gnome.spill classes

        """
        #fixme: where should we check if a spill is in a valid location on the map?
        self._spill_container.add_spill(spill)

    def remove_spill(self, spill_id):
        """
        remove the passed-in spill from the spill list
        """
        ##fixme: what if we want to remove by reference, rather than id?
        self._spill_container.remove_spill_by_id(spill_id)

    def get_wind(self, id):
        """
        Return a :class:`gnome.weather.Wind` in the ``self._wind`` dict with
        the key ``id`` if one exists.
        """
        return self._wind.get(id, None)

    def add_wind(self, obj):
        """
        add a new Wind to the model -- at the end of the stack
        """
        self._wind[obj.id] = obj
        return obj.id

    def remove_wind(self, id):
        """
        remove the passed-in Wind from the wind list
        """
        if id in self._wind:
            del self._wind[id]

    def replace_wind(self, id, new_obj):
        """
        replace a given Wind with a new one
        """
        self._wind[id] = new_obj

    def setup_model_run(self):
        """
        Sets up each mover for the model run
        
        Currently, only movers need to initialize at the beginning of the run
        """
        for mover in self.movers:
            mover.prepare_for_model_run()

        self._spill_container.reset()

    def setup_time_step(self):
        """
        sets up everything for the current time_step:
        
        right now only prepares the movers -- maybe more later?.
        """
        
        # initialize movers differently if model uncertainty is on
        for mover in self.movers:
            mover.prepare_for_model_step(self.model_time, self.time_step)
                                
    def move_elements(self):
        """ 
        Moves elements: loops through all the movers. and moves the elements
            -- sets new_position array for each spill
            -- calls the beaching code to beach the elements that need beaching.
            -- sets the new position
        """
        ## if there are no spills, there is nothing to do:
        if self._spill_container.spills:
            containers = [ self._spill_container ]
            if self.is_uncertain:
                containers.append( self._uncertain_spill_container )
            for sc in containers: # either one or two, depending on uncertaintly or not
                # reset next_positions
                sc['next_positions'][:] = sc['positions']

                # loop through the movers
                for mover in self.movers:
                    delta = mover.get_move(sc, self.time_step, self.model_time)
                    sc['next_positions'] += delta
            
                self.map.beach_elements(sc)

                # the final move to the new positions
                sc['positions'][:] = sc['next_positions']

    def step_is_done(self):
        """
        loop through movers and call model_step_is_done
        """

        for mover in self.movers:
            mover.model_step_is_done()
    
    # def write_output(self):
    #     """
    #     write the output of the current time step to whatever output
    #     methods have been selected
    #     """
    #     for output_method in self.output_types:
    #         if output_method == "image":
    #             self.write_image()
    #         else:
    #             raise ValueError("%s output type not supported"%output_method)
    #     return (self.current_time_step, filename, self.model_time.isoformat())

    def write_image(self, images_dir):
        ##fixme: put this in an "Output" class?
        """
        render the map image, according to current parameters

        :param images_dir: directory to write the image to.

        """
        if self.output_map is None:
            raise ValueError("You must have an ouput map to use the image output")
        if self.current_time_step == 0:
            self.output_map.draw_background()
            self.output_map.save_background(os.path.join(images_dir, "background_map.png"))

        filename = os.path.join(images_dir, 'foreground_%05i.png'%self.current_time_step)

        self.output_map.create_foreground_image()
        if self.is_uncertain:
            self.output_map.draw_elements(self._uncertain_spill_container)
        self.output_map.draw_elements(self._spill_container)
            
        self.output_map.save_foreground(filename)
        return filename

    def step(self):
        """
        Steps the model forward (or backward) in time. Needs testing for hindcasting.
                
        """
        if self.current_time_step >= self._num_time_steps:
            return False
        
        if self.current_time_step == -1:
            self.setup_model_run() # that's all we need to do for the zeroth time step
        else:    
            self.setup_time_step()
            self.move_elements()
            self.step_is_done()
        self.current_time_step += 1        
        self._spill_container.release_elements(self.model_time)
        if self.is_uncertain:
            self._uncertain_spill_container.release_elements(self.model_time)
        return True
    
    def __iter__(self):
        """
        for compatibility with Python's iterator protocol
        
        rewinds the model and returns itself so it can be iterated over. 
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

                
    def next_image(self, images_dir):
        """
        compute the next model step, render an image, and return info about the
        step rendered
        :param images_dir: directory to write the image too.
        """
        # run the next step:
        if not self.step():
            raise StopIteration
        filename = self.write_image(images_dir)
        return (self.current_time_step, filename, self.model_time.isoformat())

    def full_run_with_image_output(self, output_dir):
        """
        Do a full run of the model, outputting an image per time step.
        """
        
        # run the model
        while True:
            try:
                image_info = model.next_image()
            except StopIteration:
                print "Done with the model run"
                break


        
