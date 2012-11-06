#!/usr/bin/env python
import os
from datetime import datetime, timedelta
from collections import OrderedDict

import gnome

from gnome import greenwich

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
        self._movers = OrderedDict()
        self._spills = OrderedDict()
        self._uncertain_spills = OrderedDict()

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
    def uncertain(self):
        return self._uncertain
    
    @uncertain.setter
    def uncertain(self, uncertain_value):
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
        return self._spills.values()

    @property
    def uncertain_spills(self):
        return self._uncertain_spills.values()
        
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
        self._movers[mover.id] = mover
        return mover.id

    def remove_mover(self, mover_id):
        """
        remove the passed-in mover from the mover list
        """
        if mover_id in self._movers:
            del self._movers[mover_id]

    def replace_mover(self, mover_id, new_mover):
        """
        replace a given mover with a new one
        """
        self._movers[mover_id] = new_mover
        return new_mover 
   
    def add_spill(self, spill):
        #fixme: where should we check if a spill is in a valic location on the map?
        """
        add a spill to the model
        
        :param spill: an instance of the gnome.Spill class
        
        """
        self._spills[spill.id] = spill
        
        
    def setup_model_run(self):
        """
        Sets up each mover for the model run
        
        Currently, only movers need to initialize at the beginning of the run
        """
        for mover in self.movers:
            mover.prepare_for_model_run()
            
        self._uncertain_spills = OrderedDict()
        if self.uncertain:
            self._uncertain_spill_id_map = []   # a list mapping the order in which list is added to it's unique 'id'
            for spill in self.spills:
                uSpill = copy.deepcopy(spill)
                uSpill.is_uncertain = True
                self._uncertain_spills[uSpill.id] = uSpill   # should spill ID get updated? Does this effect how movers applies uncertainty?
                
                if self._uncertain_spill_id_map.count(uSpill.id) != 0:
                    raise ValueError("An uncertain spill with this id has been defined. spill.id should be unique")
                 
                self._uncertain_spill_id_map.append(uSpill.id)
            

    
    def setup_time_step(self):
        """
        sets up everything for the current time_step:
        
        releases elements, refloats, prepares the movers, etc.
        """
        self.model_time = self._start_time + timedelta(seconds=self._current_time_step*self.time_step)
        
        for spill in self.spills:
            spill.prepare_for_model_step(self.model_time, self.time_step)
        
        # if model is uncertain, update following defaults
        num_uSpills = 0
        uSpill_size = None
        if self.uncertain:
            num_uSpills = len(self.uncertain_spills)
            uSpill_size = np.zeros((num_uSpills,), dtype=np.int)
            
            for i in range(0, num_uSpills):
                self.uncertain_spills[i].prepare_for_model_step(self.model_time, self.time_step)
                uSpill_size[i] = spill.num_LEs
        
        # initialize movers differently if model uncertainty is on
        for mover in self.movers:
            mover.prepare_for_model_step(self.model_time, self.time_step, num_uSpills, uSpill_size)
                
                
    def move_elements(self):
        """ 
        Moves elements: loops through all the movers. and moves the elements
            -- sets new_position array for each spill
            -- calls the beaching code to beach the elements that need beaching.
            -- sets the new position
        """
        for spills in (self.spills,self.uncertain_spills):
            
            for spill in spills:
                spill['next_positions'][:] = spill['positions']
    
            uncertain_spill_number = -1 # only used by get_move for uncertain spills
            for mover in self.movers:
                for spill in spills:
                    if spill.is_uncertain:
                        uncertain_spill_number = self._uncertain_spill_id_map.index((spill.id))
                        
                    delta = mover.get_move(spill, self.time_step, self.model_time, uncertain_spill_number)  # spill ID that get_move expects
                    spill['next_positions'] += delta
                    # print "in move loop"
                    # print "pos:", spill['positions']
                    # print "next_pos:", spill['next_positions']
            for spill in spills:
                self.map.beach_elements(spill)
                # print "in map loop"
                # print "pos:", spill['positions']
                # print "next_pos:", spill['next_positions']
    
            # the final move to the new positions
            for spill in spills:
                spill['positions'][:] = spill['next_positions']
        

    def step_is_done(self):
        """
        loop through movers and call model_step_is_done
        """
        for mover in self.movers:
            mover.model_step_is_done()
    
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
        
        if self._current_time_step == -1:
            self.setup_model_run()
        
        self._current_time_step += 1
        self.setup_time_step()
        self.move_elements()
        self.step_is_done()

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
        

        