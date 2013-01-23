#!/usr/bin/env python
import os
from datetime import datetime, timedelta

import gnome

from gnome.utilities.time_utils import round_time
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.gnomeobject import GnomeObject

class Model(GnomeObject):
    """ 
    PyGNOME Model Class
    
    """
    _uncertain = False
    output_map = None
    _map = None
    
    def __init__(self):
        """ 
        Initializes model attributes. 

        All this does is call reset() which initializes eveything to defaults
        """
        self.reset() # initializes everything to defaults/nothing

    def reset(self):
        """
        Resets model to defaults -- Caution -- clears all movers, spills, etc.
        
        """
        self._uncertain = False # sets whether uncertainty is on or not.
        self.output_map = None
        self._map = None
        self.winds = OrderedCollection(dtype=gnome.weather.Wind)  #list of wind objects
        self.movers = OrderedCollection(dtype=gnome.movers.Mover)
        self._spill_container = gnome.spill_container.SpillContainer()
        self._uncertain_spill_container = None
        
        self._start_time = round_time(datetime.now(), 3600) # default to now, rounded to the nearest hour
        self._duration = timedelta(days=2) # fixme: should round to multiple of time_step?
        self.time_step = timedelta(minutes=15).total_seconds()

        self.rewind()

    def rewind(self):
        """
        Resets the model to the beginning (start_time)
        """
        ## fixme: do the movers need re-setting? -- or wait for prepare_for_model_run?

        self.current_time_step = -1 # start at -1
        self.model_time = self._start_time
        ## note: this may be redundant -- they will get reset in setup_model_run() anyway..
        self._spill_container.reset()
        try:
            self._uncertain_spill_container.reset()
        except AttributeError:
            pass # there must not be one...


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
    def spills(self):
        """
        Return a list of the spills added to this model, in order of insertion.

        :return: a list of spills
        """
        return self._spill_container.spills
        

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
    def map(self, map_in):
        self._map = map_in
        self.rewind()

    @property
    def num_time_steps(self):
        return self._num_time_steps

    def get_spill(self, spill_id):
        """
        Return a :class:`gnome.spill.Spill` in the ``self._spills`` dict with
        the key ``spill_id`` if one exists.
        """
        return self._spill_container.spills[spill_id]

    def add_spill(self, spill):
        """
        add a spill to the model

        :param spill: an instance of one of the gnome.spill classes

        """
        #fixme: where should we check if a spill is in a valid location on the map?
        self._spill_container.spills += spill
        ## fixme -- this may not be strictly required, but it's safer.
        self.rewind() 

    def remove_spill(self, spill_id):
        """
        remove the passed-in spill from the spill list
        """
        ##fixme: what if we want to remove by reference, rather than id?
        del self._spill_container.spills[spill_id]

    def setup_model_run(self):
        """
        Sets up each mover for the model run

        Currently, only movers need to initialize at the beginning of the run
        """
        for mover in self.movers:
            mover.prepare_for_model_run()
        self._spill_container.reset()
        if self._uncertain:
            self._uncertain_spill_container = self._spill_container.uncertain_copy()
        else:
            self._uncertain_spill_container = None

    def setup_time_step(self):
        """
        sets up everything for the current time_step:
        
        right now only prepares the movers -- maybe more later?.
        """
        
        # initialize movers differently if model uncertainty is on
        for mover in self.movers:
            mover.prepare_for_model_step(self._spill_container, self.time_step, self.model_time)
            if self.is_uncertain:
                mover.prepare_for_model_step(self._uncertain_spill_container, self.time_step, self.model_time)
                                
    def move_elements(self):
        """

        Moves elements:
         - loops through all the movers. and moves the elements
         - sets new_position array for each spill
         - calls the beaching code to beach the elements that need beaching.
         - sets the new position
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
        Loop through movers and call model_step_is_done
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
        Render the map image, according to current parameters

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
        Compute the next model step, render an image, and return info about the
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
                self.next_image(output_dir)
            except StopIteration:
                print "Done with the model run"
                break

