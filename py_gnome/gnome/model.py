#!/usr/bin/env python
import os
from datetime import datetime, timedelta
import copy

import gnome
import gnome.utilities.cache

from gnome import GnomeId
from gnome.utilities.time_utils import round_time
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.environment import Environment, Wind
from gnome.movers import Mover
from gnome.spill_container import SpillContainerPair
from gnome.utilities import serializable

class Model(serializable.Serializable):
    """ 
    PyGNOME Model Class
    
    """
    _update = ['time_step',
               'start_time',
               'duration',
               'uncertain',
               'movers',
               'environment',
               'spills',
               'maps'
               ]
    _create = []
    _create.extend(_update)
    state = copy.deepcopy(serializable.Serializable.state)
    state.add(create=_create,
              update=_update)   # no need to copy parent's state in tis case
    
    #===========================================================================
    # @classmethod
    # def new_from_dict(cls, dict_):
    #    """create compound objects like map, output_map from dict, then pass it onto new_from_dict"""
    #    if 'map' in dict_:
    #        obj_ = dict_.pop('map')
    #        to_eval = "{0}.new_from_dict( obj_)".format( obj_.pop('obj_type'))
    #        map = eval(to_eval)
    #        dict_['map'] = map  # update dict with object
    #        
    #    return super(Model, cls).new_from_dict( dict_)
    #===========================================================================
    
    def __init__(self,
                 time_step=timedelta(minutes=15), 
                 start_time=round_time(datetime.now(), 3600), # default to now, rounded to the nearest hour
                 duration=timedelta(days=1),
                 map=gnome.map.GnomeMap(),
                 renderer=None,
                 uncertain=False,
                 cache_enabled=False,
                 **kwargs):
        """ 
        Initializes a model. 

        :param time_step=timedelta(minutes=15): model time step in seconds or as a timedelta object
        :param start_time=datetime.now(): start time of model, datetime object
        :param duration=timedelta(days=1): how long to run the model, a timedelta object
        :param map=gnome.map.GnomeMap(): the land-water map, default is a map with no land-water
        :param output_map=None: map for drawing output
        :param uncertain=False: flag for setting uncertainty
        :param cache_enabled=False: flag for setting whether the mocel should cache results to disk.

        All this does is call reset() which initializes eveything to defaults
        
        Optional keyword parameters (kwargs):
        :param id: Unique Id identifying the newly created mover (a UUID as a string). 
                   This is used when loading an object from a persisted model
        """
        # making sure basic stuff is in place before properties are set
        self.environment = OrderedCollection(dtype=Environment)  
        self.movers = OrderedCollection(dtype=Mover)
        self.spills = SpillContainerPair(uncertain)   # contains both certain/uncertain spills 
        self._cache = gnome.utilities.cache.ElementCache()
        self._cache.enabled = cache_enabled
        self.outputters = OrderedCollection(dtype=gnome.outputter.Outputter) # list of output objects

        self._start_time = start_time # default to now, rounded to the nearest hour
        self._duration = duration
        self._map = map
        if renderer is not None:
            self.outputters += renderer
        self.time_step = time_step # this calls rewind() !
        self._gnome_id = GnomeId(id=kwargs.pop('id',None))
        

    def reset(self, **kwargs):
        """
        Resets model to defaults -- Caution -- clears all movers, spills, etc.

        Takes same keyword arguments as __init__
        """
        self.__init__(**kwargs)

    def rewind(self):
        """
        Rewinds the model to the beginning (start_time)
        """
        ## fixme: do the movers need re-setting? -- or wait for prepare_for_model_run?

        self.current_time_step = -1 # start at -1
        self.model_time = self._start_time
        ## note: this may be redundant -- they will get reset in setup_model_run() anyway..
        self.spills.rewind()
        #clear the cache:
        self._cache.rewind()
        [outputter.rewind() for outputter in self.outputters]

    def write_from_cache(self, filetype='netcdf', time_step='all'):
        """
        write the already-cached data to an output files.
        """
        pass

    ### Assorted properties
    @property
    def uncertain(self):
        return self.spills.uncertain
    @uncertain.setter
    def uncertain(self, uncertain_value):
        """
        only if uncertainty switch is toggled, then restart model
        """
        if self.spills.uncertain != uncertain_value:
            self.spills.uncertain = uncertain_value # update uncertainty
            self.rewind()

    @property
    def cache_enabled(self):
        return self._cache.enabled
    @cache_enabled.setter
    def cache_enabled(self, enabled):
        self._cache.enabled = enabled
    
    @property
    def id(self):
        return self._gnome_id.id

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

    def setup_model_run(self):
        """
        Sets up each mover for the model run

        """
        [mover.prepare_for_model_run() for mover in self.movers]
        [outputter.prepare_for_model_run(self._cache) for outputter in self.outputters]
        
        self.spills.rewind()

    def setup_time_step(self):
        """
        sets up everything for the current time_step:
        
        right now only prepares the movers -- maybe more later?.
        """
        
        # initialize movers differently if model uncertainty is on
        for mover in self.movers:
            for sc in self.spills.items():
                mover.prepare_for_model_step(sc, self.time_step, self.model_time)
        [outputter.prepare_for_model_step() for outputter in self.outputters]

                                
    def move_elements(self):
        """

        Moves elements:
         - loops through all the movers. and moves the elements
         - sets new_position array for each spill
         - calls the beaching code to beach the elements that need beaching.
         - sets the new position
        """
        ## if there are no spills, there is nothing to do:
        if len(self.spills) > 0:        # can this check be removed?
            for sc in self.spills.items():
                if sc.num_elements > 0: # can this check be removed?
                    # possibly refloat elements
                    self.map.refloat_elements(sc,self.time_step)
                    
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
        for outputter in self.outputters:
            outputter.model_step_is_done()

    def write_output(self):
        for outputter in self.outputters:
            outputter.write_output(self.current_time_step)

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
        ## release_elements after the time step increment so that they will be there
        ## but not yet moved, at the beginning of the release time.
        for sc in self.spills.items():
            sc.release_elements(self.model_time, self.time_step)
        # cache the results
        self._cache.save_timestep(self.current_time_step, self.spills)
        self.write_output()
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


    def next_image(self):
        """
        Compute the next model step, render an image, and return info about the
        step rendered

        :param images_dir: directory to write the image too.
        """
        # is there a renderer in the outputters list?
        for renderer in self.outputters:
            if isinstance(renderer, gnome.renderer.Renderer):
                break
        else:
            raise ValueError("There must be a renderer in the outputters list to call next_image")
        # run the next step:
        if not self.step():
            raise StopIteration
        filename = renderer.last_filename
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

    def movers_to_dict(self):
        """
        call to_dict method of OrderedCollection object
        """
        return self.movers.to_dict()
    
    def environment_to_dict(self):
        """
        call to_dict method of OrderedCollection object
        """
        return self.environment.to_dict()

    def spills_to_dict(self):
        return self.spills.to_dict()

    def maps_to_dict(self):
        """
        create a dict_ that contains:
        'map': (type, object.id)
        'ouput_map': (type, object.id)
        """
        dict_ = {'map': ("{0}.{1}".format(self.map.__module__, self.map.__class__.__name__), self.map.id)}
        if self.output_map is not None:
            dict_.update({'output_map': ("{0}.{1}".format(self.output_map.__module__, self.output_map.__class__.__name__), self.output_map.id)})
            
        return dict_
    
    def __eq__(self, other):
        """
        override serializable.Serializable.__eq__() method
        
        In addition to checking properties, also check the equality of
        objects in each collection
        """
        check = super(Model,self).__eq__(other)
        
        #=======================================================================
        # if check:
        #    """check ordered collections are equal. Currently not implemented"""
        #    if not self.movers == other.movers:
        #        return False
        #    
        #    if not self.environment == other.environment:
        #        return False
        #    
        #    if not self.spills == other.spills:
        #        return False
        #=======================================================================
        
        return check
        
