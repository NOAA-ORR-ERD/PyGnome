#!/usr/bin/env python

from datetime import datetime, timedelta
import copy

import numpy
np = numpy

from gnome.environment import Environment

import gnome.utilities.cache
from gnome.utilities.time_utils import round_time
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.utilities.serializable import Serializable

from gnome.spill_container import SpillContainerPair

from gnome.movers import Mover, WindMover, CatsMover
from gnome.weatherers import Weatherer


class Model(Serializable):
    'PyGNOME Model Class'
    _update = ['time_step',
               'weathering_substeps',
               'start_time',
               'duration',
               'uncertain',
               'movers',
               'weatherers',
               'environment',
               'spills',
               'map',
               'outputters',
               'cache_enabled']
    _create = []
    _create.extend(_update)
    state = copy.deepcopy(Serializable.state)

    # no need to copy parent's state in this case
    state.add(create=_create, update=_update)

    @classmethod
    def new_from_dict(cls, dict_):
        'Restore model from previously persisted state'
        l_env = dict_.pop('environment')
        l_out = dict_.pop('outputters')
        l_movers = dict_.pop('movers')
        l_weatherers = dict_.pop('weatherers')
        c_spills = dict_.pop('certain_spills')

        if 'uncertain_spills' in dict_:
            u_spills = dict_.pop('uncertain_spills')
            l_spills = zip(c_spills, u_spills)
        else:
            l_spills = c_spills

        model = object.__new__(cls)
        model.__restore__(**dict_)
        [model.environment.add(obj) for obj in l_env]
        [model.outputters.add(obj) for obj in l_out]
        [model.spills.add(obj) for obj in l_spills]
        [model.movers.add(obj) for obj in l_movers]
        [model.weatherers.add(obj) for obj in l_weatherers]

        # register callback with OrderedCollection
        model.movers.register_callback(model._callback_add_mover,
                                       ('add', 'replace'))

        model.weatherers.register_callback(model._callback_add_weatherer,
                                           ('add', 'replace'))

        return model

    def __init__(self,
                 time_step=timedelta(minutes=15),
                 start_time=round_time(datetime.now(), 3600),
                 duration=timedelta(days=1),
                 weathering_substeps=1,
                 map=gnome.map.GnomeMap(),
                 uncertain=False,
                 cache_enabled=False,
                 id=None):
        '''
        Initializes a model. All arguments have a default.

        :param time_step=timedelta(minutes=15): model time step in seconds
                                                or as a timedelta object
        :param start_time=datetime.now(): start time of model, datetime
                                          object. Rounded to the nearest hour.
        :param duration=timedelta(days=1): How long to run the model,
                                           a timedelta object.
        :param int weathering_substeps=1: How many weathering substeps to
                                          run inside a single model time step.
        :param map=gnome.map.GnomeMap(): The land-water map.
        :param uncertain=False: Flag for setting uncertainty.
        :param cache_enabled=False: Flag for setting whether the model should
                                    cache results to disk.
        :param id: Unique Id identifying the newly created mover (a UUID as a
                   string).  This is used when loading an object from a
                   persisted model
        '''
        self.__restore__(time_step, start_time, duration,
                         weathering_substeps,
                         map, uncertain, cache_enabled, id)

        # register callback with OrderedCollection
        self.movers.register_callback(self._callback_add_mover,
                                      ('add', 'replace'))

        self.weatherers.register_callback(self._callback_add_weatherer,
                                          ('add', 'replace'))

    def __restore__(self, time_step, start_time, duration,
                    weathering_substeps,
                    map, uncertain, cache_enabled, id):
        '''
        Take out initialization that does not register the callback here.
        This is because new_from_dict will use this to restore the model state
        when doing a midrun persistence.
        '''

        # making sure basic stuff is in place before properties are set
        self.environment = OrderedCollection(dtype=Environment)
        self.movers = OrderedCollection(dtype=Mover)
        self.weatherers = OrderedCollection(dtype=Weatherer)

        # contains both certain/uncertain spills
        self.spills = SpillContainerPair(uncertain)

        self._cache = gnome.utilities.cache.ElementCache()
        self._cache.enabled = cache_enabled

        # list of output objects
        self.outputters = OrderedCollection(dtype=gnome.outputter.Outputter)

        # default to now, rounded to the nearest hour
        self._start_time = start_time
        self._duration = duration
        self.weathering_substeps = weathering_substeps
        self._map = map
        self.time_step = time_step  # this calls rewind() !

        self._gnome_id = gnome.GnomeId(id)

    def reset(self, **kwargs):
        '''
        Resets model to defaults -- Caution -- clears all movers, spills, etc.
        Takes same keyword arguments as __init__
        '''
        self.__init__(**kwargs)

    def rewind(self):
        '''
        Rewinds the model to the beginning (start_time)
        '''

        # fixme: do the movers need re-setting? -- or wait for
        #        prepare_for_model_run?

        self.current_time_step = -1
        self.model_time = self._start_time

        # note: This may be redundant.  They will get reset in
        #       setup_model_run() anyway..

        self.spills.rewind()

        # set rand before each call so windages are set correctly
        gnome.utilities.rand.seed(1)

        # clear the cache:
        self._cache.rewind()

        for outputter in self.outputters:
            outputter.rewind()

#    def write_from_cache(self, filetype='netcdf', time_step='all'):
#        """
#        write the already-cached data to an output files.
#        """

    @property
    def uncertain(self):
        return self.spills.uncertain

    @uncertain.setter
    def uncertain(self, uncertain_value):
        '''
        only if uncertainty switch is toggled, then restart model
        '''
        if self.spills.uncertain != uncertain_value:
            self.spills.uncertain = uncertain_value  # update uncertainty
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
        '''
        Sets the time step, and rewinds the model

        :param time_step: The timestep can be a timedelta object
                          or integer seconds.
        '''
        try:
            self._time_step = time_step.total_seconds()
        except AttributeError:
            self._time_step = int(time_step)

        # there is a zeroth time step
        self._num_time_steps = int(self._duration.total_seconds()
                                   // self._time_step) + 1
        self.rewind()

    @property
    def current_time_step(self):
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, step):
        self.model_time = self._start_time + timedelta(seconds=step
                * self.time_step)
        self._current_time_step = step

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        if duration < self._duration:
            # only need to rewind if shorter than it was...
            # fixme: actually, only need to rewind if current model time
            # is beyond new time...
            self.rewind()
        self._duration = duration

        # there is a zeroth time step
        self._num_time_steps = int(self._duration.total_seconds()
                                   // self.time_step) + 1

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
        '''
        Sets up each mover for the model run
        '''
        self.spills.rewind()  # why is rewind for spills here?

        array_types = {}

        for mover in self.movers:
            mover.prepare_for_model_run()
            array_types.update(mover.array_types)

        for w in self.weatherers:
            w.prepare_for_model_run()
            array_types.update(w.array_types)

        for sc in self.spills.items():
            sc.prepare_for_model_run(array_types)

        # outputters need array_types, so this needs to come after those
        # have been updated.
        for outputter in self.outputters:
            outputter.prepare_for_model_run(model_start_time=self.start_time,
                                            cache=self._cache,
                                            uncertain=self.uncertain,
                                            spills=self.spills)

    def setup_time_step(self):
        '''
        sets up everything for the current time_step:
        '''
        # initialize movers differently if model uncertainty is on
        for m in self.movers:
            for sc in self.spills.items():
                m.prepare_for_model_step(sc, self.time_step, self.model_time)

        for w in self.weatherers:
            for sc in self.spills.items():
                # maybe we will setup a super-sampling step here???
                w.prepare_for_model_step(sc, self.time_step, self.model_time)

        for outputter in self.outputters:
            outputter.prepare_for_model_step(self.time_step, self.model_time)

    def move_elements(self):
        '''
        Moves elements:
         - loops through all the movers. and moves the elements
         - sets new_position array for each spill
         - calls the beaching code to beach the elements that need beaching.
         - sets the new position
        '''
        for sc in self.spills.items():
            if sc.num_released > 0:  # can this check be removed?

                # possibly refloat elements
                self.map.refloat_elements(sc, self.time_step)

                # reset next_positions
                (sc['next_positions'])[:] = sc['positions']

                # loop through the movers
                for m in self.movers:
                    delta = m.get_move(sc, self.time_step, self.model_time)
                    sc['next_positions'] += delta

                self.map.beach_elements(sc)

                # the final move to the new positions
                (sc['positions'])[:] = sc['next_positions']

    def weather_elements(self):
        '''
        Weathers elements:
        - loops through all the weatherers, passing in the spill_container
          and the time range
        - a weatherer modifies the data arrays in the spill container, so a
          particular time range should not be run multiple times.  It is
          expected that we are processing a sequence of contiguous time ranges.
        - Note: If there are multiple sequential weathering processes, some
                inaccuracy could occur.  A proposed solution is to
                'super-sample' the model time step so that it will be replaced
                with many smaller time steps.  We'll have to see if this pans
                out in practice.
        '''
        for sc in self.spills.items():
            for w in self.weatherers:
                for model_time, time_step in self._split_into_substeps():
                    w.weather_elements(sc, time_step, model_time)

    def _split_into_substeps(self):
        '''
        :return: sequence of (datetime, timestep)
         (Note: we divide evenly on second boundaries.
                   Thus, there will likely be a remainder
                   that needs to be included.  We include
                   this remainder, which results in
                   1 more sub-step than we requested.)
        '''
        time_step = int(self._time_step)
        sub_step = time_step / self.weathering_substeps

        indexes = [idx for idx in range(0, time_step + 1, sub_step)]
        res = [(idx, next_idx - idx)
               for idx, next_idx in zip(indexes, indexes[1:])]

        if sum(res[-1]) < time_step:
            # collect the remaining slice
            res.append((sum(res[-1]), time_step % sub_step))

        res = [(self.model_time + timedelta(seconds=idx), delta)
               for idx, delta in res]

        return res

    def step_is_done(self):
        '''
        Loop through movers and call model_step_is_done
        '''
        for mover in self.movers:
            for sc in self.spills.items():
                mover.model_step_is_done(sc)

        for w in self.weatherers:
            w.model_step_is_done()

        for sc in self.spills.items():
            'removes elements with oil_status.to_be_removed'
            sc.model_step_is_done()

            # age remaining particles
            sc['age'][:] = sc['age'][:] + self.time_step

        for outputter in self.outputters:
            outputter.model_step_is_done()

    def write_output(self):
        output_info = {'step_num': self.current_time_step}

        for outputter in self.outputters:
            if self.current_time_step == self.num_time_steps - 1:
                output = outputter.write_output(self.current_time_step, True)
            else:
                output = outputter.write_output(self.current_time_step)

            if output is not None:
                output_info.update(output)

        return output_info

    def step(self):
        '''
        Steps the model forward (or backward) in time. Needs testing for
        hind casting.
        '''
        for sc in self.spills.items():
            # Set the current time stamp only after current_time_step is
            # incremented and before the output is written. Set it to None here
            # just so we're not carrying around the old time_stamp
            sc.current_time_stamp = None

        # it gets incremented after this check
        if self.current_time_step >= self._num_time_steps - 1:
            raise StopIteration

        if self.current_time_step == -1:
            # that's all we need to do for the zeroth time step
            self.setup_model_run()
        else:
            self.setup_time_step()
            self.move_elements()
            self.weather_elements()
            self.step_is_done()

        self.current_time_step += 1

        # this is where the new step begins!
        # the elements released are during the time period:
        #    self.model_time + self.time_step
        # The else part of the loop computes values for data_arrays that
        # correspond with time_stamp:
        #    self.model_time + self.time_step
        # This is the current_time_stamp attribute of the SpillContainer
        #     [sc.current_time_stamp for sc in self.spills.items()]
        for sc in self.spills.items():
            sc.current_time_stamp = self.model_time

            # release particles for next step - these particles will be aged
            # in the next step
            sc.release_elements(self.time_step, self.model_time)

        # cache the results - current_time_step is incremented but the
        # current_time_stamp in spill_containers (self.spills) is not updated
        # till we go through the prepare_for_model_step
        self._cache.save_timestep(self.current_time_step, self.spills)
        output_info = self.write_output()
        return output_info

    def __iter__(self):
        '''
        Rewinds the model and returns itself so it can be iterated over.
        '''
        self.rewind()

        return self

    def next(self):
        '''
        (This method satisfies Python's iterator and generator protocols)

        :return: the step number
        '''
        return self.step()

    def full_run(self, rewind=True, log=False):
        '''
        Do a full run of the model.

        :param rewind=True: whether to rewind the model first
                            -- if set to false, model will be run from the
                               current step to the end
        :returns: list of outputter info dicts
        '''
        if rewind:
            self.rewind()

        # run the model
        output_data = []
        while True:
            try:
                results = self.step()
                if log:
                    print results
                output_data.append(results)
            except StopIteration:
                print 'Done with the model run'
                break

        return output_data

    def movers_to_dict(self):
        '''
        Call to_dict method of OrderedCollection object
        '''
        return self.movers.to_dict()

    def weatherers_to_dict(self):
        '''
        Call to_dict method of OrderedCollection object
        '''
        return self.weatherers.to_dict()

    def environment_to_dict(self):
        '''
        Call to_dict method of OrderedCollection object
        '''
        return self.environment.to_dict()

    def spills_to_dict(self):
        return self.spills.to_dict()

    def outputters_to_dict(self):
        '''
        Call to_dict method of OrderedCollection object
        '''
        return self.outputters.to_dict()

    def map_to_dict(self):
        '''
        Create a tuple that contains: (type, object.id)
        '''
        return ('{0}.{1}'.format(self.map.__module__,
                                 self.map.__class__.__name__),
                self.map.id)

    def _callback_add_mover(self, obj_added):
        'Callback after mover has been added'
        if isinstance(obj_added, WindMover):
            if obj_added.wind.id not in self.environment:
                self.environment += obj_added.wind

        if isinstance(obj_added, CatsMover):
            if (obj_added.tide is not None and
                obj_added.tide.id not in self.environment):
                self.environment += obj_added.tide

        self.rewind()  # rewind model if a new mover is added

    def _callback_add_weatherer(self, obj_added):
        'Callback after weatherer has been added'
        if isinstance(obj_added, Weatherer):
            # not sure what kind of dependencies we have just yet.
            pass

        self.rewind()  # rewind model if a new weatherer is added

    def __eq__(self, other):
        check = super(Model, self).__eq__(other)
        if check:
            # also check the data in spill_container object
            if type(self.spills) != type(other.spills):
                return False

            if self.spills != other.spills:
                return False

        return check

    def __ne__(self, other):
        'Compare inequality (!=) of two objects'
        if self == other:
            return False
        else:
            return True
