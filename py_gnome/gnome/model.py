#!/usr/bin/env python
import os
import shutil
from datetime import datetime, timedelta
import copy
import inspect

import numpy
np = numpy
from colander import (SchemaNode,
                      Float, Int, Bool, drop)

from gnome.environment import Environment, Water

import gnome.utilities.cache
from gnome.utilities.time_utils import round_time
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.utilities.serializable import Serializable, Field

from gnome.spill_container import SpillContainerPair
from gnome.movers import Mover
from gnome.weatherers import weatherer_sort, Weatherer, WeatheringData
from gnome.outputters import Outputter, NetCDFOutput, WeatheringOutput
from gnome.persist import (extend_colander,
                           validators,
                           References,
                           load)
from gnome.persist.base_schema import (ObjType,
                                       OrderedCollectionItemsList)


class ModelSchema(ObjType):
    'Colander schema for Model object'
    time_step = SchemaNode(Float(), missing=drop)
    weathering_substeps = SchemaNode(Int(), missing=drop)
    start_time = SchemaNode(extend_colander.LocalDateTime(),
                            validator=validators.convertible_to_seconds,
                            missing=drop)
    duration = SchemaNode(extend_colander.TimeDelta(), missing=drop)
    uncertain = SchemaNode(Bool(), missing=drop)
    cache_enabled = SchemaNode(Bool(), missing=drop)
    num_time_steps = SchemaNode(Int(), missing=drop)

    def __init__(self, json_='webapi', *args, **kwargs):
        '''
        Add default schema for orderedcollections if oc_schema=True
        The default schema works for 'save' files since it only keeps the same
        info {'obj_type' and 'id'} for all elements in OC. For 'webapi', we
        cannot do this and must deserialize each element of the collection
        using the deserialize method of each object; so these are not added to
        schema for 'webapi'
        '''
        if json_ == 'save':
            self.add(OrderedCollectionItemsList(missing=drop, name='spills'))
            self.add(OrderedCollectionItemsList(missing=drop,
                     name='uncertain_spills'))
            self.add(OrderedCollectionItemsList(missing=drop, name='movers'))
            self.add(OrderedCollectionItemsList(missing=drop,
                     name='weatherers'))
            self.add(OrderedCollectionItemsList(missing=drop,
                     name='environment'))
            self.add(OrderedCollectionItemsList(missing=drop,
                     name='outputters'))
        super(ModelSchema, self).__init__(*args, **kwargs)


class Model(Serializable):
    '''
    PyGnome Model Class
    '''
    _update = ['time_step',
               'weathering_substeps',
               'start_time',
               'duration',
               'uncertain',
               'cache_enabled',
               'weathering_substeps',
               'map',
               'movers',
               'weatherers',
               'environment',
               'outputters'
               ]
    _create = []
    _create.extend(_update)
    _state = copy.deepcopy(Serializable._state)
    _schema = ModelSchema

    # no need to copy parent's _state in this case
    _state.add(save=_create, update=_update)

    # override __eq__ since 'spills' and 'uncertain_spills' need to be checked
    # They both have _to_dict() methods to return underlying ordered
    # collections and that would not be the correct way to check equality
    _state += [Field('spills', save=True, update=True, test_for_eq=False),
               Field('uncertain_spills', save=True, test_for_eq=False),
               Field('num_time_steps', read=True),
               Field('water', update=True, save=True, save_reference=True)]

    # list of OrderedCollections
    _oc_list = ['movers', 'weatherers', 'environment', 'outputters']

    @classmethod
    def new_from_dict(cls, dict_):
        'Restore model from previously persisted _state'
        json_ = dict_.pop('json_')
        l_env = dict_.pop('environment', [])
        l_out = dict_.pop('outputters', [])
        g_objects = dict_.pop('movers', [])
        l_weatherers = dict_.pop('weatherers', [])
        c_spills = dict_.pop('spills', [])

        if 'uncertain_spills' in dict_:
            u_spills = dict_.pop('uncertain_spills')
            l_spills = zip(c_spills, u_spills)
        else:
            l_spills = c_spills

        # define defaults for properties that a location file may not contain
        kwargs = inspect.getargspec(cls.__init__)
        default_restore = dict(zip(kwargs[0][1:], kwargs[3]))

        if json_ == 'webapi':
            # default is to disable cache
            default_restore['cache_enabled'] = False

        for key in default_restore:
            default_restore[key] = dict_.pop(key, default_restore[key])

        model = object.__new__(cls)
        model.__restore__(**default_restore)

        # if there are other values in dict_, setattr
        if json_ == 'webapi':
            model.update_from_dict(dict_)
        else:
            cls._restore_attr_from_save(model, dict_)

        [model.environment.add(obj) for obj in l_env]
        [model.outputters.add(obj) for obj in l_out]
        [model.spills.add(obj) for obj in l_spills]
        [model.movers.add(obj) for obj in g_objects]
        [model.weatherers.add(obj) for obj in l_weatherers]

        # register callback with OrderedCollection after objects are added
        model.movers.register_callback(model._callback_add_mover,
                                       ('add', 'replace'))

        model.weatherers.register_callback(model._callback_add_weatherer_env,
                                           ('add', 'replace'))

        model.environment.register_callback(model._callback_add_weatherer_env,
                                            ('add', 'replace'))

        # todo: set Water / intrinsic properties
        if model.water is not None and len(model.weatherers) > 0:
            model._weathering_data = WeatheringData(model.water)
        else:
            model._weathering_data = None

        # restore the spill data outside this method - let's not try to find
        # the saveloc here
        msg = ("{0._pid} 'new_from_dict' created new model: "
               "{0.name}").format(model)
        model.logger.info(msg)
        return model

    def __init__(self,
                 time_step=None,
                 start_time=round_time(datetime.now(), 3600),
                 duration=timedelta(days=1),
                 weathering_substeps=1,
                 map=None,
                 uncertain=False,
                 cache_enabled=False,
                 name=None):
        '''

        Initializes a model.
        All arguments have a default.

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
        '''
        self.__restore__(time_step, start_time, duration,
                         weathering_substeps,
                         uncertain, cache_enabled, map, name)

        # set in setup_model_run if weatherers are added
        self._weathering_data = None

        # register callback with OrderedCollection
        self.movers.register_callback(self._callback_add_mover,
                                      ('add', 'replace'))

        self.weatherers.register_callback(self._callback_add_weatherer_env,
                                          ('add', 'replace'))

        self.environment.register_callback(self._callback_add_weatherer_env,
                                           ('add', 'replace'))

    def __restore__(self, time_step, start_time, duration,
                    weathering_substeps, uncertain, cache_enabled, map, name,
                    water=None):
        '''
        Take out initialization that does not register the callback here.
        This is because new_from_dict will use this to restore the model _state
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
        self.outputters = OrderedCollection(dtype=Outputter)

        # default to now, rounded to the nearest hour
        self._start_time = start_time
        self._duration = duration
        self.weathering_substeps = weathering_substeps
        if not map:
            map = gnome.map.GnomeMap()

        if name:
            self.name = name

        self._map = map
        self.water = water

        # reset _current_time_step
        self._current_time_step = -1
        self._time_step = None
        if time_step is not None:
            self.time_step = time_step  # this calls rewind() !
        self._reset_num_time_steps()

    def reset(self, **kwargs):
        '''
        Resets model to defaults -- Caution -- clears all movers, spills, etc.
        Takes same keyword arguments as :meth:`__init__()`
        '''
        self.__init__(**kwargs)

    def rewind(self):
        '''
        Rewinds the model to the beginning (start_time)
        '''
        self._current_time_step = -1
        self.model_time = self._start_time

        # fixme: do the movers need re-setting? -- or wait for
        #        prepare_for_model_run?

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
        '''
        Uncertainty attribute of the model. If flag is toggled, rewind model
        '''
        return self.spills.uncertain

    @uncertain.setter
    def uncertain(self, uncertain_value):
        '''
        Uncertainty attribute of the model
        '''
        if self.spills.uncertain != uncertain_value:
            self.spills.uncertain = uncertain_value  # update uncertainty
            self.rewind()

    @property
    def cache_enabled(self):
        '''
        If True, then generated data is cached
        '''
        return self._cache.enabled

    @cache_enabled.setter
    def cache_enabled(self, enabled):
        self._cache.enabled = enabled

    @property
    def has_weathering(self):
        return (any([w.on for w in self.weatherers]) and
                len([o for o in self.outputters
                     if isinstance(o, WeatheringOutput)]) > 0)

    @property
    def start_time(self):
        '''
        Start time of the simulation
        '''
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time
        self.rewind()

    @property
    def time_step(self):
        '''
        time step over which the dynamics is computed
        '''
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

        self._reset_num_time_steps()
        self.rewind()

    @property
    def current_time_step(self):
        '''
        Current timestep of the simulation
        '''
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, step):
        self.model_time = self._start_time + timedelta(seconds=step *
                                                       self.time_step)
        self._current_time_step = step

    @property
    def duration(self):
        '''
        total duration of the model run
        '''
        return self._duration

    @duration.setter
    def duration(self, duration):
        if duration < self._duration:
            # only need to rewind if shorter than it was...
            # fixme: actually, only need to rewind if current model time
            # is beyond new time...
            self.rewind()
        self._duration = duration
        self._reset_num_time_steps()

    @property
    def map(self):
        '''
        land water map used for simulation
        '''
        return self._map

    @map.setter
    def map(self, map_in):
        self._map = map_in
        self.rewind()

    @property
    def num_time_steps(self):
        '''
        Read only attribute
        computed number of timesteps based on py:attribute:`duration` and
        py:attribute:`time_step`
        '''
        return self._num_time_steps

    def _reset_num_time_steps(self):
        '''
        reset number of time steps if duration, or time_step change
        '''
        # We do not count any remainder time.
        if self.duration is not None and self.time_step is not None:
            initial_0th_step = 1
            self._num_time_steps = (initial_0th_step +
                                    int(self.duration.total_seconds()
                                        // self.time_step))
        else:
            self._num_time_steps = None

    def contains_object(self, obj_id):
        if self.map.id == obj_id:
            return True

        if self.water is not None:
            if self.water.id == obj_id:
                return True

        for collection in (self.environment,
                           self.spills,
                           self.movers,
                           self.weatherers,
                           self.outputters):
            for o in collection:
                if obj_id == o.id:
                    return True

                if (hasattr(o, 'contains_object') and
                        o.contains_object(obj_id)):
                    return True

        return False

    def _order_weatherers(self):
        'use weatherer_sort to sort the weatherers'
        s_weatherers = sorted(self.weatherers, key=weatherer_sort)
        if self.weatherers.values() != s_weatherers:
            self.weatherers.clear()
            self.weatherers += s_weatherers

    def setup_model_run(self):
        '''
        Sets up each mover for the model run
        '''

        self.spills.rewind()  # why is rewind for spills here?

        # use a set since we only want to add unique 'names' for data_arrays
        # that will be added
        array_types = set()

        # remake orderedcollections defined by model
        for oc in [self.movers, self.weatherers,
                   self.outputters, self.environment]:
            oc.remake()

        # order weatherers collection
        self._order_weatherers()
        transport = False
        for mover in self.movers:
            if mover.on:
                mover.prepare_for_model_run()
                transport = True
                array_types.update(mover.array_types)

        weathering = False
        for w in self.weatherers:
            for sc in self.spills.items():
                # weatherers will initialize 'weathering_data' key/values
                # to 0.0
                if w.on:
                    w.prepare_for_model_run(sc)
                    weathering = True
                    array_types.update(w.array_types)

        if weathering:
            if self.water is None:
                self.water = Water()

            if self._weathering_data is None:
                self._weathering_data = WeatheringData(self.water)

            # this adds 'density' array. It also adds data_arrays used to
            # compute area if Evaporation is included since it requires 'area'
            array_types.update(self._weathering_data.array_types)
        else:
            # reset to None if no weatherers found
            self._weathering_data = None

        for environment in self.environment:
            environment.prepare_for_model_run(self.start_time)

        if self.time_step is None:
            # for now hard-code this; however, it should depend on weathering
            # note: do not set time_step attribute because we don't want to
            # rewind because that will reset spill_container data
            if transport:
                self._time_step = 900
            elif weathering and not transport:
                # todo: 1 hour
                self._time_step = 3600
            else:
                # simple case with no weatherers or movers
                self._time_step = 900
            self._reset_num_time_steps()

        for sc in self.spills.items():
            sc.prepare_for_model_run(array_types)
            if self._weathering_data:
                # do this only if we have user has added spills!
                self._weathering_data.initialize(sc)

        # outputters need array_types, so this needs to come after those
        # have been updated.
        for outputter in self.outputters:
            outputter.prepare_for_model_run(model_start_time=self.start_time,
                                            cache=self._cache,
                                            uncertain=self.uncertain,
                                            spills=self.spills)
        self.logger.debug("{0._pid} setup_model_run complete for: "
                          "{0.name}".format(self))

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

        for environment in self.environment:
            environment.prepare_for_model_step(self.model_time)

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
        if len(self.weatherers) == 0:
            # if no weatherers then mass_components array may not be defined
            return

        for sc in self.spills.items():
            if self._weathering_data is not None:
                self._weathering_data.update_fate_status(sc)

            sc.reset_fate_dataview()

            for w in self.weatherers:
                for model_time, time_step in self._split_into_substeps():
                    # change 'mass_components' in weatherer
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
        Loop through movers and weatherers and call model_step_is_done

        Remove elements that marked for removal

        Output data
        '''
        for mover in self.movers:
            for sc in self.spills.items():
                mover.model_step_is_done(sc)

        for w in self.weatherers:
            w.model_step_is_done()

        for sc in self.spills.items():
            '''
            removes elements with oil_status.to_be_removed
            '''
            sc.model_step_is_done()

            # age remaining particles
            sc['age'][:] = sc['age'][:] + self.time_step

        for outputter in self.outputters:
            outputter.model_step_is_done()

    def write_output(self):
        output_info = {}

        for outputter in self.outputters:
            if self.current_time_step == self.num_time_steps - 1:
                output = outputter.write_output(self.current_time_step, True)
            else:
                output = outputter.write_output(self.current_time_step)

            if output is not None:
                output_info[outputter.__class__.__name__] = output

        if not output_info:
            return {'step_num': self.current_time_step}

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

        if self.current_time_step == -1:
            # that's all we need to do for the zeroth time step
            self.setup_model_run()

        elif self.current_time_step >= self._num_time_steps - 1:
            # _num_time_steps is set when self.time_step is set. If user does
            # not specify time_step, then setup_model_run() automatically
            # initializes it. Thus, do StopIteration check after
            # setup_model_run() is invoked
            raise StopIteration

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
            num_released = sc.release_elements(self.time_step, self.model_time)
            if self._weathering_data:
                self._weathering_data.update(num_released, sc)

            self.logger.debug("{1._pid} released {0} new elements for step:"
                              " {1.current_time_step} for {1.name}".
                              format(num_released, self))

        # cache the results - current_time_step is incremented but the
        # current_time_stamp in spill_containers (self.spills) is not updated
        # till we go through the prepare_for_model_step
        self._cache.save_timestep(self.current_time_step, self.spills)
        output_info = self.write_output()
        self.logger.debug("{0._pid} Completed step: {0.current_time_step} "
                          "for {0.name}".format(self))
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

    def full_run(self, rewind=True):
        '''
        Do a full run of the model.

        :param rewind=True: whether to rewind the model first -- if set to
            false, model will be run from the current step to the end
        :returns: list of outputter info dicts
        '''
        if rewind:
            self.rewind()

        # run the model
        output_data = []
        while True:
            try:
                results = self.step()
                self.logger.info(results)

                output_data.append(results)
            except StopIteration:
                self.logger.info('Run Complete: Stop Iteration')
                break

        return output_data

    def _add_to_environ_collec(self, obj_added):
        '''
        if an environment object exists in obj_added, but not in the Model's
        environment collection, then add it automatically.
        todo: maybe we don't want to do this - revisit this requirement
        '''
        if hasattr(obj_added, 'wind') and obj_added.wind is not None:
            if obj_added.wind.id not in self.environment:
                self.environment += obj_added.wind

        if hasattr(obj_added, 'tide') and obj_added.tide is not None:
            if obj_added.tide.id not in self.environment:
                self.environment += obj_added.tide

        if hasattr(obj_added, 'waves') and obj_added.waves is not None:
            if obj_added.waves.id not in self.environment:
                self.environment += obj_added.waves

    def _add_water(self, water):
        '''
        if Water object is found in obj_added as an attribute, then also set
        the Model's 'water' attribute to this object
        '''
        if self.water is None:
            self.water = water
        else:
            if self.water is not water:
                msg = ("{0._pid} water attribute is different from newly "
                       "added water named: {1.name}. "
                       "Model's Water object is used to update intrinsic "
                       "properties").format(self, water)
                self.logger.warning(msg)

    def _callback_add_mover(self, obj_added):
        'Callback after mover has been added'
        self._add_to_environ_collec(obj_added)
        self.rewind()  # rewind model if a new mover is added

    def _callback_add_weatherer_env(self, obj_added):
        '''
        Callback after weatherer/environment object has been added. 'waves'
        environment object contains 'wind' and 'water' so add those to
        environment collection and the 'water' attribute.
        If 'Water' object is added to environment collection, set self.water
        if it is not set.
        '''
        self._add_to_environ_collec(obj_added)
        if isinstance(obj_added, Water):
            self._add_water(obj_added)
        else:
            if hasattr(obj_added, 'water') and obj_added.water is not None:
                self._add_water(obj_added.water)
        self.rewind()  # rewind model if a new weatherer is added

    def __eq__(self, other):
        check = super(Model, self).__eq__(other)
        if check:
            # also check the data in ordered collections
            if type(self.spills) != type(other.spills):
                return False

            if self.spills != other.spills:
                return False

        return check

    def __ne__(self, other):
        return not self == other

    '''
    Following methods are for saving a Model instance or creating a new
    model instance from a saved location
    '''
    def spills_to_dict(self):
        '''
        return the spills ordered collection for serialization
        '''
        return self.spills.to_dict()['spills']

    def spills_update_from_dict(self, value):
        'invoke SpillContainerPair().update_from_dict'
        # containers don't need to be serializable; however, it was easiest to
        # put an update_from_dict method in the SpillContainerPair. Keep the
        # interface for this the same, so make it a dict
        return self.spills.update_from_dict({'spills': value})

    def uncertain_spills_to_dict(self):
        '''
        return the uncertain_spills ordered collection for serialization/save
        files
        '''
        if self.uncertain:
            dict_ = self.spills.to_dict()
            return dict_['uncertain_spills']

        return None

    def save(self, saveloc, references=None, name=None):
        # Note: Defining references=References() in the function definition
        # keeps the references object in memory between tests - it changes the
        # scope of Referneces() to be outside the Model() instance. We don't
        # want this
        references = (references, References())[references is None]
        self._make_saveloc(saveloc)
        self._empty_save_dir(saveloc)
        json_ = self.serialize('save')

        # map is the only nested structure - let's manually call
        # _move_data_file on it
        self.map._move_data_file(saveloc, json_['map'])

        for oc in self._oc_list:
            coll_ = getattr(self, oc)
            self._save_collection(saveloc, coll_, references, json_[oc])

        for sc in self.spills.items():
            if sc.uncertain:
                key = 'uncertain_spills'
            else:
                key = 'spills'

            self._save_collection(saveloc, sc.spills, references, json_[key])

        if self.current_time_step > -1:
            '''
            hard code the filename - can make this an attribute if user wants
            to change it - but not sure if that will ever be needed?
            '''
            self._save_spill_data(os.path.join(saveloc,
                                               'spills_data_arrays.nc'))

        # there should be no more references
        self._json_to_saveloc(json_, saveloc, references, name)
        return references

    def _save_collection(self, saveloc, coll_, refs, coll_json):
        """
        Reference objects inside OrderedCollections. Since the OC itself
        isn't a reference but the objects in the list are a reference, do
        something a little differently here

        :param OrderedCollection coll_: ordered collection to be saved
        """
        for count, obj in enumerate(coll_):
            json_ = obj.serialize('save')
            for field in obj._state:
                if field.save_reference:
                    '''
                    if attribute is stored as a reference to environment list,
                    then update the json_ here
                    '''
                    if getattr(obj, field.name) is not None:
                        ref_obj = getattr(obj, field.name)
                        try:
                            index = self.environment.index(ref_obj)
                            json_[field.name] = index
                        except ValueError:
                            '''
                            reference is not part of environment list, it must
                            be handled elsewhere
                            '''
                            pass
            obj_ref = refs.get_reference(obj)
            if obj_ref is None:
                # try following name - if 'fname' already exists in references,
                # then obj.save() assigns a different name to file
                fname = '{0.__class__.__name__}_{1}.json'.format(obj, count)
                obj.save(saveloc, refs, fname)
                coll_json[count]['id'] = refs.reference(obj)
            else:
                coll_json[count]['id'] = obj_ref

    def _save_spill_data(self, datafile):
        """ save the data arrays for current timestep to NetCDF """
        nc_out = NetCDFOutput(datafile, which_data='all', cache=self._cache)
        nc_out.prepare_for_model_run(model_start_time=self.start_time,
                                     uncertain=self.uncertain,
                                     spills=self.spills)
        nc_out.write_output(self.current_time_step)

    def _load_spill_data(self, spill_data):
        """
        load NetCDF file and add spill data back in - designed for savefiles
        """

        if not os.path.exists(spill_data):
            return

        if self.uncertain:
            saveloc, spill_data_fname = os.path.split(spill_data)
            spill_data_fname, ext = os.path.splitext(spill_data_fname)
            u_spill_data = os.path.join(saveloc,
                                        '{0}_uncertain{1}'
                                        .format(spill_data_fname, ext))

        array_types = set()

        for m in self.movers:
            array_types.update(m.array_types)

        for w in self.weatherers:
            array_types.update(w.array_types)

        if self._weathering_data:
            array_types.update(self._weathering_data.array_types)

        for sc in self.spills.items():
            sc.prepare_for_model_run(array_types)
            if sc.uncertain:
                (data, weather_data) = NetCDFOutput.read_data(u_spill_data,
                                                              time=None,
                                                              which_data='all')
            else:
                (data, weather_data) = NetCDFOutput.read_data(spill_data,
                                                              time=None,
                                                              which_data='all')

            sc.current_time_stamp = data.pop('current_time_stamp').item()
            sc._data_arrays = data
            sc.weathering_data = weather_data

    def _empty_save_dir(self, saveloc):
        '''
        Remove all files, directories under saveloc

        First clean out directory, then add new save files
        This should only be called by self.save()
        '''
        (dirpath, dirnames, filenames) = os.walk(saveloc).next()

        if dirnames:
            for dir_ in dirnames:
                shutil.rmtree(os.path.join(dirpath, dir_))

        if filenames:
            for file_ in filenames:
                os.remove(os.path.join(dirpath, file_))

    def serialize(self, json_='webapi'):
        '''
        Serialize Model object
        treat special-case attributes of Model.
        '''
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema(json_)
        o_json_ = schema.serialize(toserial)
        o_json_['map'] = self.map.serialize(json_)
        if self.water:
            o_json_['water'] = self.water.serialize(json_)

        if json_ == 'webapi':
            for attr in ('environment', 'outputters', 'weatherers', 'movers',
                         'spills'):
                o_json_[attr] = self.serialize_oc(attr, json_)

        return o_json_

    def serialize_oc(self, attr, json_='webapi'):
        '''
        Serialize Model attributes of type ordered collection
        '''
        json_out = []
        attr = getattr(self, attr)
        if isinstance(attr, (OrderedCollection, SpillContainerPair)):
            for item in attr:
                json_out.append(item.serialize(json_))
        return json_out

    @classmethod
    def deserialize(cls, json_):
        '''
        treat special-case attributes of Model.
        '''
        schema = cls._schema(json_['json_'])
        deserial = schema.deserialize(json_)
        for obj in ('map', 'water'):
            if obj in json_:
                d_item = cls._deserialize_nested_obj(json_[obj])
                deserial[obj] = d_item

        if json_['json_'] == 'webapi':
            for attr in ('environment', 'outputters', 'weatherers', 'movers',
                         'spills'):
                if attr in json_:
                    '''
                    even if list is empty, deserialize it because we still need
                    to sync up with client
                    '''
                    deserial[attr] = cls.deserialize_oc(json_[attr])

        return deserial

    @classmethod
    def deserialize_oc(cls, json_):
        '''
        check contents of orderered collections to figure out what schema to
        use.
        Basically, the json serialized ordered collection looks like a regular
        list.
        '''
        deserial = []
        for item in json_:
            d_item = cls._deserialize_nested_obj(item)
            deserial.append(d_item)

        return deserial

    @classmethod
    def _deserialize_nested_obj(cls, json_):
        if json_ is not None:
            fqn = json_['obj_type']
            name, scope = (list(reversed(fqn.rsplit('.', 1)))
                           if fqn.find('.') >= 0
                           else [fqn, ''])
            my_module = __import__(scope, globals(), locals(), [str(name)], -1)
            py_class = getattr(my_module, name)
            return py_class.deserialize(json_)
        else:
            return None

    @classmethod
    def load(cls, saveloc, json_data, references=None):
        '''
        Load a model from json format - the saveloc is location of save files
        for objects contained in the model
        '''
        references = (references, References())[references is None]
        ref_dict = cls._load_refs(saveloc, json_data, references)
        cls._update_datafile_path(saveloc, json_data)

        # deserialize after removing references
        _to_dict = cls.deserialize(json_data)

        if ref_dict:
            _to_dict.update(ref_dict)

        # load nested map object and add it - currently, 'load' is only used
        # for laoding save files/location files, so it assumes:
        # json_data['json_'] == 'save'
        if ('map' in json_data):
            map_obj = eval(json_data['map']['obj_type']).load(saveloc,
                                                              json_data['map'],
                                                              references)
            _to_dict['map'] = map_obj

        # load collections
        for oc in cls._oc_list:
            if oc in _to_dict:
                _to_dict[oc] = cls._load_collection(saveloc,
                                                    _to_dict[oc],
                                                    references)
        for spill in ['spills', 'uncertain_spills']:
            if spill in _to_dict:
                _to_dict[spill] = cls._load_collection(saveloc,
                                                       _to_dict[spill],
                                                       references)
            # also need to load spill data for mid-run save!

        model = cls.new_from_dict(_to_dict)

        model._load_spill_data(os.path.join(saveloc, 'spills_data_arrays.nc'))

        return model

    @classmethod
    def _load_collection(cls, saveloc, l_coll_dict, refs):
        '''
        doesn't need to be classmethod of the Model, but its only used by
        Model at present
        '''
        l_coll = []
        for item in l_coll_dict:
            i_ref = item['id']
            if refs.retrieve(i_ref):
                l_coll.append(refs.retrieve(i_ref))
            else:
                f_name = os.path.join(saveloc, item['id'])
                obj = load(f_name, refs)    # will add obj to refs
                l_coll.append(obj)
        return (l_coll)

    def merge(self, model):
        '''
        merge 'model' into self
        '''
        for attr in self.__dict__:
            if (getattr(self, attr) is None and
                    getattr(model, attr) is not None):
                setattr(self, attr, getattr(model, attr))

        # update orderedcollections
        for oc in self._oc_list:
            my_oc = getattr(self, oc)
            new_oc = getattr(model, oc)
            for item in new_oc:
                if item not in my_oc:
                    my_oc += item

        # update forecast spills in SpillContainerPair
        # Uncertain spills automatically be created if uncertainty is on
        for spill in model.spills:
            if spill not in self.spills:
                self.spills += spill

        # force rewind after merge?
        self.rewind()
