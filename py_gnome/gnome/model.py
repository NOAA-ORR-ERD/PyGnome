#!/usr/bin/env python
import os
import shutil
from datetime import datetime, timedelta
import copy
import inspect
import zipfile

import numpy as np

from colander import (SchemaNode,
                      Float, Int, Bool, drop)

from gnome.environment import Environment

import gnome.utilities.cache
from gnome.utilities.time_utils import round_time
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.utilities.serializable import Serializable, Field

from gnome.basic_types import oil_status, fate
from gnome.spill_container import SpillContainerPair
from gnome.environment import Wind
from gnome.movers import Mover
from gnome.weatherers import (weatherer_sort,
                              Weatherer,
                              WeatheringData,
                              FayGravityViscous)
from gnome.outputters import Outputter, NetCDFOutput, WeatheringOutput
from gnome.persist import (extend_colander,
                           validators,
                           References,
                           class_from_objtype)
from gnome.persist.base_schema import (ObjType,
                                       CollectionItemsList)
from gnome.exceptions import ReferencedObjectNotSet
from select import select
from sqlalchemy.sql.selectable import Select
# from aifc import data


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
    make_default_refs = SchemaNode(Bool(), missing=drop)

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
            self.add(CollectionItemsList(missing=drop, name='spills'))
            self.add(CollectionItemsList(missing=drop,
                     name='uncertain_spills'))
            self.add(CollectionItemsList(missing=drop, name='movers'))
            self.add(CollectionItemsList(missing=drop,
                     name='weatherers'))
            self.add(CollectionItemsList(missing=drop,
                     name='environment'))
            self.add(CollectionItemsList(missing=drop,
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
               Field('make_default_refs', save=True, update=True)]

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

        # OrderedCollections are being used so maintain order.
        if json_ == 'webapi':
            model.update_from_dict(dict_)
        else:
            cls._restore_attr_from_save(model, dict_)

        # restore the spill data outside this method - let's not try to find
        # the saveloc here
        msg = ("{0._pid}'new_from_dict' created new model: "
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

        # register callback with OrderedCollection
        self.movers.register_callback(self._callback_add_mover,
                                      ('add', 'replace'))

        self.weatherers.register_callback(self._callback_add_weatherer_env,
                                          ('add', 'replace'))

        self.environment.register_callback(self._callback_add_weatherer_env,
                                           ('add', 'replace'))

    def __restore__(self, time_step, start_time, duration,
                    weathering_substeps, uncertain, cache_enabled, map, name):
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

        # reset _current_time_step
        self._current_time_step = -1
        self._time_step = None
        if time_step is not None:
            self.time_step = time_step  # this calls rewind() !
        self._reset_num_time_steps()

        # default is to zip save file
        self.zipsave = True

        # model creates references to weatherers/environment if
        # make_default_refs is True
        self.make_default_refs = True

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

        self.logger.info(self._pid + "rewound model - " + self.name)

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
    def has_weathering_uncertainty(self):
        return (any([w.on for w in self.weatherers]) and
                len([o for o in self.outputters
                     if isinstance(o, WeatheringOutput)]) > 0 and
                (any([s.amount_uncertainty_scale > 0.0
                     for s in self.spills]) or
                 any([w.speed_uncertainty_scale > 0.0
                     for w in self.environment
                     if isinstance(w, Wind)]))
                )

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

    def find_by_class(self, obj, collection, ret_all=False):
        '''
        Look for an object that isinstance() of obj in specified colleciton.
        By default, it will return the first object of this type.
        To get all obects of this type, set ret_all to True
        '''
        all_objs = []
        for item in collection:
            if isinstance(item, obj):
                if not ret_all:
                    return obj
                else:
                    all_objs.append(obj)

        if len(all_objs) == 0:
            return None

        return all_objs

    def find_by_attr(self, attr, value, collection, allitems=False):
        '''
        find first object in collection where the 'attr' attribute matches
        'value'. This is primarily used to find 'wind', 'water', 'waves'
        objects in environment collection. Use the '_ref_as' attribute to
        search.

        Ignore AttributeError since all objects in collection may not contain
        the attribute over which we are searching.

        :param str attr: attribute whose value must match
        :param str value: desired value of the attribute
        :param OrderedCollection collection: the ordered collection in which
            to search
        '''
        items = []
        for item in collection:
            try:
                if getattr(item, attr) == value:
                    if allitems:
                        items.append(item)
                    else:
                        return item
            except AttributeError:
                pass
        items = None if items == [] else items

        return items

    def _order_weatherers(self):
        'use weatherer_sort to sort the weatherers'
        s_weatherers = sorted(self.weatherers, key=weatherer_sort)
        if self.weatherers.values() != s_weatherers:
            self.weatherers.clear()
            self.weatherers += s_weatherers

    def _attach_references(self):
        '''
        attach references
        '''
        attr = {'wind': None, 'water': None, 'waves': None}
        attr['wind'] = self.find_by_attr('_ref_as', 'wind', self.environment)
        attr['water'] = self.find_by_attr('_ref_as', 'water', self.environment)
        attr['waves'] = self.find_by_attr('_ref_as', 'waves', self.environment)

        weather_data = set()
        wd = None
        spread = None
        for coll in ('environment', 'weatherers', 'movers'):
            for item in getattr(self, coll):

                if coll == 'weatherers':
                    # by default turn WeatheringData and spreading object off
                    if isinstance(item, WeatheringData):
                        item.on = False
                        wd = item

                    try:
                        if item._ref_as == 'spreading':
                            item.on = False
                            spread = item

                    except AttributeError:
                        pass

                    if item.on:
                        weather_data.update(item.array_types)

                if hasattr(item, 'on') and not item.on:
                    # no need to setup references if item is not on
                    continue

                for name, val in attr.iteritems():
                    if hasattr(item, name) and item.make_default_refs:
                        setattr(item, name, val)

        # if WeatheringData object and FayGravityViscous (spreading object)
        # are not defined by user, add them automatically because most
        # weatherers will need these
        if len(weather_data) > 0:
            if wd is None:
                self.weatherers += WeatheringData(attr['water'])
            else:
                # turn mass_balance on and make references
                wd.on = True
                if wd.make_default_refs:
                    wd.water = attr['water']

        # if a weatherer is using 'area' array, make sure it is being set.
        # Objects that set 'area' are referenced as 'spreading'
        if 'area' in weather_data:
            if spread is None:
                self.weatherers += FayGravityViscous(attr['water'])
            else:
                # turn spreading on and make references
                spread.on = True
                if spread.make_default_refs:
                    for at in attr:
                        if hasattr(spread, at):
                            spread.water = attr['water']

    def setup_model_run(self):
        '''
        Sets up each mover for the model run
        '''
        # use a set since we only want to add unique 'names' for data_arrays
        # that will be added
        array_types = set()

        # attach references so objects don't raise ReferencedObjectNotSet error
        # in prepare_for_model_run()
        self._attach_references()
        self.spills.rewind()  # why is rewind for spills here?

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
                # weatherers will initialize 'mass_balance' key/values
                # to 0.0
                if w.on:
                    w.prepare_for_model_run(sc)
                    weathering = True
                    array_types.update(w.array_types)

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

                # let model mark these particles to be removed
                tbr_mask = sc['status_codes'] == oil_status.off_maps
                sc['status_codes'][tbr_mask] = oil_status.to_be_removed

                self._update_fate_status(sc)

                # the final move to the new positions
                (sc['positions'])[:] = sc['next_positions']

    def _update_fate_status(self, sc):
        '''
        WeatheringData used to perform this operation in weather_elements;
        however, WeatheringData is one of the objects in weatherers collection
        so just let model do this for now. Eventually, we want to get rid
        of 'fate_status' array and only manipulate 'status_codes'. Until then,
        update fate_status in move_elements
        '''
        if 'fate_status' in sc:
            non_w_mask = sc['status_codes'] == oil_status.on_land
            sc['fate_status'][non_w_mask] = fate.non_weather

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
            # elements may have beached to update fate_status

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
            for sc in self.spills.items():
                w.model_step_is_done(sc)

        for outputter in self.outputters:
            outputter.model_step_is_done()

        for sc in self.spills.items():
            '''
            removes elements with oil_status.to_be_removed
            '''
            sc.model_step_is_done()

            # age remaining particles
            sc['age'][:] = sc['age'][:] + self.time_step

    def write_output(self, valid, messages=None):
        output_info = {'step_num': self.current_time_step}

        for outputter in self.outputters:
            if self.current_time_step == self.num_time_steps - 1:
                output = outputter.write_output(self.current_time_step, True)
            else:
                output = outputter.write_output(self.current_time_step)

            if output is not None:
                output_info[outputter.__class__.__name__] = output

        if len(output_info) > 1:
            # append 'valid' flag to output
            output_info['valid'] = valid

        return output_info

    def step(self):
        '''
        Steps the model forward (or backward) in time. Needs testing for
        hind casting.
        '''
        isvalid = True
        for sc in self.spills.items():
            # Set the current time stamp only after current_time_step is
            # incremented and before the output is written. Set it to None here
            # just so we're not carrying around the old time_stamp
            sc.current_time_stamp = None

        if self.current_time_step == -1:
            # that's all we need to do for the zeroth time step
            self.setup_model_run()

            # let each object raise appropriate error if obj is incomplete
            # validate and send validation flag if model is invalid
            # (msgs, isvalid) = self.validate()
            # if not isvalid:
            #    raise StopIteration("Setup model run complete but model "
            #                        "is invalid", msgs)

        elif self.current_time_step >= self._num_time_steps - 1:
            # _num_time_steps is set when self.time_step is set. If user does
            # not specify time_step, then setup_model_run() automatically
            # initializes it. Thus, do StopIteration check after
            # setup_model_run() is invoked
            raise StopIteration("Run complete for {0}".format(self.name))

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

            # initialize data - currently only weatherers do this so cycle
            # over weatherers collection - in future, maybe movers can also do
            # this
            if num_released > 0:
                for item in self.weatherers:
                    item.initialize_data(sc, num_released)

            self.logger.debug("{1._pid} released {0} new elements for step:"
                              " {1.current_time_step} for {1.name}".
                              format(num_released, self))

        # cache the results - current_time_step is incremented but the
        # current_time_stamp in spill_containers (self.spills) is not updated
        # till we go through the prepare_for_model_step
        self._cache.save_timestep(self.current_time_step, self.spills)
        output_info = self.write_output(isvalid)
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

        if hasattr(obj_added, 'water') and obj_added.water is not None:
            if obj_added.water.id not in self.environment:
                self.environment += obj_added.water

    def _callback_add_mover(self, obj_added):
        'Callback after mover has been added'
        self._add_to_environ_collec(obj_added)
        self.rewind()  # rewind model if a new mover is added

    def _callback_add_weatherer_env(self, obj_added):
        '''
        Callback after weatherer/environment object has been added. 'waves'
        environment object contains 'wind' and 'water' so add those to
        environment collection and the 'water' attribute.
        todo: simplify this
        '''
        self._add_to_environ_collec(obj_added)
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

    def _create_zip(self, saveloc, name):
        '''
        create a zipfile and update saveloc to point to it. This is now
        passed down to all the objects contained within the Model so they can
        save themselves to zipfile
        '''
        if self.zipsave:
            if name is None and self.name is None:
                z_name = 'Model.zip'
            else:
                z_name = name if name is not None else self.name + '.zip'

            # create the zipfile and update saveloc - _json_to_saveloc checks
            # to see if saveloc is a zipfile
            saveloc = os.path.join(saveloc, z_name)
            z = zipfile.ZipFile(saveloc, 'w',
                                compression=zipfile.ZIP_DEFLATED,
                                allowZip64=self._allowzip64)
            z.close()

        return saveloc

    def save(self, saveloc, references=None, name=None):
        '''
        save the model state in saveloc. If self.zipsave is True, then a
        zip archive is created and model files are saved to the archive.

        This overrides the base class save(). Model contains collections and
        model must invoke save for each object in the collection. It must also
        save the data in the SpillContainer's if it is a mid-run save.

        :param saveloc: zip archive or a valid directory. Model files are
            either persisted here or a new model is re-created from the files
            stored here. The files are clobbered when save() is called.
        :type saveloc: A path as a string or unicode
        :param name=None: If data is saved to zipfile (default behavior), then
            this is name of zip file. For a zipfile, the model's state is
            always contained in Model.json. If zipsave is False, then model's
            json is stored in name.json
        :type name: str
        :param references: dict of references mapping 'id' to a string used for
            the reference. The value could be a unique integer or it could be
            a filename. It is upto the creator of the reference list to decide
            how to reference a nested object.

        :returns: references
        '''
        # if zipsave is on, the create zip and update saveloc
        saveloc = self._create_zip(saveloc, name)

        # Note: Defining references=References() in the function definition
        # keeps the references object in memory between tests - it changes the
        # scope of References() to be outside the Model() instance. We don't
        # want this so define the default here
        references = (references, References())[references is None]
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
            self._save_spill_data(saveloc, 'spills_data_arrays.nc')

        # if saved as zipfile, then store model's json in Model.json - this is
        # default if name is None
        mdl_name = 'Model.json' if self.zipsave or name is None else name
        self._json_to_saveloc(json_, saveloc, references, mdl_name)

        return references

    def _save_spill_data(self, saveloc, nc_file):
        """
        save the data arrays for current timestep to NetCDF
        If saveloc is zipfile, then move NetCDF to zipfile
        """
        zipname = None
        if zipfile.is_zipfile(saveloc):
            saveloc, zipname = os.path.split(saveloc)

        datafile = os.path.join(saveloc, nc_file)
        nc_out = NetCDFOutput(datafile, which_data='all', cache=self._cache)
        nc_out.prepare_for_model_run(model_start_time=self.start_time,
                                     uncertain=self.uncertain,
                                     spills=self.spills)
        nc_out.write_output(self.current_time_step)
        if zipname is not None:
            with zipfile.ZipFile(os.path.join(saveloc, zipname), 'a',
                                 compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=self._allowzip64) as z:
                z.write(datafile, nc_file)
                os.remove(datafile)
                if self.uncertain:
                    u_file = nc_out.uncertain_filename
                    z.write(u_file, os.path.split(u_file)[1])
                    os.remove(u_file)

    def _load_spill_data(self, saveloc, nc_file):
        """
        load NetCDF file and add spill data back in - designed for savefiles
        """
        if zipfile.is_zipfile(saveloc):
            with zipfile.ZipFile(saveloc, 'r') as z:
                if nc_file not in z.namelist():
                    return

                saveloc = os.path.split(saveloc)[0]
                z.extract(nc_file, saveloc)
                if self.uncertain:
                    spill_data_fname, ext = os.path.splitext(nc_file)
                    fname = '{0}_uncertain{1}'.format(spill_data_fname, ext)
                    z.extract(fname, saveloc)

        spill_data = os.path.join(saveloc, nc_file)
        if not os.path.exists(spill_data):
            return

        if self.uncertain:
            spill_data_fname, ext = os.path.splitext(nc_file)
            u_spill_data = os.path.join(saveloc,
                                        '{0}_uncertain{1}'
                                        .format(spill_data_fname, ext))

        array_types = set()

        for m in self.movers:
            array_types.update(m.array_types)

        for w in self.weatherers:
            array_types.update(w.array_types)

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
            sc.mass_balance = weather_data

        # delete file after data is loaded - since no longer needed
        os.remove(spill_data)
        if self.uncertain:
            os.remove(u_spill_data)

    # todo: remove following

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

        if json_ == 'webapi':
            # for webapi, we serialize forecast spills just like all other
            # collections - ignore spills in uncertain spill container
            for attr in ('environment', 'outputters', 'weatherers', 'movers',
                         'spills'):
                o_json_[attr] = self.serialize_oc(getattr(self, attr), json_)

            # validate and send validation flag
            (msgs, isvalid) = self.validate()
            o_json_['valid'] = isvalid
            if len(msgs) > 0:
                o_json_['messages'] = msgs

        return o_json_

    @classmethod
    def deserialize(cls, json_):
        '''
        treat special-case attributes of Model.
        '''
        schema = cls._schema(json_['json_'])
        deserial = schema.deserialize(json_)
        if 'map' in json_:
            #d_item = cls._deserialize_nested_obj(json_['map'])
            #deserial['map'] = d_item
            # map will be deserialized later - no need to do it twice
            # todo: clean this up
            deserial['map'] = json_['map']

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
    def loads(cls, json_data, saveloc, references=None):
        '''
        loads a model from json_data

        - load json for references from files
        - update paths of datafiles if needed
        - deserialize json_data
        - and create object with new_from_dict()

        :param saveloc: location of data files

        Optional parameter

        :param references: references object - if this is called by the Model,
            it will pass a references object. It is not required.
        '''
        references = (references, References())[references is None]
        ref_dict = cls._load_refs(json_data, saveloc, references)

        # there are no datafiles for model properties; so no need for following
        # at present
        cls._update_datafile_path(json_data, saveloc)

        # deserialize after removing references
        _to_dict = cls.deserialize(json_data)

        if ref_dict:
            _to_dict.update(ref_dict)

        # load nested map object and add it - currently, 'load' is only used
        # for laoding save files/location files, so it assumes:
        # json_data['json_'] == 'save'
        if ('map' in json_data):
            mapcls = class_from_objtype(json_data['map'].pop('obj_type'))
            map_obj = mapcls.loads(json_data['map'], saveloc, references)
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

        model._load_spill_data(saveloc, 'spills_data_arrays.nc')

        return model

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

    def validate(self):
        '''
        invoke validate for all gnome objects contained in model
        todo: should also check wind, water, waves are defined if weatherers
        are defined
        '''
        # since model does not contain wind, waves, water attributes, no need
        # to call base class method - model requires following only if an
        # object in collection requires it
        env_req = set()
        msgs = []
        isvalid = True
        for oc in self._oc_list:
            for item in getattr(self, oc):
                # if item is not on, no need to validate it
                if hasattr(item, 'on') and not item.on:
                    continue

                # validate item
                (msg, i_isvalid) = item.validate()
                if not i_isvalid:
                    isvalid = i_isvalid

                msgs.extend(msg)

                # add to set of required env objects if item's
                # make_default_refs is True
                if item.make_default_refs:
                    for attr in ('wind', 'water', 'waves'):
                        if hasattr(item, attr):
                            env_req.update({attr})

        # ensure that required objects are present in environment collection
        if len(env_req) > 0:
            (ref_msgs, ref_isvalid) = \
                self._validate_env_coll(env_req)
            if not ref_isvalid:
                isvalid = ref_isvalid
            msgs.extend(ref_msgs)

        # Spill warnings
        if len(self.spills) == 0:
            msg = '{0} contains no spills'.format(self.name)
            self.logger.warning(msg)
            msgs.append(self._warn_pre + msg)

        for spill in self.spills:
            msg = None
            if spill.get('release_time') > self.start_time:
                msg = ('{0} has release time after model start time'.
                       format(spill.name))

            elif spill.get('release_time') < self.start_time:
                msg = ('{0} has release time before model start time'
                       .format(spill.name))

            if msg is not None:
                self.logger.warning(msg)
                msgs.append(self._warn_pre + msg)

        return (msgs, isvalid)

    def _validate_env_coll(self, refs, raise_exc=False):
        '''
        validate refs + log warnings or raise error if required refs not found.
        If refs is None, model must query its weatherers/movers/environment
        collections to figure out what objects it needs to have in environment.
        '''
        msgs = []
        isvalid = True

        if refs is None:
            # need to go through orderedcollections to see if water, waves
            # and wind refs are required
            raise NotImplementedError("validate_refs() incomplete")

        for ref in refs:
            obj = self.find_by_attr('_ref_as', ref, self.environment)
            if obj is None:
                msg = ("{0} not found in environment collection".
                       format(ref))
                if raise_exc:
                    raise ReferencedObjectNotSet(msg)
                else:
                    self.logger.warning(msg)
                    msgs.append(self._warn_pre + msg)
                    isvalid = False

        return (msgs, isvalid)

    def set_make_default_refs(self, value):
        '''
        make default refs for all items in ('weatherers', 'movers',
        'environment') collections
        '''
        for attr in ('weatherers', 'movers', 'environment'):
            oc = getattr(self, attr)
            for item in oc:
                item.make_default_refs = value

    def get_spill_property(self, prop_name, ucert=0):
        '''
        Convenience method to allow user to look up properties of a spill.
        User can specify ucert as 'ucert' or 1
        '''
        if ucert == 'ucert':
            ucert = 1
        return self.spills.items()[ucert][prop_name]
    
    def get_spill_data(self, target_properties, conditions, ucert=0):
        '''
        Convenience method to allow user to write an expression to filter raw spill data
        Example case: 
        get_spill_data('position && mass','position > 50 && spill_num == 1 || status_codes == 1')
        
        WARNING: EXPENSIVE! USE AT YOUR OWN RISK ON LARGE num_elements!
        
        Example spill element properties are below. This list may not contain all properties tracked by the model.
        'positions', 'next_positions', 'last_water_positions', 'status_codes',
        'spill_num', 'id', 'mass', 'age'
        '''
        if ucert == 'ucert':
            ucert = 1
        def elem_val(prop,index):
            '''
            Gets the column containing the information on one element
            '''
            val = self.spills.items()[ucert].data_arrays[prop][index]
            return val
        
        def test_phrase(phrase):
            for sub_cond in phrase:
                    cond = sub_cond.rsplit()
                    prop_val = elem_val(cond[0],i)
                    op = cond [1]
                    test_num = cond[2]
                    if test(prop_val,op,test_num):
                        return True
                    
            return False
        
        def test(elem_value, op, test_val):
            if op in {'<','<=','>','>=','=='}:
                return eval(str(int(elem_value))+op+test_val)
        
        def num(s):
            try:
                return int(s)
            except ValueError:
                return float(s)
        conditions = conditions.rsplit('&&')
        conditions = [str(cond).rsplit('||') for cond in conditions]
        
        
        sc = self.spills.items()[ucert]
        result = {}
        for t in target_properties:
            result[t] = []
        for i in range(0,len(sc)):
            test_result = True
            for phrase in conditions:
                if not test_phrase(phrase):
                    test_result = False
                    break
            if test_result:         
                for k in result.keys():
                    n = elem_val(k,i)
                    result[k].append(n)       
        return result