#!/usr/bin/env python
import os
import shutil
import types
from datetime import datetime, timedelta
import glob
import copy
import json
import inspect

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import numpy
np = numpy
from colander import (MappingSchema, SchemaNode,
                      Float, Int, Bool, drop)

from gnome.environment import Environment

import gnome.utilities.cache
from gnome.utilities.time_utils import round_time
from gnome.utilities.orderedcollection import OrderedCollection
from gnome.utilities.serializable import Serializable, Field

from gnome.spill_container import SpillContainerPair

from gnome.movers import Mover
from gnome.weatherers import Weatherer

from gnome.outputters import Outputter, NetCDFOutput

from gnome.persist import (extend_colander,
                           validators,
                           References,
                           load)
from gnome.persist.base_schema import (ObjType,
                                       OrderedCollectionItemsList)


class SpillContainerPairSchema(MappingSchema):
    '''
    Schema for SpillContainerPair object.
    Since this is currently only used by the model, define the schema
    in this module. The SpillContainerPair object is not serializable since
    there isn't a need
    '''
    certain_spills = OrderedCollectionItemsList(name='certain_spills')
    uncertain_spills = OrderedCollectionItemsList(name='uncertain_spills',
                                                  missing=drop)


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
    spills = OrderedCollectionItemsList(missing=drop)
    uncertain_spills = OrderedCollectionItemsList(missing=drop)
    movers = OrderedCollectionItemsList(missing=drop)
    weatherers = OrderedCollectionItemsList(missing=drop)
    environment = OrderedCollectionItemsList(missing=drop)
    outputters = OrderedCollectionItemsList(missing=drop)


class Model(Serializable):
    '''
    PyGnome Model Class
    '''
    _update = ['time_step',
               'weathering_substeps',
               'start_time',
               'duration',
               'time_step',
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
               Field('uncertain_spills', save=True, test_for_eq=False)]

    # list of OrderedCollections
    _oc_list = ['movers', 'weatherers', 'environment', 'outputters']

    @classmethod
    def new_from_dict(cls, dict_):
        'Restore model from previously persisted _state'
        json_ = dict_.pop('json_')
        l_env = dict_.pop('environment', [])
        l_out = dict_.pop('outputters', [])
        l_movers = dict_.pop('movers', [])
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
            # default is to enable cache
            default_restore['cache_enabled'] = True

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
        [model.movers.add(obj) for obj in l_movers]
        [model.weatherers.add(obj) for obj in l_weatherers]

        # register callback with OrderedCollection
        model.movers.register_callback(model._callback_add_mover,
                                       ('add', 'replace'))

        model.weatherers.register_callback(model._callback_add_weatherer,
                                           ('add', 'replace'))

        # restore the spill data outside this method - let's not try to find
        # the saveloc here
        return model

    def __init__(self,
                 time_step=timedelta(minutes=15),
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

        self.weatherers.register_callback(self._callback_add_weatherer,
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
        self.time_step = time_step  # this calls rewind() !

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

        # there is a zeroth time step
        self._num_time_steps = int(self._duration.total_seconds()
                                   // self._time_step) + 1
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

        # there is a zeroth time step
        self._num_time_steps = int(self._duration.total_seconds()
                                   // self.time_step) + 1

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

    def setup_model_run(self):
        '''
        Sets up each mover for the model run
        '''
        self.spills.rewind()  # why is rewind for spills here?

        # remake orderedcollections defined by model
        for oc in [self.movers, self.weatherers,
                   self.outputters, self.environment]:
            oc.remake()

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
        output_info = {}

        for outputter in self.outputters:
            if self.current_time_step == self.num_time_steps - 1:
                output = outputter.write_output(self.current_time_step, True)
            else:
                output = outputter.write_output(self.current_time_step)

            if output is not None:
                output_info.update(output)

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

                if log:
                    print results

                output_data.append(results)
            except StopIteration:
                print 'Done with the model run'
                break

        return output_data

    def _callback_add_mover(self, obj_added):
        'Callback after mover has been added'
        if hasattr(obj_added, 'wind'):
            if obj_added.wind.id not in self.environment:
                self.environment += obj_added.wind

        if hasattr(obj_added, 'tide') and obj_added.tide is not None:
            if obj_added.tide.id not in self.environment:
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
        print 'Model.__eq__(): super check =', check
        if check:
            # also check the data in ordered collections
            if type(self.spills) != type(other.spills):
                print 'Model.__eq__(): spill types:', (type(self.spills),
                                                       type(other.spills))
                return False

            if self.spills != other.spills:
                print 'Model.__eq__(): spills:'
                pp.pprint((self.spills, other.spills))
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
        if name and references.reference(self) != name:
            # todo: want a warning here instead of an exception
            raise Exception("{0} already exists, cannot name "
                "the model's json file: {0}".format(name))
            pass
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
                    'attribute is stored as a reference to environment list'
                    if getattr(obj, field.name) is not None:
                        ref_obj = getattr(obj, field.name)
                        index = self.environment.index(ref_obj)
                        json_[field.name] = index
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
                '{0}_uncertain{1}'.format(spill_data_fname, ext))

        array_types = {}

        for m in self.movers:
            array_types.update(m.array_types)

        for w in self.weatherers:
            array_types.update(w.array_types)

        for sc in self.spills.items():
            if sc.uncertain:
                data = NetCDFOutput.read_data(u_spill_data, time=None,
                                              which_data='all')
            else:
                data = NetCDFOutput.read_data(spill_data, time=None,
                                              which_data='all')

            sc.current_time_stamp = data.pop('current_time_stamp').item()
            sc._data_arrays = data
            sc._array_types.update(array_types)

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
        schema = self.__class__._schema()
        o_json_ = schema.serialize(toserial)
        o_json_['map'] = self.map.serialize(json_)

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
        deserial = cls._schema().deserialize(json_)

        if 'map' in json_ and json_['map']:
            deserial['map'] = json_['map']

        if json_['json_'] == 'webapi':
            for attr in ('environment', 'outputters', 'weatherers', 'movers',
                         'spills'):
                if attr in json_ and json_[attr]:
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
            fqn = item['obj_type']
            name, scope = (list(reversed(fqn.rsplit('.', 1)))
                           if fqn.find('.') >= 0
                           else [fqn, ''])
            my_module = __import__(scope, globals(), locals(), [str(name)], -1)
            py_class = getattr(my_module, name)

            deserial.append(py_class.deserialize(item))

        return deserial

    @classmethod
    def load(cls, saveloc, json_data, references=None):
        '''
        '''
        references = (references, References())[references is None]

        # model has no datafiles or 'save_reference' attributes so no need to
        # do anything special for it. But let's add this as a check anyway
        datafiles = cls._state.get_field_by_attribute('isdatafile')
        ref_fields = cls._state.get_field_by_attribute('save_reference')
        if (datafiles or ref_fields):
            raise Exception("Model.load() assumes none of the attributes "
                "defining the state 'isdatafile' or is 'save_reference'. "
                "If this changes, then we need to make this more robust.")

        # deserialize after removing references
        _to_dict = cls.deserialize(json_data)

        # load nested map object and add it - currently, 'load' is only used
        # for laoding save files/location files, so it assumes:
        # json_data['json_'] == 'save'
        if ('map' in json_data):
            map_obj = eval(json_data['map']['obj_type']).load(saveloc,
                json_data['map'], references)
            _to_dict['map'] = map_obj

        # load collections
        for oc in cls._oc_list:
            if oc in _to_dict:
                _to_dict[oc] = cls._load_collection(saveloc, _to_dict[oc],
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
