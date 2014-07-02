#!/usr/bin/env python
"""
outputters.py

module to define classes for GNOME output:
  - base class
  - saving to netcdf
  - saving to other formats ?

"""
import copy
from datetime import timedelta

from colander import SchemaNode, MappingSchema, Bool, drop


from gnome.persist import base_schema, extend_colander
from gnome.utilities.serializable import Serializable, Field


class BaseSchema(base_schema.ObjType, MappingSchema):
    'Base schema for all outputters - they all contain the following'
    output_zero_step = SchemaNode(Bool())
    output_last_step = SchemaNode(Bool())
    output_timestep = SchemaNode(extend_colander.TimeDelta(), missing=drop)


class Outputter(Serializable):
    '''
    base class for all outputters
    Since this outputter doesn't do anything, it'll never be used as part
    of a gnome model. As such, it should never need to be serialized
    '''
    _state = copy.deepcopy(Serializable._state)
    _state += (Field('output_zero_step', save=True, update=True),
               Field('output_last_step', save=True, update=True),
               Field('output_timestep', save=True, update=True))
    _schema = BaseSchema

    def __init__(self,
                 cache=None,
                 output_timestep=None,
                 output_zero_step=True,
                 output_last_step=True,
                 name=None):
        """
        sets attributes for all outputters, like output_timestep, cache

        :param cache: sets the cache object from which to read data. The model
            will automatically set this param

        :param output_timestep: default is None in which case every time the
            write_output is called, output is written. If set, then output is
            written every output_timestep starting from model_start_time.
        :type output_timestep: timedelta object

        :param output_zero_step: default is True. If True then output for
            initial step (showing initial release conditions) is written
            regardless of output_timestep
        :type output_zero_step: boolean

        :param output_last_step: default is True. If True then output for
            final step is written regardless of output_timestep
        :type output_last_step: boolean
        """
        self.cache = cache
        self.output_timestep = output_timestep
        self.output_zero_step = output_zero_step
        self.output_last_step = output_last_step

        if name:
            self.name = name

        # reset internally used variables
        self.rewind()

    def prepare_for_model_run(self, model_start_time, spills=None,
                              **kwargs):
        """
        This method gets called by the model at the beginning of a new run.
        Do what you need to do to prepare.

        Required arguments - if output_timestep is changed from None, these are
        needed. Just make them required.

        :param model_start_time: (Required) start time of the model run. NetCDF
            time units calculated with respect to this time.
        :type model_start_time: datetime.datetime object

        Optional argument - in case cache needs to be updated

        :param cache=None: Sets the cache object to be used for the data.
            If None, it will use the one already set up.
        :type cache: As defined in cache module (gnome.utilities.cache).
            Currently only ElementCache is defined/used.

        also added **kwargs since a derived class like NetCDFOutput could
        require additional variables.
        """
        cache = kwargs.pop('cache', None)
        if cache is not None:
            self.cache = cache

        if model_start_time is None:
            raise TypeError('model_start_time cannot be NoneType if'
                            ' output_timestep is not None')

        self._model_start_time = model_start_time
        if self.output_timestep is not None:
            self._next_output_time = (self._model_start_time +
                                      self.output_timestep)

    def prepare_for_model_step(self, time_step, model_time):
        """
        This method gets called by the model at the beginning of each time step
        Do what you need to do to prepare for a new model step

        base class method checks to see if data for model_time should be output
        Set self._write_step flag to true if:
            model_time < self._next_output_timestep <= model_time + time_step

        :param time_step: time step in seconds
        :param model_time: current model time as datetime object

        Note: If output_timestep is not set so every timestep is in fact
            written out, then there is no need to call prepare_for_model_step.
            This is primarily only useful if user wants to write data post run
            and write out every step_num saved in the model's cache. In this
            case, user can simply call prepare_for_model_run() followed by
            write_output for every step_num in range(model.num_time_steps)
        """
        self._write_step = False

        if self._next_output_time is not None:
            end_model_time = model_time + timedelta(seconds=time_step)

            if (self._next_output_time > model_time and
                self._next_output_time <= end_model_time):
                self._write_step = True

    def model_step_is_done(self):
        '''
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc.
        '''
        pass

    def write_output(self, step_num, islast_step=False):
        """
        called by the model at the end of each time step
        This is the last operation after model_step_is_done()

        :param step_num: the model step number you want rendered.
        :type step_num: int

        :param islast_step: default is False. Flag that indicates that step_num
            is last step. If 'output_last_step' is True then this is written
            out
        :type islast_step: bool
        """
        if (step_num == 0 and self.output_zero_step):
            self._write_step = True

        if (islast_step and self.output_last_step):
            self._write_step = True

        if (self._write_step and self.cache is None):
            raise ValueError('cache object is not defined. It is required'
                             ' prior to calling write_output')

        # if output_time_step is not set, then no need to call
        # prepare_for_model_step. This is primarily useful when every timestep
        # is written post model run. In this case, it is easier to call
        # write_output() without messing with prepare_for_model_step which
        # requires model_time as input
        if self._next_output_time is None:
            self._write_step = True

    def rewind(self):
        '''
        Called by model.rewind()

        Reset variables set during prepare_for_model_run() to init conditions
        Make sure all child classes call parent rewind() first!
        '''
        self._model_start_time = None
        self._next_output_time = None
        self._write_step = False

    def _update_next_output_time(self, step_num, time_stamp):
        """
        Internal method to update self._next_output_time by:
            self._next_output_time = self.time_stamp + self.output_timestep

        Call only after data associated with time_stamp is written. This
        function updates the _next_output_time

        :param time_stamp: datetime associated with data written by
            write_output
        """
        if (step_num == 0):
            # update not required if 0th step or final step.
            # Strictly speaking this logic is not required since setting the
            # _next_output_time at time 0 doesn't change its value
            # But why reset it if not required
            return

        if self.output_timestep is not None:
            self._next_output_time = time_stamp + self.output_timestep

    def write_output_post_run(self, model_start_time, num_time_steps,
                              **kwargs):
        """
        If the model has already been run and the data is cached, then use
        this function to write output. In this case, num_time_steps is known
        so pass it into this function.

        :param model_start_time: (Required) start time of the model run. NetCDF
            time units calculated with respect to this time.
        :type model_start_time: datetime.datetime object

        :param num_time_steps: (Required) total number of time steps for the
            run. Currently this is known and fixed.
        :type num_time_steps: int

        Optional argument - depending on the outputter, the following maybe
        required. For instance, the 'spills' are required by NetCDFOutput,
        GeoJson, but not Renderer in prepare_for_model_run(). The **kwargs here
        are those required by prepare_for_model_run() for an outputter

        :param cache=None: Sets the cache object to be used for the data.
            If None, it will use the one already set up.
        :type cache: As defined in cache module (gnome.utilities.cache).
            Currently only ElementCache is defined/used.

        :param uncertain: is there uncertain data to write. Used by
            NetCDFOutput to setup attributes for uncertain data file
        :type uncertain: bool

        :param spills: SpillContainerPair object containing spill information
            Used by both the NetCDFOutput and by GeoJson to obtain spill_id
            from spill_num
        :type spills: This is the Model's spills attribute which refers to the
            SpillContainerPair object

        Follows the iteration in Model().step() for each step_num
        """
        self.prepare_for_model_run(model_start_time, **kwargs)
        model_time = model_start_time
        last_step = False

        for step_num in range(num_time_steps):
            if (step_num > 0 and step_num < num_time_steps - 1):
                next_ts = (self.cache.load_timestep(step_num + 1).items()[0].
                           current_time_stamp)
                ts = next_ts - model_time
                self.prepare_for_model_step(ts.seconds, model_time)

            if step_num == num_time_steps - 1:
                last_step = True

            self.write_output(step_num, last_step)
            model_time = (self.cache.load_timestep(step_num).items()[0].
                         current_time_stamp)
