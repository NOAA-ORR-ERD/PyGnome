#!/usr/bin/env python
"""
outputters.py

module to define classes for GNOME output:
  - base class
  - saving to netcdf
  - saving to other formats ?

"""
import os
import copy
from datetime import timedelta

from colander import SchemaNode, MappingSchema, Bool, drop


from gnome.persist import base_schema, extend_colander
from gnome.utilities.serializable import Serializable, Field


class BaseSchema(base_schema.ObjType, MappingSchema):
    'Base schema for all outputters - they all contain the following'
    on = SchemaNode(Bool(), missing=drop)
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
    _state += (Field('on', save=True, update=True),
               Field('output_zero_step', save=True, update=True),
               Field('output_last_step', save=True, update=True),
               Field('output_timestep', save=True, update=True))
    _schema = BaseSchema

    def __init__(self,
                 cache=None,
                 on=True,
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
        self.on = on
        self.output_zero_step = output_zero_step
        self.output_last_step = output_last_step
        self.output_timestep = output_timestep

        self.sc_pair = None     # set in prepare_for_model_run

        if name:
            self.name = name

        # reset internally used variables
        self.rewind()

    @property
    def output_timestep(self):
        if self._output_timestep is not None:
            return timedelta(seconds=self._output_timestep)
        else:
            return None

    @output_timestep.setter
    def output_timestep(self, value):
        '''
        make it a property so internally we keep it in seconds, easier to work
        with but let user set it as a timedelta object since that's probably
        easier for user
        '''
        if value is None:
            self._output_timestep = None
        else:
            self._output_timestep = value.seconds

    def prepare_for_model_run(self,
                              model_start_time=None,
                              spills=None,
                              model_time_step=None,
                              **kwargs):
        """
        This method gets called by the model at the beginning of a new run.
        Do what you need to do to prepare.

        :param model_start_time: (Required) start time of the model run. NetCDF
            time units calculated with respect to this time.
        :type model_start_time: datetime.datetime object

        :param spills: (Required) model.spills object (SpillContainerPair)
        :type spills: gnome.spill_container.SpillContainerPair object

        :param model_time_step: time step of the model -- used to set timespans for some outputters
        :type model_time_step: float seconds

        Optional argument - in case cache needs to be updated

        :param cache=None: Sets the cache object to be used for the data.
            If None, it will use the one already set up.
        :type cache: As defined in cache module (gnome.utilities.cache).
            Currently only ElementCache is defined/used.

        also added **kwargs since a derived class like NetCDFOutput could
        require additional variables.

        .. note:: base class doesn't use model_start_time or spills, but
        multiple outputters need spills and netcdf needs model_start_time,
        so just set them here
        """
        #check for required parameters -- they are None so that they can be out of order
        # this breaks tests -- probably should fix the tests...
        if model_start_time is None:
            raise TypeError("model_start_time is a required parameter")
        #if spills is None:
        #    raise TypeError("spills is a required parameter")
        # if model_time_step is None:
        #     raise TypeError("model_time_step is a required parameter")

        self._model_start_time = model_start_time
        self.model_timestep = model_time_step
        self.sc_pair = spills
        cache = kwargs.pop('cache', None)
        if cache is not None:
            self.cache = cache

        if self.output_timestep is None:
            self._write_step = True

        self._dt_since_lastoutput = 0

    def prepare_for_model_step(self, time_step, model_time):
        """
        This method gets called by the model at the beginning of each time step
        Do what you need to do to prepare for a new model step

        base class method checks to see if data for model_time should be output
        Set self._write_step flag to true if:
            model_time < self._dt_since_lastoutput <= model_time + time_step

        It also updates the _dt_since_lastoutput internal variable if the data
        from this step will be written to output

        :param time_step: time step in seconds
        :param model_time: current model time as datetime object

        .. note:: The write_output() method will be called after the Model
        calls model_step_is_done(). Let's set the _write_step flag here and
        update the _dt_since_lastoutput variable

        """
        if self._output_timestep is not None:
            self._write_step = False
            self._dt_since_lastoutput += time_step
            if self._dt_since_lastoutput >= self._output_timestep:
                self._write_step = True
                self._dt_since_lastoutput = (self._dt_since_lastoutput %
                                             self._output_timestep)

    def model_step_is_done(self):
        '''
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc.
        The write_output method is called by Model after all processing.
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

    def rewind(self):
        '''
        Called by model.rewind()

        Reset variables set during prepare_for_model_run() to init conditions
        Make sure all child classes call parent rewind() first!
        '''
        self._model_start_time = None
        self._dt_since_lastoutput = None
        self._write_step = True

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
                next_ts = (self.cache.load_timestep(step_num).items()[0].
                           current_time_stamp)
                ts = next_ts - model_time
                self.prepare_for_model_step(ts.seconds, model_time)

            if step_num == num_time_steps - 1:
                last_step = True

            self.write_output(step_num, last_step)
            model_time = (self.cache.load_timestep(step_num).items()[0].
                          current_time_stamp)

    # Some utilities for checking valid filenames, etc...
    def _check_filename(self, filename):
        'basic checks to make sure the filename is valid'
        if os.path.isdir(filename):
            raise ValueError('filename must be a file not a directory.')

        if not os.path.exists(os.path.realpath(os.path.dirname(filename)
                                               )):
            raise ValueError('{0} does not appear to be a valid path'
                             .format(os.path.dirname(filename)))
    def _file_exists_error(self, file_):
        """
        invoked by prepare_for_model_run. If file already exists, it will raise
        this error.

        Do this in prepare_for_model_run, because user may want to define the
        model and run it in batch mode. This will allow netcdf_outputter to be
        created, but the first time it tries to write this file, it will check
        and raise an error if file exists
        """
        if os.path.exists(file_):
            raise ValueError('{0} file exists. Enter a filename that '
                             'does not exist in which to save data.'
                             .format(file_))


