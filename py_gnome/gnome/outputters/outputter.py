#!/usr/bin/env python

"""
outputters.py

module to define classes for GNOME output:
  - base class
  - saving to netcdf
  - saving to other formats ?

"""
import os

from datetime import timedelta

import warnings

from gnome.persist import (
    SchemaNode,
    Boolean,
    drop,
    String,
    ObjTypeSchema,
    TimeDelta,
    LocalDateTime,
)
from gnome.array_types import gat

from gnome.utilities.surface_concentration import compute_surface_concentration
from gnome.gnomeobject import GnomeId


class BaseOutputterSchema(ObjTypeSchema):
    'Base schema for all outputters - they all contain the following'
    on = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )
    output_zero_step = SchemaNode(
        Boolean(), save=True, update=True
    )
    output_last_step = SchemaNode(
        Boolean(), save=True, update=True
    )
    output_single_step = SchemaNode(
        Boolean(), save=True, update=True
    )
    output_timestep = SchemaNode(
        TimeDelta(), missing=drop, save=True, update=True
    )
    output_start_time = SchemaNode(
        LocalDateTime(),
        missing=drop, save=True, update=True
    )
    surface_conc = SchemaNode(
        String(allow_empty=True), save=True, update=True
    )


class Outputter(GnomeId):
    '''
    Base class for all outputters
    Since this outputter doesn't do anything, it'll never be used as part
    of a gnome model. As such, it should never need to be serialized
    '''
    _schema = BaseOutputterSchema

    _surf_conc_computed = False

    def __init__(self,
                 cache=None,
                 on=True,
                 output_timestep=None,
                 output_zero_step=True,
                 output_last_step=True,
                 output_single_step=False,
                 output_start_time=None,
                 output_dir=None,  # Fixme: this probably shouldn't be in the base class
                 surface_conc=None,
                 *args,
                 **kwargs):
        """
        Sets attributes for outputters, like output_timestep, cache, etc.

        :param cache: sets the cache object from which to read data. The model
                      will automatically set this parameter.

        :param output_timestep=None: If ``None`` output will be written every model time step.
                                     If set, then output is written every output_timestep
                                     starting from the model start time. If the output_timestep
                                     is less than the model timestep, an Warning will be raised
                                     at runtime.
        :type output_timestep: timedelta object

        :param output_zero_step=True: If True then output for initial step (showing initial
                                      release conditions) is written regardless of
                                      ``output_timestep`` or ``output_single_step``
        :type output_zero_step: bool

        :param output_last_step=True: If True then output for final step is written
                                      regardless of ``output_timestep`` or ``output_single_step``.
                                      This is potentially an extra output, if not aligned with
                                      ``output_timestep``.
        :type output_last_step: bool

        :param output_single_step=False: If ``True`` then output is written for only one
                                         step, the output_start_time, regardless of ``output_timestep``.
                                         ``output_zero_step`` and ``output_last_step`` are still
                                         respected, set these to False if you want only one time step.
        :type output_single_step: boolean

        :param output_start_time=None: Time to start outputting restults. If None it is set to the model
                                       start time
        :type output_start_time: datetime object

        :param output_dir=None: Directory to dump output in, if it needs to do this.
        :type output_dir: PathLike

        :param surface_conc=None: Compute surface concentration
                                  Any non-empty string will compute (and output)
                                  the surface concentration. The contents of the
                                  string determine the algorithm used. "kde" is
                                  currently the only available option.
        :type surface_conc: str or None
        """

        self._middle_of_run = False

        super(Outputter, self).__init__(*args, **kwargs)

        self.cache = cache
        self.on = on
        self.output_zero_step = output_zero_step
        self.output_last_step = output_last_step
        self.output_single_step = output_single_step

        self.output_timestep = output_timestep

        if output_start_time:
            self.output_start_time = output_start_time
        else:
            self.output_start_time = None

        self.sc_pair = None     # set in prepare_for_model_run

        # make sure the output_dir exists:
        if output_dir is not None:
            try:
                os.mkdir(output_dir)
            except OSError:
                pass

        self.output_dir = output_dir

        self.surface_conc = surface_conc
        self.array_types = dict()

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
            if value <= timedelta(0):
                raise ValueError("output_timestep must be positive")
            self._output_timestep = int(value.total_seconds())

    def prepare_for_model_run(self,
                              model_start_time=None,
                              spills=None,
                              model_time_step=None,
                              map=None,
                              **kwargs):
        """
        This method gets called by the model at the beginning of a new run.
        Do what you need to do to prepare.

        :param model_start_time: (Required) start time of the model run.
                                 NetCDF time units calculated with respect
                                 to this time.
        :type model_start_time: datetime.datetime object

        :param spills: (Required) model.spills object (SpillContainerPair)
        :type spills: gnome.spill_container.SpillContainerPair object

        :param model_time_step: time step of the model
                                -- used to set timespans for some outputters
        :type model_time_step: float seconds

        Optional argument - in case cache needs to be updated

        :param cache=None: Sets the cache object to be used for the data.
                           If None, it will use the one already set up.
        :type cache: As defined in cache module (gnome.utilities.cache).
                     Currently only ElementCache is defined/used.

        also added ``**kwargs`` since a derived class like NetCDFOutput could
        require additional variables.

        .. note:: base class doesn't use model_start_time or spills, but
            multiple outputters need spills and netcdf needs model_start_time,
            so just set them here

        """
        # check for required parameters
        # -- they are None so that they can be out of order
        # this breaks tests -- probably should fix the tests...
        if model_start_time is None:
            raise TypeError("model_start_time is a required parameter")

        if hasattr(self, 'filename'):
            self._check_filename(self.filename)
        self.clean_output_files()

        self._model_start_time = model_start_time
        self.model_timestep = model_time_step
        self.sc_pair = spills

        if self._output_timestep is not None:
            if self._output_timestep < model_time_step:
                warnings.warn(f"Outputter output timestep {self.output_timestep} is less than "
                              f"model time step: {model_time_step} seconds. "
                              "Output will only occur every model timestep.",
                              RuntimeWarning)


        cache = kwargs.pop('cache', None)
        if cache is not None:
            self.cache = cache

        self.map = map
        if self.output_timestep is None:
            self._write_step = True

        self._dt_since_lastoutput = 0
        self._middle_of_run = True

        # kludge to catch special case of single output at the model start time
        if self.output_single_step and ((self.output_start_time == model_start_time)
                                        or self.output_start_time is None):
            self._single_output_at_start = True
        else:
            self._single_output_at_start = False

    def prepare_for_model_step(self, time_step, model_time):
        """
        This method gets called by the model at the beginning of each time step
        Do what you need to do to prepare for a new model step

        base class method checks to see if data for model_time should be output
        Set self._write_step flag to true if::

            model_time < self._dt_since_lastoutput <= model_time + time_step

        It also updates the _dt_since_lastoutput internal variable if the data
        from this step will be written to output

        :param time_step: time step in seconds
        :param model_time: current model time as datetime object

        .. note:: The write_output() method will be called after the Model
           calls model_step_is_done(). Let's set the _write_step flag here and
           update the _dt_since_lastoutput variable

        """

        if not self.on:
            return

        self._surf_conc_computed = False

        d = timedelta(seconds=time_step)
        if self.output_start_time is not None:
            if self.output_start_time != self._model_start_time:
                if model_time + d < self.output_start_time:
                    self._write_step = False
                    return

                if model_time + d == self.output_start_time:
                    self._write_step = True
                    self._is_first_output = False
                    return

                if model_time + d > self.output_start_time:
                    if self._is_first_output:
                        self._write_step = True
                        self._is_first_output = False
                        return

        if self.output_single_step:
            self._write_step = False
            return

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

    def post_model_run(self):
        """
        Override this method if a derived class needs to perform
        any actions after a model run is complete (StopIteration triggered)
        """
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
        if step_num == 0:
            if self.output_zero_step or self._single_output_at_start:
                self._write_step = True  # this is the default
            else:
                self._write_step = False

        if (islast_step and self.output_last_step):
            self._write_step = True

        if (self._write_step and self.cache is None):
            raise ValueError('cache object is not defined. It is required'
                             ' prior to calling write_output')

        # compute the surface_concentration if need be
        # doing this here so that it will only happen if there is an
        # output step.
        # this updates the most recent one in the cache

        if (self._write_step
                and self.surface_conc is not None
                and not self._surf_conc_computed):
            # compute the surface concentration and put it in the cache
            try:
                sc = self.cache.recent[step_num][0]  # only the certain one
            except KeyError:
                # not using the most recent one from cache
                # so no need to compute
                # fixme: it may not get into cache at all.
                pass
            else:
                compute_surface_concentration(sc, self.surface_conc)
                self._surf_conc_computed = True

    def clean_output_files(self):
        '''
        Cleans out the output dir

        This should be implemented by subclasses that dump files.

        Each outputter type dumps different types of files, and this should
        only clear out those.

        See the OutputterFilenameMixin for a simple example.

        '''
        pass

    def rewind(self):
        '''
        Called by model.rewind()

        Reset variables set during prepare_for_model_run() to init conditions
        Make sure all child classes call parent rewind() first!
        '''
        self._model_start_time = None
        self._dt_since_lastoutput = None
        self._write_step = True
        self._is_first_output = True
        self._surf_conc_computed = True
        self._middle_of_run = False

        if self.surface_conc:
            self.array_types['surface_concentration'] = gat('surface_concentration')

    def write_output_post_run(self,
                              *,
                              model_start_time,
                              model_time_step,
                              num_time_steps,
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

        Optional argument - depending on the outputter, the following may be
        required. For instance, the 'spills' are required by NetCDFOutput,
        GeoJson, but not Renderer in prepare_for_model_run(). The ``**kwargs``
        here are those required by prepare_for_model_run() for an outputter

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
        self.prepare_for_model_run(model_start_time=model_start_time,
                                   model_time_step=model_time_step,
                                   **kwargs)

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

            model_time = (self.cache.load_timestep(step_num).items()[0]
                          .current_time_stamp)

    @property
    def middle_of_run(self):
        return self._middle_of_run

    # Some utilities for checking valid filenames, etc...
    def _check_filename(self, filename):
        'basic checks to make sure the filename is valid'
        # path.isdir causes problems for shapefile; outputters call as needed
        #if os.path.isdir(filename):
            #raise ValueError('filename must be a file not a directory.')

        if not os.path.exists(os.path.realpath(os.path.dirname(filename))):
            raise ValueError('{0} does not appear to be a valid path'
                             .format(os.path.dirname(filename)))

    def _check_is_dir(self, filename):
        'split this out - causes problems for shape and most outputters dont need it'
        if os.path.isdir(filename):
            raise ValueError('filename must be a file not a directory.')

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


class OutputterFilenameMixin(object):
    """
    mixin for outputter that output to a single file
    """
    def __init__(self, filename, *args, **kwargs):
        super(OutputterFilenameMixin, self).__init__(**kwargs)
        self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, new_name):
        if self.middle_of_run:
            raise AttributeError('This attribute cannot be changed in the '
                                 'middle of a run')
        else:
            # move to prepare for model run so save files with outputters
            # can be loaded
            # self._check_filename(new_name)
            self._filename = new_name

    def clean_output_files(self):
        '''
        deletes output files that may be around

        called by prepare_for_model_run

        here in case it needs to be called from elsewhere
        '''
        # super(OutputterFilenameMixin, self).clean_output_files()
        try:
            os.remove(self.filename)
        except OSError:
            pass  # it must not be there
