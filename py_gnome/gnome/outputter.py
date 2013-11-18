
"""
outputters.py

module to define classes for GNOME output:
  - base class
  - saving to netcdf
  - saving to other formats ?

"""

from datetime import timedelta


class Outputter(object):

    """
    base class for all outputters
    """

    def __init__(self,
                 cache=None,
                 output_timestep=None,
                 output_zero_step=True,
                 output_last_step=True):
        """
        sets attributes for all outputters, like output_timestep, cache

        :param output_timestep: default is None in which case everytime the
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

        # internally used variables - set in prepare_for_model_run
        self.rewind()
        #self._model_start_time = None
        #self._num_time_steps = None
        #self._next_output_time = None
        #self._write_step = False

    def prepare_for_model_run(self,
        model_start_time,
        num_time_steps,
        cache=None,
        **kwargs):
        """
        This method gets called by the model at the beginning of a new run.
        Do what you need to do to prepare.

        Required arguments - if output_timestep is changed from None, these are
        needed. Just make them required.

        :param model_start_time: (Required) start time of the model run. NetCDF
            time units calculated with respect to this time.
        :type model_start_time: datetime.datetime object
        :param num_time_steps: (Required) total number of time steps for the
            run. Currently this is known and fixed.
        :type num_time_steps: int

        Optional argument - incase cache needs to be updated

        :param cache=None: Sets the cache object to be used for the data.
            If None, it will use the one already set up.
        :type cache: As defined in cache module (gnome.utilities.cache).
            Currently only ElementCache is defined/used.

        also added **kwargs since a derived class like NetCDFOutput could
        require additional variables.
        """
        if cache is not None:
            self.cache = cache

        if model_start_time is None or num_time_steps is None:
            raise TypeError('model_start_time or num_time_steps cannot be'
                            ' NoneType if output_timestep is not None')

        self._model_start_time = model_start_time
        self._num_time_steps = num_time_steps
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

        Note: If output_time_step is not set so every timestep is infact
            written out, then there is no need to call prepare_for_model_step.
            This is primarily only useful if user wants to write data post run
            and write out every step_num saved in the model's cache. In this
            case, user can simply call prepare_for_model_run() followed by
            write_output for every step_num in range(model.num_time_steps)
        """
        self._write_step = False

        if self._next_output_time is not None:
            if (model_time < self._next_output_time and
                self._next_output_time <= model_time + timedelta(
                                                        seconds=time_step)):
                self._write_step = True

    def model_step_is_done(self):
        """
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc.
        """

        pass

    def write_output(self, step_num):
        """
        called by the model at the end of each time step
        This is the last operation after model_step_is_done()
        """

        if (step_num == 0 and self.output_zero_step):
            self._write_step = True

        if (step_num == self._num_time_steps - 1 and self.output_last_step):
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
        """
        called by model.rewind()

        reset variables set during prepare_for_model_run() to init conditions
        """
        self._model_start_time = None
        self._num_time_steps = None
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
        if (step_num == 0 or step_num == self._num_time_steps - 1):
            # update not required if 0th step or final step.
            # Strictly speaking this logic is not required since setting the
            # _next_output_time at time 0 doesn't change its value and at
            # final step it doesn't matter. But why reset it if not required
            return

        if self.output_timestep is not None:
            self._next_output_time = time_stamp + self.output_timestep
