import copy
from datetime import datetime, timedelta

import numpy as np

from colander import (SchemaNode, MappingSchema, Bool, drop)

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime

from gnome.basic_types import (world_point,
                               world_point_type,
                               spill_type,
                               status_code_type)

from gnome.utilities import inf_datetime
from gnome.utilities import time_utils, serializable
from gnome.cy_gnome.cy_rise_velocity_mover import CyRiseVelocityMover
from gnome import AddLogger
from gnome.utilities.inf_datetime import InfTime, MinusInfTime


class ProcessSchema(MappingSchema):
    '''
    base Process schema - attributes common to all movers/weatherers
    defined at one place
    '''
    on = SchemaNode(Bool(), missing=drop)
    active_start = SchemaNode(LocalDateTime(), missing=drop,
                              validator=convertible_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), missing=drop,
                             validator=convertible_to_seconds)
    real_data_start = SchemaNode(LocalDateTime(), missing=drop,
                              validator=convertible_to_seconds)
    real_data_stop = SchemaNode(LocalDateTime(), missing=drop,
                             validator=convertible_to_seconds)


class Process(AddLogger):
    """
    Base class from which all Python movers/weatherers can inherit

    It defines the base functionality for mover/weatherer.

    NOTE: Since base class is not Serializable, it does not need
    a class level _schema attribute.
    """
    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(update=['on', 'active_start', 'active_stop', 'real_data_start', 'real_data_stop'],
               save=['on', 'active_start', 'active_stop', 'real_data_start', 'real_data_stop'],
               read=['active'])

    def __init__(self, **kwargs):   # default min + max values for timespan
        """
        Initialize default Mover/Weatherer parameters

        All parameters are optional (kwargs)

        :param on: boolean as to whether the object is on or not. Default is on
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        :param real_data_start: datetime when the mover first has data (not including extrapolation)
        :param real_data_stop: datetime after which the mover has no data (not including extrapolation)
        """
        self.on = kwargs.pop('on', True)  # turn the mover on / off for the run
        self._active = self.on  # initial value

        active_start = kwargs.pop('active_start',
                                  inf_datetime.InfDateTime('-inf'))
        active_stop = kwargs.pop('active_stop',
                                 inf_datetime.InfDateTime('inf'))

        real_data_start = kwargs.pop('real_data_start',
                                  inf_datetime.InfDateTime('-inf'))
        real_data_stop = kwargs.pop('real_data_stop',
                                 inf_datetime.InfDateTime('inf'))

        self._check_active_startstop(active_start, active_stop)

        self._active_start = active_start
        self._active_stop = active_stop

                # not sure if we would ever pass this in...
        self._check_active_startstop(real_data_start, real_data_stop)

        self.real_data_start = real_data_start
        self.real_data_stop = real_data_stop

        # empty dict since no array_types required for all movers at present
        self.array_types = set()
        self.name = kwargs.pop('name', self.__class__.__name__)
        self.make_default_refs = kwargs.pop('make_default_refs', True)

    def _check_active_startstop(self, active_start, active_stop):
        # there are no swapped-argument versions of the comare operations,
        # so it seems that InfTime and MinusInfTime cannot simply be fixed
        # to work with datetime when they are on the right side of a compare
        # operation.  So we have to put this kludge in.
        if (isinstance(active_start, datetime) and
                isinstance(active_stop, (InfTime, MinusInfTime))):
            if active_stop <= active_start:
                msg = 'active_start {0} should be smaller than active_stop {1}'
                raise ValueError(msg.format(active_start, active_stop))
        else:
            if active_start >= active_stop:
                msg = 'active_start {0} should be smaller than active_stop {1}'
                raise ValueError(msg.format(active_start, active_stop))

        return True

    # Methods for active property definition
    @property
    def active(self):
        return self._active

    @property
    def active_start(self):
        return self._active_start

    @active_start.setter
    def active_start(self, value):
        self._check_active_startstop(value, self._active_stop)
        self._active_start = value

    @property
    def active_stop(self):
        return self._active_stop

    @active_stop.setter
    def active_stop(self, value):
        self._check_active_startstop(self._active_start, value)
        self._active_stop = value

    def datetime_to_seconds(self, model_time):
        """
        Put the time conversion call here - in case we decide to change it, it
        only updates here
        """
        return time_utils.date_to_sec(model_time)

    def prepare_for_model_run(self):
        """
        Override this method if a derived mover class needs to perform any
        actions prior to a model run
        """
        pass

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        sets active flag based on time_span and on flag.
        Object is active if following hold and 'on' is True:

        1. active_start <= (model_time + time_step/2) so object is on for
           more than half the timestep
        2. (model_time + time_step/2) <= active_stop so again the object is
           on for at least half the time step
           flag to true.

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        """
        if (self.active_start <=
            (model_time_datetime + timedelta(seconds=time_step/2)) and
            self.active_stop >=
            (model_time_datetime + timedelta(seconds=time_step/2)) and
            self.on):
            self._active = True
        else:
            self._active = False

    def model_step_is_done(self, sc=None):
        """
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc in here in
        subclassed movers.
        """
        pass


class PyMover(Process):
    def get_move(self, sc, time_step, model_time_datetime):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Base class returns an array of numpy.nan for delta to indicate the
        get_move is not implemented yet.

        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        All movers must implement get_move() since that's what the model calls
        """
        positions = sc['positions']
        delta = np.zeros_like(positions)
        delta[:] = np.nan

        return delta


class Mover(Process):
    def get_move(self, sc, time_step, model_time_datetime):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Base class returns an array of numpy.nan for delta to indicate the
        get_move is not implemented yet.

        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        All movers must implement get_move() since that's what the model calls
        """
        positions = sc['positions']
        delta = np.zeros_like(positions)
        delta[:] = np.nan

        return delta


class CyMover(Mover):

    def __init__(self, **kwargs):
        """
        Base class for python wrappers around cython movers.
        Uses super(CyMover,self).__init__(\*\*kwargs) to call Mover class
        __init__ method

        All cython movers (CyWindMover, CyRandomMover) are instantiated by a
        derived class, and then contained by this class in the member 'movers'.
        They will need to extract info from spill object.

        We assumes any derived class will instantiate a 'mover' object that
        has methods like: prepare_for_model_run, prepare_for_model_step,

        All kwargs passed on to super class
        """
        super(CyMover, self).__init__(**kwargs)

        # initialize variables here for readability, though self.mover = None
        # produces errors, so that is not initialized here

        self.model_time = 0
        self.positions = np.zeros((0, 3), dtype=world_point_type)
        self.delta = np.zeros((0, 3), dtype=world_point_type)
        self.status_codes = np.zeros((0, 1), dtype=status_code_type)

        # either a 1, or 2 depending on whether spill is certain or not
        self.spill_type = 0

    def prepare_for_model_run(self):
        """
        Calls the contained cython mover's prepare_for_model_run()
        """
        self.mover.prepare_for_model_run()

    def prepare_for_model_step(self, sc, time_step, model_time_datetime):
        """
        Default implementation of prepare_for_model_step(...)
         - Sets the mover's active flag if time is within specified timespan
           (done in base class Mover)
         - Invokes the cython mover's prepare_for_model_step

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        Uses super to invoke Mover class prepare_for_model_step and does a
        couple more things specific to CyMover.
        """
        super(CyMover, self).prepare_for_model_step(sc, time_step,
                                                    model_time_datetime)
        if self.active:
            uncertain_spill_count = 0
            uncertain_spill_size = np.array((0, ), dtype=np.int32)

            if sc.uncertain:
                uncertain_spill_count = 1
                uncertain_spill_size = np.array((sc.num_released, ),
                                                dtype=np.int32)

            err = self.mover.prepare_for_model_step(
                        self.datetime_to_seconds(model_time_datetime),
                        time_step, uncertain_spill_count, uncertain_spill_size)

            if err != 0:
                msg = "No available data in the time interval that is being modeled"
                self.logger.error(msg)
                raise RuntimeError(msg)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Base implementation of Cython wrapped C++ movers
        Override for things like the WindMover since it has a different
        implementation

        :param sc: spill_container.SpillContainer object
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object
        """
        self.prepare_data_for_get_move(sc, model_time_datetime)

        # only call get_move if mover is active, it is on and there are LEs
        # that have been released

        if self.active and len(self.positions) > 0:
            self.mover.get_move(self.model_time, time_step,
                                self.positions, self.delta,
                                self.status_codes, self.spill_type)

        return self.delta.view(dtype=world_point_type).reshape((-1,
                len(world_point)))

    def prepare_data_for_get_move(self, sc, model_time_datetime):
        """
        organizes the spill object into inputs for calling with Cython
        wrapper's get_move(...)

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param model_time_datetime: current model time as datetime object
        """
        self.model_time = self.datetime_to_seconds(model_time_datetime)

        # Get the data:
        try:
            self.positions = sc['positions']
            self.status_codes = sc['status_codes']
        except KeyError, err:
            raise ValueError('The spill container does not have the required'
                             'data arrays\n' + err.message)

        if sc.uncertain:
            self.spill_type = spill_type.uncertainty
        else:
            self.spill_type = spill_type.forecast

        # Array is not the same size, change view and reshape

        self.positions = \
            self.positions.view(dtype=world_point).reshape(
                                                    (len(self.positions),))
        self.delta = np.zeros(len(self.positions),
                              dtype=world_point)

    def model_step_is_done(self, sc=None):
        """
        This method gets called by the model after everything else is done
        in a time step, and is intended to perform any necessary clean-up
        operations. Subclassed movers can override this method.
        """
        if sc is not None:
            if sc.uncertain:
                if self.active:
                    try:
                        self.status_codes = sc['status_codes']
                    except KeyError, err:
                        raise ValueError('The spill container does not have'
                                         ' the required data array\n'
                                         + err.message)
                    self.mover.model_step_is_done(self.status_codes)
            else:
                if self.active:
                    self.mover.model_step_is_done()
        else:
            if self.active:
                self.mover.model_step_is_done()
