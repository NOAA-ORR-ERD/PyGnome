import copy

import numpy
np = numpy
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


class MoverSchema(MappingSchema):
    'base mover schema - common to all movers'
    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), missing=drop,
                              validator=convertible_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), missing=drop,
                             validator=convertible_to_seconds)


class Mover(object):

    """
    Base class from which all Python movers can inherit

    It defines the interface for a Python mover. The model expects the methods
    defined here. The get_move(...) method needs to be implemented by the
    derived class.

    NOTE: Since base Mover class is not Serializable, it does not need
    a class level _schema attribute
    """

    _state = copy.deepcopy(serializable.Serializable._state)
    _state.add(update=['on', 'active_start', 'active_stop'],
              save=['on', 'active_start', 'active_stop'],
              read=['active'])

    def __init__(self, **kwargs):   # default min + max values for timespan
        """
        Initialize default Mover parameters

        All parameters are optional (kwargs)

        :param on: boolean as to whether the object is on or not. Default is on
        :param active_start: datetime when the mover should be active
        :param active_stop: datetime after which the mover should be inactive
        """

        self._active = True
        self.on = kwargs.pop('on', True)  # turn the mover on / off for the run
        active_start = kwargs.pop('active_start',
                                  inf_datetime.InfDateTime('-inf'))
        active_stop = kwargs.pop('active_stop',
                                 inf_datetime.InfDateTime('inf'))

        if active_stop <= active_start:
            msg = 'active_start {0} should be smaller than active_stop {1}'
            raise ValueError(msg.format(active_start, active_stop))

        self.active_start = active_start
        self.active_stop = active_stop

        # empty dict since no array_types required for all movers at present
        self.array_types = {}
        self.name = kwargs.pop('name', self.__class__.__name__)

    # Methods for active property definition
    @property
    def active(self):
        return self._active

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

    def prepare_for_model_step(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
        """
        sets active flag based on time_span and on flag. If
            model_time > active_start and model_time < active_stop then set
            flag to true.

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        """

        if self.active_start <= model_time_datetime \
            and self.active_stop > model_time_datetime and self.on:
            self._active = True
        else:
            self._active = False

    def get_move(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
        """
        Compute the move in (long,lat,z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Not implemented in base class
        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object
        """

        raise NotImplementedError('Each mover that derives from Mover base'
                                  ' class must implement get_move(...)')

    def model_step_is_done(self, sc=None):
        """
        This method gets called by the model when after everything else is done
        in a time step. Put any code need for clean-up, etc in here in
        subclassed movers.
        """

        pass


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

    def prepare_for_model_step(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
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

            self.mover.prepare_for_model_step(
                        self.datetime_to_seconds(model_time_datetime),
                        time_step, uncertain_spill_count, uncertain_spill_size)

    def get_move(
        self,
        sc,
        time_step,
        model_time_datetime,
        ):
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
            self.mover.get_move(
                self.model_time,
                time_step,
                self.positions,
                self.delta,
                self.status_codes,
                self.spill_type,
                )

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
