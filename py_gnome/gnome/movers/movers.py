
from datetime import datetime, timedelta

import numpy as np

from colander import (SchemaNode, TupleSchema, Bool, drop, String)

from gnome.basic_types import (world_point,
                               world_point_type,
                               spill_type,
                               status_code_type)

from gnome.utilities import time_utils
from gnome.persist.base_schema import ObjTypeSchema
from gnome.cy_gnome.cy_rise_velocity_mover import CyRiseVelocityMover
from gnome import GnomeId
from gnome.utilities.projections import FlatEarthProjection
from gnome.utilities.inf_datetime import InfDateTime, InfTime, MinusInfTime

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.extend_colander import LocalDateTime


class TimeRangeSchema(TupleSchema):
    start = SchemaNode(LocalDateTime(), validator=convertible_to_seconds)
    stop = SchemaNode(LocalDateTime(), validator=convertible_to_seconds)


class ProcessSchema(ObjTypeSchema):
    '''
    base Process schema - attributes common to all movers/weatherers
    defined at one place
    '''
    on = SchemaNode(Bool(), missing=drop, save=True, update=True)
    active_range = TimeRangeSchema()
    
    #flag for client weatherer management system
    _automanaged = SchemaNode(Bool(), missing=drop, save=True, update=True)


class PyMoverSchema(ProcessSchema):
    default_num_method = SchemaNode(String(), missing=drop, save=True, update=True)

class Process(GnomeId):
    """
    Base class from which all Python movers/weatherers can inherit

    It defines the base functionality for mover/weatherer.

    NOTE: Since base class is not Serializable, it does not need
          a class level _schema attribute.
    """

    def __init__(self,
                 on=True,
                 make_default_refs=True,
                 active_range=(InfDateTime('-inf'), InfDateTime('inf')),
                 _automanaged=True,
                 **kwargs):  # default min + max values for timespan
        """
        Initialize default Mover/Weatherer parameters

        All parameters are optional (kwargs)

        :param on: boolean as to whether the object is on or not. Default is on
        :param active_range: Range of datetimes for when the mover should be
                             active
        :type active_range: 2-tuple of datetimes
        """
        super(Process, self).__init__(**kwargs)

        self.on = on
        self._active = self.on

        self._check_active_startstop(*active_range)

        self._active_range = active_range
        self._automanaged = _automanaged

        # empty dict since no array_types required for all movers at present
        self.make_default_refs = make_default_refs

    def _check_active_startstop(self, active_start, active_stop):
        # there are no swapped-argument versions of the comare operations,
        # so it seems that InfTime and MinusInfTime cannot simply be fixed
        # to work with datetime when they are on the right side of a compare
        # operation.  So we have to put this kludge in.
        if (isinstance(active_start, datetime) and
                isinstance(active_stop, (InfTime, MinusInfTime))):
            if active_stop <= active_start:
                raise ValueError('active start time {0} should be smaller '
                                 'than the active stop time {1}'
                                 .format(active_start, active_stop))
        else:
            if active_start >= active_stop:
                raise ValueError('active start time {0} should be smaller '
                                 'than the active stop time {1}'
                                 .format(active_start, active_stop))

        return True

    # Methods for active property definition
    @property
    def active(self):
        return self._active

    @property
    def active_range(self):
        return self._active_range

    @active_range.setter
    def active_range(self, value):
        self._check_active_startstop(*value)
        self._active_range = value

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

        1. active start <= (model_time + time_step/2) so object is on for
           more than half the timestep
        2. (model_time + time_step/2) <= active_stop so again the object is
           on for at least half the time step
           flag to true.

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        """
        half_timestep = model_time_datetime + timedelta(seconds=time_step / 2)

        if (self.active_range[0] <= half_timestep and
                self.active_range[1] >= half_timestep and
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

    def post_model_run(self):
        """
        Override this method if a derived class needs to perform
        any actions after a model run is complete (StopIteration triggered)
        """
        pass

    @property
    def data_start(self):
        return MinusInfTime()

    @property
    def data_stop(self):
        return InfTime()


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

    def get_bounds(self):
        '''
            Return a bounding box surrounding the grid data.
        '''

        return ((-360, -90), (360, 90))


class PyMover(Mover):

    _schema = PyMoverSchema

    def __init__(self, default_num_method='RK2',
                 **kwargs):
        super(PyMover, self).__init__(**kwargs)

        self.num_methods = {'RK4': self.get_delta_RK4,
                            'Euler': self.get_delta_Euler,
                            'RK2': self.get_delta_RK2}
        self.default_num_method = default_num_method

        if 'env' in kwargs:
            if hasattr(self, '_req_refs'):
                for k, in self._req_refs:
                    for o in kwargs['env']:
                        if k in o._ref_as:
                            setattr(self, k, o)

    def delta_method(self, method_name=None):
        '''
            Returns a delta function based on its registered name

            Usage: delta = self.delta_method('RK2')(**kwargs)

            Note: We do not handle any key errors resulting from passing in
            a bad registered name.
        '''
        if method_name is None:
            method_name = self.default_num_method

        return self.num_methods[method_name]

    def get_delta_Euler(self, sc, time_step, model_time, pos, vel_field):
        vels = vel_field.at(pos, model_time)

        return vels * time_step

    def get_delta_RK2(self, sc, time_step, model_time, pos, vel_field):
        dt = timedelta(seconds=time_step)
        dt_s = dt.seconds
        t = model_time

        v0 = vel_field.at(pos, t)
        d0 = FlatEarthProjection.meters_to_lonlat(v0 * dt_s, pos)
        p1 = pos.copy()
        p1 += d0

        v1 = vel_field.at(p1, t + dt)

        return dt_s / 2 * (v0 + v1)

    def get_delta_RK4(self, sc, time_step, model_time, pos, vel_field):
        dt = timedelta(seconds=time_step)
        dt_s = dt.seconds
        t = model_time

        v0 = vel_field.at(pos, t)
        d0 = FlatEarthProjection.meters_to_lonlat(v0 * dt_s / 2, pos)
        p1 = pos.copy()
        p1 += d0

        v1 = vel_field.at(p1, t + dt / 2)
        d1 = FlatEarthProjection.meters_to_lonlat(v1 * dt_s / 2, pos)
        p2 = pos.copy()
        p2 += d1

        v2 = vel_field.at(p2, t + dt / 2)
        d2 = FlatEarthProjection.meters_to_lonlat(v2 * dt_s, pos)
        p3 = pos.copy()
        p3 += d2

        v3 = vel_field.at(p3, t + dt)

        return dt_s / 6 * (v0 + 2 * v1 + 2 * v2 + v3)


class CyMover(Mover):
    def __init__(self, **kwargs):
        """
        Base class for python wrappers around cython movers.
        Uses ``super(CyMover, self).__init__(**kwargs)`` to call Mover class
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
            uncertain_spill_size = np.array((0,), dtype=np.int32)

            if sc.uncertain:
                uncertain_spill_count = 1
                uncertain_spill_size = np.array((sc.num_released,),
                                                dtype=np.int32)

            seconds = self.datetime_to_seconds(model_time_datetime)

            try:
                self.mover.prepare_for_model_step(seconds,
                                                  time_step,
                                                  uncertain_spill_count,
                                                  uncertain_spill_size)
            except OSError as e:
                msg = ('No available data in the time interval '
                       'that is being modeled\n'
                       '\tModel time: {}\n'
                       '\tData available from {} to {}\n'
                       '\tMover: {} of type {}\n'
                       '\tError: {}'
                       .format(model_time_datetime,
                               self.data_start, self.data_stop,
                               self.name, self.__class__,
                               str(e)))

                self.logger.error(msg)
                raise RuntimeError(msg)

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Base implementation of Cython wrapped C++ movers
        Override for things like the PointWindMover since it has a different
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

        return (self.delta.view(dtype=world_point_type)
                .reshape((-1, len(world_point))))

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
        except KeyError as err:
            raise ValueError('The spill container does not have the required'
                             'data arrays\n' + str(err))

        if sc.uncertain:
            self.spill_type = spill_type.uncertainty
        else:
            self.spill_type = spill_type.forecast

        # Array is not the same size, change view and reshape
        self.positions = (self.positions.view(dtype=world_point)
                          .reshape((len(self.positions),)))

        self.delta = np.zeros(len(self.positions), dtype=world_point)

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
                    except KeyError as err:
                        raise ValueError('The spill container does not have'
                                         ' the required data array\n{}'
                                         .format(err))

                    self.mover.model_step_is_done(self.status_codes)
            else:
                if self.active:
                    self.mover.model_step_is_done()
        else:
            if self.active:
                self.mover.model_step_is_done()
