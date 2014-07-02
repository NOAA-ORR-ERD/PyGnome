'''
release objects that define how elements are released. A Spill() objects
is composed of a release object and an ElementType
'''
import types
import copy
from datetime import timedelta

import numpy
np = numpy

from colander import (SchemaNode, drop, Bool, Int)

from gnome.persist.base_schema import ObjType, WorldPoint
from gnome.persist.extend_colander import LocalDateTime
from gnome.persist.validators import convertible_to_seconds

from gnome.basic_types import world_point_type
from gnome.utilities.plume import Plume, PlumeGenerator

from gnome.utilities.serializable import Serializable


class ReleaseSchema(ObjType):
    'Base Class for Release Schemas'
    num_elements = SchemaNode(Int(), default=1000)

    release_time = SchemaNode(LocalDateTime(),
                              validator=convertible_to_seconds)
    name = 'release'

    def __init__(self, json_='webapi', **kwargs):
        if json_ == 'save':
            # used to create a new Release object if model is persisted mid-run
            self.add(SchemaNode(Int(), name='num_released'))
            self.add(SchemaNode(Bool(), name='start_time_invalid'))

        super(ReleaseSchema, self).__init__(**kwargs)


class PointLineReleaseSchema(ReleaseSchema):
    '''
    Contains properties required by UpdateWindMover and CreateWindMover
    TODO: also need a way to persist list of element_types
    '''
    start_position = WorldPoint()
    end_position = WorldPoint(missing=drop)
    end_release_time = SchemaNode(LocalDateTime(), missing=drop,
                                  validator=convertible_to_seconds)

    # Not sure how this will work w/ WebGnome
    #prev_release_pos = WorldPoint(missing=drop)
    description = 'PointLineRelease object schema'


class Release(object):
    """
    base class for Release classes.

    It contains interface for Release objects
    """
    _update = ['num_elements', 'release_time']

    _create = ['num_released', 'start_time_invalid']
    _create.extend(_update)

    _state = copy.deepcopy(Serializable._state)
    _state.add(save=_create, update=_update,
               read=('num_released', 'start_time_invalid'))

    def __init__(self, release_time, num_elements=0, name=None):
        self.num_elements = num_elements
        self.release_time = release_time

        # number of new particles released at each timestep
        # set/updated by the derived Release classes at each timestep
        self.num_released = 0

        # flag determines if the first time is valid. If the first call to
        # self.num_elements_to_release(current_time, time_step) has
        # current_time > self.release_time, then no particles are ever released
        # if current_time <= self.release_time, then toggle this flag since
        # model start time is valid
        self.start_time_invalid = True

        if name:
            self.name = name

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release_time={0.release_time!r}, '
                'num_elements={0.num_elements}'
                ')'.format(self))

    def num_elements_to_release(self, current_time, time_step):
        """
        Determines the number of elements to be released during:
        current_time + time_step. Base class has partial (incomplete)
        implementation.

        This base class method checks if current_time in first step
        is valid and toggles the self.start_time_invalid flag if it is valid.
        If current_time <= self.release_time the first time this is called,
        then toggle start_time_invalid to True.

        Subclasses should define the complete implementation and return number
        of new particles to be released once this check passes. Be sure to call
        the base class method first if start_time_invalid flag should be
        checked.

        :param current_time: current time
        :type current_time: datetime.datetime
        :param time_step: the time step, sometimes used to decide how many
            should get released.
        :type time_step: integer seconds

        :returns: the number of elements that will be released. This is taken
            by SpillContainer to initialize all data_arrays.

        self.num_released is updated after self.set_newparticle_values is
        called. Particles are considered released after the values are set.
        """
        if self.release_time is None:
            raise ValueError("release_time attribute cannot be None")

        if (current_time <= self.release_time and
            self.start_time_invalid):
            # start time is valid
            # It's fine to release elements in subsequent steps
            self.start_time_invalid = False

        return 0    # base class does not release any particles

    def set_newparticle_positions(self, num_new_particles,
                                  current_time, time_step,
                                  data_arrays):
        """
        derived object should set the 'positions' array for the data_arrays
        base class has no implementation
        """
        pass

    def rewind(self):
        """
        rewinds the Release to original status (before anything has been
        released).

        Base class sets 'num_released'=0 and 'start_time_invalid'=True
        properties to original _state.
        Subclasses should overload for additional functions required to reset
        _state.
        """
        self.num_released = 0
        self.start_time_invalid = True

    def serialize(self, json_='webapi'):
        'define schema based on type of desired output'
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema(json_)
        serial = schema.serialize(toserial)

        return serial

    @classmethod
    def deserialize(cls, json_):
        schema = cls._schema(json_['json_'])
        return schema.deserialize(json_)


class PointLineRelease(Release, Serializable):
    """
    The primary spill source class  --  a release of floating
    non-weathering particles, can be instantaneous or continuous, and be
    released at a single point, or over a line.
    """
    _update = ['start_position', 'end_position', 'end_release_time']

    _create = []
    _create.extend(_update)

    _state = copy.deepcopy(Release._state)
    _state.add(update=_update, save=_create)

    _schema = PointLineReleaseSchema

    def __init__(self, release_time, num_elements, start_position,
                 end_position=None, end_release_time=None, name=None):
        """
        :param num_elements: total number of elements to be released
        :type num_elements: integer

        :param release_time: time the LEs are released (datetime object)
        :type release_time: datetime.datetime

        :param start_position: initial location the elements are released
        :type start_position: 3-tuple of floats (long, lat, z)

        :param end_position=None: optional. For moving source, the end position
        :type end_position: 3-tuple of floats (long, lat, z)

        :param end_release_time=None: optional -- for a release over time, the
            end release time
        :type end_release_time: datetime.datetime

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """
        super(PointLineRelease, self).__init__(release_time,
                                               num_elements,
                                               name)

        if end_release_time is None:
            # also sets self._end_release_time
            self.end_release_time = release_time
        else:
            if release_time > end_release_time:
                raise ValueError('end_release_time must be greater than '
                                 'release_time')
            self.end_release_time = end_release_time

        if self.release_time == self.end_release_time:
            self.set_newparticle_positions = \
                self._init_positions_instantaneous_release
        else:
            self.set_newparticle_positions = \
                self._init_positions_timevarying_release

        self.start_position = np.array(start_position,
                            dtype=world_point_type).reshape((3, ))
        if end_position is None:
            # also sets self._end_position
            end_position = start_position

        self.end_position = np.array(end_position,
                dtype=world_point_type).reshape((3, ))

        # only needs to be computed once
        self.delta_pos = ((self.end_position - self.start_position) /
                          max(1, self.num_elements - 1))

        self.delta_release = (self.end_release_time
                              - self.release_time).total_seconds()

        # number of new particles released at each timestep
        #self.prev_release_pos = self.start_position.copy()

    def __getstate__(self):
        '''
            Used by pickle.dump() and pickle.dumps()
            Note: Dynamically set instance methods cannot be pickled methods.
                  They should not be present in the resulting dict.
        '''
        return dict([(k, v) for k, v in self.__dict__.iteritems()
                     if type(v) != types.MethodType])

    def __setstate__(self, d):
        '''
            Used by pickle.load() and pickle.loads()
            Note: We will need to explicitly reconstruct any instance methods
                  that were dynamically set in __init__()
        '''
        self.__dict__ = d

        # reconstruct our dynamically set methods.
        if self.release_time == self.end_release_time:
            self.set_newparticle_positions = \
                self._init_positions_instantaneous_release
        else:
            self.set_newparticle_positions = \
                self._init_positions_timevarying_release

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release_time={0.release_time!r}, '
                'num_elements={0.num_elements}, '
                'start_position={0.start_position!r}, '
                'end_position={0.end_position!r}, '
                'end_release_time={0.end_release_time!r}'
                ')'.format(self))

    @property
    def end_position(self):
        return self._end_position

    @end_position.setter
    def end_position(self, val):
        if val is None:
            self._end_position = self.start_position
        else:
            self._end_position = val

    @property
    def end_release_time(self):
        return self._end_release_time

    @end_release_time.setter
    def end_release_time(self, val):
        if val is None:
            self._end_release_time = self.release_time
        else:
            self._end_release_time = val

    def num_elements_to_release(self, current_time, time_step):
        """
        return number of particles released in current_time + time_step
        """
        # call base class method to check if start_time is valid
        super(PointLineRelease, self).num_elements_to_release(current_time,
                                                              time_step)
        if self.start_time_invalid:
            return 0

        if self.num_released >= self.num_elements:
            # nothing left to release
            return 0

        # it's been called before the release_time
        if current_time + timedelta(seconds=time_step) \
            <= self.release_time:
            #print 'not time to release yet'
            return 0

        delta_release = ((self.end_release_time - self.release_time)
                         .total_seconds())
        if delta_release <= 0:
            # instantaneous release. All particles released at this timestep
            return self.num_elements

        # time varying release
        n_0 = self.num_released  # always want to start at previous released

        # index of end of current time step
        # a tiny bit to make it open on the right.
        n_1 = int(((current_time - self.release_time).total_seconds()
                   + time_step)
                  / delta_release
                  * (self.num_elements - 1))

        n_1 = min(n_1, self.num_elements - 1)  # don't want to go over the end.
        if n_1 == self.num_released - 1:  # indexes from zero
            # none to release this time step
            return 0

        # JS: not sure why we want to release 1 extra particle?
        # but leave algorithm as it is. Since n_0 = 0 at first iteration,
        # _num_new_particles at 1st step is 1 more than _num_new_particles in
        # subsequent steps for a fixed time_step
        _num_new_particles = n_1 - n_0 + 1
        return _num_new_particles

    def _init_positions_instantaneous_release(self, num_new_particles,
                                              current_time, time_step,
                                              data_arrays):
        """
        initialize all elements in the very first instant (timestep) of the run
        The particles can be released at a single 'point' or along a 'line'

        For each axis (x,y,z), it evenly spaces all elements along a line:
            np.linspace( self.start_position, self.end_position,
                         self._num_new_particles)

        If self.start_position == self.end_position, then all particles are
        released at self.start_position.
        """
        if num_new_particles == 0:
            return

        if np.all(self.start_position == self.end_position):
            # point release
            data_arrays['positions'][-num_new_particles:, :] = \
                self.start_position
        else:
            # line release
            data_arrays['positions'][-num_new_particles:, 0] = \
                np.linspace(self.start_position[0],
                            self.end_position[0],
                            num_new_particles)
            data_arrays['positions'][-num_new_particles:, 1] = \
                np.linspace(self.start_position[1],
                            self.end_position[1],
                            num_new_particles)
            data_arrays['positions'][-num_new_particles:, 2] = \
                np.linspace(self.start_position[2],
                            self.end_position[2],
                            num_new_particles)

        # expect self.num_released to be 0 before the instantaneous release
        self.num_released += num_new_particles

    def _init_positions_timevarying_release(self, num_new_particles,
                                            current_time, time_step,
                                            data_arrays):
        """
        Time varying release of particles. Initialize particles as they are
        released in each timestep
        """
        if num_new_particles == 0:
            return

        if np.all(self.start_position == self.end_position):
            # point release
            data_arrays['positions'][-num_new_particles:] = \
                self.start_position
        else:
            # continuous line release
            n_0 = self.num_released
            n_1 = self.num_released + num_new_particles
            n = np.arange(n_0, n_1).reshape((-1, 1))
            data_arrays['positions'][-num_new_particles:] = \
                self.start_position + n * self.delta_pos

        self.num_released += num_new_particles

    def rewind(self):
        """
        reset to initial conditions -- i.e. nothing released.
        """
        super(PointLineRelease, self).rewind()
        #self.prev_release_pos = self.start_position.copy()


class SpatialRelease(Release, Serializable):
    """
    A simple release class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified
    """
    _state = copy.deepcopy(Release._state)
    _state.add(update=['start_position'], save=['start_position'])

    def __init__(self, release_time, start_position, name=None):
        """
        :param release_time: time the LEs are released
        :type release_time: datetime.datetime

        :param start_positions: locations the LEs are released
        :type start_positions: (num_elements, 3) numpy array of float64
            -- (long, lat, z)

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """
        self.start_position = np.asarray(start_position,
                dtype=world_point_type).reshape((-1, 3))
        super(SpatialRelease, self).__init__(release_time,
                                             self.start_position.shape[0],
                                             name)

    def num_elements_to_release(self, current_time, time_step):
        """
        return number of particles released in current_time + time_step
        """
        # call base class method to check if start_time is valid
        super(SpatialRelease, self).num_elements_to_release(current_time,
                                                            time_step)
        if self.start_time_invalid:
            return 0

        if (self.num_released >= self.num_elements or
            current_time + timedelta(seconds=time_step) <= self.release_time):
            return 0

        return self.num_elements

    def set_newparticle_positions(self, num_new_particles, current_time,
                                  time_step, data_arrays):
        """
        set positions for new elements added by the SpillContainer

        .. note:: this releases all the elements at their initial positions at
            the release_time
        """
        self.num_released = self.num_elements
        data_arrays['positions'][-self.num_released:] = self.start_position


class VerticalPlumeRelease(Release, Serializable):
    '''
    An Underwater Plume spill class -- a continuous release of particles,
    controlled by a contained spill generator object.
    - plume model generator will have an iteration method.  This will provide
    flexible looping and list comprehension behavior.
    '''
    _state = copy.deepcopy(Release._state)

    # what kinds of customized state attributes would we like to add here?
    #_state.add(update=['start_position'], save=['start_position'])

    def __init__(self, release_time, num_elements, start_position,
                 plume_data, end_release_time, name=None):
        '''
        :param num_elements: total number of elements to be released
        :type num_elements: integer

        :param start_position: initial location the elements are released
        :type start_position: 3-tuple of floats (long, lat, z)

        :param release_time: time the LEs are released
        :type release_time: datetime.datetime

        :param start_positions: locations the LEs are released
        :type start_positions: (num_elements, 3) numpy array of float64
            -- (long, lat, z)
        '''
        super(VerticalPlumeRelease, self).__init__(release_time,
                                                num_elements, name)

        self.start_position = np.array(start_position,
                                       dtype=world_point_type).reshape((3, ))

        plume = Plume(position=start_position, plume_data=plume_data)
        time_step_delta = timedelta(hours=1).total_seconds()
        self.plume_gen = PlumeGenerator(release_time=release_time,
                                        end_release_time=end_release_time,
                                        time_step_delta=time_step_delta,
                                        plume=plume)

        if self.num_elements:
            self.plume_gen.set_le_mass_from_total_le_count(self.num_elements)

    def _plume_elem_coords(self, current_time, time_step):
        '''
        Return a list of positions for all elements released within
        current_time + time_step
        '''
        next_time = current_time + timedelta(seconds=time_step)
        elem_counts = self.plume_gen.elems_in_range(current_time, next_time)

        for coord, count in zip(self.plume_gen.plume.coords, elem_counts):
            for c in (coord,) * count:
                yield tuple(c)

    def num_elements_to_release(self, current_time, time_step):
        '''
        Return number of particles released in current_time + time_step
        '''
        return len([e for e in self._plume_elem_coords(current_time,
                                                       time_step)])

    def set_newparticle_positions(self, num_new_particles,
                                  current_time, time_step, data_arrays):
        '''
        Set positions for new elements added by the SpillContainer
        '''
        coords = [e for e in self._plume_elem_coords(current_time, time_step)]
        self.coords = np.asarray(tuple(coords),
                                 dtype=world_point_type).reshape((-1, 3))

        if self.coords.shape[0] != num_new_particles:
            raise RuntimeError('The Specified number of new particals does not'
                               ' match the number calculated from the '
                               'time range.')

        self.num_released += num_new_particles
        data_arrays['positions'][-self.coords.shape[0]:, :] = self.coords

    def rewind(self):
        '''
        Rewind to initial conditions -- i.e. nothing released.
        '''
        self.num_released = 0
        self.start_time_invalid = True
