'''
release objects that define how elements are released. A Spill() objects
is composed of a release object and an ElementType
'''
import types
import copy
from datetime import datetime, timedelta

import numpy
np = numpy

from colander import (iso8601,
                      SchemaNode, SequenceSchema,
                      drop, Bool, Int)

from gnome.persist.base_schema import ObjType, WorldPoint
from gnome.persist.extend_colander import LocalDateTime
from gnome.persist.validators import convertible_to_seconds

from gnome.basic_types import world_point_type
from gnome.utilities.plume import Plume, PlumeGenerator

from gnome.utilities.serializable import Serializable
from gnome.outputters import NetCDFOutput


class BaseReleaseSchema(ObjType):
    'Base Class for Release Schemas'
    release_time = SchemaNode(LocalDateTime(),
                              validator=convertible_to_seconds)
    name = 'release'
    start_time_invalid = SchemaNode(Bool(), missing=drop)

    def __init__(self, json_='webapi', **kwargs):
        if json_ == 'save':
            # used to create a new Release object if model is persisted mid-run
            self.add(SchemaNode(Int(), name='num_released'))

        super(BaseReleaseSchema, self).__init__(**kwargs)


class ReleaseSchema(BaseReleaseSchema):
    'Base Class for Release Schemas'
    num_elements = SchemaNode(Int(), missing=drop)


class PointLineReleaseSchema(ReleaseSchema):
    '''
    Contains properties required by UpdateWindMover and CreateWindMover
    TODO: also need a way to persist list of element_types
    '''
    start_position = WorldPoint()
    end_position = WorldPoint(missing=drop)
    _next_release_pos = WorldPoint(missing=drop)
    end_release_time = SchemaNode(LocalDateTime(), missing=drop,
                                  validator=convertible_to_seconds)
    num_per_timestep = SchemaNode(Int(), missing=drop)
    description = 'PointLineRelease object schema'


class StartPositions(SequenceSchema):
    start_position = WorldPoint()


class SpatialReleaseSchema(BaseReleaseSchema):
    '''
    Contains properties required by UpdateWindMover and CreateWindMover
    TODO: also need a way to persist list of element_types
    '''
    description = 'SpatialRelease object schema'
    start_position = StartPositions()


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
        self._num_elements = num_elements
        self.release_time = release_time

        # number of new particles released at each timestep
        # set/updated by the derived Release classes at each timestep
        self.num_released = 0

        # flag determines if the first time is valid. If the first call to
        # self.num_elements_to_release(current_time, time_step) has
        # current_time > self.release_time, then no particles are ever released
        # if current_time <= self.release_time, then toggle this flag since
        # model start time is valid
        self.start_time_invalid = None

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

        if self.start_time_invalid is None:
            if (current_time > self.release_time):
                self.start_time_invalid = True
            else:
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
        self.start_time_invalid = None

    def serialize(self, json_='webapi'):
        'define schema based on type of desired output'
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema(json_)
        serial = schema.serialize(toserial)

        return serial

    @property
    def num_elements(self):
        return self._num_elements

    @num_elements.setter
    def num_elements(self, val):
        '''
        made it a property w/ setter/getter because derived classes may need
        to over ride the setter. See PointLineRelease() or an example
        '''
        self._num_elements = val

    @property
    def release_duration(self):
        '''
        returns a timedelta object defining the time over which the particles
        are released. The default is 0; derived classes like PointLineRelease
        must over-ride
        '''
        return 0

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
    _update = ['start_position', 'end_position', 'end_release_time',
               'num_per_timestep']

    _create = ['_next_release_pos']
    _create.extend(_update)

    _state = copy.deepcopy(Release._state)
    _state.add(update=_update, save=_create)

    _schema = PointLineReleaseSchema

    def __init__(self,
                 release_time,
                 start_position,
                 num_elements=None,
                 num_per_timestep=None,
                 end_release_time=None,
                 end_position=None,
                 name=None):
        """
        Required Arguments:

        :param release_time: time the LEs are released (datetime object)
        :type release_time: datetime.datetime

        :param start_position: initial location the elements are released
        :type start_position: 3-tuple of floats (long, lat, z)

        Optional arguments:

        .. note:: Either num_elements or num_per_timestep must be given. If
            both are None, then it defaults to num_elements=1000. If both are
            given a TypeError is raised because user can only specify one or
            the other, not both.

        :param num_elements: total number of elements to be released
        :type num_elements: integer

        :param num_per_timestep: fixed number of LEs released at each timestep
        :type num_elements: integer

        :param end_release_time=None: optional -- for a time varying release,
            the end release time. If None, then release is instantaneous
        :type end_release_time: datetime.datetime

        :param end_position=None: optional. For moving source, the end position
            If None, then release from a point source
        :type end_position: 3-tuple of floats (long, lat, z)

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """
        if num_elements is None and num_per_timestep is None:
            num_elements = 1000

        if num_elements is not None and num_per_timestep is not None:
            msg = ('Either num_elements released or a release rate, defined by'
                   ' num_per_timestep must be given, not both')
            raise TypeError(msg)

        super(PointLineRelease, self).__init__(release_time,
                                               num_elements,
                                               name)
        self._num_per_timestep = num_per_timestep

        # initializes internal variable: _end_release_time
        self.end_release_time = end_release_time

        # make attributes into numpy arrays
        self.start_position = np.array(start_position,
                                       dtype=world_point_type).reshape((3, ))
        if end_position is not None:
            end_position = np.array(end_position,
                                    dtype=world_point_type).reshape((3, ))
        self.end_position = end_position

        # need this for a line release
        self._next_release_pos = self.start_position

        # set this the first time it is used
        self._delta_pos = None

        self._reference_to_num_elements_to_release()

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
        self._reference_to_num_elements_to_release()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release_time={0.release_time!r}, '
                'num_elements={0.num_elements}, '
                'start_position={0.start_position!r}, '
                'end_position={0.end_position!r}, '
                'end_release_time={0.end_release_time!r}'
                ')'.format(self))

    @property
    def is_pointsource(self):
        '''
        if end_position - start_position == 0, point source
        otherwise it is a line source

        :returns: True if point source, false otherwise
        '''
        if self.end_position is None:
            return True

        if np.all(self.end_position == self.start_position):
            return True

        return False

    @property
    def release_duration(self):
        '''
        duration over which particles are released in seconds
        '''
        if self.end_release_time is None:
            return 0
        else:
            return (self.end_release_time - self.release_time).total_seconds()

    @property
    def end_release_time(self):
        return self._end_release_time

    @end_release_time.setter
    def end_release_time(self, val):
        '''
        Set end_release_time.
        If end_release_time is None or if end_release_time == release_time,
        it is an instantaneous release.

        Also update reference to set_newparticle_positions - if this was
        previously an instantaneous release but is now timevarying, we need
        to update this method
        '''
        if val is not None and self.release_time > val:
            raise ValueError('end_release_time must be greater than '
                             'release_time')

        self._end_release_time = val

    @property
    def num_per_timestep(self):
        return self._num_per_timestep

    @num_per_timestep.setter
    def num_per_timestep(self, val):
        '''
        Defines fixed number of LEs released per timestep

        Setter does the following:

        1. sets num_per_timestep attribute
        2. sets num_elements to None since total elements depends on duration
            and timestep
        3. invokes _reference_to_num_elements_to_release(), which updates the
            method referenced by num_elements_to_release
        '''
        self._num_per_timestep = val
        if val is not None:
            self._num_elements = None
        self._reference_to_num_elements_to_release()

    @Release.num_elements.setter
    def num_elements(self, val):
        '''
        over ride base class setter. Makes num_per_timestep None since only one
        can be set at a time
        '''
        self._num_elements = val
        if val is not None:
            self._num_per_timestep = None
        self._reference_to_num_elements_to_release()

    def _reference_to_num_elements_to_release(self):
        '''
        assign correct reference for num_elements_to_release
        '''
        if self.num_elements is None:
            # delta_pos must be set for each timestep
            self.num_elements_to_release = \
                self._num_to_release_given_timestep_rate
        else:
            self.num_elements_to_release = \
                self._num_to_release_given_total_elements

    def _release(self, current_time, time_step):
        """
        returns a bool (True) if elements can be released at this time, so
        following are True:

        1)  start_time is valid: so model start time is not after the
            release_time
        2)  release_time <= current_time + time_step so it isn't too early to
            release particles

        if either of above conditions fail, then return False
        """
        # call base class method to check if start_time is valid
        super(PointLineRelease, self).num_elements_to_release(current_time,
                                                              time_step)
        if self.start_time_invalid:
            return False

        # it's been called before the release_time
        if current_time + timedelta(seconds=time_step) <= self.release_time:
            return False

        return True

    def _num_to_release_given_total_elements(self, current_time, time_step):
        '''
        if _release() method returns True, then release particles.
        Release particles linearly where the rate of release is the total
        number of particles divided by release_duration. Also ensure a fraction
        of particle is not released; ie. an integer number of particles
        released.
        '''

        if not self._release(current_time, time_step):
            return 0

        if self.num_released >= self.num_elements:
            # nothing left to release
            return 0

        if self.release_duration == 0:
            return self.num_elements

        # time varying release
        n_0 = self.num_released  # always want to start at previous released

        # index of end of current time step
        # a tiny bit to make it open on the right.
        n_1 = int(((current_time - self.release_time).total_seconds()
                   + time_step)
                  / self.release_duration
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

    def _num_to_release_given_timestep_rate(self, current_time, time_step):
        '''
        if _release() method returns True, then release particles.
        In this case the number of particles released is always fixed
        regardless of timestep. If end_release_time >= current_time, then
        release num_per_timestep number of particles.
        '''
        if not self._release(current_time, time_step):
            return 0

        if self.end_release_time is None:
            return self.num_per_timestep

        elif self.end_release_time >= current_time:
            # todo: should we have above condition or the one listed below:
            # self.end_release_time <= (current_time +
            #                           timedelta(seconds=time_step)):
            return self.num_per_timestep

        else:
            return 0

    def _set_delta_pos(self, num_new_particles, current_time, time_step):
        '''
        This sets the self._delta_pos variable. For a line release, this is
        the amount by which each element is shifted in (x, y, z). This assumes
        a linear release rate and position rate over the release_duration.
        '''
        if self.is_pointsource:
            # point source should be handled correctly by
            # _get_start_end_position() - don't do anything if not line source
            return

        if self.num_elements is None:
            # for each timestep estimate number of total elements released
            # This should be fixed since we assume a linear rate; however,
            # if num_per_timestep is given and the timestep varies, then
            # total num_elements are not known and not changing at a constant
            # rate. For now, for this case just update _delta_pos at each
            # timestep.
            ts = min(time_step,
                     (self.end_release_time - current_time).total_seconds())
            frac = ((current_time + timedelta(seconds=ts) -
                     self.release_time).total_seconds()/self.release_duration)
            est_total_num_elems = (self.num_released + num_new_particles)/frac
            self._delta_pos = ((self.end_position - self.start_position) /
                               (est_total_num_elems-1))

        else:
            # if num_elements given, _delta_pos only set if it is None
            if self._delta_pos is None:
                self._delta_pos = ((self.end_position - self.start_position) /
                                   (self.num_elements - 1))

    def _get_start_end_position(self,
                                num_new_particles,
                                current_time,
                                time_step):
        '''
        returns a tuple containing (start_position, end_postion) for
        set_newparticle_positions to use for linspace

        1) Point source just returns start position
        if is_pointsource():
            return (self.start_position, self.start_position)

        2) For instantaneous release, return user defined start and end:
        if release_duration == 0,
            return (self.start_position, self.end_position)

        3) For timevarying line release, use _set_delta_pos() to set the delta
        (x, y, z) by which each particle is moved along the line. The delta_pos
        is given as (end_position - start_position)/(num_elements - 1)
        however, when num_elements is not defined, it is estimated based on
        the fraction of time elapsed and the number of elements released
        thus far.

        After computing (start_position, end_position), update
        self._next_release_position as end_position + delta_pos
        '''
        if self.is_pointsource:
            return (self.start_position, self.start_position)

        if self.release_duration == 0:
            # particles released from start to end since a delta_pos_rate is
            # not defined
            return (self.start_position, self.end_position)

        # for fixed timestep, estimate total number of particles and
        # compute delta_pos based on it
        self._set_delta_pos(num_new_particles, current_time, time_step)
        positions = (self._next_release_pos.copy(), self._next_release_pos +
                     (num_new_particles - 1) * self._delta_pos)

        if not np.allclose(positions[1], self.end_position, atol=1e-10):
            self._next_release_pos = positions[1] + self._delta_pos

        return positions

    def set_newparticle_positions(self,
                                  num_new_particles,
                                  current_time,
                                  time_step,
                                  data_arrays):
        '''
        sets positions for newly released particles. It evenly spaces the
        particles using numpy.linespace. Set (start_postion, end_position)
        depending on whether it is a point source, a line release
        an instantaneous release or a time varying release.
        '''
        if num_new_particles == 0:
            return

        # get start_position, end_position
        (start_position, end_position) = \
            self._get_start_end_position(num_new_particles,
                                         current_time, time_step)

        data_arrays['positions'][-num_new_particles:, 0] = \
            np.linspace(start_position[0],
                        end_position[0],
                        num_new_particles)
        data_arrays['positions'][-num_new_particles:, 1] = \
            np.linspace(start_position[1],
                        end_position[1],
                        num_new_particles)
        data_arrays['positions'][-num_new_particles:, 2] = \
            np.linspace(start_position[2],
                        end_position[2],
                        num_new_particles)

        self.num_released += num_new_particles


class SpatialRelease(Release, Serializable):
    """
    A simple release class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified
    """
    _update = ['start_position']

    _create = []
    _create.extend(_update)

    _state = copy.deepcopy(Release._state)
    _state.add(update=_update, save=_create)

    _schema = SpatialReleaseSchema

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

    @classmethod
    def new_from_dict(cls, dict_):
        '''
            Custom new_from_dict() functionality for SpatialRelease
        '''
        if ('release_time' in dict_ and
            not isinstance(dict_['release_time'], datetime)):
            print 'handling release_time...'
            dict_['release_time'] = iso8601.parse_date(dict_['release_time'],
                                                       default_timezone=None)

        return super(SpatialRelease, cls).new_from_dict(dict_)

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


def GridRelease(release_time, bounds, resolution):
    """
    Utility function that creates a SpatialRelease with a grid of elements.

    Only 2-d for now

    :param bounds: bounding box of region you want the elements in:
                   ((min_lon, min_lat),
                    (max_lon, max_lat))
    :type bounds: 2x2 numpy array or equivalent
    """
    lon = np.linspace(bounds[0][0], bounds[1][0], resolution)
    lat = np.linspace(bounds[0][1], bounds[1][1], resolution)
    lon, lat = np.meshgrid(lon, lat)
    positions = np.c_[lon.flat, lat.flat, np.zeros((resolution * resolution),)]

    return SpatialRelease(release_time, positions)


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


class InitElemsFromFile(Release):
    '''
    release object that sets the initial state of particles from a previously
    output NetCDF file
    '''
    def __init__(self, filename, release_time=None, index=None, time=None):
        '''
        Take a NetCDF file, which is an output of PyGnome's outputter:
        NetCDFOutput, and use these dataarrays as initial condition for the
        release. The release sets not only 'positions' but also all other
        arrays it finds. Arrays found in NetCDF file but not in the
        SpillContainer are ignored. Optional arguments, index and time can
        be used to initialize the release from any other record in the
        NetCDF file. Default behavior is to use the last record in the NetCDF
        to initialize the release elements.

        :param str filename: NetCDF file from which to initialize released
            elements

        Optional arguments:

        :param int index=None: index of the record from which to initialize the
            release elements. Default is to use -1 if neither time nor index is
            specified

        :param datetime time: timestamp at which the data is desired. Looks in
            the netcdf data's 'time' array and finds the closest time to this
            and use this data. If both 'time' and 'index' are None, use
            data for index = -1
        '''
        self._init_data = None
        self._read_data_file(filename, index, time)
        if release_time is None:
            release_time = self._init_data.pop('current_time_stamp').item()

        super(InitElemsFromFile, self).__init__(release_time,
                                                len(self._init_data['positions']))

        self.set_newparticle_positions = self._set_data_arrays

    def _read_data_file(self, filename, index, time):
        if time is not None:
            self._init_data = NetCDFOutput.read_data(filename, time,
                                                     which_data='all')[0]
        elif index is not None:
            self._init_data = NetCDFOutput.read_data(filename, index=index,
                                                     which_data='all')[0]
        else:
            self._init_data = NetCDFOutput.read_data(filename, index=-1,
                                                     which_data='all')[0]

    def num_elements_to_release(self, current_time, time_step):
        '''
        all elements should be released in the first timestep unless start time
        is invalid. Start time is invalid if it is after the Spill's
        releasetime
        '''
        super(InitElemsFromFile, self).num_elements_to_release(current_time,
                                                               time_step)
        if self.start_time_invalid:
            return 0

        return self.num_elements - self.num_released

    def _set_data_arrays(self, num_new_particles, current_time, time_step,
                         data_arrays):
        '''
        Will set positions and all other data arrays if data for them was found
        in the NetCDF initialization file.
        '''
        for key, val in self._init_data.iteritems():
            if key in data_arrays:
                data_arrays[key][-num_new_particles:] = val

        self.num_released = self.num_elements
