'''
release objects that define how elements are released. A Spill() objects
is composed of a release object and an ElementType
'''

import copy
from datetime import datetime, timedelta
from gnome.utilities.time_utils import asdatetime

import numpy as np

from colander import (iso8601,
                      SchemaNode, SequenceSchema,
                      drop, Bool, Int)

from gnome.persist.base_schema import ObjTypeSchema, WorldPoint, WorldPointNumpy
from gnome.persist.extend_colander import LocalDateTime
from gnome.persist.validators import convertible_to_seconds

from gnome.basic_types import world_point_type
from gnome.array_types import gat
from gnome.utilities.plume import Plume, PlumeGenerator

from gnome.outputters import NetCDFOutput
from gnome.gnomeobject import GnomeId
from gnome.environment.timeseries_objects_base import TimeseriesData,\
    TimeseriesVector
from gnome.environment.gridded_objects_base import Time


class BaseReleaseSchema(ObjTypeSchema):
    release_time = SchemaNode(
        LocalDateTime(), validator=convertible_to_seconds,
    )

class PointLineReleaseSchema(BaseReleaseSchema):
    '''
    Contains properties required for persistence
    '''
    # start_position + end_position are only persisted as WorldPoint() instead
    # of WorldPointNumpy because setting the properties converts them to Numpy
    # _next_release_pos is set when loading from 'save' file and this does have
    # a setter that automatically converts it to Numpy array so use
    # WorldPointNumpy schema for it.
    start_position = WorldPoint(
        save=True, update=True
    )
    end_position = WorldPoint(
        missing=drop, save=True, update=True
    )
    end_release_time = SchemaNode(
        LocalDateTime(), missing=drop,
        validator=convertible_to_seconds,
        save=True, update=True
    )
    num_elements = SchemaNode(Int())
    num_per_timestep = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    description = 'PointLineRelease object schema'


class ContinuousReleaseSchema(BaseReleaseSchema):
    initial_elements = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    start_position = WorldPoint(
        save=True, update=True
    )
    end_position = WorldPoint(
        missing=drop, save=True, update=True
    )
    end_release_time = SchemaNode(
        LocalDateTime(), missing=drop,
        validator=convertible_to_seconds,
        save=True, update=True
    )
    num_elements = SchemaNode(Int())
    num_per_timestep = SchemaNode(
        Int(), missing=drop, save=True, update=True
    )
    description = 'ContinuousRelease object schema'


class StartPositions(SequenceSchema):
    start_position = WorldPoint()


class SpatialReleaseSchema(BaseReleaseSchema):
    '''
    Contains properties required by UpdateWindMover and CreateWindMover
    TODO: also need a way to persist list of element_types
    '''
    description = 'SpatialRelease object schema'
    start_position = StartPositions(
        save=True, update=True
    )


class Release(GnomeId):
    """
    base class for Release classes.

    It contains interface for Release objects
    """
    _schema = BaseReleaseSchema

    def __init__(self,
                 release_time=None,
                 num_elements=0,
                 num_released=0,
                 start_time_invalid=None,
                 **kwargs):

        super(Release, self).__init__(**kwargs)

        self._num_elements = num_elements
        self.release_time = asdatetime(release_time)

        # number of new particles released at each timestep
        # set/updated by the derived Release classes at each timestep
        self.num_released = num_released

        # flag determines if the first time is valid. If the first call to
        # self.num_elements_to_release(current_time, time_step) has
        # current_time > self.release_time, then no particles are ever released
        # model start time is valid
        self.start_time_invalid = start_time_invalid
        self._prepared = False
        self.array_types.update({'positions': gat('positions'),
                                 'mass': gat('mass'),
                                 'init_mass': gat('mass')})

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release_time={0.release_time!r}, '
                'num_elements={0.num_elements}'
                ')'.format(self))

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
        return value in seconds
        '''
        return 0


class PointLineRelease(Release):
    """
    The primary spill source class  --  a release of floating
    non-weathering particles, can be instantaneous or continuous, and be
    released at a single point, or over a line.
    """
    _schema = PointLineReleaseSchema

    def __init__(self,
                 release_time=None,
                 start_position=None,
                 num_elements=None,
                 num_per_timestep=None,
                 end_release_time=None,
                 end_position=None,
                 amount=None,
                 **kwargs):
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

        :param amount=None: optional. This is the mass released in kilograms.

        :type amount: integer

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """

        if num_elements is None and num_per_timestep is None:
            num_elements = 1000
        super(PointLineRelease, self).__init__(release_time=release_time,
                                               num_elements=num_elements,
                                               **kwargs)

        if num_elements is not None and num_per_timestep is not None:
            msg = ('Either num_elements released or a release rate, defined by'
                   ' num_per_timestep must be given, not both')
            raise TypeError(msg)
        self._num_per_timestep = num_per_timestep

        # initializes internal variables: _end_release_time, _start_position,
        # _end_position
        self.end_release_time = asdatetime(end_release_time)
        self.start_position = start_position
        self.end_position = end_position

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
        if self._end_release_time is None:
            return self.release_time + timedelta(seconds=1)
        else:
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
        val = asdatetime(val)
        if val is not None and self.release_time > val:
            raise ValueError('end_release_time must be greater than '
                             'release_time')
        if val == self.release_time:
            val += timedelta(seconds=1)

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

    @property
    def start_position(self):
        return self._start_position

    @start_position.setter
    def start_position(self, val):
        '''
        set start_position and also make _delta_pos = None so it gets
        recomputed when model runs - it should be updated
        '''
        self._start_position = np.array(val,
                                        dtype=world_point_type).reshape((3, ))

    @property
    def end_position(self):
        if self._end_position is None:
            return self.start_position
        else:
            return self._end_position

    @end_position.setter
    def end_position(self, val):
        '''
        set end_position and also make _delta_pos = None so it gets
        recomputed - it should be updated

        :param val: Set end_position to val. This can be None if release is a
            point source.
        '''
        if val is not None:
            val = np.array(val, dtype=world_point_type).reshape((3, ))

        self._end_position = val

    def LE_timestep_ratio(self, ts):
        '''
        Returns the ratio
        '''
        if self.num_elements is None and self.num_per_timestep is not None:
            return self.num_per_timestep
        return 1.0 * self.num_elements / max((self.get_num_release_time_steps(ts)-1), 1)

    def get_num_release_time_steps(self, ts):
        '''
        calculates how many time steps it takes to complete the release duration
        '''
        return 1 + int(self.release_duration / ts)

    def generate_release_timeseries(self, num_ts, max_release, ts):
        '''
        Release timeseries describe release behavior as a function of time.
        _release_ts describes the number of LEs that should exist at time T
        _pos_ts describes the spill position at time T
        All use TimeseriesData objects.
        '''
        t = None
        if num_ts == 1:
            #This is a special case, when the release is short enough a single
            #timestep encompasses the whole thing.
            t = Time([self.release_time, self.end_release_time])
            num_ts = 2
        else:
            t = Time([self.release_time + timedelta(seconds=ts * step) for step in range(0,num_ts)])
            t.data[-1] = self.end_release_time
        self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                          time=t,
                                          data=np.linspace(0, max_release, num_ts).astype(int))
        lon_ts = TimeseriesData(name=self.name+'_lon_ts',
                                time=t,
                                data=np.linspace(self.start_position[0], self.end_position[0], num_ts))
        lat_ts = TimeseriesData(name=self.name+'_lat_ts',
                                time=t,
                                data=np.linspace(self.start_position[1], self.end_position[1], num_ts))
        z_ts = TimeseriesData(name=self.name+'_z_ts',
                                time=t,
                                data=np.linspace(self.start_position[2], self.end_position[2], num_ts))
        self._pos_ts = TimeseriesVector(name=self.name+'_pos_ts',
                                        time=t,
                                        variables=[lon_ts, lat_ts, z_ts])

    def rewind(self):
        self._prepared = False
        self._mass_per_le = 0
        self._release_ts = None
        self._pos_ts = None

    def prepare_for_model_run(self, ts, amount):
        '''
        :param ts: integer seconds
        :param amount: integer kilograms
        '''
        if self._prepared:
            self.rewind()
        if self.LE_timestep_ratio(ts) < 1:
            raise ValueError('Not enough LEs: Number of LEs must at least \
                be equal to the number of timesteps in the release')

        num_ts = self.get_num_release_time_steps(ts)
        max_release = 0
        if self.num_per_timestep is not None:
            max_release = self.num_per_timestep * num_ts
        else:
            max_release = self.num_elements

        self.generate_release_timeseries(num_ts, max_release, ts)
        self._prepared = True

    def num_elements_after_time(self, current_time, time_step):
        '''
        Returns the number of elements expected to exist at current_time+time_step.
        Returns 0 if prepare_for_model_run has not been called.
        :param ts: integer seconds
        :param amount: integer kilograms
        '''
        if not self._prepared:
            return 0
        return int(self._release_ts.at(None, current_time + timedelta(seconds=time_step)))

    def initialize_LEs(self, to_rel, data, current_time, time_step):
        '''
        Initializes the mass and position for num_released new LEs.
        current_time = datetime.datetime
        time_step = integer seconds
        '''
        sl = slice(-to_rel, None, 1)
        start_position = self._pos_ts.at(None, current_time)
        end_position = self._pos_ts.at(None, current_time + timedelta(seconds=time_step))
        data['positions'][sl, 0] = \
            np.linspace(start_position[0],
                        end_position[0],
                        to_rel)
        data['positions'][sl, 1] = \
            np.linspace(start_position[1],
                        end_position[1],
                        to_rel)
        data['positions'][sl, 2] = \
            np.linspace(start_position[2],
                        end_position[2],
                        to_rel)


class ContinuousRelease(Release):
    _schema = ContinuousReleaseSchema

    def __init__(self,
                 release_time=None,
                 start_position=None,
                 num_elements=None,
                 num_per_timestep=None,
                 end_release_time=None,
                 end_position=None,
                 initial_elements=None,
                 name=None):

        self.initial_release = PointLineRelease(release_time=release_time,
                                                start_position=start_position,
                                                num_elements=initial_elements)
        self.continuous = PointLineRelease(release_time=release_time,
                                           start_position=start_position,
                                           num_elements=num_elements,
                                           num_per_timestep=num_per_timestep,
                                           end_release_time=end_release_time,
                                           end_position=end_position)

        self._next_release_pos = self.start_position

        # set this the first time it is used
        self._delta_pos = None

        self.initial_done = False
        self.num_initial_released = 0
        if name:
            self.name = name

    def num_elements_to_release(self, current_time, time_step):
        num = 0
        if(self.initial_release._release(current_time, time_step) and not self.initial_done):
            self.num_initial_released += self.initial_release.num_elements_to_release(
                current_time, 1)
            num += self.initial_release.num_elements_to_release(
                current_time, 1)
        num += self.continuous.num_elements_to_release(current_time, time_step)
        return num

    def set_newparticle_positions(self, num_new_particles,
                                  current_time, time_step,
                                  data_arrays):
        cont_rel = num_new_particles
        if self.num_initial_released is not 0 and not self.initial_done:
            self.initial_release.set_newparticle_positions(
                cont_rel, current_time, 0, data_arrays)
            cont_rel -= self.num_initial_released
            self.initial_done = True
        self.continuous.set_newparticle_positions(
            cont_rel, current_time, time_step, data_arrays)

    def rewind(self):
        self.initial_release.rewind()
        self.continuous.rewind()
        self.initial_done = False

    @property
    def end_position(self):
        return self.continuous._end_position

    @end_position.setter
    def end_position(self, val):
        '''
        set end_position and also make _delta_pos = None so it gets
        recomputed - it should be updated

        :param val: Set end_position to val. This can be None if release is a
            point source.
        '''
        if val is not None:
            val = np.array(val, dtype=world_point_type).reshape((3, ))

        self.continuous._end_position = val
        self.continuous._delta_pos = None

    @property
    def end_release_time(self):
        return self.continuous._end_release_time

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
        val = asdatetime(val)
        if val is not None and self.continuous.release_time > val:
            raise ValueError('end_release_time must be greater than '
                             'release_time')

        self.continuous.end_release_time = val

    @property
    def initial_elements(self):
        return self.initial_release.num_elements

    @initial_elements.setter
    def initial_elements(self, val):
        self.initial_release.num_elements = val

    @property
    def num_elements(self):
        return self.continuous.num_elements

    @num_elements.setter
    def num_elements(self, val):
        '''
        over ride base class setter. Makes num_per_timestep None since only one
        can be set at a time
        '''
        self.continuous.num_elements = val

    @property
    def num_per_timestep(self):
        return self.continuous.num_per_timestep

    @num_per_timestep.setter
    def num_per_timestep(self, val):
        self.continuous.num_per_timestep = val

    @property
    def release_duration(self):
        return self.continuous.release_duration

    @property
    def release_time(self):
        return self.initial_release.release_time

    @release_time.setter
    def release_time(self, val):
        val = asdatetime(val)
        self.initial_release.release_time = val
        self.continuous.release_time = val

    @property
    def start_position(self):
        return self.initial_release.start_position

    @start_position.setter
    def start_position(self, val):
        '''
        set start_position and also make _delta_pos = None so it gets
        recomputed when model runs - it should be updated
        '''
        self.continuous.start_position = np.array(val,
                                                  dtype=world_point_type).reshape((3, ))
        self.initial_release.start_position = np.array(val,
                                                       dtype=world_point_type).reshape((3, ))
        self._start_position = val
        self.continuous._delta_pos = None
        self.initial_release._delta_pos = None

    @property
    def start_time_invalid(self):
        if self.initial_release.start_time_invalid is None or self.continuous.start_time_invalid is None:
            return None
        else:
            return self.initial_release.start_time_invalid

    @start_time_invalid.setter
    def start_time_invalid(self, val):
        self.initial_release.start_time_invalid = val
        self.continuous.start_time_invalid = val


class SpatialRelease(Release):
    """
    A simple release class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified
    """
    _schema = SpatialReleaseSchema

    def __init__(self,
                 release_time=None,
                 start_position=None,
                 **kwargs):
        """
        :param release_time: time the LEs are released
        :type release_time: datetime.datetime

        :param start_positions: locations the LEs are released
        :type start_positions: (num_elements, 3) numpy array of float64
            -- (long, lat, z)

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """
        super(SpatialRelease, self).__init__(release_time=release_time,**kwargs)
        self.start_position = (np.asarray(start_position,
                                          dtype=world_point_type)
                               .reshape((-1, 3)))
        self.num_elements = len(self.start_position)

    @classmethod
    def new_from_dict(cls, dict_):
        '''
            Custom new_from_dict() functionality for SpatialRelease
        '''
        if ('release_time' in dict_ and not isinstance(dict_['release_time'], datetime)):
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

        if (current_time + timedelta(seconds=time_step) <= self.release_time):
            return 0

        return self.num_elements

    def set_newparticle_positions(self, num_new_particles, current_time,
                                  time_step, data_arrays):
        """
        set positions for new elements added by the SpillContainer

        .. note:: this releases all the elements at their initial positions at
            the release_time
        """
        data_arrays['positions'][-self.num_released:] = self.start_position


def GridRelease(release_time, bounds, resolution):
    """
    Utility function that creates a SpatialRelease with a grid of elements.

    Only 2-d for now

    :param bounds: bounding box of region you want the elements in:
                   ((min_lon, min_lat),
                    (max_lon, max_lat))
    :type bounds: 2x2 numpy array or equivalent

    :param resolution: resolution of grid -- it will be a resoluiton X resolution grid
    :type resolution: integer
    """
    lon = np.linspace(bounds[0][0], bounds[1][0], resolution)
    lat = np.linspace(bounds[0][1], bounds[1][1], resolution)
    lon, lat = np.meshgrid(lon, lat)
    positions = np.c_[lon.flat, lat.flat, np.zeros((resolution * resolution),)]

    return SpatialRelease(release_time=release_time,
                          start_position=positions)


class ContinuousSpatialRelease(SpatialRelease):
    """
    continuous release of elements from specified positions
    """
    def __init__(self,
                 num_elements,
                 release_time,
                 end_release_time,
                 start_positions,
                 name="continuous spatial release"):
        """
        :param num_elements: the total number of elements to release.
                            note that this may be rounded to fit the
                            number of release points
        :type integer:

        :param release_time: the start of the release time
        :type release_time: datetime.datetime

        :param release_time: the end of the release time
        :type release_time: datetime.datetime

        :param start_positions: locations the LEs are released
        :type start_positions: (num_positions, 3) tuple or numpy array of float64
            -- (long, lat, z)

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """
        Release.__init__(release_time,
                         num_elements,
                         name)

        self._start_positions = (np.asarray(start_positions,
                                           dtype=world_point_type).reshape((-1, 3)))


    def num_elements_to_release(self, current_time, time_step):
        '''
        Return number of particles released in current_time + time_step
        '''
        return len([e for e in self._plume_elem_coords(current_time,
                                                       time_step)])

    def num_elements_to_release(self, current_time, time_step):
        num = 0
        if(self.initial_release._release(current_time, time_step) and not self.initial_done):
            self.num_initial_released += self.initial_release.num_elements_to_release(
                current_time, 1)
            num += self.initial_release.num_elements_to_release(
                current_time, 1)
        num += self.continuous.num_elements_to_release(current_time, time_step)
        return num

    def set_newparticle_positions(self,
                                  num_new_particles,
                                  current_time,
                                  time_step,
                                  data_arrays):
        '''
        Set positions for new elements added by the SpillContainer
        '''
        coords = self._start_positions
        num_rel_points = len(coords)

        # divide the number to be released by the number of release points
        # rounding down so same for each point
        num_per_point = int(num_new_particles / num_rel_points)
        coords = coords * np.zeros(num_rel_points, num_per_point, 3)
        coords.shape = (num_new_particles, 3)
        data_arrays['positions'][-num_new_particles:, :] = self.coords




class VerticalPlumeRelease(Release):
    '''
    An Underwater Plume spill class -- a continuous release of particles,
    controlled by a contained spill generator object.
    - plume model generator will have an iteration method.  This will provide
    flexible looping and list comprehension behavior.
    '''

    def __init__(self,
                 release_time=None,
                 start_position=None,
                 plume_data=None,
                 end_release_time=None,
                 **kwargs):
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
        super(VerticalPlumeRelease, self).__init__(release_time=release_time, **kwargs)

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


class InitElemsFromFile(Release):
    # fixme: This should really be a spill, not a release -- it does al of what
    # a spill does, not just the release part.
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
        self._read_data_file(filename, index, time)
        if release_time is None:
            release_time = self._init_data.pop('current_time_stamp').item()

        super(InitElemsFromFile,
              self).__init__(release_time, len(self._init_data['positions']))

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
        # if init_mass is not there, set it to mass
        # fixme: should this be a required data array?
        self._init_data.setdefault('init_mass', self._init_data['mass'].copy())

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


def release_from_splot_data(release_time, filename):
    '''
    Initialize a release object from a text file containing splots.
    The file contains 3 columns with following data:
        [longitude, latitude, num_LEs_per_splot/5000]

    For each (longitude, latitude) release num_LEs_per_splot points
    '''
    # use numpy loadtxt - much faster
    pos = np.loadtxt(filename)
    num_per_pos = np.asarray(pos[:, 2], dtype=int)
    pos[:, 2] = 0

    # 'loaded data, repeat positions for splots next'
    start_positions = np.repeat(pos, num_per_pos, axis=0)

    return SpatialRelease(release_time=release_time,
                          start_position=start_positions)
