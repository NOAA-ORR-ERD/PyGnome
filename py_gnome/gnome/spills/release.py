'''
release objects that define how elements are released. A Spill() objects
is composed of a release object and an ElementType
'''

import copy
import functools
import math
import warnings
import numpy as np
import shapefile as shp
# import trimesh # making this optional
import geojson
import zipfile
import shapely

from math import ceil
from datetime import datetime, timedelta

from shapely.geometry import Polygon, Point, MultiPoint
import shapely.ops as ops

from pyproj import Proj, transform
import pyproj

from gnome.utilities.time_utils import asdatetime
import gnome.utilities.geometry.geo_routines as geo_routines


from colander import (String, SchemaNode, SequenceSchema, drop, Int, Float,
                      Boolean)

from gnome.persist.base_schema import ObjTypeSchema, WorldPoint, FeatureCollectionSchema
from gnome.persist.extend_colander import LocalDateTime, FilenameSchema
from gnome.persist.validators import convertible_to_seconds

from gnome.basic_types import world_point_type
from gnome.array_types import gat
from gnome.utilities.plume import Plume, PlumeGenerator

from gnome.outputters import NetCDFOutput
from gnome.gnomeobject import GnomeId
from gnome.environment.timeseries_objects_base import (TimeseriesData,
                                                       TimeseriesVector)
from gnome.environment.gridded_objects_base import Time

from gnome.weatherers.spreading import FayGravityViscous
from gnome.environment import Water
from gnome.constants import gravity
from gnome.exceptions import ReferencedObjectNotSet
from .initializers import (InitRiseVelFromDropletSizeFromDist,
                           InitRiseVelFromDist)

SPREADING_CUMULATIVE_TIME_SCALE = timedelta(hours=12.)

class StartPositions(SequenceSchema):
    start_position = WorldPoint()

class BaseReleaseSchema(ObjTypeSchema):
    release_time = SchemaNode(
        LocalDateTime(), validator=convertible_to_seconds,
    )
    end_release_time = SchemaNode(
        LocalDateTime(), missing=drop,
        validator=convertible_to_seconds,
        save=True, update=True
    )
    num_elements = SchemaNode(Int(), missing=drop)
    num_per_timestep = SchemaNode(Int(), missing=drop)
    release_mass = SchemaNode(
        Float()
    )
    custom_positions = StartPositions(save=True, update=True)
    centroid = WorldPoint(save=False, update=False, read_only=True)


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
    description = 'PointLineRelease object schema'


class Release(GnomeId):
    """
    base class for Release classes.

    It contains interface for Release objects
    """
    _schema = BaseReleaseSchema

    def __init__(self,
                 release_time=None,
                 num_elements=None,
                 num_per_timestep=None,
                 end_release_time=None,
                 custom_positions=None,
                 release_mass=0,
                 retain_initial_positions=False,
                 **kwargs):
        """
        Required Arguments:

        :param release_time: time the LEs are released
        :type release_time: datetime.datetime or iso string.

        :param custom_positions: initial location(s) the elements are released
        :type custom_positions: iterable of (lon, lat, z)

        Optional arguments:

        .. note:: Either num_elements or num_per_timestep must be given. If
            both are None, then it defaults to num_elements=1000. If both are
            given a TypeError is raised because user can only specify one or
            the other, not both.

        :param num_elements: total number of elements to be released
        :type num_elements: integer default 1000

        :param num_per_timestep: fixed number of LEs released at each timestep
        :type num_elements: integer

        :param end_release_time=None: optional -- for a time varying release,
            the end release time. If None, then release is instantaneous
        :type end_release_time: datetime.datetime

        :param release_mass=0: optional. This is the mass released in kilograms.
        :type release_mass: integer

        :param retain_initial_positions: Optional. If True, each LE will retain
            information about it's originally released position
        :type retain_initial_positions: boolean
        """
        self._num_elements = self._num_per_timestep = None

        if num_elements is None and num_per_timestep is None:
            num_elements = 1000
        if num_elements is not None and num_per_timestep is not None:
            msg = ('Either num_elements released or a release rate, defined by'
                   ' num_per_timestep must be given, not both')
            raise TypeError(msg)
        self._num_per_timestep = num_per_timestep

        self.num_elements = num_elements
        self.release_time = asdatetime(release_time)
        self.end_release_time = asdatetime(end_release_time)
        if self.release_time is None:
            self.release_time = datetime.now()
        self.release_mass = release_mass
        self.custom_positions = custom_positions
        self.retain_initial_positions = retain_initial_positions
        self.rewind()
        super(Release, self).__init__(**kwargs)
        self.array_types.update({'positions': gat('positions'),
                                 'mass': gat('mass'),
                                 'init_mass': gat('mass'),
                                 'density': gat('density'),
                                 'release_rate': gat('release_rate'),
                                 'bulk_init_volume': gat('bulk_init_volume'),
                                 'area': gat('area'),
                                 'fay_area': gat('fay_area'),
                                 'frac_coverage': gat('frac_coverage'),})
        if self.retain_initial_positions:
            self.array_types.update({'init_positions': gat('positions')})

        self._previously_released = 0.0
        self.cumulative_time_scale = SPREADING_CUMULATIVE_TIME_SCALE

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'release_time={0.release_time!r}, '
                'num_elements={0.num_elements}'
                ')'.format(self))

    def rewind(self):
        self._prepared = False
        self._mass_per_le = 0
        self._release_ts = None
        self._pos_ts = None
        self._previously_released = 0.0

    @property
    def centroid(self):
        if len(self.custom_positions):
            mp = shapely.geometry.MultiPoint(self.custom_positions)
            return np.array((mp.centroid.x, mp.centroid.y, 0))

    @property
    def release_mass(self):
        return self._release_mass

    @release_mass.setter
    def release_mass(self, val):
        if val is None or val < 0:
            val = 0
        self._release_mass = val

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
        if val is not None or val < 0:
            self._num_elements = None

    @property
    def num_elements(self):
        return self._num_elements

    @num_elements.setter
    def num_elements(self, val):
        '''
        over ride base class setter. Makes num_per_timestep None since only one
        can be set at a time
        '''
        if val is None:
            self._num_elements = val
            if self._num_per_timestep is None:
                self._num_per_timestep = 1
        elif val < 0:
            raise ValueError('number of elements cannot be less than 0')
        else:
            self._num_elements = val
        if val is not None:
            self._num_per_timestep = None

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
            return self.release_time
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

        self._end_release_time = val

    def LE_timestep_ratio(self, ts):
        '''
        Returns the ratio
        '''
        if self.num_elements is None and self.num_per_timestep is not None:
            return self.num_per_timestep
        return 1.0 * self.num_elements / self.get_num_release_time_steps(ts)

    def maximum_mass_error(self, ts):
        '''
        This function returns the maximum error in mass present in the model at
        any given time. In theory, this should be the mass of 1 LE
        '''
        pass

    def get_num_release_time_steps(self, ts):
        '''
        calculates how many time steps it takes to complete the release duration
        '''
        rts = int(ceil(self.release_duration / ts))
        if rts == 0:
            rts = 1
        return rts

    def generate_release_timeseries(self, num_ts, max_release, ts):
        '''
        Release timeseries describe release behavior as a function of time.
        _release_ts describes the number of LEs that should exist at time T
        PolygonRelease does not have a _pos_ts because it uses start_positions only
        All use TimeseriesData objects.
        '''
        t = None
        if num_ts == 1:
            # This is a special case, when the release is short enough a single
            # timestep encompasses the whole thing.
            if self.release_duration == 0:
                t = Time([self.release_time,
                          self.end_release_time + timedelta(seconds=1)])
            else:
                t = Time([self.release_time, self.end_release_time])
        else:
            t = Time([self.release_time + timedelta(seconds=ts * step)
                      for step in range(0, num_ts + 1)])
            t.data[-1] = self.end_release_time
        if self.release_duration == 0:
            self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                              time=t,
                                              data=np.full(t.data.shape, max_release).astype(int))
        else:
            self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                              time=t,
                                              data=np.linspace(0, max_release, num_ts + 1).astype(int))

    def num_elements_after_time(self, current_time):
        '''
        Returns the number of elements expected to exist at current_time.
        Returns 0 if prepare_for_model_run has not been called.
        :param current_time: time of release
        :type current_time: datetime
        '''
        if not self._prepared:
            return 0
        if current_time < self.release_time:
            return 0
        return int(math.ceil(self._release_ts.at(None, current_time, extrapolate=True)))

    def prepare_for_model_run(self, ts):
        '''
        :param ts: timestep as integer seconds
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
        self._mass_per_le = self.release_mass*1.0 / max_release

        if self.__class__ is Release:
            self._prepared = True

    def initialize_LEs(self, to_rel, sc, start_time, end_time): # change data to soill container (sc) 10/24/2022
        """
        set positions for new elements added by the SpillContainer

        .. note:: this releases all the elements at their initial positions at
            the end_time
        """
        if self.custom_positions is None or len(self.custom_positions) == 0:
            raise ValueError('No positions to release particles from')
        num_locs = len(self.custom_positions)
        if to_rel < num_locs:
            warnings.warn("{0} is releasing fewer LEs than number of start positions at time: {1}".format(self, end_time))

        sl = slice(-to_rel, None, 1)
        c_p = np.asarray(self.custom_positions)
        qt = to_rel // num_locs # number of times to tile self.start_positions
        rem = to_rel % num_locs # remaining LES to distribute randomly
        qt_pos = np.tile(c_p, (qt, 1))
        rem_pos = c_p[np.random.randint(0, len(c_p), rem)]
        pos = np.vstack((qt_pos, rem_pos))
        assert len(pos) == to_rel

        sc['positions'][sl] = pos
        sc['mass'][sl] = self._mass_per_le
        sc['init_mass'][sl] = self._mass_per_le

        if self.retain_initial_positions:
            sc['init_positions'][sl] = pos

    def initialize_LEs_post_substance(self, to_rel, sc, start_time, end_time, environment):

        # compute initial spreading area based terminal oil thickness
        sl = slice(-to_rel, None, 1)

        if sc.substance.is_weatherable:
           if environment['water'] is not None:
              water = environment['water']
           else:
              raise ReferencedObjectNotSet("water object not found in environment collection")

           visc = sc.substance.kvis_at_temp(temp_k=water.get('temperature'))
           thickness_limit = FayGravityViscous.get_thickness_limit(visc)

           sc['fay_area'][sl] = (sc['init_mass'][sl] / sc['density'][sl]) / thickness_limit
           sc['area'][sl] = sc['fay_area'][sl]

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
                 release_mass=0,
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

        :param release_mass=0: optional. This is the mass released in kilograms.
        :type release_mass: integer
        """

        super(PointLineRelease, self).__init__(release_time=release_time,
                                               end_release_time=end_release_time,
                                               num_elements=num_elements,
                                               release_mass = release_mass,
                                               **kwargs)

        # initializes internal variables: _end_release_time, _start_position,
        # _end_position
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
    def centroid(self):
        return self.start_position

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

    def generate_release_timeseries(self, num_ts, max_release, ts):
        '''
        Release timeseries describe release behavior as a function of time.
        _release_ts describes the number of LEs that should exist at time T
        _pos_ts describes the spill position at time T
        All use TimeseriesData objects.
        '''
        super(PointLineRelease, self).generate_release_timeseries(num_ts, max_release, ts)
        t = self._release_ts.time
        lon_ts = TimeseriesData(name=self.name+'_lon_ts',
                                time=t,
                                data=np.linspace(self.start_position[0], self.end_position[0], num_ts + 1))
        lat_ts = TimeseriesData(name=self.name+'_lat_ts',
                                time=t,
                                data=np.linspace(self.start_position[1], self.end_position[1], num_ts + 1))
        z_ts = TimeseriesData(name=self.name+'_z_ts',
                                time=t,
                                data=np.linspace(self.start_position[2], self.end_position[2], num_ts + 1))
        self._pos_ts = TimeseriesVector(name=self.name+'_pos_ts',
                                        time=t,
                                        variables=[lon_ts, lat_ts, z_ts])

    def rewind(self):
        self._prepared = False
        self._mass_per_le = 0
        self._release_ts = None
        self._pos_ts = None

    def prepare_for_model_run(self, ts):
        super(PointLineRelease, self).prepare_for_model_run(ts)
        self._prepared = True

    def initialize_LEs(self, to_rel, sc, start_time, end_time):
        '''
        Initializes the mass and position for to_rel new LEs.
        :param data: spill container with data arrays
        :param to_rel: number of elements to initialize
        :param start_time: initial time of release
        :param end_time: final time of release
        '''
        # if time_step == 0:
        #     time_step = 1  # to deal with initializing position in instantaneous release case
        if start_time == end_time:
            end_time += timedelta(seconds=1)
        sl = slice(-to_rel, None, 1)
        # if we have an interpolator -- why use linspace later?
        start_position = self._pos_ts.at(None, start_time, extrapolate=True)
        end_position = self._pos_ts.at(None, end_time, extrapolate=True)
        sc['positions'][sl, 0] = \
            np.linspace(start_position[0],
                        end_position[0],
                        to_rel)
        sc['positions'][sl, 1] = \
            np.linspace(start_position[1],
                        end_position[1],
                        to_rel)
        sc['positions'][sl, 2] = \
            np.linspace(start_position[2],
                        end_position[2],
                        to_rel)
        sc['mass'][sl] = self._mass_per_le
        sc['init_mass'][sl] = self._mass_per_le

        if self.retain_initial_positions:
            sc['init_positions'][sl] = sc['positions'][sl]

    def initialize_LEs_post_substance(self, to_rel, sc, start_time, end_time, environment):

        # compute initial spreading area based on Fay
        sl = slice(-to_rel, None, 1)

        if sc.substance.is_weatherable:
           if environment['water'] is not None:
              water = environment['water']
           else:
              raise ReferencedObjectNotSet("water object not found in environment collection")

           spread = FayGravityViscous(water=water)
           spread.prepare_for_model_run(sc)
           spread._set_init_relative_buoyancy(sc.substance)

           # compute release rate
           if self.release_duration > 0:
            sc['release_rate'][sl] = sum(sc['init_mass'][sl] / sc['density'][sl]) / (end_time-start_time).total_seconds()
           else:
            sc['release_rate'][sl] = np.nan

           if end_time <= self.release_time + self.cumulative_time_scale:
            self._previously_released = self._previously_released + sum(sc['init_mass'][sl] / sc['density'][sl])
           # compute release rate

           # change the computation of bulk_init_volume
           if not np.isnan(sc['release_rate'][sl][0]):
                sc['bulk_init_volume'][sl] = self._previously_released
           else:
                sc['bulk_init_volume'][sl] = sum(sc['init_mass'][sl] / sc['density'][sl])
           # change the computation of bulk_init_volume

           if sc['bulk_init_volume'][sl][0] > 0:
                sc['vol_frac_le_st'][sl] = (sc['init_mass'][sl] / sc['density'][sl]) / sc['bulk_init_volume'][sl]
           else:
                sc['vol_frac_le_st'][sl] = 0

           init_blob_area = spread.init_area(sc.substance.kvis_at_temp(temp_k=water.get('temperature')), spread._init_relative_buoyancy, sc['bulk_init_volume'][sl][0])

           sc['fay_area'][sl] = init_blob_area * sc['vol_frac_le_st'][sl]
           sc['area'][sl] = sc['fay_area'][sl]
        # compute initial spreading area based on Fay


class PolygonReleaseSchema(BaseReleaseSchema):
    filename = FilenameSchema(save=False, update=False, test_equal=False, missing=drop)
    features = FeatureCollectionSchema(save=True, update=True, test_equal=True, missing=drop)


class PolygonRelease(Release):
    """
    A release of elements into a set of provided polygons.

    When X particles are determined to be released, they are into the polygons
    randomly. For each LE, pick a polygon, weighted by it's proportional area
    and place the LE randomly within it. By default the PolygonRelease uses
    simple area for polygon weighting. Other classes (NESDISRelease for example)
    may use other weighting functions.
    """
    _schema = PolygonReleaseSchema

    def __init__(self,
                 filename=None,
                 features=None,
                 polygons=None,
                 weights=None,
                 thicknesses=None,
                 **kwargs):
        """
        Required Arguments:

        :param release_time: time the LEs are released (datetime object)
        :type release_time: datetime.datetime

        :param polygons: polygons to use in this release
        :type polygons: list of shapely.Polygon or shapely.MultiPolygon.

        Optional arguments:

        :param filename: (optional) shapefile
        :type filename: string name of a zip file. Polygons loaded are concatenated after polygons from kwarg

        :param weights: (optional) LE placement probability weighting for each polygon. Must be the same length as the polygons kwarg, and must sum to 1. If None, weights are generated at runtime based on area proportion.

        :param num_elements: total number of elements to be released
        :type num_elements: integer default 1000

        :param num_per_timestep: fixed number of LEs released at each timestep
        :type num_elements: integer

        :param end_release_time=None: optional -- for a time varying release, the end release time. If None, then release is instantaneous
        :type end_release_time: datetime.datetime

        :param release_mass=0: optional. This is the mass released in kilograms.
        :type release_mass: integer
        """
        if filename is not None and features is not None:
            raise ValueError('Cannot pass both a filename and FeatureCollection to PolygonRelease')
        if filename is not None:
            file_fc = geo_routines.load_shapefile(filename)
            self.features = file_fc
            self.filename = filename
        elif features is not None:
            self.features = features
        else: #construction via kwargs...need to check some possible conflicts
            if polygons is not None:
                if weights is not None:
                    if thicknesses is not None:
                        raise ValueError('Cannot use both thicknesses and weights in PolygonRelease')
                    if len(weights) != len(polygons):
                        raise ValueError('Weights must be equal in length to provided Polygons')
            else:
                raise ValueError('Must provide polygons to PolygonRelease')
            self.features = self.gen_fc_from_kwargs(
                {'polygons': polygons,
                 'weights': weights,
                 'thicknesses': thicknesses}
            )

            #attach the feature_index so webgnomeclient can associate the polygons of a Feature
            #back to the Feature
            for i, feature in enumerate(self.features[:]):
                if 'feature_index' not in feature.properties:
                    feature.properties['feature_index'] = i

        super(PolygonRelease, self).__init__(
            **kwargs
        )

    @property
    def filename(self):
        if not hasattr(self, '_filename'):
            self._filename=None
        return self._filename

    @filename.setter
    def filename(self, f):
        self._filename = f

    @property
    def centroid(self):
        return np.array([self.polygons[0].centroid.x, self.polygons[0].centroid.y, 0])

    @property
    def __geo_interface__(self):
        return self.features.__geo_interface__


    def gen_fc_from_kwargs(self, kwargs):
        fc = geojson.FeatureCollection(features=[])
        feats = [geojson.Feature(geometry=poly) for poly in kwargs.pop('polygons',[])]
        attrnames = {'thicknesses': 'thickness',
                     'weights': 'weight',
                     'names': 'name'}
        for k, v in attrnames.items():
            if kwargs.get(k, False):
                for f, val in zip(feats, kwargs.get(k)):
                    f.properties[v] = val
        fc.features = feats
        return fc

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, fc):
        self._features = fc

    @property
    def polygons(self):
        return [shapely.geometry.shape(feat.geometry) for feat in self.features[:]]

    @polygons.setter
    def polygons(self, polys):
        #polygons must be list of shapely or geojson (Multi)Polygon
        for feat, poly in zip(self.features[:], poly):
            feat.geometry = geojson.loads(geojson.dumps(poly.__geo_interface__))

    @property
    def thicknesses(self):
        rv = [feat.properties.get('thickness', None) for feat in self.features[:]]
        return None if all([r == None for r in rv]) else rv

    @thicknesses.setter
    def thicknesses(self, vals):
        if vals is None:
            for feat in self.features[:]:
                del feat.properties['thickness']
            return
        if self.weights is not None:
            raise ValueError('Cannot assign thicknesses to {} due to previously assigned weights'.format(self.name))
        for feat, t in zip(self.features[:], vals):
            feat.properties['thickness'] = t

    @property
    def weights(self):
        rv = [feat.properties.get('weight', None) for feat in self.features[:]]
        return None if all([r == None for r in rv]) else rv

    @weights.setter
    def weights(self, vals):
        if vals is None:
            for feat in self.features[:]:
                del feat.properties['weight']
            return
        if self.thicknesses is not None:
            raise ValueError('Cannot assign thicknesses to {} due to previously assigned weights'.format(self.name))
        for feat, w in zip(self.features[:], vals):
            feat.properties['weight'] = w

    @property
    def areas(self):
        return [geo_routines.geo_area_of_polygon(p) for p in self.polygons]

    def rewind(self):
        self._prepared = False
        self._mass_per_le = 0
        self._release_ts = None
        self._tris = None
        self._weights = None
        #self._pos_ts = None

    def get_polys_as_tris(self, polys, weights=None):
        #decomposes a
        _tris = []
        _weights = []
        if weights is not None:
            #user provided custom per-(multi)polygon weighting
            if len(weights) != len(polys):
                raise(ValueError('{0}:{1} Number of weights and polygons are not equal {2} vs {3}'
                .format(self.obj_type, self.name, len(weights), len(polys))))
            for p, w in zip(polys, weights):
                tris = geo_routines.triangulate_poly(p)

                #scale weight of triangles by parent poly weight
                ws = [w * tri_weight for tri_weight in geo_routines.poly_area_weight(tris)]

                _tris += tris
                _weights += ws
        else:
            #use default weight-by-area-proportion
            _tris = sum([geo_routines.triangulate_poly(p) for p in self.polygons], _tris)
            _weights = geo_routines.poly_area_weight(_tris)

        assert np.isclose(sum(_weights), 1.0)

        return _tris, _weights

    def compute_distribution(self):
        #computes polygon probability weight distribution by volume
        areas = [geo_routines.geo_area_of_polygon(p) for p in self.polygons]
        #it is possible for the areas computed above to be nans, if the polygons
        #are invalid somehow. If this is the case, raise an error
        if any(np.isnan(areas)):
            raise ValueError('Invalid polygon in {}. Area computed is NaN'.format(self.name))

        volumes = [a * t for a, t in zip(areas, self.thicknesses)]
        total_vol = sum(volumes)

        weights = [v / total_vol for v in volumes]
        return weights

    def prepare_for_model_run(self, ts):
        '''
        :param ts: timestep as integer seconds
        '''
        super(PolygonRelease, self).prepare_for_model_run(ts)
        #first a sanity check. The release only makes sense if using wgs84 (lon, lat).
        #for example nesdis files come in pseudo-mercator coordinates.

        for poly in self.polygons:
            geo_routines.check_valid_polygon(poly)

        #unless user explicitly assigned weights, compute the distribution now
        if self.thicknesses:
            weights = self.compute_distribution()
        else:
            weights = self.weights
        self._tris, self._weights = self.get_polys_as_tris(self.polygons, weights)

        self._prepared = True

    def initialize_LEs(self, to_rel, data, start_time, end_time):
        """
        set positions for new elements added by the SpillContainer

        .. note:: this releases all the elements at their initial positions at
            the end_time
        """

        sl = slice(-to_rel, None, 1)
        pts = [geo_routines.random_pt_in_tri(s) for s in np.random.choice(self._tris, to_rel, p=self._weights)]
        pts = [np.append(pt, 0) for pt in pts] #add Z coordinate

        data['positions'][sl] = pts

        if self.retain_initial_positions:
            data['init_positions'][sl] = data['positions'][sl]

        data['mass'][sl] = self._mass_per_le
        data['init_mass'][sl] = self._mass_per_le

    def get_polygons(self):
        '''
        Returns an array of lengths, and a list of line arrays.
        The first array sequentially indexes the second array.
        When the second array is split up using the first array
        and the resulting lines are drawn, you should end up with a picture of
        the polygons.
        '''
        uniq_polys = geo_routines.mixed_polys_to_polygon(self.polygons)
        polycoords = [np.array(p.exterior.xy).T.astype(np.float32) for p in uniq_polys]
        lengths = np.array([len(p) for p in polycoords]).astype(np.int32)
        # weights = self.weights if self.weights is not None else []
        # thicknesses = self.thicknesses if self.thicknesses is not None else []
        return lengths, polycoords

    def get_metadata(self):
        weights = [f.properties.get('weight',0) for f in self.features[:]]
        thicknesses = [f.properties.get('thickness',0) for f in self.features[:]]
        rw = []
        rt = []
        for p, w, t in zip(self.polygons, weights, thicknesses):
            if isinstance(p, shapely.geometry.MultiPolygon):
                for subp in p:
                    rw.append(w)
                    rt.append(t)
            else:
                rw.append(w)
                rt.append(t)

        return {'weights': rw, 'thicknesses': rt}

    @classmethod
    def new_from_dict(cls, dict_):
        if 'filename' in dict_ and 'features' in dict_:
            dict_.pop('filename')
        return super(PolygonRelease, cls).new_from_dict(dict_)

# PolygonRelease = SpatialRelease

def GridRelease(release_time, bounds, resolution):
    """
    Utility function that creates a release with a grid of elements.

    Only 2-d for now

    :param bounds: bounding box of region you want the elements in:
                   ((min_lon, min_lat),
                    (max_lon, max_lat))
    :type bounds: 2x2 numpy array or equivalent

    :param resolution: resolution of grid -- it will be a resolution X resolution grid
    :type resolution: integer
    """
    lon = np.linspace(bounds[0][0], bounds[1][0], resolution)
    lat = np.linspace(bounds[0][1], bounds[1][1], resolution)
    lon, lat = np.meshgrid(lon, lat)
    positions = np.c_[lon.flat, lat.flat, np.zeros((resolution * resolution),)]

    return Release(release_time=release_time,
                          custom_positions=positions,
                          num_elements=len(positions),
                          )


class NESDISReleaseSchema(PolygonReleaseSchema):
    thicknesses = SequenceSchema(
        SchemaNode(Float()), save=False
    )
    record_areas = SequenceSchema(
        SchemaNode(Float()), save=False, read_only=True, update=False
    )
    oil_types = SequenceSchema(
        SchemaNode(String()), save=False
    )


class NESDISRelease(PolygonRelease):
    '''
    A PolygonRelease subclass that has functions and data specifically for
    representing NESDIS shapefiles within GNOME
    '''
    _schema = NESDISReleaseSchema

    def __init__(self,
                 filename=None,
                 features=None,
                 **kwargs):
        """
        :param filename: NESDIS shapefile
        :type filename: string filename

        :param feature: FeatureCollection representation of a NESDIS shapefile
        :type feature: geojson.FeatureCollection

        """

        for kw in ('thicknesses', 'weights', 'polygons'):
            if kwargs.get(kw):
                warnings.warn('{} passed to NESDISRelease init are ignored'.format(kw))

        if filename is not None and features is not None:
            raise ValueError('Cannot pass both a filename and FeatureCollection to NESDISRelease')
        if filename is None and features is None:
            raise ValueError('''NESDISRelease must be provided a filename
                or FeatureCollection to "filename" or "features" respectively''')
        if filename is not None:
            file_fc = NESDISRelease.load_nesdis(filename, kwargs.get('release_time', None))
            features = file_fc
            self.filename = filename
            kwargs['release_time'] = datetime.fromisoformat(features[0].properties['release_time'])
            kwargs['end_release_time'] = kwargs['release_time']
        if features and 'release_time' not in kwargs:
            kwargs['release_time'] = datetime.fromisoformat(features[0].properties['release_time'])

        super(NESDISRelease, self).__init__(
            features=features,
            **kwargs
        )

    @staticmethod
    def load_nesdis(filename, release_time=None):
        '''
        1. load a nesdis shapefile
        2. Translates the time in the property array
        3. Add extra properties as necessary

        filename should be a zipfile
        returns a geojson.FeatureCollection
        '''
        fc = geo_routines.load_shapefile(filename)
        release_time = datetime.now() if release_time is None else release_time
        for i, feature in enumerate(fc.features):
            im_date = feature.properties.get('DATE', None)
            im_time = feature.properties.get('TIME', None)
            if im_date and im_time:
                parsed_time = ''.join([d for d in im_time if d.isdigit()])
                try:
                    #if time found in file, it takes precedence over supplied release_time
                    #this is necessary for webgnome (for now?)
                    release_time = datetime.strptime(im_date + ' ' + parsed_time, '%m/%d/%Y %H%M')
                except ValueError as ve:
                    warnings.warn('Could not parse shapefile time: ' + str(ve))

            #append webgnomeclient or pygnome specific properties
            feature.properties['feature_index'] = i
            feature.properties['thickness'] = 50e-6 if feature.properties.get('OILTYPE', '').lower() == 'thick' else 5e-6
            feature.properties['release_time'] = release_time.isoformat()

        return fc

    @property
    def record_areas(self):
        return self.areas

    @property
    def oil_types(self):
        return [
            feat.properties.get('OILTYPE') if 'OILTYPE' in feat.properties
            else 'Group_{}'.format(feat.properties.get('feature_index'))
            for feat in self.features[:]
            ]

    @oil_types.setter
    def oil_types(self, ot):
        for o, feat in zip(ot, self.features[:]):
            feat.properties['OILTYPE'] = o

    def to_dict(self, json_=None):
        dct = super(NESDISRelease, self).to_dict(json_=json_)
        if json_ == 'save':
            #stick the geojson in the file for now
            fc = geojson.FeatureCollection(self.polygons)
            fc.thicknesses = self.thicknesses
            fc.record_areas = self.record_areas
            fc.oil_types = self.oil_types
            dct['json_file'] = geojson.dumps(fc)
        return dct


class ContinuousPolygonRelease(PolygonRelease):
    """
    continuous release of elements from specified positions
    NOTE 3/23/2021: THIS IS NOT FUNCTIONAL
    """
    def __init__(self,
                 release_time=None,
                 start_positions=None,
                 num_elements=10000,
                 end_release_time=None,
                 LE_timeseries=None,
                 **kwargs):
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
        super(self, PolygonRelease).__init__(
            release_time=release_time,
            num_elements=num_elements,
            end_release_time=end_release_time
        )
        Release.__init__(release_time,
                         num_elements,
                         **kwargs)

        self._start_positions = (np.asarray(start_positions,
                                           dtype=world_point_type).reshape((-1, 3)))

    @property
    def release_duration(self):
        '''
        duration over which particles are released in seconds
        '''
        if self.end_release_time is None:
            return 0
        else:
            return (self.end_release_time - self.release_time).total_seconds()

    def LE_timestep_ratio(self, ts):
        '''
        Returns the ratio
        '''
        return 1.0 * self.num_elements / self.get_num_release_time_steps(ts)


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


'''
Subsurface release draft
'''
class SubsurfaceReleaseSchema(BaseReleaseSchema):
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
    description = 'SubsurfaceRelease object schema'

class SubsurfaceRelease(PointLineRelease):
    _schema = SubsurfaceReleaseSchema

    def __init__(self,
                 distribution = None,
                 distribution_type = 'droplet_size',
                 release_time=None,
                 start_position=None,
                 num_elements=None,
                 num_per_timestep=None,
                 end_release_time=None,
                 end_position=None,
                 release_mass=0,
                 **kwargs):
        """
        released as PointLinearRelease with additional features
        """

        super(SubsurfaceRelease, self).__init__(release_time=release_time,
                                               end_release_time=end_release_time,
                                               num_elements=num_elements,
                                               release_mass = release_mass,
                                               start_position = start_position,
                                               end_position = end_position,
                                               **kwargs)
        self.distribution = distribution
        self.distribution_type = distribution_type

        # remove plume_initializers method and move stuff here
        if distribution_type == 'droplet_size':
               self._init_rise_vel = InitRiseVelFromDropletSizeFromDist(distribution=distribution, **kwargs)
        elif distribution_type == 'rise_velocity':
               self._init_rise_vel = InitRiseVelFromDist(distribution=distribution,**kwargs)
        else:
               raise TypeError('distribution_type must be either droplet_size or '
                               'rise_velocity')

        self.array_types.update(self._init_rise_vel.array_types)

    def initialize_LEs_post_substance(self, to_rel, sc, start_time, end_time, environment):
        sl = slice(-to_rel, None, 1)

        Release.initialize_LEs_post_substance(self, to_rel, sc, start_time, end_time, environment)
        self._init_rise_vel.initialize(to_rel, sc, sc.substance)

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
        for key, val in self._init_data.items():
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

    return Release(release_time=release_time,
                   custom_positions=start_positions)
