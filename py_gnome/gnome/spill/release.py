'''
release objects that define how elements are released. A Spill() objects
is composed of a release object and an ElementType
'''

import copy
import math
import os
import warnings
import numpy as np
import shapefile as shp
import trimesh
import geojson
import zipfile
import tempfile
from math import ceil
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point, MultiPoint
from pyproj import Proj, transform

from gnome.utilities.time_utils import asdatetime
from gnome.utilities.geometry.geo_routines import random_pt_in_tri


from colander import (iso8601, String,
                      SchemaNode, SequenceSchema,
                      drop, Bool, Int, Float, Boolean)

from gnome.persist.base_schema import ObjTypeSchema, WorldPoint, WorldPointNumpy
from gnome.persist.extend_colander import LocalDateTime, FilenameSchema
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
    num_elements = SchemaNode(Int(), missing=drop)
    num_per_timestep = SchemaNode(Int(), missing=drop)
    release_mass = SchemaNode(
        Float()
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
                 num_elements=0,
                 release_mass=0,
                 end_release_time=None,
                 **kwargs):

        self.num_elements = num_elements
        self.release_time = asdatetime(release_time)
        self.end_release_time = asdatetime(end_release_time)
        self.release_mass = release_mass
        self.rewind()
        super(Release, self).__init__(**kwargs)
        self.array_types.update({'positions': gat('positions'),
                                 'mass': gat('mass'),
                                 'init_mass': gat('mass')})

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

    @property
    def release_mass(self):
        return self._release_mass

    @release_mass.setter
    def release_mass(self, val):
        if val is None or val < 0:
            val = 0
        self._release_mass = val

    @property
    def num_elements(self):
        return self._num_elements

    @num_elements.setter
    def num_elements(self, val):
        '''
        made it a property w/ setter/getter because derived classes may need
        to over ride the setter. See PointLineRelease() or an example
        '''
        if val < 0:
            raise ValueError('number of elements cannot be less than 0')
        self._num_elements = val

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

        num_elements and release_time passed to base class __init__ using super
        See base :class:`Release` documentation
        """

        self._num_elements = self._num_per_timestep = None

        if num_elements is None and num_per_timestep is None:
            num_elements = 1000
        super(PointLineRelease, self).__init__(release_time=release_time,
                                               end_release_time=end_release_time,
                                               num_elements=num_elements,
                                               release_mass = release_mass,
                                               **kwargs)

        if num_elements is not None and num_per_timestep is not None:
            msg = ('Either num_elements released or a release rate, defined by'
                   ' num_per_timestep must be given, not both')
            raise TypeError(msg)
        self._num_per_timestep = num_per_timestep

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


    @Release.num_elements.setter
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
        return 1.0 * self.num_elements / self.get_num_release_time_steps(ts)

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
            if self.release_duration == 0:
                t = Time([self.release_time, self.end_release_time+timedelta(seconds=1)])
            else:
                t = Time([self.release_time, self.end_release_time])
        else:
            t = Time([self.release_time + timedelta(seconds=ts * step) for step in range(0,num_ts + 1)])
            t.data[-1] = self.end_release_time
        if self.release_duration == 0:
            self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                              time=t,
                                              data=np.full(t.data.shape, max_release).astype(int))
        else:
            self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                            time=t,
                                            data=np.linspace(0, max_release, num_ts + 1).astype(int))
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
        self._mass_per_le = self.release_mass*1.0 / max_release

    def num_elements_after_time(self, current_time, time_step):
        '''
        Returns the number of elements expected to exist at current_time+time_step.
        Returns 0 if prepare_for_model_run has not been called.
        :param ts: integer seconds
        :param amount: integer kilograms
        '''
        if not self._prepared:
            return 0
        if current_time < self.release_time:
            return 0
        return int(math.ceil(self._release_ts.at(None, current_time + timedelta(seconds=time_step), extrapolate=True)))

    def initialize_LEs(self, to_rel, data, current_time, time_step):
        '''
        Initializes the mass and position for num_released new LEs.
        current_time = datetime.datetime
        time_step = integer seconds
        '''
        if(time_step == 0):
            time_step = 1 #to deal with initializing position in instantaneous release case 

        sl = slice(-to_rel, None, 1)
        start_position = self._pos_ts.at(None, current_time, extrapolate=True)
        end_position = self._pos_ts.at(None, current_time + timedelta(seconds=time_step), extrapolate=True)
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
        data['mass'][sl] = self._mass_per_le
        data['init_mass'][sl] = self._mass_per_le


class StartPositions(SequenceSchema):
    start_position = WorldPoint()


class SpatialReleaseSchema(BaseReleaseSchema):
    '''
    Contains properties required by UpdateWindMover and CreateWindMover
    TODO: also need a way to persist list of element_types
    '''
    start_position = WorldPoint(
        save=False, update=False
    )
    end_position = WorldPoint(
        save=False, update=False
    )
    random_distribute = SchemaNode(Boolean())
    filename = FilenameSchema(save=False, missing=drop, isdatafile=False, update=False, test_equal=False)
    #json_file = FilenameSchema(save=True, missing=drop, isdatafile=True, update=False, test_equal=False)
    # Because file generation on save isn't supported yet
    json_file = SchemaNode(String(), save=True, update=False, test_equal=False, missing=drop) 
    custom_positions = StartPositions(save=True, update=True)


class SpatialRelease(Release):
    """
    A simple release class  --  a release of floating non-weathering particles,
    with their initial positions pre-specified
    """
    _schema = SpatialReleaseSchema

    def __init__(self,
                 filename=None,
                 polygons=None,
                 weights=None,
                 thicknesses=None,
                 json_file=None,
                 custom_positions=None,
                 random_distribute=True,
                 num_elements=1000,
                 **kwargs):
        """
        :param filename: NESDIS shapefile
        :type filename: string or list of strings

        :param polygons: polygons to use in this release
        :type polygons: list of shapely.Polygon

        :param weights: probability weighting for each polygon. Must be the same
        length as the polygons kwarg, and must sum to 1. If None, weights are
        generated based on area proportion.

        :param start_positions: Nx3 array of release coordinates (lon, lat, z)
        :type start_positions: np.ndarray 

        :param random_distribute: If True, all LEs will always be distributed
        among all release locations. Otherwise, LEs will be equally distributed,
        and only remainder will be placed randomly

        :param num_elements: If passed as None, number of elements will be equivalent
        to number of start positions. For backward compatibility.
        """
        kwargs.pop('start_position', None)
        kwargs.pop('end_position', None)
        super(SpatialRelease, self).__init__(
            **kwargs
        )
        self.filename = None
        if filename is not None and json_file is not None:
            raise ValueError('May only provide filename or json_file to SpatialRelease')
        elif filename is not None:
            polygons, weights, thicknesses = self.__class__.load_shapefile(filename)
            self.filename = filename
        elif json_file is not None:
            polygons, weights, thicknesses = self.__class__.load_geojson(json_file)

        self.polygons = polygons
        if weights is None and self.polygons is not None:
            weights = self.gen_default_weights(self.polygons)
        if self.polygons is not None and len(weights) != len(self.polygons):
            raise ValueError('Weights must be equal in length to provided Polygons')
        
        self.thicknesses = thicknesses
        self.weights = weights
        self.random_distribute = random_distribute
        self.num_elements = num_elements
        self.custom_positions = np.array(custom_positions) if custom_positions is not None else None
        self._start_positions = self.gen_start_positions()

    @classmethod
    def load_geojson(cls, filename):
        #gj = geojson.load(filename)
        # Currently (9/16/2020), the json_file init parameter and this filename parameter
        # should always contain the raw GeoJSON. This is in lieu of developing a new
        # system to generate files when saving

        fc = geojson.FeatureCollection(geojson.loads(filename))
        weights = fc.weights
        thicknesses = fc.thicknesses
        polygons = None
        if fc.features is not None:
            polygons = map(lambda f: Polygon(f.coordinates[0]), fc.features)
        return polygons, weights, thicknesses

    @classmethod
    def load_shapefile(cls, filename):
        with zipfile.ZipFile(filename, 'r') as zsf:
            basename = ''.join(filename.split('.')[:-1])
            shpfile = filter(lambda f: f.split('.')[-1] == 'shp', zsf.namelist())
            if len(shpfile) > 0:
                shpfile = zsf.open(shpfile[0], 'r')
            else:
                raise ValueError('No .shp file found')
            dbffile = filter(lambda f: f.split('.')[-1] == 'dbf', zsf.namelist())[0]
            dbffile = zsf.open(dbffile, 'r')
            sf = shp.Reader(shp=shpfile, dbf=dbffile)
            shapes = sf.shapes()
            oil_polys = []
            oil_amounts = []
            field_names = [field[0] for field in sf.fields[1:]]
            date_id = field_names.index('DATE')
            time_id = field_names.index('TIME')
            type_id = field_names.index('OILTYPE')
            area_id = field_names.index('AREA_GEO')
            im_date = sf.record()[date_id]
            im_time = sf.record()[time_id]
            all_oil_polys = []
            all_oil_weights = []
            all_oil_thicknesses = []
            shape_oil_thickness = []

            oil_amounts = []
            for i, shape in enumerate(shapes):
                oil_type = sf.records()[i][type_id]
                oil_area = sf.records()[i][area_id] * 1000**2 #area in m2
                if oil_type == "Thin":
                    thickness = 5e-6
                else:
                    thickness = 200e-6
                oil_amounts.append(thickness * oil_area) #oil amount in cubic meters
                shape_oil_thickness.append(thickness)

            #percentage of mass in each Shape. Later this is further broken down per Polygon
            oil_amount_weights = map(lambda w: w/sum(oil_amounts), oil_amounts)

            #Each Shape contains multiple Polygons. The following extracts these Polygons
            #and determines the per Polygon weighting out of the total
            for shape, weight, thickness in zip(shapes, oil_amount_weights, shape_oil_thickness):
                shape_polys = []
                shape_amounts = []
                shape_poly_area_weights = []
                shape_poly_thickness = []
                total_poly_area = 0
                for i, start_idx in enumerate(shape.parts):
                    sl = None
                    if i < len(shape.parts) - 1:
                        sl = slice(start_idx, shape.parts[i+1])
                    else:
                        sl = slice(start_idx, None)
                    points = shape.points[sl]
                    pts = map(lambda pt: transform(Proj(init='epsg:3857'), Proj(init='epsg:4326'), pt[0], pt[1]), points)
                    poly = Polygon(pts)
                    shape_polys.append(poly)
                    shape_poly_thickness.append(thickness)

                    total_poly_area += poly.area
                areas = map(lambda s: s.area, shape_polys)
                #percentage of area each poly contributes to total shape area
                shape_poly_area_weights = map(lambda s: s/total_poly_area, areas)
                #percentage of mass each poly contributes to total mass
                oil_poly_weights = map(lambda w: w * weight, shape_poly_area_weights)
                all_oil_polys.extend(shape_polys)
                all_oil_weights.extend(oil_poly_weights)

            return all_oil_polys, all_oil_weights, all_oil_thicknesses

    @classmethod
    def from_shapefile(cls,
                       filename=None,
                       **kwargs):
        polys, weights, thicknesses = cls.load_shapefile(filename)
        return cls(
            polygons=polys,
            weights=weights,
            thicknesses = thicknesses,
            **kwargs
        )

    @property
    def json_file(self):
        #Placeholder value for the serialization system
        return None

    @property
    def start_positions(self):
        if not hasattr(self, "_start_positions") or self._start_positions is None:
            self._start_positions = self.gen_start_positions()
        return self._start_positions
    
    @start_positions.setter
    def start_positions(self, val):
        self._start_positions = val

    @property
    def start_position(self):
        if hasattr(self, '_start_positions'):
            ctr = MultiPoint(self.gen_combined_start_positions()).centroid
            return np.array([ctr.x, ctr.y, 0])
        else:
            return np.array([0, 0, 0])

    @property
    def end_position(self):
        return self.start_position

    @start_position.setter
    def start_position(self, val):
        '''
        dummy setter for web client
        '''
        pass 

    @property
    def end_release_time(self):
        if not hasattr(self, '_end_release_time') or self._end_release_time is None:
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
        
    @property
    def num_per_timestep(self):
        return None

    @num_per_timestep.setter
    def num_per_timestep(self, val):
        raise TypeError('num_per_timestep not supported on SpatialRelease')

    def LE_timestep_ratio(self, ts):
        '''
        Returns the ratio
        '''
        return 1.0 * self.num_elements / self.get_num_release_time_steps(ts)

    def gen_default_weights(self, polygons):
        if polygons is None:
            return
        tot_area = sum(map(lambda p: p.area, polygons))
        weights = map(lambda p: p.area/tot_area, polygons)
        return weights

    def gen_start_positions(self):
        if self.polygons is None:
            return
        if self.weights is None:
            self.weights = self.gen_default_weights(self.polygons)
        #generates the start positions for this release. Must be called before usage in a model
        def gen_release_pts_in_poly(num_pts, poly):
            pts, tris = trimesh.creation.triangulate_polygon(poly, engine='earcut')
            tris = map(lambda k: Polygon(k), pts[tris])
            areas = map(lambda s: s.area, tris)
            t_area = sum(areas)
            weights = map(lambda s: s/t_area, areas)
            rv = map(random_pt_in_tri, np.random.choice(tris, num_pts, p=weights))
            rv = map(lambda pt: np.append(pt, 0), rv) #add Z coordinate
            return rv
        num_pts = self.num_elements
        weights = self.weights
        polys = self.polygons
        pts_per_poly = map(lambda w: int(math.ceil(w*num_pts)), weights)
        release_pts = []
        for n, poly in zip(pts_per_poly, polys):
            release_pts.extend(gen_release_pts_in_poly(n, poly))
        return np.array(release_pts)

    def rewind(self):
        self._prepared = False
        self._mass_per_le = 0
        self._release_ts = None
        self._combined_positions = None
        #self._pos_ts = None

    def gen_combined_start_positions(self):
        self.start_positions #generates start_positions if not done already via property

        if self.start_positions is None:
            if self.custom_positions is None:
                raise ValueError('No polygons or custom positions specified, unable to generate release positions')
            else:
                return self.custom_positions
        else:
            if self.custom_positions is None:
                return self.start_positions
            else:
                return np.vstack((self.start_positions, self.custom_positions))


    def prepare_for_model_run(self, ts):
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

        if self.weights is None:
            self.weights = self.gen_default_weights(self.polygons)

        self._combined_positions = self.gen_combined_start_positions()

        if self.start_positions is None:
            if self.custom_positions is None:
                raise ValueError('No polygons or custom positions specified, unable to generate release positions')
            else:
                self._combined_positions = self.custom_positions
        else:
            if self.custom_positions is None:
                self._combined_positions = self.start_positions
            else:
                self._combined_positions = np.vstack((self.start_positions, self.custom_positions))

        self._mass_per_le = self.release_mass*1.0 / max_release
        self._prepared = True

    def generate_release_timeseries(self, num_ts, max_release, ts):
        '''
        Release timeseries describe release behavior as a function of time.
        _release_ts describes the number of LEs that should exist at time T
        SpatialRelease does not have a _pos_ts because it uses start_positions only
        All use TimeseriesData objects.
        '''
        t = None
        if num_ts == 1:
            #This is a special case, when the release is short enough a single
            #timestep encompasses the whole thing.
            if self.release_duration == 0:
                t = Time([self.release_time, self.end_release_time+timedelta(seconds=1)])
            else:
                t = Time([self.release_time, self.end_release_time])
        else:
            t = Time([self.release_time + timedelta(seconds=ts * step) for step in range(0,num_ts + 1)])
            t.data[-1] = self.end_release_time
        if self.release_duration == 0:
            self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                              time=t,
                                              data=np.full(t.data.shape, max_release).astype(int))
        else:
            self._release_ts = TimeseriesData(name=self.name+'_release_ts',
                                            time=t,
                                            data=np.linspace(0, max_release, num_ts + 1).astype(int))

    def num_elements_after_time(self, current_time, time_step):
        '''
        Returns the number of elements expected to exist at current_time+time_step.
        Returns 0 if prepare_for_model_run has not been called.
        :param ts: integer seconds
        :param amount: integer kilograms
        '''
        if not self._prepared:
            return 0
        if current_time < self.release_time:
            return 0
        return int(math.ceil(self._release_ts.at(None, current_time + timedelta(seconds=time_step), extrapolate=True)))


    def initialize_LEs(self, to_rel, data, current_time, time_step):
        """
        set positions for new elements added by the SpillContainer

        .. note:: this releases all the elements at their initial positions at
            the release_time
        """

        num_locs = len(self._combined_positions)
        if to_rel < num_locs:
            warnings.warn("{0} is releasing fewer LEs than number of start positions at time: {1}".format(self, current_time))

        sl = slice(-to_rel, None, 1)
        if self.random_distribute or to_rel < num_locs:
            data['positions'][sl] = self._combined_positions[np.random.randint(0,len(self._combined_positions), to_rel)]
        else:
            qt = num_locs / to_rel #number of times to tile self.start_positions
            rem = num_locs % to_rel #remaining LES to distribute randomly
            qt_pos = np.tile(self.start_positions, (qt, 1))
            rem_pos = self._combined_positions[np.random.randint(0,len(self._combined_positions), rem)]
            pos = np.vstack((qt_pos, rem_pos))
            assert len(pos) == to_rel
            data['positions'][sl] = pos


        data['mass'][sl] = self._mass_per_le
        data['init_mass'][sl] = self._mass_per_le

    def to_dict(self, json_=None):
        dct = super(SpatialRelease, self).to_dict(json_=json_)
        if json_ == 'save':
            #stick the geojson in the file for now
            fc = geojson.FeatureCollection(self.polygons)
            fc.weights = self.weights
            fc.thicknesses = self.thicknesses
            dct['json_file'] = geojson.dumps(fc)
        return dct


    def get_polygons(self):
        '''
        Returns an array of lengths, and a list of line arrays.
        The first array sequentially indexes the second array.
        When the second array is split up using the first array
        and the resulting lines are drawn, you should end up with a picture of
        the polygons.
        '''
        polycoords = map(lambda p: np.array(p.exterior.xy).T, self.polygons)
        lengths = map(len, polycoords)
        weights = self.weights if self.weights is not None else []
        thicknesses = self.thicknesses if self.thicknesses is not None else []
        return lengths, polycoords
    
    def get_metadata(self):
        return np.array(self.weights), np.array(self.thicknesses)

    def get_start_positions(self):
        #returns all combined start positions in binary form for the API
        return np.ascontiguousarray(self.gen_combined_start_positions().astype(np.float32))



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
        super(self, SpatialRelease).__init__(
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
                          custom_positions=start_positions)
