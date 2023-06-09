
"""
GNOME land-water map.

Maps for the GNOME ``Model`` -- maps are responsible for
determining where / when elements impact the shoreline

The module includes a base map that is all water -- no land,
As well as a few implementations of actual maps
"""

# This is a re-write of the C++ raster map approach

# Features:
#  - Map now handles spillable area and map bounds as polygons
#  - raster is the same aspect ratio as the land
#  - internally, raster is a numpy array
#  - land raster is only as big as the land -- if the map bounds are bigger,
#    extra space is not in the land map
#
# NOTES:
#  - Perhaps we just use non-projected coordinates for the raster map?
#    It makes for a little less computation at every step.


import os
import math
import warnings

import py_gd
import numpy as np
from gnome.persist import SchemaNode, String, Float, Boolean, drop

from geojson import FeatureCollection, Feature, MultiPolygon

import nucos as uc

from gnome.gnomeobject import GnomeId

from gnome.environment.gridded_objects_base import PyGrid


from gnome import _valid_units
from gnome.basic_types import oil_status, world_point_type

from gnome.utilities.projections import (FlatEarthProjection,
                                         RectangularGridProjection,
                                         RegularGridProjection)
from gnome.utilities.map_canvas import MapCanvas
from gnome.utilities.file_tools import haz_files
# from gnome.utilities.file_tools.osgeo_helpers import (ogr_layers)
# from gnome.utilities.file_tools.osgeo_helpers import (ogr_features)
# from gnome.utilities.file_tools.osgeo_helpers import (ogr_open_file)

from gnome.utilities.geometry.polygons import PolygonSet
from gnome.utilities.geometry import points_in_poly, point_in_poly
from gnome.utilities.appearance import AppearanceSchema

from gnome.cy_gnome.cy_land_check import check_land_layers, move_particles

from gnome.persist import (base_schema, SchemaNode, String, Float, Int,
                           Boolean, drop)


class GnomeMapSchema(base_schema.ObjTypeSchema):
    map_bounds = base_schema.LongLatBounds(save_reference=False)
    # .MetaDataList is not serialized at all
    spillable_area = base_schema.PolygonSetSchema(save_reference=False,
                                                  test_equal=False)
    _appearance = AppearanceSchema(test_equal=False)
    # land_polys = base_schema.PolygonSet(missing=drop)
    # crosses_dateline = SchemaNode(Boolean(), test_equal=False, read_only=True, save=False),


class ParamMapSchema(GnomeMapSchema):
    center = base_schema.WorldPoint()
    distance = SchemaNode(Int())
    bearing = SchemaNode(Int())
    units = SchemaNode(String())


# fixme: shouldn't the rasterMap have the raster size? and the raster itself?
class RasterMapSchema(GnomeMapSchema):
    pass


class MapFromBNASchema(RasterMapSchema):
    filename = SchemaNode(
        String(), isdatafile=True, test_equal=False)
    refloat_halflife = SchemaNode(Float())
    raster_size = SchemaNode(Float())
    shift_lons = SchemaNode(Int(), missing=drop)
    approximate_raster_interval = SchemaNode(Float(), save=False, update=False, read_only=True)


class MapFromUGridSchema(RasterMapSchema):
    filename = SchemaNode(
        String(), read_only=True, isdatafile=True, test_equal=False
    )
    refloat_halflife = SchemaNode(Float())


class GnomeMap(GnomeId):
    """
    The very simplest map for GNOME -- all water
    with only a bounding box for the map bounds.

    This also serves as a description of the interface and
    base class for more complex maps
    """
    _schema = GnomeMapSchema

    refloat_halflife = None  # note -- no land, so never used
    _ref_as = 'map'

    def __init__(self,
                 map_bounds=None,
                 spillable_area=None,
                 land_polys=None,
                 **kwargs):
        """
        The __init__ will be different for other implementations

        :param map_bounds: The polygon bounding the map if any elements are
                           outside the map bounds, they are removed from the
                           simulation.

        :param spillable_area: The PolygonSet bounding the spillable_area.
        :type spillable_area: Either a PolygonSet object or a list of lists
            from which a polygon set can be created. Each element in the list
            is a list of points defining a polygon.

        :param land_polys: The PolygonSet holding the land polygons
        :type land_polys: Either a PolygonSet object or a list of lists
            from which a polygon set can be created. Each element in the list
            is a list of points defining a polygon.

        Note on 'map_bounds':
            ``[(lon, lat), (lon, lat), (lon, lat), ..``
            An NX2 array of points that describe a polygon
            if no map bounds is provided -- the whole world is valid
        """
        super(GnomeMap, self).__init__(**kwargs)
        if map_bounds is not None:
            self.map_bounds = np.asarray(map_bounds,
                                         dtype=np.float64).reshape(-1, 2)
        else:
            # using -360 to 360 to allow stuff to cross the dateline..
            self.map_bounds = np.array(((-360, -90), (-360, 90),
                                        (360, 90), (360, -90)),
                                       dtype=np.float64)

        self.spillable_area = spillable_area
        self.land_polys = land_polys

    def get_polygons(self):
        polys = {}
        polys['spillable_area'] = self.spillable_area
        polys['map_bounds'] = self.map_bounds
        polys['land_polys'] = self.land_polys
        return polys


    # def _polygon_set_from_points(self, poly):
    #     # hmm -- this doesn't seem to be used at all
    #     # fixme: is this not in the geometry module??
    #     #.       but maybe should get rid of the PolygonSet object anyway
    #     '''
    #     create PolygonSet() object from list of polygons which in turn is a
    #     list of points
    #     :returns: PolygonSet() object
    #     '''
    #     x = PolygonSet()
    #     for p in poly:
    #         x.append(p)
    #     return x

    # does not seem to be used anywhere
    # def _attr_array_to_dict(self, np_array):
    #     '''
    #     convert np_array to list of tuples, used for map_bounds,
    #     spillable_area

    #     Fixme: why is this needed? numpy arrays are already a
    #            sequence of sequences.
    #            And why is this called _to_dict ???
    #     '''
    #     return list(map(tuple, np_array.tolist()))

    def _attr_from_list_to_array(self, l_):
        '''
        dict returned as list of tuples to be converted to numpy array
        Again used to update_from_dict map_bounds and spillable_area
        '''
        return np.asarray(l_, dtype=np.float64).reshape(-1, 2)

    @property
    def map_bounds(self):
        """
        Bounds of the map -- an NX2 array of float describing a polygon.

        [(lon, lat), (lon, lat), ...]
        """
        return self._map_bounds

    @map_bounds.setter
    def map_bounds(self, mb):
        if mb is None:
            mb = np.array(((-360, -90), (-360, 90),
                           (360, 90), (360, -90)),
                          dtype=np.float64)
        self._map_bounds = np.array(mb)

    def get_map_bounding_box(self):
        """
        return a bounding box of the map

        needed because map_bounds can be a polygon

        bounding box is a 2x2 tuple:

        ((left, bottom), (right, top))
        """

        bounds = self._map_bounds
        longs = bounds[:, 0]
        lats = bounds[:, 1]
        left, right = longs.min(), longs.max()
        bottom, top = lats.min(), lats.max()

        return ((left, bottom), (right, top))

    @property
    def spillable_area(self):
        """
        List of polygons defining the region in which it is
        legal to set spills
        """
        return self._spillable_area

    @spillable_area.setter
    def spillable_area(self, sa):
        # it's OK to have No spillable area
        # if sa is None or (isinstance(sa, list) and len(sa) == 0):
        #     sa = np.array([[(-360, -90), (-360, 90),
        #                    (360, 90), (360, -90)]],
        #                    dtype=np.float64)

        if sa is None:
            self._spillable_area = None
        elif isinstance(sa, PolygonSet):
            self._spillable_area = sa
        else:  # make it a PolygonSet
            # fixme: this should be in the geometry package
            #        -- or stop using PolygonSet
            ps = PolygonSet()
            try:  #
                sa[0][0][0]
            except TypeError:
                # probably a single polygon -- put it in a list
                sa = [sa]
            for poly in sa:
                ps.append(poly)
            self._spillable_area = ps

    @property
    def land_polys(self):
        """
        Sequence of polygons surrounding land -- mostly used
        for rendering.
        """
        return self._land_polys

    @land_polys.setter
    def land_polys(self, sa):
        if sa is None or (isinstance(sa, list) and len(sa) == 0):
            sa = PolygonSet()
        if isinstance(sa, PolygonSet):
            self._land_polys = sa
            return
        ps = PolygonSet()
        try:
            sa[0][0][0]
        except TypeError:
            # probably a single polygon -- put it in a list
            sa = [sa]
        for poly in sa:
            ps.append(poly)
        self._land_polys = ps

    def on_map(self, coords):
        """
        Determine whether points are on the map or not

        :param coords: location for test.
        :type coords: NX3 numpy array of floats: (long, lat, depth) or
                      something that can be turned into one.

        :return: bool array: True if the location is on the map,
                             False otherwise

        Note:
          coords are 3-d, but the concept of "on the map" is 2-d in this context,
          so depth is ignored.
        """
        coords = np.asarray(coords, dtype=world_point_type)

        return points_in_poly(self.map_bounds, coords)

    def on_land(self, coord):
        """
        :param coord: location for test.
        :type coord: 3-tuple of floats: (long, lat, depth)

        :return:
         - Always returns False-- no land in this implementation
        """
        return False

    def in_water(self, coords):
        """
        Determines if the points are in water (rather than on land)

        :param coords: location for test.
        :type coords: 3-tuple of floats: (long, lat, depth)
                      or an Nx3 array

        :returns:
         - True if the point is in the water,
         - False if the point is on land (or off map?)

         This implementation has no land, so always True in on the map.
        """
        return self.on_map(coords)

    def allowable_spill_position(self, coord):
        """
        :param coord: location for test.
        :type coord: 3-tuple of floats: (long, lat, depth)

        :return:
         - True if the point is an allowable spill position
         - False if the point is not an allowable spill position

        .. note:: it could be either off the map, or in a location that
                  spills aren't allowed
        """
        if self.spillable_area is not None:
            for poly in self.spillable_area:
                if point_in_poly(poly.points, coord):
                    return True
        else:
            if point_in_poly(self.map_bounds, coord):
                return True
        return False

    def _set_off_map_status(self, spill):
        """
        Determines which elements moved off the map

        Called by beach_elements after checking for land-hits

        :param spill: current SpillContainer
        :type spill:  :class:`gnome.spill_container.SpillContainer`
        """
        next_positions = spill['next_positions']
        status_codes = spill['status_codes']
        off_map = np.logical_not(self.on_map(next_positions))
        if len(next_positions) != 0 and np.all(off_map):
            self.logger.warning("All particles left the map this timestep.")

        # let model decide if we want to remove elements marked as off-map
        status_codes[off_map] = oil_status.off_maps

    def beach_elements(self, sc, model_time=None):
        """
        Determines which LEs were or weren't beached or moved off_map.
        status_code is changed to oil_status.off_maps if off the map.

        Called by the model in the main time loop, after all movers have acted.

        :param spill: current SpillContainer
        :type spill:  :class:`gnome.spill_container.SpillContainer`

        This map class has no land, so only the map check and
        resurface_airborn elements is done: noting else changes.

        subclasses that override this probably want to make sure that:

        self.resurface_airborne_elements(spill)
        self._set_off_map_status(spill)

        are called.
        """
        self.resurface_airborne_elements(sc)
        self._set_off_map_status(sc)

    def refloat_elements(self, spill_container, time_step, model_time=None):
        """
        This method performs the re-float logic -- changing the element
        status flag, and moving the element to the last known water position

        :param spill_container: current SpillContainer
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`

        .. note::
            This map class has no land, and so is a no-op.
        """
        pass

    def resurface_airborne_elements(self, spill_container):
        """
        Takes any elements that are left above the water surface (z < 0.0)
        and puts them on the surface (z == 0.0)

        :param spill_container: current SpillContainer
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`

        .. note::
            While this shouldn't occur according to the physics we're modeling,
            some movers may push elements up too high, or multiple movers may
            add vertical movement that adds up to over the surface. e.g rise
            velocity.
        """
        next_positions = spill_container['next_positions']

        np.maximum(next_positions[:, 2], 0.0, out=next_positions[:, 2])
        return None

    def to_geojson(self):
        return FeatureCollection([])


class ParamMap(GnomeMap):

    _schema = ParamMapSchema

    _valid_dist_units = _valid_units("Length")

    def __init__(self, center=(0.0, 0.0), distance=30000, bearing=90,
                 units=None, **kwargs):
        """
        Creates a parameratized map, essentially a straight shoreline set a
        certain distance and bearing from a location, usually a spill.
        It is up to the user to put the spill and the map center in the same
        location.

        Required arguments:

        :param center: tuple of coordinates describing the center point
                       of the map

        :param distance: The distance in meters the closest point on the
                         shoreline is from the center

        :param bearing: The bearing the closest point on the shoreline is
                        from the center.
        """
        refloat_halflife = kwargs.pop('refloat_halflife', 1)
        self._refloat_halflife = refloat_halflife * 3600
        self.build(center, distance, bearing, units)

    def build(self, center, distance, bearing, units):
        self.units = units if units is not None else 'm'
        self._check_units(self.units)

        if units != 'm':
            self._distance = uc.convert("Length", self.units, 'm', distance)
        else:
            self._distance = distance

        if self._distance < 30:
            raise ValueError("Distance must cover at least 1 second arc")

        if abs(center[0]) > 360 or abs(center[1]) > 90:
            raise ValueError("Center must be within (-360,-90) to (360,90)")

        # basically, direction vector to shore
        self.center = center = (center[0], center[1], 0)
        self.bearing = bearing

        map_dist = (self._distance, 0.0, 0)
        d = FlatEarthProjection.meters_to_lonlat(map_dist, center)[0][0]

        init_points = [(d, -16 * d), (d, 16 * d),
                       (16 * d, 16 * d), (16 * d, -16 * d)]

        ang = np.deg2rad(90 - bearing)
        rot_matrix = [(np.cos(ang), np.sin(ang)), (-np.sin(ang), np.cos(ang))]

        self.land_points = np.dot(init_points, rot_matrix)
        self.land_points = np.array([(x + center[0], y + center[1])
                                     for (x, y) in self.land_points])

        land_polys = PolygonSet((self.land_points, [0, 4], []))
        land_polys._MetaDataList = [('polygon', '1', '1')]

        map_bounds = np.array(((-16 * d, -16 * d), (-16 * d, 16 * d),
                               (16 * d, 16 * d), (16 * d, -16 * d)),
                              dtype=np.float64) + (center[0], center[1])

        GnomeMap.__init__(self, map_bounds=map_bounds, land_polys=land_polys)

    @property
    def distance(self):
        return uc.convert('Length', 'm', self.units, self._distance)

    def _check_units(self, units):
        """
        Checks the user provided units are in list of valid volume
        or mass units
        """
        if units in self._valid_dist_units:
            return True
        else:
            ex = uc.InvalidUnitError((units, 'Length'))
            self.logger.exception(ex, exc_info=True)

            # this should be raised since run will fail otherwise
            raise ex

    def get_map_bounds(self):
        return (self.map_bounds[0], self.map_bounds[2])

    def get_land_polygon(self):
        poly = PolygonSet((self.land_points, [0, 4], []))
        poly._MetaDataList = [('polygon', '1', '1')]

        return poly

    def on_map(self, coord):
        """
        :param coord: location for test.
        :type coord: 3-tuple of floats: (long, lat, depth)

        :return: bool array: True if the location is on the map,
                             False otherwise

        Note:
          coord is 3-d, but the concept of "on the map" is 2-d in this context,
          so depth is ignored.
        """
        return points_in_poly(self.map_bounds, coord)

    def on_land(self, coord):
        """
        :param coord: location for test.
        :type coord: 3-tuple of floats: (long, lat, depth) or Nx3 numpy array

        :return:
         - Always returns False-- no land in this implementation
        """
        return self.on_map(coord) and points_in_poly(self.land_points, coord)

    def in_water(self, coord):
        """
        :param coord: location for test.
        :type coord: 3-tuple of floats: (long, lat, depth)

        :returns:
         - True if the point is in the water,
         - False if the point is on land (or off map?)

         This implementation has no land, so always True in on the map.
        """
        return self.on_map(coord) and not self.on_land(coord)

    def allowable_spill_position(self, coord):
        """
        :param coord: location for test.
        :type coord: 3-tuple of floats: (long, lat, depth)

        :return:
         - True if the point is an allowable spill position
         - False if the point is not an allowable spill position

        .. note:: it could be either off the map, or in a location that
                  spills aren't allowed
        """
        if (coord == self.center):
            return True
        else:
            # Only allowable location for spill is the center
            # that this map was built with
            return False

    def _set_off_map_status(self, spill):
        """
        Determines which LEs moved off the map

        Called by beach_elements after checking for land-hits

        :param spill: current SpillContainer
        :type spill:  :class:`gnome.spill_container.SpillContainer`
        """
        next_positions = spill['next_positions']
        status_codes = spill['status_codes']
        off_map = np.logical_not(self.on_map(next_positions))

        # let model decide if we want to remove elements marked as off-map
        status_codes[off_map] = oil_status.off_maps

    def find_last_water_pos(self, starts, ends):
        return starts + (ends - starts) * 0.000001

    def beach_elements(self, sc, model_time=None):
        """
        Determines which LEs were or weren't beached or moved off_map.
        status_code is changed to oil_status.off_maps if off the map.

        Called by the model in the main time loop, after all movers have acted.

        :param sc: current SpillContainer
        :type sc:  :class:`gnome.spill_container.SpillContainer`

        This map class does not use a raster map for collision detection. Since
        the land is so simple the collisions are detected via intersection.
        """
        self.resurface_airborne_elements(sc)
        self._set_off_map_status(sc)

        start_pos = sc['positions']
        next_pos = sc['next_positions']
        status_codes = sc['status_codes']
        last_water_positions = sc['last_water_positions']

        # beached = 1xN numpy array of bool, elem is true if on water
        # and next pos is on land
        move_particles(start_pos, next_pos, status_codes,
                       last_water_positions, self.land_points)

        self._set_off_map_status(sc)

        # todo: need a prepare_for_model_run() so map adds these keys to
        #     mass_balance as opposed to SpillContainer
        # update 'off_maps'/'beached' in mass_balance
        sc.mass_balance['beached'] = \
            sc['mass'][sc['status_codes'] == oil_status.on_land].sum()
        sc.mass_balance['off_maps'] += \
            sc['mass'][sc['status_codes'] == oil_status.off_maps].sum()

    def refloat_elements(self, spill_container, time_step, model_time=None):
        """
        This method performs the re-float logic -- changing the element
        status flag, and moving the element to the last known water position

        :param spill_container: the current spill container
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`
        """
        r_idx = np.where(spill_container['status_codes'] ==
                         oil_status.on_land)[0]

        if r_idx.size == 0:  # no particles on land
            return

        if self._refloat_halflife > 0.0:
            # if 0.0, then r_idx is all of them -- they will all refloat.
            # refloat particles based on probability
            refloat_probability = 1.0 - 0.5 ** (float(time_step) /
                                                self._refloat_halflife)
            rnd = np.random.uniform(0, 1, len(r_idx))

            # subset of indices that will refloat
            # maybe we should rename refloat_probability since
            # rnd <= refloat_probability to
            # refloat, maybe call it stay_on_land_probability
            r_idx = r_idx[np.where(rnd <= refloat_probability)[0]]
        elif self._refloat_halflife < 0.0:
            # fake for nothing gets refloated.
            r_idx = np.array((), bool)

        if r_idx.size > 0:
            # check is not required, but why do this operation if no particles
            # need to be refloated
            spill_container['positions'][r_idx] = \
                spill_container['last_water_positions'][r_idx]
            spill_container['status_codes'][r_idx] = oil_status.in_water

    def update_from_dict(self, data):
        if ('center' in data or
            'distance' in data or
            'bearing' in data or
            'units' in data):
            self.build(
                data.pop('center', self.center),
                data.pop('distance', self.distance),
                data.pop('bearing', self.bearing),
                data.pop('units', self.units)
            )
        else:
            # for the case when instantiating a param map using new_from_dict,
            # since new_from_dict will call update_from_dict
            super(ParamMap, self).update_from_dict(data)

    def to_geojson(self):
        shoreline_geo = [p.points.tolist() for p in self.land_polys]

        shoreline = Feature(id="1",
                            properties={'name': 'Shoreline Polys'},
                            geometry=MultiPolygon(coordinates=[shoreline_geo]))

        stuff = FeatureCollection([shoreline])

        return stuff


class RasterMap(GnomeMap):
    """
    A land water map implemented as a raster

    This one uses a numpy array of uint8, so there are 8 bits to choose from...

    It requires a constant refloat half-life in hours

    This will usually be initialized in a sub-class (from a BNA, etc)
    NOTE: Nothing new added to _state attribute for serialization
    """
    # NOTE: spillable area can be both larger and smaller than land raster:
    #       map bounds can also be larger or smaller:
    #            both are done with a point in polygon check
    #       if map is smaller than land polygons, no need for raster to be
    #       larger than map -- but no impimented yet.

    # flags for what's in the raster
    # in theory -- it could be used for other data:
    #  refloat, other properties?
    # note the BW map_canvas only does 1, though.
    seconds_in_hour = 60 * 60

    land_flag = 1

    def __init__(self,
                 raster=None,
                 projection=None,
                 refloat_halflife=1,
                 **kwargs):
        """
        create a new RasterMap

        :param raster: A numpy array that stores the land-water map
                             0 is water. 1 is land. In theory, other values
                             could be used for other purposes. If the array
                             is not C-contiguous, it will be copied to a
                             C-contiguus array.
        :type raster: a (W,H) numpy array of type uint8

        :param projection: A Projection object -- used to convert from
                           lat-long to pixels in the array
        :type projection: :class:`gnome.map_canvas.Projection`

        Optional arguments (kwargs)

        :param refloat_halflife: The halflife for refloating off land
                                 -- assumed to be the same for all land.
                                 0.0 means all refloat every time step
                                 < 0.0 means never re-float.
        :type refloat_halflife: float. Units are hours

        :param map_bounds: The polygon bounding the map -- could be larger
                           or smaller than the land raster
        :type map_bounds: (N,2) numpy array of floats

        :param spillable_area: The polygon bounding the spillable_area
        :type spillable_area: (N,2) numpy array of floats

        :param id: unique ID of the object. Using UUID as a string.
                   This is only used when loading object from save file.

        :type id: string
        """
        super(RasterMap, self).__init__(**kwargs)
        self._refloat_halflife = refloat_halflife * self.seconds_in_hour

        if raster is None:
            self.raster = np.zeros((1024, 1024))
        else:
            self.raster = np.ascontiguousarray(raster)

        self.projection = projection

    def build_coarser_rasters(self):
        """
        Builds the list which contains the different resolution raster maps.
        Scale -> raster
        example for base map of 1024 x 1024:
        0 -> 1/16th raster 64 base cells per cell
        1 -> 1/32nd raster 32:1
        2 -> 1/64th raster 16:1
        3 -> 1/128th raster 8:1
        4 -> 1/256th raster 4:1
        5 -> 1/512th raster 2:1
        6 -> 1/1024th raster (== base map 1:1)

        The general idea is that the particle position (an int) can quickly
        be mapped into any scale and the path can begin from there.
        For example, if your path begins offshore and ends in a narrow inlet,
        your scale might begin on the 32:1 map.
        But as soon as the path crosses into the (32:1) raster cell containing
        the inlet (which will register as a land cell on that raster),
        the scale will decrease to 4:1, when the cell is completely water.
        In the end, if the scale decreases to 1:1 and there's still a land hit,
        then land was hit.
        """
        self.logger.info('generating coarser rasters')
        self.layers = []
        base_w = self.raster.shape[0]
        base_h = self.raster.shape[1]

        for ratio in self.ratios[:-1]:
            genned_layer = np.zeros((int(math.ceil(float(base_w) / ratio)),
                                     int(math.ceil(float(base_h) / ratio))),
                                    dtype=np.uint8, order='C')

            for j in range(0, genned_layer.shape[1]):
                for i in range(0, genned_layer.shape[0]):
                    genned_layer[i, j] = np.any(self.raster[
                                                i * ratio:(i + 1) * ratio,
                                                j * ratio:(j + 1) * ratio
                                                ])

            self.layers.append(genned_layer)

        self.layers.append(self.raster)
        # self.layers = np.array(self.layers)
        # print("the layers array:", self.layers)

    @property
    def ratios(self):
        if self._ratios is None:
            self._ratios = (16, 1)
        return self._ratios

    @ratios.setter
    def ratios(self, r):
        self._ratios = r
        self.build_coarser_rasters()

    @property
    def raster(self):
        return self._raster

    @raster.setter
    def raster(self, arr):
        if arr.size > 16000000:
            self._ratios = np.array((128, 32, 1,), dtype=np.int32)
        elif arr.size > 1000000:
            self._ratios = np.array((32, 1,), dtype=np.int32)
        else:
            self._ratios = np.array((16, 1,), dtype=np.int32)

        self._raster = np.ascontiguousarray(arr)
        self.build_coarser_rasters()

    @property
    def refloat_halflife(self):
        return self._refloat_halflife / self.seconds_in_hour

    @refloat_halflife.setter
    def refloat_halflife(self, value):
        self._refloat_halflife = value * self.seconds_in_hour

    @property
    def approximate_raster_interval(self):
        orig, diag = self.projection.to_lonlat(np.array([(0, 0), (1, 1)]))
        orig = np.r_[orig, 0]  # make (lat, lon) into (lat, lon, z)
        diag = np.r_[diag, 0]
        diff = self.projection.lonlat_to_meters(orig - diag, orig)

        return np.average(np.abs(diff))

    def save_as_image(self, filename):
        '''
        Save the land-water raster as a PNG save_as_image

        :param filename: the name of the file to save to.
        '''
        raster = self.raster.copy()

        # change anything not zero to 255 - to get black and white
        np.putmask(raster, self.raster > 0, 2)

        im = py_gd.from_array(raster)

        im.save(filename, 'bmp')

    def _off_raster(self, coord):
        """
        are these pixel coordinates on the raster

        We can't just use an IndexError, as negative
        indexes can be legal with numpy, but aren't expected here.
        """
        shape = self.raster.shape
        return (coord[0] < 0 or
                coord[1] < 0 or
                coord[0] >= shape[0] or
                coord[1] >= shape[1])

    def _on_land_pixel(self, coord):
        """
        returns 1 if the point is on land, 0 otherwise

        :param coord: pixel coordinates of point of interest
        :type coord: tuple: (row, col)

        .. note:: Only used internally or for testing -- no need for external
                  API to use pixel coordinates.
        """
        # if pixel coords are negative, then off the raster,
        # so can't be on land
        if self._off_raster(coord):
            return False
        else:
            return self.raster[coord[0], coord[1]] & self.land_flag

    def on_land(self, coord):
        """
        :param coord: (long, lat, depth) location -- depth is ignored here.
        :type coord: 3-tuple of floats -- (long, lat, depth)

        :return:
         - 1 if point on land
         - 0 if not on land

        .. note:: to_pixel() converts to array of points...
        """
        return self._on_land_pixel(self.projection.to_pixel(coord,
                                                            asint=True)[0])
    # does not appear to be used anymore
    # def _on_land_pixel_array(self, coords):
    #     """
    #     determines which LEs are on land

    #     :param coords:  pixel coords matching the raster
    #     :type coords:  Nx2 numpy int array

    #     returns: a (N,) array of bools - true for particles that are on land
    #     """
    #     mask = list(map(point_in_poly, [self.map_bounds] * len(coords), coords))
    #     racpy = np.copy(coords)[mask]
    #     mskgph = self.raster[racpy[:, 0], racpy[:, 1]]

    #     chrmgph = np.array([0] * len(coords))
    #     chrmgph[np.array(mask)] = mskgph

    #     return chrmgph

    def _in_water_pixel(self, coord):
        # if  off the raster, so must be in water,
        # unless not on map, which should have already been checked.
        if not self._off_raster:
            return True
        else:
            return not self.raster[coord[0], coord[1]] & self.land_flag

    def in_water(self, coord):
        """
        checks if it's on the map, first.
            (depth is ignored in this version)

        :param coord: (lon, lat, depth) coordinate

        :return: true if the point given by coord is in the water
        """
        if not self.on_map(coord):
            return False
        else:
            # to_pixel makes a NX2 array
            return self._in_water_pixel(self.projection.to_pixel(coord,
                                                                 asint=True)[0]
                                        )
    def beach_elements(self, sc, model_time=None):
        """
        Determines which elements were or weren't beached.

        Any that are beached have the beached flag set, and a
        "last known water position" (lkwp) is computed

        This version uses a modified Bresenham algorithm to find out
        which pixels the LE may have crossed.

        :param sc: the current spill container
        :type sc:  :class:`gnome.spill_container.SpillContainer`
            It must have the following data arrays:
            ('prev_position', 'positions', 'last_water_pt', 'status_code')
        """

        self.resurface_airborne_elements(sc)

        # pull the data from the sc
        # Is the last water point the same as the previous position? why not??
        # If beached, it won't move, if not, then we can use it?

        start_pos = sc['positions']
        next_pos = sc['next_positions']
        status_codes = sc['status_codes']
        last_water_positions = sc['last_water_positions']

        # transform to pixel coords:
        # NOTE: must be integers!

        start_pos_pixel = self.projection.to_pixel(start_pos, asint=True)
        next_pos_pixel = self.projection.to_pixel(next_pos, asint=True)
        last_water_pos_pixel = self.projection.to_pixel(last_water_positions,
                                                        asint=True)

        # call the actual hit code:
        # the status_code and last_water_point arrays are altered in-place
        # only check the ones that aren't already beached?
        self._check_land_layers(self.layers, self.ratios,
                                start_pos_pixel, next_pos_pixel,
                                status_codes, last_water_pos_pixel)

        # transform the points back to lat-long.
        beached = status_codes == oil_status.on_land
        next_pos[beached, :2] = \
            self.projection.to_lonlat(next_pos_pixel[beached])
        last_water_positions[beached, :2] = \
            self.projection.to_lonlat(last_water_pos_pixel[beached, :2])

        self._set_off_map_status(sc)

        # todo: need a prepare_for_model_run() so map adds these keys to
        #     mass_balance as opposed to SpillContainer
        # update 'off_maps'/'beached' in mass_balance
        sc.mass_balance['beached'] = \
            sc['mass'][sc['status_codes'] == oil_status.on_land].sum()
        sc.mass_balance['off_maps'] += \
            sc['mass'][sc['status_codes'] == oil_status.off_maps].sum()

    def refloat_elements(self, spill_container, time_step, model_time=None):
        """
        This method performs the re-float logic -- changing the element
        status flag, and moving the element to the last known water position

        :param spill_container: the current spill container
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`
        """
        # index into array of particles on_land

        r_idx = np.where(spill_container['status_codes'] ==
                         oil_status.on_land)[0]

        if r_idx.size == 0:  # no particles on land
            return

        if self._refloat_halflife > 0.0:
            # if 0.0, then r_idx is all of them -- they will all refloat.
            # refloat particles based on probability

            refloat_probability = 1.0 - 0.5 ** (float(time_step) /
                                                self._refloat_halflife)
            rnd = np.random.uniform(0, 1, len(r_idx))

            # subset of indices that will refloat
            # maybe we should rename refloat_probability since
            # rnd <= refloat_probability to
            # refloat, maybe call it stay_on_land_probability
            r_idx = r_idx[np.where(rnd <= refloat_probability)[0]]
        elif self._refloat_halflife < 0.0:
            # fake for nothing gets refloated.
            r_idx = np.array((), bool)

        if r_idx.size > 0:
            # check is not required, but why do this operation if no particles
            # need to be refloated
            spill_container['positions'][r_idx] = \
                spill_container['last_water_positions'][r_idx]
            spill_container['status_codes'][r_idx] = oil_status.in_water

    def _check_land_layers(self, raster_map_layers, ratios,
                           positions, end_positions,
                           status_codes, last_water_positions):
        """
        Do the actual land-checking.
        This method simply calls a Cython version:
            gnome.cy_gnome.cy_land_check.check_land()

        The arguments 'status_codes', 'positions' and 'last_water_positions'
        are altered in place.
        """
        check_land_layers(raster_map_layers, ratios,
                          positions, end_positions,
                          status_codes, last_water_positions)

    def allowable_spill_position(self, coord):
        """
        Returns true if the spill position is in the allowable spill area

        .. note::
            This may not be the same as in_water!

        :param coord: (lon, lat, depth) coordinate
        """
        if self.on_map(coord):
            if not self.on_land(coord):
                return (super(RasterMap, self).allowable_spill_position(coord))
            else:
                return False
        else:
            return False

    def to_pixel_array(self, coords):
        """
        Projects an array of (lon, lat) tuples onto the raster,
        and modifies it in place to hold the corresponding projected values.

        :param coords:  a numpy array of (lon, lat, depth) points

        :return: a numpy array of (x, y) pixel values
        """
        return self.projection.to_pixel(coords)


def ShiftLon360(points):
    try:
        points[points[:,0]<0,0] = points[:,0]+360
    except ValueError:
        pass
    return points

def ShiftLon180(points):
    try:
        points[points[:,0]>180,0] = points[:,0]-360
    except ValueError:
        pass
    return points

class MapFromBNA(RasterMap):
    """
    A raster land-water map, created from file with polygons in it.

    Currently only support BNA, but could be shapefile, or ???
    """
    _schema = MapFromBNASchema

    def __init__(self,
                 filename,
                 raster_size=4096 * 4096,
                 map_bounds=None,
                 spillable_area=None,
                 shift_lons=0,
                 **kwargs):
        """
        Creates a RasterMap from a data file.
        It is expected that you will get the spillable area and map bounds
        from the data file -- if they exist

        Required arguments:

        :param filename: full path to the data file

        :param raster_size: the total number of pixels (bytes) to make the
                            raster -- the actual size will match the
                            aspect ratio of the bounding box of the land
        :type raster_size: integer

        :param shiftLons: shift longitudes to be in -180 to 180 coords or 0 to 360.
                          180, or 360 are valid inputs
        :type shiftLons: integer

        Optional arguments (kwargs):

        :param refloat_halflife: the half-life (in hours) for the re-floating.

        :param map_bounds: The polygon bounding the map -- could be larger or
                           smaller than the land raster

        :param spillable_area: The polygon bounding the spillable_area

        :param id: unique ID of the object. Using UUID as a string.
                   This is only used when loading object from save file.
        :type id: string
        """
        self.filename = filename
        self._raster_size = raster_size
        self.shift_lons = shift_lons

        # fixme: do some file type checking here.
        polygons = haz_files.ReadBNA(filename, 'PolygonSet')

        if kwargs.get('name', False):
            self.name = os.path.split(filename)[1]

        # find the spillable area and map bounds:
        # and create a new polygonset without them
        #  fixme -- adding a "pop" method to PolygonSet might be better
        #      or a gnome_map_data object...

        land_polys = PolygonSet()  # and lakes....
        spillable_area_bna = PolygonSet()

        #add if based on input param
        tf = ShiftLon360 if shift_lons == 360 else ShiftLon180 if shift_lons == 180 else None
        if tf is not None:
            polygons.TransformData(tf)
            if map_bounds:
                map_bounds = tf(np.array(map_bounds)).tolist()

        for p in polygons:
            if p.metadata[1].lower().replace(' ', '') == 'spillablearea':
                spillable_area_bna.append(p)

            elif p.metadata[1].lower().replace(' ', '') == 'mapbounds':
                if map_bounds is not None:
                    warnings.warn('Provided map bounds superscede map bounds found in file. Please double check.')
                else:
                    map_bounds = p
            else:
                #  Fixme: we could do something with the polylines....
                if len(p) > 2:
                    land_polys.append(p)
                else:
                    self.logger.debug("invalid polygon ignored:"
                                      "{} points: {}, ".format(len(p), p.metadata))


        # Draw the raster map with a map_canvas:
        # determine the size:
        BB = land_polys.bounding_box

        if not spillable_area:  # not passed in
            # use the one in the bna
            if len(spillable_area_bna) == 0:
                # no spillable_area in the file
                spillable_area = None
            else:
                spillable_area = spillable_area_bna

        # fixme: map bounds might not be a rectangle -- is this doing the right thing?
        if map_bounds is None:
            if spillable_area:  # add the spillable area to the bounds
                saBB = spillable_area.bounding_box
                saBB.Merge(BB)
                map_bounds = saBB.AsPoly()
            else:
                map_bounds = BB.AsPoly()

        # get the raster as a numpy array:
        raster, projection = self.build_raster(land_polys, BB)

        super(MapFromBNA, self).__init__(
            raster=raster,
            projection=projection,
            map_bounds=map_bounds,
            spillable_area=spillable_area,
            land_polys=land_polys,
            **kwargs)
        return None


    def build_raster(self, land_polys=None, BB=None):
        """
        Build the underlying raster used for the map

        This should be called if the resolution or land polygons change
        """
        # now draw the raster map with a map_canvas:
        # determine the size:

        # stretch the bounding box, to get approximate aspect ratio in
        # projected coords.
        land_polys = self.land_polys if land_polys is None else land_polys
        BB = land_polys.bounding_box if BB is None else BB
        raster_size = self.raster_size

        aspect_ratio = np.cos(BB.Center[1] * np.pi / 180) * (BB.Width / BB.Height)

        w = int(np.sqrt(raster_size * aspect_ratio))
        h = raster_size // w

        canvas = MapCanvas(image_size=(w, h),
                           preset_colors=None,
                           background_color='water',
                           viewport=BB)
        # color doesn't matter here, only index
        canvas.add_colors((('water', (0, 255, 255)),  # aqua
                           ('land', (255, 204, 153)),  # brown
                           ))
        canvas.clear_background()

        # draw the land to the background
        for poly in land_polys:
            # fixme -- this should be something like "land"
            if poly.metadata[2] == '1':
                canvas.draw_polygon(poly,
                                    line_color='land',
                                    fill_color='land',
                                    line_width=1,
                                    background=True)
            # fixme -- this should be something like "lake"
            elif poly.metadata[2] == '2':
                # this is a lake, draw as water
                canvas.draw_polygon(poly,
                                    line_color='water',
                                    fill_color='water',
                                    line_width=1,
                                    background=True)

                # get the raster as a numpy array:
        raster_array = canvas.back_asarray()

        # need to return and use projection used to create the raster, or the to_pixel/from_pixel functions
        # will give incorrect results going forward.
        return raster_array, canvas.projection

    @property
    def raster_size(self):
        '''
        Maximum physical size of the raster in bytes
        '''
        return self._raster_size

    @raster_size.setter
    def raster_size(self, size):
        if size != self._raster_size:
            self._raster_size = size
            #should trigger base class to recreate coarser rasters
            self.raster, self.projection = self.build_raster()

    def to_geojson(self):
        """
        Output the vector version of the shoreline polygons.

        This is what gets drawn in the WebGNOME client, for example

        This version directly writes the polygons already stored in the map
        object -- keeping the door open to that data coming from something
        other than a bna file.

        FIXME: Technically, geojson recommends ccw polygons -- but putting that
               check in was pretty slow, so it's commented out.

        FIXME: This really should export the map_bounds and spillable_area
        as well.
        """
        land_coords = []
        lake_coords = []
        crds=None

        for poly in self.land_polys:
            if poly.metadata[2] == '1':
                crds = land_coords
            elif poly.metadata[2] == '2':
                crds = lake_coords
            else:
                continue
            pts = poly.points.tolist()
            pts.append(pts[0])
            crds.append([pts])
            # FIXME: this is a good idea, but really slow...
            # the is_clockwise() code could be cythonized, maybe that would help?
            # # geojson polygons should be counter-clockwise
            # if is_clockwise(poly):
            #     p.reverse()

        features = []
        if land_coords:
            land = Feature(id="1",
                        properties={'name': 'Shoreline Polys'},
                        geometry=MultiPolygon(land_coords)
                    )
            lakes = Feature(id="2",
                            properties={'name': 'Lakes'},
                            geometry=MultiPolygon(lake_coords)
                        )
            features.append(land)
            features.append(lakes)
        return FeatureCollection(features)



class MapFromUGrid(RasterMap):
    """
    A raster land-water map, created from netcdf File of a UGrid
    """
    _schema = MapFromUGridSchema

    def __init__(self, filename, raster_size=1024 * 1024, **kwargs):
        """
        Creates a GnomeMap (specifically a RasterMap) from a netcdf
        data file with a triangular mesh grid in it.
        The spillable area and map bounds need to be supplied -- there is
        currently no way to express that in a netcdf file.

        Required arguments:

        :param filename: full path to the data file

        :param refloat_halflife: the half-life (in hours) for the re-floating.

        :param raster_size: the total number of pixels (bytes) to make the
                            raster -- the actual size will match the
                            aspect ratio of the bounding box of the land
        :type raster_size: integer

        Optional arguments (kwargs):

        :param map_bounds: The polygon bounding the map -- could be larger or
                           smaller than the land raster

        :param spillable_area: The polygon bounding the spillable_area

        :param id: unique ID of the object. Using UUID as a string.
                   This is only used when loading object from save file.
        :type id: string
        """
        self.filename = filename

        grid = PyGrid.from_netCDF(filename)

        polygons = haz_files.ReadBNA(filename, 'PolygonSet')
        map_bounds = None
        self.name = kwargs.pop('name', os.path.split(filename)[1])

        # find the spillable area and map bounds:
        # and create a new polygonset without them
        #  fixme -- adding a "pop" method to PolygonSet might be better
        #      or a gnome_map_data object...

        land_polys = PolygonSet()  # and lakes....
        spillable_area = PolygonSet()

        for p in polygons:
            if p.metadata[1].lower() == 'spillablearea':
                spillable_area.append(p)
            elif p.metadata[1].lower() == 'map bounds':
                map_bounds = p
            else:
                land_polys.append(p)

        # now draw the raster map with a map_canvas:
        # determine the size:

        BB = land_polys.bounding_box

        # create spillable area and  bounds if they weren't in the BNA
        if map_bounds is None:
            map_bounds = BB.AsPoly()

        if len(spillable_area) == 0:
            spillable_area = None

        # user defined spillable_area, map_bounds overrides data obtained
        # from polygons

        # todo: should there be a check between spillable_area read from BNA
        # versus what the user entered. if this is within spillable_area for
        # BNA, then include it? else ignore
        spillable_area = kwargs.pop('spillable_area', spillable_area)

        map_bounds = kwargs.pop('map_bounds', map_bounds)

        # stretch the bounding box, to get approximate aspect ratio in
        # projected coords.
        aspect_ratio = (np.cos(BB.Center[1] * np.pi / 180) * (BB.Width / BB.Height))

        w = int(np.sqrt(raster_size * aspect_ratio))
        h = raster_size // w

        canvas = MapCanvas(image_size=(w, h),
                           preset_colors=None,
                           background_color='water',
                           viewport=BB,
                           )
        # color doesn't matter here, only index
        canvas.add_colors((('water', (0, 255, 255)),  # aqua
                           ('land', (255, 204, 153)),  # brown
                           ))
        canvas.clear_background()

        # draw the land to the background
        for poly in land_polys:
            # fixme -- this should be something like "land"
            if poly.metadata[2] == '1':
                canvas.draw_polygon(poly,
                                    line_color='land',
                                    fill_color='land',
                                    line_width=1,
                                    background=True)
            # fixme -- this should be something like "lake"
            elif poly.metadata[2] == '2':
                # this is a lake, draw as water
                canvas.draw_polygon(poly,
                                    line_color='water',
                                    fill_color='water',
                                    line_width=1,
                                    background=True)

        # just for testing
        # canvas.save_background("raster_map_test.png")

        # get the raster as a numpy array:
        raster_array = canvas.back_asarray()

        RasterMap.__init__(self, raster_array, canvas.projection,
                           map_bounds=map_bounds,
                           spillable_area=spillable_area,
                           land_polys=land_polys,
                           ** kwargs)

        return None


def map_from_rectangular_grid(mask, lon, lat, refine=1, **kwargs):
    """
    Suitable for a rectangular, but not fully regular, grid

    Such that it can be described by single longitude and latitude vectors

    :param mask: the land-water mask as a numpy array

    :param lon: longitude array

    :param lon: latitude array

    :param refine=1: amount to refine grid -- 4 will give 4 times the
                     resolution
    :type refine: integer

    :param kwargs: Other keyword arguments are passed on to RasterMap
    """
    # expand the grid mask
    grid = np.repeat(mask, refine, axis=0)
    grid = np.repeat(grid, refine, axis=1)

    # refine the axes:
    lon = refine_axis(lon, refine)
    lat = refine_axis(lat, refine)

    nlon, nlat = grid.shape

    map_bounds = np.array(((lon[0], lat[0]), (lon[-1], lat[0]),
                           (lon[-1], lat[-1]), (lon[0], lat[-1])),
                          dtype=np.float64)

    # generating projection for raster map
    proj = RectangularGridProjection(lon, lat)

    return RasterMap(grid, proj,
                               map_bounds=map_bounds,
                               **kwargs)


def grid_from_nc(filename):
    """
    generates a grid_mask and lat lon from a conforming netcdf file

    NOTE: poorly tested, and will only work on a regular grid
    """
    import netCDF4

    nc = netCDF4.Dataset(filename)

    lat_var = nc.variables['lat']
    lon_var = nc.variables['lon']

    nx, ny = lat_var.shape

    # check for regular grid:
    # all rows should be same:
    for r in range(nx):
        if not np.array_equal(lon_var[r, :], lon_var[0, :]):
            raise ValueError("Row: %i isn't equal!" % r)

    for c in range(ny):
        if not np.array_equal(lat_var[:, c], lat_var[:, 0]):
            raise ValueError("column: %i isn't equal!" % c)

    mask = nc.variables['mask'][:]

    # Re-shuffle for gnome raster map orientation:
    # create the raster
    # raster_array = np.zeros( (nlon, nlat), dtype=np.uint8 )
    mask = (mask == 0).astype(np.uint8)  # swap water/land
    mask = np.ascontiguousarray(np.fliplr(mask.T))  # to get oriented right.

    # extra point to fill for last grid cell
    # note: values can be variable, so not *quite* right
    lon = lon_var[0, :]
    lat = lat_var[:, 0]
    lon = np.r_[lon, [2 * lon[-1] - lon[-2]]]
    lat = np.r_[lat, [2 * lat[-1] - lat[-2]]]

    return mask, lon, lat


def map_from_rectangular_grid_nc_file(filename, refine=1, **kwargs):
    """
    builds a raster map from a rectangular grid in a netcdf file

    only tested with the HYCOM grid

    :param filename: the full path or opendap url for the netcdf file
    :type filename: string

    :param refine: how much to refine the grid. 1 means keep it as it is,
        otherwise is will scale
    :type refine: integer

    :param kwargs: other key word arguemnts -- passed on to RasterMap class
        constructor
    """
    grid, lon, lat = grid_from_nc(filename)
    map_ = map_from_rectangular_grid(grid, lon, lat, refine, **kwargs)

    return map_


def refine_axis(old_axis, refine):
    """
    refines the axis be interpolating points between each axis points

    :param old_axis: the axis values
    :type old_axis: 1-d numpy array of floats

    :param refine: amount to refine grid -- 4 will give 4 times the resolution
    :type refine: integer
    """
    refine = int(refine)

    axis = old_axis.reshape((-1, 1))
    axis = ((axis[1:] - axis[:-1]) / refine) * np.arange(refine) + axis[:-1]
    axis.shape = (-1,)
    axis = np.r_[axis, old_axis[-1]]

    return axis


def map_from_regular_grid(grid_mask, lon, lat, refine=4, refloat_halflife=1,
                          map_bounds=None):
    """
    note: untested -- here to save it in case we need it in the future

    makes a raster map from a regular grid: i.e delta_lon and delta-lat are
    constant.
    """
    nlon, nlat = grid_mask.shape
    dlon = (lon[-1] - lon[0]) / (len(lon) - 1)
    dlat = (lat[-1] - lat[0]) / (len(lat) - 1)

    # create the raster
    raster_array = np.zeros((nlon * resolution, nlat * resolution),
                            dtype=np.uint8)

    # add the land to the raster
    for i in range(resolution):
        for j in range(resolution):
            raster_array[i::resolution, j::resolution] = grid_mask

    # compute projection
    bounding_box = np.array(((lon[0], lat[0]),
                             (lon[-1] + dlon, lat[-1] + dlat)),
                            dtype=np.float64)  # adjust for last grid cell

    proj = RegularGridProjection(bounding_box,
                                 image_size=raster_array.shape)

    return RasterMap(
        raster=raster_array,
        projection=proj,
        refloat_halflife=refloat_halflife)
