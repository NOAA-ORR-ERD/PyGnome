from gnome.utilities.projections import FlatEarthProjection
from numpy import sin, deg2rad
from sympy.geometry.util import intersection

###fixme:
### needs some refactoring -- more clear distinction between public and
###                           private API:

# on_land, in_water, etc -- need that?

# NOTES:
#  - Should we just use non-projected coordinates for the raster map?
#    It makes for a little less computation at every step.
#  - Do we want to treat lakes differently than regular water?

# New features:
#  - Map now handles spillable area and map bounds as polygons
#  - raster is the same aspect ratio as the land
#  - internally, raster is a numpy array
#  - land raster is only as big as the land -- if the map bounds are bigger,
#    extra space is not in the land map
#  Question: what if map-bounds is smaller than land? wasted bitmap space?
#            (though it should work)

"""
An implementation of the GNOME land-water map.

This is a re-write of the C++ raster map approach
"""

import copy
import os

import numpy
np = numpy
from colander import SchemaNode, String, Float, drop

from gnome.persist import base_schema

import gnome.map

from gnome.utilities.map_canvas import BW_MapCanvas
from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.file_tools import haz_files

from gnome.utilities import projections

from gnome.basic_types import oil_status, world_point_type
from gnome.cy_gnome.cy_land_check import check_land
from gnome.cy_gnome.cy_land_check import do_landings

from gnome.utilities.geometry.cy_point_in_polygon import (points_in_poly,
                                                          point_in_poly)
from gnome.utilities.geometry.polygons import PolygonSet


class GnomeMapSchema(base_schema.ObjType):
    map_bounds = base_schema.LongLatBounds(missing=drop)
    spillable_area = base_schema.PolygonSet(missing=drop)


class MapFromBNASchema(GnomeMapSchema):
    filename = SchemaNode(String())
    refloat_halflife = SchemaNode(Float(), missing=drop)


class GnomeMap(Serializable):
    """
    The very simplest map for GNOME -- all water
    with only a bounding box for the map bounds.

    This also serves as a description of the interface
    """
    _update = ['map_bounds', 'spillable_area']
    _create = []
    _create.extend(_update)
    _state = copy.deepcopy(Serializable._state)
    _state.add(save=_create, update=_update)
    _schema = GnomeMapSchema

    refloat_halflife = None  # note -- no land, so never used

    def __init__(self, map_bounds=None, spillable_area=None, name=None):
        """
        This __init__ will be different for other implementations

        Optional parameters (kwargs)

        :param map_bounds: The polygon bounding the map -- could be larger
                           or smaller than the land raster

        :param spillable_area: The PolygonSet bounding the spillable_area.
        :type spillable_area: Either a PolygonSet object or a list of lists
            from which a polygon set can be created. Each element in the list
            is a list of points defining a polygon.

        Note on 'map_bounds':
            ( (x1,y1), (x2,y2),(x3,y3),..)
            An NX2 array of points that describe a polygon
            if no map bounds is provided -- the whole world is valid
        """
        if map_bounds is not None:
            self.map_bounds = np.asarray(map_bounds,
                    dtype=np.float64).reshape(-1, 2)
        else:
            # using -360 to 360 to allow stuff to cross the dateline..
            self.map_bounds = np.array(((-360, 90),
                                        (360, 90),
                                        (360, -90),
                                        (-360, -90)),
                                       dtype=np.float64)

        if spillable_area is None:
            #self.spillable_area = self.map_bounds
            self.spillable_area = PolygonSet()
            self.spillable_area.append(self.map_bounds)
        else:
            if not isinstance(spillable_area, PolygonSet):
                spillable_area = self._polygon_set_from_points(spillable_area)

            self.spillable_area = spillable_area

    def _polygon_set_from_points(self, poly):
        '''
        create PolygonSet() object from list of polygons which in turn is a
        list of points
        :returns: PolygonSet() object
        '''
        x = PolygonSet()
        for p in poly:
            x.append(p)
        return x

    def _attr_array_to_dict(self, np_array):
        '''convert np_array to list of tuples, used for map_bounds,
        spillable_area'''
        return map(tuple, np_array.tolist())

    def _attr_from_list_to_array(self, l_):
        '''
        dict returned as list of tuples to be converted to numpy array
        Again used to update_from_dict map_bounds and spillable_area
        '''
        return np.asarray(l_, dtype=np.float64).reshape(-1, 2)

    def map_bounds_to_dict(self):
        'convert numpy array to a list for serializing'
        return self._attr_array_to_dict(self.map_bounds)

    def map_bounds_update_from_dict(self, val):
        'convert list of tuples back to numpy array'
        new_arr = self._attr_from_list_to_array(val)
        if np.any(self.map_bounds != new_arr):
            self.map_bounds = new_arr
            return True

        return False

    def spillable_area_to_dict(self):
        'convert numpy array to a list for serializing'
        #return self._attr_array_to_dict(self.spillable_area)
        x = []
        for poly in self.spillable_area:
            x.append(poly.points.tolist())
        return x

    def spillable_area_update_from_dict(self, poly_set):
        'convert list of tuples back to numpy array'
        # since metadata will not match, let's create a new PolygonSet,
        # check equality on _PointsArray and update if not equal
        ps = PolygonSet()
        for poly in poly_set:
            ps.append(poly)

        if not np.array_equal(self.spillable_area._PointsArray,
                              ps._PointsArray):
            self.spillable_area = ps
            return True

        return False

    def on_map(self, coords):
        """
        :param coords: location for test.
        :type coords: 3-tuple of floats: (long, lat, depth) or a
                                         NX3 numpy array

        :return: bool array: True if the location is on the map,
                             False otherwise

        Note:
          coord is 3-d, but the concept of "on the map" is 2-d in this context,
          so depth is ignored.
        """
        coords = np.asarray(coords, dtype=world_point_type)
        on_map_mask = points_in_poly(self.map_bounds, coords)
        return on_map_mask

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
        for poly in self.spillable_area:
            if points_in_poly(poly.points, coord):
                return True

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

    def beach_elements(self, spill):
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
        self.resurface_airborne_elements(spill)
        self._set_off_map_status(spill)

    def refloat_elements(self, spill_container, time_step):
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


class Param_Map(GnomeMap):
    
    def __init__(self, center = (0,0), distance = 30000, bearing = 90):
        #basically, direction vector to shore
        center = (center[0], center[1], 0)
        distance = (distance, 0, 0)
        d = FlatEarthProjection.meters_to_lonlat(distance, center)[0][0]
        init_points = [(720,720),(720,-720),(d,-720),(d,720)]
        ang = deg2rad(90 - bearing)
        rot_matrix = [(np.cos(ang), np.sin(ang)),(-np.sin(ang),np.cos(ang))]
        self.land_points = np.dot(init_points, rot_matrix)
        self.land_points = np.array([(x + center[0], y + center[1]) for (x,y) in self.land_points])
        self.map_bounds = np.array(((center[0] - 2*d, center[1] + 2*d),
                                        (center[0] + 2*d, center[1] + 2*d),
                                        (center[0] + 2*d, center[1] - 2*d),
                                        (center[0] - 2*d, center[1] - 2*d)),
                                       dtype=np.float64)
        self._refloat_halflife = 0.5
    
    def get_map_bounds(self):
        return (self.map_bounds[1], self.map_bounds[3])
    
    def get_land_polygon(self):
        poly = PolygonSet((self.land_points,[0,4],[]))
        poly._MetaDataList = [('polygon','1','1')]
        return poly


    def on_map(self, coords):
        """
        :param coords: location for test.
        :type coords: 3-tuple of floats: (long, lat, depth)

        :return: bool array: True if the location is on the map,
                             False otherwise

        Note:
          coord is 3-d, but the concept of "on the map" is 2-d in this context,
          so depth is ignored.
        """
        return points_in_poly(self.map_bounds, coords)

    def on_land(self, coords):
        """
        :param coord: location for test.
        :type coords: 3-tuple of floats: (long, lat, depth) or Nx3 numpy array

        :return:
         - Always returns False-- no land in this implementation
        """
        return  self.on_map(coords) * points_in_poly(self.land_points, coords)

    def in_water(self, coords):
        """
        :param coords: location for test.
        :type coords: 3-tuple of floats: (long, lat, depth)

        :returns:
         - True if the point is in the water,
         - False if the point is on land (or off map?)

         This implementation has no land, so always True in on the map.
        """
        return self.on_map(coords) and not self.on_land(coords)

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
            print "Only allowable location for spill is the center this map was built with"
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
    
    #line intersection on each numpy element 
    def find_land_intersection(self, starts, ends):
        lines = np.array(zip(starts, ends)) #each index is (x1,y1,x2,y2)
        sl = [self.land_points[2], self.land_points[3]]
        [x1, y1], [x2, y2] = sl[0], sl[1]
        a = sl[1] - sl[0]
        
        inters = np.ndarray((ends.shape[0],3), dtype = np.float64)
        for i in range(0,lines.shape[0]):
            p1 = lines[i][0]
            p2 = lines[i][1]
            [x3,y3],[x4,y4] = p1[0:2],p2[0:2]
            b = p2[0:2] - p1[0:2]
            den = a[0]*b[1] - b[0]*a[1]
            if (den is 0):
                raise ValueError("den is 0")
            u = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3))/den
            v = y1 + u * (y2-y1)
            inters[i] = [u,v,0]
        
        return inters
    
    def find_last_water_pos(self,starts, ends):
        return starts + (ends-starts) * 0.000001
    
    def beach_elements(self, sc):
        """
        Determines which LEs were or weren't beached or moved off_map.
        status_code is changed to oil_status.off_maps if off the map.

        Called by the model in the main time loop, after all movers have acted.

        :param sc: current SpillContainer
        :type sc:  :class:`gnome.spill_container.SpillContainer`

        This map class has no land, so only the map check and
        resurface_airborn elements is done: noting else changes.

        subclasses that override this probably want to make sure that:

        self.resurface_airborne_elements(sc)
        self._set_off_map_status(sc)

        are called.
        """
        self.resurface_airborne_elements(sc)
        self._set_off_map_status(sc)
        
        start_pos = sc['positions']
        next_pos = sc['next_positions']
        status_codes = sc['status_codes']
        last_water_positions = sc['last_water_positions']
        
        
        # beached = 1xN numpy array of bool, elem is true if on water and next pos is on land
        new_beached = ((status_codes == oil_status.in_water) * self.on_land(next_pos))
        land = np.array((self.land_points[2], self.land_points[3]))
        do_landings(start_pos, next_pos, status_codes, last_water_positions, new_beached, land)
        #status_codes[new_beached] = oil_status.on_land
         
#         next_pos[new_beached] = self.find_land_intersection(start_pos[new_beached], next_pos[new_beached])
#         last_water_positions[new_beached] = self.find_last_water_pos(start_pos[new_beached], next_pos[new_beached])
        
    def refloat_elements(self, spill_container, time_step):
        """
        This method performs the re-float logic -- changing the element
        status flag, and moving the element to the last known water position

        :param spill_container: the current spill container
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`
        """
        # index into array of particles on_land

        r_idx = np.where(spill_container['status_codes']
                         == oil_status.on_land)[0]

        if r_idx.size == 0:  # no particles on land
            return

        if self._refloat_halflife > 0.0:
            #if 0.0, then r_idx is all of them -- they will all refloat.
            # refloat particles based on probability

            refloat_probability = 1.0 - 0.5 ** (float(time_step)
                                                / self._refloat_halflife)
            rnd = np.random.uniform(0, 1, len(r_idx))

            # subset of indices that will refloat
            # maybe we should rename refloat_probability since
            # rnd <= refloat_probability to
            # refloat, maybe call it stay_on_land_probability
            r_idx = r_idx[np.where(rnd <= refloat_probability)[0]]
        elif self._refloat_halflife < 0.0:
            # fake for nothing gets refloated.
            r_idx = np.array((), np.bool)

        if r_idx.size > 0:
            # check is not required, but why do this operation if no particles
            # need to be refloated
            spill_container['positions'][r_idx] = \
                spill_container['last_water_positions'][r_idx]
            spill_container['status_codes'][r_idx] = oil_status.in_water


class RasterMap(GnomeMap):
    """01
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

    # flags for what's in the bitmap
    # in theory -- it could be used for other data:
    #  refloat, other properties?
    # note the BW map_canvas only does 1, though.
    seconds_in_hour = 60 * 60

    land_flag = 1

    def __init__(self, bitmap_array, projection, **kwargs):
        """
        create a new RasterMap


        :param bitmap_array: A numpy array that stores the land-water map
        :type bitmap_array: a (W,H) numpy array of type uint8

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
        refloat_halflife = kwargs.pop('refloat_halflife', 1)
        self._refloat_halflife = refloat_halflife * self.seconds_in_hour

        self.bitmap = bitmap_array
        self.projection = projection

        GnomeMap.__init__(self, **kwargs)

    @property
    def refloat_halflife(self):
        return self._refloat_halflife / self.seconds_in_hour

    @refloat_halflife.setter
    def refloat_halflife(self, value):
        self._refloat_halflife = value * self.seconds_in_hour

    def save_as_image(self, filename):
        '''
        Save the land-water raster as a PNG save_as_image

        :param filename: the name of the file to save to.
        '''
        from PIL import Image

        bitmap = self.bitmap.copy()

        #change anyting not zero to 255 - to get black and white
        np.putmask(bitmap, self.bitmap > 0, 255)
        im = Image.fromarray(bitmap, mode='L')

        # to get it oriented right...
        im = im.transpose(Image.ROTATE_90)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

        im.save(filename, format='PNG')

    def _on_land_pixel(self, coord):
        """
        returns 1 if the point is on land, 0 otherwise

        :param coord: pixel coordinates of point of interest
        :type coord: tuple: (row, col)

        .. note:: Only used internally or for testing -- no need for external
                  API to use pixel coordinates.
        """
        try:
            return self.bitmap[coord[0], coord[1]] & self.land_flag
        except IndexError:
            # not on land if outside the land raster. (Might be off the map!)
            return False

    def on_land(self, coord):
        """
        :param coord: (long, lat, depth) location -- depth is ignored here.
        :type coord: 3-tyuple of floats -- (long, lat, depth)

        :return:
         - 1 if point on land
         - 0 if not on land

        .. note:: to_pixel() converts to array of points...
        """
        return self._on_land_pixel(self.projection.to_pixel(coord)[0])

    def _on_land_pixel_array(self, coords):
        """
        determines which LEs are on lond

        :param coords:  Nx2 numpy int array of pixel coords matching the bitmap
        :type coords:  Nx2 numpy int array of pixel coords matching the bitmap

        returns: a (N,) array of bools - true for particles that are on land
        """
        mask = map(point_in_poly, [self.map_bounds] * len(coords), coords)
        racpy = np.copy(coords)[mask]
        mskgph = self.bitmap[racpy[:, 0], racpy[:, 1]]
        chrmgph = np.array([0] * len(coords))
        chrmgph[np.array(mask)] = mskgph

        return chrmgph

    def _in_water_pixel(self, coord):
        try:
            return not self.bitmap[coord[0], coord[1]] & self.land_flag
        except IndexError:
            # Note: this could be off map, which may be a different thing
            #       than on land....but off the map should have been tested
            #       first
            return True

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

    def beach_elements(self, sc):
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
        self._check_land(self.bitmap, start_pos_pixel, next_pos_pixel,
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

    def refloat_elements(self, spill_container, time_step):
        """
        This method performs the re-float logic -- changing the element
        status flag, and moving the element to the last known water position

        :param spill_container: the current spill container
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`
        """
        # index into array of particles on_land

        r_idx = np.where(spill_container['status_codes']
                         == oil_status.on_land)[0]

        if r_idx.size == 0:  # no particles on land
            return

        if self._refloat_halflife > 0.0:
            #if 0.0, then r_idx is all of them -- they will all refloat.
            # refloat particles based on probability

            refloat_probability = 1.0 - 0.5 ** (float(time_step)
                                                / self._refloat_halflife)
            rnd = np.random.uniform(0, 1, len(r_idx))

            # subset of indices that will refloat
            # maybe we should rename refloat_probability since
            # rnd <= refloat_probability to
            # refloat, maybe call it stay_on_land_probability
            r_idx = r_idx[np.where(rnd <= refloat_probability)[0]]
        elif self._refloat_halflife < 0.0:
            # fake for nothing gets refloated.
            r_idx = np.array((), np.bool)

        if r_idx.size > 0:
            # check is not required, but why do this operation if no particles
            # need to be refloated
            spill_container['positions'][r_idx] = \
                spill_container['last_water_positions'][r_idx]
            spill_container['status_codes'][r_idx] = oil_status.in_water

    def _check_land(self, raster_map, positions, end_positions,
                    status_codes, last_water_positions):
        """
        Do the actual land-checking.  This method calls a Cython version:
            gnome.cy_gnome.cy_land_check.check_land()

        The arguments 'status_codes', 'positions' and 'last_water_positions'
        are altered in place.
        """
        check_land(raster_map, positions, end_positions, status_codes,
                   last_water_positions)

    def allowable_spill_position(self, coord):
        """
        Returns true is the spill position is in the allowable spill area

        .. note::
            This may not be the same as in_water!

        :param coord: (lon, lat, depth) coordinate
        """
        if self.on_map(coord):
            if not self.on_land(coord):
                if self.spillable_area is None:
                    return True
                else:
                    return super(RasterMap, self).allowable_spill_position(coord)
            else:
                return False
        else:
            return False

    def to_pixel_array(self, coords):
        """
        Projects an array of (lon, lat) tuples onto the bitmap,
        and modifies it in place to hold the corresponding projected values.

        :param coords:  a numpy array of (lon, lat, depth) points

        :return: a numpy array of (x, y) pixel values
        """
        return self.projection.to_pixel(coords)


class MapFromBNA(RasterMap):
    """
    A raster land-water map, created from a BNA file
    """
    _state = copy.deepcopy(RasterMap._state)
    _state.update(['map_bounds', 'spillable_area'], save=False)
    _state.add(save=['refloat_halflife'], update=['refloat_halflife'])
    _state.add_field(Field('filename', isdatafile=True, save=True,
                           read=True, test_for_eq=False))
    _schema = MapFromBNASchema

    def __init__(self, filename, raster_size=1024 * 1024, **kwargs):
        """
        Creates a GnomeMap (specifically a RasterMap) from a bna file.
        It is expected that you will get the spillable area and map bounds
        from the BNA -- if they exist

        Required arguments:

        :param bna_file: full path to a bna file
        :param refloat_halflife: the half-life (in hours) for the re-floating.
        :param raster_size: the total number of pixels (bytes) to make the
                            raster -- the actual size will match the
                            aspect ratio of the bounding box of the land

        Optional arguments (kwargs):

        :param map_bounds: The polygon bounding the map -- could be larger or
                           smaller than the land raster
        :param spillable_area: The polygon bounding the spillable_area
        :param id: unique ID of the object. Using UUID as a string.
                   This is only used when loading object from save file.
        :type id: string
        """
        self.filename = filename
        polygons = haz_files.ReadBNA(filename, 'PolygonSet')
        map_bounds = None
        self.name = kwargs.pop('name', os.path.split(filename)[1])

        # find the spillable area and map bounds:
        # and create a new polygonset without them
        #  fixme -- adding a "pop" method to PolygonSet might be better
        #      or a gnome_map_data object...

        just_land = PolygonSet()  # and lakes....
        spillable_area = PolygonSet()

        for p in polygons:
            if p.metadata[1].lower() == 'spillablearea':
                spillable_area.append(p)

            elif p.metadata[1].lower() == 'map bounds':
                map_bounds = p
            else:
                just_land.append(p)

        # now draw the raster map with a map_canvas:
        # determine the size:

        BB = just_land.bounding_box

        # create spillable area and  bounds if they weren't in the BNA
        if map_bounds is None:
            map_bounds = BB.AsPoly()

        if len(spillable_area) == 0:
            spillable_area.append(map_bounds)

        # user defined spillable_area, map_bounds overrides data obtained
        # from polygons

        # todo: should there be a check between spillable_area read from BNA
        # versus what the user entered. if this is within spillable_area for
        # BNA, then include it? else ignore
        #spillable_area = kwargs.pop('spillable_area', spillable_area)
        spillable_area = kwargs.pop('spillable_area', spillable_area)
        map_bounds = kwargs.pop('map_bounds', map_bounds)

        # stretch the bounding box, to get approximate aspect ratio in
        # projected coords.

        aspect_ratio = (np.cos(BB.Center[1] * np.pi / 180) *
                        (BB.Width / BB.Height))
        w = int(np.sqrt(raster_size * aspect_ratio))
        h = int(raster_size / w)

        canvas = BW_MapCanvas((w, h), land_polygons=just_land)
        canvas.draw_background()

        # canvas.save_background("raster_map_test.png")

        # # get the bitmap as a numpy array:

        bitmap_array = canvas.as_array()

        # __init__ the  RasterMap

        # hours
        RasterMap.__init__(self,
                           bitmap_array,
                           canvas.projection,
                           map_bounds=map_bounds,
                           spillable_area=spillable_area,
                           **kwargs)

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

    map_bounds = np.array(((lon[0], lat[0]),
                           (lon[-1], lat[0]),
                           (lon[-1], lat[-1]),
                           (lon[0], lat[-1]),
                           ), dtype=np.float)

    # generating projection for raster map
    proj = projections.RectangularGridProjection(lon, lat)

    return gnome.map.RasterMap(grid,
                               proj,
                               map_bounds=map_bounds,
                               **kwargs)


def grid_from_nc(filename):
    """
    generates a grid_mask and lat lon from a conforming netcdf file
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
    # bitmap_array = np.zeros( (nlon, nlat), dtype=np.uint8 )
    mask = (mask == 0).astype(np.uint8)  # swap water/land
    mask = np.ascontiguousarray(np.fliplr(mask.T))  # to get oriented right.

    # extra point to fill for last grid cell
    # note: values can be variable, so not *quite* right
    lon = lon_var[0, :]
    lat = lat_var[:, 0]
    lon = np.r_[lon, [2*lon[-1] - lon[-2]]]
    lat = np.r_[lat, [2*lat[-1] - lat[-2]]]

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


def map_from_regular_grid(grid_mask, lon, lat, refine=4, refloat_halflife=6,
                          map_bounds=None):
    """
    note: poorly tested -- here to save it in case we need it in the future

    makes a raster map from a regular grid: i.e delta_lon and delta-lat are
    constant.

    """
    nlon, nlat = grid_mask.shape
    dlon = (lon[-1] - lon[0]) / (len(lon)-1)
    dlat = (lat[-1] - lat[0]) / (len(lat)-1)

    # create the raster
    bitmap_array = np.zeros((nlon*resolution, nlat*resolution), dtype=np.uint8)
    # add the land to the raster
    for i in range(resolution):
        for j in range(resolution):
            bitmap_array[i::resolution, j::resolution] = grid_mask

    # compute projection
    bounding_box = np.array(((lon[0], lat[0]),
                             (lon[-1]+dlon, lat[-1]+dlat),
                             ), dtype=np.float64)  # adjust for last grid cell
    proj = RegularGridProjection(bounding_box,
                                 image_size=bitmap_array.shape,
                                 )

    return gnome.map.RasterMap(bitmap_array,
                               proj,
                               refloat_halflife=refloat_halflife,
                               )
