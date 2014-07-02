
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

from gnome.utilities.map_canvas import BW_MapCanvas
from gnome.utilities.serializable import Serializable, Field
from gnome.utilities.file_tools import haz_files

from gnome.basic_types import oil_status, world_point_type
from gnome.cy_gnome.cy_land_check import check_land

from gnome.utilities.geometry.cy_point_in_polygon import (points_in_poly,
                                                          point_in_poly)
from gnome.utilities.geometry.polygons import PolygonSet


class GnomeMapSchema(base_schema.ObjType):
    map_bounds = base_schema.LongLatBounds()
    spillable_area = base_schema.LongLatBounds(missing=drop)


class MapFromBNASchema(base_schema.ObjType):
    filename = SchemaNode(String())
    refloat_halflife = SchemaNode(Float())


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

        :param spillable_area: The polygon bounding the spillable_area

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
            self.spillable_area = self.map_bounds
        else:
            self.spillable_area = np.asarray(spillable_area,
                                             dtype=np.float64).reshape(-1, 2)

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
        return points_in_poly(self.spillable_area, coord)

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

        status_codes[off_map] = oil_status.to_be_removed

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

    # flags for what's in the bitmap
    # in theory -- it could be used for other data:
    #  refloat, other properties?
    # note the BW map_canvas only does 1, though.
    seconds_in_hour = 60 * 60

    land_flag = 1

    def __init__(self, refloat_halflife, bitmap_array, projection,
                 **kwargs):
        """
        create a new RasterMap

        :param refloat_halflife: The halflife for refloating off land
                                 -- assumed to be the same for all land.
                                 0.0 means all refloat every time step
                                 < 0.0 means never re-float.
        :type refloat_halflife: float. Units are hours

        :param bitmap_array: A numpy array that stores the land-water map
        :type bitmap_array: a (W,H) numpy array of type uint8

        :param projection: A Projection object -- used to convert from
                           lat-long to pixels in the array
        :type projection: :class:`gnome.map_canvas.Projection`

        Optional arguments (kwargs)

        :param map_bounds: The polygon bounding the map -- could be larger
                           or smaller than the land raster
        :type map_bounds: (N,2) numpy array of floats

        :param spillable_area: The polygon bounding the spillable_area
        :type spillable_area: (N,2) numpy array of floats

        :param id: unique ID of the object. Using UUID as a string.
                   This is only used when loading object from save file.

        :type id: string
        """
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

    def beach_elements(self, spill):
        """
        Determines which elements were or weren't beached.

        Any that are beached have the beached flag set, and a
        "last known water position" (lkwp) is computed

        This version uses a modified Bresenham algorithm to find out
        which pixels the LE may have crossed.

        :param spill: the current spill container
        :type spill:  :class:`gnome.spill_container.SpillContainer`
            It must have the following data arrays:
            ('prev_position', 'positions', 'last_water_pt', 'status_code')
        """
        self.resurface_airborne_elements(spill)

        # pull the data from the spill
        # Is the last water point the same as the previous position? why not??
        # If beached, it won't move, if not, then we can use it?

        start_pos = spill['positions']
        next_pos = spill['next_positions']
        status_codes = spill['status_codes']
        last_water_positions = spill['last_water_positions']

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

        self._set_off_map_status(spill)

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
                    return points_in_poly(self.spillable_area, coord)
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

    def __init__(self, filename, refloat_halflife, raster_size=1024 * 1024,
                 **kwargs):
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
        spillable_area = None
        self.name = kwargs.pop('name', os.path.split(filename)[1])

        # find the spillable area and map bounds:
        # and create a new polygonset without them
        #  fixme -- adding a "pop" method to PolygonSet might be better
        #      or a gnome_map_data object...

        just_land = PolygonSet()  # and lakes....

        for p in polygons:
            if p.metadata[1].lower() == 'spillablearea':
                spillable_area = p
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

        if spillable_area is None:
            spillable_area = map_bounds

        # user defined spillable_area, map_bounds overrides data obtained
        # from polygons

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
        RasterMap.__init__(self, refloat_halflife, bitmap_array,
                           canvas.projection,
                           map_bounds=map_bounds,
                           spillable_area=spillable_area,
                           **kwargs)

        return None
