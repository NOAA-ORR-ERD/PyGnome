"""
Module to hold classes and supporting code for projections used in GNOME.

Only:

  * no projection
  * geo-projection (just scaling to pixels)
  * a simple "flat earth" projection for

Also a bit of code for scaling lat-long to meters, etc.

Used by map_canvas code and map code.

NOTE: all coordinates are takes as (lon, lat, depth)
      even though depth is always ignored
"""







import numpy as np
from gnome.gnomeobject import GnomeId
from gnome.persist.base_schema import ObjTypeSchema
from colander import drop, TupleSchema, Float, SchemaNode, Int


def to_2d_coords(coords):
    """
    utility function to take whatever the heck someone may pass in
    for coordinates and make it an Nx2 array

    :param input: the input coordinates. they should be one of:
                  a (lon, lat) pair
                  a (lon, lat, depth) triple
                  a Nx2 array-like object of (lon,lat) pairs
                  a Nx3 array-like object of (lon, lat, depth) triples

    The depth is ignored in all cases

    This is probably overly convenient, but the legacy is there...
    """
    coords = np.atleast_2d(coords)
    if coords.shape[1] == 2:
        return coords
    if coords.shape[1] == 3:
        return coords[:, :2]
    else:
        raise ValueError("coords must be one of:\n"
                         "a (lon, lat) pair\n"
                         "a (lon, lat, depth) triple\n"
                         "a Nx2 array-like object of (lon, lat)\n"
                         "a Nx3 array-like object of (lon, lat, depth)\n"
                         )


class ProjectionSchema(ObjTypeSchema):
    bounding_box = TupleSchema(
        missing=drop, save=True, update=False,
        children=[TupleSchema(children=[SchemaNode(Float()),
                                        SchemaNode(Float())]),
                  TupleSchema(children=[SchemaNode(Float()),
                                        SchemaNode(Float())])
                  ])
    image_size = TupleSchema(save=True, update=True, missing=drop,
                             children=[SchemaNode(Int()), SchemaNode(Int())])


class NoProjection(GnomeId):
    """
    This is do-nothing projection class -- returns what it gets.

    It optionally rounds down to integer (pixel) coordinates

    used for testing, primarily, and as a definition of the interface
    """
    _schema = ProjectionSchema

    def set_scale(self, bounding_box, image_size=None):
        """
        Does nothing
        """
        pass

    @property
    def bounding_box(self):
        return self.image_box

    def to_pixel(self, coords, asint=False):
        """
        returns the same (lon, lat) coords, but as an np.array, if they aren't
        already

        :param coords: The coords to project
        :type coords: Nx3 numpy array or compatible sequence (lon, lat, depth)

        :param asint: Flag to set whether to convert to a integer or not
                      default is to leave it as the same type it came in,
                      so you can have fractional pixels
        """
        if asint:
            # C ordering to make sure it's contiguous
            return np.asarray(to_2d_coords(coords), dtype=np.int32, order='C')
        else:
            # C ordering to make sure it's contiguous
            return np.asarray(to_2d_coords(coords), order='C')

    def to_pixel_2D(self, coords, asint=False):
        # FIXME: this is not longer required, what with to_2d_coords
        """
        same as to_pixel, but expects only (lon, lat) coords as input.

        :param coords: The coords to project
        :type coords: Nx2 numpy array or compatible sequence (lon, lat)

        :param asint: flag to set whether to convert to a integer or not
                      default is to leave it as the same type it came in,
                      so you can have fractional pixels
        """
        raise NotImplementedError("no longer required, use to_pixel()")

    def to_lonlat(self, coords):
        """
        returns the same coords, but as a np.array of float64,
        if they aren't already
        """
        return np.asarray(coords, dtype=np.float64, order='C')


class GeoProjection(GnomeId):
    """
    This acts as the base class for other projections

    This one doesn't really project, but does convert to pixel coords
    i.e. "geo-coordinates"
    """
    _schema = ProjectionSchema

    def __init__(self, bounding_box=None, image_size=None, *args, **kwargs):
        """
        Create a new projection

        Projection(bounding_box, image_size)

        :param bounding_box: The bounding box of the map
        :type bounding_box: Struct of the form::

                                ((min_long, min_lat),
                                 (max_lon,  max_lat))

                            or a BoundingBox Object

        :param image_size: The size of the map image
        :type image_size: Struct of the form (width, height)

        """
        super(GeoProjection, self).__init__(*args, **kwargs)
        self.center = None
        self.offset = None
        self.scale = None

        if image_size is None:
            self.image_size = (600, 600)

        if bounding_box is None:
            bounding_box = ((-180, -90), (180, 90))

        self.image_box = bounding_box
        self.set_scale(bounding_box, image_size)

    @property
    def bounding_box(self):
        return self.image_box

    def __eq__(self, other):
        """
        provide an equality check for checking
        saved state of renderers, etc
        """
        if type(self) is not type(other):
            return False
        elif not np.allclose(self.center, other.center, rtol=1e-4, atol=1e-4):
            return False
        elif not np.array_equal(self.offset, other.offset):
            return False
        elif not np.allclose(self.scale, other.scale, rtol=1e-4, atol=1e-4):
            return False
        elif not np.array_equal(self.image_size, other.image_size):
            return False
        else:
            return True

    def __ne__(self, other):
        return not self == other

    def set_scale(self, bounding_box, image_size=None):
        """
        Set the scaling, etc. of the projection

        This should be called whenever the bounding box of the map,
        or the size of the image is changed

        :param bounding_box: bounding box of the visual portion of the map
        :type bounding_box: Struct of the form: ((min_long, min_lat),
                                                 (max_long, max_lat))

        :param image_size=None: The size of the image that will be drawn to.
                                if not given, the previous size will be used.
        """
        if image_size is None:
            image_size = self.image_size

        bounding_box = np.asarray(bounding_box, dtype=np.float64)

        self.center = np.mean(bounding_box, axis=0)
        self.offset = np.array(image_size, dtype=np.float64) / 2

        # compute BB to fit image
        h = bounding_box[1, 1] - bounding_box[0, 1]

        # width scaled to longitude
        w = bounding_box[1, 0] - bounding_box[0, 0]

        if w / h > image_size[0] / image_size[1]:
            s = image_size[0] / w
        else:
            s = image_size[1] / h

        self.scale = (s, -s)

        # doing this at the end, in case there is a problem with the input.

        self.image_box = (self.to_lonlat((0, image_size[1])),
                          self.to_lonlat((image_size[0], 0)))
        self.image_size = image_size

    def to_pixel(self, coords, asint=False):
        """
        Converts input coordinates to pixel coords

        :param coords: An array of coordinates
        :type coords: Sequence of NX3

        ::
            ((long1, lat1, z1),
             (long2, lat2, z2),
             (long3, lat3, z3),
             )

        (z is ignored, and there is no z in the returned array)

        :returns: The pixel (x, y) coords as a similar Nx2 array of integer
                  (using the y = 0 at the top, and y increasing down)

        NOTE: The values between the minimum of a pixel value to less than the
              max of a pixel range are in that pixel, so  a point exactly at
              the minimum of the bounding box will be in the zeroth pixel, but
              a point  exactly at the max of the bounding box will be
              considered outside the map

        """
        coords = to_2d_coords(coords)  # strip off depth, if it's there

        # shift to center:
        coords = coords - self.center

        # scale to pixels:
        coords *= self.scale

        # shift to pixel coords
        coords += self.offset

        if asint:
            # NOTE: using "floor" as it rounds negative numbers towards -inf
            #      simple casting rounds toward zero
            #      we may need the negative coords to work right for locations
            #      off the grid.
            #      (used for the raster map code)
            return np.floor(coords, coords).astype(np.int32)
        else:
            return coords

    def to_pixel_2D(self, coords, asint=False):
        """
        # Fixme: depreciated, as we have to_2d_coords
        """
        raise NotImplementedError("no longer required, use to_pixel()")

    def to_pixel_multipoint(self, coords, asint=False):
        """
        does the to_pixel operation, but on a generic shaped array
        """
        coords = coords - self.center

        coords *= self.scale

        coords += self.offset

        if asint:
            return np.floor(coords, coords).astype(np.int32)
        else:
            return coords

    def to_lonlat(self, coords):
        """
        converts pixel coords to long-lat coords

        :param coords: An array of pixel coordinates (usually integer type
        :type coords: Sequence of NX2::
                          ((long1, lat1),
                          (long2, lat2),
                          (long3, lat3),
                          ...
                          )

        (as produced by to_pixel)

        :returns: The pixel coords as a similar Nx2 array of floating point
                  x,y coordinates
                  (using the y = 0 at the top, and y increasing down)

        NOTE: there is not depth in input -- pixels are always 2-d!


        NOTE: to_lonlat(to_pixel(coords)) != coords, due to rounding.
              If the input is integers, a 0.5 is added to "shift" the location
              to mid-pixel.
        """
        coords = np.asarray(coords)

        if np.issubdtype(coords.dtype, np.integer):
            # convert to float64:
            coords = coords.astype(np.float64)

            # add 0.5 to shift to center of pixel
            coords += 0.5

        # shift to pixel center coords
        coords -= self.offset

        # scale to lat-lon
        coords /= self.scale

        # shift from center:
        coords += self.center

        return coords


class FlatEarthProjection(GeoProjection):
    """
    class to define a "flat earth" projection:

        longitude is scaled to the cosine of the mid-latitude -- but that's it.

        not conforming to equal area, distance, bearing, or any other nifty
        map properties -- but easy to compute, and it looks OK.
    """

    @staticmethod
    def meters_to_lonlat(meters, ref_positions):
        """
        Converts from delta meters to delta latitude-longitude,
        using the Flat-Earth projection.

        dlat = dy * 8.9992801e-06
        dlon = dy * 8.9992801e-06 * cos(ref_lat)
        (based on previous GNOME value: and/or average radius of the earth of
        6366706.989  m)

        :param meters: Distances in meters
        :type meters: NX3 numpy array of (dx, dy, dz)
                      (dz is passed through untouched)

        :param ref_positions: Reference positions in degrees
        :type ref_positions: NX3, numpy array (Only lat is used here)

        :returns delta_lon_lat: Differential (delta) positional values
                                Nx3 numpy array of (delta-lon, delta-lat, delta-z)
        """

        # make a copy -- don't change meters
        delta_lon_lat = np.array(meters, dtype=np.float64)
        if len(delta_lon_lat.shape) == 1:
            if delta_lon_lat.shape[0] == 2:
                delta_lon_lat = delta_lon_lat.reshape(1, 2)
            else:
                delta_lon_lat = delta_lon_lat.reshape(1, 3)
        # reference is possible for reference positions
        ref_positions = np.asarray(ref_positions,
                                   dtype=np.float64)

        if len(ref_positions.shape) == 1:
            if ref_positions.shape[0] == 2:
                ref_positions = ref_positions.reshape(1, 2)
            else:
                ref_positions = ref_positions.reshape(1, 3)

        delta_lon_lat[:, :2] *= 8.9992801e-06
        delta_lon_lat[:, 0] /= np.cos(np.deg2rad(ref_positions[:, 1]))

        return delta_lon_lat

    @staticmethod
    def lonlat_to_meters(lon_lat, ref_positions):
        """
        Converts from delta longitude-latitude to delta meters, using the
        Flat-Earth projection. This should be a reversal of meters_to_latlon.

        This function mainly used for testing

        dy = dlon / 8.9992801e-06
        dx = dlat / ( 8.9992801e-06 * cos(ref_lat) )

        (based on previous GNOME value: and/or average radius of the earth
         of 6366706.989 m)

        NOTE: the input is in units of longitude and latitude, but they are
              relative -- no absolute -- so 0 means zero distance,
              not on the equator

        :param lon_lat: Distances in meters
        :type lon_lat: NX3 numpy array of (dlon, dlat, dz)
                       (dz is passed through untouched)

        :param ref_positions: Reference positions in degrees
        :type ref_positions: NX3, numpy array of (lon,lat,z)
                             (Only lat is used here)

        :returns delta_meters: Differential (delta) positional values in meters
                               Nx3 numpy array of (delta-x, delta-y, delta-z)
                               triples
        """
        # make a copy -- don't change input
        delta_meters = np.array(lon_lat, dtype=np.float64).reshape(-1, 3)

        # reference is possible for reference positions
        ref_positions = np.asarray(ref_positions,
                                   dtype=np.float64).reshape(-1, 3)

        delta_meters[:, :2] /= 8.9992801e-06
        delta_meters[:, 0] *= np.cos(np.deg2rad(ref_positions[:, 1]))

        return delta_meters

    @staticmethod
    def geodesic_sphere(lon, lat,
                        distance,
                        bearing):
        """
        Given a start point, initial bearing, and distance, returns the
        destination point along a (shortest distance) great circle arc --
        assuming a spherical earth. Similar to how GNOME does it.

        :param lon: longitude in decimal degrees.
        :param lat: latitude in decimal degrees.
        :param distance:  meters.
        :param bearing: in decimal degrees, measured clockwise from north.

        :returns longitude, latitude: in degrees.

        Code from Brian Zelenke

        NOTE: performance could be improved a lot here if need be
              (lots of data copies)
        """
        # EarthRadius = 6371010.0 # Value I"ve looked up
        # Matches the value used above -- GNOME value
        EarthRadius = 6366706.989

        # Convert from degrees to radians.
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        bearing = np.deg2rad(bearing)

        # Convert linear distance to angular distance (in radians).
        distance = distance / EarthRadius

        latout = np.arcsin(np.sin(lat) * np.cos(distance) + np.cos(lat) *
                           np.sin(distance) * np.cos(bearing))
        lonout = (lon +
                  np.arctan2(np.sin(bearing) * np.sin(distance) * np.cos(lat),
                             np.cos(distance) - np.sin(lat) * np.sin(latout)))

        # Convert from radians to degrees.
        lonout = np.rad2deg(lonout)
        latout = np.rad2deg(latout)

        return (lonout, latout)

    def set_scale(self, bounding_box, image_size=None):
        """
        set the scaling, etc. of the projection

        This should be called whenever the bounding box of the map,
        or the size of the image is changed

        :param bounding_box: bounding box of the visual portion of the map
        :type bounding_box: Structure of the form: ((min_long, min_lat),
                                                    (max_long, max_lat))

        :param image_size=None: the size of the image that will be drawn to.
                                if not given, the previous size will be used.
        """
        if image_size is None:
            image_size = self.image_size

        bb = np.asarray(bounding_box, dtype=np.float64)

        self.center = np.mean(bb, axis=0)
        self.offset = np.array(image_size, dtype=np.float64) / 2.0

        lon_scale = np.cos(np.deg2rad(self.center[1]))

        # compute BB to fit image
        h = bb[1, 1] - bb[0, 1]

        # width scaled to longitude
        w = (bb[1, 0] - bb[0, 0]) * lon_scale

        if w / h > image_size[0] / image_size[1]:
            s = image_size[0] / w
            self.scale = (s * lon_scale, -s)
        else:
            s = image_size[1] / h
            self.scale = (s * lon_scale, -s)

        # doing this at the end, in case there is a problem with the input.

        self.image_box = (self.to_lonlat((0, image_size[1])),
                          self.to_lonlat((image_size[0], 0)))
        self.image_size = image_size


class RectangularGridProjection(NoProjection):
    """
    Projection for lat-lon to pixel and back for a rectangular but not regular
    grid.

    i.e. a rectangular grid that can be defined by a single vector each of
         latitude and longitude

    This is a totally different type of projection.  It requires a linear
    interpolation for the latitude and longitude.

    Primarily used for making a raster land-water map from a non-regular
    rectangular grid.
    """

    def __init__(self, longitude, latitude):
        """
        Create a new Rectangular Grid projection

        :param longitude: the vector of longitudes
        :param latitude: the vector of latitudes

        It is assumed that the largest and smallest values define the
        bounds of the raster.
        """
        import scipy.interpolate

        latitude = np.array(latitude, dtype=np.float64)
        longitude = np.array(longitude, dtype=np.float64)

        self.max_lat_index = len(latitude) - 1  # height of bitmap
        self.max_lon_index = len(longitude) - 1  # width of bitmap

        self.min_lon = longitude.min()
        self.max_lon = longitude.max()

        self.min_lat = latitude.min()
        self.max_lat = latitude.max()

        # pixels = (np.arange(len(latitude)).reshape(-1, 1) *
        #           np.arange(len(longitude)))

        # Create interpolators:

        # fill value of None means use NaN
        self._lon_to_pixel_interp = (scipy.interpolate
                                     .interp1d(longitude,
                                               np.arange(len(longitude)),
                                               kind='linear',
                                               copy=True,
                                               bounds_error=False,
                                               fill_value=None))

        self._lat_to_pixel_interp = (scipy.interpolate
                                     .interp1d(latitude,
                                               np.arange(len(latitude)),
                                               kind='linear',
                                               copy=True,
                                               bounds_error=False,
                                               fill_value=None))

        self._pixel_to_lon_interp = (scipy.interpolate
                                     .interp1d(np.arange(len(longitude)),
                                               longitude,
                                               kind='linear',
                                               copy=True,
                                               bounds_error=False,
                                               fill_value=None))

        self._pixel_to_lat_interp = (scipy.interpolate
                                     .interp1d(np.arange(len(latitude)),
                                               latitude,
                                               kind='linear',
                                               copy=True,
                                               bounds_error=False,
                                               fill_value=None))

    def set_scale(self, bounding_box, image_size=None):
        """
        Does nothing
        """
        raise NotImplementedError('you can not reset the scale on a '
                                  'RectangularGridProjection object\n'
                                  'create a new one if you need a new scale')

    def to_pixel(self, coords, asint=False):
        """
        Returns the pixel coordinates in the grid for the given
        lat-lon location.

        :param coords: The coords to project
        :type coords: Nx3 numpy array or compatible sequence (lon, lat, depth)

        :param asint: Flag to set whether to convert to a integer or not
                      default is to leave it as the same type it came in,
                      so you can have fractional pixels
        """
        coords = to_2d_coords(coords)
        # All computation in floats -- convert to int if asked for
        coords = np.asarray(coords, dtype=np.float64)
        pixel_coords = np.zeros_like(coords, dtype=np.float64)

        np.putmask(coords,
                   coords < (self.min_lon, self.min_lat),
                   (self.min_lon, self.min_lat))
        np.putmask(coords,
                   coords > (self.max_lon, self.max_lat),
                   (self.max_lon, self.max_lat))

        np.clip(coords,
                (self.min_lon, self.min_lat),
                (self.max_lon, self.max_lat),
                out=coords)

        pixel_coords[:, 0] = self._lon_to_pixel_interp(coords[:, 0])
        pixel_coords[:, 1] = (self.max_lat_index -
                              self._lat_to_pixel_interp(coords[:, 1]))

        if asint:
            # NOTE: using "floor" as it rounds negative numbers towards -inf
            #       simple casting rounds toward zero
            #       we may need the negative coords to work right for locations
            #       off the grid.
            #       (used for the raster map code)
            pixel_coords = (np.floor(pixel_coords, pixel_coords)
                            .astype(np.int32))

        return pixel_coords

    def to_pixel_2D(self, coords, asint=False):
        """
        same as to_pixel, but expects only (lon, lat) coords as input.

        :param coords: The coords to project
        :type coords: Nx2 numpy array or compatible sequence (lon, lat)

        :param asint: Flag to set whether to convert to a integer or not
                      default is to leave it as the same type it came in,
                      so you can have fractional pixels
        """
        raise NotImplementedError

    def to_lonlat(self, coords):
        """
        Converts pixel coords to long-lat coords

        :param coords: An array of pixel coordinates (as produced by to_pixel)
        :type coords: Sequence of Nx2 (usually integer type)::
                          ((long1, lat1),
                          (long2, lat2),
                          (long3, lat3),
                          ...
                          )

        :returns: the pixel coords as a similar Nx2 array of floating point
                  x,y coordinates
                  (using the y = 0 at the top, and y increasing down)

        NOTE: there is not depth in input -- pixels are always 2-d!

        NOTE: to_lonlat(to_pixel(coords)) != coords, due to rounding.
              If the input is integers, a 0.5 is added to "shift" the location
              to mid-pixel.
         """
        coords = to_2d_coords(coords)

        if np.issubdtype(coords.dtype, np.integer):
            # convert to float64:
            coords = coords.astype(np.float64)

            # add 0.5 to shift to center of pixel
            coords += 0.5

        # out of bounds gets clipped to boundary

        np.clip(coords,
                (0, 0), (self.max_lon_index, self.max_lat_index),
                out=coords)

        # interpolate to lon-lat_coords
        lon = self._pixel_to_lon_interp(coords[:, 0])
        lat = self._pixel_to_lat_interp(self.max_lat_index - coords[:, 1])

        return np.c_[lon, lat]


class RegularGridProjection(GeoProjection):
    """
    projection for lat-lon to pixel and back for a pre-defined regular grid.

    This differs from the other projections in that it doesn't try to
    match the bounding box aspect ratio -- it simply uses the one
    already defined by the grid.

    You  could use a RectangularGridProjection here as well, but this is
    simpler and should be faster.
    """

    def set_scale(self, bounding_box, image_size=None):
        """
        Set the scaling, etc. of the projection

        This should be called whenever the bounding box of the map,
        or the size of the image is changed

        :param bounding_box: Bounding box of the visual portion of the map
        :type bounding_box: Structure of the form: ((min_long, min_lat),
                                                    (max_long, max_lat))

        :param image_size=None: The size of the image that will be drawn to.
                                if not given, the previous size will be used.
        """
        if image_size is None:
            image_size = self.image_size

        bounding_box = np.asarray(bounding_box, dtype=np.float64)

        self.center = np.mean(bounding_box, axis=0)
        self.offset = np.array(image_size, dtype=np.float64) / 2

        h = bounding_box[1, 1] - bounding_box[0, 1]
        w = bounding_box[1, 0] - bounding_box[0, 0]

        self.scale = (image_size[0] / w, - image_size[1] / h)

        # doing this at the end, in case there is a problem with the input.
        self.image_size = image_size
