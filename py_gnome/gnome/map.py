 #!/usr/bin/env python

"""
An implementation of the GNOME land-water map.

This is a re-write of the C++ raster map approach

NOTES:
 - Should we just use non-projected coordinates for the raster map?
   It makes for a little less computation at every step.
 - Do we want to treat lakes differently than regular water?

New features:
 - Map now handles spillable area and map bounds as polygons
 - raster is the same aspect ratio as the land
 - internally, raster is a numpy array
 - land raster is only as big as the land -- if the map bounds are bigger, extra space is not in the land map
    Question: what if map-bounds is smaller than land? wasted bitmap space? (though it should work)
"""
 
import numpy as np

import gnome
from gnome.utilities import map_canvas
from gnome.basic_types import world_point_type, oil_status

from gnome.utilities.file_tools import haz_files
#from gnome.utilities.geometry import BBox
from gnome.utilities.geometry.PinP import CrossingsTest as point_in_poly

from gnome.utilities.geometry.polygons import PolygonSet

            
class GnomeMap(object):
    """
    The very simplest map for GNOME -- all water
    with only a bounding box for the map bounds.
    
    This also serves as a description of the interface
    """
    
    refloat_halflife = None # note -- no land, so never used

    def __init__(self, map_bounds = None):
        """
        This __init__ will be different for other implementations
        
        :param map_bounds: the bounds of the map:
            ( (x1,y1), (x2,y2),(x3,y3),..)
            An NX2 array of points that describe a polygon
            if no map bounds is provided -- the whole world is valid
        
        """
        if map_bounds is not None:
            self.map_bounds = np.asarray(map_bounds, dtype=np.float64).reshape(-1, 2)
        else:
            # using -360 to 360 to allow stuff to cross the dateline..
            self.map_bounds = np.array( ( (-360, 90),
                                          ( 360, 90),
                                          ( 360, -90),
                                          (-360, -90),
                                          ), dtype=np.float64 )

    def on_map(self, coord):
        """
        .. note::
            should this support non-rectangular maps? -- point in polygon?
                
        :param coord: (long, lat, depth) location
        :return:
         - True if the location is on the map
        """
        return point_in_poly(self.map_bounds, coord[:2])

    def on_land(self, coord):
        """
        .. note::
            what should this give if it is off map?
        
        :param coord: (long, lat, depth) location
        
        :return:
         - Always returns False-- no land in this implementation
        
        """
        return False

    def in_water(self, coord):
        """
        :param coord: (long, lat, depth) location
        
        :return:
         - True if the point is in the water,
         - False if the point is on land (or off map?)
        
        """
        return self.on_map(coord)

    def allowable_spill_position(self, coord):
        """
        .. note::
            it could be either off the map, or in a location that spills aren't allowed
        
        :param coord: (long, lat, depth) location
        
        :return:
         - True if the point is an allowable spill position
         - False if the point is not an allowable spill position
        """
        return self.on_map(coord)

    def beach_elements(self, spill):
        """
        Determines which LEs were or weren't beached.
        
        :param spill: an object of or inheriting from :class:`gnome.spill.Spill`
            This map class  has no land, so nothing changes
        """
        return None

    def refloat_elements(self, spill):
        """
        This method performs the re-float logic -- changing the element status flag,
        and moving the element to the last known water position
        
        .. note::
            This map class has no land, and so is a no-op.
        
        :param spill: an object of or inheriting from :class:`gnome.spill.Spill`
            This object holds the elements that need refloating
        """
        pass


from gnome.cy_gnome import cy_land_check as land_check
# from gnome import land_check

class RasterMap(GnomeMap):
    """
    A land water map implemented as a raster
    
    This one uses us a numpy array of uint8 -- so there are 8 bits to choose from...
    
    It requires a constant refloat half-life
    
    This will usually be initialized in a sub-class (from a BNA, etc)
    """

    ## NOTE: spillable area can be both larger and smaller than land raster:
    ##       map bounds can also be larger or smaller:
    ##            both are done with a point in polygon check
    ##       if map is smaller than land polygons, no need for raster to be
    ##       larger than map -- but no impimented yet.
    
    ## flags for what's in the bitmap
    ## in theory -- it could be used for other data:
    ##  refloat, other properties?   
    ## note the BW map_canvas only does 1, though.

    land_flag  = 1    
    def __init__(self,
                 refloat_halflife,    #seconds
                 bitmap_array,
                 projection,
                 map_bounds = None,   # defaults to bounding box of raster
                 spillable_area=None, # defaults to any water
                 ):
        """
        create a new RasterMap
        
        :param refloat_halflife: The halflife for refloating off land -- given in seconds.
                This is assumed to be the same everywhere at this point
        
        :param bitmap_array: A numpy array that stores the land-water map
        
        :param projection: A gnome.map_canvas.Projection object -- used to convert from lat-long to pixels in the array
        
        :param map_bounds: The polygon bounding the map -- could be larger or smaller than the land raster
        """
        
        self.refloat_halflife = refloat_halflife
        self.bitmap = bitmap_array
        self.projection = projection
        if map_bounds is not None:
            # make sure map bounds in a numpy array
            self.map_bounds = np.asarray(map_bounds, dtype=np.float64).reshape(-1, 2)
        else:
            self.map_bounds = None
        if spillable_area is not None:
            # make sure spillable_area is a numpy array
            self.spillable_area = np.asarray(spillable_area, dtype=np.float64).reshape(-1, 2)
        else:
            self.spillable_area = None

    def _on_land_pixel(self, coord):
        """
        returns 1 if the point is on land, 0 otherwise
        
        coord is on pixel coordinates of the bitmap
        
        """  
        try:
            return self.bitmap[coord[0], coord[1]] & self.land_flag
        except IndexError:
            return False # not on land if outside the land raster. (Might be off the map!) 

    ## fixme: just for compatibility with old code -- nothing outside this class should know about pixels...
    ##        on_land_pixel = _on_land_pixel
    
    def on_land(self, coord):
        """
        :param coord: (long, lat, depth) location
        
        :return:
         - 1 if point on land
         - 0 if not on land
        
        """
        return self._on_land_pixel(self.projection.to_pixel(coord)[0]) # to_pixel converts to array of points...
    
    def _on_land_pixel_array(self, coords):
        """
        determines which LEs are on lond
        
        param: coords  Nx2 numpy integer array of pixel coords (matching the bitmap)
        
        returns: a (N,) array of booleans- true for the particles that are on land
        """
        mask = map(point_in_poly, [self.map_bounds,]*len(coords), coords)
        racpy = np.copy(coords)[mask]
        mskgph = self.bitmap[racpy[:,0], racpy[:,1]]
        chrmgph = np.array([0,]*len(coords))
        chrmgph[np.array(mask)] = mskgph
        return chrmgph

    def _in_water_pixel(self, coord):
        try:
            return not (self.bitmap[coord[0], coord[1]] & self.land_flag)
        except IndexError:
            # Note: this could be off map, which may be a different thing than on land....
            #       but off the map should have been tested first
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
            return self._in_water_pixel(self.projection.to_pixel(coord, asint=True)[0]) # to_pixel makes a NX2 array
    
#    def beach_element(self, p, lwp):
#        """ 
#        Beaches an element that has been landed.
#        
#        param: p: current position (see basic_types.world_point dtype)
#        param: lwp: last water position (see basic_types.world_point dtype)
#        
#        """
#        in_water = self.in_water
#        displacement = ((p['p_long'] - lwp['p_long']), (p['p_lat'] - lwp['p_lat']))
#        while not in_water((p['p_long'], p['p_lat'])):
#            displacement = (displacement[0]/2, displacement[1]/2)
#            p['p_long'] = lwp['p_long'] + displacement[0]
#            p['p_lat'] = lwp['p_lat'] + displacement[1]

#    def beach_elements(self, start_positions, end_positions, last_water_positions, status_codes):
#        
#        """ 
#        Beaches all the elements that have crossed over land
#        
#        Checks to see if any land is crossed between the start and end positions.
#           if land is crossed, the beached flag is set, and the last  water postion returned
#        
#        param: start_positions -- (long, lat) positions elements begin the time step at (NX2 array)
#        param: end_positions   -- (long, lat) positions elements end the time step at (NX2 array)
#        param: last_water_positions -- (long, lat) positions elements end the time step at (NX2 array)
#        param: status_codes    -- the status flag for the LEs
#
#        last_water_positions and status_codes are changed in-place
#        """
#        
#        # project the positions:
#        start_positions_px = self.projection.to_pixel(start_positions)
#        end_positions_px   = self.projection.to_pixel(end_positions)
#
#        last_water_positions_px = np.zeros_like(start_positions_px)
#        # do the inner loop
#        for i in range( len(start_positions) ):
#            #do the check...
#            result = land_check.find_first_pixel(self.bitmap, start_positions_px[i], end_positions_px[i], draw=False)
#            if result is not None:
#                last_water_positions_px[i], end_positions_px[i] = result
#                status_codes[i] = oil_status.on_land
#
#        # put the data back in the arrays
#        beached_mask =  ( status_codes == oil_status.on_land )
#        end_positions[beached_mask] = self.projection.to_lonlat(end_positions_px[beached_mask])
#        last_water_positions[beached_mask] = self.projection.to_lonlat(last_water_positions_px[beached_mask])
#        
#        return None
#                                          
            
    def beach_elements(self, spill):
        """
        Determines which LEs were or weren't beached.
        
        Any that are beached have the beached flag set, and a "last known water position" (lkwp) is computed
        
        This version uses a modified Bresenham algorithm to find out which pixels the LE may have crossed.
        
        :param spill: an object of or inheriting from :class:`gnome.spill.Spill`
            It must have the following data arrays:
            ('prev_position', 'positions', 'last_water_pt', 'status_code')
        """
        # pull the data from the spill 
        ## is the last water point the same as the previos position? why not?? if beached, it won't move, if not, then we can use it?
        start_pos     = spill['positions']
        next_pos      = spill['next_positions']
        status_codes  = spill['status_codes']
        last_water_positions = spill['last_water_positions']
        
        
        # transform to pixel coords:
        # NOTE: must be integers!
        start_pos_pixel = self.projection.to_pixel(start_pos, asint=True)
        next_pos_pixel  = self.projection.to_pixel(next_pos, asint=True)
        last_water_pos_pixel = self.projection.to_pixel(last_water_positions, asint=True)
        
        # call the actual hit code:
        # the status_code and last_water_point arrays are altered in-place
        # only check the ones that aren't already beached?
        self.check_land(self.bitmap, start_pos_pixel, next_pos_pixel, status_codes, last_water_pos_pixel)

        #transform the points back to lat-long.
        beached = ( status_codes == oil_status.on_land )
        next_pos[beached, :2]= self.projection.to_lonlat(next_pos_pixel[beached])
        last_water_positions[beached, :2] = self.projection.to_lonlat(last_water_pos_pixel[beached,:2])

        ##fixme -- add off-map check here


    def check_land(self, raster_map, positions, end_positions, status_codes, last_water_positions):
        """
        Do the actual land-checking.  This method calls a Cython version:
            gnome.cy_gnome.cy_land_check.check_land()

        The arguments **status_codes**, **positions** and **last_water_positions** are altered in place.
        """
        gnome.cy_gnome.cy_land_check.check_land(raster_map,
                                                positions,
                                                end_positions,
                                                status_codes,
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
                    return point_in_poly(self.spillable_area, coord[:2]) # point_in_poly is 2-d
            else:
                 return False
        else:
            return False

    def to_pixel_array(self, coords):
        """ 
        Projects an array of (lon, lat) tuples onto the bitmap, and modifies it in 
        place to hold the corresponding projected values.
        
        :param coords:  a numpy array of (lon, lat, depth) points
        
        :return: a numpy array of (x, y) pixel values
        """
        
        return self.projection.to_pixel(coords)

#    def movement_check(self, spill):
#        """ 
#        After moving a spill (after superposing each of the movers' contributions),
#        we determine which of the particles have been landed, ie., that are in the
#        current time step on land but were not in the previous one. Chromgph is a list
#        of boolean values, determining which of the particles need treatment. Particles
#        that have been landed are beached.
#        
#        param: spill -- a gnome.spill object (with LE info, etc)
#        """        
#        # make a regular Nx2 numpy array 
#        coords = np.copy(spill.npra['p']).view(world_point_type).reshape((-1, 2),)
#        coords = self.to_pixel_array(coords)
#        chromgph = self._on_land_pixel_array(coords)
#        sra = spill.npra['status_code']
#        for i in xrange(0, spill.num_particles):
#            if chromgph[i]:
#                sra[i] = oil_status.on_land
#        if spill.chromgph == None:
#            spill.chromgph = chromgph
#        merg = [int(chromgph[x] and not spill.chromgph[x]) for x in xrange(0, len(chromgph))]
#        self.chromgph = chromgph
#        return merg
#
#    def movement_check2(self, current_pos, prev_pos, status):
#        """
#        checks the movement of the LEs to see if they have:
#        
#        hit land or gone off the map. The status code is changed if they are beached or off map
#        
#        param: current_pos -- a Nx2 array of lat-long coordinates of the current positions
#        param: prev_pos -- a Nx2 array of lat-long coordinates of the previous positions
#        param: status -- a N, array of status codes.
#        """
#        # check which ones are still on the map:
#        on_map = self.on_map(current_pos)
#        # check which ones are on land:
#        pixel_coords = self.to_pixel_array(current_pos)
#        on_land = self.on_land(current_pos)
#        
#        

        
class MapFromBNA(RasterMap):
    """
    A raster land-water map, created from a BNA file
    """
    def __init__(self,
                 bna_filename,
                 refloat_halflife, #seconds
                 raster_size = 1024*1024, # default to 1MB raster
                 ):
        """
        Creates a GnomeMap (specifically a RasterMap) from a bna file.
        It is expected that you will get the spillable area and map bounds from the BNA -- if they exist
        
        :param bna_file: full path to a bna file
        :param refloat_halflife: the half-life (in seconds) for the re-floating.
        :param raster_size: the total number of pixels (bytes) to make the raster -- the actual size
        will match the aspect ratio of the bounding box of the land
        """
        polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")

        # find the spillable area and map bounds:
        # and create a new polygonset without them
        #  fixme -- adding a "pop" method to PolygonSet might be better
        #      or a gnome_map_data object...
        just_land = PolygonSet() # and lakes....
        spillable_area = None
        map_bounds = None
        for p in polygons:
            if p.metadata[1].lower() == "spillablearea":
                spillable_area = p
            elif p.metadata[1].lower() == "map bounds":
                map_bounds = p
            else:
                just_land.append(p)
        # now draw the raster map with a map_canvas:
        #determine the size:
        BB = just_land.bounding_box
        # create spillable area and  bounds if they weren't in the BNA
        if map_bounds is None:
            map_bounds = BB.AsPoly()
        if spillable_area is None:
            spillable_area = map_bounds

        ##fixme: should we just stick with non-projected coord for the raster map?
        W, H = BB.Width, BB.Height
        # stretch the bounding box, to get approximate aspect ratio in projected coords.
        aspect_ratio = np.cos(BB.Center[1] * np.pi / 180 ) * BB.Width / BB.Height
        w = int(np.sqrt(raster_size*aspect_ratio))
        h = int(raster_size / w)

        canvas = map_canvas.BW_MapCanvas( (w, h) )
        canvas.set_land(just_land)
        canvas.draw_background()
        #canvas.save_background("raster_map_test.png")

        ## get the bitmap as a numpy array:
        bitmap_array = canvas.as_array()
        
        # __init__ the  RasterMap
        RasterMap.__init__(self,
                           refloat_halflife, #hours
                           bitmap_array,
                           canvas.projection,
                           map_bounds, 
                           spillable_area, 
                           )
        
        return None

#########################################
##
## Original Code -- depricated now
##    
#########################################    
#class gnome_map(map_canvas.MapCanvas):
#    """
#        Basics pyGNOME color bitmap.
#        (End-user visualization.)
#    """     
#    def __init__(self, image_size, bna_filename, color_mode='RGB'):
#        """
#            Initializes color map attributes. Calls on parent class initialization 
#            method in order to handle projection scaling.
#        """
#        map_canvas.MapCanvas.__init__(self,
#                                      image_size,
#                                      projection=map_canvas.FlatEarthProjection,
#                                      mode=color_mode)
#        self.polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")
#        self.filename = bna_filename        
#        self.draw_land(self.polygons)
#
#    def to_pixel(self, coord):
#        """ Projects a (lon, lat) tuple onto the bitmap, and returns the resultant tuple. """
#        coord = tuple(self.projection.to_pixel(np.array((coord[0], coord[1]))))
#        coord = (int(coord[0]), int(coord[1]))
#        return coord
#        
#    def to_pixel_array(self, coords):
#        """ 
#            Projects an array of (lon, lat) tuples onto the bitmap, and modifies it in 
#            place to hold the corresponding projected values.
#        """
#        coords['p_long'] -= self.projection.center[0]
#        coords['p_lat']  -= self.projection.center[1]
#        coords['p_long'] *= self.projection.scale[0]
#        coords['p_lat']  *= self.projection.scale[1]
#        coords['p_long'] += self.projection.offset[0]
#        coords['p_lat']  += self.projection.offset[1]
#        coords['p_long'] = np.round(coords['p_long']).astype(np.int)
#        coords['p_lat']  = np.round(coords['p_lat']).astype(np.int)
#        
#    def _type(self):
#        """ This requires an explanation. """
#        return ' color bitmap'
#        
#class lw_map(gnome_map):
#
#    """land-water bitmap."""
#        
#    background_color = 0
#    lake_color = 0
#    land_color = 1
#    
#    def __init__(self, image_size, bna_filename, refloat_halflife, color_mode='1'):
#        """
#            Initializes land-water map attributes. Calls on parent class initialization 
#            method in order to handle projection scaling. Caches its bounding_box so that
#            it doesn't need to be computed repeatedly.
#        """
#        gnome_map.__init__(self, image_size, bna_filename, color_mode)
#        self.bounding_box = self.polygons.bounding_box
#        self.refloat_halflife = refloat_halflife
#        self.spills = []
#
#    def _type(self):
#        """ Returns the map type. (Either 'color' or 'land-water'.) """
#        return ' land-water bitmap'
#        
#    def on_map(self, pixel_coord):
#        """ 
#            Given a tuple in pixel coordinates, determines whether the position is on the map.
#            It is actually not behaving correctly at the moment: the map bounds and the bounding box
#            may not necessarily coincide. Needs fixing!
#        """
#        bounding_box = self.bounding_box
#        if  ( pixel_coord[0] > bounding_box[0][0] and 
#              pixel_coord[0] < bounding_box[1][0] and
#              pixel_coord[1] > bounding_box[0][1] and
#              pixel_coord[1] < bounding_box[1][1] ):
#            return True
#        return False
#
#    def on_land(self, coord):
#        """ Given lat-lon coordinates, determines whether the position is on land. """
#        return not self.in_water(coord)
#
#    def on_land_pixel(self, coord):
#        """ Given a tuple in pixel coordinates, determines whether the position is on land. """
#        return not self.in_water_pixel(coord)
#        
#    def in_water(self, coord):
#        """ Given lat-lon coordinates, determines whether the position is in water. """
#        coord = self.to_pixel(coord)
#        if not self.on_map(coord):
#            return False
#        try:
#            chrom = self.image.getpixel(coord)
#            if not chrom:
#                return True
#            else:
#                return False
#        except:
#            print 'exception!',  sys.exc_info()[0]
#            return False
#            
#    def in_water_pixel(self, coord):
#        """ Given a tuple in pixel coordinates, determines whether the position is in water. """
#        coord = coord.tolist()
#        if not self.on_map(coord):
#            return False
#        try:
#            chrom = self.image.getpixel(coord)
#            if not chrom:
#                return True
#            else:
#                return False
#        except:
#            print 'exception!',  sys.exc_info()[0]
#            return False
#
#    def allowable_spill_position(self, coord):
#        """
#        Determines whether a position given in lat-lon coordinates is an allowable spill location.
#        """
#        return self.in_water(coord)
#        
#    def set_spill(self, coord, num_particles, release_time):
#        ##fixme: why is this in the map class?
#        """ 
#        Sets a spill.
#        
#        param: coord: (lon, lat)
#        param: release_time in seconds.
#        """
#        if not self.allowable_spill_position(coord):
#            print  "spill " + str(dict((('position', coord), ('num_particles', num_particles), ('release_time', release_time)))) + " ignored."
#        else:
#            self.spills += [(coord, num_particles, release_time)]

#
# #   def beach_element(self, p, lwp):
#        """ 
#        Beaches an element that has been landed.
#        
#        param: p: current position (see basic_types.world_point dtype)
#        param: lwp: last water position (see basic_types.world_point dtype)
#        
#        """
#        in_water = self.in_water
#        displacement = ((p['p_long'] - lwp['p_long']), (p['p_lat'] - lwp['p_lat']))
#        while not in_water((p['p_long'], p['p_lat'])):
#            displacement = (displacement[0]/2, displacement[1]/2)
#            p['p_long'] = lwp['p_long'] + displacement[0]
#            p['p_lat'] = lwp['p_lat'] + displacement[1]

