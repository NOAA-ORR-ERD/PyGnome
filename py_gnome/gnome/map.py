 #!/usr/bin/env python

"""
 An implementation of the GNOME land-water map.
 
 This is a port of the C++ raster map approach
 
 
 NOTES:

Should we just use non-projected coordinates for the raster map? --
   it makes for a little less computation at every step.

Do we want to treat lakes differently than regular water?

New features:
 - Map now handles spillable area and map bounds as polygons
 - raster is the same aspect ratio os the land
 - internally, raster is a numpy array
 - land raster is only as big as the land -- if the map bounds are bigger, extra space is not in the land map
    Question: what if map-bounds is smaller than land? wasted bitmap space? (though it should work)

"""
 
import sys
import os
import numpy as np
import random
from numpy import ma
from PIL import Image, ImageDraw

from gnome.utilities import map_canvas
from gnome.basic_types import world_point_type, status_on_land

from hazpy.file_tools import haz_files
from gnome.utilities.geometry import BBox
from gnome.utilities.geometry.PinP import CrossingsTest as point_in_poly

from gnome.utilities.geometry.polygons import PolygonSet

#from hazpy.geometry import polygons

            
class GnomeMap:
    """
    The very simplest map for GNOME -- all water,
    only a bounding box for the map bounds.
    
    This also serves as a description of the inteface
    """
    
    refloat_halflife = None # note -- no land, so never used

    def __init__(self, map_bounds = None):
        """
        This __init__ will be different for other implementations
        
        map_bounds is the bounds of the map:
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
        returns True is the location is on the map

        coord is a (long, lat) location.
                
        note: should this support no-rectangular maps? -- point in polygon?
        """
        """
        returns true is the spill position is in the allowable spill area
        
        This may not be the same as in_water!
        
        """
        print "in on_map"
        print self.map_bounds
        print coord
        print point_in_poly(self.map_bounds, coord)
        return point_in_poly(self.map_bounds, coord)
        
    def on_land(self, coord):
        """
        returns a Boolean result:
        
        True if the point is on land,
        False if the point is on water
        
        coord is a (long, lat) location
        
        Always returns False-- no land in this implementation
        
        ## note: what should this give if off map?
        
        """
        return False

    def in_water(self, coord):
        """
        returns a Boolean result:
        
        True if the point is in the water,
        False if the point is on land (or off map?)
        
        coord is a (long, lat) location
        
        """
        return self.on_map(coord)
    
    def allowable_spill_position(self, coord):
        """
        returns a Boolean result:
        
        True if the point is an allowable spill position
        False if the point is not an allowable spill position

        (Note: it could be either off the map, or in a location that spills
        aren't allowed)
        
        coord is a (long, lat) location
        
        """
        return self.on_map(coord)

    def beach_LEs(self, spill):
        """
        beach_LEs
        
        determines which LEs were or weren't beached.
        
        Any that are beached have the beached flag set, and a "last know water position" (lkwp) is computed
        
        param: spill  - a spill object -- it must have:
               'prev_position', 'positions', 'last_water_pt' and 'status_code' data arrays
        
        The default base map has no land, so nothing changes
        """
        return None
        
        
        
import land_check

class RasterMap(GnomeMap):
    """
    A land water map implemented as a raster
    
    This one uses us a numpy array of uint8 -- so there are 8 bits to choose from...
    
    It requires a constant refloat half-life
    
    This will usually be initialized in a sub-class (froma BNA, etc)
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
        
        refloat_halflife is the halflife for refloating off land -- given in seconds
                This is assumed to be the same everywhere at this point
        
        bitmap_array is a numpy array that stores the land-water map
        
        projection is a gnome.map_canvas.Projection object -- used to convert from lat-long to pixels in the array
        
        map_bounds is the polygon boudning the map -- could be larger or smaller than the land raster
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
        print "in _on_land_pixel", coord 
        print self.bitmap.shape, self.bitmap.dtype 
        print self.bitmap[coord[0], coord[1]]
        print self.bitmap
        try:
            return self.bitmap[coord[0], coord[1]] & self.land_flag
        except IndexError:
            return 0 # not on land if outside the land raster. (Might be off the map!) 

    ## fixme: just for compatibility with old code -- nothing outside this class should know about pixels...
    ##        on_land_pixel = _on_land_pixel
    
    def on_land(self, coord):
        """
        returns 1 if point on land
        returns 0 if not on land
        
        coord is (long, lat) location
        
        """
        print "in on_land"
        print coord
        return self._on_land_pixel(self.projection.to_pixel(coord))
    
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
        #print "in RasterMap.in_water_pixel", coord
        #print self.bitmap.shape
        #print self.bitmap[coord[0], coord[1]]
        try:
            return not (self.bitmap[coord[0], coord[1]] & self.land_flag)
        except IndexError:
            # Note: this could be off map, which may be a different thing than on land....
            #       but off the map should have been tested first
            return True

    def in_water(self, coord):
        """
        returns true if the point given by coord is in the water
        
        checks if it's on the map, first.
        """
        if not self.on_map(coord):
            return False
        else:
            return self._in_water_pixel(self.projection.to_pixel(coord))
    
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

    def beach_elements(self, start_positions, end_positions, status_codes):
        
        """ 
        Beaches all the elements that have crossed over land
        
        Checks to see if any land is crossed between the start and end positions.
           if land is crossed, the beached flag is set, and the last  water postion returned
        
        param: start_positions -- (long, lat) positions elements begin the time step at (NX2 array)
        param: end_positions   -- (long, lat) positions elements end the time step at (NX2 array)
        param: status_codes    -- the status flag for the LEs

        : returns last_water_positions 
        
        """
        
        # project the positions:
        start_positions_px = self.projection.to_pixel(start_positions)
        end_positions_px   = self.projection.to_pixel(end_positions)

        last_water_positions_px = np.array_like(start_positions_px)
        # do the inner loop
        for i in range(len(start_positions)):
            #do the check...
            result = land_check.find_first_pixel(self.bitmap, start_positions_px[i], end_positions_px[i], draw=False)
            if result is not None:
                last_water_positions_px[i], end_positions_px[i] = result
                status_codes[i] = STATUS_CODE_BEACHED
        # put the data back in the arrays
        beached_mask =  status_codes[i] == STATUS_CODE_BEACHED
        end_positions[beached_mask] = self.projection.to_lat_long(end_positions_px[beached_mask])
        last_water_positions = np.zeros_like(start_positions)
        last_water_positions = self.projection.to_lat_long(end_positions_px[beached_mask])
        
        return last_water_positions
              
                            
            
    def beach_LEs(self, spill):
        """
        beach_LEs
        
        determines which LEs were or weren't beached.
        
        Any that are beached have the beached flag set, and a "last know water position" (lkwp) is computed
        
        param: spill  - a spill object -- it must have:
               'prev_position', 'positions', 'last_water_pt' and 'status_code' data arrays
        
        This version uses a modified Bresenham algorythm to find out which pixels the LE may have crossed.
        
        """
        # pull the data from the spill 
        ## is the last water point the same as the previos position? why not?? if beached, it won't move, if not, then we can use it?
        start_pos, end_pos, status_code = spill.get_data_arrays( ('positions',
                                                                  'prev_positions',
                                                                  'last_water_pt',
                                                                  'status_code',
                                                                  ) )
        # transform to pixel coords:
        start_pos = self.projection.to_pixel(start_pos)
        end_pos   = self.projection.to_pixel(end_pos)
        
        # call the actual hit code:
        # the status_code and last_water_point arrays are altered in-place
        self.check_land(self.bitmap, start_pos, end_pos, status_code, last_water_pt)
                
        #transform the points back to lat-long.
        ##fixme -- only transform those that were changed -- no need to introcude rounding error otherwise
        beached = status_code == STATUS_CODE_BEACHED
        end_pos[beached]= self.projection.to_lat_long(end_pos[beached])
        last_water_pt[beached] = self.projection.to_lat_long(end_pos[beached])
        
        
        
        
    
    def allowable_spill_position(self, coord):
        """
        returns true is the spill position is in the allowable spill area
        
        This may not be the same as in_water!
        
        """
        if self.on_map(coord):
            if not self.on_land(coord):
                if self.spillable_area is None:
                    return True
                else:
                    return point_in_poly(self.spillable_area, coord)
            else:
                 return False
        else:
            return False
    def to_pixel_array(self, coords):
        """ 
        Projects an array of (lon, lat) tuples onto the bitmap, and modifies it in 
        place to hold the corresponding projected values.
        
        param: coords is a Nx2 numpy array
        
        returns: NX2 numpy array of integer pixel values
        """
        
        return self.projection.to_pixel(coords)

    def movement_check(self, spill):
        """ 
        After moving a spill (after superposing each of the movers' contributions),
        we determine which of the particles have been landed, ie., that are in the
        current time step on land but were not in the previous one. Chromgph is a list
        of boolean values, determining which of the particles need treatment. Particles
        that have been landed are beached.
        
        param: spill -- a gnome.spill object (with LE info, etc)
        """        
        # make a regular Nx2 numpy array 
        coords = np.copy(spill.npra['p']).view(world_point_type).reshape((-1, 2),)
        coords = self.to_pixel_array(coords)
        chromgph = self._on_land_pixel_array(coords)
        sra = spill.npra['status_code']
        for i in xrange(0, spill.num_particles):
            if chromgph[i]:
                sra[i] = status_on_land
        if spill.chromgph == None:
            spill.chromgph = chromgph
        merg = [int(chromgph[x] and not spill.chromgph[x]) for x in xrange(0, len(chromgph))]
        self.chromgph = chromgph
        return merg

    def movement_check2(self, current_pos, prev_pos, status):
        """
        checks the movement of the LEs to see if they have:
        
        hit land or gone off the map. The status code is changed if they are beached or off map
        
        param: current_pos -- a Nx2 array of lat-long coordinates of the current positions
        param: prev_pos -- a Nx2 array of lat-long coordinates of the previous positions
        param: status -- a N, array of status codes.
        """
        # check which ones are still on the map:
        on_map = self.on_map(current_pos)
        # check which ones are on land:
        pixel_coords = self.to_pixel_array(current_pos)
        on_land = self.on_land(current_pos)
        
        

        
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
        Creates a GnomeMap (specifically a RasterMap) from a bna file
        
        bna_file: full path to a bna file
        
        refloat_halflife: the half-life (in seconds) for the re-floating.
        
        raster_size: total number of pixels (bytes) to make the raster -- the actual size
        will match the aspect ratio of the bounding box of the land
        
        It is expected that you will get the spillable area and map bounds from the BNA -- if they exist
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
        canvas.draw_land(just_land)
        canvas.save("Failedtest.png")

        # uncomment for diagnostics
        # canvas.save("TestLandWaterMap.png")
        
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

