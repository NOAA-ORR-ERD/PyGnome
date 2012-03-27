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
from PIL import Image, ImageDraw

from gnome.utilities import map_canvas
from hazpy.file_tools import haz_files
from hazpy.geometry import BBox
from hazpy.geometry.PinP import CrossingsTest as point_in_poly
from hazpy.geometry.polygons import PolygonSet
#from hazpy.geometry import polygons

class gnome_map(map_canvas.MapCanvas):
    """
        Basics pyGNOME color bitmap.
        (End-user visualization.)
    """     
    def __init__(self, image_size, bna_filename, color_mode='RGB'):
        """
            Initializes color map attributes. Calls on parent class initialization 
            method in order to handle projection scaling.
        """
        map_canvas.MapCanvas.__init__(self,
                                      image_size,
                                      projection=map_canvas.FlatEarthProjection,
                                      mode=color_mode)
        self.polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")
        self.filename = bna_filename        
        self.draw_land(self.polygons)

    def to_pixel(self, coord):
        """ Projects a (lon, lat) tuple onto the bitmap, and returns the resultant tuple. """
        coord = tuple(self.projection.to_pixel(np.array((coord[0], coord[1]))))
        coord = (int(coord[0]), int(coord[1]))
        return coord
        
    def to_pixel_array(self, coords):
        """ 
            Projects an array of (lon, lat) tuples onto the bitmap, and modifies it in 
            place to hold the corresponding projected values.
        """
        coords['p_long'] -= self.projection.center[0]
        coords['p_lat']  -= self.projection.center[1]
        coords['p_long'] *= self.projection.scale[0]
        coords['p_lat']  *= self.projection.scale[1]
        coords['p_long'] += self.projection.offset[0]
        coords['p_lat']  += self.projection.offset[1]
        coords['p_long'] = np.round(coords['p_long']).astype(np.int)
        coords['p_lat']  = np.round(coords['p_lat']).astype(np.int)
        
    def _type(self):
        """ This requires an explanation. """
        return ' color bitmap'
        
class lw_map(gnome_map):

    """land-water bitmap."""
        
    background_color = 0
    lake_color = 0
    land_color = 1
    
    def __init__(self, image_size, bna_filename, refloat_halflife, color_mode='1'):
        """
            Initializes land-water map attributes. Calls on parent class initialization 
            method in order to handle projection scaling. Caches its bounding_box so that
            it doesn't need to be computed repeatedly.
        """
        gnome_map.__init__(self, image_size, bna_filename, color_mode)
        self.bounding_box = self.polygons.bounding_box
        self.refloat_halflife = refloat_halflife
        self.spills = []

    def _type(self):
        """ Returns the map type. (Either 'color' or 'land-water'.) """
        return ' land-water bitmap'
        
    def on_map(self, pixel_coord):
        """ 
            Given a tuple in pixel coordinates, determines whether the position is on the map.
            It is actually not behaving correctly at the moment: the map bounds and the bounding box
            may not necessarily coincide. Needs fixing!
        """
        bounding_box = self.bounding_box
        if  ( pixel_coord[0] > bounding_box[0][0] and 
              pixel_coord[0] < bounding_box[1][0] and
              pixel_coord[1] > bounding_box[0][1] and
              pixel_coord[1] < bounding_box[1][1] ):
            return True
        return False

    def on_land(self, coord):
        """ Given lat-lon coordinates, determines whether the position is on land. """
        return not self.in_water(coord)

    def on_land_pixel(self, coord):
        """ Given a tuple in pixel coordinates, determines whether the position is on land. """
        return not self.in_water_pixel(coord)
        
    def in_water(self, coord):
        """ Given lat-lon coordinates, determines whether the position is in water. """
        coord = self.to_pixel(coord)
        if not self.on_map(coord):
            return False
        try:
            chrom = self.image.getpixel(coord)
            if not chrom:
                return True
            else:
                return False
        except:
            print 'exception!',  sys.exc_info()[0]
            return False
            
    def in_water_pixel(self, coord):
        """ Given a tuple in pixel coordinates, determines whether the position is in water. """
        coord = coord.tolist()
        if not self.on_map(coord):
            return False
        try:
            chrom = self.image.getpixel(coord)
            if not chrom:
                return True
            else:
                return False
        except:
            print 'exception!',  sys.exc_info()[0]
            return False

    def allowable_spill_position(self, coord):
        """ Determines whether a position given in lat-lon coordinates is an allowable spill location. """
        return self.in_water(coord)
        
    def set_spill(self, coord, num_particles, release_time):
        """ 
            Sets a spill.
        ++args:
            coord: (lon, lat)
            release_time in seconds.
        """
        if not self.allowable_spill_position(coord):
            print  "spill " + str(dict((('position', coord), ('num_particles', num_particles), ('release_time', release_time)))) + " ignored."
        else:
            self.spills += [(coord, num_particles, release_time)]

    def beach_element(self, p, lwp):
        """ 
            Beaches an element that has been landed.
        ++args:
            p: current position (see basic_types.world_point dtype)
            lwp: last water position (see basic_types.world_point dtype)
        """
        in_water = self.in_water
        displacement = ((p['p_long'] - lwp['p_long']), (p['p_lat'] - lwp['p_lat']))
        while not in_water((p['p_long'], p['p_lat'])):
            displacement = (displacement[0]/2, displacement[1]/2)
            p['p_long'] = lwp['p_long'] + displacement[0]
            p['p_lat'] = lwp['p_lat'] + displacement[1]

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
        print "checking on_map:", coord
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

    land_flag  = 1

    # spillable_area_flag = 2
    # something_flag = 4
    # others....
    
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
        try:
            return self.bitmap[coord[0], coord[1]] & self.land_flag
        except IndexError:
            return 0 # not on land if outside the land raster. (Might be off the map!) 
        
    def on_land(self, coord):
        """
        returns 1 if point on land
        returns 0 if not on land
        
        coord is (long, lat) location
        
        """
        return self._on_land_pixel(self.projection.to_pixel(coord))
        
    def _in_water_pixel(self, coord):
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
    
    def allowable_spill_position(self, coord):
        """
        returns true is the spill position is in the allowable spill area
        
        This may not be the same as in_water!
        
        """
        print "checking spillable:", coord
         
#        if self.spillable_area is None:
#            if self.on_map(coord):
#                return not self.on_land(coord)
#            else:
#                return False
#        else:
#            if point_in_poly(self.spillable_area, coord):
#                return self.in_water(coord)
#            else:
#                return False
        
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

        
class MapFromBNA(RasterMap):
    """
    A raster land-water map, created from a BNA file
    """
    def __init__(self,
                 bna_filename,
                 refloat_halflife, #hours
                 raster_size = 1024*1024, # default to 1MB raster
                 ):
        """
        Creates a GnomeMap (specifically a RasterMap) from a bna file
        
        bna_file: full path to a bna file
        
        refloat_halflife: the half-life (in hours) for the re-floating.
        
        raster_size: total number of pixels (bytes) to make the raster -- the actual size
        will match the aspect ratio of the bounding box of the land
        
        It is expected that you will get the spillable area and map bounds from the BNA -- if they exist
        """
        polygons = haz_files.ReadBNA(bna_filename, "PolygonSet")

        #find the spillable area and map bounds:
        # and create a new polygonset with out them
        #  fixme -- adding a "pop" method to PolygonSet might be better
        #      or a gnome_map_data object...
        just_land = PolygonSet()
        spillable_area = None
        map_bounds = None
        for p in polygons:
            print p.metadata
            if p.metadata[1].lower() == "spillablearea":
                spillable_area = p
            elif p.metadata[1].lower() == "map bounds":
                map_bounds = p
            else:
                just_land.append(p)

        # now draw the raster map with a map_canvas:
        #determine the size:
        BB = just_land.bounding_box
        ##fixme: should we just stick with non-projected coord for the raster map?
        W, H = BB.Width, BB.Height
        # stretch the bounding box, to get approximate aspect ratio in projected coords.
        aspect_ratio = np.cos(BB.Center[1] * np.pi / 180 ) * BB.Width / BB.Height
        w = int(np.sqrt(raster_size*aspect_ratio))
        h = int(raster_size / w)
        canvas = map_canvas.BW_MapCanvas( (w, h) )
        canvas.draw_land(just_land)

        #canvas.save("TestLandWaterMap.png")
        
        ## get the bitmap as a numpy array:
        bitmap_array = canvas.as_array()
        
        print bitmap_array
        
        # __init__ the  RasterMap
        RasterMap.__init__(self,
                                 refloat_halflife, #hours
                                 bitmap_array,
                                 canvas.projection,
                                 map_bounds, 
                                 spillable_area, 
                                 )
        
        return None
    
        
        
        
=======
            self.spills += [(coord, num_particles, release_time)]

>>>>>>> e24ce5dbd0eabfdc926c84609ca3b6364ec5debf
