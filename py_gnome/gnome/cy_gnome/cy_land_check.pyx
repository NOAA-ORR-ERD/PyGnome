
"""
cython version of code that uses Bresenham's line algorithm to check if LEs
have crossed land on the raster map

"""

import cython

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int16_t, int32_t, uint8_t, uint32_t
from libcpp cimport bool

cimport type_defs

def overlap_grid(int32_t m, int32_t n, pt1, pt2):
    """
    check if the line segment from pt1 to pt could overlap the grid of
    size (m,n).
    
    returns True is both points are all the way to the left, right top or bottom of the grid

    This version calls the cdef version -- jsut so we do'nt ahve to do teh tuple unpacking
    of the points in the cdef version

    """
    cdef int32_t x1 = pt1[0]
    cdef int32_t y1 = pt1[1]
    cdef int32_t x2 = pt2[0]
    cdef int32_t y2 = pt2[1]

    return c_overlap_grid(m, n, x1, y1, x2, y2)

cdef int32_t c_overlap_grid(int32_t m,
                            int32_t n,
                            int32_t x1,
                            int32_t y1,
                            int32_t x2,
                            int32_t y2):
    """
    check if the line segment from pt1 to pt could overlap the grid of
    size (m,n).
    
    returns True is both points are all the way to the left, right top or bottom of the grid

    """

    if x1 < 0 and x2 < 0: # both left
        return 0
    elif y1 < 0 and y2 < 0: # both below 
        return 0
    elif x1 >= m and x2 >= m: # both right
        return 0 
    elif y1 >= n and y2 >= n: # both above
        return 0
    else:
        return 1


@cython.boundscheck(False)
cdef bool c_find_first_pixel( cnp.ndarray[uint8_t, ndim=2, mode="c"] grid,
                             int32_t m,
                             int32_t n,
                             int32_t x0,
                             int32_t y0,
                             int32_t x1,
                             int32_t y1,
                             int32_t *prev_x,
                             int32_t *prev_y,
                             int32_t *hit_x,
                             int32_t *hit_y,
                             ):

    cdef int32_t dx, dy, sx, sy, err, e2,
    cdef int32_t pt1_x, pt1_y, pt2_x, pt2_y

    # check if totally off the grid
    if not c_overlap_grid(m, n, x0, y0, x1, y1):
        return False

    #pixels = []
    #hit_point = previous_pt = None
    #Set it up to march in the right direction depending on slope.

    dx = abs(x1-x0)
    dy = abs(y1-y0) 
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy
    # check the first point
    if not (x0 < 0 or x0 >= m or y0 < 0 or y0 >= n):#  is the point off the grid? if so, it's not land!
        ##fixme: we should never be starting on land! 
        ## should this raise an Error instead ?
        if grid[x0, y0] == 1: #we've hit "land"
            prev_x[0] = x0
            prev_y[0] = y0
            hit_x[0] = x0
            hit_y[0] = y0
            return True
            #return (x0, y0), (x0, y0)
    
    while True: #keep going till hit land or the final point
        if x0 == x1 and y0 == y1:
            break
        prev_x[0] = x0
        prev_y[0] = y0
        # advance to next point
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 <  dx:
            err = err + dx
            y0 = y0 + sy
        # check for land hit

        if x0 < 0 or x0 >= m or y0 < 0 or y0 >= n:# is the point off the grid? if so, it's not land!
            ## fixme -- if we've moved off the grid for good, no need to keep going.
            continue                             # note: a diagonal movement, off the grid wouldn't be a hit either
        else:
            if grid[x0, y0] == 1:
                hit_x[0] = x0
                hit_y[0] = y0
                # return (*prev_x, *prev_y), (*hit_x, *hit_y)
                return True
            else:
                if (e2 > -dy) and (e2 < dx): # there is a diagonal move -- test adjacent points also
                    ## only call it a hit if BOTH adjacent points are land.
                    pt1_x = x0
                    pt1_y = y0-sy
                    pt2_x = x0-sx
                    pt2_y = y0
                    try: # replace with real check???
                        if ( (grid[pt1_x, pt1_y] == 1) and #is the y-adjacent point on land? 
                             (grid[pt2_x, pt2_y] == 1)     #is the x-adjacent point on land?
                            ): 
                            hit_x[0] = pt1_x # we have to pick one -- this is arbitrary
                            hit_y[0] = pt1_y # we have to pick one -- this is arbitrary
                            #return (*prev_x, *prev_y), (*hit_x, *hit_y)
                            return True
                    except IndexError:
                        pass

    # if we get here, no hit
    return False



def find_first_pixel(grid, pt1, pt2):
    """
    This finds the first non-zero pixel that is is encountered when followoing
    a line from pt1 to pt2.
    
    param: grid  -- a numpy integer array -- the raster were'e working with
           zero eveywhere there is not considered a "hit"
    param: pt1 -- the start point -- an integer (i,j) tuple     
    param: pt2 -- the end point   -- an integer (i,j) tuple
    
    return: None if land is not hit
            (previous_pt, hit_point) if land is hit
    
    This is an adaptation of "Bresenham's line algorithm":
    
   (http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)
    
    Usually used for drawing lines in graphics.  It's been adapted to do an
    extra check when the algorythm puts two points diagonal to each-other, so
    as to avoid an exact match with a diagonal line of land skipping thorough.
    If _both_ the points diagonal to the move are land, it is considered a hit.
    
    """

    cdef int32_t m, n
    m, n = grid.shape

    cdef int32_t  prev_x, prev_y, hit_x, hit_y  

    cdef int32_t x1 = pt1[0]
    cdef int32_t y1 = pt1[1]
    cdef int32_t x2 = pt2[0]
    cdef int32_t y2 = pt2[1]

    result = c_find_first_pixel(grid,
                                m,
                                n,
                                x1,
                                y1,
                                x2,
                                y2,
                                &prev_x,
                                &prev_y,
                                &hit_x,
                                &hit_y,
                                )

    if result:
        return (prev_x, prev_y), (hit_x, hit_y)
    else:
        return None

## called by a method in gnome.map.RasterMap class
from gnome.basic_types import world_point_type, oil_status
@cython.boundscheck(False)
def check_land(cnp.ndarray[uint8_t, ndim=2, mode='c'] grid not None,
               cnp.ndarray[int32_t, ndim=2, mode='c'] positions not None,
               cnp.ndarray[int32_t, ndim=2, mode='c'] end_positions not None,
               cnp.ndarray[int16_t, ndim=1, mode='c'] status_codes not None,
               cnp.ndarray[int32_t, ndim=2, mode='c'] last_water_positions not None):
        """
        do the actual land-checking
                
        status_codes, positions and last_water_positions are altered in place.
        
        NOTE: these are the integer versions -- having already have been projected to the raster coordinates

        """
        cdef int32_t  prev_x, prev_y, hit_x, hit_y  
        cdef uint32_t i, num_le
        cdef int32_t m, n
        cdef bool did_hit

        num_le = positions.shape[0]
        m = grid.shape[0]
        n = grid.shape[1]

        for i in range(num_le):
            if status_codes[i] == type_defs.OILSTAT_ONLAND:
                continue
            
            did_hit = c_find_first_pixel(grid,
                                         m,
                                         n,
                                         positions[i, 0],
                                         positions[i, 1],
                                         end_positions[i, 0],
                                         end_positions[i, 1],
                                         &prev_x,
                                         &prev_y,
                                         &hit_x,
                                         &hit_y,
                                         )
            if did_hit:
                last_water_positions[i, 0] = prev_x
                last_water_positions[i, 1] = prev_y
                end_positions[i,0] = hit_x
                end_positions[i,1] = hit_y
                status_codes[i] = type_defs.OILSTAT_ONLAND
            else:
                # didn't hit land -- can move the LE
                positions[i, 0] = end_positions[i, 0]
                positions[i, 1] = end_positions[i, 1]
        return None


