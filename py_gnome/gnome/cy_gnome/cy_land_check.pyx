
"""
cython version of code that uses Bresenham's line algorithm to check if LEs
have crossed land on the raster map

"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, uint8_t

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




cdef c_find_first_pixel( cnp.ndarray[uint8_t, ndim=2] grid,
                         int32_t m,
                         int32_t n,
                         int32_t x0,
                         int32_t y0,
                         int32_t x1,
                         int32_t y1):

    cdef int32_t dx, dy, sx, sy, err, e2,
    cdef int32_t hit_x, hit_y, prev_x, prev_y, pt1_x, pt1_y, pt2_x, pt2_y

    # check if totally off the grid
    if not c_overlap_grid(m, n, x0, y0, x1, y1):
        return None

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
            return (x0, y0), (x0, y0)
    
    while True: #keep going till hit land or the final point
        if x0 == x1 and y0 == y1:
            break
        prev_x = x0
        prev_y = y0
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
                hit_x = x0
                hit_y = y0
                return (prev_x, prev_y), (hit_x, hit_y)
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
                            hit_x = pt1_x # we have to pick one -- this is arbitrary
                            hit_y = pt1_y # we have to pick one -- this is arbitrary
                            return (prev_x, prev_y), (hit_x, hit_y)
                    except IndexError:
                        pass

    # if we get here, no hit
    return None



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
                                y2)

    return result



# if __name__ == "__main__":
#     #provide a raster:
    
#     tests = [  [ (0,0), (9,11) ],
#                [ (0,11), (9,0) ],
#                [ (5,0), (5,11) ],
#                [ (0,6), (9,6) ],
#                [ (5,0), (6,11) ],
#                [ (6,0), (5,11) ],
#                [ (0,6), (9,5)  ],
#                [ (0,6), (9,7)  ],
#                [ (0,0), (0,11)  ],
#                [ (0,0), (1,11)  ],
#                [ (0,0), (2,11)  ],
#                [ (0,0), (3,11)  ],
#                [ (0,0), (4,11)  ],
#                [ (0,0), (5,11)  ],
#                [ (0,0), (6,11)  ],
#                [ (0,0), (7,11)  ],
#                [ (0,0), (8,11)  ],
#                [ (9,11), (0,0)  ],
#                [ (9,10), (0,1)  ],
#                [ (9,9), (0,2)  ],
#                [ (9,8), (0,3)  ],
#                [ (9,7), (0,4)  ],
#                [ (9, 6), (0,5)  ],
#                [ (9, 5), (0,6)  ],
#                [ (9, 4), (0,7)  ],
#                [ (9, 3), (0,8)  ],
#                [ (9, 2), (0,9)  ],
#                [ (9, 1), (0,10)  ],
#                [ (9,0), (0,11)  ],
#                ]
    
# #    for points in tests:
# #    
# #        grid = np.zeros( (10, 12), dtype= np.uint8 )
# #        pixels = draw_line( grid, *points)
# #        print grid
# #        print points
# #        print pixels
# #        if not pixels[0] == points[0] and pixels[-1] == points[1]:
# #            raise Exceptions("end and start points not the same")
# #        points.reverse()
# #        grid = np.zeros( (10, 12), dtype= np.uint8 )
# #        pixels = draw_line( grid, *points)
# #        print grid
# #        print points
# #        print pixels
# #        if not pixels[0] == points[0] and pixels[-1] == points[1]:
# #            raise Exceptions("end and start points not the same")

# ### Test the Thickline code
# #    for points in tests:
# #        grid = np.zeros( (10, 12), dtype= np.uint8 )
# #        pixels = thick_line( grid, *points, line_type=2, color=1)
# #        print grid
# #        print points
# ##        print pixels
# ##        if not pixels[0] == points[0] and pixels[-1] == points[1]:
# ##            raise Exceptions("end and start points not the same")
# ##        points.reverse()
# ##        grid = np.zeros( (10, 12), dtype= np.uint8 )
# ##        pixels = draw_line( grid, *points)
# ##        print grid
# ##        print points
# ##        print pixels
# ##        if not pixels[0] == points[0] and pixels[-1] == points[1]:
# ##            raise Exceptions("end and start points not the same")        
# #
# #    raise Exception("stopping")

# ### create a simple example:
# #    grid = np.zeros((9, 11), dtype=np.uint8)
# #    # vertical line of land:
# #    grid[4, :] = 1
# #    
# #    #tests:
# #    #horizontal line
# #    points = [(2, 4), (8,4)]
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])
# #    points = [(8, 4), (2,4)]
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])
# #    points = [(0, 0), (8,10)]
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])
# #    points = [(8, 10), (0, 0)]
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])
# #
# #    points = [(4, 5), (0, 0)]
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])
# #
# #    #diagonal line of land
# #    grid = np.zeros((9, 9), dtype=np.uint8)
# #
# #    for i in range(9):
# #        grid[i,i] = 1
# #    points = [(0,8), (8, 0)]
# #    print grid
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])
# #
# #    # no land 
# #    grid = np.zeros((9, 9), dtype=np.uint8)
# #
# #    points = [(0,8), (0, 8)]
# #    print grid
# #    print "first hit at:", find_first_pixel(grid, points[0], points[1])

# #    draw_line( grid, (0,11), (9,0))
# #    print grid
# #    draw_line( grid, (5,5), (5,11))
# #    print grid

#     # test for "slipping through the corners":
#     m = 10
#     grid = np.zeros((m, m), dtype=np.uint8)
#     # a diagonal line:
#     for i in range(m):
#         grid[i,i] = 1
    
#     #now loop through all the possible diagonals:
#     for i in range(1,m):
#         for j in range(1,m):
# #    for i,j in ( (9, 3),):
#             new_grid = grid.copy()
#             #p1, p2 = (0,i), (m-1,j)
#             p2, p1 = (0,i), (m-1,j)
#             result = find_first_pixel(new_grid, p1, p2, draw=True, keep_going=True)
#             draw_grid("test%i-%i.png"%(i,j), new_grid)
#             print new_grid
#             if result is None:
#                 print "point fell through: %s to %s"%(p1, p2)
#                 break
#             else:
#                 print "this worked!:", p1, p2, "hit at:", result
    
#     ## plot out ones that got through
    
# #    new_grid = grid.copy()            
# #    #draw_line(new_grid, (0,6) , (9,0), draw_val=2)
# #    print new_grid
# #    find_first_pixel(new_grid, (0,6) , (9,0), draw=True, keep_going=True)
# #    print new_grid
# #    draw_grid("test1.png", new_grid)
    
# #    new_grid = grid.copy()            
# #    #draw_line(new_grid, (0,0) , (9,9), draw_val=2)
# #    find_first_pixel(new_grid, (9,0) , (0,9), draw=True, keep_going=True)
# #    print new_grid
# #    draw_grid("test2.png", new_grid)
    
    
            
        
