#!/usr/bin/env python

"""
code that uses Bresenham's line algorithm to check if LEs have crossed land
on the raster map

This will most likely be translated to Cython
"""

import numpy as np

def overlap_grid(m, n, pt1, pt2):
    """
    check if the line segment from pt1 to pt could overlap the grid of
    size (m,n).
    
    returns True is both points are all the way to the left, right top or bottom of the grid

    """
    if pt1[0] < 0 and pt2[0] < 0: # both left
        return False
    elif pt1[1] < 0 and pt2[1] < 0: # both below 
        return False
    elif pt1[0] >= m and pt2[0] >= m: # both right
        return False 
    elif pt1[1] >= n and pt2[1] >= n: # both above
        return False
    else:
        return True

def find_first_pixel(grid, pt1, pt2, draw=False):
    """
    This finds the first non-zero pixel that is is encountered when followoing
    a line from pt1 to pt2.
    
    param: grid  -- a numpy integer array -- the raster were'e working with
           zero eveywhere there is not considered a "hit"
    param: pt1 -- the start point -- an integer (i,j) tuple     
    param: pt2 -- the end point   -- an integer (i,j) tuple     
    param: draw [False] -- should the line be drawn to the grid?
    
    return: None if land is not hit
            previous_pt, hit_point
    
    This is an adaptation of "Bresenham's line algorithm":
    
   (http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)
    
    Usually used for drawing lines in graphics.  It's been adapted to do an
    extra check when the algorythm puts two points diagonal to each-other, so
    as to avoid an exact match with a diagonal line of land skipping thorough.
    If _both_ the points diagonal to the move are land, it is considered a hit.
    
    """
    # check if totally off the grid
    # print "in python find_first_pixel"
    m, n = grid.shape
    if not overlap_grid(m, n, pt1, pt2):
        return None
    
    pixels = []
    hit_point = previous_pt = None
    #Set it up to march in the right direction depending on slope.
    x0, y0 = pt1
    x1, y1 = pt2

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
    last_pt  = pt1
    # check the first point
    if not (x0 < 0 or x0 >= m or y0 < 0 or y0 >= n):#  is the point off the grid? if so, it's not land!
        if draw: 
            grid[x0, y0] += 2
        ##fixme: we should never be starting on land! 
        ## should this raise an Error instead ?
        if grid[x0, y0] == 1: #we've hit "land"
            return (x0, y0), (x0, y0)
    
    while True: #keep going till hit land or the final point
        if x0 == x1 and y0 == y1:
            break
        previous_pt = (x0, y0) 
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
            continue                             # note: a diagonal movement, off the grid wouldn't be a hit either
        else:
            if grid[x0, y0] == 1:
                hit_point = (x0, y0)
                if draw:
                    grid[x0, y0] += 2
                return previous_pt, hit_point
            else:
                if (e2 > -dy) and (e2 < dx): # there is a diagonal move -- test adjacent points also
                    ## only call it a hit if BOTH adjacent points are land.
                    pt1 = (x0, y0-sy)
                    pt2 = (x0-sx, y0)
                    try:
                        if ( (grid[pt1[0], pt1[1]] == 1) and #is the y-adjacent point on land? 
                             (grid[pt2[0], pt2[1]] == 1)     #is the x-adjacent point on land?
                            ): 
                            hit_point = pt1 # we have to pick one -- this is arbitrary
                            if draw:
                                grid[hit_point[0], hit_point[1]] += 2
                            return previous_pt, hit_point
                    except IndexError:
                        pass
            if draw:
                grid[x0, y0] += 2

    # if we get here, no hit
    return None

def draw_line(grid, pt1, pt2, draw_val=1):
    """
    this is a version of "Bresenham's line algorithm"
    
    This one just finds the line.
    
    it is used to draw lines on a screen, but also should tell us which pixels an LE path has crossed
    """
    pixels = []
    x0, y0 = pt1
    x1, y1 = pt2

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
 
    while True:
        grid[x0, y0] = draw_val
        pixels.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 <  dx:
            err = err + dx
            y0 = y0 + sy 
    return pixels



def thick_line(grid, (x_1, y_1), (x_2, y_2), line_type=0, color=1):
    """
    draws a thick line -- line_type is integer width
    
    adapted from C++ code from:
    http://mtshome.sw3solutions.com/cppComputerGraphics.html#Line

    This turned out not to work so well, -- it needed some tweaking, and is
    probably not the way to go anywya -- resuscied resolutin everywhere, why do that?
    """ 
    
    def putpixel(x, y, color):
        try:
            grid[x,y] = color
        except IndexError:
            pass
    x1=x_1
    y1=y_1

    x2=x_2
    y2=y_2

    if (x_1 > x_2):
        x1=x_2
        y1=y_2
        
        x2=x_1
        y2=y_1

    dx = abs(x2-x1)
    dy = abs(y2-y1)
    #inc_dec = ((y2 >= y1) ?1:-1 )
    if y2 >= y1: 
        inc_dec = 1
    else:
        inc_dec = -1

    if  dx > dy :
        two_dy=(2*dy)
        two_dy_dx=(2*(dy-dx))
        p=((2*dy)-dx)

        x=x1
        y=y1

        while (x <= x2) :
            if(line_type==0):
                putpixel(x,y,color)
            elif(line_type==1):
                putpixel(x,y,color)
                putpixel(x,(y+1),color)
            elif(line_type==2):
                putpixel(x,(y-1),color)
                putpixel(x,(y),color)
                putpixel(x,(y+1),color)
            elif(line_type==3):
                putpixel(x,(y-1),color)
                putpixel(x,(y),color)
                putpixel(x,(y+1),color)
                putpixel(x,(y+2),color)
            elif(line_type==4):
                putpixel(x,(y-2),color)
                putpixel(x,(y-1),color)
                putpixel(x,(y),color)
                putpixel(x,(y+1),color)
                putpixel(x,(y+2),color)

            x += 1 

            if p < 0 :
                p += two_dy
            else:
                y += inc_dec
                p += two_dy_dx
    else:
        #raise NotImplementedError("can't go that way yet: %s"%( ((x_1, y_1), (x_2, y_2)), ))
        two_dx=(2*dx);
        two_dx_dy=(2*(dx-dy));
        p=((2*dx)-dy);
        
        x=x1;
        y=y1;

        while( y != y2 ):
            if(line_type==0):
                putpixel(x,y,color)
            elif(line_type==1):
                putpixel(x,y,color)
                putpixel(x,(y+1),color)
            elif(line_type==2):
                putpixel(x,(y-1),color)
                putpixel(x,(y),color)
                putpixel(x,(y+1),color)
            elif(line_type==3):
                putpixel(x,(y-1),color)
                putpixel(x,(y),color)
                putpixel(x,(y+1),color)
                putpixel(x,(y+2),color)
            elif(line_type==4):
                putpixel(x,(y-2),color)
                putpixel(x,(y-1),color)
                putpixel(x,(y),color)
                putpixel(x,(y+1),color)
                putpixel(x,(y+2),color)


            y += inc_dec

            if p < 0 :
                p += two_dx
            else:
                x += 1
                p += two_dx_dy 

def draw_grid(filename, grid):
    """
    draw a grid in large scale with PIL
    """
    from PIL import Image, ImageDraw
    
    bs = 20 # block size

    colors = {0: (255,255,255),
              1: (255, 0, 0),
              2: (0, 0, 255),
              3: (255, 0, 255),
              4: (0, 255, 0),
              }
    shape = grid.shape[0]*bs , grid.shape[1] * bs
    im = Image.new('P', shape)
    draw = ImageDraw.Draw(im)
    
    for i in range(grid.shape[0]):
        for j in range (grid.shape[1]):
            box =  ( (j*bs, i*bs), ((j+1)*bs, (i+1)*bs) )
            draw.rectangle(box, fill=colors[grid[i,j]])

    im.save(filename)
    

if __name__ == "__main__":
    #provide a raster:
    
    tests = [  [ (0,0), (9,11) ],
               [ (0,11), (9,0) ],
               [ (5,0), (5,11) ],
               [ (0,6), (9,6) ],
               [ (5,0), (6,11) ],
               [ (6,0), (5,11) ],
               [ (0,6), (9,5)  ],
               [ (0,6), (9,7)  ],
               [ (0,0), (0,11)  ],
               [ (0,0), (1,11)  ],
               [ (0,0), (2,11)  ],
               [ (0,0), (3,11)  ],
               [ (0,0), (4,11)  ],
               [ (0,0), (5,11)  ],
               [ (0,0), (6,11)  ],
               [ (0,0), (7,11)  ],
               [ (0,0), (8,11)  ],
               [ (9,11), (0,0)  ],
               [ (9,10), (0,1)  ],
               [ (9,9), (0,2)  ],
               [ (9,8), (0,3)  ],
               [ (9,7), (0,4)  ],
               [ (9, 6), (0,5)  ],
               [ (9, 5), (0,6)  ],
               [ (9, 4), (0,7)  ],
               [ (9, 3), (0,8)  ],
               [ (9, 2), (0,9)  ],
               [ (9, 1), (0,10)  ],
               [ (9,0), (0,11)  ],
               ]
    
#    for points in tests:
#    
#        grid = np.zeros( (10, 12), dtype= np.uint8 )
#        pixels = draw_line( grid, *points)
#        print grid
#        print points
#        print pixels
#        if not pixels[0] == points[0] and pixels[-1] == points[1]:
#            raise Exceptions("end and start points not the same")
#        points.reverse()
#        grid = np.zeros( (10, 12), dtype= np.uint8 )
#        pixels = draw_line( grid, *points)
#        print grid
#        print points
#        print pixels
#        if not pixels[0] == points[0] and pixels[-1] == points[1]:
#            raise Exceptions("end and start points not the same")

### Test the Thickline code
#    for points in tests:
#        grid = np.zeros( (10, 12), dtype= np.uint8 )
#        pixels = thick_line( grid, *points, line_type=2, color=1)
#        print grid
#        print points
##        print pixels
##        if not pixels[0] == points[0] and pixels[-1] == points[1]:
##            raise Exceptions("end and start points not the same")
##        points.reverse()
##        grid = np.zeros( (10, 12), dtype= np.uint8 )
##        pixels = draw_line( grid, *points)
##        print grid
##        print points
##        print pixels
##        if not pixels[0] == points[0] and pixels[-1] == points[1]:
##            raise Exceptions("end and start points not the same")        
#
#    raise Exception("stopping")

### create a simple example:
#    grid = np.zeros((9, 11), dtype=np.uint8)
#    # vertical line of land:
#    grid[4, :] = 1
#    
#    #tests:
#    #horizontal line
#    points = [(2, 4), (8,4)]
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])
#    points = [(8, 4), (2,4)]
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])
#    points = [(0, 0), (8,10)]
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])
#    points = [(8, 10), (0, 0)]
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])
#
#    points = [(4, 5), (0, 0)]
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])
#
#    #diagonal line of land
#    grid = np.zeros((9, 9), dtype=np.uint8)
#
#    for i in range(9):
#        grid[i,i] = 1
#    points = [(0,8), (8, 0)]
#    print grid
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])
#
#    # no land 
#    grid = np.zeros((9, 9), dtype=np.uint8)
#
#    points = [(0,8), (0, 8)]
#    print grid
#    print "first hit at:", find_first_pixel(grid, points[0], points[1])

#    draw_line( grid, (0,11), (9,0))
#    print grid
#    draw_line( grid, (5,5), (5,11))
#    print grid

    # test for "slipping through the corners":
    m = 10
    grid = np.zeros((m, m), dtype=np.uint8)
    # a diagonal line:
    for i in range(m):
        grid[i,i] = 1
    
    #now loop through all the possible diagonals:
    for i in range(1,m):
        for j in range(1,m):
#    for i,j in ( (9, 3),):
            new_grid = grid.copy()
            #p1, p2 = (0,i), (m-1,j)
            p2, p1 = (0,i), (m-1,j)
            result = find_first_pixel(new_grid, p1, p2, draw=True, keep_going=True)
            draw_grid("test%i-%i.png"%(i,j), new_grid)
            print new_grid
            if result is None:
                print "point fell through: %s to %s"%(p1, p2)
                break
            else:
                print "this worked!:", p1, p2, "hit at:", result
    
    ## plot out ones that got through
    
#    new_grid = grid.copy()            
#    #draw_line(new_grid, (0,6) , (9,0), draw_val=2)
#    print new_grid
#    find_first_pixel(new_grid, (0,6) , (9,0), draw=True, keep_going=True)
#    print new_grid
#    draw_grid("test1.png", new_grid)
    
#    new_grid = grid.copy()            
#    #draw_line(new_grid, (0,0) , (9,9), draw_val=2)
#    find_first_pixel(new_grid, (9,0) , (0,9), draw=True, keep_going=True)
#    print new_grid
#    draw_grid("test2.png", new_grid)
    
    
            
        
