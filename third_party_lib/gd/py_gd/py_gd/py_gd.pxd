"""
declarations for the cython wrapper around the gd drawing lib
"""

from libc.stdio cimport FILE

## access the gd header files:
cdef extern from "gd.h":

    cdef struct gdImageStruct:
        pass # for now, all I need is to know it exist to pass along...
        # Palette-based image pixels
        unsigned char **pixels
        ## lots more we might want here, but for now...
    ctypedef gdImageStruct *gdImagePtr
    
    cdef struct gdPoint:
        int x, y
    ctypedef gdPoint *gdPointPtr

    cdef struct gdFont
    ctypedef gdFont *gdFontPtr

    ## The functions we want to wrap

    # utilities for creating, etc, images
    gdImagePtr gdImageCreatePalette(int width, int height)
    
    void gdImageDestroy (gdImagePtr im)
    
    int gdImageColorAllocate (gdImagePtr im, int r, int g, int b)
    int gdImageColorAllocateAlpha(gdImagePtr im, int r, int g, int b, int a)
    int gdImageGetPixel(gdImagePtr im, int x, int y)

    # drawing functions
    ## to set up line drawing
    void gdImageSetThickness(gdImagePtr im, int thickness)
    
    void gdImageSetPixel (gdImagePtr im, int x, int y, int color)

    void gdImageLine (gdImagePtr im, int x1, int y1, int x2, int y2, int color)

    void gdImagePolygon (gdImagePtr im, gdPointPtr p, int num_points, int color)
    void gdImageFilledPolygon (gdImagePtr im, gdPointPtr p, int num_points, int color)

    void gdImageOpenPolygon(gdImagePtr im, gdPointPtr points, int pointsTotal, int color)

    void gdImageRectangle(gdImagePtr im, int x1, int y1, int x2, int y2, int color)
    void gdImageFilledRectangle(gdImagePtr im, int x1, int y1, int x2, int y2, int color)

    void gdImageArc(gdImagePtr im, int cx, int cy, int w, int h, int s, int e, int color)
    void gdImageFilledArc(gdImagePtr im, int cx, int cy, int w, int h, int s, int e, int color, int style)

    # text drawing, etc.
    void gdImageString(gdImagePtr im, gdFontPtr font, int x, int y, unsigned char *s, int color)

    # image saving functions
    void gdImageBmp  (gdImagePtr im, FILE *outFile,  int compression)
    void gdImageJpeg (gdImagePtr im, FILE * outFile, int quality)
    void gdImageGif  (gdImagePtr im, FILE *outFile) 
    void gdImagePng  (gdImagePtr im, FILE *outFile)

    # constants (these are #define in gd.h)
    cpdef int gdArc
    cpdef int gdPie
    cpdef int gdChord 
    cpdef int gdNoFill
    cpdef int gdEdged 

# fonts are in extra headers:
cdef extern from "gdfontt.h":
    gdFontPtr gdFontTiny
cdef extern from "gdfonts.h":
    gdFontPtr gdFontSmall
cdef extern from "gdfontmb.h":
    gdFontPtr gdFontMediumBold
cdef extern from "gdfontl.h":
    gdFontPtr gdFontLarge
cdef extern from "gdfontg.h":
    gdFontPtr gdFontGiant

## synonm
#cdef int gdPie = gdArc






