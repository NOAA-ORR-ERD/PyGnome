/*
 *  Basics.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __gnome_basics__
#define __gnome_basics__

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdlib>
#include <ios>

#define nil NULL
#define noErr 0

#define TRUE 1
#define FALSE 0

#ifndef MAC
#define _min(a,b) ((a) < (b) ? (a) : (b))
#define _max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifdef IBM
#define DIRDELIMITER '\\'
#define OPPOSITEDIRDELIMITER ':'
#else
#define DIRDELIMITER ':'
#define OPPOSITEDIRDELIMITER '\\'
#endif

#ifdef pyGNOME

/*****************************************************/
// This is temporary:

typedef unsigned char Boolean;
struct Point {
	short               v;
	short               h;
};
typedef struct Point                    Point;
typedef Point *                         PointPtr;
struct Rect {
	short               top;
	short               left;
	short               bottom;
	short               right;
};
typedef struct Rect						Rect;
typedef Rect *                          RectPtr;
struct DateTimeRec {
	short               year;
	short               month;
	short               day;
	short               hour;
	short               minute;
	short               second;
	short               dayOfWeek;
};
typedef struct DateTimeRec              DateTimeRec;

struct RGBColor {
	unsigned short      red;                    /*magnitude of red component*/
	unsigned short      green;                  /*magnitude of green component*/
	unsigned short      blue;                   /*magnitude of blue component*/
};

typedef struct RGBColor                 RGBColor;
typedef RGBColor *                      RGBColorPtr;
typedef RGBColorPtr *                   RGBColorHdl;

struct Pattern {
	unsigned long               pat[8];
};
typedef struct Pattern                  Pattern;
typedef char *OSType;
typedef void *WindowPtr;
typedef short							OSErr;

/*******************************************************/

#endif

typedef void *VOIDPTR, **VOIDH;
typedef char *CHARPTR, **CHARH;
typedef CHARH *CHARHP;
typedef short *SHORTPTR, **SHORTH;
typedef long *LONGPTR, **LONGH;
typedef float *FLOATPTR, **FLOATH;
typedef double *DOUBLEPTR, **DOUBLEH;
typedef Point *POINTPTR, **POINTH;
typedef Rect *RECTPTR, **RECTH;

#endif