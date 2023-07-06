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

//#define _USE_32BIT_TIME_T	1

#include <cstdlib>
#include <stdint.h>
#include <math.h>
#include <cstring>
#include <ios>
#include <iostream>
#include <fstream>
#include <vector>

#define nil NULL

#define TRUE 1
#define FALSE 0

#ifndef MAC
#define _min(a,b) ((a) < (b) ? (a) : (b))
#define _max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifdef _WIN32
	#define DIRDELIMITER '\\'		// This probably designates the separation characters for directories in a full path
	#define NEWDIRDELIMITER '\\'	// I dunno, is this the same as DIRDELIMITER?
	#define OPPOSITEDIRDELIMITER ':' // OK, this looks like a drive letter delimiter here.  Why the obscure name?
#else
	#define DIRDELIMITER ':'		// but why would the DIRDELIMITER look like a drive letter delimiter on a non-windows system?
	#define NEWDIRDELIMITER '/'
	#define OPPOSITEDIRDELIMITER '\\'
#endif

#ifndef noErr
#define noErr 0
#endif

#ifdef pyGNOME
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
//#define noErr 0

enum {
	/* Memory Manager errors*/
	memROZWarn                    = -99,  /*soft error in ROZ*/
	memROZError                   = -99,  /*hard error in ROZ*/
	memROZErr                     = -99,  /*hard error in ROZ*/
	memFullErr                    = -108, /*Not enough room in heap zone*/
	nilHandleErr                  = -109, /*Master Pointer was NIL in HandleZone or other*/
	memWZErr                      = -111, /*WhichZone failed (applied to free block)*/
	memPurErr                     = -112, /*trying to purge a locked or non-purgeable block*/
	memAdrErr                     = -110, /*address was odd; or out of range*/
	memAZErr                      = -113, /*Address in zone check failed*/
	memPCErr                      = -114, /*Pointer Check failed*/
	memBCErr                      = -115, /*Block Check failed*/
	memSCErr                      = -116, /*Size Check failed*/
	memLockedErr                  = -117  /*trying to move a locked block (
										   MoveHHi)*/
};

/*******************************************************/

#endif

#ifdef IBM

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

enum {
	/* Memory Manager errors*/
	memROZWarn                    = -99,  /*soft error in ROZ*/
	memROZError                   = -99,  /*hard error in ROZ*/
	memROZErr                     = -99,  /*hard error in ROZ*/
	memFullErr                    = -108, /*Not enough room in heap zone*/
	nilHandleErr                  = -109, /*Master Pointer was NIL in HandleZone or other*/
	memWZErr                      = -111, /*WhichZone failed (applied to free block)*/
	memPurErr                     = -112, /*trying to purge a locked or non-purgeable block*/
	memAdrErr                     = -110, /*address was odd; or out of range*/
	memAZErr                      = -113, /*Address in zone check failed*/
	memPCErr                      = -114, /*Pointer Check failed*/
	memBCErr                      = -115, /*Block Check failed*/
	memSCErr                      = -116, /*Size Check failed*/
	memLockedErr                  = -117  /*trying to move a locked block (
										   MoveHHi)*/
};

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
#ifndef MAC
typedef CHARPTR Ptr;
typedef CHARH Handle;
typedef Handle *HANDLEPTR;
#endif

#endif
