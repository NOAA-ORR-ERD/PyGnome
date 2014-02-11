/*
 *  TypeDefs.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TypeDefs__
#define __TypeDefs__

#include <time.h>

#ifndef pyGNOME
#include "Earl.h"
#endif

//++ Carry-over from basic definitions.

#define DEREFH(h) (*(h))
#define INDEXH(h, i) (*(h))[i]

//--

#define kMaxNameLen 256
#define TIMEVALUE_TOLERANCE 0.00001

#define INFINITE_DEPTH	5000.


typedef short OilType;
typedef short OilStatus;
typedef short LandType;

// we need to key on the bus width here i386 vs. x86_64
#ifdef _MSC_VER
	#if _WIN64
		// for now we will not change the type for windows64
		typedef long Seconds; // duration in seconds, or seconds since 1904
	#else
		#ifndef pyGNOME
			typedef unsigned long Seconds;
		#else
			typedef long Seconds; // duration in seconds, or seconds since 1904
		#endif
	#endif
#else
	#if __x86_64__ || __ppc64__
		typedef time_t Seconds; // duration in seconds, or seconds since 1904
	#else
		#ifndef pyGNOME
			typedef unsigned long Seconds;
		#else
			// this is the value we are using for 32 bit architecture
			typedef time_t Seconds; // duration in seconds, or seconds since 1904
		#endif
	#endif
#endif


typedef unsigned long LETYPE ;

///// TYPES ///////////////////////////////////////////////////////////////////////

extern Rect CATSgridRect;

typedef struct VelocityRec
{
	double u;
	double v;
} VelocityRec, *VelocityP, **VelocityH;

typedef struct VelocityFRec
{
	float u;
	float v;
} VelocityFRec, *VelocityFP, **VelocityFH;

typedef struct VelocityRec3D
{
	double u;
	double v;
	double w;
} VelocityRec3D, *VelocityP3D, **VelocityH3D;

typedef struct VelocityFRec3D
{
	float u;
	float v;
	float w;
} VelocityFRec3D, *VelocityFP3D, **VelocityFH3D;

typedef struct
{
	Seconds time;
	VelocityRec value;
} TimeValuePair, *TimeValuePairP, **TimeValuePairH;

typedef struct
{
	Seconds time;
	VelocityRec3D value;
} TimeValuePair3D, *TimeValuePairP3D, **TimeValuePairH3D;

// old ossm pollutant types
enum { //OSSMOIL_UNKNOWN = 0,  JLM 3/11/99, there was no such thing as unkown in old OSSM
	OSSMOIL_GAS = 1, OSSMOIL_JETFUELS, OSSMOIL_DIESEL,
	OSSMOIL_4, OSSMOIL_CRUDE, OSSMOIL_6, OSSMOIL_USER1, OSSMOIL_USER2,
	OSSMOIL_CONSERVATIVE};
// OSSMOIL_COMBINATION = 1000 };JLM 3/11/99, there was no such thing as combination in old OSSM

// new GNOME pollutant types

// alphabetical
//enum { OIL_UNKNOWN = 0, OIL_DIESEL, OIL_4, OIL_6, OIL_GAS, OIL_JETFUELS,
//	   OIL_CRUDE, OIL_CONSERVATIVE, OIL_USER1=5000, OIL_USER2 = 5001,
//	   OIL_COMBINATION = 1000 };

// light to heavy
enum { OIL_UNKNOWN = 0, OIL_GAS, OIL_JETFUELS, OIL_DIESEL, OIL_4, OIL_CRUDE, OIL_6,  
	OIL_CONSERVATIVE, OIL_USER1=5000, OIL_USER2 = 5001,
	OIL_COMBINATION = 1000, CHEMICAL = 8 };


enum { LT_LAND = 1, LT_WATER = 2, LT_UNDEFINED = -1 };

// OSSM land types 
enum { OSSM_L0 = 1, OSSM_L1 = 2, OSSM_L2 = 3, OSSM_L3 = 4, OSSM_L4 = 5, OSSM_L5 = 6, 
	OSSM_L6 = 7, OSSM_L7 = 8, OSSM_L8 = 9, OSSM_L9 = 10, OSSM_LL = 11, 
	OSSM_W0 = 12, OSSM_W1 = 13, OSSM_W2 = 14, OSSM_W3 = 15, OSSM_WW = 16};

enum LEStatus { OILSTAT_NOTRELEASED = 0, OILSTAT_INWATER = 2, OILSTAT_ONLAND = 3,
	OILSTAT_OFFMAPS = 7, OILSTAT_EVAPORATED = 10, OILSTAT_TO_BE_REMOVED = 12};
//JLM note: on why these number are what they are
// OILSTAT_INAIR (which was 1) meant evaporated, and got confused with the 
// with OILSTAT_EVAPORATED (which was 10, because of old ossm)
// Using these numbers will keep our testers save files in sync 
// 
// old OSSM (and hence TAT) use the codes 
// nMap 7 for off maps,  beachHeight -50 for beached 
// and adds 10 to the pollutant code to indicate EVAPORATED
enum { OLD_OSSM_OFFMAPS = 7, OLD_OSSM_EVAPORATED = 10, OLD_OSSM_BEACHED = -50};

enum {CURRENTS_MOVERTYPE=1,WIND_MOVERTYPE,CONSTANT_MOVERTYPE,RANDOM_MOVERTYPE,COMPONENT_MOVERTYPE,COMPOUND_MOVERTYPE};

enum {NOTIMEFILE=1, SHIOCURRENTSFILE, SHIOHEIGHTSFILE, OSSMTIMEFILE, HYDROLOGYFILE, PROGRESSIVETIDEFILE, ADCPTIMEFILE};

enum { BULLET_NONE = 0, BULLET_DASH, BULLET_EMPTYBOX, BULLET_FILLEDBOX,
	BULLET_OPENTRIANGLE, BULLET_CLOSEDTRIANGLE };

enum { SCALE_NONE, SCALE_CONSTANT, SCALE_OTHERGRID };

enum { LINEAR, HERMITE, TBD };

enum {DONT_DISPERSE, DISPERSE, HAVE_DISPERSED, DISPERSE_NAT, HAVE_DISPERSED_NAT, EVAPORATE, HAVE_EVAPORATED, REMOVE, HAVE_REMOVED};	// LE dispersion status

enum LEType {FORECAST_LE = 1, UNCERTAINTY_LE = 2};
///// CONSTANTS /////////////////////////////////////////////////////////////////

#ifdef MAC
#ifdef MPW
#define NEWLINESTRING "\n"
#define UNIXNEWLINESTRING "\r"
#define MACNEWLINESTRING "\n"
#define IBMNEWLINESTRING "\n\r"
#else
#define NEWLINESTRING "\r"
#define UNIXNEWLINESTRING "\n"
#define MACNEWLINESTRING "\r"
#define IBMNEWLINESTRING "\r\n"
#endif
#else
#define NEWLINESTRING "\r\n"
#define UNIXNEWLINESTRING "\n"
#define MACNEWLINESTRING "\r"
#define IBMNEWLINESTRING "\r\n"
#endif

#ifndef MYKEYS
#define MYKEYS
enum { ENTER = 3, DEL = 8, TAB = 9, LINEFEED = 10, RETURN = 13, ESC = 27,
	LEFTARROW = 28, RIGHTARROW = 29, UPARROW = 30, DOWNARROW = 31 };
enum { ESCAPEKEY = 0X35, ENTERKEY = 0X4C, RETURNKEY = 0X24, PERIODKEY = 0X2F,
	UPKEY = 0X7E, UPKEYPLUS = 0X4D, DOWNKEY = 0X7D, DOWNKEYPLUS = 0X48,
	PAGEUPKEY = 116, PAGEDOWNKEY = 121, HOMEKEY = 115, ENDKEY = 119,
	AKEY = 0X00, IKEY = 0X22 };
#endif

#ifdef _MSC_VER
  #include <float.h>  // for _isnan() on VC++
  #define isnan(x) _isnan(x)  // VC++ uses _isnan() instead of isnan()
  #define round(n) floor((n) + 0.5) // round not defined in VC++
#else
  #include <math.h>  // for isnan() everywhere else
#endif

//#define round(n) floor((n) + 0.5)

#define PI 3.14159265359

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
	double 	uncertaintyValue;
} TR_OPTIMZE;

//++ Geometry

typedef struct LongPoint
{
	long 					h;
	long 					v;
} LongPoint, *LongPointPtr, **LongPointHdl;

#ifndef pyGNOME
typedef struct {
	long pLong;
	long pLat;
} WorldPoint, *WORLDPOINTP, **WORLDPOINTH;
#else
typedef struct {
	double pLong;
	double pLat;
} WorldPoint, *WORLDPOINTP, **WORLDPOINTH;
#endif

typedef struct {
	float pLong;
	float pLat;
} WorldPointF, *WORLDPOINTFP, **WORLDPOINTFH;

typedef struct {
	WorldPoint p;
	double z;
} WorldPoint3D, *WORLDPOINT3DP, **WORLDPOINT3DH;

typedef struct {
	Boolean newPiece;
	WorldPoint p;
} PointType, *PointP, **PointH;

typedef struct {
	long loLong;
	long loLat;
	long hiLong;
	long hiLat;
} WorldRect, *WORLDRECTP, **WORLDRECTH;

typedef struct {
	float loLong;
	float loLat;
	float hiLong;
	float hiLat;
} WorldRectF, *WORLDRECTFP, **WORLDRECTFH;
//--

///////////////////////////////////////////////////////////////////////////

typedef struct
{
	long		leUnits;
	long		leKey; 			// index to identify LE
	long		leCustomData; 	// space for custom LE data, default = 0
	WorldPoint	p; 				// x and y location
	double		z; 				// z position
	Seconds		releaseTime; 	// time of release, seconds since 1904
	double		ageInHrsWhenReleased;// age of oil in hours at time of release
	Seconds		clockRef; 		// time offset in seconds (for statistically varying use of time files)
	OilType		pollutantType; 	// L.E. pollutant type
	double		mass; 			// amount of pollutant (what units ?)
	double		density; 		// density in grams/cc
	double		windage;
	long 		dropletSize;		// microns
	short 		dispersionStatus;
	double		riseVelocity;	// cm/s
	OilStatus	statusCode; 	// not-released, floating, beached, etc.
	WorldPoint	lastWaterPt; 	// last on-water point before L.E. was beached
	Seconds		beachTime; 		// time when L.E. was beached
} LERec, *LERecP, **LERecH;

typedef struct
{
	float downStream;
	float crossStream;
	//	float angle;
} LEUncertainRec,*LEUncertainRecP,**LEUncertainRecH;

typedef struct
{
	float randCos;
	float randSin;
} LEWindUncertainRec,*LEWindUncertainRecP,**LEWindUncertainRecH;

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	pat1ValScale;
	double 	pat2ValScale;
} TC_OPTIMZE;

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
} TCM_OPTIMZE;

#define		kUCode			0					// supplied to UorV routine
#define		kVCode			1

typedef struct GridCellInfo
{
	long cellNum;	//index of rectangle
	long 	topLeft;	//top left index
	long 	topRight;	//top right index
	long 	bottomLeft;	//bottom left index
	long 	bottomRight;	//bottom right index
} GridCellInfo, *GridCellInfoPtr, **GridCellInfoHdl;

typedef struct SegInfo
{
	long pt1;	//index of first point
	long 	pt2;	//index of second point
	long 	islandNumber;	//land block
	Boolean 	isWater;	//land/water boudary
} SegInfo, *SegInfoPtr, **SegInfoHdl;

#define kPtCurUserNameLen 64
#define UNASSIGNEDINDEX -1
#define BOTTOMINDEX -2	// below last data value, but above bottom
#define CONSTANTCURRENT 0
#define CONSTANTWIND 0

enum { REGULAR=1, REGULAR_SWAFS, CURVILINEAR, TRIANGULAR, REGRIDDED};	// maybe eliminate regridded option
enum {TWO_D=1, BAROTROPIC, SIGMA, MULTILAYER, SIGMA_ROMS};	// gridtypes


typedef struct {
	double 	alongCurUncertainty;	
	double 	crossCurUncertainty;	
	double 	uncertMinimumInMPS;	
	double 	startTimeInHrs;	
	double 	durationInHrs;	
} UncertaintyParameters;

typedef struct {
	long fileOffsetToStartOfData;
	long lengthOfData; // implicit from the next one
	Seconds time;
} PtCurTimeData,*PtCurTimeDataP,**PtCurTimeDataHdl;


typedef struct {
	long timeIndex;
	VelocityFH dataHdl; // numVertices
}  LoadedData,*LoadedDataP;

typedef struct {
	long timeIndex;
	VelocityFH3D dataHdl; // numVertices
}  LoadedData3D,*LoadedDataP3D;


typedef struct {
	float totalDepth;
	long indexToDepthData; 
	long numDepths;
}  DepthDataInfo,*DepthDataInfoP,**DepthDataInfoH;

typedef struct {
	char pathName[kMaxNameLen];
	Seconds startTime; 
	Seconds endTime;
}  PtCurFileInfo,*PtCurFileInfoP,**PtCurFileInfoH;

typedef struct ScaleRec
{
	double					XScale;	/* Y = mX + b type linear scale and offsets */
	double					YScale;
	double					XOffset;
	double					YOffset;
} ScaleRec, *ScaleRecPtr;

typedef struct
{
	Boolean			setEddyValues;
	Seconds			fUncertainStartTime;
	double			fDuration; 				// duration time for uncertainty;
	double			fEddyDiffusion;		
	double			fEddyV0;			
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	
} CurrentUncertainyInfo;


// According to the manual this directive should be embedded 
// inside the structure definition, so may not do anything here (here pasted from OSSM.H):

#ifdef MAC
#pragma options align=mac68k
#endif

enum { KILOGRAMS = 1, METRICTONS, SHORTTONS,
	GALLONS, BARRELS, CUBICMETERS, LES }; // mass/volume units

enum {VOLUMETYPE,MASSTYPE};


enum { DIR_N = 1, DIR_NNE, DIR_NE, DIR_ENE, DIR_E, DIR_ESE, DIR_SE, DIR_SSE, DIR_S,
	DIR_SSW, DIR_SW, DIR_WSW, DIR_W, DIR_WNW, DIR_NW, DIR_NNW };

enum {DAYLIGHTSAVINGSON = 0, DAYLIGHTSAVINGSOFF = 1};	// allow users to turn off daylight savings time for shio tides

enum { NOVICEMODE = 1, INTERMEDIATEMODE, ADVANCEDMODE };  // model modes

enum {	kLinkToNone = 1, kLinkToTimeFile, kLinkToWindMover };

enum { NONE = 0, WINDSPEED, WINDSTRESS };	// for scaling

#define M19 1900

enum { M19REALREAL = 1, M19HILITEDEFAULT,
	
	M19MAGNITUDEDEGREES, M19DEGREESMAGNITUDE,
	
	M19MAGNITUDEDIRECTION, M19DIRECTIONMAGNITUDE,
	
	M19CANCEL, M19LABEL };

enum {TERMINATED = 2,NONTERMINATED};


#endif
