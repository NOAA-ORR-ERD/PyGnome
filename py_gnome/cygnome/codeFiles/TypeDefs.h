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

#define kMaxNameLen 256
#define TIMEVALUE_TOLERANCE 0.00001
#define kNumOSSMLandTypes 11
#define kNumOSSMWaterTypes 5
#define kOMapWidth 80
#define kOMapHeight 48
#define kOMapColorInd 8

#define	kOCurWidth			40
#define	kOCurHeight			24
#define	kVelsPerLine		10 // # of velocity numbers pairs per line
#define ITEM_OFFSET(index) ((index) * elementSize)
#define ITEM_PTR(index) (&(*L)[ITEM_OFFSET(index)])

#define COMPLETE_LE  (FORECAST_LE + UNCERTAINTY_LE)

typedef long ClassID;

const ClassID TYPE_UNDENTIFIED	 	= 0;
const ClassID TYPE_MODEL	 		= 100;
const ClassID TYPE_LELISTLIST	 	= 200;
const ClassID TYPE_MAPLIST 			= 201;
const ClassID TYPE_MOVERLIST	 	= 202;
const ClassID TYPE_LELIST 			= 300;
const ClassID TYPE_OSSMLELIST		= 301;
const ClassID TYPE_SPRAYLELIST		= 302;
const ClassID TYPE_CDOGLELIST		= 303;
const ClassID TYPE_MAP 				= 400;
const ClassID TYPE_OSSMMAP			= 401;
const ClassID TYPE_VECTORMAP		= 402;
const ClassID TYPE_PTCURMAP			= 403;
const ClassID TYPE_COMPOUNDMAP		= 404;

const ClassID TYPE_MOVER 			= 500;
const ClassID TYPE_RANDOMMOVER		= 501;
const ClassID TYPE_CATSMOVER		= 502;
const ClassID TYPE_WINDMOVER		= 503;
//const ClassID TYPE_CONSTANTMOVER	= 504; // no longer supported, replaced by an enhanced TYPE_WINDMOVER, JLM 2/18/00
const ClassID TYPE_COMPONENTMOVER	= 505;
const ClassID TYPE_PTCURMOVER		= 506;
const ClassID TYPE_CURRENTMOVER		= 507;
const ClassID TYPE_RANDOMMOVER3D	= 508;
const ClassID TYPE_CATSMOVER3D		= 509;
const ClassID TYPE_GRIDCURMOVER		= 510;
const ClassID TYPE_NETCDFMOVER		= 511;
const ClassID TYPE_NETCDFMOVERCURV	= 512;
const ClassID TYPE_NETCDFMOVERTRI	= 513;
const ClassID TYPE_NETCDFWINDMOVER	= 514;
const ClassID TYPE_GRIDWINDMOVER	= 515;
const ClassID TYPE_NETCDFWINDMOVERCURV	= 516;
const ClassID TYPE_TRICURMOVER	= 517;
const ClassID TYPE_TIDECURCYCLEMOVER	= 518;
const ClassID TYPE_COMPOUNDMOVER	= 519;
const ClassID TYPE_ADCPMOVER		= 520;

const ClassID TYPE_TIMEVALUES		= 600;
const ClassID TYPE_OSSMTIMEVALUES	= 601;
const ClassID TYPE_SHIOTIMEVALUES	= 602;
const ClassID TYPE_ADCPTIMEVALUES	= 602;
const ClassID TYPE_WEATHERER		= 700;
const ClassID TYPE_OSSMWEATHERER	= 701;
const ClassID TYPE_GRIDVEL			= 800;
const ClassID TYPE_RECTGRIDVEL		= 801;
const ClassID TYPE_TRIGRIDVEL		= 802;
const ClassID TYPE_TRIGRIDVEL3D		= 803;
const ClassID TYPE_CMAPLAYER 		= 901; //JLM

const ClassID TYPE_OVERLAY	= 910; //JLM
const ClassID TYPE_NESDIS_OVERLAY	= 920; //JLM
const ClassID TYPE_BUOY_OVERLAY	= 930; //JLM
const ClassID TYPE_BP_BUOY_OVERLAY	= 931; //JLM
const ClassID TYPE_SLDMB_BUOY_OVERLAY	= 932; //JLM
const ClassID TYPE_OVERFLIGHT_OVERLAY	= 940; //JLM

typedef short OilType;
typedef short OilStatus;
typedef short LandType;
typedef unsigned long Seconds; // duration in seconds, or seconds since 1904
typedef  unsigned long LETYPE ;

///// TYPES ///////////////////////////////////////////////////////////////////////

typedef struct {
	short f;
	CHARH buf;
	long bufSize;
	long base;
	long index;
	long fileLength;
	Boolean bufModified;
} BFPB, *BFPBP;

extern Rect CATSgridRect;
extern BFPB gRunSpillForecastFile;

typedef struct {
	char name[30];
	short previewLength;
	short (*compare)(VOIDPTR rec1, VOIDPTR rec2);
	short fp;
} IndexFile, *IndexFileP;

typedef struct {
	char name[30];
	BFPB bfpb;
	IndexFileP indexList;
} DataBase, *DataBaseP;


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

typedef struct
{	// for CDOG
	float depth;
	VelocityRec value;	//u,v
	double w;
	float temp;
	float sal;
} DepthValuesSet, *DepthValuesSetP, **DepthValuesSetH;

typedef struct
{	// for CDOG
	double time;	// time after model start in hours
	double q_oil;	// oil discharge rate
	double q_gas;	// gas discharge rate
	double temp;	// release temp (deg C)
	double diam;	// orifice diameter (m)
	double rho_oil;	// density of oil (kg/m^3)
	long n_den;	// release oil (n_den>=0) or water (n_den<0)
	long output_int; // output files after output_int steps
} DischargeData, *DischargeDataP, **DischargeDataH;

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

enum { OILSTAT_NOTRELEASED = 0, OILSTAT_INWATER = 2, OILSTAT_ONLAND,
	OILSTAT_OFFMAPS = 7, OILSTAT_EVAPORATED = 10};

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

enum { // moved TModel defines to TModel.cpp // JLM
	I_CLASSNAME = 0, I_CLASSACTIVE,
	
	I_MAPNAME = 0, I_REFLOATHALFLIFE, I_MOVERS,
	I_MAPS = 10, I_UMOVERS = 15, I_WIND = 16, I_LESETS = 20,I_MASSBALANCETOTALS = 21 ,I_MASSBALANCELINE,  I_WEATHERING = 30, I_OVERLAYS = 35,
	
	I_OVERLAYNAME = 0,
	
	// vector map overrides standard map items
	I_VMAPNAME = 0, I_VDRAWLANDBITMAP, I_VMOVERS, I_VSPILLBITMAP, I_VSPILLBITMAPACTIVE, I_VDRAWALLOWABLESPILLBITMAP, I_VREFLOATHALFLIFE, I_VDRAWESIBITMAP, I_VSHOWESILEGEND,
	
	// ptcur map overrides standard map items
	// I_PMAPNAME = 0, I_PDRAWLANDWATERBITMAP, I_PMOVERS, I_PREFLOATHALFLIFE,
	I_PMAPNAME = 0, I_PDRAWLANDWATERBITMAP, I_PMOVERS, I_PDRAWCONTOURS, I_PSETCONTOURS, I_PCONCTABLE, I_PSHOWLEGEND, I_PSHOWSURFACELES,
	I_PWATERDENSITY, I_PMIXEDLAYERDEPTH, I_PBREAKINGWAVEHT, I_PDIAGNOSTICSTRING, I_PDROPLETINFO, I_PREFLOATHALFLIFE, I_PDRAWCONTOURSFORMAP,
	
	I_LEFIRSTLINE = 1, I_LEACTIVE, I_LEWINDAGE, I_LESHOWHIDE, I_LEDISPERSE, I_LEDRAWRECT, I_LENATURALDISP, I_LERELEASE_TIMEPOSITION,I_LERELEASE_MASSBALANCE, I_LELIST = 100,
	
	I_RANDOMNAME = 0, I_RANDOMACTIVE, I_RANDOMUFACTOR,
	I_RANDOMAREA, I_RANDOMMAGNITUDE, I_RANDOMANGLE,
	I_RANDOMDURATION, I_RANDOMVERTAREA,
	
	I_COMPONENTNAME = 0, I_COMPONENTACTIVE, I_COMPONENTSCALEBY, I_COMPONENTREFERENCE, I_COMPONENTLAT, I_COMPONENTLONG,
	I_COMPONENT1NAME, I_COMPONENT1GRID, I_COMPONENT1ARROWS, I_COMPONENT1DIRECTION, I_COMPONENT1SCALE,
	I_COMPONENT2NAME, I_COMPONENT2GRID, I_COMPONENT2ARROWS, I_COMPONENT2DIRECTION, I_COMPONENT2SCALE,
	
	I_COMPOUNDNAME = 0, I_COMPOUNDACTIVE, I_COMPOUNDCURRENT,
	
	I_COMPOUNDMAPNAME = 0, I_COMPOUNDMAP, I_COMPOUNDMOVERS,
	
	I_CATSNAME = 0 ,I_CATSACTIVE, I_CATSGRID, I_CATSARROWS,
	I_CATSREFERENCE, I_CATSSCALING, I_CATSLAT, I_CATSLONG,
	I_CATSTIMEFILE, I_CATSTIMEFILEACTIVE, 
	I_CATSUNCERTAINTY,I_CATSSTARTTIME,I_CATSDURATION, I_CATSDOWNCUR, I_CATSCROSSCUR, I_CATSDIFFUSIONCOEFFICIENT,I_CATSEDDYV0,
	I_CATSTIMEENTRIES = 100,
	
	I_ADCPNAME = 0 ,I_ADCPACTIVE, I_ADCPGRID, I_ADCPARROWS,
	I_ADCPREFERENCE, I_ADCPSCALING, I_ADCPLAT, I_ADCPLONG,
	I_ADCPTIMEFILE, I_ADCPTIMEFILEACTIVE, 
	I_ADCPUNCERTAINTY,I_ADCPSTARTTIME,I_ADCPDURATION, I_ADCPDOWNCUR, I_ADCPCROSSCUR, I_ADCPDIFFUSIONCOEFFICIENT,I_ADCPEDDYV0,
	I_ADCPTIMEENTRIES = 100,
	
	I_ADCPSTATIONNAME = 0 ,I_ADCPSTATIONACTIVE, /*I_ADCPSTATIONGRID, I_ADCPSTATIONARROWS,*/
	I_ADCPSTATIONREFERENCE, /*I_ADCPSTATIONSCALING,*/ I_ADCPSTATIONLAT, I_ADCPSTATIONLONG, I_ADCPSTATIONDATA,
	
	I_CONSTNAME = 0, I_CONSTACTIVE, I_CONSTWINDAGE, I_CONSTMAGDIRECTION,
	
	I_WINDNAME = 0, I_WINDUNCERTAIN,I_WINDACTIVE,I_SUBSURFACEWINDACTIVE,
	I_WINDCONVERSION, I_WINDWINDAGE, 
	I_WINDTIMEFILE,I_WINDSPEEDSCALE,I_WINDANGLESCALE, I_WINDSTARTTIME,I_WINDDURATION,
	//I_WINDTIMEENTRIES = 100, //LE & JLM 6/18/98
	I_WINDTIMEENTRIES, I_WINDBARB,
	
	I_WEATHERNAME = 0, I_WEATHERACTIVE };

enum {DONT_DISPERSE, DISPERSE, HAVE_DISPERSED, DISPERSE_NAT, HAVE_DISPERSED_NAT, EVAPORATE, HAVE_EVAPORATED, REMOVE, HAVE_REMOVED};	// LE dispersion status

enum {FORECAST_LE = 1, UNCERTAINTY_LE = 2, nextOne = 4};


class TClassID;
class TTriGridVel;
class TTriGridVel3D;
class TCurrentMover;

typedef struct {
	TClassID *owner;
	long index;
	short indent;
	short bullet;
} ListItem;

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
	double 	uncertaintyValue;
} TR_OPTIMZE;

//++ Geometry


typedef struct {
	float pLong;
	float pLat;
} WorldPoint, *WORLDPOINTP, **WORLDPOINTH;

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

typedef struct GeoPoint
{ 	// right handed coordinate system
	long				h; 			// Latitude  (dec_deg*10e6)
	long				v;  		// Longitude (dec_deg*10e6)
}GeoPoint,  *GeoPointPtr,  **GeoPointHdl;

typedef struct {
	long fromLong;
	long fromLat;
	long toLong;
	long toLat;
} Segment, *SEGMENTP, **SEGMENTH, **SegmentsHdl;

typedef struct { long x; long y; } Vector;

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
	WorldPoint	p; 				// x and y location
	double		z; 				// z position
	Seconds		releaseTime; 	// time of release, seconds since 1904
	double		ageInHrsWhenReleased;// age of oil in hours at time of release
	OilType		pollutantType; 	// L.E. pollutant type
	double		density; 		// density in same units as everything else, added 1/23/03
	//double		riseVelocity;
} InitialLEInfoRec, *InitialLEInfoRecPtr, **InitialLEInfoRecHdl;

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

typedef struct
{
	Boolean bDisperseOil;
	Seconds timeToDisperse;
	Seconds duration;
	float amountToDisperse;
	WorldRect areaToDisperse;
	Boolean lassoSelectedLEsToDisperse;
	double api;
} DispersionRec,*DispersionRecP,**DispersionRecH;

typedef struct
{
	Seconds timeAfterSpill;
	//Seconds duration;
	float amountDispersed;
	float amountEvaporated;
	float amountRemoved;
	//double api;
} AdiosInfoRec,**AdiosInfoRecH;

typedef struct
{
	Seconds timeAfterSpill;
	float amountReleased;
	float amountFloating;
	float amountDispersed;
	float amountEvaporated;
	float amountBeached;
	float amountOffMap;
	float amountRemoved;
} BudgetTableData,**BudgetTableDataH;

typedef struct
{
	double dropletSize;
	double probability;
} DropletInfoRec,**DropletInfoRecH;

typedef struct
{
	double windageA;
	double windageB;	
	double persistence;	// in hours, .25 or infinite
} WindageRec,*WindageRecP;

typedef struct
{
	long 		numOfLEs;			// number of L.E.'s in this set
	OilType		pollutantType;		// type of pollutant
	double		totalMass; 			// total mass of all le's in list (in massUnits)
	short		massUnits; 			// units for mass quantity
	Seconds		startRelTime;		// start release time
	Seconds		endRelTime;			// end release time
	WorldPoint	startRelPos;		// start L.E. position
	WorldPoint	endRelPos;			// end   L.E. position
	Boolean		bWantEndRelTime;
	Boolean		bWantEndRelPosition;
	
	double		z;
	double		density;
	double		ageInHrsWhenReleased;
	
	double		riseVelocity;	// will this vary over a set? - maybe a flag if so
	
	char			spillName[kMaxNameLen];
} LESetSummary; 					// record used to summarize LE sets

typedef struct
{
	char		pollutant [kMaxNameLen];
	double		halfLife [3];
	double		percent [3];
	double		XK [3];
	double		EPRAC;
	Boolean		bModified;
	
} OilComponent;

typedef struct
{
	unsigned long ticksAtCreation;
	short counter;
} UNIQUEID;

struct InterpolationVal {	
	long ptIndex1;
	long ptIndex2;
	long ptIndex3;
	double alpha1;
	double alpha2;
	double alpha3;
};

#endif