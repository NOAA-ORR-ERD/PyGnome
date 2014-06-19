/*
 *  GuiTypeDefs.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GuiTypeDefs__
#define __GuiTypeDefs__

#include <time.h>

#include "Earl.h"
#include "Typedefs.h"

///////////////////////////////////////////////////////////////////////////

#define kDefSaveFileName "Untitled.sav"		// STH
#define	kDefLEFileName "LEFile"				// STH

#define kNumOSSMLandTypes 11
#define kNumOSSMWaterTypes 5
#define kOMapWidth 80
#define kOMapHeight 48
#define kOMapColorInd 8

#define	kOCurWidth			40
#define	kOCurHeight			24
#define	kVelsPerLine		10 // # of velocity numbers pairs per line

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
	short f;
	CHARH buf;
	long bufSize;
	long base;
	long index;
	long fileLength;
	Boolean bufModified;
} BFPB, *BFPBP;
extern BFPB gRunSpillForecastFile;

typedef struct
{
	Seconds		frameTime;
	char		frameLEFName [255];
} LEFrameRec, *LEFrameRecPtr;

typedef struct
{
	Seconds	startTime;			// time to start model run
	Seconds	duration;			// duration of model run
	Seconds computeTimeStep;	// time step for computation
	Boolean	bUncertain; 		// uncertainty on / off
	Boolean preventLandJumping;	// use the code that checks for land jumping
} TModelDialogVariables;

typedef struct {
	char		pathName[kMaxNameLen];
	char		userName[kPtCurUserNameLen]; // user name for the file
	double 	alongCurUncertainty;	
	double 	crossCurUncertainty;	
	double 	uncertMinimumInMPS;	
	double 	curScale;	
	double 	startTimeInHrs;	
	double 	durationInHrs;	
	//
	long		numLandPts; // 0 if boundary velocities defined, else set boundary velocity to zero
	long		maxNumDepths;
	short		gridType;
	double	bLayerThickness;
	//
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	Boolean	bUncertaintyPointOpen;
	double 	arrowScale;
	double 	arrowDepth;	// depth level where velocities will be shown
} PTCurVariables;

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
	double		halfLife;	// hours
	
	char			spillName[kMaxNameLen];
} LESetSummary; 					// record used to summarize LE sets

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
	char		pollutant [kMaxNameLen];
	double		halfLife [3];
	double		percent [3];
	double		XK [3];
	double		EPRAC;
	Boolean		bModified;
	
} OilComponent;

typedef struct {
	long segNo;
	long startPt;
	long endPt;
	long numBeachedLEs;
	float segmentLengthInKm;
	float	gallonsOnSegment;
	//Seconds time;
} OiledShorelineData,*OiledShorelineDataP,**OiledShorelineDataHdl;

typedef struct {
	short		color;
	char		code[2];
	WorldRect	bounds;
	long		numPoints;
	long		firstPoint;
} PolygonType, *PolygonP, **PolygonH;

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
const ClassID TYPE_MAP3D			= 405;
const ClassID TYPE_GRIDMAP			= 406;

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
const ClassID TYPE_GRIDWNDMOVER	= 515;
const ClassID TYPE_NETCDFWINDMOVERCURV	= 516;
const ClassID TYPE_TRICURMOVER	= 517;
const ClassID TYPE_TIDECURCYCLEMOVER	= 518;
const ClassID TYPE_COMPOUNDMOVER	= 519;
const ClassID TYPE_ADCPMOVER		= 520;
const ClassID TYPE_GRIDCURRENTMOVER	= 521;
const ClassID TYPE_GRIDWINDMOVER	= 522;
const ClassID TYPE_CURRENTCYCLEMOVER	= 523;

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
const ClassID TYPE_TIMEGRIDVEL		= 810;
const ClassID TYPE_TIMEGRIDVELRECT		= 811;
const ClassID TYPE_TIMEGRIDVELCURV		= 812;
const ClassID TYPE_TIMEGRIDVELTRI		= 813;
const ClassID TYPE_TIMEGRIDWINDRECT		= 814;
const ClassID TYPE_TIMEGRIDWINDCURV		= 815;
const ClassID TYPE_TIMEGRIDCURRECT		= 816;
const ClassID TYPE_TIMEGRIDCURTRI		= 817;
const ClassID TYPE_CMAPLAYER 		= 901; //JLM

const ClassID TYPE_OVERLAY	= 910; //JLM
const ClassID TYPE_NESDIS_OVERLAY	= 920; //JLM
const ClassID TYPE_BUOY_OVERLAY	= 930; //JLM
const ClassID TYPE_BP_BUOY_OVERLAY	= 931; //JLM
const ClassID TYPE_SLDMB_BUOY_OVERLAY	= 932; //JLM
const ClassID TYPE_OVERFLIGHT_OVERLAY	= 940; //JLM

typedef struct
{
	unsigned long ticksAtCreation;
	short counter;
} UNIQUEID;

#endif
