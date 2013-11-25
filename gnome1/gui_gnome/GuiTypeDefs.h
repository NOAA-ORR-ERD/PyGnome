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

#ifndef pyGNOME
#include "Earl.h"
#include "Typedefs.h"
#endif

///////////////////////////////////////////////////////////////////////////

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

#endif
