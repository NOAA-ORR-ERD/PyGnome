/*
 *  NetCDFMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMover_b__
#define __NetCDFMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover/CurrentMover_b.h"
#include "PtCurMover/PtCurMover.h"

typedef struct {
	char		pathName[kMaxNameLen];
	char		userName[kPtCurUserNameLen]; // user name for the file, or short file name
	//char		userName[kMaxNameLen]; // user name for the file, or short file name - might want to allow longer names...
	double 	alongCurUncertainty;	
	double 	crossCurUncertainty;	
	double 	uncertMinimumInMPS;	
	double 	curScale;	// use value from file? 	
	double 	startTimeInHrs;	
	double 	durationInHrs;	
	//
	//long		numLandPts; // 0 if boundary velocities defined, else set boundary velocity to zero
	long		maxNumDepths;
	short		gridType;
	//double	bLayerThickness;
	//
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	Boolean	bUncertaintyPointOpen;
	double 	arrowScale;
	double 	arrowDepth;	// depth level where velocities will be shown
} NetCDFVariables;


class NetCDFMover_b : virtual public CurrentMover_b {

public:
	long fNumRows;
	long fNumCols;
	long fNumDepthLevels;
	NetCDFVariables fVar;
	Boolean bShowDepthContours;
	Boolean bShowDepthContourLabels;
	TGridVel	*fGrid;	//VelocityH		grid; 
	//PtCurTimeDataHdl fTimeDataHdl;
	Seconds **fTimeHdl;
	float **fDepthLevelsHdl;	// can be depth levels, sigma, or sc_r (for roms formula)
	float **fDepthLevelsHdl2;	// Cs_r (for roms formula)
	float hc;	// parameter for roms formula
	LoadedData fStartData; 
	LoadedData fEndData;
	FLOATH fDepthsH;	// check what this is, maybe rename
	DepthDataInfoH fDepthDataInfo;
	float fFillValue;
	Boolean fIsNavy;	// special variable names for Navy, maybe change to grid type depending on Navy options
	Boolean fIsOptimizedForStep;
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl;
	//Seconds fTimeShift;		// to convert GMT to local time
	long fTimeShift;		// to convert GMT to local time
	Boolean fAllowExtrapolationOfCurrentsInTime;
	Boolean fAllowVerticalExtrapolationOfCurrents;
	float	fMaxDepthForExtrapolation;
	Rect fLegendRect;
	//double fOffset_u;
	//double fOffset_v;
	//double fCurScale_u;
	//double fCurScale_v;
	
};


enum { REGULAR=1, REGULAR_SWAFS, CURVILINEAR, TRIANGULAR, REGRIDDED};	// maybe eliminate regridded option

enum {
	I_NETCDFNAME = 0 ,
	I_NETCDFACTIVE, 
	I_NETCDFGRID, 
	I_NETCDFARROWS,
	I_NETCDFSCALE,
	I_NETCDFUNCERTAINTY,
	I_NETCDFSTARTTIME,
	I_NETCDFDURATION, 
	I_NETCDFALONGCUR,
	I_NETCDFCROSSCUR,
	//I_NETCDFMINCURRENT
};

#endif
