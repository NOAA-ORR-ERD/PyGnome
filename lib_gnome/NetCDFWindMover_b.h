/*
 *  NetCDFWindMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMover_b__
#define __NetCDFWindMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "WindMover/WindMover_b.h"
#include "PtCurMover/PtCurMover.h"

class NetCDFWindMover_b : virtual public WindMover_b {

public:
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	
	////// start: new fields to support multi-file NetCDFPathsFile
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl; 
	////// end:  multi-file fields
	
	char		fPathName[kMaxNameLen];
	char		fFileName[kPtCurUserNameLen]; // short file name
	//char		fFileName[kMaxNameLen]; // short file name - might want to allow longer names
	
	long fNumRows;
	long fNumCols;
	//NetCDFVariables fVar;
	TGridVel	*fGrid;	//VelocityH		grid; 
	//PtCurTimeDataHdl fTimeDataHdl;
	Seconds **fTimeHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	short fUserUnits;
	//double fFillValue;
	float fFillValue;
	float fWindScale;
	float fArrowScale;
	long fTimeShift;		// to convert GMT to local time
	Boolean fAllowExtrapolationOfWinds;
	Boolean fIsOptimizedForStep;

};


enum {
	I_NETCDFWINDNAME = 0, I_NETCDFWINDACTIVE, I_NETCDFWINDSHOWGRID, I_NETCDFWINDSHOWARROWS, I_NETCDFWINDUNCERTAIN,
	I_NETCDFWINDSPEEDSCALE,I_NETCDFWINDANGLESCALE, I_NETCDFWINDSTARTTIME,I_NETCDFWINDDURATION
};

#endif
