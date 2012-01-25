/*
 *  GridWindMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridWindMover_b__
#define __GridWindMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "WindMover/WindMover_b.h"
#include "PtCurMover/PtCurMover.h"

class GridWindMover_b : virtual public WindMover_b {

public:
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	char		fPathName[kMaxNameLen];
	char		fFileName[kPtCurUserNameLen]; // short file name
	
	long fNumRows;
	long fNumCols;
	TGridVel	*fGrid;	//VelocityH		grid; 
	PtCurTimeDataHdl fTimeDataHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	short	fUserUnits; 
	//float fFillValue;
	float fWindScale;	// not using this
	float fArrowScale;	// not using this
	Boolean fIsOptimizedForStep;
	
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl;
		

};


enum {
	I_GRIDWINDNAME = 0, I_GRIDWINDACTIVE, I_GRIDWINDSHOWGRID, I_GRIDWINDSHOWARROWS, I_GRIDWINDUNCERTAIN,
	I_GRIDWINDSPEEDSCALE,I_GRIDWINDANGLESCALE, I_GRIDWINDSTARTTIME,I_GRIDWINDDURATION
};

#endif
