/*
 *  GridWindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridWindMover_c__
#define __GridWindMover_c__

#include "Basics.h"
#include "PtCurMover.h"
#include "WindMover_c.h"

enum {
	I_GRIDWINDNAME = 0, I_GRIDWINDACTIVE, I_GRIDWINDSHOWGRID, I_GRIDWINDSHOWARROWS, I_GRIDWINDUNCERTAIN,
	I_GRIDWINDSPEEDSCALE,I_GRIDWINDANGLESCALE, I_GRIDWINDSTARTTIME,I_GRIDWINDDURATION
};

class GridWindMover_c : virtual public WindMover_c {

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
	
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList);
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual long 		GetVelocityIndex(WorldPoint p);

	
};

#endif
