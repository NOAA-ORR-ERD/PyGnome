/*
 *  GridCurMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridCurMover_c__
#define __GridCurMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "PtCurMover.h"
#include "CATSMover_c.h"

enum {
	I_GRIDCURNAME = 0 ,
	I_GRIDCURACTIVE, 
	I_GRIDCURGRID, 
	I_GRIDCURARROWS,
	//I_GRIDCURSCALE,
	I_GRIDCURUNCERTAINTY,
	I_GRIDCURSTARTTIME,
	I_GRIDCURDURATION, 
	I_GRIDCURALONGCUR,
	I_GRIDCURCROSSCUR,
	//I_GRIDCURMINCURRENT
};

class GridCurMover_c : virtual public CATSMover_c {

public:
	long fNumRows;
	long fNumCols;
	PtCurTimeDataHdl fTimeDataHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	short fUserUnits;
	char fPathName[kMaxNameLen];
	char fFileName[kMaxNameLen];
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl;
	
	
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	
	long 				GetVelocityIndex(WorldPoint p);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	
	virtual OSErr		GetStartTime(Seconds *startTime);
	virtual OSErr		GetEndTime(Seconds *endTime);
	virtual double		GetStartUVelocity(long index);
	virtual double		GetStartVVelocity(long index);
	virtual double		GetEndUVelocity(long index);
	virtual double		GetEndVVelocity(long index);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);	
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	
};

#endif
