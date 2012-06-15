/*
 *  TideCurCycleMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TideCurCycleMover_c__
#define __TideCurCycleMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "TideCurCycleMover_b.h"
#include "CATSMover_c.h"

#ifndef pyGNOME
#include "TMap.h"
#else
#include "Map_c.h"
#define TMap Map_c
#endif

class TideCurCycleMover_c : virtual public TideCurCycleMover_b, virtual public CATSMover_c {

public:
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	
	LongPointHdl 		GetPointsHdl();
	//long 					GetVelocityIndex(WorldPoint p);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	
	/*virtual OSErr		GetStartTime(Seconds *startTime);
	 virtual OSErr		GetEndTime(Seconds *endTime);*/
	virtual double		GetStartUVelocity(long index);
	virtual double		GetStartVVelocity(long index);
	virtual double		GetEndUVelocity(long index);
	virtual double		GetEndVVelocity(long index);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual OSErr 		ComputeVelocityScale();
	
	Boolean 			IsDryTriangle(long index1, long index2, long index3, float timeAlpha);
	Boolean 			IsDryTri(long triIndex);
	VelocityRec 		GetStartVelocity(long index, Boolean *isDryPt);
	VelocityRec 		GetEndVelocity(long index, Boolean *isDryPt);
	
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, bool); // AH 04/16/12
	virtual void 		ModelStepIsDone();
	OSErr 				ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts); 
	virtual Boolean 	CheckInterval(long &timeDataInterval);
	virtual OSErr	 	SetInterval(char *errmsg);
	long				GetNumTimesInFile();
	void 				DisposeLoadedData(LoadedData * dataPtr);	
	void 				ClearLoadedData(LoadedData * dataPtr);	
	OSErr				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 

};

#endif
