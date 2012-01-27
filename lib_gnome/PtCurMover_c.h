/*
 *  PtCurMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __PtCurMover_c__
#define __PtCurMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "GridVel.h"
#include "DagTree.h"

#ifdef pyGNOME
	#define TMap Map_c
#endif


class PtCurMover_c : virtual public CurrentMover_c {

public:
	
	PTCurVariables fVar;
	TGridVel	*fGrid;	
	PtCurTimeDataHdl fTimeDataHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	FLOATH fDepthsH;
	DepthDataInfoH fDepthDataInfo;
	Boolean fIsOptimizedForStep;
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl;
	
	PtCurMover_c (TMap *owner, char *name);
	PtCurMover_c () {}
	virtual ClassID 	GetClassID () { return TYPE_PTCURMOVER; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMOVER) return TRUE; return CurrentMover_c::IAm(id); }
	
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	virtual WorldRect	GetGridBounds(){return fGrid->GetBounds();}	

	
	long 					GetNumTimesInFile();
	long 					GetNumFiles();
	long					GetNumDepths(void);
	LongPointHdl 		GetPointsHdl();
	TopologyHdl 		GetTopologyHdl();
	long			 		WhatTriAmIIn(WorldPoint p);
	void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	
	virtual void 		ModelStepIsDone();
	
	void 					DisposeLoadedData(LoadedData * dataPtr);	
	void 					ClearLoadedData(LoadedData * dataPtr);
	

};


#undef TMap
#endif
