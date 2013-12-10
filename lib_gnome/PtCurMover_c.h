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

#include "Basics.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "CurrentMover_c.h"
#include "DagTree.h"

#ifndef pyGNOME
#include "GridVel.h"
#else
#include "GridVel_c.h"
#include "Map_c.h"
#define TGridVel GridVel_c
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
	//virtual ClassID 	GetClassID () { return TYPE_PTCURMOVER; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMOVER) return TRUE; return TCurrentMover::IAm(id); }
	
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	virtual WorldRect	GetGridBounds(){return fGrid->GetBounds();}	

	
	long 					GetNumTimesInFile();
	long 					GetNumFiles();
	long					GetNumDepths(void);
	LongPointHdl 		GetPointsHdl();
	TopologyHdl 		GetTopologyHdl();
	long			 		WhatTriAmIIn(WorldPoint p);
	void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList);
	virtual OSErr 		PrepareForModelRun(); 
	virtual void 		ModelStepIsDone();
	
	void 					DisposeLoadedData(LoadedData * dataPtr);	
	void 					ClearLoadedData(LoadedData * dataPtr);
	
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& model_time);	// AH 07/17/2012
	OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	OSErr 				ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataHdl,Boolean setStartTime);	// AH 07/17/2012

};


#undef TMap
#endif
