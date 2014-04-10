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
#include "CATSMover_c.h"

#ifndef pyGNOME
#include "TMap.h"
#else
#include "Map_c.h"
#define TMap Map_c
#endif

class TideCurCycleMover_c : virtual public CATSMover_c {

public:
	//long fNumRows;
	//long fNumCols;
	//PtCurTimeDataHdl fTimeDataHdl;
	Seconds **fTimeHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	float fFillValue;
	float fDryValue;
	short fUserUnits;
	char fPathName[kMaxNameLen];
	char fFileName[kMaxNameLen];
	LONGH fVerdatToNetCDFH;		// these two fields will be in curvilinear if we extend there
	WORLDPOINTFH fVertexPtsH;	// may not need this if set pts in dagtree	
	long fNumNodes;
	short fPatternStartPoint;	// maxflood, maxebb, etc
	float fTimeAlpha;
	char fTopFilePath[kMaxNameLen];
	//Seconds model_start_time;	// for the diagnostic case - no time file look at the patterns in the file that have no absolute time associated with them
	
#ifndef pyGNOME
	TideCurCycleMover_c (TMap *owner, char *name);
#endif
	TideCurCycleMover_c ();
	~TideCurCycleMover_c () { Dispose (); }
	virtual void		Dispose ();

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	
	LongPointHdl 		GetPointsHdl();
	//long 					GetVelocityIndex(WorldPoint p);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	
	/*virtual OSErr		GetStartTime(Seconds *startTime);
	 virtual OSErr		GetEndTime(Seconds *endTime);*/
	virtual double		GetStartUVelocity(long index);
	virtual double		GetStartVVelocity(long index);
	virtual double		GetEndUVelocity(long index);
	virtual double		GetEndVVelocity(long index);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual OSErr       ComputeVelocityScale(const Seconds& model_time);
	
	Boolean 			IsDryTriangle(long index1, long index2, long index3, float timeAlpha);
	Boolean 			IsDryTri(long triIndex);
	VelocityRec 		GetStartVelocity(long index, Boolean *isDryPt);
	VelocityRec 		GetEndVelocity(long index, Boolean *isDryPt);
	
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	OSErr 				ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts); 
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time); // AH 07/17/2012
	long				GetNumTimesInFile();
	void 				DisposeLoadedData(LoadedData * dataPtr);	
	void 				ClearLoadedData(LoadedData * dataPtr);	
	OSErr				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 

};

#endif
