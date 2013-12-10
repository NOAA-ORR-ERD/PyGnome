/*
 *  NetCDFWindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMover_c__
#define __NetCDFWindMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "WindMover_c.h"


#ifndef pyGNOME
#include "GridVel.h"
#else
#include "GridVel_c.h"
#define TGridVel GridVel_c
#endif


enum {
	I_NETCDFWINDNAME = 0, I_NETCDFWINDACTIVE, I_NETCDFWINDSHOWGRID, I_NETCDFWINDSHOWARROWS, I_NETCDFWINDUNCERTAIN,
	I_NETCDFWINDSPEEDSCALE,I_NETCDFWINDANGLESCALE, I_NETCDFWINDSTARTTIME,I_NETCDFWINDDURATION
};

class NetCDFWindMover_c : virtual public WindMover_c {

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

	NetCDFWindMover_c (TMap *owner, char* name);
	NetCDFWindMover_c () {}
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual long 		GetVelocityIndex(WorldPoint p);
	virtual LongPoint 		GetVelocityIndices(WorldPoint wp); /*{LongPoint lp = {-1,-1}; printError("GetVelocityIndices not defined for windmover"); return lp;}*/
	Seconds 			GetTimeValue(long index);
	//virtual OSErr		GetStartTime(Seconds *startTime);
	//virtual OSErr		GetEndTime(Seconds *endTime);
	virtual OSErr		GetStartTime(Seconds *startTime);
	virtual OSErr		GetEndTime(Seconds *endTime);
	virtual double 	GetStartUVelocity(long index);
	virtual double 	GetStartVVelocity(long index);
	virtual double 	GetEndUVelocity(long index);
	virtual double 	GetEndVVelocity(long index);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	//virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVER; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVER) return TRUE; return TWindMover::IAm(id); }

	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
};

#endif
