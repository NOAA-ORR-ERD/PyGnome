/*
 *  TriCurMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TriCurMover_c__
#define __TriCurMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "DagTree.h"
#include "CurrentMover_c.h"


#ifndef pyGNOME
#include "GridVel.h"
#else
#include "GridVel_c.h"
#define TGridVel GridVel_c
#define TMap Map_c
#endif

Boolean IsTriCurFile (char *path);
Boolean IsTriCurVerticesHeaderLine(const char *s, long* numPts);

enum {ONELAYER_CONSTDENS=1, ONELAYER_VARDENS, TWOLAYER_CONSTDENS, TWOLAYER_VARDENS};	// gridtypes

typedef struct {
	char		curFilePathName[kMaxNameLen]; // currents
	char		sshFilePathName[kMaxNameLen]; // sea surface height
	char		pycFilePathName[kMaxNameLen]; // pycnocline depth
	char		lldFilePathName[kMaxNameLen];	// lower level density
	char		uldFilePathName[kMaxNameLen]; // upper level density
	short 	modelType;	// 1 layer constant density, 1 layer variable density, 2 layer constant density, 2 layer variable density	
	double 	scaleVel;	// cm/s
	double 	bottomBLThickness;	// cm
	double 	upperEddyViscosity;	// cm^2/s
	double 	lowerEddyViscosity;	// cm^2/s
	double 	upperLevelDensity;	// gm/cm^3
	double 	lowerLevelDensity;	// gm/cm^3
} BaromodesParameters;

class TriCurMover_c : virtual public CurrentMover_c {

public:
	PTCurVariables fVar;	// not sure if this is really necessary
	BaromodesParameters fInputValues;
	TGridVel	*fGrid;	
	PtCurTimeDataHdl fTimeDataHdl;
	LoadedData fStartData; 
	LoadedData fEndData;
	FLOATH fDepthsH;	
	DepthDataInfoH fDepthDataInfo;	// triangle info?
	Boolean fIsOptimizedForStep;
	//Boolean fOverLap;
	//Seconds fOverLapStartTime;
	//PtCurFileInfoH	fInputFilesHdl;
	Rect fLegendRect;
	Boolean bShowDepthContours;
	Boolean bShowDepthContourLabels;
	
	TriCurMover_c (TMap *owner, char *name);
	TriCurMover_c () {}
	
	//virtual ClassID 	GetClassID () { return TYPE_TRICURMOVER; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_TRICURMOVER) return TRUE; return TCurrentMover::IAm(id); }
	virtual Boolean		IAmA3DMover(){return true;}

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	
	long					GetNumDepths(void);
	float 				GetMaxDepth(void);
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	virtual LongPointHdl GetPointsHdl();
	TopologyHdl 		GetTopologyHdl();
	long			 		WhatTriAmIIn(WorldPoint p);
	OSErr 				GetTriangleCentroid(long trinum, LongPoint *p);
	void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	OSErr				CalculateVerticalGrid(LongPointHdl ptsH, FLOATH totalDepthH, TopologyHdl topH, long numTri,FLOATH sigmaLevels, long numSigmaLevels);
	long				CreateDepthSlice(long triNum, float **depthSlice);
	void 					DisposeLoadedData(LoadedData * dataPtr);	
	void 					ClearLoadedData(LoadedData * dataPtr);
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time);	// AH 07/17/2012
	OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	long 				GetNumTimesInFile();
	
};

#undef TMap
#undef TGridVel
#endif
