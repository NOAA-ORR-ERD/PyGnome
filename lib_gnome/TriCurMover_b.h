/*
 *  TriCurMover_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TriCurMover_b__
#define __TriCurMover_b__

#include "Earl.h"
#include "TypeDefs.h"
#include "DagTree.h"
#include "my_build_list.h"
#include "GridVel.h"
#include "PtCurMover/PtCurMover.h"
#include "DagTree.h"
#include "CurrentMover/CurrentMover_b.h"

Boolean IsTriCurFile (char *path);
Boolean IsTriCurVerticesHeaderLine(char *s, long* numPts);

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

class TriCurMover_b : virtual public CurrentMover_b {

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
	
};


#endif
