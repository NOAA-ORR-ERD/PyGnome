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

#include "Earl.h"
#include "TypeDefs.h"
#include "TriCurMover_b.h"
#include "CurrentMover/CurrentMover_c.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

class TriCurMover_c : virtual public TriCurMover_b, virtual public CurrentMover_c {

public:
	TriCurMover_c (TMap *owner, char *name);
	TriCurMover_c () {}
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	
	long					GetNumDepths(void);
	float 				GetMaxDepth(void);
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	virtual LongPointHdl GetPointsHdl();
	TopologyHdl 		GetTopologyHdl();
	long			 		WhatTriAmIIn(WorldPoint p);
	OSErr 				GetTriangleCentroid(long trinum, LongPoint *p);
	void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	OSErr 			CalculateVerticalGrid(LongPointHdl ptsH, FLOATH totalDepthH, TopologyHdl topH, long numTri,FLOATH sigmaLevels, long numSigmaLevels);
	long			CreateDepthSlice(long triNum, float **depthSlice);
	
	
	
};

#undef TMap
#endif
