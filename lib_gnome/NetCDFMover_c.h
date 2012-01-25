/*
 *  NetCDFMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMover_c__
#define __NetCDFMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFMover_b.h"
#include "CurrentMover/CurrentMover_c.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

class NetCDFMover_c : virtual public NetCDFMover_b, virtual public CurrentMover_c {

public:
	NetCDFMover_c (TMap *owner, char *name);
	NetCDFMover_c () {}
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	virtual long 		GetVelocityIndex(WorldPoint p);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	void 				GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	float 				GetMaxDepth();
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	long 					GetNumDepths(void);
	virtual long 		GetNumDepthLevels();
	Seconds 				GetTimeValue(long index);
	virtual OSErr		GetStartTime(Seconds *startTime);
	virtual OSErr		GetEndTime(Seconds *endTime);
	virtual double 	GetStartUVelocity(long index);
	virtual double 	GetStartVVelocity(long index);
	virtual double 	GetEndUVelocity(long index);
	virtual double 	GetEndVVelocity(long index);
	virtual double	GetDepthAtIndex(long depthIndex, double totalDepth);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	float		GetTotalDepth(WorldPoint refPoint, long triNum);
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	
	
};

#undef TMap
#endif
