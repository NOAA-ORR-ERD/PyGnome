/*
 *  CompoundMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CompoundMover_c__
#define __CompoundMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "CompoundMover_b.h"
#include "CurrentMover_c.h"

#ifdef pyGNOME
#define TMap Map_c
#define TCurrentMover CurrentMover_c
#endif

class TCurrentMover;
class TTriGridVel;
class TCompoundMap;

class CompoundMover_c : virtual public CompoundMover_b, virtual public CurrentMover_c {

public:
	CompoundMover_c (TMap *owner, char *name);
	CompoundMover_c () {}
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual float		GetArrowDepth();

	virtual	LongPointHdl	GetPointsHdl();
	TTriGridVel*		GetGrid(Boolean wantRefinedGrid);
	TTriGridVel3D*		GetGrid3D(Boolean wantRefinedGrid);
	TCurrentMover*		Get3DCurrentMover();
	TTriGridVel3D*		GetGrid3DFromMoverIndex(long moverIndex);
	TCurrentMover*		Get3DCurrentMoverFromIndex(long moverIndex);
	virtual Boolean		IAmA3DMover();
	virtual OSErr		AddMover (TMover *theMover, short where);

};

#undef TMap
#undef TCurrentMover
#endif
