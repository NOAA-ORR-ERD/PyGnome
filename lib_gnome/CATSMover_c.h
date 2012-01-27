/*
 *  CATSMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover_c__
#define __CATSMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "CATSMover_b.h"
#include "CurrentMover_c.h"

class CATSMover_c : virtual public CATSMover_b, virtual public CurrentMover_c {

	public:

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	VelocityRec			GetSmoothVelocity (WorldPoint p);
	//OSErr				ComputeVelocityScale ();
	virtual OSErr		ComputeVelocityScale ();
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);

};

#endif
