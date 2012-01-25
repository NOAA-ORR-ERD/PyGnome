/*
 *  ADCPMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPMover_c__
#define __ADCPMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "ADCPMover_b.h"
#include "CurrentMover/CurrentMover_c.h"
#include "ADCPTimeValue/ADCPTimeValue.h"

class ADCPMover_c : virtual public ADCPMover_b, virtual public CurrentMover_c {

public:
	OSErr				AddTimeDep(ADCPTimeValue *theTimeDep, short where);
	OSErr				DropTimeDep(ADCPTimeValue *theTimeDep);
	ADCPTimeValue *		AddADCP(OSErr *err);
	
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	
	//void				SetTimeDep (ADCPTimeValue *newTimeDep) { timeDep = newTimeDep; }
	//ADCPTimeValue		*GetTimeDep () { return (timeDep); }
	//void				DeleteTimeDep ();
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);
	VelocityRec			GetVelocityAtPoint(WorldPoint3D p);
	//OSErr				ComputeVelocityScale ();
	virtual OSErr		ComputeVelocityScale ();
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);


};

#endif
