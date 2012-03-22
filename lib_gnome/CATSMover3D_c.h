/*
 *  CATSMover3D_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __CATSMover3D_c__
#define __CATSMover3D_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CATSMover3D_b.h"
#include "CATSMover_c.h"

class CATSMover3D_c : virtual public CATSMover3D_b, virtual public CATSMover_c {

	public:

	
	//virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	
	//virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
	//virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	
	/*void				SetRefPosition (WorldPoint p, long z) { refP = p; refZ = z; }
	 void				GetRefPosition (WorldPoint *p, long *z) { (*p) = refP; (*z) = refZ; }
	 
	 void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	 TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	 void				DeleteTimeDep ();
	 VelocityRec			GetPatValue (WorldPoint p);
	 VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	 VelocityRec			GetSmoothVelocity (WorldPoint p);
	 OSErr				ComputeVelocityScale ();
	 virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	 */		virtual OSErr 		PrepareForModelStep();
	//virtual void 		ModelStepIsDone();
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	
	//LongPointHdl 		GetPointsHdl(Boolean useRefinedGrid);
	virtual 			LongPointHdl 		GetPointsHdl();
	

};

#endif

