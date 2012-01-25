/*
 *  WindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __WindMover_c__
#define __WindMover_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "WindMover_b.h"
#include "Mover/Mover_c.h"

#ifdef pyGNOME
#define TOSSMTimeValue OSSMTimeValue_c
#endif

class TOSSMTimeValue;

class WindMover_c : virtual public WindMover_b, virtual public Mover_c {

public:
	
	virtual OSErr		AllocateUncertainty ();
	virtual void		DisposeUncertainty ();
	virtual OSErr		AddUncertainty(long setIndex,long leIndex,VelocityRec *v);
	virtual void 		UpdateUncertaintyValues(Seconds elapsedTime);
	virtual OSErr		UpdateUncertainty(void);

	virtual OSErr 		PrepareForModelStep();
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	void				SetTimeDep (TOSSMTimeValue *newTimeDep) { timeDep = newTimeDep; }
	TOSSMTimeValue		*GetTimeDep () { return (timeDep); }
	void				DeleteTimeDep ();
	void				ClearWindValues (); 
	void				SetIsConstantWind (Boolean isConstantWind) { fIsConstantWind = isConstantWind; }
	OSErr				GetTimeValue(Seconds time, VelocityRec *value);
	OSErr				CheckStartTime(Seconds time);

	
};

#ifdef pyGNOME
#undef TOSSMTimeValue
#endif
#endif