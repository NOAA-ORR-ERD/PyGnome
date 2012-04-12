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

#include "Basics.h"
#include "TypeDefs.h"
#include "WindMover_b.h"
#include "Mover_c.h"

#ifdef pyGNOME
#define TOSSMTimeValue OSSMTimeValue_c
#define TMap Map_c
#endif

class TOSSMTimeValue;
class TMap;
class WindMover_c : virtual public WindMover_b, virtual public Mover_c {

public:
	WindMover_c (TMap *owner, char* name);
	WindMover_c ();
	virtual ClassID 	GetClassID () { return TYPE_WINDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_WINDMOVER) return TRUE; return Mover_c::IAm(id); }
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

#undef TOSSMTimeValue
#undef TMap
#endif
