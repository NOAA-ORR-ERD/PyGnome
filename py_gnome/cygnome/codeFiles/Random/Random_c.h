/*
 *  Random_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Random_c__
#define __Random_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "Mover/Mover_c.h"

class Random_c : virtual public Mover_c {
	
public:
	double fDiffusionCoefficient; //cm**2/s
	TR_OPTIMZE fOptimize; // this does not need to be saved to the save file
	double fUncertaintyFactor;		// multiplicative factor applied when uncertainty is on
	Boolean bUseDepthDependent;
	
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual ClassID 	GetClassID () { return TYPE_RANDOMMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMMOVER) return TRUE; return Mover_c::IAm(id); }	
	
};

#endif