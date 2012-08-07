/*
 *  Random3D_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Random3D_c__
#define __Random3D_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "Random_c.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

class Random3D_c : virtual public Random_c {

public:
	double fVerticalDiffusionCoefficient; //cm**2/s
	double fHorizontalDiffusionCoefficient; //cm**2/s
	double fVerticalBottomDiffusionCoefficient; //cm**2/s
	Boolean bUseDepthDependentDiffusion;
	//double fDiffusionCoefficient; //cm**2/s
	//TR_OPTIMZE fOptimize; // this does not need to be saved to the save file
	//double fUncertaintyFactor;		// multiplicative factor applied when uncertainty is on
	
	Random3D_c (TMap *owner, char *name);
	Random3D_c () {}
	virtual OSErr 		PrepareForModelRun(); 
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool); // AH 07/10/2012
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);

};


#undef TMap
#endif
