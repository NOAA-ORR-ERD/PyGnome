/*
 *  Random3D_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Random3D_b__
#define __Random3D_b__

#include "Basics.h"
#include "TypeDefs.h"
#include "Random_b.h"

class Random3D_b : virtual public Random_b {

public:
	double fVerticalDiffusionCoefficient; //cm**2/s
	double fHorizontalDiffusionCoefficient; //cm**2/s
	double fVerticalBottomDiffusionCoefficient; //cm**2/s
	Boolean bUseDepthDependentDiffusion;
	//double fDiffusionCoefficient; //cm**2/s
	//TR_OPTIMZE fOptimize; // this does not need to be saved to the save file
	//double fUncertaintyFactor;		// multiplicative factor applied when uncertainty is on


};

#endif
