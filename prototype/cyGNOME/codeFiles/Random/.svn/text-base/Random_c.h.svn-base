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
#include "Random_b.h"
#include "Mover_c.h"

class Random_c : virtual public Random_b, virtual public Mover_c {
	
public:
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	
	
};

#endif