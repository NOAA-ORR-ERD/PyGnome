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

#include "Earl.h"
#include "TypeDefs.h"
#include "Random3D_b.h"
#include "Random_c.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

class Random3D_c : virtual public Random3D_b, virtual public Random_c {

public:
	
	Random3D_c (TMap *owner, char *name);
	Random3D_c () {}
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);

};


#undef TMap
#endif
