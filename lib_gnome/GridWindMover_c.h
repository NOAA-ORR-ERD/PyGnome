/*
 *  GridWindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridWindMover_c__
#define __GridWindMover_c__

#include "Basics.h"
#include "GridWindMover_b.h"
#include "WindMover_c.h"

class GridWindMover_c : virtual public GridWindMover_b, virtual public WindMover_c {

public:
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual long 		GetVelocityIndex(WorldPoint p);

	
};

#endif
