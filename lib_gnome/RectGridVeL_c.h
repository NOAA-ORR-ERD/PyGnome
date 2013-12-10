/*
 *  RectGridVel_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __RectGridVel_c__
#define __RectGridVel_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "GridVel_c.h"

class RectGridVel_c : virtual public GridVel_c {

protected:
	VelocityH 	fGridHdl;
	long 		fNumRows;
	long 		fNumCols;

public:
	RectGridVel_c();
	//virtual ClassID GetClassID 	() { return TYPE_RECTGRIDVEL; }
	virtual void 	SetBounds(WorldRect bounds);			
	long 			NumVelsInGridHdl(void);
	VelocityRec 	GetPatValue(WorldPoint p);
	VelocityRec 	GetSmoothVelocity(WorldPoint p);
	virtual void 	Dispose ();

	
};

#endif