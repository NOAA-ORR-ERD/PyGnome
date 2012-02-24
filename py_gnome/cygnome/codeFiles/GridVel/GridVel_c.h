/*
 *  GridVel_c.h
 *  c_gnome
 *
 *  Created by Alex Hadjilambris on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridVel_c__
#define __GridVel_c__

#include "Earl.h"
#include "TypeDefs.h"

extern WorldRect emptyWorldRect;

class GridVel_c { 
	
protected:
	WorldRect fGridBounds;

public:
	GridVel_c() { 	fGridBounds = emptyWorldRect; }
	virtual ClassID 	GetClassID 	() { return TYPE_GRIDVEL; }
	virtual  VelocityRec GetPatValue(WorldPoint p)=0;
	virtual VelocityRec GetSmoothVelocity(WorldPoint p)=0;
	virtual void SetBounds(WorldRect bounds){fGridBounds = bounds;}	
	virtual WorldRect GetBounds(){return fGridBounds;}	
	virtual InterpolationVal GetInterpolationValues(WorldPoint ref){InterpolationVal ival; memset(&ival,0,sizeof(ival)); return ival;}
	virtual void 	Dispose () {}

};

#endif