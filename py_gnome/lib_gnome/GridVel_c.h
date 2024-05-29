/*
 *  GridVel_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridVel_c__
#define __GridVel_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "DagTree.h"

struct InterpolationVal {	
	long ptIndex1;
	long ptIndex2;
	long ptIndex3;
	double alpha1;
	double alpha2;
	double alpha3;
};

struct InterpolationValBilinear {	
	long ptIndex1;
	long ptIndex2;
	long ptIndex3;
	long ptIndex4;
	double alpha1;
	double alpha2;
	double alpha3;
	double alpha4;
};

class GridVel_c {
	
protected:
	WorldRect fGridBounds;	
public:
	GridVel_c();
	virtual	~GridVel_c() { Dispose (); }
	//virtual ClassID 	GetClassID 	() { return TYPE_GRIDVEL; }
	virtual  VelocityRec GetPatValue(WorldPoint p)=0;
	virtual VelocityRec GetSmoothVelocity(WorldPoint p)=0;
	virtual void SetBounds(WorldRect bounds){fGridBounds = bounds;}	
	virtual WorldRect GetBounds(){return fGridBounds;}	
	virtual InterpolationValBilinear GetBilinearInterpolationValues(WorldPoint ref){InterpolationValBilinear ival; memset(&ival,0,sizeof(ival)); return ival;}
	virtual InterpolationVal GetInterpolationValues(WorldPoint ref){InterpolationVal ival; memset(&ival,0,sizeof(ival)); return ival;}
	virtual LongPointHdl GetPointsHdl(void){return 0;}
	virtual WORLDPOINTH GetWorldPointsHdl(void){return 0;}
	virtual TopologyHdl GetTopologyHdl(void){return 0;}
	virtual WORLDPOINTH	GetCenterPointsHdl(void){return 0;}
	virtual double GetDepthAtPoint(WorldPoint p){return 0;}
	virtual void	Dispose() { return; }
	
};


#endif