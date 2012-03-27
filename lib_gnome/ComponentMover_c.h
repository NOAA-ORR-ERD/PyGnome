/*
 *  ComponentMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ComponentMover_c__
#define __ComponentMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "ComponentMover_b.h"
#include "CurrentMover_c.h"

class ComponentMover_c : virtual public ComponentMover_b, virtual public CurrentMover_c {

public:
	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	OSErr				SetOptimizeVariables (char *errmsg);
	OSErr				CalculateAveragedWindsHdl(char *errmsg);
	OSErr				GetAveragedWindValue(Seconds time, VelocityRec *avValue);
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep);

	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual	Boolean 		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);	
	
};

#endif
