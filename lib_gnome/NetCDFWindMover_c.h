/*
 *  NetCDFWindMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMover_c__
#define __NetCDFWindMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "NetCDFWindMover_b.h"
#include "WindMover_c.h"

class NetCDFWindMover_c : virtual public NetCDFWindMover_b, virtual public WindMover_c {

public:
	NetCDFWindMover_c (TMap *owner, char* name);
	NetCDFWindMover_c () {}
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, bool); // AH 04/16/12
	virtual void 		ModelStepIsDone();
	virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
	virtual long 		GetVelocityIndex(WorldPoint p);
	virtual LongPoint 		GetVelocityIndices(WorldPoint wp); /*{LongPoint lp = {-1,-1}; printError("GetVelocityIndices not defined for windmover"); return lp;}*/
	Seconds 			GetTimeValue(long index);
	//virtual OSErr		GetStartTime(Seconds *startTime);
	//virtual OSErr		GetEndTime(Seconds *endTime);
	virtual OSErr		GetStartTime(Seconds *startTime);
	virtual OSErr		GetEndTime(Seconds *endTime);
	virtual double 	GetStartUVelocity(long index);
	virtual double 	GetStartVVelocity(long index);
	virtual double 	GetEndUVelocity(long index);
	virtual double 	GetEndVVelocity(long index);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVER) return TRUE; return WindMover_c::IAm(id); }

};

#endif
