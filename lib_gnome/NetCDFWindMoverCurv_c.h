/*
 *  NetCDFWindMoverCurv_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMoverCurv_c__
#define __NetCDFWindMoverCurv_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "NetCDFWindMover_c.h"

class NetCDFWindMoverCurv_c : virtual public NetCDFWindMover_c {

public:
	LONGH fVerdatToNetCDFH;	// for curvilinear
	WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file

	NetCDFWindMoverCurv_c (TMap *owner, char *name);
	NetCDFWindMoverCurv_c () {}
	//virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVERCURV; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVERCURV) return TRUE; return NetCDFWindMover::IAm(id); }

	LongPointHdl		GetPointsHdl();
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	OSErr 				ReorderPoints(TMap **newMap, char* errmsg); 
	Seconds 				GetTimeValue(long index);
	virtual long 		GetVelocityIndex(WorldPoint wp);
	virtual LongPoint 		GetVelocityIndices(WorldPoint wp);
	OSErr 				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);
	

};


#endif
