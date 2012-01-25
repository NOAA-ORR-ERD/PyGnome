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

#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFWindMoverCurv_b.h"
#include "NetCDFWindMover/NetCDFWindMover_c.h"

class NetCDFWindMoverCurv_c : virtual public NetCDFWindMoverCurv_b, virtual public NetCDFWindMover_c {

public:
	LongPointHdl		GetPointsHdl();
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	long 					CheckSurroundingPoints(LONGH maskH, long row, long col) ;
	Boolean 				InteriorLandPoint(LONGH maskH, long row, long col); 
	Boolean 				ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) ;
	Boolean 				ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) ;
	void 					ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin);
	OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long *numIslands);
	OSErr 				ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	Seconds 				GetTimeValue(long index);
	virtual long 		GetVelocityIndex(WorldPoint wp);
	virtual LongPoint 		GetVelocityIndices(WorldPoint wp);  /*{LongPoint lp = {-1,-1}; printError("GetVelocityIndices not defined for windmover"); return lp;}*/
	OSErr 				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);
	

};


#endif
