/*
 *  NetCDFMoverCurv_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMoverCurv_c__
#define __NetCDFMoverCurv_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFMoverCurv_b.h"
#include "NetCDFMover_c.h"

class NetCDFMoverCurv_c : virtual public NetCDFMoverCurv_b, virtual public NetCDFMover_c {

public:
	LongPointHdl		GetPointsHdl();
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	long 				CheckSurroundingPoints(LONGH maskH, long row, long col) ;
	Boolean 			InteriorLandPoint(LONGH maskH, long row, long col); 
	Boolean 			ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) ;
	Boolean 			ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) ;
	void 				ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin);
	OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long *numIslands);
	OSErr 				ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	OSErr 				ReorderPointsNoMask(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	OSErr 				ReorderPointsNoMask2(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	//double GetTopDepth(long depthIndex, double totalDepth);
	//double GetBottomDepth(long depthIndex, double totalDepth);
	
	virtual long 		GetVelocityIndex(WorldPoint wp);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
	OSErr 				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);
	virtual long 		GetNumDepthLevels();
	virtual OSErr 		GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH);
	void 				GetDepthIndices(long ptIndex, float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2);
	float		GetTotalDepthFromTriIndex(long triIndex);
	float		GetTotalDepth(WorldPoint refPoint,long ptIndex);

	

};

#endif
