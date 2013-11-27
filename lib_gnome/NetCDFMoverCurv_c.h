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

#include "Basics.h"
#include "TypeDefs.h"
#include "NetCDFMover_c.h"
#include "GridMapUtils.h"

#ifndef pyGNOME
#include "TMap.h"
#else
#include "Map_c.h"
#define TMap Map_c
#endif

class NetCDFMoverCurv_c : virtual public NetCDFMover_c {

public:
	
	LONGH fVerdatToNetCDFH;	// for curvilinear
	WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file
	//LONGH fVerdatToNetCDFH_2;	// for curvilinear
	Boolean bIsCOOPSWaterMask;
	
	
	NetCDFMoverCurv_c (TMap *owner, char *name);
	NetCDFMoverCurv_c () {}
	//virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVERCURV; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVERCURV) return TRUE; return NetCDFMover::IAm(id); }

	LongPointHdl		GetPointsHdl();
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	/*long 				CheckSurroundingPoints(LONGH maskH, long numRows, long  numCols, long row, long col) ;
	Boolean 			InteriorLandPoint(LONGH maskH, long numRows, long  numCols, long row, long col); 
	//Boolean 			ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long numRows, long  numCols, long row, long col) ;
	Boolean 			ThereIsAdjacentLand2(LONGH maskH, DOUBLEH landmaskH, long numRows, long  numCols, long row, long col) ;
	Boolean 			ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long numRows, long  numCols, long row, long col) ;
	void 				ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin,long numRows,long numCols);
	//OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);
	OSErr 				NumberIslands(LONGH *islandNumberH, DOUBLEH landmaskH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);*/
	//OSErr 				ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	OSErr 				ReorderPoints(DOUBLEH landmaskH, TMap **newMap, char* errmsg); 
	//OSErr 				ReorderPointsNoMask(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	OSErr 				ReorderPointsNoMask(TMap **newMap, char* errmsg); 
	//OSErr 				ReorderPointsCOOPSMask(VelocityFH velocityH, TMap **newMap, char* errmsg); 
	OSErr 				ReorderPointsCOOPSMask(DOUBLEH landmaskH, TMap **newMap, char* errmsg); 
	Boolean				IsCOOPSFile();
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
	float		GetTotalDepth2(WorldPoint refPoint);

	

};
/*long 				CheckSurroundingPoints(LONGH maskH, long numRows, long  numCols, long row, long col) ;
Boolean 			InteriorLandPoint(LONGH maskH, long numRows, long  numCols, long row, long col); 
//Boolean 			ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long numRows, long  numCols, long row, long col) ;
Boolean 			ThereIsAdjacentLand2(LONGH maskH, DOUBLEH landmaskH, long numRows, long  numCols, long row, long col) ;
Boolean 			ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long numRows, long  numCols, long row, long col) ;
void 				ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin,long numRows,long numCols);
//OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);
OSErr 				NumberIslands(LONGH *islandNumberH, DOUBLEH landmaskH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);
*/
#endif
