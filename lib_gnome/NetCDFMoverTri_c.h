/*
 *  NetCDFMoverTri_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/2/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMoverTri_c__
#define __NetCDFMoverTri_c__

#include "Earl.h"
#include "NetCDFMoverTri_b.h"
#include "NetCDFMoverCurv_c.h"


class NetCDFMoverTri_c : virtual public NetCDFMoverTri_b, virtual public NetCDFMoverCurv_c {

public:
	LongPointHdl			GetPointsHdl();
	Boolean 				VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	VelocityRec				GetMove3D(InterpolationVal interpolationVal,float depth);
	void					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	//OSErr 				ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts); 
	OSErr					ReorderPoints(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts); 
	OSErr					ReorderPoints2(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri);
	
	virtual long			GetNumDepthLevels();
	virtual OSErr			GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH) {*profilesH=nil; return 0;}
	float					GetTotalDepth(WorldPoint refPoint, long triNum);
	

};

#endif
