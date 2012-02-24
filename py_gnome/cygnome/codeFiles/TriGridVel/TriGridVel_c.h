/*
 *  TriGridVel_c.h
 *  c_gnome
 *
 *  Created by Alex Hadjilambris on 2/22/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TriGridVel_c__
#define __TriGridVel_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "GridVel/GridVel_c.h"
#include "DagTree/DagTree.h"

class TriGridVel_c : virtual public GridVel_c {

protected:
	FLOATH fBathymetryH;
	TDagTree *fDagTree;
	
public:
	TriGridVel_c() {fDagTree = 0; fBathymetryH=0;}
	virtual ClassID 	GetClassID 	() { return TYPE_TRIGRIDVEL; }
	void SetDagTree(TDagTree *dagTree){fDagTree=dagTree;}
	TDagTree*  GetDagTree(){return fDagTree;}
	LongPointHdl GetPointsHdl(void);
	TopologyHdl GetTopologyHdl(void);
	//DAGHdl GetDagTreeHdl(void);
	virtual long GetNumTriangles(void);
	void SetBathymetry(FLOATH depthsH){fBathymetryH=depthsH;}
	FLOATH  GetBathymetry(){return fBathymetryH;}
	VelocityRec GetPatValue(WorldPoint p);
	VelocityRec GetSmoothVelocity(WorldPoint p);
	virtual InterpolationVal GetInterpolationValues(WorldPoint refPoint);
	virtual	long GetRectIndexFromTriIndex(WorldPoint refPoint, LONGH ptrVerdatToNetCDFH, long numCols_ext);
	virtual	long GetRectIndexFromTriIndex2(long triIndex, LONGH ptrVerdatToNetCDFH, long numCols_ext);
	virtual LongPoint GetRectIndicesFromTriIndex(WorldPoint refPoint,LONGH ptrVerdatToNetCDFH,long numCols_ext);
	OSErr	GetRectCornersFromTriIndexOrPoint(long *index1, long *index2, long *index3, long *index4, WorldPoint refPoint,long triNum, Boolean useTriNum, LONGH ptrVerdatToNetCDFH,long numCols_ext);
	virtual void 		Dispose ();
	
};


#endif