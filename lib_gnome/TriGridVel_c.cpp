/*
 *  TriGridVel_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriGridVel_c.h"
#include "RectUtils.h"
#include "MemUtils.h"
using std::cout;

/*GridVel_c::GridVel_c() 
{
	WorldRect emptyRect = { 0, 0, 0, 0 };
	fGridBounds = emptyRect;
}*/

LongPointHdl TriGridVel_c::GetPointsHdl(void)
{
	if(!fDagTree) return nil;
	
	return fDagTree->GetPointsHdl();
}

TopologyHdl TriGridVel_c::GetTopologyHdl(void)
{
	if(!fDagTree) return nil;
	
	return fDagTree->GetTopologyHdl();
}

/*DAGHdl TriGridVel_c::GetDagTreeHdl(void)
 {
 if(!fDagTree) return nil;
 
 return fDagTree->GetDagTreeHdl();
 }*/

long TriGridVel_c::GetNumTriangles(void)
{
	long numTriangles = 0;
	TopologyHdl topoH = fDagTree->GetTopologyHdl();
	if (topoH) numTriangles = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
	
	return numTriangles;
}

long TriGridVel_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fBathymetryH) numDepths = _GetHandleSize((Handle)fBathymetryH)/sizeof(**fBathymetryH);
	
	return numDepths;
}

InterpolationVal TriGridVel_c::GetInterpolationValues(WorldPoint refPoint)
{
	InterpolationVal interpolationVal;
	LongPoint lp;
	long ntri;
	ExPoint vertex1,vertex2,vertex3;
	double denom,refLon,refLat;
	double num1,num2,num3;
	
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	
	memset(&interpolationVal,0,sizeof(interpolationVal));
	
	if(!fDagTree) return interpolationVal;
	
	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		interpolationVal.ptIndex1 = ntri; // flag it
		return interpolationVal;
	}
	
	refLon = lp.h/1000000.;
	refLat = lp.v/1000000.;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return interpolationVal;
	
	// get the index into the pts handle for each vertex
	
	interpolationVal.ptIndex1 = (*topH)[ntri].vertex1;
	interpolationVal.ptIndex2 = (*topH)[ntri].vertex2;
	interpolationVal.ptIndex3 = (*topH)[ntri].vertex3;
	
	// get the vertices from fPtsH and figure out the interpolation coefficients
	
	vertex1.h = (*ptsH)[interpolationVal.ptIndex1].h/1000000.;
	vertex1.v = (*ptsH)[interpolationVal.ptIndex1].v/1000000.;
	vertex2.h = (*ptsH)[interpolationVal.ptIndex2].h/1000000.;
	vertex2.v = (*ptsH)[interpolationVal.ptIndex2].v/1000000.;
	vertex3.h = (*ptsH)[interpolationVal.ptIndex3].h/1000000.;
	vertex3.v = (*ptsH)[interpolationVal.ptIndex3].v/1000000.;
	
	
	// use a1*x1+a2*x2+a3*x3=x_ref, a1*y1+a2*y2+a3*y3=y_ref, and a1+a2+a3=1
	
	denom = (vertex3.v-vertex1.v)*(vertex2.h-vertex1.h)-(vertex3.h-vertex1.h)*(vertex2.v-vertex1.v);
	
	num1 = ((refLat-vertex3.v)*(vertex3.h-vertex2.h)-(refLon-vertex3.h)*(vertex3.v-vertex2.v));
	num2 = ((refLon-vertex1.h)*(vertex3.v-vertex1.v)-(refLat-vertex1.v)*(vertex3.h-vertex1.h));
	num3 = ((refLat-vertex1.v)*(vertex2.h-vertex1.h)-(refLon-vertex1.h)*(vertex2.v-vertex1.v));
	
	interpolationVal.alpha1 = num1/denom;
	interpolationVal.alpha2 = num2/denom;
	interpolationVal.alpha3 = num3/denom;
	
	return interpolationVal;
}

long TriGridVel_c::GetRectIndexFromTriIndex2(long triIndex,LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp;
	long i, n, ntri = triIndex, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ;
	double refLon,refLat;
	
	TopologyHdl topH ;
	
	if(!fDagTree) return -1;
	
	if (ntri < 0) return ntri;
	
	topH = fDagTree->GetTopologyHdl();
	
	if(!topH) return -1;
	
	// get the index into the pts handle for each vertex
	
	//ptIndex1 = (*topH)[ntri].vertex1;
	//ptIndex2 = (*topH)[ntri].vertex2;
	//ptIndex3 = (*topH)[ntri].vertex3;
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;
	
	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	return index;
}

long TriGridVel_c::GetRectIndexFromTriIndex(WorldPoint refPoint,LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp;
	long i, n, ntri, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ;
	double refLon,refLat;
	
	TopologyHdl topH ;
	
	if(!fDagTree) return -1;
	
	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		index = ntri; // flag it
		return index;
	}
	
	topH = fDagTree->GetTopologyHdl();
	
	if(!topH) return -1;
	
	// get the index into the pts handle for each vertex
	
	//ptIndex1 = (*topH)[ntri].vertex1;
	//ptIndex2 = (*topH)[ntri].vertex2;
	//ptIndex3 = (*topH)[ntri].vertex3;
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;
	
	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	return index;
}

OSErr TriGridVel_c::GetRectCornersFromTriIndexOrPoint(long *index1, long *index2, long *index3, long *index4, WorldPoint refPoint,long triNum, Boolean useTriNum, LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp;
	long i, n, ntri, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long debug_ptIndex1, debug_ptIndex2, debug_ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ, numCols = numCols_ext-1;
	double refLon,refLat;
	
	TopologyHdl topH ;
	
	if(!fDagTree) return -1;
	
	if (!useTriNum)
	{
		lp.h = refPoint.pLong;
		lp.v = refPoint.pLat;
		ntri = fDagTree->WhatTriAmIIn(lp);
	}
	else ntri = triNum;
	if (ntri < 0) 
	{
		index = ntri; // flag it
		return index;
	}
	
	topH = fDagTree->GetTopologyHdl();
	
	if(!topH) return -1;
	
	// get the index into the pts handle for each vertex
	
	debug_ptIndex1 = (*topH)[ntri].vertex1;
	debug_ptIndex2 = (*topH)[ntri].vertex2;
	debug_ptIndex3 = (*topH)[ntri].vertex3;
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;
	
	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	//index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	
	/**index1 = (largestI-1)*(numCols_ext-1)+smallestJ;
	 *index2 = (largestI)*(numCols_ext-1)+smallestJ;
	 *index3 = (largestI-1)*(numCols_ext-1)+smallestJ+1;
	 *index4 = (largestI)*(numCols_ext-1)+smallestJ+1;*/
	
	if (smallestJ>=numCols-1)
	{
		if (smallestJ==numCols)
			*index1=0;
	}
	if (largestI<=1) 
	{
		if (largestI==0) 
			*index2 = -1;
	}
	*index1 = (largestI-2)*(numCols_ext-1)+smallestJ;
	*index2 = (largestI-1)*(numCols_ext-1)+smallestJ;
	*index3 = (largestI-2)*(numCols_ext-1)+smallestJ+1;
	*index4 = (largestI-1)*(numCols_ext-1)+smallestJ+1;
	
	if (largestI==1) {*index1=-1; *index3=-1;}
	if (smallestJ==numCols-1) {*index3=-1;*index4=-1;}
	
	return 0;
}

LongPoint TriGridVel_c::GetRectIndicesFromTriIndex(WorldPoint refPoint,LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp={-1,-1}, indices;
	long i, n, ntri, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ;
	double refLon,refLat;
	
	TopologyHdl topH ;
	
	if(!fDagTree) return lp;
	
	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		index = ntri; // flag it
		return lp;
	}
	
	topH = fDagTree->GetTopologyHdl();
	
	if(!topH) return lp;
	
	// get the index into the pts handle for each vertex
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;
	
	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	//index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	indices.h = smallestJ;
	indices.v = largestI-1;
	return indices;
}

VelocityRec TriGridVel_c::GetSmoothVelocity(WorldPoint p)
{
	VelocityRec r;
	LongPoint lp;

	lp.h = p.pLong;
	lp.v = p.pLat;

	fDagTree->GetVelocity(lp,&r);

	return r;
}

VelocityRec TriGridVel_c::GetPatValue(WorldPoint p)
{
	VelocityRec r;
	LongPoint lp;

	lp.h = p.pLong;
	lp.v = p.pLat;

	fDagTree->GetVelocity(lp,&r);

	return r;
}

double TriGridVel_c::GetDepthAtPoint(WorldPoint p)
{
	double depthAtPoint = 0;
	long ptIndex1,ptIndex2,ptIndex3; 
	float depth1,depth2,depth3;
	InterpolationVal interpolationVal;

	interpolationVal = this->GetInterpolationValues(p);

	depth1 = (*fBathymetryH)[interpolationVal.ptIndex1];
	depth2 = (*fBathymetryH)[interpolationVal.ptIndex2];
	depth3 = (*fBathymetryH)[interpolationVal.ptIndex3];
	depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;

	return depthAtPoint;
}

void TriGridVel_c::Dispose ()
{
	if (fDagTree)
	{
		fDagTree->Dispose();
		delete fDagTree;
		fDagTree = nil;
	}
	if (fBathymetryH)
	{
		DisposeHandle((Handle)fBathymetryH);
		fBathymetryH = 0;
	}
	GridVel_c::Dispose();
}

