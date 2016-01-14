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
#include "Replacements.h"
using std::cout;

GridVel_c::GridVel_c() 
{
	WorldRect emptyRect = { 0, 0, 0, 0 };
	fGridBounds = emptyRect;
}

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

VelocityFH TriGridVel_c::GetVelocityHdl(void)
{
	if(!fDagTree) return nil;
	
	return fDagTree->GetVelocityHdl();
}

DAGHdl TriGridVel_c::GetDagTreeHdl(void)
 {
	 if(!fDagTree) return nil;
	 
	 return fDagTree->GetDagTreeHdl();
 }

long TriGridVel_c::GetNumTriangles(void)
{
	long numTriangles = 0;
	TopologyHdl topoH = fDagTree->GetTopologyHdl();
	if (topoH) numTriangles = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
	
	return numTriangles;
}

WORLDPOINTH TriGridVel_c::GetWorldPointsHdl(void)
{
	OSErr err = 0;
	int32_t numPts = 0, numTri = 0;
	WorldPoint wp;
	LongPoint lp;
	
	if (WPtH) return WPtH;
		
	LongPointHdl ptsH = GetPointsHdl();
	TopologyHdl topoH = GetTopologyHdl();
	if (ptsH) numPts = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	if (topoH) numTri = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
	
	WPtH = (WORLDPOINTH)_NewHandle(numPts * sizeof(WorldPoint));
	if (!WPtH) {
		err = -1;
		TechError("TriGridVel_c::GetWorldPointsHdl()", "_NewHandle()", 0);
		goto done;
	}
	for (int i=0; i<numPts; i++)
	{
		lp = (*ptsH)[i];
#ifndef pyGNOME
		wp.pLong = lp.h;
		wp.pLat = lp.v;
#else
		wp.pLong = (double)lp.h / 1.e6;
		wp.pLat = (double)lp.v / 1.e6;
#endif
		INDEXH(WPtH,i) = wp;
	}
done:
	return WPtH;
}

WORLDPOINTH	TriGridVel_c::GetCenterPointsHdl()
{
	OSErr err = 0;
	LongPointHdl ptsH = 0;
	WORLDPOINTH wpH = 0;
	TopologyHdl topH ;
	LongPoint wp1,wp2,wp3;
	WorldPoint wp;
	int32_t numPts = 0, numTri = 0;
	
	if (CenterPtsH) return CenterPtsH;
	
	topH = GetTopologyHdl();
	ptsH = GetPointsHdl();
	numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
	numPts = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
	CenterPtsH = (WORLDPOINTH)_NewHandle(numTri * sizeof(WorldPoint));
	if (!CenterPtsH) {
		err = -1;
		TechError("TriGridVel_c::GetCenterPointsHdl()", "_NewHandle()", 0);
		goto done;
	}
	
	for (int i=0; i<numTri; i++)
	{
		wp1 = (*ptsH)[(*topH)[i].vertex1];
		wp2 = (*ptsH)[(*topH)[i].vertex2];
		wp3 = (*ptsH)[(*topH)[i].vertex3];

#ifndef pyGNOME
		wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
		wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
#else
		wp.pLong = (double)(wp1.h+wp2.h+wp3.h)/3.e6;
		wp.pLat = (double)(wp1.v+wp2.v+wp3.v)/3.e6;
#endif
		INDEXH(CenterPtsH,i) = wp;
	}
	
done:
	return CenterPtsH;
}

long TriGridVel_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fBathymetryH) numDepths = _GetHandleSize((Handle)fBathymetryH)/sizeof(**fBathymetryH);
	
	return numDepths;
}

InterpolationValBilinear TriGridVel_c::GetBilinearInterpolationValues(WorldPoint refPoint)
{
	InterpolationValBilinear interpolationVal;
	LongPoint lp;
	long ntri, adj_tri;
	long ptIndex1,ptIndex2,ptIndex3,ptIndex4;
	ExPoint vertex1,vertex2,vertex3,vertex4;
	double largest_lat, smallest_lat, largest_lon, smallest_lon;
	double denom,refLon,refLat;
	double num1,num2,num3,num4;
	
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	
	memset(&interpolationVal,0,sizeof(interpolationVal));
	
	if(!fDagTree) return interpolationVal;
	// get four corners (or just diagonals - x1,y1, x2,y2)
	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		interpolationVal.ptIndex1 = ntri; // flag it
		return interpolationVal;
	}
	if ((ntri+1)%2==0) adj_tri = ntri-1; else adj_tri = ntri+1; //odd, even
	
	refLon = lp.h/1000000.;
	refLat = lp.v/1000000.;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return interpolationVal;
	
	// get the index into the pts handle for each vertex
	
	//ptIndex1 = (*topH)[ntri].vertex1;
	//ptIndex2 = (*topH)[ntri].vertex2;
	//ptIndex3 = (*topH)[ntri].vertex3;
	//ptIndex4 = (*topH)[adj_tri].vertex2;
	
	if (((ntri+1)%2==0))
	{
		ptIndex1 = (*topH)[ntri].vertex1;//bL
		ptIndex2 = (*topH)[ntri].vertex2;//bR
		ptIndex4 = (*topH)[ntri].vertex3;//tR
		ptIndex3 = (*topH)[adj_tri].vertex2;//tL
	}
	else
	{
		ptIndex4 = (*topH)[ntri].vertex1;//tR
		ptIndex3 = (*topH)[ntri].vertex2;//tL
		ptIndex1 = (*topH)[ntri].vertex3;//bL
		ptIndex2 = (*topH)[adj_tri].vertex2;//bR
	}
	// top left - tR, tL, bL - adj_tri = ntri+1
	// bottom right - bL, bR, tR - adj_tri = ntri-1
	
	//interpolationVal.ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	//interpolationVal.ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	//interpolationVal.ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//interpolationVal.ptIndex4 = INDEXH(ptrVerdatToNetCDFH,(*topH)[adj_tri].vertex2);
	interpolationVal.ptIndex1 = ptIndex1;
	interpolationVal.ptIndex2 = ptIndex2;
	interpolationVal.ptIndex3 = ptIndex3;
	interpolationVal.ptIndex4 = ptIndex4;
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	// get the vertices from fPtsH and figure out the interpolation coefficients
	// to get the points use the triangle indices, for the velocities use the grid indicies for now
	
	vertex1.h = (*ptsH)[ptIndex1].h/1000000.;
	vertex1.v = (*ptsH)[ptIndex1].v/1000000.;
	vertex2.h = (*ptsH)[ptIndex2].h/1000000.;
	vertex2.v = (*ptsH)[ptIndex2].v/1000000.;
	vertex3.h = (*ptsH)[ptIndex3].h/1000000.;
	vertex3.v = (*ptsH)[ptIndex3].v/1000000.;
	vertex4.h = (*ptsH)[ptIndex4].h/1000000.;
	vertex4.v = (*ptsH)[ptIndex4].v/1000000.;

	// code goes here, find largest and smallest lat and lon
	
	largest_lat = vertex1.v;
	smallest_lat = vertex1.v;
	if (vertex2.v > largest_lat)
		largest_lat = vertex2.v;
	if (vertex2.v < smallest_lat)
		smallest_lat = vertex2.v;
	if (vertex3.v > largest_lat)
		largest_lat = vertex3.v;
	if (vertex3.v < smallest_lat)
		smallest_lat = vertex3.v;
	if (vertex4.v > largest_lat)
		largest_lat = vertex4.v;
	if (vertex4.v < smallest_lat)
		smallest_lat = vertex4.v;
	
	largest_lon = vertex1.h;
	smallest_lon = vertex1.h;
	if (vertex2.h > largest_lon)
		largest_lon = vertex2.h;
	if (vertex2.h < smallest_lat)
		smallest_lon = vertex2.h;
	if (vertex3.h > largest_lon)
		largest_lon = vertex3.h;
	if (vertex3.h < smallest_lon)
		smallest_lon = vertex3.h;
	if (vertex4.h > largest_lon)
		largest_lon = vertex4.h;
	if (vertex4.h < smallest_lon)
		smallest_lon = vertex4.h;
	
	denom = (largest_lon-smallest_lon)*(largest_lat-smallest_lat);
	num1 = (largest_lon-refLon)*(largest_lat-refLat);
	num2 = (refLon-smallest_lon)*(largest_lat-refLat);
	num3 = (largest_lon-refLon)*(refLat-smallest_lat);
	num4 = (refLon-smallest_lon)*(refLat-smallest_lat);
	// just use 1 and 4 or 2 and 3, the diagonals
	// use bilinear interpolation
	/*denom = (vertex4.h-vertex1.h)*(vertex4.v-vertex1.v);
	// check this - make sure the pts match the alphas
	num1 = (vertex4.h-refLon)*(vertex4.v-refLat);
	num2 = (refLon-vertex1.h)*(vertex4.v-refLat);
	num3 = (vertex4.h-refLon)*(refLat-vertex1.v);
	num4 = (refLon-vertex1.h)*(refLat-vertex1.v);*/
	//(x2-x)(y2-y)q11 - lower left (pt3)
	//(x-x1)(y2-y)q21 - lower right (pt4)
	//(x2-x)(y-y1)q12 - upper left (pt2)
	//(x-x1)(y-y1)q22 - upper right (pt1)
	interpolationVal.alpha1 = num1/denom;
	interpolationVal.alpha2 = num2/denom;
	interpolationVal.alpha3 = num3/denom;
	interpolationVal.alpha4 = num4/denom;
	
	return interpolationVal;
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

InterpolationVal TriGridVel_c::GetInterpolationValuesFromIndex(long triNum)
{
	InterpolationVal interpolationVal;
	LongPoint lp;
	ExPoint vertex1,vertex2,vertex3;
	double denom,refLon,refLat;
	double num1,num2,num3;
	
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	
	memset(&interpolationVal,0,sizeof(interpolationVal));
	
	if(!fDagTree) return interpolationVal;
	
	if (triNum < 0) 
	{
		interpolationVal.ptIndex1 = triNum; // flag it
		return interpolationVal;
	}
	
	refLon = lp.h/1000000.;
	refLat = lp.v/1000000.;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return interpolationVal;
	
	// get the index into the pts handle for each vertex
	
	interpolationVal.ptIndex1 = (*topH)[triNum].vertex1;
	interpolationVal.ptIndex2 = (*topH)[triNum].vertex2;
	interpolationVal.ptIndex3 = (*topH)[triNum].vertex3;
	
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

	if (interpolationVal.ptIndex1 < 0) return depthAtPoint;

	if (!fBathymetryH) return depthAtPoint;
	
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
	if (CenterPtsH)
	{
		DisposeHandle((Handle)CenterPtsH);
		CenterPtsH = 0;
	}
	if (WPtH)
	{
		DisposeHandle((Handle)WPtH);
		WPtH = 0;
	}
	GridVel_c::Dispose();
}

