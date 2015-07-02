/*
 *  TriGridVel3D_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriGridVel3D_c.h"
#include "MemUtils.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

TriGridVel3D_c::TriGridVel3D_c() 
{
	fDepthsH=0; 
	fTriSelected = 0;
	fPtsSelected = 0;
	fOilConcHdl = 0;
	fMaxLayerDataHdl = 0;
	fTriAreaHdl = 0;
	fDosageHdl = 0;
	bShowSelectedTriangles = true;
	//fPercentileForMaxConcentration = .9;
	fPercentileForMaxConcentration = 1.;	// make user decide if they want to fudge this
	bCalculateDosage = false;
	bShowDosage = false;
	fDosageThreshold = .2;
	fMaxTri = -1;
	bShowMaxTri = false;
}


long TriGridVel3D_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

double TriGridVel3D_c::GetDepthAtPoint(WorldPoint p)
{
	double depthAtPoint = 0;
	long ptIndex1,ptIndex2,ptIndex3; 
	float depth1,depth2,depth3;
	InterpolationVal interpolationVal;

	interpolationVal = this->GetInterpolationValues(p);

	if (interpolationVal.ptIndex1 < 0) return depthAtPoint;
	
	if (!fDepthsH) 
	{
		//return depthAtPoint;	// see if depths are on the parent grid
		return TriGridVel_c::GetDepthAtPoint(p);
	}
	
	depth1 = (*fDepthsH)[interpolationVal.ptIndex1];
	depth2 = (*fDepthsH)[interpolationVal.ptIndex2];
	depth3 = (*fDepthsH)[interpolationVal.ptIndex3];
	depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;

	return depthAtPoint;
}

void TriGridVel3D_c::ScaleDepths(double scaleFactor)
{
	long i, numDepths;
	if (!fDepthsH) return;
	numDepths = GetNumDepths();
	for (i=0;i<numDepths;i++)
	{
		(*fDepthsH)[i] *= scaleFactor;
	}
	return;
}

long TriGridVel3D_c::GetNumOutputDataValues(void)
{
	long numOutputDataValues = 0;
	if (fOilConcHdl) numOutputDataValues = _GetHandleSize((Handle)fOilConcHdl)/sizeof(**fOilConcHdl);
	
	return numOutputDataValues;
}

/*long TriGridVel3D_c::GetNumTriangles(void)
 {
 long numTriangles = 0;
 TopologyHdl topoH = fDagTree->GetTopologyHdl();
 if (topoH) numTriangles = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
 
 return numTriangles;
 }*/

long TriGridVel3D_c::GetNumPoints(void)
{
	long numPts = 0;
	LongPointHdl ptsH ;
	
	ptsH = fDagTree->GetPointsHdl();
	if (ptsH) numPts = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
	
	return numPts;
}

OSErr TriGridVel3D_c::GetTriangleVertices(long i, long *x, long *y)
{
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	long ptIndex1, ptIndex2, ptIndex3;
	
	if(!fDagTree) return -1;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return -1;
	
	ptIndex1 = (*topH)[i].vertex1;
	ptIndex2 = (*topH)[i].vertex2;
	ptIndex3 = (*topH)[i].vertex3;
	
	x[0] = (*ptsH)[ptIndex1].h;
	y[0] = (*ptsH)[ptIndex1].v;
	x[1] = (*ptsH)[ptIndex2].h;
	y[1] = (*ptsH)[ptIndex2].v;
	x[2] = (*ptsH)[ptIndex3].h;
	y[2] = (*ptsH)[ptIndex3].v;
	
	return noErr;
}	

OSErr TriGridVel3D_c::GetTriangleVertices3D(long i, long *x, long *y, long *z)
{
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	long ptIndex1, ptIndex2, ptIndex3;
	
	if(!fDagTree) return -1;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return -1;
	
	ptIndex1 = (*topH)[i].vertex1;
	ptIndex2 = (*topH)[i].vertex2;
	ptIndex3 = (*topH)[i].vertex3;
	
	x[0] = (*ptsH)[ptIndex1].h;
	y[0] = (*ptsH)[ptIndex1].v;
	x[1] = (*ptsH)[ptIndex2].h;
	y[1] = (*ptsH)[ptIndex2].v;
	x[2] = (*ptsH)[ptIndex3].h;
	y[2] = (*ptsH)[ptIndex3].v;
	
	z[0] = (*fDepthsH)[ptIndex1];
	z[1] = (*fDepthsH)[ptIndex2];
	z[2] = (*fDepthsH)[ptIndex3];
	
	return noErr;
}	

double GetTriangleArea(WorldPoint pt1, WorldPoint pt2, WorldPoint pt3)
{
	double sideA, sideB, sideC, angle3;
	double cp, triArea;
	WorldPoint center1,center2;
	// flat earth or spherical earth?
	sideC = DistanceBetweenWorldPoints(pt1,pt2);	// kilometers
	sideB = DistanceBetweenWorldPoints(pt1,pt3);
	sideA = DistanceBetweenWorldPoints(pt2,pt3);
	
	center1.pLat = (pt1.pLat + pt2.pLat) / 2.;	// center of map or center of line?
	center1.pLong = (pt1.pLong + pt2.pLong) / 2.;
	center2.pLat = (pt1.pLat + pt3.pLat) / 2.;
	center2.pLong = (pt1.pLong + pt3.pLong) / 2.;
	cp = LongToDistance(pt2.pLong - pt1.pLong, center1) * LatToDistance(pt3.pLat - pt1.pLat) - LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt2.pLat - pt1.pLat);
	//angle3 = acos((sideA*sideA + sideB*sideB - sideC*sideC)/(2*sideA*sideB));
	//triArea = sin(angle3)*sideA*sideB/2.;
	triArea = fabs(cp)/2.;
	
	return triArea;
}

// combine with GetTriangleArea
double GetQuadArea(WorldPoint pt1, WorldPoint pt2, WorldPoint pt3, WorldPoint pt4)
{
	double cp, quadArea;
	WorldPoint center1,center2,center3;
	
	center1.pLat = (pt1.pLat + pt2.pLat) / 2.;
	center1.pLong = (pt1.pLong + pt2.pLong) / 2.;
	center2.pLat = (pt1.pLat + pt3.pLat) / 2.;
	center2.pLong = (pt1.pLong + pt3.pLong) / 2.;
	center3.pLat = (pt1.pLat + pt4.pLat) / 2.;
	center3.pLong = (pt1.pLong + pt4.pLong) / 2.;
	cp =  LongToDistance(pt2.pLong - pt1.pLong, center1) * LatToDistance(pt3.pLat - pt1.pLat) - LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt2.pLat - pt1.pLat)
	+ LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt4.pLat - pt1.pLat) - LongToDistance(pt4.pLong - pt1.pLong, center3) * LatToDistance(pt3.pLat - pt1.pLat);
	
	quadArea = fabs(cp)/2.;
	
	return quadArea;
}

int WorldPoint3DCompare(void const *x1, void const *x2)
{
	WorldPoint3D *p1,*p2;	
	p1 = (WorldPoint3D*)x1;
	p2 = (WorldPoint3D*)x2;
	
	if ((*p1).z < (*p2).z) 
		return -1;  // first less than second
	else if ((*p1).z > (*p2).z)
		return 1;
	else return 0;// equivalent	
}

void FindPolygonPoints(short polygonType, WorldPoint3D *pts, double upperDepth, double lowerDepth, double *midPtArea, double *bottomArea)
{
	long k;
	double offset, dist, len;
	WorldPoint3D ptOnT1B3,ptOnT3B3,ptOnB2B3,ptOnT1B2,ptOnT2B2,center1;
	WorldPoint3D T1,T2,T3,B1,B2,B3;
	double h = lowerDepth -  upperDepth;
	
	T1 = pts[0]; T2 = pts[1]; T3 = pts[2];
	B1 = pts[0]; B2 = pts[1]; B3 = pts[2];
	T2.z = T1.z; T3.z = T1.z;
	
	*bottomArea = 0;
	*midPtArea = 0;
	for (k = 0; k<2; k++)
	{
		if (k==0) offset = h/2;
		else offset = 0;
		if (k==1 && lowerDepth == B3.z) break;
		dist = lowerDepth - offset - T1.z;
		len = B3.z - T1.z;
		ptOnT1B3.z = lowerDepth - offset;
		ptOnT1B3.p.pLat = T1.p.pLat + dist/len * (B3.p.pLat - T1.p.pLat);
		ptOnT1B3.p.pLong = T1.p.pLong + dist/len * (B3.p.pLong - T1.p.pLong);
		dist = lowerDepth - offset - T3.z;
		len = B3.z - T3.z;
		//  here lat/lon same at both points
		ptOnT3B3.p.pLat = T3.p.pLat;
		ptOnT3B3.p.pLong = T3.p.pLong;
		ptOnT3B3.z =  lowerDepth - offset;
		
		if (polygonType==0)
		{	// triangle
			dist = lowerDepth - offset - B2.z;
			len = B3.z - B2.z;
			//ptOnB2B3.p.pLat = B2.p.pLat + DistanceToLat(dist/len * LatToDistance(B3.p.pLat - B2.p.pLat, center),center);
			ptOnB2B3.p.pLat = B2.p.pLat + dist/len * (B3.p.pLat - B2.p.pLat);
			ptOnB2B3.p.pLong = B2.p.pLong + dist/len * (B3.p.pLong - B2.p.pLong);
			ptOnB2B3.z = lowerDepth - offset;
			if (k==1) *bottomArea = GetTriangleArea(ptOnT1B3.p,ptOnT3B3.p,ptOnB2B3.p);
			if (k==0) *midPtArea = GetTriangleArea(ptOnT1B3.p,ptOnT3B3.p,ptOnB2B3.p);
		}
		else
		{	// quadrilateral
			dist = lowerDepth - offset - T1.z;
			len = B2.z - T1.z;
			//ptOnT1B2.p.pLat = T1.p.pLat + DistanceToLat(dist/len * LatToDistance(B2.p.pLat - T1.p.pLat));
			ptOnT1B2.p.pLat = T1.p.pLat + dist/len * (B2.p.pLat - T1.p.pLat);
			//center1.pLong = (B2.p.pLong + T1.p.pLong) / 2.; center1.pLat = (B2.p.pLat + T1.p.pLat)/2.;
			//ptOnT1B2.p.pLong = T1.p.pLong + DistanceToLong(dist/len * LongToDistance(B2.p.pLong - T1.p.pLong,center1),center1);
			ptOnT1B2.p.pLong = T1.p.pLong + dist/len * (B2.p.pLong - T1.p.pLong);
			ptOnT1B2.z = lowerDepth - offset;
			dist = lowerDepth - offset - T2.z;
			len = B2.z - T2.z;
			// here lat/lon same at both points
			ptOnT2B2.p.pLat = T2.p.pLat;
			ptOnT2B2.p.pLong = T2.p.pLong;
			ptOnT2B2.z = lowerDepth - offset;
			if (k==0) *midPtArea = GetQuadArea(ptOnT1B2.p, ptOnT2B2.p, ptOnT3B3.p, ptOnT1B3.p);
			if (k==1) *bottomArea = GetQuadArea(ptOnT1B2.p, ptOnT2B2.p, ptOnT3B3.p, ptOnT1B3.p);
		}
	}
	return;
}

OSErr TriGridVel3D_c::CalculateDepthSliceVolume(double *triVol, long triNum,float origUpperDepth, float origLowerDepth)
{
	double h, dist, len, debugTriVol;
	WorldPoint3D ptOnT1B2, ptOnT2B2, ptOnT1B3, ptOnT3B3, ptOnB2B3;
	long i,j,k, shallowIndex, midLevelIndex, deepIndex;
	double triArea, topTriArea, botTriArea, midTriArea, lastTriArea, offset = 0;
	if (triNum < 0) return -1;
	WorldPoint center1,center2;
	float upperDepth = origUpperDepth, lowerDepth = origLowerDepth; 
	OSErr err = 0;
	
	WorldPoint3D T1,T2,T3,B1,B2,B3, wp[3];
	
	err = GetTriangleVerticesWP3D(triNum, wp);
	qsort(wp,3,sizeof(WorldPoint3D),WorldPoint3DCompare);
	
	T1 = wp[0]; T2 = wp[1]; T3 = wp[2];
	B1 = wp[0]; B2 = wp[1]; B3 = wp[2];
	T2.z = T1.z; T3.z = T1.z;
	
	triArea = GetTriArea(triNum);	// kilometers
	lastTriArea = triArea;
	
	h = lowerDepth - upperDepth;	// usually 1 for depth profile
	if (h<=0) {*triVol = 0; return -1;}
	if (upperDepth > B3.z) {*triVol = 0; return noErr;}	// range is below bottom
	if (lowerDepth <= B1.z)	// shallowest depth
	{ 
		double theTriArea = GetTriangleArea(T1.p,T2.p,T3.p);
		*triVol = triArea * h * 1000 * 1000;	//	convert to meters 
		return noErr;
	}
	// need to deal with non-uniform volume once depth of shallowest vertex is reached
	else 
	{
		double firstPart = 0, secondPart = 0, thirdPart = 0;
		topTriArea = lastTriArea;
		if (lowerDepth <= B2.z)
		{
			// here bottom shape will be quadrilateral
			// get points where depth line intersects prism
			// get points where half depth line intersects prism - check if upperDepth > B1.z, otherwise need pieces
			// also check if j - B1.z < 0, then part of the region is above, ...
			if (upperDepth < B1.z)
			{
				firstPart = triArea * (B1.z - upperDepth);
				h = lowerDepth - B1.z;
			}
			FindPolygonPoints(1, wp, upperDepth, lowerDepth, &midTriArea, &botTriArea);
			secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
		}
		//else if ((j+1) <= B3.z)
		else	// bottom depth below second depth
		{	// special cases first
			if (lowerDepth > B3.z) 
			{
				// don't go below bottom
				h = B3.z - upperDepth;
				lowerDepth = B3.z;
			}
			if (upperDepth < B2.z)	// check B2 == B3 too
			{
				if (B2.z == B1.z)
				{
					firstPart = triArea * (B1.z - upperDepth);
					h = lowerDepth - B1.z;	
					upperDepth = B1.z;
					// fall to triArea calculation
				}
				else if (B2.z == B3.z)
				{
					if (upperDepth < B1.z)
					{
						firstPart = triArea * (B1.z - upperDepth);
						h = lowerDepth - B1.z;	// lower depth must be B2.z = B3.z
					}
					else
					{
						firstPart = 0;
						h = lowerDepth - upperDepth;
					}
					FindPolygonPoints(1, wp, upperDepth, lowerDepth, &midTriArea, &botTriArea);
					secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
					// no third part
				}
				else
				{
					// calculate the quad area, check if all points fall inside region
					if (upperDepth < B1.z)
					{	// three pieces
						firstPart = triArea * (B1.z - upperDepth);
						h = B2.z - B1.z;
						// calculate quad area
						FindPolygonPoints(1, wp, B1.z, B2.z, &midTriArea, &botTriArea);
						secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
						h = lowerDepth - B2.z;
						upperDepth = B2.z;
						topTriArea = botTriArea;
						// calculate tri area
					}
					else
					{
						h = B2.z - upperDepth;
						//lowerDepth = B2.z;
						FindPolygonPoints(1, wp, upperDepth, B2.z, &midTriArea, &botTriArea);
						secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
						// Calculate quad stuff
						// then calcuate tri stuff with
						topTriArea = botTriArea;
						h = lowerDepth - B2.z;
						upperDepth = B2.z;
					}
				}
			}
			if (B2.z != B3.z) {FindPolygonPoints(0, wp, upperDepth, B2.z, &midTriArea, &botTriArea);
				thirdPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);}
			
		}
		lastTriArea = botTriArea;
		//debugTriVol = (firstPart + h/6. * (topTriArea + 4.*midTriArea + botTriArea)) * 1000 * 1000;	
		//*triVol = (firstPart +  h/6. * (topTriArea + 4.*midTriArea + botTriArea)) * 1000 * 1000;	// convert to meters
		debugTriVol = (firstPart + secondPart + thirdPart) * 1000 * 1000;	
		*triVol = (firstPart + secondPart + thirdPart) * 1000 * 1000;	// convert to meters
	}
	
	return noErr;
}

OSErr TriGridVel3D_c::GetTriangleDepths(long i, float *z)
{
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	long ptIndex1, ptIndex2, ptIndex3;
	
	if(!fDagTree) return -1;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return -1;
	
	ptIndex1 = (*topH)[i].vertex1;
	ptIndex2 = (*topH)[i].vertex2;
	ptIndex3 = (*topH)[i].vertex3;
	
	z[0] = (*fDepthsH)[ptIndex1];
	z[1] = (*fDepthsH)[ptIndex2];
	z[2] = (*fDepthsH)[ptIndex3];
	
	return noErr;
}	

OSErr TriGridVel3D_c::GetMaxDepthForTriangle(long triNum, double *maxDepth)
{
	TopologyHdl topH ;
	long i, ptIndex[3];
	double z;
	
	if(!fDagTree) return -1;
	
	topH = fDagTree->GetTopologyHdl();
	
	if(!topH) return -1;
	
	*maxDepth = 0;
	ptIndex[0] = (*topH)[triNum].vertex1;
	ptIndex[1] = (*topH)[triNum].vertex2;
	ptIndex[2] = (*topH)[triNum].vertex3;
	
	for (i=0;i<3;i++)
	{
		z = (*fDepthsH)[ptIndex[i]];
		if (z > *maxDepth) *maxDepth = z;
	}	
	
	return noErr;
}	

OSErr TriGridVel3D_c::GetTriangleCentroidWC(long trinum, WorldPoint *p)
{	
	long x[3],y[3];
	OSErr err = GetTriangleVertices(trinum,x,y);
	p->pLat = (y[0]+y[1]+y[2])/3;
	p->pLong =(x[0]+x[1]+x[2])/3;
	return err;
}

double TriGridVel3D_c::GetTriArea(long triNum)
{
	WorldPoint pt1,pt2,pt3,center1,center2;
	long ptIndex1, ptIndex2, ptIndex3;
	//double sideA, sideB, sideC, angle3;
	double cp, triArea;
	
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	
	if(!fDagTree) return -1;
	
	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	
	if(!topH || !ptsH) return -1;
	if (triNum < 0) return -1;
	
	// get the index into the pts handle for each vertex
	ptIndex1 = (*topH)[triNum].vertex1;
	ptIndex2 = (*topH)[triNum].vertex2;
	ptIndex3 = (*topH)[triNum].vertex3;
	
	// get the vertices from the points handle
	
	pt1.pLong = (*ptsH)[ptIndex1].h;
	pt1.pLat = (*ptsH)[ptIndex1].v;
	pt2.pLong = (*ptsH)[ptIndex2].h;
	pt2.pLat = (*ptsH)[ptIndex2].v;
	pt3.pLong = (*ptsH)[ptIndex3].h;
	pt3.pLat = (*ptsH)[ptIndex3].v;
	
	center1.pLong = (pt2.pLong+pt1.pLong) / 2;
	center1.pLat = (pt2.pLat+pt1.pLat) / 2;
	center2.pLong = (pt3.pLong+pt1.pLong) / 2;
	center2.pLat = (pt3.pLat+pt1.pLat) / 2;
	//sideC = DistanceBetweenWorldPoints(pt1,pt2);	// kilometers
	//sideB = DistanceBetweenWorldPoints(pt1,pt3);
	//sideA = DistanceBetweenWorldPoints(pt2,pt3);
	
	cp = LongToDistance(pt2.pLong - pt1.pLong, center1) * LatToDistance(pt3.pLat - pt1.pLat) - LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt2.pLat - pt1.pLat);
	//angle3 = acos((sideA*sideA + sideB*sideB - sideC*sideC)/(2.*sideA*sideB));
	//triArea = sin(angle3)*sideA*sideB/2.;
	triArea = fabs(cp)/2.;
	return triArea;
}

double **TriGridVel3D_c::GetDosageHdl(Boolean initHdl)
{
	if (fDosageHdl) return fDosageHdl;
	else if (initHdl)
	{
		long i;
		long ntri = GetNumTriangles();
		fDosageHdl =(double **)_NewHandle(sizeof(double)*ntri);
		if(fDosageHdl)
		{
			for(i=0; i < ntri; i++)
			{
				(*fDosageHdl)[i] = 0.;
			}
			return fDosageHdl;
		}
		else {printError("Not enough memory to create dosage handle"); return nil;}
	}
	return nil;
	
}


double TriGridVel3D_c::GetMaxAtPreviousTimeStep(Seconds time)
{
	long sizeOfHdl;
	float prevMax = -1;
	OSErr err = 0;
	outputData data;
	if(!fOilConcHdl)
	{
		return -1;
	}
	sizeOfHdl = _GetHandleSize((Handle)fOilConcHdl)/sizeof(outputData);
	if (sizeOfHdl>0) data = (*fOilConcHdl)[sizeOfHdl-1];
	if (sizeOfHdl>1 && time==(*fOilConcHdl)[sizeOfHdl-1].time)	
		prevMax = (*fOilConcHdl)[sizeOfHdl-2].maxOilConcOverSelectedTri;
	return prevMax;
}


void TriGridVel3D_c::AddToOutputHdl(double avConcOverSelectedTriangles, double maxConcOverSelectedTriangles, Seconds time)
{
	long sizeOfHdl;
	OSErr err = 0;
	if(!fOilConcHdl)
	{
		fOilConcHdl = (outputDataHdl)_NewHandle(0);
		if(!fOilConcHdl) {TechError("TTriGridVel3D::AddToOutputHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fOilConcHdl)/sizeof(outputData);
	//if (sizeOfHdl>0 && time==(*fOilConcHdl)[sizeOfHdl-1].time) return;	// code goes here, check all times
	_SetHandleSize((Handle) fOilConcHdl, (sizeOfHdl+1)*sizeof(outputData));
	if (_MemError()) { TechError("TTriGridVel3D::AddToOutputHdl()", "_SetHandleSize()", 0); return; }
	//(*fOilConcHdl)[sizeOfHdl].oilConcAtSelectedTri = concentrationInSelectedTriangles;	// should add to old value??			
	(*fOilConcHdl)[sizeOfHdl].avOilConcOverSelectedTri = avConcOverSelectedTriangles;	// should add to old value??			
	(*fOilConcHdl)[sizeOfHdl].maxOilConcOverSelectedTri = maxConcOverSelectedTriangles;	// should add to old value??			
	(*fOilConcHdl)[sizeOfHdl].time = time;				
}

/*void TriGridVel3D_c::AddToMaxLayerHdl(long maxLayer, long maxTri, Seconds time)
 {	// want top/bottom ? 
 long sizeOfHdl;
 OSErr err = 0;
 if(!fMaxLayerDataHdl)
 {
 fMaxLayerDataHdl = (maxLayerDataHdl)_NewHandle(0);
 if(!fMaxLayerDataHdl) {TechError("TTriGridVel3D::AddToMaxLayerHdl()", "_NewHandle()", 0); err = memFullErr; return;}
 }
 sizeOfHdl = _GetHandleSize((Handle)fMaxLayerDataHdl)/sizeof(maxLayerData);
 if (sizeOfHdl>0 && time==(*fMaxLayerDataHdl)[sizeOfHdl-1].time) return;	// code goes here, check all times
 _SetHandleSize((Handle) fMaxLayerDataHdl, (sizeOfHdl+1)*sizeof(maxLayerData));
 if (_MemError()) { TechError("TTriGridVel3D::AddToMaxLayerHdl()", "_SetHandleSize()", 0); return; }
 (*fMaxLayerDataHdl)[sizeOfHdl].maxLayer = maxLayer;	// should add to old value??			
 (*fMaxLayerDataHdl)[sizeOfHdl].maxTri = maxTri;	// should add to old value??			
 (*fMaxLayerDataHdl)[sizeOfHdl].time = time;				
 }
 Boolean TriGridVel3D_c::GetMaxLayerInfo(long *maxLayer, long *maxTri, Seconds time)
 {
 long i, sizeOfHdl = _GetHandleSize((Handle)fMaxLayerDataHdl)/sizeof(maxLayerData);
 *maxLayer = -1; *maxTri = -1;
 if (time > (*fMaxLayerDataHdl)[sizeOfHdl-1].time) return false;	// will need to calculate the info
 
 for (i=0;i<sizeOfHdl;i++)
 {
 if (time==(*fMaxLayerDataHdl)[i].time)
 {
 *maxLayer = (*fMaxLayerDataHdl)[i].maxLayer;
 *maxTri = (*fMaxLayerDataHdl)[i].maxTri;
 return true;
 }
 }
 return false;	// error message? if time step has changed either need to rerun
 }
 */
void TriGridVel3D_c::AddToTriAreaHdl(double *triAreaArray, long numValues)
{
	long i,sizeOfHdl;
	OSErr err = 0;
	if(!fTriAreaHdl)
	{
		fTriAreaHdl = (double**)_NewHandle(0);
		if(!fTriAreaHdl) {TechError("TTriGridVel3D::AddToTriAreaHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fTriAreaHdl)/sizeof(double);
	//if (sizeOfHdl>0 && time==(*fTriAreaHdl)[sizeOfHdl-1].time) return;	// code goes here, check all times
	_SetHandleSize((Handle) fTriAreaHdl, (sizeOfHdl+numValues)*sizeof(double));
	if (_MemError()) { TechError("TTriGridVel3D::AddToTriAreaHdl()", "_SetHandleSize()", 0); return; }
	for (i=0;i<numValues;i++)
	{
		(*fTriAreaHdl)[sizeOfHdl+i] = triAreaArray[i];				
	}
}

void TriGridVel3D_c::ClearOutputHandles()
{
	if(fOilConcHdl) 
	{
		DisposeHandle((Handle)fOilConcHdl); 
		fOilConcHdl = 0;
	}
	if(fTriAreaHdl) 
	{
		DisposeHandle((Handle)fTriAreaHdl); 
		fTriAreaHdl = 0;
	}
	if(fDosageHdl) 
	{
		DisposeHandle((Handle)fDosageHdl); 
		fDosageHdl = 0;
	}
	if(fMaxLayerDataHdl)
	{
		DisposeHandle((Handle)fMaxLayerDataHdl); 
		fMaxLayerDataHdl = 0;
	}
}


void TriGridVel3D_c::GetTriangleVerticesWP(long i, WorldPoint *w)
{
	TopologyHdl topH;
	LongPointHdl ptsH;
	topH = fDagTree->GetTopologyHdl();	
	ptsH = fDagTree->GetPointsHdl();
	if(!topH || !ptsH) return;
	w[0].pLong = (*ptsH)[(*topH)[i].vertex1].h;
	w[0].pLat = (*ptsH)[(*topH)[i].vertex1].v;
	w[1].pLong = (*ptsH)[(*topH)[i].vertex2].h;
	w[1].pLat = (*ptsH)[(*topH)[i].vertex2].v;
	w[2].pLong = (*ptsH)[(*topH)[i].vertex3].h;
	w[2].pLat = (*ptsH)[(*topH)[i].vertex3].v;
	return;
}

OSErr TriGridVel3D_c::GetTriangleVerticesWP3D(long i, WorldPoint3D *w)
{
	TopologyHdl topH;
	LongPointHdl ptsH;
	topH = fDagTree->GetTopologyHdl();	
	ptsH = fDagTree->GetPointsHdl();
	if(!topH || !ptsH) return -1;
	w[0].p.pLong = (*ptsH)[(*topH)[i].vertex1].h;
	w[0].p.pLat = (*ptsH)[(*topH)[i].vertex1].v;
	w[1].p.pLong = (*ptsH)[(*topH)[i].vertex2].h;
	w[1].p.pLat = (*ptsH)[(*topH)[i].vertex2].v;
	w[2].p.pLong = (*ptsH)[(*topH)[i].vertex3].h;
	w[2].p.pLat = (*ptsH)[(*topH)[i].vertex3].v;
	w[0].z = (*fDepthsH)[(*topH)[i].vertex1];
	w[1].z = (*fDepthsH)[(*topH)[i].vertex2];
	w[2].z = (*fDepthsH)[(*topH)[i].vertex3];
	return 0;
}

Boolean **TriGridVel3D_c::GetTriSelection(Boolean initHdl) 
{
	if (fTriSelected)	
		return fTriSelected;
	else if (initHdl)
	{
		long i;
		long ntri = GetNumTriangles();
		fTriSelected =(Boolean **)_NewHandle(sizeof(Boolean)*ntri);
		if(fTriSelected)
		{
			for(i=0; i < ntri; i++)
			{
				(*fTriSelected)[i] = false;
			}
			return fTriSelected;
		}
	}
	return nil;
}


void TriGridVel3D_c::Dispose ()
{
	if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
	//if(fDepthContoursH) {DisposeHandle((Handle)fDepthContoursH); fDepthContoursH=0;}	// moved to TriGridVel
	if(fTriSelected) {DisposeHandle((Handle)fTriSelected); fTriSelected=0;}
	if(fPtsSelected) {DisposeHandle((Handle)fPtsSelected); fPtsSelected=0;}
	if(fOilConcHdl) {DisposeHandle((Handle)fOilConcHdl); fOilConcHdl=0;}
	if(fMaxLayerDataHdl) {DisposeHandle((Handle)fMaxLayerDataHdl); fMaxLayerDataHdl=0;}
	if(fTriAreaHdl) {DisposeHandle((Handle)fTriAreaHdl); fTriAreaHdl=0;}
	if(fDosageHdl) {DisposeHandle((Handle)fDosageHdl); fDosageHdl=0;}
	
	TriGridVel_c::Dispose ();
}