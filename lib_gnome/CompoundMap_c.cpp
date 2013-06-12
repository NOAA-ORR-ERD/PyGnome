/*
 *  CompoundMap_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "CompoundMap_c.h"
#include "CompFunctions.h"
#include "MemUtils.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

static long theSegno,theSegStart,theSegEnd,theIndex,theBndryStart,theBndryEnd;
static Boolean IsClockWise;

float DistFromWPointToSegment(long pLong, long pLat, long long1, long lat1, 
							  long long2, long lat2, long dLong, long dLat);

CompoundMap_c::CompoundMap_c (char *name, WorldRect bounds) : PtCurMap_c (name, bounds), Map_c (name,bounds)
{
	mapList = 0;
	
	bMapsOpen = FALSE;
	
	return;
}
TCurrentMover* CompoundMap_c::GetCompoundMover()
{
	TMover *thisMover = nil;
	long i,d;
	for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	{
		this -> moverList -> GetListItem ((Ptr) &thisMover, i);
		if(thisMover -> IAm(TYPE_COMPOUNDMOVER)) return dynamic_cast<TCurrentMover*>(thisMover);
	}
	return nil;
}
// will need to deal with this for new curvilinear algorithm when start using subsurface movement
long CompoundMap_c::PointOnWhichSeg(long point)	// This is really which boundary
{
	long numSegs = GetNumBoundarySegs(),jseg;
	for(jseg = 0; jseg < numSegs; jseg++)
	{
		if(point <= (*fBoundarySegmentsH)[jseg])
		{
			return jseg;
		}
	}
	return -1;
}

Boolean CompoundMap_c::ContiguousPoints(long p1, long p2)
{
	
	long segno1 = PointOnWhichSeg(p1);
	long segno2 = PointOnWhichSeg(p2);
	
	if(segno1 != segno2)return false;
	return (p2 == PrevPointOnSeg(segno1,p1)) || (p2 == NextPointOnSeg(segno1,p1));
}


long CompoundMap_c::PointOnWhichSeg(long longVal, long latVal, long *startver, long *endver, float *distToSeg)
{
	long numSegs = GetNumBoundarySegs(), jseg;
	long firstPoint, lastPoint, segNo, endPt, x1, y1, x2, y2, closestSeg = -1;
	
	long dLong, dLat;
	float dist, smallestDist = 100.;
	long oneSecond = (1000000/3600); // map border is several pixels wide
	//long oneSecond = 0;
	
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return -1;
	//dLong = dLat = oneSecond * 5;
	dLong = dLat = oneSecond * 50;
	*distToSeg = -1;
	
	for(jseg = 0; jseg < numSegs; jseg++)	// loop through the boundaries
	{
		firstPoint = jseg == 0? 0: (*fBoundarySegmentsH)[jseg-1] + 1;
		lastPoint = (*fBoundarySegmentsH)[jseg]+1;
		// check each segment on the boundary
		for(segNo = firstPoint; segNo < lastPoint; segNo++)
		{
			if (segNo == lastPoint-1)
				endPt = firstPoint;
			else
				endPt = segNo+1;
			x1 = (*ptsHdl)[segNo].h;
			y1 = (*ptsHdl)[segNo].v;
			x2 = (*ptsHdl)[endPt].h;
			y2 = (*ptsHdl)[endPt].v;
			
			dist = DistFromWPointToSegment(longVal, latVal, x1, y1, x2, y2, dLong, dLat);
			if (dist==-1) continue;	// not within range
			
			if (dist<smallestDist)
			{
				smallestDist = dist;
				*startver = segNo;
				*endver = endPt;
				closestSeg = jseg;
				*distToSeg = smallestDist;
			}
		}
	}
	return closestSeg;
}

TMover* CompoundMap_c::GetMover(ClassID desiredClassID)
{
	// loop through each mover in the map
	TMover *thisMover = nil;
	long k;
	for (k = 0; k < moverList -> GetItemCount (); k++)
	{
		moverList -> GetListItem ((Ptr) &thisMover, k);
		if(thisMover -> IAm(desiredClassID)) return thisMover;
	}
	return nil;
}


TTriGridVel* CompoundMap_c::GetGrid(Boolean wantRefinedGrid)
{
	TTriGridVel* triGrid = 0;	
	TMover *mover = 0;
	
	// Figure out if this map has a PtCurMover or TCATSMover3D current
	mover = this->GetMover(TYPE_COMPOUNDMOVER);
	if (mover)
	{
		return  (dynamic_cast<TCompoundMover *>(mover)) -> GetGrid(false);
	}
	else	
	{
	}
	
	return triGrid;
}

TTriGridVel3D* CompoundMap_c::GetGrid3D(Boolean wantRefinedGrid)
{
	TTriGridVel3D* triGrid = 0;	
	TMover *mover = 0;
	
	// Figure out if this map has a PtCurMover or TCATSMover3D current
	// code goes here, make sure the mover is 3D...
	mover = this->GetMover(TYPE_COMPOUNDMOVER);
	if (mover)
	{
		return (dynamic_cast<TCompoundMover *>(mover)) -> GetGrid3D(false);
	}
	else	
	{
	}
	
	return triGrid;
}

TTriGridVel3D* CompoundMap_c::GetGrid3DFromMapIndex(long mapIndex)
{
	TTriGridVel3D* triGrid = 0;	
	TMover *mover = 0;
	
	if (mapIndex < 0) return nil;
	// Figure out if this map has a PtCurMover or TCATSMover3D current
	// code goes here, make sure the mover is 3D...
	mover = this->GetMover(TYPE_COMPOUNDMOVER);
	if (mover)
	{
		return (dynamic_cast<TCompoundMover *>(mover)) -> GetGrid3DFromMoverIndex(mapIndex);
	}
	else	
	{
	}
	
	return triGrid;
}

TCurrentMover* CompoundMap_c::Get3DCurrentMover()
{	
	TMover *thisMover = nil;
	long i,d;
	for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	{
		this -> moverList -> GetListItem ((Ptr) &thisMover, i);
		if (thisMover->IAm(TYPE_COMPOUNDMOVER))
			return  (dynamic_cast<TCompoundMover*>(thisMover))->Get3DCurrentMover();	// which one to use?
	}
	return nil;
}

TCurrentMover* CompoundMap_c::Get3DCurrentMoverFromIndex(long moverIndex)
{	
	TMover *thisMover = nil;
	long i,d;
	for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	{
		this -> moverList -> GetListItem ((Ptr) &thisMover, i);
		if (thisMover->IAm(TYPE_COMPOUNDMOVER))
			return  (dynamic_cast<TCompoundMover*>(thisMover))->Get3DCurrentMoverFromIndex(moverIndex);	// which one to use?
	}
	return nil;
}

LongPointHdl CompoundMap_c::GetPointsHdl(Boolean useRefinedGrid)	// always false at this point...
{
	LongPointHdl ptsHdl = 0;
	TMover *mover=0;
	
	// Figure out if this map has a PtCurMover or TCATSMover3D current
	mover = this->GetMover(TYPE_COMPOUNDMOVER);
	if (mover)
		ptsHdl = (dynamic_cast<TCompoundMover *>(mover))->GetPointsHdl();
	else
	{
	}
	
	return ptsHdl;
}

Boolean CompoundMap_c::InVerticalMap(WorldPoint3D wp)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel3D* triGrid = GetGrid3D(false);	// don't use refined grid, depths aren't refined
	NetCDFMover *mover = dynamic_cast<NetCDFMover*>(model->GetMover(TYPE_NETCDFMOVER));
	
	if (mover && mover->fVar.gridType==SIGMA_ROMS)	// really need to get priority grid
		depthAtPoint = ((NetCDFMoverCurv*)mover)->GetTotalDepth(wp.p,-1);
	else
	{
		if (!triGrid) return false; // some error alert, no depth info to check
		interpolationVal = triGrid->GetInterpolationValues(wp.p);
		depthsHdl = triGrid->GetDepths();
		//depthsHdl = triGrid->GetBathymetry();
		if (!depthsHdl) return false;	// some error alert, no depth info to check
		if (interpolationVal.ptIndex1<0)	
		{
			//printError("Couldn't find point in dagtree"); 
			return false;
		}
		
		depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
		depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
		depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
		depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;
	}
	if (wp.z >= depthAtPoint || wp.z < 0)	// allow surface but not bottom
		return false;
	else
		return true;
}


double CompoundMap_c::DepthAtPoint(WorldPoint wp, NetCDFMover *mover, TTriGridVel3D *triGrid)
{	// here need to check by priority
	float depth1,depth2,depth3;
	double depthAtPoint;	
	long mapIndex;
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	//TTriGridVel3D* triGrid = GetGrid3D(false);	// don't use refined grid, depths aren't refined
	//NetCDFMover *mover = (NetCDFMover*)(model->GetMover(TYPE_NETCDFMOVER));
	//NetCDFMover *mover = (NetCDFMover*)(Get3DCurrentMover());
	
	if (mover && mover->fVar.gridType==SIGMA_ROMS)
		return ((NetCDFMoverCurv*)mover)->GetTotalDepth(wp,-1);	// expand options here
	
	if (!triGrid) return -1; // some error alert, no depth info to check
	interpolationVal = triGrid->GetInterpolationValues(wp);
	depthsHdl = triGrid->GetDepths();
	if (!depthsHdl) return -1;	// some error alert, no depth info to check
	if (interpolationVal.ptIndex1<0)	
	{
		//printError("Couldn't find point in dagtree"); 
		return -1;
	}
	
	depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
	depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
	depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
	depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;
	
	return depthAtPoint;
}

/*float CompoundMap_c::GetMaxDepth(void)
 {	// 2D grid
 long i,numDepths;
 float depth, maxDepth=0;
 FLOATH depthsHdl = 0;
 TTriGridVel* triGrid = GetGrid(false);	// don't use refined grid, depths aren't refined
 
 if (!triGrid) return 0; // some error alert, no depth info to check
 
 //depthsHdl = triGrid->GetDepths();
 depthsHdl = triGrid->GetBathymetry(); // I think this is only for CATS grids ...
 if (!depthsHdl) return 0;	// some error alert, no depth info to check
 
 numDepths = _GetHandleSize((Handle)depthsHdl)/sizeof(**depthsHdl);
 for (i=0;i<numDepths;i++)
 {
 depth = INDEXH(depthsHdl,i);
 if (depth > maxDepth) 
 maxDepth = depth;
 }
 return maxDepth;
 }*/

float CompoundMap_c::GetMaxDepth2(void)
{	// may want to extend for SIGMA_ROMS (all ROMS?) to check the cell depths rather than point depths
	long i,n;
	float maxDepth, overallMaxDepth = 0.;
	TMap *map = 0;
	
	// draw each of the maps contours (in reverse order to show priority)		
	for (i = 0, n = mapList->GetItemCount() ; i < n ; i++) 
	{
		mapList->GetListItem((Ptr)&map, i);
		if (map) maxDepth = (dynamic_cast<PtCurMap *>(map)) -> GetMaxDepth2();
		if (maxDepth > overallMaxDepth) overallMaxDepth = maxDepth;
	}
	return overallMaxDepth;		
}

WorldPoint3D CompoundMap_c::TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint)
{
	WorldPoint3D movedPoint = {0,0,0.}, firstEndPoint = {0,0,0.}, secondEndPoint = {0,0,0.};
	WorldPoint3D testPt = {0,0,0.}, realBeachedPt = {0,0,0.};
	double alpha, sideA, sideB, sideC, sideD, shorelineLength;
	long startver, endver, x1, y1, x2, y2, testcase = 0;
	
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return waterPoint;
	
	if (!InMap (beachedPoint.p))
		return waterPoint;	// something went wrong don't do anything
	
	if (OnLand (beachedPoint.p))	
	{
		// Find shoreline segment where LE has beached and get the endpoints
		// Then move LE parallel to shoreline in the direction the beaching vector tends towards
		// May only want to do this for current movement...
		WorldPoint center;
		float dist;
		long segNo = PointOnWhichSeg(beachedPoint.p.pLong,beachedPoint.p.pLat,&startver,&endver,&dist);
		if (segNo==-1) return waterPoint;	// this should probably be an error
		
		firstEndPoint.p.pLong = (*ptsHdl)[startver].h;
		firstEndPoint.p.pLat = (*ptsHdl)[startver].v;
		secondEndPoint.p.pLong = (*ptsHdl)[endver].h;
		secondEndPoint.p.pLat = (*ptsHdl)[endver].v;
		center.pLong = (waterPoint.p.pLong + beachedPoint.p.pLong) / 2;
		center.pLat = (waterPoint.p.pLat + beachedPoint.p.pLat) / 2;
		
		sideA = DistanceBetweenWorldPoints(waterPoint.p,firstEndPoint.p);
		sideB = DistanceBetweenWorldPoints(beachedPoint.p,waterPoint.p);
		sideC = DistanceBetweenWorldPoints(beachedPoint.p,firstEndPoint.p);
		sideD = DistanceBetweenWorldPoints(waterPoint.p,toPoint.p);
		
		shorelineLength = DistanceBetweenWorldPoints(secondEndPoint.p,firstEndPoint.p);
		
		testPt.p.pLat = beachedPoint.p.pLat + DistanceToLat(LongToDistance(firstEndPoint.p.pLong - secondEndPoint.p.pLong,center)*(sideD/shorelineLength));
		testPt.p.pLong = beachedPoint.p.pLong - DistanceToLong(LatToDistance(firstEndPoint.p.pLat - secondEndPoint.p.pLat)*(sideD/shorelineLength),center);
		if (InMap(testPt.p) && !OnLand(testPt.p))
		{
			testcase = 1;
		}
		else
		{
			testPt.p.pLat = beachedPoint.p.pLat + DistanceToLat(LongToDistance(secondEndPoint.p.pLong - firstEndPoint.p.pLong,center)*(sideD/shorelineLength));
			testPt.p.pLong = beachedPoint.p.pLong - DistanceToLong(LatToDistance(secondEndPoint.p.pLat - firstEndPoint.p.pLat)*(sideD/shorelineLength),center);
			if (InMap(testPt.p) && !OnLand(testPt.p))
			{
				testcase = 2;
			}
		}
		if (testcase==1)
		{
			realBeachedPt.p.pLat = beachedPoint.p.pLat - DistanceToLat(LongToDistance(firstEndPoint.p.pLong - secondEndPoint.p.pLong,center)*(dist/shorelineLength));
			realBeachedPt.p.pLong = beachedPoint.p.pLong + DistanceToLong(LatToDistance(firstEndPoint.p.pLat - secondEndPoint.p.pLat)*(dist/shorelineLength),center);
			sideB = DistanceBetweenWorldPoints(realBeachedPt.p,waterPoint.p);
		}
		else if (testcase==2)
		{
			realBeachedPt.p.pLat = beachedPoint.p.pLat - DistanceToLat(LongToDistance(secondEndPoint.p.pLong - firstEndPoint.p.pLong,center)*(dist/shorelineLength));
			realBeachedPt.p.pLong = beachedPoint.p.pLong + DistanceToLong(LatToDistance(secondEndPoint.p.pLat - firstEndPoint.p.pLat)*(dist/shorelineLength),center);
			sideB = DistanceBetweenWorldPoints(realBeachedPt.p,waterPoint.p);
		}
		
		alpha = acos((sideB*sideB + sideC*sideC - sideA*sideA)/(2*sideB*sideC));
		
		// turn direction determined by which is greater, alpha or 90, towards larger one, if same?
		if (alpha > PI/2.)
		{
			movedPoint.p.pLat = waterPoint.p.pLat + DistanceToLat(LatToDistance(firstEndPoint.p.pLat - secondEndPoint.p.pLat)*(sideB/shorelineLength));
			movedPoint.p.pLong = waterPoint.p.pLong + DistanceToLong(LongToDistance(firstEndPoint.p.pLong - secondEndPoint.p.pLong,center)*(sideB/shorelineLength),center);
		}
		else
		{
			movedPoint.p.pLat = waterPoint.p.pLat + DistanceToLat(LatToDistance(secondEndPoint.p.pLat - firstEndPoint.p.pLat)*(sideB/shorelineLength));
			movedPoint.p.pLong = waterPoint.p.pLong + DistanceToLong(LongToDistance(secondEndPoint.p.pLong - firstEndPoint.p.pLong,center)*(sideB/shorelineLength),center);
		}
		movedPoint.z = beachedPoint.z;
		// check that movedPoint is not onLand
		if (InMap(movedPoint.p) && !OnLand(movedPoint.p))
			return movedPoint;
		else // try again
		{
			/*WorldRect wBounds = this -> GetMapBounds(); // use bounds to determine how far offshore to move point
			 double latDiff = fabs(float(wBounds.hiLat - wBounds.loLat)/1000000);
			 double lonDiff = fabs(float(wBounds.loLong - wBounds.hiLong)/1000000);
			 double distOffshore;	// probably want an option for user to set this value
			 if (latDiff >=1 || lonDiff >=1){ if (sideD<1) distOffshore = 1; else distOffshore = sideD;}
			 else if (latDiff >=.1 || lonDiff >=.1) { if (sideD<.5) distOffshore = .5; else distOffshore = sideD;}
			 else if (latDiff >=.01 || lonDiff >=.01) { if (sideD<.05) distOffshore = .05; else distOffshore = sideD;}*/
			
			double distOffshore;	// probably want an option for user to set this value
			//if (sideD<1) distOffshore = fMinDistOffshore; else distOffshore = sideD;	
			if (sideD<fMinDistOffshore) distOffshore = fMinDistOffshore; else distOffshore = sideD;	
			//if (sideD < 1) distOffshore = 1.;	// at least 1km
			//if (sideD < 1) distOffshore = .05;	// at least 1km
			//if (sideD < 5) distOffshore = 5.;	// at least 1km
			//else distOffshore = sideD;
			{
				movedPoint.p.pLat = beachedPoint.p.pLat + DistanceToLat(LongToDistance(firstEndPoint.p.pLong - secondEndPoint.p.pLong,center)*(distOffshore/shorelineLength));
				movedPoint.p.pLong = beachedPoint.p.pLong - DistanceToLong(LatToDistance(firstEndPoint.p.pLat - secondEndPoint.p.pLat)*(distOffshore/shorelineLength),center);
			}
			if (InMap(movedPoint.p) && !OnLand(movedPoint.p))
				return movedPoint;
			else
			{
				movedPoint.p.pLat = beachedPoint.p.pLat + DistanceToLat(LongToDistance(secondEndPoint.p.pLong - firstEndPoint.p.pLong,center)*(distOffshore/shorelineLength));
				movedPoint.p.pLong = beachedPoint.p.pLong - DistanceToLong(LatToDistance(secondEndPoint.p.pLat - firstEndPoint.p.pLat)*(distOffshore/shorelineLength),center);
			}
			if (InMap(movedPoint.p) && !OnLand(movedPoint.p))
				return movedPoint;
			else
			{
				distOffshore = 2*distOffshore;
				movedPoint.p.pLat = beachedPoint.p.pLat + DistanceToLat(LongToDistance(firstEndPoint.p.pLong - secondEndPoint.p.pLong,center)*(distOffshore/shorelineLength));
				movedPoint.p.pLong = beachedPoint.p.pLong - DistanceToLong(LatToDistance(firstEndPoint.p.pLat - secondEndPoint.p.pLat)*(distOffshore/shorelineLength),center);
				if (InMap(movedPoint.p) && !OnLand(movedPoint.p))
					return movedPoint;
				else
				{
					movedPoint.p.pLat = beachedPoint.p.pLat + DistanceToLat(LongToDistance(secondEndPoint.p.pLong - firstEndPoint.p.pLong,center)*(distOffshore/shorelineLength));
					movedPoint.p.pLong = beachedPoint.p.pLong - DistanceToLong(LatToDistance(secondEndPoint.p.pLat - firstEndPoint.p.pLat)*(distOffshore/shorelineLength),center);
				}
				if (InMap(movedPoint.p) && !OnLand(movedPoint.p))
					return movedPoint;
				else
					return waterPoint;
			}
		}
	}
	return waterPoint;	// shouldn't get here
}

double CompoundMap_c::DepthAtCentroid(long triNum)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	long ptIndex1,ptIndex2,ptIndex3;
	FLOATH depthsHdl = 0;
	TTriGridVel3D* triGrid = GetGrid3D(false);	// don't use refined grid, depths aren't refined
	
	TopologyHdl topH ;
	
	//NetCDFMover *mover = (NetCDFMover*)(model->GetMover(TYPE_NETCDFMOVER));
	TCurrentMover* compoundMover = this->GetCompoundMover();
	TCurrentMover* mover = ((TCompoundMover *)compoundMover)->Get3DCurrentMoverFromIndex(0);
	
	if (mover && ((NetCDFMoverCurv*)mover)->fVar.gridType==SIGMA_ROMS)
		return ((NetCDFMoverCurv*)mover)->GetTotalDepthFromTriIndex(triNum);
	
	if (triNum < 0) return -1;
	if (!triGrid) return -1; // some error alert, no depth info to check
	
	topH = triGrid -> GetTopologyHdl();
	if (!topH) return -1;
	
	ptIndex1 = (*topH)[triNum].vertex1;
	ptIndex2 = (*topH)[triNum].vertex2;
	ptIndex3 = (*topH)[triNum].vertex3;
	
	depthsHdl = triGrid->GetDepths();
	if (!depthsHdl) return -1;	// some error alert, no depth info to check
	
	depth1 = (*depthsHdl)[ptIndex1];
	depth2 = (*depthsHdl)[ptIndex2];
	depth3 = (*depthsHdl)[ptIndex3];
	depthAtPoint = (depth1 + depth2 + depth3) / 3.;
	
	return depthAtPoint;
}

double CompoundMap_c::DepthAtCentroidFromMapIndex(long triNum,long mapIndex)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	long ptIndex1,ptIndex2,ptIndex3;
	FLOATH depthsHdl = 0;
	TTriGridVel3D* triGrid = GetGrid3DFromMapIndex(mapIndex);	// don't use refined grid, depths aren't refined
	
	TopologyHdl topH ;
	
	//NetCDFMover *mover = (NetCDFMover*)(model->GetMover(TYPE_NETCDFMOVER));
	TCurrentMover *compoundMover = this->GetCompoundMover();
	TCurrentMover* mover = ((TCompoundMover *)compoundMover)->Get3DCurrentMoverFromIndex(mapIndex);
	
	if (mover && ((NetCDFMoverCurv*)mover)->fVar.gridType==SIGMA_ROMS)
		return ((NetCDFMoverCurv*)mover)->GetTotalDepthFromTriIndex(triNum);
	
	if (triNum < 0) return -1;
	if (!triGrid) return -1; // some error alert, no depth info to check
	
	topH = triGrid -> GetTopologyHdl();
	if (!topH) return -1;
	
	ptIndex1 = (*topH)[triNum].vertex1;
	ptIndex2 = (*topH)[triNum].vertex2;
	ptIndex3 = (*topH)[triNum].vertex3;
	
	depthsHdl = triGrid->GetDepths();
	if (!depthsHdl) return -1;	// some error alert, no depth info to check
	
	depth1 = (*depthsHdl)[ptIndex1];
	depth2 = (*depthsHdl)[ptIndex2];
	depth3 = (*depthsHdl)[ptIndex3];
	depthAtPoint = (depth1 + depth2 + depth3) / 3.;
	
	return depthAtPoint;
}

WorldPoint3D CompoundMap_c::ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp, NetCDFMover *mover, TTriGridVel3D *triGrid3D)
{
	//WorldPoint3D movedPoint = model->TurnLEAlongShoreLine(fromWPt, wp, this);	// use length of fromWPt to beached point or to toWPt?
	WorldPoint3D movedPoint = TurnLEAlongShoreLine(fromWPt, wp, toWPt);	// use length of fromWPt to beached point or to toWPt?
	/*if (!InVerticalMap(movedPoint)) 
	 {
	 movedPoint.z = fromWPt.z;	// try not changing depth
	 if (!InVerticalMap(movedPoint))
	 movedPoint.p = fromWPt.p;	// use original point
	 }*/
	//movedPoint.z = toWPt.z; // attempt the z move
	// code goes here, check mixedLayerDepth?
	if (!InVerticalMap(movedPoint) || movedPoint.z == 0) // these points are supposed to be below the surface
	{
		double depthAtPt = DepthAtPoint(movedPoint.p, mover, triGrid3D);	// code goes here, a check on return value
		if (depthAtPt < 0) 
		{
			OSErr err = 0;
			return fromWPt;
		}
		if (depthAtPt==0)
			movedPoint.z = .1;
		if (movedPoint.z > depthAtPt) movedPoint.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
		//if (movedPoint.z > depthAtPt) movedPoint.z = GetRandomFloat(.7*depthAtPt,.99*depthAtPt);
		if (movedPoint.z <= 0) movedPoint.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
		//movedPoint.z = fromWPt.z;	// try not changing depth
		//if (!InVerticalMap(movedPoint))
		//movedPoint.p = fromWPt.p;	// use original point - code goes here, need to find a z in the map
	}
	return movedPoint;
}

/////////////////////////////////////////////////

/*double CompoundMap_c::GetBreakingWaveHeight(void)
 {
 double velAt10meters=0, windStress, significantWaveHt, breakingWaveHt = 0;
 VelocityRec windVel;
 OSErr err = 0;
 if (fWaveHtInput>0) 	// user input value by hand
 return fBreakingWaveHeight;
 else
 {
 TWindMover *wind = model -> GetWindMover(false);
 if (wind) err = wind -> GetTimeValue(model->GetModelTime(),&windVel);
 if (err || !wind) 
 {
 velAt10meters = 1;	// set to a minimum wind value
 //printNote("There is no wind, breaking wave height is zero");	// have to decide what to do in this case
 //return 0;
 }
 else 
 {
 velAt10meters = sqrt(windVel.u*windVel.u + windVel.v*windVel.v);	// m/s
 // if wind speed is known at other than 10m U_10 = U_z * (10/z)^(1/7) for z up to 20m
 // for now we assume wind is at 10m
 }
 windStress = .71 * velAt10meters;
 significantWaveHt = .0248 * (windStress * windStress);
 breakingWaveHt = significantWaveHt / 1.42;
 }
 
 return breakingWaveHt;
 }*/

long CompoundMap_c::GetNumBoundarySegs(void)
{
	long i,numMaps;
	TMap *map = 0;
	
	numMaps = mapList->GetItemCount();
	mapList->GetListItem((Ptr)&map, numMaps-1);	// get lowest priority map ? (probably biggest)
	
	return (dynamic_cast<PtCurMap *>(map))->GetNumBoundarySegs();
}

long CompoundMap_c::GetNumPointsInBoundarySeg(long segno)
{
	long i,numMaps;
	TMap *map = 0;
	
	numMaps = mapList->GetItemCount();
	mapList->GetListItem((Ptr)&map, numMaps-1);	// get lowest priority map ? (probably biggest)
	
	return (dynamic_cast<PtCurMap *>(map))->GetNumPointsInBoundarySeg(segno);
}

long CompoundMap_c::GetNumBoundaryPts(void)
{
	long numMaps;
	TMap *map = 0;
	
	numMaps = mapList->GetItemCount();
	mapList->GetListItem((Ptr)&map, numMaps-1);	// get lowest priority map ? (probably biggest)
	
	return (dynamic_cast<PtCurMap *>(map))->GetNumBoundaryPts();
}

Boolean CompoundMap_c::IsBoundaryPoint(long pt)
{
	return pt < GetNumBoundaryPts();
}

void CompoundMap_c::FindStartEndSeg(long ptnum,long *startPt, long *endPt)
{
	long jseg, nSegs = GetNumBoundarySegs(),segno;
	for(jseg = 0; jseg < nSegs; jseg++)
	{
		if(ptnum <= (*fBoundarySegmentsH)[jseg])
		{
			segno = jseg;
			break;
		}
	}
	
	*startPt = segno==0? 0: (*fBoundarySegmentsH)[segno-1]+1;
	*endPt = (*fBoundarySegmentsH)[segno];
}

long CompoundMap_c::NextPointOnSeg(long segno, long point)
{
	long incr = segno == 0? 1 : -1;
	long startno, endno;
	
	startno = segno== 0 ? 0 : (*fBoundarySegmentsH)[segno-1]+1;
	endno  = (*fBoundarySegmentsH)[segno];
	
	point++;
	if(point > endno)point = startno;
	return point;
}

long CompoundMap_c::PrevPointOnSeg(long segno, long point)
{
	long incr = segno == 0? 1 : -1;
	long startno, endno;
	
	startno = segno== 0 ? 0 : (*fBoundarySegmentsH)[segno-1]+1;
	endno  = (*fBoundarySegmentsH)[segno];
	
	point--;
	if(point < startno)point = endno;
	return point;
}

/*Boolean CompoundMap_c::MoreSegments(LONGH segh,long *startIndex, long *endIndex,long *curIndex)
 {
 //long i,numitems = GetNumLONGHItems(segh);
 long i,numitems;
 
 if (segh) numitems = _GetHandleSize((Handle)segh)/sizeof(**segh); 
 if(*curIndex >=numitems) return false;
 
 *startIndex = *curIndex;
 for(i=*curIndex; i < numitems; i++)
 {
 if((*segh)[i] == -1)break;
 }
 
 *endIndex = i;
 *curIndex = i+1;
 return true;
 }*/


void CompoundMap_c::InitBoundaryIter(Boolean clockwise,long segno, long startno, long endno)
{
	theSegno = segno;
	theSegStart = startno;
	theSegEnd = endno;
	theIndex = startno;
	theBndryStart = segno==0? 0: (*fBoundarySegmentsH)[segno-1]+1;
	theBndryEnd = (*fBoundarySegmentsH)[segno];
	IsClockWise = clockwise;
}

Boolean CompoundMap_c::MoreBoundarySegments(long *a,long *b)
{
	long j;
	if(theSegStart == theSegEnd || theIndex == theSegEnd) return false;
	if(theSegno == 0)
	{
		j = IsClockWise ? theIndex - 1 : theIndex + 1;
	}
	else
	{
		j = IsClockWise ? theIndex + 1 : theIndex - 1;
	}
	if( j > theBndryEnd) j = theBndryStart;
	if(j < theBndryStart)j = theBndryEnd;
	*a = theIndex;
	*b = j;
	theIndex = j;
	return true;
}

void  CompoundMap_c::FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo)
{
	long startVer = 0,i,jseg;
	//WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);
	WorldPoint wp2;
	LongPoint lp;
	long lastVer = GetNumBoundaryPts();
	//long nbounds = GetNumBoundaries();
	long nSegs = GetNumBoundarySegs();	
	float wdist = LatToDistance(ScreenToWorldDistance(4));
	LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
	if(!ptsHdl) return;
	*verNum= -1;
	*segNo =-1;
	for(i = 0; i < lastVer; i++)
	{
		//wp2 = (*gVertices)[i];
		lp = (*ptsHdl)[i];
		wp2.pLat = lp.v;
		wp2.pLong = lp.h;
		
		if(WPointNearWPoint(wp,wp2 ,wdist))
		{
			//for(jseg = 0; jseg < nbounds; jseg++)
			for(jseg = 0; jseg < nSegs; jseg++)
			{
				if(i <= (*fBoundarySegmentsH)[jseg])
				{
					*verNum  = i;
					*segNo = jseg;
					break;
				}
			}
		}
	} 
}




double CompoundMap_c::PathLength(Boolean selectionDirection,long segNo, long startno, long endno)
{
	long p1,p2;
	double x1,x2,y1,y2,len=0;
	InitBoundaryIter(selectionDirection, segNo,  startno, endno);
	LongPointHdl ptsHdl = GetPointsHdl(false);	
	if(!ptsHdl) return -1;
	while(MoreBoundarySegments(&p1,&p2))		
	{
		x1 = (*ptsHdl)[p1].h/1000000.;
		y1 = (*ptsHdl)[p1].v/1000000.;
		x2 = (*ptsHdl)[p2].h/1000000.;
		y2 = (*ptsHdl)[p2].v/1000000.;
		len += sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
	}
	return len;
}

/*long CompoundMap_c::WhichSelectedSegAmIIn(long index)
 {
 long i, startIndex, endIndex, curIndex=0, p, afterP, firstSegIndex, lastSegIndex, index1;
 long segNo,lastPtOnSeg,firstPtOnSeg,selectionNumber=0,midSelectionIndex;
 
 while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
 {
 if(endIndex <= startIndex)continue;
 
 selectionNumber++;
 
 // use the p/afterp to ensure points match to correct segment if endpoints touch
 segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
 firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
 lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
 for(i=startIndex; i< endIndex-1; i++)
 {
 index1 = (*fSelectedBeachHdl)[i]; 
 p = (*fSelectedBeachHdl)[i];
 afterP = (*fSelectedBeachHdl)[i+1];
 if ((p<afterP && !(p==firstPtOnSeg && afterP==lastPtOnSeg)) || (afterP==firstPtOnSeg && p==lastPtOnSeg))
 {
 if (afterP==index) 
 return selectionNumber;
 }
 else if ((p>afterP && !(afterP==firstPtOnSeg && p==lastPtOnSeg)) || (p==firstPtOnSeg && afterP==lastPtOnSeg))
 {
 if (p==index) 
 return selectionNumber;
 }
 }
 }
 return -1;	// this is an error
 }
 
 #define POINTDRAWFLAG 0
 void CompoundMap_c::DrawSegmentLabels(Rect r)
 {
 long i, startIndex, endIndex, curIndex=0, p, afterP, firstSegIndex, lastSegIndex;
 long segNo,lastPtOnSeg,firstPtOnSeg,selectionNumber=0,midSelectionIndex;
 RGBColor sc;
 char numstr[40];
 short x,y;
 Point pt;
 Boolean offQuickDrawPlane = false;
 LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
 if(!ptsHdl) return;
 
 GetForeColor(&sc);
 RGBForeColor(&colors[RED]);
 //TextSizeTiny();	
 
 while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
 {
 if(endIndex <= startIndex)continue;
 
 selectionNumber++;
 segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
 firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
 lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
 firstSegIndex = (*fSelectedBeachHdl)[startIndex];
 lastSegIndex = (*fSelectedBeachHdl)[endIndex-1];
 midSelectionIndex = (firstSegIndex+lastSegIndex)/2;
 pt = GetQuickDrawPt((*ptsHdl)[midSelectionIndex].h,(*ptsHdl)[midSelectionIndex].v,&r,&offQuickDrawPlane);
 //MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 MyNumToStr(selectionNumber,numstr);
 x = pt.h;
 y = pt.v;
 MyDrawString(x,y,numstr,true,POINTDRAWFLAG);
 }
 RGBForeColor(&sc);
 }
 
 void CompoundMap_c::DrawPointLabels(Rect r)
 {
 long i, startIndex, endIndex, curIndex=0, firstSegIndex, lastSegIndex;
 long segNo,lastPtOnSeg,firstPtOnSeg,selectionNumber=0;
 RGBColor sc;
 char numstr[40];
 short x,y;
 Point pt;
 Boolean offQuickDrawPlane = false;
 LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
 if(!ptsHdl) return;
 
 GetForeColor(&sc);
 RGBForeColor(&colors[BLUE]);
 TextSizeTiny();	
 
 while(MoreSegments(fSelectedBeachHdl,&startIndex, &endIndex, &curIndex))
 {
 if(endIndex <= startIndex)continue;
 
 selectionNumber++;
 segNo = PointOnWhichSeg((*fSelectedBeachHdl)[startIndex]);
 firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
 lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
 firstSegIndex = (*fSelectedBeachHdl)[startIndex];
 lastSegIndex = (*fSelectedBeachHdl)[endIndex-1];
 if (firstSegIndex < lastSegIndex)  {startIndex = firstSegIndex; endIndex = lastSegIndex;}
 else {startIndex = lastSegIndex; endIndex = firstSegIndex;}
 for (i=startIndex;i<=endIndex;i++)
 {
 pt = GetQuickDrawPt((*ptsHdl)[i].h,(*ptsHdl)[i].v,&r,&offQuickDrawPlane);
 //MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 MyNumToStr(i,numstr);
 x = pt.h;
 y = pt.v;
 MyDrawString(x,y,numstr,false,POINTDRAWFLAG);
 }
 }
 RGBForeColor(&sc);
 }
 
 void CompoundMap_c::DrawBoundaries(Rect r)
 {
 long nSegs = GetNumBoundarySegs();	
 long theSeg,startver,endver,j;
 long x,y;
 Point pt;
 Boolean offQuickDrawPlane = false;
 
 long penWidth = 3;
 long halfPenWidth = penWidth/2;
 
 PenNormal();
 RGBColor sc;
 GetForeColor(&sc);
 
 // to support new curvilinear algorithm
 if (fBoundaryPointsH)
 {
 DrawBoundaries2(r);
 return;
 }
 
 LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
 if(!ptsHdl) return;
 
 #ifdef MAC
 PenSize(penWidth,penWidth);
 #else
 PenStyle(BLACK,penWidth);
 #endif
 
 // have each seg be a polygon with a fill option - land only, maybe fill with a pattern?
 for(theSeg = 0; theSeg < nSegs; theSeg++)
 {
 startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
 endver = (*fBoundarySegmentsH)[theSeg]+1;
 
 pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
 MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 for(j = startver + 1; j < endver; j++)
 {
 if ((*fBoundaryTypeH)[j]==2)	// a water boundary
 RGBForeColor(&colors[BLUE]);
 else// add option to change color, light or dark depending on which is easier to see , see premerge GNOME_beta
 {
 RGBForeColor(&colors[BROWN]);	// land
 }
 if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[j]==1)
 RGBForeColor(&colors[DARKGREEN]);
 pt = GetQuickDrawPt((*ptsHdl)[j].h,(*ptsHdl)[j].v,&r,&offQuickDrawPlane);
 if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
 {
 MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 else
 MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
 RGBForeColor(&colors[BLUE]);
 else
 {
 RGBForeColor(&colors[BROWN]);	// land
 }
 if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[startver]==1)
 RGBForeColor(&colors[DARKGREEN]);
 pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
 if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
 {
 MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 else
 MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 
 #ifdef MAC
 PenSize(1,1);
 #else
 PenStyle(BLACK,1);
 #endif
 RGBForeColor(&sc);
 if (fSelectedBeachFlagHdl) DrawSegmentLabels(r);
 if (fSelectedBeachFlagHdl && fDiagnosticStrType==SHORELINEPTNUMS) DrawPointLabels(r);
 }
 
 void CompoundMap_c::DrawBoundaries2(Rect r)
 {
 // should combine into original DrawBoundaries, just check for fBoundaryPointsH
 PenNormal();
 RGBColor sc;
 GetForeColor(&sc);
 
 TMover *mover=0;
 
 long nSegs = GetNumBoundarySegs();	
 long theSeg,startver,endver,j;
 long x,y,index1,index;
 Point pt;
 Boolean offQuickDrawPlane = false;
 
 long penWidth = 3;
 long halfPenWidth = penWidth/2;
 
 LongPointHdl ptsHdl = GetPointsHdl(false);	// will use refined grid if there is one
 if(!ptsHdl) return;
 
 //mover = this->GetMover(TYPE_PTCURMOVER);
 //if (mover)
 //ptsHdl = ((PtCurMover *)mover)->GetPointsHdl();
 //else	return; // some error alert
 //if(!ptsHdl) return;
 
 #ifdef MAC
 PenSize(penWidth,penWidth);
 #else
 PenStyle(BLACK,penWidth);
 #endif
 
 
 for(theSeg = 0; theSeg < nSegs; theSeg++)
 {
 startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
 endver = (*fBoundarySegmentsH)[theSeg]+1;
 index1 = (*fBoundaryPointsH)[startver];
 pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
 MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 for(j = startver + 1; j < endver; j++)
 {
 index = (*fBoundaryPointsH)[j];
 if ((*fBoundaryTypeH)[j]==2)	// a water boundary
 RGBForeColor(&colors[BLUE]);
 else
 RGBForeColor(&colors[BROWN]);	// land
 if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[j]==1)
 RGBForeColor(&colors[DARKGREEN]);
 pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
 if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
 {
 MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 else
 MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
 RGBForeColor(&colors[BLUE]);
 else
 RGBForeColor(&colors[BROWN]);	// land
 if (fSelectedBeachFlagHdl && (*fSelectedBeachFlagHdl)[startver]==1)
 RGBForeColor(&colors[DARKGREEN]);
 pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
 if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
 {
 MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 else
 MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
 }
 
 #ifdef MAC
 PenSize(1,1);
 #else
 PenStyle(BLACK,1);
 #endif
 RGBForeColor(&sc);
 }*/

OSErr CompoundMap_c::GetDepthAtMaxTri(long *maxTriIndex,double *depthAtPnt, NetCDFMover *mover, TTriGridVel3D *triGrid3D)	
{	// 
	long i,j,n,numOfLEs=0,numLESets,numDepths=0,numTri;
	TTriGridVel3D* triGrid = GetGrid3D(false);
	TDagTree *dagTree = 0;
	LONGH numLEsInTri = 0;
	DOUBLEH massInTriInGrams = 0;
	TopologyHdl topH = 0;
	LERec LE;
	OSErr err = 0;
	double triArea, triVol, oilDensityInWaterColumn, massInGrams, totalVol=0, depthAtPt = 0;
	long numLEsInTriangle,numLevels,totalLEs=0,maxTriNum=-1;
	double concInSelectedTriangles=0,maxConc=0;
	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
	short massunits;
	double density, LEmass;
	TLEList *thisLEList = 0;
	//short massunits = thisLEList->GetMassUnits();
	//double density =  thisLEList->fSetSummary.density;	// density set from API
	//double LEmass =  thisLEList->fSetSummary.totalMass / (double)(thisLEList->fSetSummary.numOfLEs);	
	
	dagTree = triGrid -> GetDagTree();
	if(!dagTree) return -1;
	topH = dagTree->GetTopologyHdl();
	if(!topH)	return -1;
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	numLEsInTri = (LONGH)_NewHandleClear(sizeof(long)*numTri);
	if (!numLEsInTri)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); err = -1; goto done; }
	massInTriInGrams = (DOUBLEH)_NewHandleClear(sizeof(double)*numTri);
	if (!massInTriInGrams)
	{ TechError("TCompoundMap::DrawContour()", "_NewHandleClear()", 0); err = -1; goto done; }
	//numOfLEs = thisLEList->numOfLEs;
	//massInGrams = VolumeMassToGrams(LEmass, density, massunits);
	if (!fContourLevelsH)
		if (!InitContourLevels()) {err = -1; goto done;}
	numLevels = GetNumDoubleHdlItems(fContourLevelsH);
	
	numLESets = model->LESetsList->GetItemCount();
	for (i = 0; i < numLESets; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		if (thisLEList->fLeType == UNCERTAINTY_LE)	
			continue;	
		if (!((*(dynamic_cast<TOLEList*>(thisLEList))).fDispersantData.bDisperseOil && ((model->GetModelTime() - model->GetStartTime()) >= (*(dynamic_cast<TOLEList*>(thisLEList))).fDispersantData.timeToDisperse ) )
			&& !(*(dynamic_cast<TOLEList*>(thisLEList))).fAdiosDataH && !((*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.z > 0)) 
			continue;
		numOfLEs = thisLEList->numOfLEs;
		// density set from API
		//density =  GetPollutantDensity(thisLEList->GetOilType());	
		density = ((dynamic_cast<TOLEList*>(thisLEList)))->fSetSummary.density;	
		massunits = thisLEList->GetMassUnits();
		
		for (j = 0 ; j < numOfLEs ; j++) 
		{
			LongPoint lp;
			long triIndex;
			thisLEList -> GetLE (j, &LE);
			//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
			if (!(LE.statusCode == OILSTAT_INWATER)) continue;// Windows compiler requires extra parentheses
			lp.h = LE.p.pLong;
			lp.v = LE.p.pLat;
			LEmass = GetLEMass(LE);	// will only vary for chemical with different release end time
			massInGrams = VolumeMassToGrams(LEmass, density, massunits);	// need to do this above too
			if (fContourDepth1==BOTTOMINDEX)
			{
				double depthAtLE = DepthAtPoint(LE.p, mover, triGrid3D);
				if (depthAtLE <= 0) continue;
				//if (LE.z > (depthAtLE-1.) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				if (LE.z > (depthAtLE-fBottomRange) && LE.z > 0 && LE.z <= depthAtLE) // assume it's in map, careful with 2 grids...
				{
					triIndex = dagTree -> WhatTriAmIIn(lp);
					//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
					//if (triIndex>=0 && LE.pollutantType == CHEMICAL) (*massInTri)[triIndex]+=GetLEMass(LE);	// use weathering information
					if (triIndex>=0) 
					{
						(*numLEsInTri)[triIndex]++;
						(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
					}
				}
			}
			else if (LE.z>fContourDepth1 && LE.z<=fContourDepth2) 
			{
				triIndex = dagTree -> WhatTriAmIIn(lp);
				//if (triIndex>=0) (*numLEsInTri)[triIndex]++;
				//if (triIndex>=0 && LE.pollutantType == CHEMICAL) (*massInTri)[triIndex]+=GetLEMass(LE);	// use weathering information
				if (triIndex>=0) 
				{
					(*numLEsInTri)[triIndex]++;
					(*massInTriInGrams)[triIndex]+=massInGrams;	// use weathering information
				}
			}
		}
	}
	
	for (i=0;i<numTri;i++)
	{	
		depthAtPt=0;
		double depthRange;
		//WorldPoint centroid = {0,0};
		if (triSelected && !(*triSelected)[i]) continue;	
		//if (!triGrid->GetTriangleCentroidWC(i,&centroid))
		{
			//depthAtPt = DepthAtPoint(centroid);
			depthAtPt = DepthAtCentroid(i);
		}
		triArea = (triGrid -> GetTriArea(i)) * 1000 * 1000;	// convert to meters
		if (!(fContourDepth1==BOTTOMINDEX))
		{
			depthRange = fContourDepth2 - fContourDepth1;
		}
		else
		{
			//depthRange = 1.; // for bottom will always contour 1m 
			//if (depthAtPt<1 && depthAtPt>0) depthRange = depthAtPt;
			depthRange = fBottomRange; // for bottom will always contour 1m 
			if (depthAtPt<fBottomRange && depthAtPt>0) depthRange = depthAtPt;
		}
		//if (depthAtPt < depthRange && depthAtPt != 0) depthRange = depthAtPt;
		if (depthAtPt < fContourDepth2 && depthAtPt > fContourDepth1 && depthAtPt > 0) depthRange = depthAtPt - fContourDepth1;
		triVol = triArea * depthRange; 
		//triVol = triArea * (fContourDepth2 - fContourDepth1); // code goes here, check this depth range is ok at all vertices
		numLEsInTriangle = (*numLEsInTri)[i];
		if (!(fContourDepth1==BOTTOMINDEX))		// need to decide what to do for bottom contour
			if (triGrid->CalculateDepthSliceVolume(&triVol,i,fContourDepth1,fContourDepth2)) goto done;
		/*if (thisLEList->GetOilType() == CHEMICAL) 
		 {
		 massInGrams = VolumeMassToGrams((*massInTri)[i], density, massunits);
		 oilDensityInWaterColumn = massInGrams / triVol;
		 }
		 else
		 oilDensityInWaterColumn = numLEsInTriangle * massInGrams / triVol; // units? milligrams/liter ?? for now gm/m^3
		 */
		if (numLEsInTriangle==0)
			continue;
		oilDensityInWaterColumn = (*massInTriInGrams)[i] / triVol;	// units? milligrams/liter ?? for now gm/m^3
		
		for (j=0;j<numLevels;j++)
		{
			if (oilDensityInWaterColumn>(*fContourLevelsH)[j] && (j==numLevels-1 || oilDensityInWaterColumn <= (*fContourLevelsH)[j+1]))
			{
				//fTriAreaArray[j] = fTriAreaArray[j] + triArea/1000000;
				//totalLEs += numLEsInTriangle;
				//totalVol += triVol;
				//concInSelectedTriangles += oilDensityInWaterColumn;	// sum or track each one?
				if (oilDensityInWaterColumn > maxConc) 
				{
					maxConc = oilDensityInWaterColumn;
					//numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
					maxTriNum = i;
				}
			}
		}
	}
	
	*depthAtPnt = depthAtPt;
	*maxTriIndex = maxTriNum;
done:
	if(numLEsInTri) {DisposeHandle((Handle)numLEsInTri); numLEsInTri=0;}
	if(massInTriInGrams) {DisposeHandle((Handle)massInTriInGrams); massInTriInGrams=0;}
	return err;
}

//OSErr CompoundMap_c::CreateDepthSlice(TLEList *thisLEList, long triNum)	
OSErr CompoundMap_c::CreateDepthSlice(long triNum, float **depthSlice)	
//OSErr CompoundMap_c::CreateDepthSlice(long triNum, float *depthSlice)	
{
	LERec LE;
	LongPoint lp;
	long i, j, k, triIndex, numOfLEs, numLESets, numDepths, numDepths2;
	short massunits;
	double density, LEmass, depthAtPt;
	double triArea, triVol, oilDensityInWaterColumn, massInGrams;;
	//TTriGridVel3D* triGrid = GetGrid(false);	
	TTriGridVel3D* triGrid = GetGrid3D(true);	
	TDagTree *dagTree = 0;
	WorldPoint centroid = {0,0};
	double *triVolumes = 0;
	TLEList *thisLEList = 0;
	Boolean bLEsInSelectedTri = false;
	float *depthSliceArray = 0;
	OSErr err = 0;
	
	if (!triGrid) return -1;
	dagTree = triGrid -> GetDagTree();
	if(!dagTree) return -1;
	
	if (triNum < 0) return -1;
	
	err = triGrid->GetMaxDepthForTriangle(triNum,&depthAtPt);
	if (err) return -1;
	numDepths = floor(depthAtPt)+1;	// split into 1m increments to track vertical slice
	if (numDepths-1 == depthAtPt) numDepths -= 1;
	
	if (numDepths>0)
	{
		triVolumes = new double[numDepths];
		if (!triVolumes) {TechError("TCompoundMap::CreateDepthSlice()", "new[]", 0); err = memFullErr; goto done;}
		
		if (*depthSlice)
		{delete [] *depthSlice; *depthSlice = 0;}
		
		//if (depthSlice)
		//{delete [] depthSlice; depthSlice = 0;}
		
		//if (depthSliceArray)
		//{delete [] depthSliceArray; depthSliceArray = 0;}
		
		depthSliceArray = new float[numDepths+1];
		if (!depthSliceArray) {TechError("TCompoundMap::CreateDepthSlice()", "new[]", 0); err = memFullErr; goto done;}
		
		depthSliceArray[0]=numDepths;	//store size here, maybe store triNum too
		for (j=0;j<numDepths;j++)
		{
			depthSliceArray[j+1]=0;
			err = triGrid->CalculateDepthSliceVolume(&triVol, triNum, j, j+1);
			if (!err && triVol>0) triVolumes[j] = triVol; else {err = -1; goto done;}
		}
		// code goes here, loop over all LELists
		numLESets = model->LESetsList->GetItemCount();
		for (k = 0; k < numLESets; k++)
		{
			model -> LESetsList -> GetListItem ((Ptr) &thisLEList, k);
			if (thisLEList->fLeType == UNCERTAINTY_LE)	
				continue;	// don't draw uncertainty for now...
			if (! ((*(dynamic_cast<TOLEList*>(thisLEList))).fDispersantData.bDisperseOil && model->GetModelTime() - model->GetStartTime() >= (*(dynamic_cast<TOLEList*>(thisLEList))).fDispersantData.timeToDisperse
				   || (*(dynamic_cast<TOLEList*>(thisLEList))).fAdiosDataH
				   || (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.z > 0	))// for bottom spill
				continue;	// this list has no subsurface LEs
			
			numOfLEs = thisLEList->numOfLEs;
			// density set from API
			//density =  GetPollutantDensity(thisLEList->GetOilType());	
			density = ((dynamic_cast<TOLEList*>(thisLEList)))->fSetSummary.density;	
			massunits = thisLEList->GetMassUnits();
			for (i = 0 ; i < numOfLEs ; i++) 
			{
				thisLEList -> GetLE (i, &LE);
				//if (LE.statusCode == OILSTAT_NOTRELEASED) continue;
				if (!(LE.statusCode == OILSTAT_INWATER)) continue;// Windows compiler requires extra parentheses
				lp.h = LE.p.pLong;
				lp.v = LE.p.pLat;
				triIndex = dagTree -> WhatTriAmIIn(lp);	
				if (!(triIndex==triNum)) continue;	// compare to selected tri
				//massunits = thisLEList->GetMassUnits();
				//density =  (((TOLEList*)thisLEList)->fSetSummary).density;	// density set from API
				//LEmass =  (((TOLEList*)thisLEList)->fSetSummary).totalMass / (double)(((TOLEList*)thisLEList)->fSetSummary).numOfLEs;	
				LEmass =  GetLEMass(LE); // will only vary for chemical with different release end time
				//if (LE.pollutantType == CHEMICAL) LEmass = GetLEMass(LE);
				massInGrams = VolumeMassToGrams(LEmass, density, massunits);
				//triArea = (triGrid -> GetTriArea(triIndex)) * 1000 * 1000;	// convert to meters
				//triVol = triArea*1.; 	//for now always 1m depth intervals, except possibly the last one...
				// code goes here, need to deal with non-uniform volume once depth of shallowest vertex is reached
				//oilDensityInWaterColumn = 1. * massInGrams / triVol; // units? milligrams/liter ?? for now gm/m^3
				for (j=0;j<numDepths;j++)
				{
					//if (LE.z>j && LE.z<=j+1) fDepthSliceArray[j+1]++;
					if (LE.z>j && (LE.z<=j+1 || j==numDepths-1)) 
					{
						oilDensityInWaterColumn = 1. * massInGrams / triVolumes[j]; // units? milligrams/liter ?? for now gm/m^3
						depthSliceArray[j+1]+= oilDensityInWaterColumn;
						bLEsInSelectedTri = true;
					}
					// include the LEs that are below the centroid depth, but don't change the volume for now
				}
			}
			//if ((numDepths-1) < depthAtPt) fDepthSliceArray[numDepths] = fDepthSliceArray[numDepths] / (depthAtPt - (numDepths-1));
		}
	}
done:
	//(*depthSlice) = depthSliceArray;
	if (triVolumes) delete [] triVolumes; triVolumes = 0;
	if (!bLEsInSelectedTri) 
	{
		if (depthSliceArray)
		{delete [] depthSliceArray; depthSliceArray = 0;}
		return -1;
	}
	(*depthSlice) = depthSliceArray;
	//depthSlice = depthSliceArray;
	return err;
}
long CompoundMap_c::CountLEsOnSelectedBeach()
{
	long i,j,k,c,n,numLEs = 0,numBoundaryPts = GetNumBoundaryPts(), dLat, dLong;
	long thisLELat, thisLELong, startPtLat, startPtLong, endPtLat, endPtLong,segno,index;
	long nSegs = GetNumBoundarySegs();	
	long theSeg,segStartver,segEndver;
	float d =0.,distToSeg;
	LERec thisLE;
	TLEList	*thisLEList;
	LETYPE leType;
	LongPointHdl ptsHdl = GetPointsHdl(false);	
	LongPoint startPt,endPt,beachedPt;
	long numBeachedLEs=0, numSelSegs = 0, startver, endver, closestSeg,triIndex;
	double segLengthInKm;
	Boolean firstTimeThrough = true;
	WorldPoint segStart, segEnd;
	char msg[64];
	OiledShorelineDataHdl oiledShorelineHdl = 0;
	OiledShorelineData data;
	OSErr err = 0, triErr = 0;
	TDagTree *dagTree = 0;
	TTriGridVel3D* triGrid = GetGrid3D(true);	
	double density,massFrac,amtBeachedOnSegment,LEMass;
	short massunits;
	long p,afterP;
	long numSelectedSegs=0;
	
	if (!triGrid) return -1; // some error alert, no depth info to check
	
	dagTree = triGrid -> GetDagTree();
	if(!dagTree)	return -1;
	
	oiledShorelineHdl = (OiledShorelineDataHdl)_NewHandleClear(sizeof(OiledShorelineData)*numBoundaryPts);
	if(!oiledShorelineHdl) {TechError("TCompoundMap::CountLEsOnSelectedBeach()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
	
	// code goes here, clean this up
	// fill all selected segments even if no beached LEs, do that first, then add to the numLEs
	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		segStartver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		segEndver = (*fBoundarySegmentsH)[theSeg]+1;
		for(j = segStartver + 1; j < segEndver; j++)
		{
 			if (INDEXH(fSelectedBeachFlagHdl,j)==1)	// endver is what marks the segment as selected
			{
				endver = j;
				startver = j-1;
				startPt = INDEXH(ptsHdl,startver);
				endPt = INDEXH(ptsHdl,endver);
				segStart.pLat = startPt.v;
				segStart.pLong = startPt.h;
				segEnd.pLat = endPt.v;
				segEnd.pLong = endPt.h;
				segLengthInKm = DistanceBetweenWorldPoints(segStart, segEnd);
				segno = WhichSelectedSegAmIIn(j);
				//data.segNo = theSeg; 
				data.segNo = segno; 
				data.startPt = startver; 
				data.endPt = endver; 
				data.numBeachedLEs = 0; 
				data.segmentLengthInKm = segLengthInKm; 
				data.gallonsOnSegment = 0;
				INDEXH(oiledShorelineHdl,endver) = data;
			}
		}
		if (INDEXH(fSelectedBeachFlagHdl,segStartver)==1)	// endver is what marks the segment as selected
		{
			endver = segStartver;
			startver = segEndver-1;
			startPt = INDEXH(ptsHdl,startver);
			endPt = INDEXH(ptsHdl,endver);
			segStart.pLat = startPt.v;
			segStart.pLong = startPt.h;
			segEnd.pLat = endPt.v;
			segEnd.pLong = endPt.h;
			segLengthInKm = DistanceBetweenWorldPoints(segStart, segEnd);
			segno = WhichSelectedSegAmIIn(segStartver);
			data.segNo = segno; 
			//data.segNo = theSeg; 
			data.startPt = startver; 
			data.endPt = endver; 
			data.numBeachedLEs = 0; 
			data.segmentLengthInKm = segLengthInKm; 
			data.gallonsOnSegment = 0;
			INDEXH(oiledShorelineHdl,endver) = data;
		}
	}
	
	// for each beached le find a corresponding segment, then see if this is selected
	// code goes here, find out what triangle beached point is in and get segment from that, if can't find then try second option
	for (i = 0, n = model -> LESetsList -> GetItemCount (); i < n; i++)
	{
		model -> LESetsList -> GetListItem ((Ptr) &thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE && !model->IsUncertain()) continue; //JLM 9/10/98
		density = ((dynamic_cast<TOLEList*>(thisLEList)))->fSetSummary.density;	
		massunits = thisLEList->GetMassUnits();
		massFrac = thisLEList->GetTotalMass()/thisLEList->GetNumOfLEs();
		for (j = 0, c = thisLEList -> numOfLEs; j < c; j++)
		{
			thisLEList -> GetLE (j, &thisLE);
			if (thisLE.statusCode == OILSTAT_ONLAND)
			{
				// for each selected boundary
				beachedPt.h = thisLE.p.pLong;
				beachedPt.v = thisLE.p.pLat;
				triIndex = dagTree -> WhatTriAmIIn(beachedPt);
				if (triIndex>=0)
				{
					TopologyHdl topH ;
					long adjTri1,adjTri2,adjTri3,vertex1,vertex2,vertex3,lastBoundaryVer,index1=-1,index2=-1;
					
					topH = dagTree->GetTopologyHdl();
					if (!topH) {return -1;/*triErr=-1;*/}
					triErr=0;
					adjTri1 = (*topH)[triIndex].adjTri1;
					adjTri2 = (*topH)[triIndex].adjTri2;
					adjTri3 = (*topH)[triIndex].adjTri3;
					vertex1 = (*topH)[triIndex].vertex1;
					vertex2 = (*topH)[triIndex].vertex2;
					vertex3 = (*topH)[triIndex].vertex3;
					lastBoundaryVer = GetNumBoundaryPts();
					if (vertex1 < lastBoundaryVer)
						index1=vertex1;
					if (vertex2 < lastBoundaryVer)
					{
						if (index1==-1) index1=vertex2; else index2 = vertex2;
					}
					if (vertex3 < lastBoundaryVer)
					{
						if (index1==-1) index1=vertex3; else if (index2==-1) index2 = vertex3; else triErr = -1;
					}
					if (index1==-1) triErr=-1;
					if (index2==-1)
					{
						triErr=-1;
					}
					if (triErr==0)
					{
						long segNo, firstPtOnSeg, lastPtOnSeg;
						//p = (*fSelectedBeachHdl)[index1];
						//afterP = (*fSelectedBeachHdl)[index2];
						segNo = PointOnWhichSeg(index1);
						firstPtOnSeg = segNo == 0 ? 0: (*fBoundarySegmentsH)[segNo-1] + 1;
						lastPtOnSeg = (*fBoundarySegmentsH)[segNo];
						//if ((p<afterP && !(p==firstPtOnSeg && afterP==lastPtOnSeg)) || (afterP==firstPtOnSeg && p==lastPtOnSeg))
						if ((index1<index2 && !(index1==firstPtOnSeg && index2==lastPtOnSeg)) || (index2==firstPtOnSeg && index1==lastPtOnSeg))
						{
							if( (*fSelectedBeachFlagHdl)[index2]==1 )
							{
								numLEs++;
								// store this information
								startPt = INDEXH(ptsHdl,index1);
								endPt = INDEXH(ptsHdl,index2);
								segStart.pLat = startPt.v;
								segStart.pLong = startPt.h;
								segEnd.pLat = endPt.v;
								segEnd.pLong = endPt.h;
								segLengthInKm = DistanceBetweenWorldPoints(segStart, segEnd);
								segno = WhichSelectedSegAmIIn(index2);
								//(*oiledShorelineHdl)[endver].segNo = closestSeg; 
								(*oiledShorelineHdl)[index2].segNo = segno; 
								(*oiledShorelineHdl)[index2].startPt = index1; 
								(*oiledShorelineHdl)[index2].endPt = index2; 
								(*oiledShorelineHdl)[index2].numBeachedLEs++; 
								(*oiledShorelineHdl)[index2].segmentLengthInKm = segLengthInKm; 
								LEMass = GetLEMass(thisLE);
								//amtBeachedOnSegment = VolumeMassToVolumeMass(1*massFrac,density,massunits,GALLONS);	// a single LE in gallons
								amtBeachedOnSegment = VolumeMassToVolumeMass(1*LEMass,density,massunits,GALLONS);	// a single LE in gallons
								(*oiledShorelineHdl)[index2].gallonsOnSegment += amtBeachedOnSegment; 
							}
						}
						//else if ((p>afterP && !(afterP==firstPtOnSeg && p==lastPtOnSeg)) || (p==firstPtOnSeg && afterP==lastPtOnSeg))
						else if ((index1>index2 && !(index2==firstPtOnSeg && index1==lastPtOnSeg)) || (index1==firstPtOnSeg && index2==lastPtOnSeg))
						{
							if( (*fSelectedBeachFlagHdl)[index1]==1 )
							{
								numLEs++;
								// store this information
								startPt = INDEXH(ptsHdl,index2);
								endPt = INDEXH(ptsHdl,index1);
								segStart.pLat = startPt.v;
								segStart.pLong = startPt.h;
								segEnd.pLat = endPt.v;
								segEnd.pLong = endPt.h;
								segLengthInKm = DistanceBetweenWorldPoints(segStart, segEnd);
								segno = WhichSelectedSegAmIIn(index1);
								//(*oiledShorelineHdl)[endver].segNo = closestSeg; 
								(*oiledShorelineHdl)[index1].segNo = segno; 
								(*oiledShorelineHdl)[index1].startPt = index2; 
								(*oiledShorelineHdl)[index1].endPt = index1; 
								(*oiledShorelineHdl)[index1].numBeachedLEs++; 
								(*oiledShorelineHdl)[index1].segmentLengthInKm = segLengthInKm; 
								LEMass = GetLEMass(thisLE);
								//amtBeachedOnSegment = VolumeMassToVolumeMass(1*massFrac,density,massunits,GALLONS);	// a single LE in gallons
								amtBeachedOnSegment = VolumeMassToVolumeMass(1*LEMass,density,massunits,GALLONS);	// a single LE in gallons
								(*oiledShorelineHdl)[index1].gallonsOnSegment += amtBeachedOnSegment; 
							}
						}
					}
					//if ((*topoHdl)[i].adjTri3 < i)
					
				}
				else {triErr=-1;}
				
				numBeachedLEs++;
				closestSeg = PointOnWhichSeg(thisLE.p.pLong, thisLE.p.pLat, &startver, &endver, &distToSeg);
				/*if (closestSeg<0)
				 {	// this should be an error
				 numBeachedLEs--;
				 continue;
				 }*/
				// check here to see if no seg returned
				if (triErr==-1 && INDEXH(fSelectedBeachFlagHdl,endver)==1)	// endver is what marks the segment as selected
				{
					numLEs++;
					// store this information
					startPt = INDEXH(ptsHdl,startver);
					endPt = INDEXH(ptsHdl,endver);
					segStart.pLat = startPt.v;
					segStart.pLong = startPt.h;
					segEnd.pLat = endPt.v;
					segEnd.pLong = endPt.h;
					segLengthInKm = DistanceBetweenWorldPoints(segStart, segEnd);
					// should do this later in a separate loop
					segno = WhichSelectedSegAmIIn(endver);
					//(*oiledShorelineHdl)[endver].segNo = closestSeg; 
					(*oiledShorelineHdl)[endver].segNo = segno; 
					(*oiledShorelineHdl)[endver].startPt = startver; 
					(*oiledShorelineHdl)[endver].endPt = endver; 
					(*oiledShorelineHdl)[endver].numBeachedLEs++; 
					(*oiledShorelineHdl)[endver].segmentLengthInKm = segLengthInKm; 
					LEMass = GetLEMass(thisLE);
					//amtBeachedOnSegment = VolumeMassToVolumeMass(1*massFrac,density,massunits,GALLONS);	// a single LE in gallons
					amtBeachedOnSegment = VolumeMassToVolumeMass(1*LEMass,density,massunits,GALLONS);	// a single LE in gallons
					(*oiledShorelineHdl)[endver].gallonsOnSegment += amtBeachedOnSegment; 
				}
				firstTimeThrough = false;
			}
			else
				continue;
		}
	}
	for (i=0;i<numBoundaryPts;i++)
	{
		data = INDEXH(oiledShorelineHdl,i);
		//endPt = data.endPt;
		//if (INDEXH(fSelectedBeachFlagHdl,endPt)==0) continue;
		if (INDEXH(fSelectedBeachFlagHdl,i)==0) continue;
		INDEXH(oiledShorelineHdl,numSelectedSegs) = data;
		numSelectedSegs++;
	}
	_SetHandleSize((Handle)oiledShorelineHdl,numSelectedSegs*sizeof(OiledShorelineData));
	if (_MemError()) { TechError("CountLEsOnSelectedBeach()", "_SetHandleSize()", 0); return -1; }
	//if (numLEs>0) err = ExportOiledShorelineData(oiledShorelineHdl);	// this call should be moved outside of this function
	//if (numLEs>0) err = OiledShorelineTable(oiledShorelineHdl);
	//#ifdef MAC
	if (numLEs>0) err = OSPlotDialog(oiledShorelineHdl);
	//#endif
	sprintf(msg,"Number of LEs beached on selected segment = %ld, %ld", numLEs, numBeachedLEs);
	// bring up a graph here, or output a table, need segment lengths too
	// make another dialog similar to plotdialog but simpler, still use the graphing functions
	printNote(msg);
	
done:
	if (err)
	{
	}
	if(oiledShorelineHdl) {DisposeHandle((Handle)oiledShorelineHdl); oiledShorelineHdl=0;}	// may want to save this to draw or whatever
	return numLEs;
}


/*void CompoundMap_c::AddSegmentToSegHdl(long startno)
 {
 AppendToLONGH(&fSegSelectedH,startno);
 return;
 }*/
/////////////////////////////////////////////////

/////////////////////////////////////////////////
