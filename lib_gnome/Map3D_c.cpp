/*
 *  Map3D_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Map3D_c.h"
#include "Map3D.h"
#include "CurrentMover_c.h"
#include "MemUtils.h"
#include "StringFunctions.h"
#include "CompFunctions.h"

#ifndef pyGNOME
#include "CROSS.H"
#include "TideCurCycleMover.h"
#else
#include "Replacements.h"
#endif

float DistFromWPointToSegment(long pLong, long pLat, long long1, long lat1, 
							  long long2, long lat2, long dLong, long dLat);

Map3D* CreateAndInitMap3D(char *path, WorldRect bounds)
{
	char 		nameStr[256];
	OSErr		err = noErr;
	Map3D 	*map = nil;
	char fileName[256],s[256];
	
	if (path[0])
	{
		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcpy (nameStr, "BathymetryMap: ");
		strcat (nameStr, fileName);
	}
	else
		strcpy(nameStr,"Bathymetry Map");
	map = new Map3D(nameStr, bounds);
	if (!map)
	{ TechError("AddMap3D()", "new Map3D()", 0); return nil; }
	
	if (err = map->InitMap()) { delete map; return nil; }
	
	return map;
}


Map3D_c::Map3D_c(char* name, WorldRect bounds) : Map_c(name, bounds)
{
	fGrid = 0;

	fBoundarySegmentsH = 0;
	fBoundaryTypeH = 0;
	fBoundaryPointsH = 0;

	bDrawLandBitMap = false;	// combined option for now
	bDrawWaterBitMap = false;
	
	bShowLegend = true;
	bShowGrid = false;
	bShowDepthContours = false;
	
	bDrawContours = true;
	
	fDropletSizesH = 0;
	
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
	
	fWaterDensity = 1020;
	fMixedLayerDepth = 10.;	//meters
	fBreakingWaveHeight = 1.;	// meters
	fDiagnosticStrType = 0;
	
	fMinDistOffshore = 0.;	//km - use bounds to set default
	bUseLineCrossAlgorithm = false;
	bUseSmoothing = false;
	
	fWaveHtInput = 0;	// default compute from wind speed

	fVerticalGridType = TWO_D;
	fGridType = CURVILINEAR;
}


// will need to deal with this for new curvilinear algorithm when start using subsurface movement
long Map3D_c::PointOnWhichSeg(long longVal, long latVal, long *startver, long *endver, float *distToSeg)
{
	long numSegs = GetNumBoundarySegs(), jseg;
	long firstPoint, lastPoint, segNo, endPt, x1, y1, x2, y2, closestSeg = -1;
	
	long dLong, dLat;
	float dist, smallestDist = 100.;
	long oneSecond = (1000000/3600); // map border is several pixels wide
	//long oneSecond = 0;
	
	LongPointHdl ptsHdl = GetPointsHdl();	
	if(!ptsHdl) return -1;
	//dLong = dLat = oneSecond * 5;
	dLong = dLat = oneSecond * 50;
	*distToSeg = -1;
	
	// to support new curvilinear algorithm
	if (fBoundaryPointsH)
	{
		//long theSeg,startver,endver,j,index1;
		long index,index1;
		//return PointOnWhichSeg2();
		for(jseg = 0; jseg < numSegs; jseg++)
		{
			firstPoint = jseg == 0? 0: (*fBoundarySegmentsH)[jseg-1] + 1;
			lastPoint = (*fBoundarySegmentsH)[jseg]+1;
			//index1 = (*fBoundaryPointsH)[startver];
			//pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
			//MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			for(segNo = firstPoint; segNo < lastPoint; segNo++)
			{
				index = (*fBoundaryPointsH)[segNo];
				
				if (segNo == lastPoint-1)
					endPt = firstPoint;
				else
					endPt = segNo+1;
				index1 = (*fBoundaryPointsH)[endPt];
				x1 = (*ptsHdl)[index].h;
				y1 = (*ptsHdl)[index].v;
				x2 = (*ptsHdl)[index1].h;
				y2 = (*ptsHdl)[index1].v;
				
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
	}
	else {
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
	}
	return closestSeg;
}

TMover* Map3D_c::GetMover(ClassID desiredClassID)
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

Boolean Map3D_c::ThereIsADispersedSpill()
{
	long i, n;
	TLEList *thisLEList;
	LETYPE leType;
	
	// also need to check if spill is going to be dispersed, go through all spills
	// actually this is now the only way spill can get below surface
	for (i = 0, n = model->LESetsList->GetItemCount(); i < n; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE /*&& !this->IsUncertain()*/) continue;
		if ((*(dynamic_cast<TOLEList*>(thisLEList))).fDispersantData.bDisperseOil || (*(dynamic_cast<TOLEList*>(thisLEList))).fAdiosDataH)
			return true;
		// will need to consider spill set below the surface
		if ((*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.z > 0)
			return true;
	}
	return false;
}

double Map3D_c::GetSpillStartDepth()
{
	long i, n;
	TLEList *thisLEList;
	LETYPE leType;
	double spillStartDepth = 0.;
	
	// also need to check if spill is going to be dispersed, go through all spills
	// actually this is now the only way spill can get below surface
	for (i = 0, n = model->LESetsList->GetItemCount(); i < n; i++) 
	{
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		leType = thisLEList -> GetLEType();
		if(leType == UNCERTAINTY_LE /*&& !this->IsUncertain()*/) continue;
		//if ((*(TOLEList*)thisLEList).fDispersantData.bDisperseOil || (*(TOLEList*)thisLEList).fAdiosDataH)
		//return true;
		// will need to consider spill set below the surface
		if ((*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.z > 0)
			return (*(dynamic_cast<TOLEList*>(thisLEList))).fSetSummary.z;
	}
	return spillStartDepth;
}


Boolean Map3D_c::CanReFloat(Seconds time, LERec *theLE) 
{ 
	//if (ThereIsADispersedSpill())
	if ((*theLE).dispersionStatus == HAVE_DISPERSED)	// these LEs shouldn't be allowed to beach in the first place...
	{
		OSErr err = 0;
		return false;
	}
	return true; 
}

TTriGridVel* Map3D_c::GetGrid()
{
	TTriGridVel* triGrid = 0;	
	
	triGrid = (TTriGridVel*)fGrid;	// are we sure this is a TriGrid?
	return triGrid;
}


TCurrentMover* Map3D_c::Get3DCurrentMover()
{
	TMover *thisMover = nil;
	long i,d;
	for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	{
		this -> moverList -> GetListItem ((Ptr) &thisMover, i);
		//classID = thisMover -> GetClassID ();
		//if(classID == desiredClassID) return thisMover(;
		//if (thisMover -> IAm(TYPE_CURRENTMOVER)) return ((TCurrentMover*)thisMover);	// show movement only handles currents, not wind and dispersion
		// might want to be specific since this could allow CATSMovers...
		if(thisMover -> IAm(TYPE_PTCURMOVER) || thisMover -> IAm(TYPE_TRICURMOVER) || thisMover -> IAm(TYPE_CATSMOVER3D)
		   || thisMover -> IAm(TYPE_NETCDFMOVERCURV) || thisMover -> IAm(TYPE_NETCDFMOVERTRI)) return dynamic_cast<TCurrentMover*>(thisMover);
	}
	return nil;
}

LongPointHdl Map3D_c::GetPointsHdl()	
{
	LongPointHdl ptsHdl = 0;
	TMover *mover=0;
	
	ptsHdl = ((TTriGridVel*)fGrid) -> GetPointsHdl();

	
	return ptsHdl;
}

/*float Map3D_c::GetTotalDepth(WorldPoint refPoint,long ptIndex)
{
	long index1, index2, index3, index4, numDepths;
	OSErr err = 0;
	float totalDepth = 0;
	Boolean useTriNum = false;
	long triNum = 0;
	
	if (fGridType == SIGMA_ROMS)
	{
		//if (triNum < 0) useTriNum = false;
		err = ((TTriGridVel*)fGrid)->GetRectCornersFromTriIndexOrPoint(&index1, &index2, &index3, &index4, refPoint, triNum, useTriNum, fVerdatToNetCDFH, fNumCols+1);
		
		if (err) return -1;
		if (fDepthsH)
		{	// issue with extended grid not having depths - probably need to rework that idea
			long numCorners = 4;
			numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
			if (index1<numDepths && index1>=0) totalDepth += INDEXH(fDepthsH,index1); else numCorners--;
			if (index2<numDepths && index2>=0) totalDepth += INDEXH(fDepthsH,index2); else numCorners--;
			if (index3<numDepths && index3>=0) totalDepth += INDEXH(fDepthsH,index3); else numCorners--;
			if (index4<numDepths && index4>=0) totalDepth += INDEXH(fDepthsH,index4); else numCorners--;
			if (numCorners>0)
				totalDepth = totalDepth/(float)numCorners;
		}
	}
	else 
	{
		if (fDepthsH) totalDepth = INDEXH(fDepthsH,ptIndex);
	}
	return totalDepth;
	
}*/

Boolean Map3D_c::InVerticalMap(WorldPoint3D wp)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	// don't use refined grid, depths aren't refined
	//TCurrentMover *mover = Get3DCurrentMover();
	
	//if (fGridType==SIGMA_ROMS)
		//depthAtPoint = (double)((NetCDFMoverCurv*)mover)->GetTotalDepth(wp.p,-1);
	//else
	{
		if (!triGrid) return false; // some error alert, no depth info to check
		interpolationVal = triGrid->GetInterpolationValues(wp.p);
		depthsHdl = triGrid->GetBathymetry();
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

double Map3D_c::DepthAtPoint(WorldPoint wp)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	// don't use refined grid, depths aren't refined
	//TCurrentMover* mover = Get3DCurrentMover();
	
	//if (mover && mover->IAm(TYPE_NETCDFMOVERCURV) && ((NetCDFMoverCurv*)mover)->fVar.gridType==SIGMA_ROMS)
		//return (double)((NetCDFMoverCurv*)mover)->GetTotalDepth(wp,-1);
	
	if (!triGrid) return -1; // some error alert, no depth info to check
	interpolationVal = triGrid->GetInterpolationValues(wp);
	depthsHdl = triGrid->GetBathymetry();
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

float Map3D_c::GetMaxDepth2(void)
{	// may want to extend for SIGMA_ROMS (all ROMS?) to check the cell depths rather than point depths
	long i,numDepths;
	float depth, maxDepth=0;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	// don't use refined grid, depths aren't refined
	
	if (!triGrid) return 0; // some error alert, no depth info to check
	
	depthsHdl = triGrid->GetBathymetry();
	if (!depthsHdl) return 0;	// some error alert, no depth info to check
	
	numDepths = _GetHandleSize((Handle)depthsHdl)/sizeof(**depthsHdl);
	for (i=0;i<numDepths;i++)
	{
		depth = INDEXH(depthsHdl,i);
		if (depth > maxDepth) 
			maxDepth = depth;
	}
	return maxDepth;
}

WorldPoint3D Map3D_c::TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint)
{
	WorldPoint3D movedPoint = {0,0,0.}, firstEndPoint = {0,0,0.}, secondEndPoint = {0,0,0.};
	WorldPoint3D testPt = {0,0,0.}, realBeachedPt = {0,0,0.};
	double alpha, sideA, sideB, sideC, sideD, shorelineLength;
	long startver, endver, x1, y1, x2, y2, testcase = 0;
	
	LongPointHdl ptsHdl = GetPointsHdl();	
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

double Map3D_c::DepthAtCentroid(long triNum)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	long ptIndex1,ptIndex2,ptIndex3;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	
	
	TopologyHdl topH ;
	
	//NetCDFMover *mover = (NetCDFMover*)(model->GetMover(TYPE_NETCDFMOVER));
	//TCurrentMover* mover = Get3DCurrentMover();
	
	//if (mover && mover->fVar.gridType==SIGMA_ROMS)
	/*if (mover && mover->IAm(TYPE_NETCDFMOVERCURV) && ((NetCDFMoverCurv*)mover)->fVar.gridType==SIGMA_ROMS)
		return (double)((NetCDFMoverCurv*)mover)->GetTotalDepthFromTriIndex(triNum);*/
	
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

WorldPoint3D Map3D_c::ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp)
{
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
		double depthAtPt = DepthAtPoint(movedPoint.p);	// code goes here, a check on return value
		if (depthAtPt <= 0) 
		{
			OSErr err = 0;
			return fromWPt;	// code goes here, may want to force the point back into map somehow
		}
		//if (depthAtPt==0)
		//movedPoint.z = .1;
		if (movedPoint.z > depthAtPt) movedPoint.z = GetRandomFloat(.9*depthAtPt,.99*depthAtPt);
		//if (movedPoint.z > depthAtPt) movedPoint.z = GetRandomFloat(.7*depthAtPt,.99*depthAtPt);
		//if (movedPoint.z <= 0) movedPoint.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
		if (movedPoint.z <= 0) 
			//movedPoint.z = GetRandomFloat(.01*depthAtPt,.1*depthAtPt);
			movedPoint.z = 0;	// let points surface or resurface (redispersing is taken care of in TRandom3D)
		//movedPoint.z = fromWPt.z;	// try not changing depth
		//if (!InVerticalMap(movedPoint))
		//movedPoint.p = fromWPt.p;	// use original point - code goes here, need to find a z in the map
	}
	return movedPoint;
}

/////////////////////////////////////////////////

double Map3D_c::GetBreakingWaveHeight(void)
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
}

long Map3D_c::GetNumBoundarySegs(void)
{
	long numInHdl = 0;
	if (fBoundarySegmentsH) numInHdl = _GetHandleSize((Handle)fBoundarySegmentsH)/sizeof(**fBoundarySegmentsH);
	
	return numInHdl;
}

long Map3D_c::GetNumPointsInBoundarySeg(long segno)
{
	if (fBoundarySegmentsH) return (*fBoundarySegmentsH)[segno] - (segno==0? 0: (*fBoundarySegmentsH)[segno-1]+1) + 1;
	else return 0;
}

long Map3D_c::GetNumBoundaryPts(void)
{
	long numInHdl = 0;
	if (fBoundaryTypeH) numInHdl = _GetHandleSize((Handle)fBoundaryTypeH)/sizeof(**fBoundaryTypeH);
	
	return numInHdl;
}

Boolean Map3D_c::IsBoundaryPoint(long pt)
{
	return pt < GetNumBoundaryPts();
}

long Map3D_c::GetNumContourLevels(void)
{
	long numInHdl = 0;
	if (fContourLevelsH) numInHdl = _GetHandleSize((Handle)fContourLevelsH)/sizeof(**fContourLevelsH);
	
	return numInHdl;
}


OSErr Map3D_c::InitMap()
{
	OSErr err = 0;
	//	code goes here, only if there is a 3D mover?
	return Map_c::InitMap();
}


void Map3D_c::FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo)
{
	long startVer = 0,i,jseg;
	//WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);
	WorldPoint wp2;
	LongPoint lp;
	long lastVer = GetNumBoundaryPts();
	//long nbounds = GetNumBoundaries();
	long nSegs = GetNumBoundarySegs();	
	float wdist = LatToDistance(ScreenToWorldDistance(4));
	LongPointHdl ptsHdl = GetPointsHdl();
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