/*
 *  VectMap_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "VectMap_c.h"
#include "MemUtils.h"

#ifndef pyGNOME
#include "TModel.h"
#include "TMover.h"
#include "TCATSMover.h"
#include "TMap.h"
extern TModel *model;
#else
#include "Replacements.h"
#endif

VectorMap_c::VectorMap_c (char* name, WorldRect bounds): Map_c(name, bounds)
{
	thisMapLayer = nil;
	allowableSpillLayer = nil;
	mapBoundsLayer = nil;
	esiMapLayer = nil;
	map = nil;
	bDrawLandBitMap = false;
	bDrawAllowableSpillBitMap = false;
	bSpillableAreaActive = true;
	bSpillableAreaOpen = false;
	
	fExtendedMapBounds = bounds;
	fUseExtendedBounds = false;
	
	bShowLegend = false;
	bDrawESIBitMap = false;
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
	
	//fBitMapResMultiple = 1;

	return;
}

long VectorMap_c::GetLandType(WorldPoint p)
{
	return LT_WATER;
}

/////////////////////////////////////////////////

double DistFromWPointToSegmentDouble(long pLong, long pLat, long long1, long lat1, 
									 long long2, long lat2, long dLong, long dLat)
{
	double a, b, x, y, h, dist, numer;
	WorldPoint p;
	
	if (long1 < long2) { if (pLong < (long1 - dLong) ||
							 pLong > (long2 + dLong)) return -1; }
	else			   { if (pLong < (long2 - dLong) ||
							 pLong > (long1 + dLong)) return -1; }
	
	if (lat1 < lat2) { if (pLat < (lat1 - dLat) ||
						   pLat > (lat2 + dLat)) return -1; }
	else			 { if (pLat < (lat2 - dLat) ||
						   pLat > (lat1 + dLat)) return -1; }
	
	p.pLong = pLong;
	p.pLat = pLat;
	
	// translate origin to start of segment
	
	a = LongToDistance(long2 - long1, p);
	b = LatToDistance(lat2 - lat1);
	x = LongToDistance(pLong - long1, p);
	y = LatToDistance(pLat - lat1);
	h = sqrt(a * a + b * b);
	
	// distance from point to segment
	numer = abs(a * y - b * x);
	dist = numer / h;
	return dist;
}

long CheckShoreline (long longVal, long latVal,CMapLayer *mapLayer)
{
	long	ObjectIndex, ObjectCount;
	ObjectRecHdl	thisObjectHdl = nil;
	CMyList	*thisObjectList = nil;
	long	PointCount, PointIndex;
	LongPoint	MatrixPt1, MatrixPt2;
	LongPoint	**RgnPtsHdl;
	long 	x1, y1, x2, y2;
	long dLong, dLat, objectNum=-1;
	double dist, smallestDistWithinPolygon = 100.,smallestDist = 100;
	long oneSecond = (1000000/3600); // map border is several pixels wide
	//dLong = dLat = oneSecond * 50;
	//dLong = dLat = oneSecond * 5;
	dLong = dLat = oneSecond / 5;
	//dLong = dLat = oneSecond;	// need to figure how to set this
	//dLong = dLat = 0;
	long theESICode = 0; // default value
	
	thisObjectList = mapLayer->GetLayerObjectList (); //- if don't put into CMapLayer
	//thisObjectList = GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = 0; ObjectIndex < ObjectCount; ObjectIndex++)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		//PointCount = GetPolyPointCount ((PolyObjectHdl) thisObjectHdl); AH 03/16/2012
		//RgnPtsHdl = GetPolyPointsHdl ((PolyObjectHdl) thisObjectHdl); AH 03/16/2012
		
		PointCount = (**((PolyObjectHdl)thisObjectHdl)).pointCount;	// AH 03/16/2012
		RgnPtsHdl = (LongPoint **) ((**thisObjectHdl).objectDataHdl); // AH 03/16/2012
		
		if (RgnPtsHdl != nil)
		{
			if (ObjectIndex>201)
			{
				dist = -1;
			}
			
			// send in point to compare against
			for (PointIndex = 0;PointIndex < PointCount-1; ++PointIndex)
			{
				MatrixPt1 = (*RgnPtsHdl) [PointIndex];
				MatrixPt2 = (*RgnPtsHdl) [PointIndex+1];
				
				x1 = MatrixPt1.h;
				y1 = MatrixPt1.v;
				x2 = MatrixPt2.h;
				y2 = MatrixPt2.v;
				
				dist = DistFromWPointToSegmentDouble(longVal, latVal, x1, y1, x2, y2, dLong, dLat);
				if (dist==-1) continue;	// not within range
				
				if (dist<smallestDistWithinPolygon)
				{
					smallestDistWithinPolygon = dist;
				}
			}
		}
		if (smallestDistWithinPolygon<smallestDist)
		{
			smallestDist = smallestDistWithinPolygon;
			objectNum = ObjectIndex;
		}
	}
	if (objectNum>=0)
	{	
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, objectNum);
		//GetObjectESICode (thisObjectHdl, &theESICode); AH 03/16/2012
		theESICode = (**thisObjectHdl).objectESICode;	// AH 03/16/2012
	}
	return theESICode; // need an error code?
}

float GetRefloatTimeInHoursFromESICode(long code)
{
	switch(code) 
	{	// will have 16 codes corresponding to all the colors
			// need to decide what refloat goes with each ESI code
		case 1: return 0;
			//case 1: return 8760;
			//case 2: return 1;
			//case 3: return kRedColorInd;
			//case 3: return 100;
			//case 3: return 8760;
		case 3: return 0;
			//case 4: return 3600;
			//case 5: return 1;
			//case 6: return 1;
			//case 7: return 8760;
		case 7: return 0;
			//case 7: return 1;
			//case 8: return 1;
			//case 9: return 8760;
		case 9: return 0;
			//case 10: return 8760;
			//case 10: return 0;
	}
	//return 1; // should have a default, maybe fRefloat?
	return 8760; // should have a default, maybe fRefloat?
}

float VectorMap_c::RefloatHalfLifeInHrs(WorldPoint wp) 
{ 
	float refloatHalfLifeInHrs = fRefloatHalfLifeInHrs;
	long esiCode = 1;
	if (HaveESIMapLayer())	// change to ESILayer
	{
		esiCode = CheckShoreline(wp.pLong,wp.pLat,esiMapLayer);
		refloatHalfLifeInHrs = GetRefloatTimeInHoursFromESICode(esiCode);
	}
	return refloatHalfLifeInHrs;
}

Boolean VectorMap_c::HaveAllowableSpillLayer(void)
{
	CMyList *thisObjectList = nil;
	long numObjs;
	
	if(!this->allowableSpillLayer) return false;
	thisObjectList = this->allowableSpillLayer->GetLayerObjectList ();
	numObjs = thisObjectList -> GetItemCount ();
	if(numObjs>0) return true;
	else return false;
}

/////////////////////////////////////////////////

Boolean VectorMap_c::HaveMapBoundsLayer(void)
{
	CMyList *thisObjectList = nil;
	long numObjs;
	
	if(!this->mapBoundsLayer) return false;
	thisObjectList = this->mapBoundsLayer->GetLayerObjectList ();
	numObjs = thisObjectList -> GetItemCount ();
	if(numObjs>0) return true;
	else return false;
}

/////////////////////////////////////////////////

Boolean VectorMap_c::HaveLandWaterLayer(void)
{
	CMyList *thisObjectList = nil;
	long numObjs;
	
	if(!this->thisMapLayer) return false;
	thisObjectList = this->thisMapLayer->GetLayerObjectList ();
	numObjs = thisObjectList -> GetItemCount ();
	if(numObjs>0) return true;
	else return false;
}

/////////////////////////////////////////////////

Boolean VectorMap_c::HaveESIMapLayer(void)
{
	CMyList *thisObjectList = nil;
	long numObjs;
	
	if(!this->esiMapLayer) return false;
	thisObjectList = this->esiMapLayer->GetLayerObjectList ();
	numObjs = thisObjectList -> GetItemCount ();
	if(numObjs>0) return true;
	else return false;
}

TMover* VectorMap_c::GetMover(ClassID desiredClassID)
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

/////////////////////////////////////////////////
TTriGridVel* VectorMap_c::GetGrid()
{
	TTriGridVel* triGrid = 0;	
	TMover *mover = 0;
	
	// Figure out if this map has a TCATSMover current
	mover = this->GetMover(TYPE_CATSMOVER);	// get first one, assume all grids are the same
	if (mover)
	{
		triGrid = dynamic_cast<TTriGridVel*>(((dynamic_cast<TCATSMover *>(mover)) -> fGrid));
	}
	return triGrid;
}

double VectorMap_c::DepthAtPoint(WorldPoint wp)
{
	float depth1,depth2,depth3;
	double depthAtPoint = 0;	
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	// don't use refined grid, depths aren't refined
	
	if (!triGrid) return 0; // some error alert, no depth info to check
	interpolationVal = triGrid->GetInterpolationValues(wp);
	if (interpolationVal.ptIndex1<0)	// couldn't find point in dag tree
	{
		return 0;
	}
	depthsHdl = triGrid->GetBathymetry();
	if (!depthsHdl) return 0;	// some error alert, no depth info to check
	
	depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
	depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
	depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
	depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;

	return depthAtPoint;
}

VectorMap_c* GetNthVectorMap(long desiredNum0Relative)
{
	long i,n;
	TMap *map;
	long numVectorMaps = 0;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_VECTORMAP)) 
		{
			if(desiredNum0Relative == numVectorMaps)
				return dynamic_cast<VectorMap_c*>(map);
			numVectorMaps++;
		}
	}
	return nil;
}

