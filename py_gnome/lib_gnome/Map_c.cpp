/*
 *  Map_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Map_c.h"
#include "GEOMETRY.H"

#ifdef pyGNOME
#define TMover Mover_c
#define TechError(a, b, c) printf(a)
#else
#include "CROSS.H"
#endif

Map_c::Map_c(char *name, WorldRect bounds) 
{
	SetMapName(name);
	fMapBounds = bounds;
	
	moverList = 0;
#ifndef pyGNOME	
//	SetDirty(FALSE);
#endif
	bOpen = TRUE;
	bMoversOpen = TRUE;
	
	fRefloatHalfLifeInHrs = 1.0;
	
	bIAmPartOfACompoundMap = false;
}

OSErr Map_c::AddMover(TMover *theMover, short where)
{
	OSErr err = 0;
	if (!moverList) return -1;
	
	if (err = moverList->AppendItem((Ptr)&theMover))
	{ TechError("TMap::AddMover()", "AppendItem()", err); return err; }
#ifndef pyGNOME	
	SetDirty (true);
	
	SelectListItemOfOwner(theMover);
#endif
	return 0;
}
Boolean Map_c::InMap(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	return WPointInWRect(p.pLong, p.pLat, &ourBounds);
}

WorldPoint3D Map_c::MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// this is a primitive check that is fairly fast but can jump land.
	
	if (this -> InMap (toWPt.p) && this -> OnLand (toWPt.p))
	{
		// beach near shore by using binary search for land
		
		WorldPoint3D  midPt;
		Map_c *midPtBestMap = this;
#define BEACHINGDISTANCE 0.05 // in kilometers 
		// fromWPt is the water point
		// toWPt is the land point
		double distanceInKm = DistanceBetweenWorldPoints(fromWPt.p,toWPt.p);
		while(distanceInKm > BEACHINGDISTANCE)
		{
			midPt.p.pLong = (fromWPt.p.pLong + toWPt.p.pLong)/2;
			midPt.p.pLat = (fromWPt.p.pLat + toWPt.p.pLat)/2;
			midPt.z = (fromWPt.z + toWPt.z)/2;
			if (!this -> InMap (midPt.p))
			{	// unusual case, the midPt is not on this map
				// this usually means it is coming onto this map
				// from another map.  
				// We will assume that map has checked the movement and it is OK
				// so we assume the midPt is a water point
				fromWPt = midPt;// midpt is water
			}
			else
			{
				if (midPtBestMap -> OnLand (midPt.p)) 
					toWPt = midPt;
				else 
					fromWPt = midPt;// midpt is water
			}
			distanceInKm = DistanceBetweenWorldPoints(fromWPt.p,toWPt.p);
		}
	}
	return toWPt;
}

Boolean Map_c::IsAllowableSpillPoint(WorldPoint p)
{
	if(!this->InMap(p)) return false;// not on this map
	if(this->OnLand(p)) return false;// on land
	return true; // a water point
}


OSErr Map_c::InitMap()
{
	OSErr err = 0;
	moverList = new CMyList(sizeof(TMover *));
	if (!moverList)
	{ TechError("TMap::InitMap()", "new CMyList()", 0); return -1; }
	if (err = moverList->IList())
	{ TechError("TMap::InitMap()", "IList()", 0); return -1; }
	
	return 0;
}
