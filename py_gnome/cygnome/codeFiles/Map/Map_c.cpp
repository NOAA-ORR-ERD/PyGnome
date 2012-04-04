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

Map_c::Map_c(char *name, WorldRect bounds) 
{
//	SetMapName(name);
	fMapBounds = bounds;
	
	moverList = 0;
	
//	SetDirty(FALSE);
	
	bOpen = TRUE;
	bMoversOpen = TRUE;
	
	fRefloatHalfLifeInHrs = 1.0;
	
	bIAmPartOfACompoundMap = false;
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