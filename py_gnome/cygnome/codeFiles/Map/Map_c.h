/*
 *  Map_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Map_c__
#define __Map_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "ClassID/ClassID_c.h"
#include "CMyList.h"

class Map_c : virtual public ClassID_c {

public:
	WorldRect			fMapBounds; 				// bounding rectangle of map
	CMyList				*moverList;					// list of this map's movers
	Boolean				bMoversOpen;				// movers list open (display)	
	float				fRefloatHalfLifeInHrs;	
	Boolean				bIAmPartOfACompoundMap;

public:
	Map_c() {}
	Map_c(char *name, WorldRect bounds);
	virtual ClassID	GetClassID () { return TYPE_MAP; }
	virtual Boolean	IAm(ClassID id) {if(id==TYPE_MAP) return TRUE; return ClassID_c::IAm(id); }
	virtual	WorldRect	GetMapBounds () { return fMapBounds; }
	void				SetMapBounds (WorldRect newBounds) { fMapBounds = newBounds; }
	virtual	Boolean 	InMap (WorldPoint p);
		
	virtual Boolean		OnLand (WorldPoint p) { return false; }
	virtual float		RefloatHalfLifeInHrs(WorldPoint p) { return fRefloatHalfLifeInHrs;}
	virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	virtual Boolean 	IsAllowableSpillPoint(WorldPoint p);
	virtual Boolean 	HaveAllowableSpillLayer(void) {return false;}
	virtual Boolean		CanReFloat (Seconds time, LERec *theLE) { return true; }
	virtual long		GetLandType (WorldPoint p) { return LT_WATER; }
	virtual Boolean 	HaveMapBoundsLayer(void) { return false; }
	//virtual Boolean 	IsIceMap(void) { return false; }
	
	virtual double		DepthAtPoint(WorldPoint wp) {return 0;}
	virtual	float		GetMaxDepth2(void) {return 0;}	
	
};

#endif