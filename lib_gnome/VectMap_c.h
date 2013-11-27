/*
 *  VectMap_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __VectorMap_c__
#define __VectorMap_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "Map_c.h"
#include "ObjectUtils.h"

#ifndef pyGNOME
#include "TMover.h"
#include "GridVel.h"
#else
#include "TriGridVel_c.h"
#include "Mover_c.h"
#include "ObjectUtils.h"
#define TTriGridVel TriGridVel_c
#define TMover Mover_c
#define CMapLayer CMapLayer_c
#endif

class CMapLayer;

class VectorMap_c : virtual public Map_c  {

public:
	CMapLayer			*thisMapLayer;
	CMapLayer			*allowableSpillLayer;
	CMapLayer			*mapBoundsLayer;
	CMapLayer			*esiMapLayer;
	CMap				*map;

	Boolean				bDrawLandBitMap;
	Boolean				bDrawAllowableSpillBitMap;
	Boolean				bSpillableAreaOpen;
	Boolean				bSpillableAreaActive;
	
	WorldRect			fExtendedMapBounds;
	Boolean				fUseExtendedBounds;
	
	Rect				fLegendRect;
	Boolean				bShowLegend;
	Boolean				bDrawESIBitMap;
	
	//long				fBitMapResMultiple;
	
	VectorMap_c (char* name, WorldRect bounds);
	VectorMap_c () {}
	void		 		SetExtendedBounds (WorldRect newBounds) {fExtendedMapBounds = newBounds;}
	virtual WorldRect	GetMapBounds () { if (fUseExtendedBounds) return fExtendedMapBounds; else return fMapBounds; }
	//virtual ClassID 	GetClassID () { return TYPE_VECTORMAP; }
	//virtual Boolean		IAm(ClassID id) { if(id==TYPE_VECTORMAP) return TRUE; return TMap::IAm(id); }	

	virtual float		RefloatHalfLifeInHrs(WorldPoint p);
	virtual long 		GetLandType (WorldPoint p);
	virtual Boolean 	HaveAllowableSpillLayer(void);
	virtual Boolean 	HaveMapBoundsLayer(void);
	virtual	Boolean 	HaveLandWaterLayer(void);
	virtual	Boolean 	HaveESIMapLayer(void);
	//virtual	Boolean 	IsIceMap(void);

	
	TMover* 			GetMover(ClassID desiredClassID);
	TTriGridVel*		GetGrid();
	virtual	double 				DepthAtPoint(WorldPoint wp);


	
};

VectorMap_c* GetNthVectorMap(long desiredNum0Relative);

#undef TTriGridVel
#undef TMover
#endif