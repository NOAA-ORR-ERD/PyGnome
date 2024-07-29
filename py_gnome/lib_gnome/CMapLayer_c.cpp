/*
 *  CMapLayer_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "MemUtils.h"
#ifndef pyGNOME
#include "ObjectUtilsPD.h"
#else
#include "ObjectUtils.h"
#endif

/**************************************************************************************************/
void GetObjectType (ObjectRecHdl ObjectHdl, OSType *ObjectTypePtr)
{
	*ObjectTypePtr = (**ObjectHdl).objectType;
	
	return;
}
/**************************************************************************************************/
long GetPolyPointCount (PolyObjectHdl thePolyHdl)
{
	if (thePolyHdl != nil)
		return ((**thePolyHdl).pointCount);
	else
		return (0);
}
/**************************************************************************************************/
void GetObjectLRect (ObjectRecHdl ObjectHdl, LongRect *ObjectLRectPtr)
{
	*ObjectLRectPtr = (**ObjectHdl).objectLRect;
	
	return;
}
/**************************************************************************************************/
CMyList* CMapLayer_c::GetLayerObjectList ()
{
	CMyList		*thisObjectList = nil;
	
	thisObjectList = layerObjectList;
	
	return (thisObjectList);
}
/**************************************************************************************************/
CMyList* CMapLayer_c::GetLayerGroupList ()
{
	CMyList		*thisGroupList = nil;
	
	thisGroupList = layerGroupList;
	
	return (thisGroupList);
}
/**************************************************************************************************/
long CMapLayer_c::GetLayerObjectCount ()
{
	long			ObjectCount = 0;
	CMyList			*thisObjectList = nil;
	
	thisObjectList = GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();
	
	return (ObjectCount);
}

/**************************************************************************************************/
long CMapLayer_c::GetLayerPolyPtsCount ()
{
	long			ObjectIndex, ObjectCount, TotalPointsCount = 0;
	CMyList			*thisObjectList = nil;
	ObjectRecHdl	thisObjectHdl;
	OSType			thisObjectType;
	
	thisObjectList = GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		GetObjectType (thisObjectHdl, &thisObjectType);
		if (thisObjectType == kPolyType)
			TotalPointsCount  += GetPolyPointCount ((PolyObjectHdl) thisObjectHdl);
	}
	
	return (TotalPointsCount);
}
/**************************************************************************************************/
void CMapLayer_c::GetLayerScope (LongRect *LayerLBoundsPtr, Boolean bRecalc)
{
	if (!bRecalc)
		*LayerLBoundsPtr = layerScopeLRect;
	else
	{
		long			ObjectCount, ObjectIndex;
		LongRect		ObjectLRect, layerBoundsLRect;
		ObjectRecHdl	thisObjectHdl;
		CMyList			*thisObjectList = nil;
		
		/* set layer bounds rect to empty to start with */
		SetLRect (&layerBoundsLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);
		
		thisObjectList = GetLayerObjectList ();
		ObjectCount = thisObjectList -> GetItemCount ();
		
		for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
		{
			thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
			GetObjectLRect (thisObjectHdl, &ObjectLRect);
			
			UnionLRect (&ObjectLRect, &layerBoundsLRect, &layerBoundsLRect);
		}
		
		layerScopeLRect = layerBoundsLRect;		/* update internal class field */
		*LayerLBoundsPtr = layerBoundsLRect;	/* send LRect back */
	}
	
	return;
}

/**************************************************************************************************/
Boolean CMapLayer_c::IsLayerModified ()
{
	return (bLayerModified);
}