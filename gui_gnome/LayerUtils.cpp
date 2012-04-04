
#include "Cross.h"
#include "Classes.h"
#include "GenDefs.h"
#include "ObjectUtilsPD.h"
/**************************************************************************************************/
CMapLayer *GetMapLayer (CMyList *theLayerList, long LayerNum)
{
	CMapLayer 	*thisLayer = nil;
	OSErr		ErrCode = 0;
	
	ErrCode = theLayerList -> GetListItem ((Ptr) &thisLayer, LayerNum);
	if (ErrCode)
		thisLayer = nil;

	return (thisLayer);
}
/**************************************************************************************************/
void GetMapLBounds (CMyList *theLayerList, LongRect *MapLRectPtr)
{
	short		LayerCount, LayerIndex;
	LongRect	MapBoundsLRect, LayerBoundsLRect;
	CMapLayer	*thisMapLayer = nil;
	
	/* set map bounds to empty to start with */
	SetLRect (&MapBoundsLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);
	
	LayerCount = theLayerList -> GetItemCount ();
	for (LayerIndex = 1; LayerIndex <= LayerCount; ++LayerIndex)
	{
		thisMapLayer = GetMapLayer (theLayerList, LayerIndex);
		thisMapLayer -> GetLayerScope (&LayerBoundsLRect, true);
		UnionLRect (&MapBoundsLRect, &LayerBoundsLRect, &MapBoundsLRect);
	}

	*MapLRectPtr = MapBoundsLRect;

	return;
}
/**************************************************************************************************/
CMyList *GetNewLayerList ()
{
	OSErr	 ErrCode = 0;
	CMyList	*thisLayerList = nil;

	thisLayerList = new CMyList (sizeof (CMapLayer*));
	if (thisLayerList == nil)
	{
		ErrCode = memFullErr;
		TechError ("GetNewLayerList ()", "Out of memory while creating new map layer!", ErrCode);
	}

	return (thisLayerList);
}
/**************************************************************************************************/
void MarkAllObjects (CMyList *theLayerList, Boolean MarkFlag)
{
	short		LayerCount, LayerIndex;
	CMyList		*thisObjectList;
	CMapLayer	*thisMapLayer = nil;

	LayerCount = theLayerList -> GetItemCount ();
	for (LayerIndex = 1; LayerIndex <= LayerCount; ++LayerIndex)
	{
		/* get this layer's objects list */
		thisMapLayer = GetMapLayer (theLayerList, LayerIndex);
		if (thisMapLayer != nil)
		{
			thisObjectList = thisMapLayer -> GetLayerObjectList ();	
			MarkObjectsInList (thisObjectList, MarkFlag);
		}
	}

	return;
}
/**************************************************************************************************/
CMapLayer* AddNewLayer (CMap *theMap, CMyList *theLayerList, Boolean LayerVisFlag,
						Boolean LOLabelVisFlag)
/* this subroutine adds a new layer to the current layer list.  It also allocates a new objects
	list of the new layer.
	Note: Upon return, *layer-info-ptr contains the settings of the new layer */
{
	OSErr		ErrCode = 0;
	CMapLayer	*thisMapLayer = nil;
	WindowPtr	mapWPtr;

	/* fist try and allocate a new layer record handle */
	thisMapLayer = new (CMapLayer);
	if (thisMapLayer != nil)
	{
		mapWPtr = theMap -> GetCMapWindow();
		ErrCode = thisMapLayer -> IMapLayer (mapWPtr, theMap, LayerVisFlag, LOLabelVisFlag);
		if (ErrCode)
		{
			delete (thisMapLayer);
			thisMapLayer = nil;
		}
	}
	else
	{
		ErrCode = memFullErr;
		TechError ("AddNewLayer ()", "Out of memory while in creating new layer!", ErrCode);
	}

	if (!ErrCode)		/* if allocation was successful */
	{
		if (theLayerList != nil)
		{
			/* append this layer onto the layers list (Map-Layer-List) */
			ErrCode = theLayerList -> AppendItem ((Ptr) &thisMapLayer);
			if (ErrCode)
				TechError ("AddNewLayer ()", "Error while trying to add new layer to map layers list!", ErrCode);
		}
	}

	if (ErrCode)		/* do clean up before exiting with an error */
	{
		if (thisMapLayer != nil)
		{
			thisMapLayer -> Dispose ();
			delete (thisMapLayer);
			thisMapLayer = nil;
		}
	}

	return (thisMapLayer);
}
/**************************************************************************************************/
Boolean ListContainsList (CMyList *theOjbectList, CMyList **NextSubList)
{
	LinkDataRec		thisGroupData;
	long			groupObjectCount, objectIndex;
	Boolean			ListEncountered = false;

	groupObjectCount = theOjbectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < groupObjectCount; ++objectIndex)
	{
		theOjbectList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
		{
			ListEncountered = true;
			*NextSubList = thisGroupData.objectList;
			break;
		}
	}

	return (ListEncountered);
}
/**************************************************************************************************/
void PurgeObjectList (CMyList *theOjbectList)
/* recursive routine to dispose of all sublists in a given list including the list itself */
{
	LinkDataRec		thisObjectData;
	long			theObjectCount, objectIndex;
	CMyList			*theSubList = nil;
	Boolean			ListEncountered = false;

	theObjectCount = theOjbectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < theObjectCount; ++objectIndex)
	{
		theOjbectList -> GetListItem ((Ptr) &thisObjectData, objectIndex);
		if (thisObjectData.objectList != nil)
		{
			if (ListContainsList (thisObjectData.objectList, &theSubList))
				PurgeObjectList (theSubList);
			else
			{
				thisObjectData.objectList -> Dispose ();
				thisObjectData.objectList = nil;
//				theOjbectList -> SetListItem ((Ptr) &thisObjectData, objectIndex);
//				Debug ("List Disposed\n");
			}
		}
	}

	return;
}
/**************************************************************************************************/
void SetGroupSelected (CMyList *groupObjectList, Boolean bSelected)
/* Recursive routine to set the selected flag of each object in the group to the supplied value.  */
{
	LinkDataRec		thisGroupData;
	long			groupObjectCount, objectIndex;

	groupObjectCount = groupObjectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < groupObjectCount; ++objectIndex)
	{
		groupObjectList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
			SetGroupSelected (thisGroupData.objectList, bSelected);
		else
			SetObjectSelected (thisGroupData.objectHandle, bSelected);
	}

	return;
}
/**************************************************************************************************/
void MarkObjectsInGroup (CMyList *groupObjectList)
/* recursive routine to mark all objects in given group list */
{
	LinkDataRec		thisGroupData;
	long			groupObjectCount, objectIndex;
	
	groupObjectCount = groupObjectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < groupObjectCount; ++objectIndex)
	{
		groupObjectList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
			MarkObjectsInGroup (thisGroupData.objectList);
		else
			SetObjectMarked (thisGroupData.objectHandle, true);
	}

	return;
}
/**************************************************************************************************/
void GetGroupListLRect (CMyList *groupObjectList, LongRect *theLRectPtr)
/* Warning: Due to the recursive nature of this routine, the-L-Rect-Ptr MUST be set to an empty rect
		 	before this routine is called. */
{
	LinkDataRec		thisGroupData;
	long			groupObjectCount, objectIndex;
	LongRect		thisObjectLRect;
	
	groupObjectCount = groupObjectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < groupObjectCount; ++objectIndex)
	{
		groupObjectList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
			GetGroupListLRect (thisGroupData.objectList, theLRectPtr);
		else
		{
			GetObjectLRect (thisGroupData.objectHandle, &thisObjectLRect);
			UnionLRect (&thisObjectLRect, theLRectPtr, theLRectPtr);
		}
	}

	return;
}
/**************************************************************************************************/
Boolean IsObjectInList (ObjectRecHdl theObjectHdl, CMyList *groupObjectList)
/* Recursive routine to check if the given object is in the objects list provided.  */
{
	LinkDataRec		thisGroupData;
	long			groupObjectCount, objectIndex;

	groupObjectCount = groupObjectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < groupObjectCount; ++objectIndex)
	{
		groupObjectList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
		{
			if (IsObjectInList (theObjectHdl, thisGroupData.objectList))
				return (true);
		}
		else if (thisGroupData.objectHandle == theObjectHdl)
			return (true);
	}

	return (false);		/* lists have been exhausted */
}
/**************************************************************************************************/
Boolean IsListInList (CMyList *keyList, CMyList *mainList)
/* Recursive routine to check if the given list is in the objects list provided.  */
{
	LinkDataRec		thisGroupData;
	long			groupObjectCount, objectIndex;

	groupObjectCount = mainList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < groupObjectCount; ++objectIndex)
	{
		mainList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
		{
			if (thisGroupData.objectList == keyList)
				return (true);
			else
				return (IsListInList (keyList, mainList));
		}
	}

	return (false);
}
/**************************************************************************************************/
void RemoveObjectFromList (CMyList *theObjectList, ObjectRecHdl theObjectHdl)
/* Recursive routine to remove the given object from the objects list provided.  */
{
	LinkDataRec		thisObjectData;
	long			groupObjectCount, objectIndex;

	groupObjectCount = theObjectList -> GetItemCount ();
	for (objectIndex = groupObjectCount - 1; objectIndex >= 0; --objectIndex)
	{
		theObjectList -> GetListItem ((Ptr) &thisObjectData, objectIndex);
		if (thisObjectData.objectList != nil)
			RemoveObjectFromList (thisObjectData.objectList, theObjectHdl);
		else if (thisObjectData.objectHandle == theObjectHdl)
		{
			theObjectList -> DeleteItem (objectIndex);
			return;
		}
	}

	return;
}
/**************************************************************************************************/
void DisposeLayerList (CMyList *theLayerList)
{
	short		LayerCount, LayerIndex;
	CMapLayer	*thisLayer;

	LayerCount = theLayerList -> GetItemCount ();
	for (LayerIndex = 0; LayerIndex < LayerCount; ++LayerIndex)
	{
		theLayerList -> GetListItem ((Ptr) &thisLayer, LayerIndex);
		thisLayer -> Dispose ();
		delete (thisLayer);
	}

	theLayerList -> Dispose ();
	delete (theLayerList);

	return;
}
/**************************************************************************************************/
void DrawObjectsInList (CMyList *thisObjectList, CMap *theMap, LongRect *UpdateLRectPtr,
					    DrawSpecRecPtr drawSettings)
{
	long		ObjectCount, ObjectIndex;
	LinkDataRec	thisSelData;
	
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisSelData, ObjectIndex);
		if (thisSelData.objectList != nil)
			DrawObjectsInList (thisSelData.objectList, theMap, UpdateLRectPtr, drawSettings);
		else
			DrawObject (thisSelData.objectHandle, theMap, UpdateLRectPtr, drawSettings);
	}

	return;
}
/**************************************************************************************************/
void CalcSetOLabelRect (CMap *Map, ObjectRecHdl theObjectHdl, Rect *newLabelRect)
// Note: current port is assumed to be the one being drawn onto
{
	char		objectLabel [kObjectNameLen];
	Rect		labelRect;
	LongPoint	LabelLPoint;
	Point		ScrPoint;
	short		CurrFontNum, CurrSize, CurrStyle;
	long		labelJustCode;
	//WindowPtr	WPtr;
	GrafPtr		WPtr;
	
	GetPortGrafPtr (&WPtr);
	
	GetObjectLabel (theObjectHdl, objectLabel);
	if (strlen (objectLabel) > 0)
	{
#ifdef MAC 
#if TARGET_API_MAC_CARBON
		CurrFontNum = GetPortTextFont(WPtr);		// save current font and size
		CurrSize = GetPortTextSize(WPtr);
		CurrStyle = GetPortTextFace(WPtr);
#else
		CurrFontNum = WPtr -> txFont;		// save current font and size
		CurrSize = WPtr -> txSize;
		CurrStyle = WPtr -> txFace;
#endif
#endif // maybe code goes here
		
		TextFontSizeFace (kFontIDGeneva,9,normal);
		
		GetOLabelLPoint (theObjectHdl, &LabelLPoint);
		labelJustCode = GetOLabelJust (theObjectHdl);
		Map -> GetScrPoint (&LabelLPoint, &ScrPoint);
		
		// compute and set the object's label rect
		labelRect.top    = ScrPoint.v - 6;
		labelRect.bottom = ScrPoint.v + 9;
		
		if (labelJustCode == teJustCenter)
		{
			labelRect.left   = ScrPoint.h - stringwidth (objectLabel) / 2;
			labelRect.right  = labelRect.left + stringwidth (objectLabel) + 2;
		}
		else if (labelJustCode == teJustLeft)
		{
			labelRect.left   = ScrPoint.h + 7;
			labelRect.right  = labelRect.left + stringwidth (objectLabel) + 2;
		}
		
		SetObjectLabelRect (theObjectHdl, &labelRect);
		if (newLabelRect != nil)
			*newLabelRect = labelRect;		// new rect to be sent back
		
		// restore old font and size
#ifdef MAC 
		TextFontSizeFace (CurrFontNum,CurrSize,CurrStyle);
	#endif // maybe code goes here
	}
	
	return;
}

/**************************************************************************************************/
void OffsetObjectsInList (CMyList *thisObjectList, CMap *theMap, Boolean bInvalidate,
						  long LDx, long LDy)
{
	long			ObjectCount, ObjectIndex;
	LinkDataRec		thisSelData;
	
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisSelData, ObjectIndex);
		if (thisSelData.objectList != nil)
			OffsetObjectsInList (thisSelData.objectList, theMap, bInvalidate, LDx, LDy);
		else
			OffsetObject (thisSelData.objectHandle, theMap, LDx, LDy, bInvalidate);
	}

	return;
}
/**************************************************************************************************/
void UndoObjectsInList (CMyList *thisObjectList, Boolean bUndoFlag)
{
	long			ObjectCount, ObjectIndex;
	LinkDataRec		thisSelData;
	
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisSelData, ObjectIndex);
		if (thisSelData.objectList != nil)
			UndoObjectsInList (thisSelData.objectList, bUndoFlag);
		else
			SetObjectUndone (thisSelData.objectHandle, bUndoFlag);
	}

	return;
}
/**************************************************************************************************/
void ScaleObjectsInList (CMyList *thisObjectList, CMap *theMap, Boolean bInvalidate,
						 ScaleRec *ScaleInfo)
{
	long			ObjectCount, ObjectIndex;
	LinkDataRec		thisSelData;
	
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisSelData, ObjectIndex);
		if (thisSelData.objectList != nil)
			ScaleObjectsInList (thisSelData.objectList, theMap, bInvalidate, ScaleInfo);
		else
			ScaleObject (thisSelData.objectHandle, theMap, ScaleInfo, bInvalidate);
	}

	return;
}
/**************************************************************************************************/
void TextSpecObjectsInList (CMyList *thisObjectList, CMap *theMap, long SpecCode, long NewSpec,
							Boolean bUpdate)
{
	long			ObjectCount, ObjectIndex;
	LinkDataRec		thisSelData;
	
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisSelData, ObjectIndex);
		if (thisSelData.objectList != nil)
			TextSpecObjectsInList (thisSelData.objectList, theMap, SpecCode, NewSpec, bUpdate);
//		else
//JLM			SetObjectTextSpec ((TextObjectRecHdl) thisSelData.objectHandle, theMap, SpecCode,
//								NewSpec, bUpdate);
	}

	return;
}
/**************************************************************************************************/
Boolean IsObjectTypeInList (CMyList *theObjectList, OSType theObjectType,
							ObjectRecHdl *theObjectHdl)
/* Recursive routine to check if at least one object of the given type is in the list provided.  */
{
	LinkDataRec		thisGroupData;
	long			ObjectCount, objectIndex;
	OSType			thisObjectType;

	ObjectCount = theObjectList -> GetItemCount ();
	for (objectIndex = 0; objectIndex < ObjectCount; ++objectIndex)
	{
		theObjectList -> GetListItem ((Ptr) &thisGroupData, objectIndex);
		if (thisGroupData.objectList != nil)
		{
			if (IsObjectTypeInList (theObjectList, theObjectType, theObjectHdl))
				return (true);
		}
		else
		{
			GetObjectType (thisGroupData.objectHandle, &thisObjectType);
			if (thisObjectType == theObjectType)
			{
				*theObjectHdl = thisGroupData.objectHandle;
				return (true);
			}
		}
	}

	return (false);		/* lists have been exhausted */
}
/**************************************************************************************************/
void CMapLayer::GetDrawSettings (DrawSpecRecPtr drawSettings, ObjectRecHdl theObjectHdl,
								 long theMode)
{
	if (theObjectHdl == nil)	/* set general settings */
	{
		drawSettings -> mode = theMode;

		if (theMode == kPrintMode)
			drawSettings -> bColor = false;
		else
			drawSettings -> bColor = Map -> IsColorMap ();

		drawSettings -> bClosed = true;	
		drawSettings -> frameCode = kPaintFrameCode;
		drawSettings -> fillCode = kPaintFillCode;
		drawSettings -> offsetDx = 0;
		drawSettings -> offsetDy = 0;
	#ifdef MAC	
		StuffHex (&(drawSettings -> backPattern), kBlackPat);
		StuffHex (&(drawSettings -> forePattern), kStdGrayPat);
	#else
		drawSettings -> backPattern = BLACK_BRUSH; //JLM ??
		drawSettings -> forePattern = GRAY_BRUSH; //JLM ??
	#endif
		
		if (drawSettings -> mode == kDragMode)
			drawSettings -> bDrawOLabels = false;	/* don't draw labels when dragging objects */
		// JLMelse
			//JLM drawSettings -> bDrawOLabels = IsLayerOLabelVisible ();
	}
	else						/* set object specific settings */
	{
		OSType	thisObjectType;

		drawSettings -> foreColorInd = GetObjectColor  (theObjectHdl);
		drawSettings -> backColorInd = GetObjectBColor (theObjectHdl);
		drawSettings -> bErase = false;

		GetObjectType (theObjectHdl, &thisObjectType);
		if (thisObjectType == kPolyType)
			drawSettings -> bClosed = IsPolyClosed ((PolyObjectHdl) theObjectHdl);
		else
			drawSettings -> bClosed = true;

		/* OPAQUE object being drawn in B&W */
		if ((drawSettings -> mode == kPrintMode || !drawSettings -> bColor)
			 && (drawSettings -> backColorInd != kNoColorInd &&
			 	 drawSettings -> backColorInd != kWaterColorInd &&
			 	 drawSettings -> bClosed))
		{
			drawSettings -> bErase = true;				/* erase before filling */
			drawSettings -> fillCode = kPatFillCode;	/* use fill pattern */

			if (drawSettings -> mode == kPrintMode)
			{
				/* for printing, use very light gray */
	#ifdef MAC	
				StuffHex (&(drawSettings -> backPattern), kVeryLgtGrayPat);
	#else
				drawSettings -> backPattern = LTGRAY_BRUSH; //JLM ??
	#endif
			}
			else
			{
				/* for screen and pict modes, use rainy hash pattern */
	#ifdef MAC	
				StuffHex (&(drawSettings -> backPattern), "\p4020100000000000");
	#else
				drawSettings -> backPattern = LTGRAY_BRUSH; //JLM ??
	#endif
			}
		}
		else if (!drawSettings -> bClosed || drawSettings -> backColorInd == kNoColorInd)
			/* TRANSPARENT polygon with no color fill code */
			drawSettings -> fillCode = kNoFillCode;
		else if (drawSettings -> mode == kScreenMode || drawSettings -> mode == kPictMode)
		{
			/* OPAQUE object being drawn in COLOR to SCREEN or PICT */
			if (drawSettings -> backColorInd == kWaterColorInd) 
				drawSettings -> fillCode = kNoFillCode;	// here set no fill for lakes so can set spill
			else 	
				drawSettings -> fillCode = kPaintFillCode;
			GetObjectFillPat (theObjectHdl, &(drawSettings -> backPattern));
		}
		else if (drawSettings -> mode == kPrintMode)
		{
			/* opaque object being drawn to printer */
			drawSettings -> bErase = true;
	#ifdef MAC	
			StuffHex (&(drawSettings -> backPattern), kWhitePat);
	#else
				drawSettings -> backPattern = WHITE_BRUSH; //JLM ??
	#endif
		}
	}

	return;
}
/**************************************************************************************************/
ObjectRecHdl GetObjectWithID (CMyList *MapLayerList, ObjectIDRecPtr ObjectIDPtr)
/* given the ID record of an object, this subroutine attempts to find the object with the given ID.
	If such an object cannot be found, it returns nil. */
{
	short			LayerCount, LayerIndex;
	CMyList			*thisObjectList;
	long			ObjectCount, ObjectIndex;
	ObjectRecHdl	thisObjectHdl, IDObjectHdl = nil;
	ObjectIDRec		thisObjectID;
	CMapLayer		*thisMapLayer = nil;
	
	LayerCount = MapLayerList -> GetItemCount ();
	for (LayerIndex = 1; LayerIndex <= LayerCount; ++LayerIndex)
	{
		/* get this layer's objects list */
		thisMapLayer = GetMapLayer (MapLayerList, LayerIndex);
		thisObjectList = thisMapLayer -> GetLayerObjectList ();

		ObjectCount = thisObjectList -> GetItemCount ();
		for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
		{
			thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
			GetObjectID (thisObjectHdl, &thisObjectID);
			if (thisObjectID == *ObjectIDPtr)
			{
				IDObjectHdl = thisObjectHdl;
				break;
			}
		}
	}

	return (IDObjectHdl);
}
