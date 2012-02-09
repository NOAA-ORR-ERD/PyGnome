
#include "Cross.h"
#include "ObjectUtils.h"
#include "GenDefs.h"
/**************************************************************************************************/

#define CMapLayerREADWRITEVERSION 1 //JLM



OSErr CMapLayer::Read(BFPB *bfpb) // JLM
{

	long version;
	ClassID id;
	long 	numObjs,numPolyPts,i,ptIndex;
	OSType thisObjectType;
	CMyList	*objectList;
	OSErr err = 0;	

	StartReadWriteSequence("CMapLayer::Read()");

	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != this->GetClassID()) goto cantReadFile;

	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version != CMapLayerREADWRITEVERSION) goto cantReadFile;
	
	// read in the  map-specfic fields
	if (err = ReadMacValue(bfpb, &numObjs)) return err;
	objectList = this -> GetLayerObjectList ();
	if(numObjs< 0) goto cantReadFile;
	for (i = 0; i < numObjs; ++i)
	{
		if (err = ReadMacValue(bfpb, &thisObjectType)) return err;//JLM
		
		switch(thisObjectType)
		{
			case kPolyType:
			{
				// allocate and read the header fields for this polygon
				
				PolyObjectHdl	thisObjectHdl = 0;
				LongRect	ObjectLRect;
				LongPoint	**thisPtsHdl = nil;
				PolyObjectRec 	localPolyObject;
				
				// set region rect to empty to start with
				SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);
				thisObjectHdl = (PolyObjectHdl) this -> AddNewObject (kPolyType, &ObjectLRect, false);
				memset(&localPolyObject,0,sizeof(localPolyObject));
				if (!thisObjectHdl) 
					{printError("Out of memory");return -1;}
				
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectType))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLPoint))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLabelLPoint))) return err;

				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLRect.top))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLRect.left))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLRect.bottom))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLRect.right))) return err;

				if (err = ReadMacValue(bfpb, &(localPolyObject.objectSymbol))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectFlags))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectColorInd))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectBColorInd))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectLabelJust))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectHotLinkData))) return err;
				if (err = ReadMacValue(bfpb, localPolyObject.objectLabel, kObjectNameLen)) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectCustData))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.objectESICode))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.pointCount))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.bClosedPoly))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.bWaterPoly))) return err;
				if (err = ReadMacValue(bfpb, &(localPolyObject.polyLineWidth))) return err;
	
				// read the points for this polygon
				numPolyPts = localPolyObject.pointCount;
				if(numPolyPts< 0) goto cantReadFile;
				
				**thisObjectHdl = localPolyObject;
				
				thisPtsHdl = (LongPoint**) _NewHandle(numPolyPts * sizeof(LongPoint));
				if (!thisPtsHdl) 
					{printError("Out of memory");return -1;}
				
				for (ptIndex = 0; ptIndex < numPolyPts; ptIndex++)
				{
					LongPoint thisLPoint;
					if (err = ReadMacValue(bfpb, &(thisLPoint.h))) return err;
					if (err = ReadMacValue(bfpb, &(thisLPoint.v))) return err;
					 (*thisPtsHdl) [ptIndex] = thisLPoint;
				}
				//assign handle to polygon
				SetPolyPointsHdl (thisObjectHdl,thisPtsHdl);
				break;
			}
			default:
				goto cantReadFile;
				break;
		}
	}
	
		
	return 0;
	
cantReadFile:
		printSaveFileVersionError(); 
		return -1; 
	

}

OSErr CMapLayer::Write (BFPB *bfpb) //JLM
{

	long 			version = CMapLayerREADWRITEVERSION, i, ptIndex;
	ClassID 		id = this->GetClassID();
	CMyList			*objectList;
	OSErr			err = noErr;
	PolyObjectHdl	thisObjectHdl;
	PolyObjectRec	localCopy;
	OSType			thisObjectType;
	LongPoint		thisLPoint;
	LongPoint		**thisPtsHdl = nil;
	long 	num;

	StartReadWriteSequence("CMapLayer::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;

	// begin writing map-specfic fields
	objectList = this -> GetLayerObjectList ();
	num = objectList -> GetItemCount (); //JLM
	if (err = WriteMacValue(bfpb, num)) return err;
	
	for (i = 0; i < num; ++i)
	{
		objectList -> GetListItem ((Ptr) &thisObjectHdl, i);
		GetObjectType ((ObjectRecHdl) thisObjectHdl, &thisObjectType);
		
		if (err = WriteMacValue(bfpb, thisObjectType)) return err;//JLM
		
		switch(thisObjectType)
		{
			case kPolyType:
				// write the header fields for this polygon
				localCopy = **thisObjectHdl;
				if (err = WriteMacValue(bfpb, localCopy.objectType)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLPoint)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLabelLPoint)) return err;

				if (err = WriteMacValue(bfpb, localCopy.objectLRect.top)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLRect.left)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLRect.bottom)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLRect.right)) return err;

				if (err = WriteMacValue(bfpb, localCopy.objectSymbol)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectFlags)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectColorInd)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectBColorInd)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLabelJust)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectHotLinkData)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectLabel, kObjectNameLen)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectCustData)) return err;
				if (err = WriteMacValue(bfpb, localCopy.objectESICode)) return err;
				if (err = WriteMacValue(bfpb, localCopy.pointCount)) return err;
				if (err = WriteMacValue(bfpb, localCopy.bClosedPoly)) return err;
				if (err = WriteMacValue(bfpb, localCopy.bWaterPoly)) return err;
				if (err = WriteMacValue(bfpb, localCopy.polyLineWidth)) return err;
	
				// write the points for this polygon
				thisPtsHdl = GetPolyPointsHdl (thisObjectHdl);
				if(!thisPtsHdl) {printError("Programmer error: thisPtsHdl nil in CMapLayer::Write");return -1;}

				for (ptIndex = 0; ptIndex < localCopy.pointCount; ptIndex++)
				{
					thisLPoint = (*thisPtsHdl) [ptIndex];
					if (err = WriteMacValue(bfpb, (thisLPoint.h))) return err;
					if (err = WriteMacValue(bfpb, (thisLPoint.v))) return err;
				}
				break;
			default:
				printError("Programmer error: Unexpected type in CMapLayer::Write");
				return -1;
				
		}
	}

	return err;

}




OSErr CMapLayer::IMapLayer (WindowPtr mapWPtr, CMap *theMap,
							Boolean bLayerVisible, Boolean bLayerOLabelVisble)
{
	OSErr		ErrCode = 0;
	CMyList		*thisObjectList = nil;
	CMyList		*thisGroupList = nil;
	CMyList		*thisLayerSelList = nil;

	Map  = theMap;			/* copy the map object    to private layer field */
	WPtr = mapWPtr;			/* copy the map windowPtr to private layer field */
	
	/* allocate an undo object for this layer */
	layerUndoObjectHdl = (UndoObjectRecHdl) _NewHandleClear (sizeof (UndoObjectRec));
	if (layerUndoObjectHdl == nil)
	{
		ErrCode = memFullErr;
		TechError ("CMapLayer::IMapLayer", "Error creating undo object for new layer!", ErrCode);
	}
	else
		SetObjectDataHdl ((ObjectRecHdl) layerUndoObjectHdl, nil);

	if (!ErrCode)
	{
		/* now allocate the necessary lists to manage this layer */
		thisObjectList = new CMyList (sizeof (ObjectRecHdl));
		if (thisObjectList == nil)
		{
			ErrCode = memFullErr;
			TechError ("CMapLayer::IMapLayer", "Error in creating objects list for new layer!", ErrCode);
		}
		else
			ErrCode = thisObjectList -> IList ();
	}

	if (!ErrCode)		/* now try and allocate a groups list for this layer */
	{
		thisGroupList = new CMyList (sizeof (CMyList*));
		if (thisGroupList == nil)
		{
			ErrCode = memFullErr;
			TechError ("CMapLayer::IMapLayer", "Error in creating new groups list for new layer!", ErrCode);
		}
		else
			ErrCode = thisGroupList -> IList ();
	}

	if (!ErrCode)		/* try to allocate a list to store selected objects */
	{
		thisLayerSelList = new CMyList (sizeof (LinkDataRec));
		if (thisLayerSelList == nil)
		{
			ErrCode = memFullErr;
			TechError ("CMapLayer::IMapLayer", "Error in allocating object selection list for new layer!", ErrCode);
		}
		else
			ErrCode = thisLayerSelList -> IList ();
	}

	if (!ErrCode)		/* if allocation was successful */
	{
		LongRect	thisLayerLRect;
		char fontLabel[32];
		strcpy(fontLabel,"Geneva");

		SetLayerVisible (bLayerVisible);
		SetLayerMinVisScale (-1);
		SetLayerMaxVisScale (-1);

		SetLayerName ("Untitled Layer");
		SetLRect (&thisLayerLRect, 0, 0, 0, 0);
		SetLayerScope (&thisLayerLRect);

		SetLayerOLabelVisible (bLayerOLabelVisble);
		SetLayerOLMinScale (-1);			/* no minimum if -1 */
		SetLayerOLMaxScale (-1);			/* no maximum if -1 */

		lastTextFont = kFontIDGeneva;
		lastTextStyle = normal;
		lastTextSize = 9;
		lastTextJust = teJustCenter;

		//SetLayerOLabelFont ("Geneva");
		SetLayerOLabelFont (fontLabel);
		SetLayerOLabelFSize (9);
		SetLayerOLabelFStyle (normal);
		SetLayerOLabelColor (kBlackColorInd);

		SetLayerObjectList (thisObjectList);
		SetLayerGroupList (thisGroupList);
		SetLayerSelectList (thisLayerSelList);

		SetLayerModified (false);
		layerUndoCode = 0;

//		layerTextHandle = nil;
		layerTextItemNum = -1;
	}

	if (ErrCode)		/* do clean up before exiting with an error */
	{
		if (thisObjectList != nil)
		{
			delete (thisObjectList);
			SetLayerObjectList (nil);
		}

		if (thisGroupList != nil)
		{
			delete (thisGroupList);
			SetLayerGroupList (nil);
		}

		if (thisLayerSelList != nil)
		{
			delete (thisLayerSelList);
			SetLayerSelectList (nil);
		}
	}

	return (ErrCode);
}
/**************************************************************************************************/
void CMapLayer::Dispose ()
{
	long			ObjectCount, ObjectIndex;
	ObjectRecHdl	thisObjectHdl;
	
	if (layerObjectList != nil)
	{
		ObjectCount = layerObjectList -> GetItemCount ();
		for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
		{
			layerObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
			KillObject (thisObjectHdl);
		}

		/* now dispose and delete the objects list */
		layerObjectList -> Dispose ();
		delete (layerObjectList);
		layerObjectList = nil;
	}

	if (layerGroupList != nil)
	{
		long		GroupIndex, LayerGroupCount;
		CMyList		*thisGroupOList = nil;

		LayerGroupCount = GetLayerGroupCount ();
		for (GroupIndex = LayerGroupCount; GroupIndex >= 1; --GroupIndex)
		{
			thisGroupOList = GetGroupObjectList (GroupIndex);
			thisGroupOList -> Dispose ();
			delete (thisGroupOList);
		}

		layerGroupList -> Dispose ();
		delete (layerGroupList);
		SetLayerGroupList ((CMyList*) nil);
	}

	if (layerSelectList != nil)
	{
		delete (layerSelectList);
		layerSelectList = nil;
	}
	
	if (layerUndoObjectHdl != nil)
	{
		KillObject ((ObjectRecHdl) layerUndoObjectHdl);
		layerUndoObjectHdl = nil;
		layerUndoCode = 0;
	}

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerName (char *LayerNamePtr)
{
	strcpy (layerName, LayerNamePtr);

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerVisible (Boolean Visible)
{
	bLayerVisible = Visible;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerMinVisScale (long theScale)
{
	layerMinVisScale = theScale;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerMaxVisScale (long theScale)
{
	layerMaxVisScale = theScale;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLabelVisible (Boolean OLabelVisFlag)
{
	bLayerOLabelVisible = OLabelVisFlag;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLabelColor (long OLabelColorInd)
{
	layerLabelColorInd = OLabelColorInd;

	return;
}
/**************************************************************************************************/
Boolean CMapLayer::IsLayerOLabelVisible ()
{
	if (!bLayerOLabelVisible)
		return (false);
	else
	{
		long	CurMapScale;

		CurMapScale = Map -> GetMapScale ();
		if (oLabelMinVisScale != -1 && CurMapScale < oLabelMinVisScale)
			return (false);

		if (oLabelMaxVisScale != -1 && CurMapScale > oLabelMaxVisScale)
			return (false);
	}

	return (true);
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLabelFont (char *LayerLabelFont)
{
	strcpy (LayerLabelFont, LayerLabelFont);

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLabelFSize (long ObjectFSize)
{
	layerLabelFontSize = ObjectFSize;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLabelFStyle (long ObjectFStyle)
{
	layerLabelFontStyle = ObjectFStyle;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerScope (LongRect *thisLayerLRectPtr)
{
	layerScopeLRect = *thisLayerLRectPtr;
	
	return;
}
/**************************************************************************************************/
void CMapLayer::GetLayerScope (LongRect *LayerLBoundsPtr, Boolean bRecalc)
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
long CMapLayer::GetLayerPolyPtsCount ()
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
void CMapLayer::SetLayerModified (Boolean theModFlag)
{
	bLayerModified = theModFlag;

	return;
}
/**************************************************************************************************/
Boolean CMapLayer::IsLayerModified ()
{
	return (bLayerModified);
}
/**************************************************************************************************/
void CMapLayer::SetLayerObjectList (CMyList *ObjectList)
{
	layerObjectList = ObjectList;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerGroupList (CMyList *theGroupList)
/* LayerGroupList is a list of groups contained in this layer.  Each list contains one or more
	group-data-rec type records */
{
	layerGroupList = theGroupList;

	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerSelectList (CMyList *theSelectList)
{
	layerSelectList = theSelectList;

	return;
}
/**************************************************************************************************/
CMyList* CMapLayer::GetLayerObjectList ()
{
	CMyList		*thisObjectList = nil;

	thisObjectList = layerObjectList;

	return (thisObjectList);
}
/**************************************************************************************************/
CMyList* CMapLayer::GetLayerGroupList ()
{
	CMyList		*thisGroupList = nil;

	thisGroupList = layerGroupList;

	return (thisGroupList);
}
/**************************************************************************************************/
long CMapLayer::GetLayerObjectCount ()
{
	long			ObjectCount = 0;
	CMyList			*thisObjectList = nil;
	
	thisObjectList = GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();

	return (ObjectCount);
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLMaxScale (long theScale)
{
	oLabelMaxVisScale = theScale;
	
	return;
}
/**************************************************************************************************/
void CMapLayer::SetLayerOLMinScale (long theScale)
{
	oLabelMinVisScale = theScale;

	return;
}
/**************************************************************************************************/
CMyList* CMapLayer::AddNewGroup ()
/* returns the object list of new group or nil if an error occurs */
{
	CMyList 		*newObjectList = nil;
	OSErr			ErrCode = 0;
	
	newObjectList = new CMyList (sizeof (LinkDataRec));		/* try to create a new objects list for new group */
	if (newObjectList == nil)
		ErrCode = memFullErr;
	else
		ErrCode = newObjectList -> IList ();

	if (!ErrCode)	/* try and append new object list to layer groups list */
		ErrCode = layerGroupList -> InsertAtIndex ((Ptr) &newObjectList, 1);

	if (ErrCode)
	{
		if (newObjectList)
		{
			newObjectList -> Dispose ();
			delete (newObjectList);
			newObjectList = nil;
		}

		TechError ("CMapLayer:;IMapLayer", "Not enough memory to create a new group!", ErrCode);
	}

	return (newObjectList);
}
/**************************************************************************************************/
long CMapLayer::GetLayerGroupCount ()
/* returns the total number of group definitions in the layer */
{
	long		groupCount;
	CMyList		*thisLayerGroupList = nil;

	thisLayerGroupList = GetLayerGroupList ();
	groupCount = thisLayerGroupList -> GetItemCount ();

	return (groupCount);
}
/**************************************************************************************************/
OSErr CMapLayer::AddObjectToGroup (CMyList *theObjectList, ObjectRecHdl theObjectHdl)
{
	OSErr			ErrCode = 0;
	LinkDataRec		newGroupData;

	newGroupData.objectHandle = theObjectHdl;
	newGroupData.objectList = nil;

	ErrCode = theObjectList -> AppendItem ((Ptr) &newGroupData);
	if (!ErrCode)
		SetObjectGrouped (theObjectHdl, true);

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::AddOListToGroup (CMyList *mainObjectList, CMyList *groupObjectList)
/* this subroutine simply appends the given object list onto the given main-object-list list */
{
	OSErr			ErrCode = 0;
	LinkDataRec		newGroupData;

	newGroupData.objectHandle = nil;
	newGroupData.objectList = groupObjectList;
	ErrCode = mainObjectList -> AppendItem ((Ptr) &newGroupData);

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::AddObjectToSelList (ObjectRecHdl theObjectHdl, Boolean bInvert, Boolean bSetFlags)
/* if b-set-flags is true, the objects selected flag is set to true */
{
	OSErr			ErrCode = 0;
	LinkDataRec		thisSelData;
	CMyList			*thisObjGroupList = nil;
	long			theGroupNum;

	if (IsObjectGrouped (theObjectHdl, &thisObjGroupList, &theGroupNum))
	{
		thisSelData.objectList = thisObjGroupList;
		thisSelData.objectHandle = nil;

		ErrCode = layerSelectList -> AppendItem ((Ptr) &thisSelData);
		if (!ErrCode)			/* set the object's selected flag */
		{
			if (bSetFlags)
				SelectGroup (thisObjGroupList, bInvert, true, true);
		}
	}
	else
	{
		thisSelData.objectList = nil;
		thisSelData.objectHandle = theObjectHdl;

		ErrCode = layerSelectList -> AppendItem ((Ptr) &thisSelData);
		if (!ErrCode)			/* set the object's selected flag */
		{
			if (bInvert)
				InvertObject (Map, theObjectHdl);
				
			if (bSetFlags)
				SetObjectSelected (theObjectHdl, true);
		}
	}

	return (ErrCode);
}
/**************************************************************************************************/
void CMapLayer::RemObjectFromSelList (ObjectRecHdl theObjectHdl, Boolean bInvert, Boolean bSetFlags)
/* this subroutine simply adds the given object to the layer object selection list.
	Note: It does NOT change any of the object's settings */
{
	LinkDataRec		thisSelData;
	long			ObjectIndex, theGroupNum;
	CMyList			*theObjectList = nil;	
	
	if (IsObjectGrouped (theObjectHdl, &theObjectList, &theGroupNum))
	{
		thisSelData.objectList = theObjectList;
		thisSelData.objectHandle = nil;

		if (layerSelectList -> IsItemInList ((Ptr) &thisSelData, &ObjectIndex))
		{
			layerSelectList -> DeleteItem (ObjectIndex);
			if (bSetFlags)
				SelectGroup (theObjectList, bInvert, true, false);
		}
	}
	else
	{
		thisSelData.objectList = nil;
		thisSelData.objectHandle = theObjectHdl;

		if (layerSelectList -> IsItemInList ((Ptr) &thisSelData, &ObjectIndex))
		{
			layerSelectList -> DeleteItem (ObjectIndex);
			if (bInvert)
				InvertObject (Map, theObjectHdl);

			if (bSetFlags)
				SetObjectSelected (theObjectHdl, false);
		}
	}

	return;
}
/**************************************************************************************************/
OSErr CMapLayer::AddGroupToSelList (CMyList *theGroupOList)
/* this subroutine simply adds the given group to the layer's object selection list.
	Note: It does NOT change any of the object's settings */
{
	OSErr			ErrCode = 0;
	LinkDataRec		thisSelData;

	thisSelData.objectList = theGroupOList;
	thisSelData.objectHandle = nil;

	ErrCode = layerSelectList -> AppendItem ((Ptr) &thisSelData);

	return (ErrCode);
}
/**************************************************************************************************/
void CMapLayer::RemGroupFromSelList (CMyList *theGroupOList)
/* this subroutine simply removes the given group to the layer's object selection list.
	Note: It does NOT change any of the object's settings */
{
	LinkDataRec		thisSelData;
	long			itemNum;

	thisSelData.objectList = theGroupOList;
	thisSelData.objectHandle = nil;

	if (layerSelectList -> IsItemInList ((Ptr) &thisSelData, &itemNum))
		layerSelectList -> DeleteItem (itemNum);

	return;
}
/**************************************************************************************************/
CMyList* CMapLayer::GetGroupObjectList (long theGroupNum)
{
	CMyList		*thisGroupObjList = nil;

	layerGroupList -> GetListItem ((Ptr) &thisGroupObjList, theGroupNum);

	return (thisGroupObjList);
}
/**************************************************************************************************/
void CMapLayer::SelectGroup (CMyList *theObjectList, Boolean bInvert, Boolean bSetFlag,
							 Boolean bSelected)
/* subroutine to invert the given group and to set the selected flag of each object in the group to
   the flag supplied in b-Selected.
	Note: If b-setflag is false, the value of b-selected is not relevant. */
{
	if (bInvert)
		InvertGroup (theObjectList);

	if (bSetFlag)
		SetGroupSelected (theObjectList, bSelected);			/* recursive routine */

	return;
}
/**************************************************************************************************/
void CMapLayer::InvertGroup (CMyList *theObjectList)
{
	LongRect	thisGroupLRect;
	GrafPtr		SavePort;
	PenState	penStatus;

	GetGroupLRect (theObjectList, &thisGroupLRect);

	GetPortGrafPtr (&SavePort);
	GetPenState (&penStatus);	/* save original pen state */

	/* now draw the selection points on the grouped / single region */
	PenSize (4, 4);
	PenMode (srcXor);

	Map -> LMoveTo (thisGroupLRect.left, thisGroupLRect.top);
	Map -> LLineTo (thisGroupLRect.left, thisGroupLRect.top);

	Map -> LMoveTo (thisGroupLRect.right, thisGroupLRect.top);
	Map -> LLineTo (thisGroupLRect.right, thisGroupLRect.top);

	Map -> LMoveTo (thisGroupLRect.right, thisGroupLRect.bottom);
	Map -> LLineTo (thisGroupLRect.right, thisGroupLRect.bottom);

	Map -> LMoveTo (thisGroupLRect.left, thisGroupLRect.bottom);
	Map -> LLineTo (thisGroupLRect.left, thisGroupLRect.bottom);

	SetPenState (&penStatus);
	SetPortGrafPort (SavePort);

	return;
}
/**************************************************************************************************/
ObjectRecHdl CMapLayer::AddNewObject (OSType ObjectType, LongRect *thisObjectLRectPtr,
									  Boolean bPutAtTop)
/* subroutine to create and add a new object of the given type.
	Note: default parameters are also inserted into object data fields */
{
	ObjectRecHdl	NewObjectHdl = nil;
	OSErr			err = noErr;
	ObjectIDRec		thisObjectID;
	long			SpaceNeeded;
	LongPoint		**thisPointsHdl;
	Pattern			thisPattern;

	switch (ObjectType)
	{
		case kPolyType:
			SpaceNeeded = sizeof (PolyObjectRec);
		break;

		case kTextType:
			SpaceNeeded = sizeof (TextObjectRec);
		break;

		case kLineType:
			SpaceNeeded = sizeof (LineObjectRec);
		break;

		default:
			SpaceNeeded = sizeof (ObjectInfoRec);
		break;
	}

//	if (IsMemAvailable (SpaceNeeded, true) && layerObjectList != nil)
	{
		NewObjectHdl = (ObjectRecHdl) _NewHandleClear (SpaceNeeded);
		if (NewObjectHdl != nil)
		{
			if (bPutAtTop)
				err = layerObjectList -> InsertAtIndex ((Ptr) &NewObjectHdl, 1);
			else
				err = layerObjectList -> AppendItem ((Ptr) &NewObjectHdl);
	
			if (err)
			{
				TechError ("CMapLayer::AddObject", "Error in adding new object to objects list", err);
				DisposeHandle ((Handle) NewObjectHdl);
				NewObjectHdl = nil;
			}
			else	/* insert default parameters into object handle */
			{
				/* set object fields using supplied parameters */
				SetObjectType (NewObjectHdl, ObjectType);
				
				if (thisObjectLRectPtr != nil)
				{
					LongPoint	objectCenterLPoint;

					SetObjectLRect (NewObjectHdl, thisObjectLRectPtr);
					GetLRectCenter (thisObjectLRectPtr, &objectCenterLPoint);
					SetObjectLPoint (NewObjectHdl, &objectCenterLPoint);
					SetOLabelLPoint (NewObjectHdl, &objectCenterLPoint);
				}

				SetOLabelJust (NewObjectHdl, teJustCenter);

				/* set default flags */
				GetUniqueID (&thisObjectID, 1);			/* get a new id to assign to this object */
				SetObjectID (NewObjectHdl, &thisObjectID);

				SetObjectSymbol (NewObjectHdl, -1);		/* no defualt symbol */
				SetObjectSelected (NewObjectHdl, false);
				SetObjectMarked (NewObjectHdl, false);
				SetObjectVisible (NewObjectHdl, true);
				SetOLabelVisible (NewObjectHdl, true);
				SetOSymbolVisible (NewObjectHdl, false);
				SetOHotLinked (NewObjectHdl, false);

				/* set remaining defaults */
				SetObjectColor (NewObjectHdl, kLandColorInd);
				//SetObjectBColor (NewObjectHdl, kNoColorInd);
				SetObjectBColor (NewObjectHdl, kLandColorInd); //JLM 1/14/99

				/* black pattern for default object fill */
				#ifdef MAC
					StuffHex (&thisPattern, kBlackPat);
				#else
					thisPattern = BLACK_BRUSH; // JLM
				#endif
				SetObjectFillPat (NewObjectHdl, &thisPattern);
				SetObjectLabel (NewObjectHdl, "Untitled");
				
				SetObjectCustData (NewObjectHdl, 0);	/* no custom data by default */
				SetObjectDataHdl (NewObjectHdl, nil);
				SetObjectESICode (NewObjectHdl, -500);

				/* initialize future use variables */
				(**NewObjectHdl).futureLong1 = 0;
				(**NewObjectHdl).futureLong2 = 0;
				(**NewObjectHdl).futureLong3 = 0;
				(**NewObjectHdl).futureLong4 = 0;
			}

			if (!err)	/* initialize additional fields based on object type if necessary */
			{				
				switch (ObjectType)
				{
					case kPolyType:
						(**((PolyObjectHdl) NewObjectHdl)).pointCount = 0;
						(**((PolyObjectHdl) NewObjectHdl)).bClosedPoly = true;
						(**((PolyObjectHdl) NewObjectHdl)).bWaterPoly = false;
						(**((PolyObjectHdl) NewObjectHdl)).polyLineWidth = 1;
						
						CreateNewPtsHdl ((PolyObjectHdl) NewObjectHdl, 0);	// create handle to store points
					break;
					
					case kSymbolType:
						SetOLabelJust (NewObjectHdl, teJustLeft);
						
						// custom-data field used to store plot-icon ID
//						SetObjectCustData (NewObjectHdl, SymbolToolCursID);
					break;
					
					case kTextType:
						TextSpecRec	TextSpec;
						
						_HLock ((Handle) NewObjectHdl);
						
						strcpy (TextSpec.textFont, "Times");
						TextSpec.textStyle = normal;
						TextSpec.textJust = teJustCenter;
						
						(**((TextObjectRecHdl) NewObjectHdl)).textSpec = TextSpec;
						(**((TextObjectRecHdl) NewObjectHdl)).textStr [0] = '\0';
										
						_HUnlock ((Handle) NewObjectHdl);
						
						SetOLabelVisible (NewObjectHdl, false);
					break;
					
					case kLineType:
						SetOLabelVisible (NewObjectHdl, false);
					break;
					
					case kRectType:
						SetOLabelVisible (NewObjectHdl, false);
					break;
				}
			}
		}
		else
			TechError ("CMapLayer::AddNewObject", "Could not allocate space for new object!", err);
	}
//	else
//		TechError ("CMapLayer::AddNewObject", "Not enough memory to create and store another object!", err);

	if (!err)
		SetLayerModified (true);

	return (NewObjectHdl);
}
/**************************************************************************************************/
Boolean CMapLayer::DoClick (char *currToolName, Point LocalPt, Boolean *bModified)
{
	Boolean			bHandled = false, bWasSelected = false;
	ObjectRecHdl	ClickedObjectHdl = nil;
	LongRect		MapViewLRect;
	DrawSpecRec		drawSettings;
	
	*bModified = false;

	if (!strcmpnocase (currToolName, "Arrow Tool"))
	{
		ClickedObjectHdl = FindClickedObject (LocalPt);
		if (ClickedObjectHdl != nil)
		{
			if (!ShiftKeyDown ())
			{
				bWasSelected = IsObjectSelected (ClickedObjectHdl);

				if (bWasSelected)
					bHandled = DoGrowSelection (LocalPt);

				if (bWasSelected)
				{
					bHandled = DoDragSelection (LocalPt);	// returns true if moved
					if (bHandled)
						*bModified = true;					// object(s) moved
				}
				
//				if (!bHandled && bWasSelected && IsDoubleClick () && !ShiftKeyDown ())
//					bHandled = DoObjectDblClick (ClickedObjectHdl);

				if (!bHandled)
				{
					ClearSelection (true, true);
					AddObjectToSelList (ClickedObjectHdl, true, true);
				}

//				if (StillDown ())		/* drag newly selected object if necessary */
//				{
//					if (!bHandled)
//						bHandled = DoDragSelection (LocalPt);
//				}

				*bModified = bHandled;	/* drag or grow took place */
			}
			else	/* shift key is down, so toggle selection */
			{
				if (IsObjectSelected (ClickedObjectHdl))
				{
					RemObjectFromSelList (ClickedObjectHdl, true, true);
				}
				else
				{
					AddObjectToSelList (ClickedObjectHdl, true, true);
				}
			}

			bHandled = true;
		}
		else	/* no valid object was clicked on */
		{
			if (!ShiftKeyDown ())
				ClearSelection (true, true);
		}
	}
	else if (!strcmpnocase (currToolName, "Rect Tool"))
	{
		Rect			ObjectRect;
		LongRect		ObjectLRect;
		OSErr			err = noErr;
		ObjectRecHdl	theObjectHdl;
		GrafPtr			SavePort;

		GetPortGrafPtr (&SavePort);
		
//JLM		err = Map -> GetDragRect (LocalPt, false, 0, &ObjectRect);
		if (!err)
		{
			Map -> GetMatrixLRect (&ObjectRect, &ObjectLRect);
			theObjectHdl = AddNewObject (kRectType, &ObjectLRect, true);
			if (theObjectHdl)
			{
				Map -> GetMapViewLRect (&MapViewLRect);
				GetDrawSettings (&drawSettings, nil, kScreenMode);
				GetDrawSettings (&drawSettings, theObjectHdl, kScreenMode);
				DrawObject (theObjectHdl, Map, &MapViewLRect, &drawSettings);
				ClearSelection (true, true);
				AddObjectToSelList (theObjectHdl, true, true);
			
				/* change tool to arrow after creating new object */
//				mapToolSet -> SetCurrTool (ArrowTool);
				
				// set layer-modified flag
				*bModified = true;
			}
		}
		
		SetPortGrafPort (SavePort);
	}
	else if (!strcmpnocase (currToolName, "Text Tool"))
	{
		OSType	ObjectType;
		GrafPtr	SavePort;
		
		GetPortGrafPtr (&SavePort);
		SetPortWindowPort (WPtr);
		
		if (!bHandled)	/* new text needs to be started */
		{
			ClickedObjectHdl = FindClickedObject (LocalPt);
			if (ClickedObjectHdl != nil)
			{
				GetObjectType (ClickedObjectHdl, &ObjectType);
				if (ObjectType != kTextType)
					ClickedObjectHdl = nil;			/* non text objects don't count */
			}

			/* click object handle is nil if no object was clicked on */
//JLM			StartNewText (LocalPt, (TextObjectRecHdl) ClickedObjectHdl);
		}
	
		SetPortGrafPort (SavePort);
	}
	else if (!strcmpnocase (currToolName, "Line Tool"))
	{
		Rect				LineRect;
		Point				LineStartPt, LineEndPt;
		LongPoint			LineStartLPt, LineEndLPt;
		LongRect			LineLRect;
		GrafPtr				SavePort;
		OSErr				ErrCode = 0;
		LineObjectRecHdl	lineObjectHdl;

		GetPortGrafPtr (&SavePort);
		SetPortWindowPort (WPtr);
		
//JLM		ErrCode = Map -> GetDragLine (LocalPt, &LineStartPt, &LineEndPt);
		// then the rest of this doesn't make sense...
		/*if (!ErrCode)
		{
			// get the matrix rect for new line object 
			LineRect.left   = LineStartPt.h;
			LineRect.top    = LineStartPt.v;
			LineRect.right  = LineEndPt.h;
			LineRect.bottom = LineEndPt.v;
			NormalizeRect (&LineRect);
			
			Map -> GetMatrixLRect (&LineRect, &LineLRect);
			lineObjectHdl = (LineObjectRecHdl) AddNewObject (kLineType, &LineLRect, true);
			if (lineObjectHdl)
			{
				// set the line's start and end points 
				Map -> GetMatrixPoint (&LineStartPt, &LineStartLPt);
				Map -> GetMatrixPoint (&LineEndPt, &LineEndLPt);

				(**lineObjectHdl).lineStartLPoint = LineStartLPt;
				(**lineObjectHdl).lineEndLPoint = LineEndLPt;

				Map -> GetMapViewLRect (&MapViewLRect);
				GetDrawSettings (&drawSettings, nil, kScreenMode);
				GetDrawSettings (&drawSettings, (ObjectRecHdl) lineObjectHdl, kScreenMode);
				DrawObject ((ObjectRecHdl) lineObjectHdl, Map, &MapViewLRect, &drawSettings);
				ClearSelection (true, true);
				AddObjectToSelList ((ObjectRecHdl) lineObjectHdl, true, true);
			
				// change tool to arrow after creating new object 
//				mapToolSet -> SetCurrTool (ArrowTool);
				
				*bModified = true;		// new object was added 
			}
			
			bHandled = true;
		}*/
	
		SetPortGrafPort (SavePort);
	}
	else if (!strcmpnocase (currToolName, "Symbol Tool"))
	{}

	return (bHandled);
}
/**************************************************************************************************/
void CMapLayer::GetGroupLRect (CMyList *thisGroupOList, LongRect *theGroupLRectPtr)
{
	LongRect	thisGroupLRect;

	/* set group bounds to empty to start with */
	SetLRect (&thisGroupLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);

	GetGroupListLRect (thisGroupOList, &thisGroupLRect);

	*theGroupLRectPtr = thisGroupLRect;

	return;
}
/**************************************************************************************************/
void CMapLayer::ClearSelection (Boolean bInvert, Boolean bSetFlag)
{
	if (bInvert)
		InvertSelection (bSetFlag, false);

	layerSelectList -> ClearList ();

	return;
}
/**************************************************************************************************/
void CMapLayer::InvertSelection (Boolean bSetFlag, Boolean bTheFlag)
/* this subroutine inverts the current list of groups and objects, and if b-setflag is true, sets
	the flag of each object selected to b-theflag. */
{
	long			SelectObjCount, SelectObjIndex;
	LinkDataRec		thisSelData;
	CMyList			*thisGroupOList = nil;

	SelectObjCount = GetSelObjectCount ();
	for (SelectObjIndex = 0; SelectObjIndex < SelectObjCount; ++SelectObjIndex)
	{
		layerSelectList -> GetListItem ((Ptr) &thisSelData, SelectObjIndex);
		if (thisSelData.objectHandle)
		{
			InvertObject (Map, thisSelData.objectHandle);
			if (bSetFlag)
				SetObjectSelected (thisSelData.objectHandle, bTheFlag);
		}
		else if (thisSelData.objectList)
		{
			InvertGroup (thisSelData.objectList);
			if (bSetFlag)
				SetGroupSelected (thisSelData.objectList, bTheFlag);		/* recursive routine */
		}
	}

	return;
}
/**************************************************************************************************/
Boolean CMapLayer::IsObjectGrouped (ObjectRecHdl theObjectHdl, CMyList **theGroupListPtr,
									long *theGroupNumPtr)
{
	long	LayerGroupCount, GroupIndex;
	CMyList	*thisGroupOList = nil;
	Boolean	bIsInGroup = false;

	if (IsObjectFGrouped (theObjectHdl))
	{
		/* search each group to find the object */
		LayerGroupCount = GetLayerGroupCount ();
		for (GroupIndex = 1; GroupIndex <= LayerGroupCount; ++GroupIndex)
		{
			thisGroupOList = GetGroupObjectList (GroupIndex);
			if (IsObjectInList (theObjectHdl, thisGroupOList))
			{
				*theGroupListPtr = thisGroupOList;
				*theGroupNumPtr = GroupIndex;
				bIsInGroup = true;
				break;
			}
		}
	}

	return (bIsInGroup);
}
/**************************************************************************************************/
OSErr CMapLayer::AddObjectsToSelList (CMyList *theObjectList)
/* this subroutine add one-by-one each item in the given object list to the selection list */
{
	OSErr			ErrCode = 0;
	LinkDataRec		thisSelData, thisGroupData;
	CMyList			*thisObjGroupList = nil;
	long			ObjectCount, ObjectIndex;

	ObjectCount = theObjectList -> GetItemCount ();
	for (ObjectIndex = 0; ObjectIndex < ObjectCount; ++ObjectIndex)
	{
		theObjectList -> GetListItem ((Ptr) &thisGroupData, ObjectIndex);
		thisSelData.objectHandle = thisGroupData.objectHandle;
		thisSelData.objectList = thisGroupData.objectList;

		ErrCode = layerSelectList -> AppendItem ((Ptr) &thisSelData);
		if (ErrCode)
			break;
	}

	return (ErrCode);
}
/**************************************************************************************************/
Boolean CMapLayer::GroupSelection (Boolean bInvert)
/* this subroutine constructs a new group and adds it to the groups list.  It then adds all of the
	groups / objects that are currently selected (in the selection list).  It then clears the
	selection list and then adds the single new group to the selection list */
{
	OSErr			ErrCode = 0;
	long			SelectObjCount, SelectObjIndex;
	LinkDataRec		thisSelData;
	CMyList		   *thisGroupOList = nil, *NewGroupOList = nil;
	Boolean			bHandled = false;

	SelectObjCount = GetSelObjectCount ();
	if (SelectObjCount > 0)
	{
		NewGroupOList = AddNewGroup ();
		if (NewGroupOList != nil)
		{
			/* add all currently selected objects / groups to newly created group list */
			for (SelectObjIndex = SelectObjCount - 1; SelectObjIndex >= 0; --SelectObjIndex)
			{
				layerSelectList -> GetListItem ((Ptr) &thisSelData, SelectObjIndex);
				if (thisSelData.objectHandle != nil)
				{
					RemObjectFromSelList (thisSelData.objectHandle, bInvert, false);
					ErrCode = AddObjectToGroup (NewGroupOList, thisSelData.objectHandle);
					if (ErrCode)
						break;
				}
				else if (thisSelData.objectList != nil)
				{
					if (bInvert)
						InvertGroup (thisSelData.objectList);
						
					RemGroupFromSelList (thisSelData.objectList);
					ErrCode = AddOListToGroup (NewGroupOList, thisSelData.objectList);
					if (ErrCode)
						break;
				}
			}
	
			if (!ErrCode)	/* select the newly created group */
			{
				AddGroupToSelList (NewGroupOList);
				SelectGroup (NewGroupOList, bInvert, false, false);
				bHandled = true;			/* grouping was performed by this routine */
			}
		}
		else
			ErrCode = memFullErr;
	}

	return (bHandled);
}
/**************************************************************************************************/
Boolean CMapLayer::UngroupSelection (Boolean bInvert)
/* this subroutine */
{
	long			SelectObjCount, SelectObjIndex, GroupObjCount, GroupObjIndex;
	LinkDataRec		thisSelData, thisGroupData;
	long			theGroupNum;
	CMyList		   *thisGroupOList = nil, *thisObjGroupList = nil;
	ObjectRecHdl	firstObjectHdl;
	Boolean			bHandled = false;

	if (bInvert)
		InvertSelection (false, false);

	/* loop through the selection list to find all grouped objects */
	SelectObjCount = GetSelObjectCount ();
	for (SelectObjIndex = SelectObjCount - 1; SelectObjIndex >= 0; --SelectObjIndex)
	{
		layerSelectList -> GetListItem ((Ptr) &thisSelData, SelectObjIndex);
		if (thisSelData.objectList != nil)
		{			
			RemGroupFromSelList (thisSelData.objectList);
			AddObjectsToSelList (thisSelData.objectList);
			
			/* set the grouped-flag to false for each newly loosened object */
			GroupObjCount = thisSelData.objectList -> GetItemCount ();
			for (GroupObjIndex = 1; GroupObjIndex <= GroupObjCount; ++GroupObjIndex)
			{
				thisSelData.objectList -> GetListItem ((Ptr) &thisGroupData, GroupObjIndex);
				if (thisGroupData.objectHandle != nil)
					SetObjectGrouped (thisGroupData.objectHandle, false);
			}
			
			/* dispose of this list and remove it from the groups list */
			DeleteGroup (thisSelData.objectList);
			bHandled = true;		/* ungrouping was performed by this routine */
		}
	}

	if (bInvert)
		InvertSelection (false, false);

	return (bHandled);
}
/**************************************************************************************************/
Boolean CMapLayer::IsGroupSelected (CMyList *theObjectList)
/* Note: Only the top level of the selection is searched. */
{
	long			SelectObjCount, SelectObjIndex;
	LinkDataRec		thisSelData;
	Boolean			bIsSelected = false;

	SelectObjCount = GetSelObjectCount ();
	for (SelectObjIndex = SelectObjCount - 1; SelectObjIndex >= 0; --SelectObjIndex)
	{
		layerSelectList -> GetListItem ((Ptr) &thisSelData, SelectObjIndex);
		if (thisSelData.objectList != nil)
		{
			if (thisSelData.objectList == theObjectList)
			{
				bIsSelected = true;
				break;
			}
		}
	}

	return (bIsSelected);
}
/**************************************************************************************************/
void CMapLayer::DeleteGroup (CMyList *thisGroupOList)
{
	long	itemNum;

	if (layerGroupList -> IsItemInList ((Ptr) &thisGroupOList, &itemNum))
	{
		thisGroupOList -> Dispose ();
		layerGroupList -> DeleteItem (itemNum);
	}

	return;
}
/**************************************************************************************************/
OSErr CMapLayer::WriteLinkedList (short FRefNum, CMyList *theObjectList)
/* Recursive routine to write the given object list to disk.  This information should be written
	after the objects themselves so that valid handles may be assigned when reading the group link
	information.
*/
{
	OSErr			ErrCode = 0;
	LinkDataRec		thisObjectData, writeLinkData;
	long			objectCount, objectIndex, byteCount;
		
	/* get and write the item count for this list */
	objectCount = theObjectList -> GetItemCount ();
	byteCount = sizeof (objectCount);
	ErrCode = FSWrite (FRefNum, &byteCount, &objectCount);
	
	/* now write each element in the list */
	byteCount = sizeof (LinkDataRec);
	
	for (objectIndex = 0; objectIndex < objectCount; ++objectIndex)
	{
		theObjectList -> GetListItem ((Ptr) &thisObjectData, objectIndex);
		if (thisObjectData.objectList != nil)
		{
			/* write a null record signifying a new list */
			writeLinkData.objectList = nil;
			writeLinkData.objectHandle = nil;
			ErrCode = FSWrite (FRefNum, &byteCount, &writeLinkData);
			if (ErrCode)
				break;
			else
				WriteLinkedList (FRefNum, thisObjectData.objectList);
		}
		else if (thisObjectData.objectHandle != nil)
		{
			writeLinkData.objectList = nil;
			writeLinkData.objectHandle = (ObjectRecHdl) GetObjectIndex (thisObjectData.objectHandle, layerObjectList);

			ErrCode = FSWrite (FRefNum, &byteCount, &writeLinkData);
			if (ErrCode)
				break;
		}
	}

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::ReadLinkedList (short FRefNum, CMyList *theObjectList)
/* Recursive routine to read the given object list to disk. */
{
	OSErr			ErrCode = 0;
	LinkDataRec		thisObjectData, readDataLink;
	long			objectCount, objectIndex, byteCount;
	
	/* read the number of items in the list to be read */
	byteCount = sizeof (objectCount);
	ErrCode = FSRead (FRefNum, &byteCount, &objectCount);

	byteCount = sizeof (LinkDataRec);
	
	for (objectIndex = 1; objectIndex <= objectCount; ++objectIndex)
	{
		ErrCode = FSRead (FRefNum, &byteCount, &readDataLink);
		if (!ErrCode)
		{
			if (readDataLink.objectList == nil && readDataLink.objectHandle == nil)
			{
				thisObjectData.objectHandle = nil;
				thisObjectData.objectList = new CMyList (sizeof (LinkDataRec));
				if (thisObjectData.objectList != nil)
				{
					theObjectList -> AppendItem ((Ptr) &thisObjectData);
					ReadLinkedList (FRefNum, thisObjectData.objectList);
				}
				else
				{
					ErrCode = memFullErr;
					break;
				}
			}
			else if (readDataLink.objectHandle != nil)
			{
				thisObjectData.objectHandle = GetObjectHandle (layerObjectList, (long) readDataLink.objectHandle);
				thisObjectData.objectList = nil;
				theObjectList -> AppendItem ((Ptr) &thisObjectData);
			}
		}
		else
			break;
	}
	
	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::WriteLayerGroups (short FRefNum)
{
	long	LayerGroupCount, GroupIndex;
	CMyList	*thisGroupOList = nil;
	OSErr	ErrCode = 0;
	long	byteCount;
	OSType	groupTag;

	/* write tag marking start of groups data */
	byteCount = sizeof (groupTag);
	groupTag = 'GRPS';
	ErrCode = FSWrite (FRefNum, &byteCount, &groupTag);

	/* write the number of groups in the layer */
	LayerGroupCount = GetLayerGroupCount ();
	byteCount = sizeof (long);
	ErrCode = FSWrite (FRefNum, &byteCount, &LayerGroupCount);

	for (GroupIndex = LayerGroupCount; GroupIndex >= 1; --GroupIndex)
	{
		thisGroupOList = GetGroupObjectList (GroupIndex);
		ErrCode = WriteLinkedList (FRefNum, thisGroupOList);

		if (ErrCode)
			break;
	}

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::ReadLayerGroups (short FRefNum)
/* should be called after a 'GRPS' tag is read from the file.  This subroutine adds a new group to
	the layer and reads the data from the given file into the group list
*/
{
	long	LayerGroupCount, GroupIndex;
	CMyList	*newGroupOList = nil;
	OSErr	ErrCode = 0;
	long	byteCount, PastGroupsFPos;
	OSType	groupTag;

	/* read the file position marking end of groups data */
	byteCount = sizeof (long);
	ErrCode = FSRead (FRefNum, &byteCount, &PastGroupsFPos);

	/* read and verify groups data start tag */
	byteCount = sizeof (groupTag);
	ErrCode = FSRead (FRefNum, &byteCount, &groupTag);

	if (groupTag == 'GRPS')
	{
		/* read the number of groups in this layer */
		byteCount = sizeof (long);
		ErrCode = FSRead (FRefNum, &byteCount, &LayerGroupCount);
		if (!ErrCode)
		{
			for (GroupIndex = LayerGroupCount; GroupIndex >= 1; --GroupIndex)
			{
				newGroupOList = AddNewGroup ();
				if (newGroupOList != nil)
					ErrCode = ReadLinkedList (FRefNum, newGroupOList);
				else
					ErrCode = memFullErr;
		
				if (ErrCode)
					break;
			}
		}
	}
	else
		ErrCode = 1;		/* format error */

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::WriteLayer (short FRefNum)
/* subroutine to write out all objects in the layer as well as any grouping relationship between the
	objects within */
{
	OSErr			ErrCode = 0;
	ObjectRecHdl	thisObjectHdl;
	OSType			LayerTag;
	long			byteCount, structSize, LayerObjectCount, dummyLong,
					SaveFPos, CurrFPos, objectIndex;

	LayerObjectCount = GetLayerObjectCount ();
	if (LayerObjectCount > 0)
	{
		/* write the layer tag before saving the layer objects */
		LayerTag = 'LAYR';
		byteCount = sizeof (LayerTag);
		ErrCode = FSWrite (FRefNum, &byteCount, &LayerTag);
		
		/* write the current size of the layer header */
		byteCount = sizeof (long);
		structSize = sizeof (LayerInfoRec);
		ErrCode = FSWrite (FRefNum, &byteCount, &structSize);
		
		/* write the layer header */
		byteCount = sizeof (LayerInfoRec);
		ErrCode = FSWrite (FRefNum, &byteCount, (Ptr) &bLayerFirstField);

		if (!ErrCode)
		{
			/* write the object count before writing the object data */
			byteCount = sizeof (long);
			ErrCode = FSWrite (FRefNum, &byteCount, &LayerObjectCount);
		}
		
		if (!ErrCode)
		{
			for (objectIndex = 0; objectIndex < LayerObjectCount; ++objectIndex)
			{
				layerObjectList -> GetListItem ((Ptr) &thisObjectHdl, objectIndex);
				ErrCode = WriteObjectInfo (FRefNum, thisObjectHdl);
				if (ErrCode)	
					break;
			}
		}
		
		if (!ErrCode)
		{
			/* save a bogus placeholder to contain the fpos marking end of groups data */
			GetFPos (FRefNum, &SaveFPos);
			byteCount = sizeof (long);
			ErrCode = FSWrite (FRefNum, &byteCount, &dummyLong);
	
			if (!ErrCode)
				ErrCode = WriteLayerGroups (FRefNum);	/* write the layer groups info */
		}
			
		if (!ErrCode)	/* update file position (written above) marking end of groups data */
		{
			GetFPos (FRefNum, &CurrFPos);
			SetFPos (FRefNum, fsFromStart, SaveFPos);
			byteCount = sizeof (long);
			FSWrite (FRefNum, &byteCount, &CurrFPos);
			SetFPos (FRefNum, fsFromStart, CurrFPos);
		}
	}

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMapLayer::ReadLayer (short FRefNum)
/* subroutine to read in all objects in the layer as well as any grouping relationship between the
	objects within.  Note: This subroutine should be called after the layer tag has been read in
*/
{
	OSErr			ErrCode = 0;
	ObjectRecHdl	thisObjectHdl;
	LongRect		dummyLRect;
	long			structSize, byteCount, LayerObjectCount, objectIndex;
					
	/* read the size of the layer header at the time it was written */
	byteCount = sizeof (long);
	ErrCode = FSRead (FRefNum, &byteCount, &structSize);
	
	/* read the layer header */
	byteCount = structSize;
	ErrCode = FSRead (FRefNum, &byteCount, (Ptr) &bLayerFirstField);

	/* read the object-count for this layer */
	byteCount = sizeof (long);
	ErrCode = FSRead (FRefNum, &byteCount, &LayerObjectCount);
					
	for (objectIndex = 1; objectIndex <= LayerObjectCount; ++objectIndex)
	{
		ErrCode = ReadObjectInfo (FRefNum, &thisObjectHdl);
		if (ErrCode)
			break;
		else
			layerObjectList -> AppendItem ((Ptr) &thisObjectHdl);
	}

	if (!ErrCode)
		ErrCode = ReadLayerGroups (FRefNum);
	
	if (!ErrCode)
		GetLayerScope (&dummyLRect, true);

	return (ErrCode);
}
/**************************************************************************************************/
Boolean CMapLayer::DrawLayer (LongRect *UpdateLRectPtr, long Mode)
/* returns true if user interrupts drawing using escape or cmd period */
{
	long			ObjectIndex, ObjectCount, RefreshObjectCount = 0;
	Boolean			StopFlag = false;
	ObjectRecHdl	thisObjectHdl = nil;
	CMyList			*thisObjectList = nil;
	DrawSpecRec		drawSettings;

	GetDrawSettings (&drawSettings, (ObjectRecHdl) nil, Mode);

	thisObjectList = GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();
	//for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	// JLM 7/13/99 the smaller polygons are typically at the bottom of the BNA file
	// so we need to draw from the top to the bottom
	for (ObjectIndex = 0; ObjectIndex < ObjectCount; ObjectIndex++)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		if (ObjectSectLRect (thisObjectHdl, UpdateLRectPtr))
		{
			++RefreshObjectCount;
//			if (RefreshObjectCount > 2)		/* if more than two objects have needed updating */
//				SetWatchCursor ();
			
			/* get the settings needed for this particular object */
			GetDrawSettings (&drawSettings, thisObjectHdl, Mode);

			DrawObject (thisObjectHdl, Map, UpdateLRectPtr, &drawSettings);
		}
	}
	
	/* invert any currently selected objects in this layer */
	if (Mode == kScreenMode)
		InvertSelection (false, false);

//	Debug ("SelObjectCount = %d\n", GetSelObjectCount ());

	/* update active text if any */
//	UpdateText (WPtr, layerTextHandle);

	return (StopFlag);
}
/**************************************************************************************************/
Boolean CMapLayer::DoDragSelection (Point LocalClickPt)
{
	Rect		DeltaRect;
	LongRect	MapViewLRect, DeltaLRect;
	long		LDx, LDy;
	GrafPtr		SavePort;
	Point		MousePoint, LastPoint;
	Boolean		bMoved = false;
	PenState	penStatus;
	Pattern		GrayPattern;
	DrawSpecRec	drawSettings;
	
	GetPortGrafPtr (&SavePort);

	// keep looping until mouse moves past tolerence-pixels or button is released
	bMoved = false;
	while (StillDown ())
	{
		GetMouse (&MousePoint);
		if (abs (MousePoint.h - LocalClickPt.h) > 4 ||
			abs (MousePoint.v - LocalClickPt.v) > 4)
		{
			bMoved = true;
			LastPoint = LocalClickPt;
			break;
		}
	}

	if (bMoved)		// start dragging the selection
	{
		InvertSelection (false, false);				// remove the object sel-points

		GetPenState (&penStatus);					// save original pen state
#ifdef MAC
		StuffHex (&GrayPattern, kStdGrayPat);		// initialize gray pattern bits
		PenPat (&GrayPattern);						// gray pen for a dotted rectangle perimeter
#else
		FillPat(DARKGRAY);
		PenStyle(DARKGRAY, 2);
#endif
		PenMode (srcXor);							// exclusive or pen mode reverses pixel states

		GetDrawSettings (&drawSettings, nil, kDragMode);
		drawSettings.fillCode = kNoFillCode;

		// draw the first set of objects in xor mode
		Map -> GetMapViewLRect (&MapViewLRect);
		DrawObjectsInList (layerSelectList, Map, &MapViewLRect, &drawSettings);

		do
		{
			//if (LastPoint != MousePoint)
			if (!EqualPoints(LastPoint,MousePoint))
			{
				// erase the objects that were drawn before
				DrawObjectsInList (layerSelectList, Map, &MapViewLRect, &drawSettings);

				drawSettings.offsetDx = MousePoint.h - LocalClickPt.h;
				drawSettings.offsetDy = MousePoint.v - LocalClickPt.v;

				// draw a new set of objects in xor mode
				DrawObjectsInList (layerSelectList, Map, &MapViewLRect, &drawSettings);

				LastPoint = MousePoint;
			}

			GetMouse (&MousePoint);
		}
		while (Button ());

		// erase the objects that were drawn last
		DrawObjectsInList (layerSelectList, Map, &MapViewLRect, &drawSettings);

		// now calculate the delta X and delta Y in terms of matrix coordinates
		DeltaRect.top = 0;
		DeltaRect.left = 0;
		DeltaRect.bottom = drawSettings.offsetDy;
		DeltaRect.right = drawSettings.offsetDx;
		Map -> GetMatrixLRect (&DeltaRect, &DeltaLRect);
		LDx = DeltaLRect.right  - DeltaLRect.left;
		LDy = DeltaLRect.bottom - DeltaLRect.top;

		// now offset each selected object by the delta values
		OffsetObjectsInList (layerSelectList, Map, true, LDx, LDy);
		SetLayerModified (true);
//		EventDP -> DPCommand (RecalcDocBoundsCmd, nil, nil, nil, nil);

		// now set the undo codes and offsets
		UndoLayerObjects ();
		UndoObjectsInList (layerSelectList, true);
		SetObjectLRect ((ObjectRecHdl) layerUndoObjectHdl, &DeltaLRect);
		layerUndoCode = kUndoDragCode;

		SetPenState (&penStatus);					// restore original pen state / mode
		InvertSelection (false, false);				// redraw the object sel-points
	}

	SetPortGrafPort (SavePort);

	return (bMoved);
}
/**************************************************************************************************/
long CMapLayer::GetSelObjectCount ()
{
	long	NumSelObjects;

	NumSelObjects = layerSelectList -> GetItemCount ();

	return (NumSelObjects);
}
/**************************************************************************************************/
Boolean CMapLayer::DoGrowSelection (Point LocalClickPt)
{
	LongRect	SelectionLRect, NewSelectLRect;
	Rect		SelectionRect, WindowDrawRect;
	Boolean		bHandled = false;
	LinkDataRec	thisSelData;
	OSErr		ErrCode = 0;
	ScaleRec	GrowScaleInfo;

	if (GetSelObjectCount () == 1)
	{
		layerSelectList -> GetListItem ((Ptr) &thisSelData, 0);
		if (thisSelData.objectHandle)
			GetObjectLRect (thisSelData.objectHandle, &SelectionLRect);
		else
			GetGroupLRect (thisSelData.objectList, &SelectionLRect);

		Map -> GetMapDrawRect (&WindowDrawRect);
		Map -> GetScrRect (&SelectionLRect, &SelectionRect);

//JLM		ErrCode = Map -> DoGrowRect (&SelectionRect, LocalClickPt, &WindowDrawRect);
		if (!ErrCode)
		{
			Map -> GetMatrixLRect (&SelectionRect, &NewSelectLRect);
			GetLScaleAndOffsets (&SelectionLRect, &NewSelectLRect, &GrowScaleInfo);
			
			ScaleObjectsInList (layerSelectList, Map, true, &GrowScaleInfo);
			SetLayerModified (true);
//			EventDP -> DPCommand (RecalcDocBoundsCmd, nil, nil, nil, nil);
			
			/* set the undo object parameters */
			UndoLayerObjects ();
			UndoObjectsInList (layerSelectList, true);
			
			(**layerUndoObjectHdl).growFromLRect = NewSelectLRect;
			(**layerUndoObjectHdl).growToLRect   = SelectionLRect;

			layerUndoCode = kUndoGrowCode;

			bHandled = true;
		}
	}

	return (bHandled);
}
/**************************************************************************************************/
ObjectRecHdl CMapLayer::FindClickedObject (Point LocalClickPt)
{
	long			ObjectCount, ObjectIndex;
	LongRect		thisObjectLRect;
	//Rect			thisObjectRect;
	ObjectRecHdl	ClickedObjectHdl = nil;
	//LinkDataRec		thisSelData;

	/* check to see if any selected-object's corners have been clicked on */
//	ObjectCount = GetSelObjectCount ();
//	if (ObjectCount > 0)
//	{
//		for (ObjectIndex = 0; ObjectIndex < ObjectCount; ++ObjectIndex)
//		{
//			layerSelectList -> GetListItem ((Ptr) &thisSelData, ObjectIndex);
//			if (thisSelData.objectHandle)
//				GetObjectRect (thisSelData.objectHandle, &thisObjectRect);
//			else
//			{
//				GetGroupLRect (thisSelData.objectList, &thisObjectLRect);
//				Map -> GetScrRect (&thisObjectLRect, &thisObjectRect);
//			}
//
//			MyInsetRect (&thisObjectRect, -2, -2);
//			if (PtInRect (LocalClickPt, &thisObjectRect))
//			{
//				MyInsetRect (&thisObjectRect, 5, 5);
//				if (!PtInRect (LocalClickPt, &thisObjectRect))
//				{
//					if (thisSelData.objectHandle)
//					{
//						ClickedObjectHdl = thisSelData.objectHandle;
//					}
//					else
//						ClickedObjectHdl = thisSelData.objectList;
//				}
//			}
//		}
//	}
	
	if (ClickedObjectHdl == nil)
	{
		CMyList		*thisObjectList = nil;

		/* get this layer's objects list */		
		thisObjectList = GetLayerObjectList ();
		if (thisObjectList != nil)
		{
			ObjectRecHdl	thisObjectHdl;
			long			clickedPart;
	
			ObjectCount = thisObjectList -> GetItemCount ();
			for (ObjectIndex = 0; ObjectIndex < ObjectCount; ++ObjectIndex)
			{
				thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
				GetObjectLRect (thisObjectHdl, &thisObjectLRect);
				if (PointOnObject (Map, thisObjectHdl, LocalClickPt, &clickedPart))
				{
					ClickedObjectHdl = thisObjectHdl;
					break;
				}
			}
		}
	}

	return (ClickedObjectHdl);
}
/**************************************************************************************************/
Boolean CMapLayer::PerformUndo ()
{
	long			ObjectCount, ObjectIndex;
	ObjectRecHdl	thisObjectHdl;
	Boolean			bHandled = false;
	
	if (GetSelObjectCount () > 0)
	{
		switch (layerUndoCode)
		{
			case kUndoDragCode:
			{
				LongRect	UndoObjectLRect;
				long		LDx, LDy, tempLong;

				GetObjectLRect ((ObjectRecHdl) layerUndoObjectHdl, &UndoObjectLRect);
				LDx = UndoObjectLRect.left - UndoObjectLRect.right;
				LDy = UndoObjectLRect.top  - UndoObjectLRect.bottom;
	
				InvertSelection (false, false);		/* unhilite before moving */
	
				ObjectCount = layerObjectList -> GetItemCount ();
				for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
				{
					layerObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
					if (IsObjectUndone (thisObjectHdl))
						OffsetObject (thisObjectHdl, Map, LDx, LDy, true);
				}

				/* flip the undo object handle's offsets */
				tempLong = UndoObjectLRect.left;
				UndoObjectLRect.left = UndoObjectLRect.right;
				UndoObjectLRect.right = tempLong;

				tempLong = UndoObjectLRect.top;
				UndoObjectLRect.top = UndoObjectLRect.bottom;
				UndoObjectLRect.bottom = tempLong;
				SetObjectLRect ((ObjectRecHdl) layerUndoObjectHdl, &UndoObjectLRect);

				InvertSelection (false, false);	/* rehilite after moving */

				bHandled = true;
			}
			break;

			case kUndoGrowCode:
			{
				ScaleRec	growScaleInfo;
				LongRect	growFromLRect, growToLRect;
				
				growFromLRect = (**layerUndoObjectHdl).growFromLRect;
				growToLRect   = (**layerUndoObjectHdl).growToLRect;
				
				GetLScaleAndOffsets (&growFromLRect, &growToLRect, &growScaleInfo);
	
				InvertSelection (false, false);		/* unhilite before moving */
	
				ObjectCount = layerObjectList -> GetItemCount ();
				for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
				{
					layerObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
					if (IsObjectUndone (thisObjectHdl))
						ScaleObject (thisObjectHdl, Map, &growScaleInfo, true);
				}
	
				/* swap the undo object's to / from rects */
				(**layerUndoObjectHdl).growFromLRect = growToLRect;
				(**layerUndoObjectHdl).growToLRect   = growFromLRect;
				
				InvertSelection (false, false);	/* rehilite after moving */

				bHandled = true;
			}
			break;
		}
	}
	
	return (bHandled);
}
/**************************************************************************************************/
void CMapLayer::UndoLayerObjects ()
/* subroutine to create and add a new poly region handle.
	Note: default parameters are also inserted into certain poly data fields */
{
	long			ObjectCount, ObjectIndex;
	Handle			ListDataHdl;
	ObjectRecHdl	thisObjectHdl;

	ObjectCount = layerObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		layerObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		SetObjectUndone (thisObjectHdl, false);
	}

	return;
}
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
void CMapLayer::DoIdle ()
{
	return;
}
/**************************************************************************************************/
Boolean CMapLayer::DoMenu (long menuResult, char *ChoiceStr, Boolean *modified)
{
	short	menuID, menuItem;
	Boolean	bHandled = false;

	/* get menu id and item number for checking modified view save menu */
	menuID = HiWord (menuResult);
	menuItem = LoWord (menuResult);
			
	if (GetSelObjectCount () > 0)
	{
		if (!strcmpnocase (ChoiceStr, "Group"))
			bHandled = GroupSelection (true);
	
		if (!strcmpnocase (ChoiceStr, "Ungroup"))
			bHandled = UngroupSelection (true);
			
//		if (!strcmpnocase (ChoiceStr, "Bring to Front"))
//			bHandled = DoBringToFront (Map, layerObjectList);
	
//		if (!strcmpnocase (ChoiceStr, "Send to Back"))
//			bHandled = DoSendToBack (Map, layerObjectList);
	
		if (!strcmpnocase (ChoiceStr, "Shuffle Up"))
		{
//			DoBringToFront (layerObjectList);
//			bHandled = true;
		}
	
		if (!strcmpnocase (ChoiceStr, "Shuffle Down"))
		{
//			DoBringToFront (layerObjectList);
//			bHandled = true;
		}

		if (!strcmpnocase (ChoiceStr, "Remove") ||
			!strcmpnocase (ChoiceStr, "Clear"))
		{
			DeleteSelection ();
			bHandled = true;
		}
	}

	if (!strcmpnocase (ChoiceStr, "Undo"))
		bHandled = PerformUndo ();

	if (!bHandled)
		bHandled = DoTextMenu (menuResult, ChoiceStr);

	if (bHandled)
	{
		SetLayerModified (true);
		*modified = true;
	}

	return (bHandled);
}
/**************************************************************************************************/
Boolean CMapLayer::DoTextMenu (long menuResult, char *ChoiceStr)
/* returns true if handled */
{
	short	menuID, menuItem;
	Boolean	bHandled = false;

	/* get menu id and item number for checking modified view save menu */
	menuID = HiWord (menuResult);
	menuItem = LoWord (menuResult);
		
	if (bHandled)
		SetLayerModified (true);
	
	return (bHandled);
}
/**************************************************************************************************/
void CMapLayer::DeleteSelection ()
{
	OSErr			ErrCode = 0;
	ObjectRecHdl	thisObjectHdl;
	
	if (GetSelObjectCount () > 0)
	{
		InvertSelection (false, false);
		
		ErrCode = layerObjectList -> GetLastItem ((Ptr) &thisObjectHdl);
		while (!ErrCode)
		{
			if (IsObjectSelected (thisObjectHdl))
				DeleteObject (thisObjectHdl, true);

			ErrCode = layerObjectList -> GetNextItem ((Ptr) &thisObjectHdl);
		}

		ClearSelection (false, false);
	}
	
	return;
}
/**************************************************************************************************/
Boolean CMapLayer::DoObjectDblClick (ObjectRecHdl ClickedObjectHdl)
{
	Boolean	bHandled = false;
	OSType	thisObjectType;
	
	GetObjectType (ClickedObjectHdl, &thisObjectType);
	if (thisObjectType == kSymbolType)
	{
		Rect	SymbolScrRect;
		//GrafPtr	SavePort;
		
		InvertSelection (false, false);		/* deselect object */
		
		/* get and store the symbol's current screen rect */
		GetObjectScrRect (Map, ClickedObjectHdl, &SymbolScrRect);

/*		if (DoDialog (EventDP, SymbolDlogID, "", (Ptr) &ClickedObjectHdl) == OKButton)
		{
			// inval old symbol rect before drawing in a possibly new position
			GetPortGrafPtr (&SavePort);
			InvalRect (&SymbolScrRect);
			SetPortGrafPort (SavePort);
			
			// inval a possibly new symbol rect
			InvalObjectRect (Map, ClickedObjectHdl);
		}
*/		
		InvertSelection (false, false);		/* reselect object */

		bHandled = true;
	}

	return (bHandled);
}
/**************************************************************************************************/
void CMapLayer::DeleteObject (ObjectRecHdl theObjectHdl, Boolean bInvalidate)
{
	long	ObjectIndex;
	
	if (bInvalidate)
		InvalObjectRect (Map, theObjectHdl);
	
	if (layerObjectList -> IsItemInList ((Ptr) &theObjectHdl, &ObjectIndex))
	{
		layerObjectList -> DeleteItem (ObjectIndex);
		RemoveObjectFromList (layerSelectList, theObjectHdl);
		RemoveObjectFromList (layerGroupList,  theObjectHdl);
	}

	return;
}
/**************************************************************************************************/
OSErr CMapLayer::ClipLayerToScreen ()
{
	long			ObjectIndex, ObjectCount;
	CMyList			*thisObjectList = nil;
	ObjectRecHdl	thisObjectHdl;
	OSType			thisObjectType;
	LongRect		sectLRect;
	Rect			ScrPolyRect;
	OSErr			err = noErr;

	Map -> GetMapDrawRect (&ScrPolyRect);
	MyInsetRect (&ScrPolyRect, -1, -1);
	Map -> GetMatrixLRect (&ScrPolyRect, &sectLRect);

	thisObjectList = GetLayerObjectList ();
	MarkObjectsInList (thisObjectList, false);		/* unmark all objects */
	
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		GetObjectType (thisObjectHdl, &thisObjectType);
		if (thisObjectType == kPolyType)
		{
			err = IntersectPoly ((PolyObjectHdl) thisObjectHdl, &sectLRect, thisObjectList);
			if (err)
				break;
		}
	}
	
	/* now removed all unmarked polygons from the objects list */
	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		GetObjectType (thisObjectHdl, &thisObjectType);
		if (thisObjectType == kPolyType)
		{
			if (!IsObjectMarked ((ObjectRecHdl) thisObjectHdl))
				thisObjectList -> DeleteItem (ObjectIndex);
		}
	}

	return err;
}
