
#include "Cross.h"
#include	"ObjectUtils.h"
#include	"GenDefs.h"
/**************************************************************************************************/

#ifdef IBM
void BitClr(void *bytePtr, long bitNum)
{
	long byteNum = bitNum / 8;
	long bitWithinByte = bitNum % 8;
	unsigned char* p = (unsigned char*)bytePtr;
	unsigned char byte = p[byteNum];
	
	//if (byte & (1 << bitWithinByte))  // IBM style
	if (byte & (1 << (7-bitWithinByte))) // MAC style
	{
		//byte -= 1 << bitWithinByte;	 // IBM style
		byte -= 1 << (7-bitWithinByte);	 // MAC style
		p[byteNum] = byte;
	}
}
void BitSet(void *bytePtr, long bitNum)
{
	long byteNum = bitNum / 8;
	long bitWithinByte = bitNum % 8;
	char* p = (char*)bytePtr;
	char byte = p[byteNum];

	//byte |= 1 << bitWithinByte;	 // IBM style
	byte |= 1 << (7-bitWithinByte);	 // MAC style
	p[byteNum] = byte;
}
Boolean BitTst(void *bytePtr, long bitNum)
{
	long byteNum = bitNum / 8;
	long bitWithinByte = bitNum % 8;
	char* p = (char*)bytePtr;
	char byte = p[byteNum];

	//return (byte & (1 << bitWithinByte)) != 0;	 // IBM style
	return (byte & (1 << (7-bitWithinByte))) != 0;	 // MAC style
}
#endif


Boolean operator == (ObjectIDRec Obj1ID, ObjectIDRec Obj2ID)
{
	if (Obj1ID.hi == Obj2ID.hi && Obj1ID.lo == Obj2ID.lo)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
void GetUniqueID (ObjectIDRecPtr idPtr, unsigned short idCounter)
// this subroutine returns an id that is probably going to be very unique.  It based on current date
//	and time as well as the time since the machine was turned on.  It also takes into account a
//	global id counter that is passed into the routine
{
	static long		ticks = 0;
	unsigned long	seconds;
	long			pair;
	short			count, sTicks;

	if (ticks == 0)
		ticks = TickCount ();

	sTicks = (short) ticks;
	GetDateTime (&seconds);
	count = idCounter;
	pair = count;

	*(short*) &pair = sTicks;
	idPtr -> lo = pair;
	idPtr -> hi = seconds;

	return;
}
/**************************************************************************************************/
Boolean IsIdZero (ObjectIDRecPtr idPtr)
{
	if (idPtr -> lo == -1 && idPtr -> hi == -1)
		return (true);
	else
		return (false);
}
/**************************************************************************************************/
void GetObjectID (ObjectRecHdl ObjectHdl, ObjectIDRecPtr ObjectIDPtr)
{
	*ObjectIDPtr = (**ObjectHdl).objectID;

	return;
}
/**************************************************************************************************/
void SetObjectID (ObjectRecHdl ObjectHdl, ObjectIDRecPtr ObjectIDPtr)
{
	(**ObjectHdl).objectID = *ObjectIDPtr;

	return;
}
/**************************************************************************************************/
long GetOLabelJust (ObjectRecHdl ObjectHdl)
{
	long	theJustCode;

	theJustCode = (**ObjectHdl).objectLabelJust;

	return (theJustCode);
}
/**************************************************************************************************/
void SetOLabelJust (ObjectRecHdl ObjectHdl, long justCode)
{
	(**ObjectHdl).objectLabelJust = justCode;
	
	return;
}
/**************************************************************************************************/
ObjectRecHdl GetOGroupToHdl (ObjectRecHdl objectHdl)
{
	ObjectRecHdl	groupToObjectHdl;
	
	groupToObjectHdl = (**objectHdl).groupToObjectHdl;
	
	return (groupToObjectHdl);
}
/**************************************************************************************************/
void SetObjectGrouped (ObjectRecHdl thisObjectHdl, Boolean bGroupedFlag)
{
	short	objectFlags;

	objectFlags = (**thisObjectHdl).objectFlags;
	if (bGroupedFlag)
		BitSet (&objectFlags, kGroupedMask);
	else
		BitClr (&objectFlags, kGroupedMask);
	(**thisObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
Boolean IsObjectFGrouped (ObjectRecHdl thisObjectHdl)
{
	Boolean	GroupedFlag;
	short	objectFlags;
	
	objectFlags = (**thisObjectHdl).objectFlags;
	GroupedFlag = BitTst (&objectFlags, kGroupedMask);	
	
	return (GroupedFlag);
}
/**************************************************************************************************/
void KillObject (ObjectRecHdl ObjectHdl)
// this subroutine releases the memory occupied by the given object handle
{
	OSType			thisObjectType;
	PolyObjectHdl	thisPolyHdl;
	LongPoint		**thisPointsHdl;
	
	if (ObjectHdl != nil)
	{
		GetObjectType (ObjectHdl, &thisObjectType);
		
		// dispose of any non-standard object data structures first
		switch (thisObjectType)
		{
			case kPolyType:
				thisPolyHdl = (PolyObjectHdl) ObjectHdl;
				thisPointsHdl = (LongPoint**) (**thisPolyHdl).objectDataHdl;
				if (thisPointsHdl != nil)
				{
					DisposeHandle ((Handle) thisPointsHdl);
					(**thisPolyHdl).objectDataHdl = nil;
				}
			break;
		}
		
		DisposeHandle ((Handle) ObjectHdl);		// finally, dispose the object handle
	}
	
	return;
}
/**************************************************************************************************/
void SetObjectLabel (ObjectRecHdl ObjectHdl, char *ObjectName)
{
	_HLock ((Handle) ObjectHdl);
	
	if (strlen (ObjectName) < kObjectNameLen)
		strcpy ((**ObjectHdl).objectLabel, ObjectName);
	else
		strncpy ((**ObjectHdl).objectLabel, ObjectName, kObjectNameLen - 1);

	_HUnlock ((Handle) ObjectHdl);
	
	return;
}
/**************************************************************************************************/
void GetObjectLabel (ObjectRecHdl objectHdl, char *objectName)
{
	_HLock ((Handle) objectHdl);
	strcpy (objectName, (**objectHdl).objectLabel);
	_HUnlock ((Handle) objectHdl);

	return;
}
/**************************************************************************************************/
void GetObjectType (ObjectRecHdl ObjectHdl, OSType *ObjectTypePtr)
{
	*ObjectTypePtr = (**ObjectHdl).objectType;

	return;
}
/**************************************************************************************************/
void SetObjectType (ObjectRecHdl ObjectHdl, OSType theObjectType)
{
	(**ObjectHdl).objectType = theObjectType;

	return;
}
/**************************************************************************************************/
void GetObjectLRect (ObjectRecHdl ObjectHdl, LongRect *ObjectLRectPtr)
{
	*ObjectLRectPtr = (**ObjectHdl).objectLRect;

	return;
}
/**************************************************************************************************/
void GetObjectLPoint (ObjectRecHdl ObjectHdl, LongPoint *ObjectLPointPtr)
{
	*ObjectLPointPtr = (**ObjectHdl).objectLPoint;

	return;
}
/**************************************************************************************************/
void SetObjectLPoint (ObjectRecHdl ObjectHdl, LongPoint *objectLPointPtr)
{
	OSType	thisObjectType;
		
	(**ObjectHdl).objectLPoint = *objectLPointPtr;
	GetObjectType (ObjectHdl, &thisObjectType);
	if (thisObjectType == kSymbolType)
	{
		LongRect	thisObjectLRect;
		
		SetLRect (&thisObjectLRect, objectLPointPtr -> h, objectLPointPtr -> v,
									objectLPointPtr -> h, objectLPointPtr -> v);
									
		SetObjectLRect (ObjectHdl, &thisObjectLRect);
	}
	
	return;
}
/**************************************************************************************************/
void SetObjectLRect (ObjectRecHdl ObjectHdl, LongRect *ObjectLRectPtr)
{
	(**ObjectHdl).objectLRect = *ObjectLRectPtr;

	return;
}
/**************************************************************************************************/
void SetObjectVisible (ObjectRecHdl ObjectHdl, Boolean VisibleFlag)
{
	short	objectFlags;
	
	objectFlags = (**ObjectHdl).objectFlags;
	if (VisibleFlag)
		BitSet (&objectFlags, kVisibleMask);
	else
		BitSet (&objectFlags, kVisibleMask);
	(**ObjectHdl).objectFlags = objectFlags;

	return;
}
/**************************************************************************************************/
Boolean IsOLabelVisible (ObjectRecHdl ObjectHdl)
{
	Boolean VisibleFlag;
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	VisibleFlag = BitTst (&objectFlags, kLabelVisMask);	

	return (VisibleFlag);
}
/**************************************************************************************************/
void SetOLabelVisible (ObjectRecHdl ObjectHdl, Boolean LabelVisFlag)
{
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	if (LabelVisFlag)
		BitSet (&objectFlags, kLabelVisMask);
	else
		BitClr (&objectFlags, kLabelVisMask);
	(**ObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
void SetOHotLinked (ObjectRecHdl ObjectHdl, Boolean HotLinkedFlag)
{
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	if (HotLinkedFlag)
		BitSet (&objectFlags, kHotLinkedMask);
	else
		BitClr (&objectFlags, kHotLinkedMask);
	(**ObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
void SetOSymbolVisible (ObjectRecHdl ObjectHdl, Boolean SymVisFlag)
{
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	if (SymVisFlag)
		BitSet (&objectFlags, kSymbolVisMask);
	else
		BitClr (&objectFlags, kSymbolVisMask);
	(**ObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
Boolean IsObjectVisible (ObjectRecHdl ObjectHdl)
{
	Boolean VisibleFlag;
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	VisibleFlag = BitTst (&objectFlags, kVisibleMask);	

	return (VisibleFlag);
}
/**************************************************************************************************/
Boolean IsObjectUndone (ObjectRecHdl ObjectHdl)
{
	Boolean bUndoFlag;
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	bUndoFlag = BitTst (&objectFlags, kUndoMask);	

	return (bUndoFlag);
}
/**************************************************************************************************/
void SetObjectUndone (ObjectRecHdl ObjectHdl, Boolean bUndoFlag)
{
	short	objectFlags;
	
	objectFlags = (**ObjectHdl).objectFlags;
	if (bUndoFlag)
		BitSet (&objectFlags, kUndoMask);
	else
		BitClr (&objectFlags, kUndoMask);
	(**ObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
void SetObjectSelected (ObjectRecHdl ObjectHdl, Boolean SelectedFlag)
{
	short	objectFlags;
	
	objectFlags = (**ObjectHdl).objectFlags;
	if (SelectedFlag)
		BitSet (&objectFlags, kSelectedMask);
	else
		BitClr (&objectFlags, kSelectedMask);
	(**ObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
Boolean IsObjectSelected (ObjectRecHdl ObjectHdl)
{
	Boolean	SelectedFlag;
	short	objectFlags;
	
	objectFlags = (**ObjectHdl).objectFlags;
	SelectedFlag = BitTst (&objectFlags, kSelectedMask);	
	
	return (SelectedFlag);
}
/**************************************************************************************************/
void SetObjectMarked (ObjectRecHdl ObjectHdl, Boolean MarkedFlag)
{
	short	objectFlags;

	objectFlags = (**ObjectHdl).objectFlags;
	if (MarkedFlag)
		BitSet (&objectFlags, kMarkedMask);
	else
		BitClr (&objectFlags, kMarkedMask);
	(**ObjectHdl).objectFlags = objectFlags;	

	return;
}
/**************************************************************************************************/
Boolean IsObjectMarked (ObjectRecHdl ObjectHdl)
{
	Boolean	MarkedFlag;
	short	objectFlags;
	
	objectFlags = (**ObjectHdl).objectFlags;
	MarkedFlag = BitTst (&objectFlags, kMarkedMask);
	
	return (MarkedFlag);
}
/**************************************************************************************************/
void SetObjectColor (ObjectRecHdl ObjectHdl, long objectColorInd)
{
	(**ObjectHdl).objectColorInd = objectColorInd;
		
	return;
}
/**************************************************************************************************/
void SetObjectESICode (ObjectRecHdl ObjectHdl, long theESICode)
{
	(**ObjectHdl).objectESICode = theESICode;
		
	return;
}
/**************************************************************************************************/
void GetObjectESICode (ObjectRecHdl ObjectHdl, long *theESICode)
{
	*theESICode = (**ObjectHdl).objectESICode;
		
	return;
}
/**************************************************************************************************/
void SetObjectBColor (ObjectRecHdl ObjectHdl, long objectBColorInd)
{
	(**ObjectHdl).objectBColorInd = objectBColorInd;
		
	return;
}
/**************************************************************************************************/
void SetObjectLabelRect (ObjectRecHdl ObjectHdl, Rect *theScrRect)
// usually called from object's draw procedure
{
	(**ObjectHdl).objectLabelRect = *theScrRect;

	return;
}
/**************************************************************************************************/
void GetObjectLabelRect (ObjectRecHdl ObjectHdl, Rect *theScrRect)
{
	*theScrRect = ((**ObjectHdl).objectLabelRect);

	return;
}
/**************************************************************************************************/
long GetObjectColor (ObjectRecHdl ObjectHdl)
{
	return ((**ObjectHdl).objectColorInd);
}
/**************************************************************************************************/
long GetObjectBColor (ObjectRecHdl ObjectHdl)
{
	return ((**ObjectHdl).objectBColorInd);
}
/**************************************************************************************************/
void SetObjectSymbol (ObjectRecHdl thisObjectHdl, long objectSymbol)
{
	(**thisObjectHdl).objectSymbol = objectSymbol;
	
	return;
}
/**************************************************************************************************/
long GetObjectSymbol (ObjectRecHdl thisObjectHdl)
{
	long	objectSymbol;
	
	objectSymbol = (**thisObjectHdl).objectSymbol;
	
	return (objectSymbol);
}
/**************************************************************************************************/
double GetObjectArea (ObjectRecHdl thisObjectHdl)
{
	double		objectArea;
	LongRect	objectLRect;
	
	GetObjectLRect (thisObjectHdl, &objectLRect);
	
	objectArea = ((double) (objectLRect.top   - objectLRect.bottom)) *
			     ((double) (objectLRect.right - objectLRect.left));

	return objectArea;
}
/**************************************************************************************************/
void SortObjectsBySize (CMyList *thisObjectList)
// subroutine to create and add a new poly region handle.
//	Note: default parameters are also inserted into certain poly data fields
{
	long		objectCount;
	Handle		ListDataHdl;
	Ptr			ListDataPtr;
	Size		ListDataSize;

	objectCount = thisObjectList -> GetItemCount ();
	ListDataHdl = thisObjectList -> GetListDataHdl ();
	ListDataSize = thisObjectList -> GetItemSize ();

	_HLock (ListDataHdl);

	ListDataPtr = *ListDataHdl;
	qsort (ListDataPtr, objectCount, ListDataSize, ObjSizeComp);

	_HUnlock (ListDataHdl);

	return;
}
/**************************************************************************************************/
int ObjSizeComp (const void *Object1HdlPtr, const void *Object2HdlPtr)
{
	double			Object1Area, Object2Area;
	ObjectRecHdl	Object1Hdl, Object2Hdl;
	
	// copy / convert object handles
	BlockMove ((VOIDPTR)Object1HdlPtr, &Object1Hdl, 4);
	BlockMove ((VOIDPTR)Object2HdlPtr, &Object2Hdl, 4);
	
	// get object areas to be compared
	Object1Area = GetObjectArea (Object1Hdl);
	Object2Area = GetObjectArea (Object2Hdl);

	if (Object1Area < Object2Area)
		return (-1);
	else if (Object1Area == Object2Area)
		return (0);
	else
		return (1);
}
/**************************************************************************************************/
void GetObjectCenter (ObjectRecHdl thisObjectHdl, LongPoint *CenterLPtPtr)
{
	LongRect	ObjectLRect;
	
	GetObjectLRect (thisObjectHdl, &ObjectLRect);

	CenterLPtPtr -> h = ObjectLRect.left   + ((ObjectLRect.right - ObjectLRect.left)   / 2);
	CenterLPtPtr -> v = ObjectLRect.bottom + ((ObjectLRect.top   - ObjectLRect.bottom) / 2);

	return;
}
/**************************************************************************************************/
void MarkObjectsInList (CMyList *thisObjectList, Boolean bMark)
// subroutine to create and add a new poly region handle.
//	Note: default parameters are also inserted into certain poly data fields
{
	long			ObjectCount, ObjectIndex;
	Handle			ListDataHdl;
	ObjectRecHdl	thisObjectHdl;

	ObjectCount = thisObjectList -> GetItemCount ();
	for (ObjectIndex = ObjectCount; ObjectIndex >= 1; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		SetObjectMarked (thisObjectHdl, bMark);
	}

	return;
}
/**************************************************************************************************/
void SetOLabelLPoint (ObjectRecHdl thisObjectHdl, LongPoint *LabelLPointPtr)
// initializes the top-left coordinates of the object label rect to be the supplied point.  This is
//	needed for objects that only have a label point and do not have their label-zoom flag set
{
	(**thisObjectHdl).objectLabelLPoint = *LabelLPointPtr;

	return;
}
/**************************************************************************************************/
void GetOLabelLPoint (ObjectRecHdl thisObjectHdl, LongPoint *LabelLPointPtr)
// returns the top-left coordinates of the object label rect.  This is needed for objects that only
//	have a label point and do not have their label-zoom flag set
{
	*LabelLPointPtr = (**thisObjectHdl).objectLabelLPoint;

	return;
}
/**************************************************************************************************/
void SetObjectDataHdl (ObjectRecHdl thisObjectHdl, Handle theDataHdl)
// initializes the top-left coordinates of the object label rect to be the supplied point.  This is
//	needed for objects that only have a label point and do not have their label-zoom flag set
{
	if ((**thisObjectHdl).objectDataHdl != nil)
		DisposeHandle ((**thisObjectHdl).objectDataHdl);

	(**thisObjectHdl).objectDataHdl = theDataHdl;

	return;
}
/**************************************************************************************************/
Handle GetObjectDataHdl (ObjectRecHdl thisObjectHdl)
// initializes the top-left coordinates of the object label rect to be the supplied point.  This is
//	needed for objects that only have a label point and do not have their label-zoom flag set
{
	return ((**thisObjectHdl).objectDataHdl);
}
/**************************************************************************************************/
void SetObjectCustData (ObjectRecHdl thisObjectHdl, long theCustData)
{
	(**thisObjectHdl).objectCustData = theCustData;

	return;
}
/**************************************************************************************************/
long GetObjectCustData (ObjectRecHdl thisObjectHdl)
{
	return ((**thisObjectHdl).objectCustData);
}
/**************************************************************************************************/
void SelectObject (CMap *Map, ObjectRecHdl thisObjectHdl, Boolean Invert, Boolean bSetFlag,
				   Boolean bTheFlag)
{
	if (bSetFlag)
		SetObjectSelected (thisObjectHdl, bTheFlag);
 
	if (Invert)
		InvertObject (Map, thisObjectHdl);

	return;
}
/**************************************************************************************************/
void GetObjectScrRect (CMap *Map, ObjectRecHdl theObjectHdl, Rect *theScrRect)
{
	GrafPtr		SavePort;
	LongRect	ObjectLRect;
	OSType		thisObjectType;
	
	GetPort (&SavePort);
	
	GetObjectType (theObjectHdl, &thisObjectType);
	if (thisObjectType == kSymbolType)
	{
		LongPoint	SymbolLPoint;
		Point		SymbolScrPoint;
		Rect		LabelScrRect;

		GetObjectLPoint (theObjectHdl, &SymbolLPoint);
		Map -> GetScrPoint (&SymbolLPoint, &SymbolScrPoint);
		theScrRect -> top    = SymbolScrPoint.v - 8;
		theScrRect -> bottom = SymbolScrPoint.v + 8;
		theScrRect -> left   = SymbolScrPoint.h - 8;
		theScrRect -> right  = SymbolScrPoint.h + 8;
		
		// now add object's label rect if label is visible
		if (IsOLabelVisible (theObjectHdl))
		{
			GetObjectLabelRect (theObjectHdl, &LabelScrRect);
			if (!EmptyRect (&LabelScrRect))
				MyUnionRect (theScrRect, &LabelScrRect, theScrRect);
		}
	}
	else
	{
		GetObjectLRect (theObjectHdl, &ObjectLRect);
		Map -> GetScrRect (&ObjectLRect, theScrRect);
	}
	
	SetPort (SavePort);
	
	return;
}
/**************************************************************************************************/
void InvertObject (CMap *Map, ObjectRecHdl theObjectHdl)
{
	GrafPtr		SavePort;
	PenState	penStatus;
	Rect		thisObjectRect;
	OSType		thisObjectType;

	GetPort (&SavePort);
	
	// now draw the selection points on the grouped / single region
	GetPenState (&penStatus);	// save original pen state
	PenSize (4, 4);
	PenMode (srcXor);

	GetObjectType (theObjectHdl, &thisObjectType);
	if (thisObjectType == kLineType)
	{
		LineObjectRecHdl	lineObjectHdl;
		LongPoint			LineStartLPoint, LineEndLPoint;
		Point				LineStartPoint, LineEndPoint;
		
		lineObjectHdl = (LineObjectRecHdl) theObjectHdl;

		LineStartLPoint = (**lineObjectHdl).lineStartLPoint;
		LineEndLPoint   = (**lineObjectHdl).lineEndLPoint;
		Map -> GetScrPoint (&LineStartLPoint, &LineStartPoint);
		Map -> GetScrPoint (&LineEndLPoint,   &LineEndPoint);
		
		MyMoveTo (LineStartPoint.h, LineStartPoint.v);
		MyLineTo (LineStartPoint.h, LineStartPoint.v);
		
		MyMoveTo (LineEndPoint.h, LineEndPoint.v);
		MyLineTo (LineEndPoint.h, LineEndPoint.v);
	}
	else
	{
		GetObjectScrRect (Map, theObjectHdl, &thisObjectRect);
	
		thisObjectRect.top -= 1;
		thisObjectRect.left -= 1;
		thisObjectRect.right -= 3;
		thisObjectRect.bottom -= 3;
	
		MyMoveTo (thisObjectRect.left, thisObjectRect.top);
		MyLineTo (thisObjectRect.left, thisObjectRect.top);
	
		MyMoveTo (thisObjectRect.right, thisObjectRect.top);
		MyLineTo (thisObjectRect.right, thisObjectRect.top);
	
		MyMoveTo (thisObjectRect.right, thisObjectRect.bottom);
		MyLineTo (thisObjectRect.right, thisObjectRect.bottom);
	
		MyMoveTo (thisObjectRect.left, thisObjectRect.bottom);
		MyLineTo (thisObjectRect.left, thisObjectRect.bottom);
	}

	SetPenState (&penStatus);

	SetPort (SavePort);

	return;
}
/**************************************************************************************************/
void UnmarkObjects (CMyList *ObjectList)
{
	long			ObjectCount, ObjectIndex;
	ObjectRecHdl	thisObjectHdl;
	
	ObjectCount = ObjectList -> GetItemCount ();
	for (ObjectIndex = 1; ObjectIndex <= ObjectCount; ++ObjectCount)
	{
		ObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		SetObjectMarked (thisObjectHdl, false);
	}

	return;
}
/**************************************************************************************************/
OSErr WriteObjectInfo (short FRefNum, ObjectRecHdl thisObjectHdl)
{
	OSErr	ErrCode = 0;
	long	byteCount, structSize;

	// write the size of this object
	structSize = _GetHandleSize ((Handle) thisObjectHdl);
	byteCount = sizeof (long);
	ErrCode = FSWrite (FRefNum, &byteCount, (Ptr) &structSize);

	if (!ErrCode)
	{	
		_HLock ((Handle) thisObjectHdl);
		
		ErrCode = FSWrite (FRefNum, &structSize, (Ptr) *thisObjectHdl);
		
		_HUnlock ((Handle) thisObjectHdl);

		if (!ErrCode)	// write the contents of the data handle if necessary
		{
			Handle	thisObjectDataHdl;
			
			thisObjectDataHdl = GetObjectDataHdl (thisObjectHdl);
			if (thisObjectDataHdl != nil)
			{
				// write the size of the data handle first
				structSize = _GetHandleSize (thisObjectDataHdl);
				byteCount = sizeof (long);
				ErrCode = FSWrite (FRefNum, &byteCount, (Ptr) &structSize);
			
				if (!ErrCode)
				{
					_HLock ((Handle) thisObjectDataHdl);
					
					ErrCode = FSWrite (FRefNum, &structSize, (Ptr) *thisObjectDataHdl);
					
					_HUnlock ((Handle) thisObjectDataHdl);
				}
			}
		}
	}
	
	return (ErrCode);
}
/**************************************************************************************************/
OSErr ReadObjectInfo (short FRefNum, ObjectRecHdl *thisObjectHdlPtr)
{
	OSErr	ErrCode = 0;
	long	byteCount, structSize;
	Handle	thisObjectHdl = nil, thisObjectDataHdl = nil;

	// read the size of this object's info handle
	byteCount = sizeof (long);
	ErrCode = FSRead (FRefNum, &byteCount, (Ptr) &structSize);
	if (!ErrCode)
	{
		thisObjectHdl = _NewHandleClear (structSize);
		if (thisObjectHdl != nil)
		{
			_HLock (thisObjectHdl);
		
			ErrCode = FSRead (FRefNum, &structSize, (Ptr) *thisObjectHdl);
		
			_HUnlock (thisObjectHdl);
		}
		else
			ErrCode = memFullErr;
	}
	
	if (!ErrCode)	// read the object's data handle if any
	{
		if (GetObjectDataHdl ((ObjectRecHdl) thisObjectHdl) != nil)
		{
			// read the size of this object's data handle
			byteCount = sizeof (long);
			ErrCode = FSRead (FRefNum, &byteCount, (Ptr) &structSize);
		
			if (!ErrCode)
			{
				thisObjectDataHdl = _NewHandleClear (structSize);
				if (thisObjectDataHdl != nil)
				{
					_HLock (thisObjectDataHdl);
				
					ErrCode = FSRead (FRefNum, &structSize, (Ptr) *thisObjectDataHdl);

					_HUnlock (thisObjectDataHdl);

					if (!ErrCode)
						SetObjectDataHdl ((ObjectRecHdl) thisObjectHdl, thisObjectDataHdl);
				}
				else
					ErrCode = memFullErr;
			}
		}
	}
	
	if (ErrCode)
	{
		if (thisObjectHdl != nil)
			DisposeHandle (thisObjectDataHdl);	

		if (thisObjectHdl != nil)
			DisposeHandle (thisObjectDataHdl);	
	}
	
	if (!ErrCode)
		*thisObjectHdlPtr = (ObjectRecHdl) thisObjectHdl;	// send this handle back
	
	return (ErrCode);
}
/**************************************************************************************************/
long GetObjectIndex (ObjectRecHdl theObjectHandle, CMyList *LayerObjectList)
{
	long			theObjectIndex = 0, objectIndex, objectCount;
	ObjectRecHdl	thisObjectHdl;
	
	objectCount = LayerObjectList -> GetItemCount ();
	for (objectIndex = 1; objectIndex <= objectCount; ++objectIndex)
	{
		LayerObjectList -> GetListItem ((Ptr) &thisObjectHdl, objectIndex);
		if (thisObjectHdl == theObjectHandle)
		{
			theObjectIndex = objectIndex;
			break;
		}
	}

	return (theObjectIndex);
}
/**************************************************************************************************/
ObjectRecHdl GetObjectHandle (CMyList *theObjectList, long ObjectIndex)
{
	ObjectRecHdl	theObjectHdl = nil;

	theObjectList -> GetListItem ((Ptr) &theObjectHdl, ObjectIndex);

	return (theObjectHdl);
}
/**************************************************************************************************/

void DrawObject (ObjectRecHdl theObjectHdl, CMap *Map, LongRect *UpdateLRectPtr,
				 DrawSpecRecPtr drawSettings)
{
	OSType		thisObjectType;
	RGBColor	SaveColor;

	GetForeColor (&SaveColor);		// save original forecolor

	GetObjectType (theObjectHdl, &thisObjectType);
	switch (thisObjectType)
	{
		case kPolyType:
			DrawMapPoly (Map, (PolyObjectHdl) theObjectHdl, drawSettings);
		break;

		case kBeachedLEType:
			DrawBeachLEs (Map, (PolyObjectHdl) theObjectHdl, drawSettings);
		break;

		case kRectType:
		{
			Rect	ObjectRect;

			GetObjectScrRect (Map, theObjectHdl, &ObjectRect);
			MyOffsetRect (&ObjectRect, drawSettings -> offsetDx, drawSettings -> offsetDy);

			if (drawSettings -> fillCode == kPaintFillCode)
			{
				if (drawSettings -> backColorInd > 0)
				{
					Our_PmForeColor (drawSettings -> backColorInd);
					PaintRect (&ObjectRect);
				}
			}
			else if (drawSettings -> fillCode == kPatFillCode)
			{
#ifdef MAC
				PenPat (&drawSettings -> backPattern);
#else
				SetPenPat (drawSettings -> backPattern);
#endif
				PaintRect (&ObjectRect);
			}

			if (drawSettings -> frameCode == kPatFrameCode)
#ifdef MAC
				PenPat (&drawSettings -> forePattern);
#else
				SetPenPat (drawSettings -> forePattern);
#endif

			Our_PmForeColor (drawSettings -> foreColorInd);
			MyFrameRect (&ObjectRect);
		}
		break;

		case kTextType:
		{
			Rect		ObjectRect;

			GetObjectScrRect (Map, theObjectHdl, &ObjectRect);
			MyOffsetRect (&ObjectRect, drawSettings -> offsetDx, drawSettings -> offsetDy);

			if (drawSettings -> fillCode == kPaintFillCode)
			{
				if (drawSettings -> backColorInd > 0)
				{
					Our_PmForeColor (drawSettings -> backColorInd);
					PaintRect (&ObjectRect);
				}
			}
			else if (drawSettings -> fillCode == kPatFillCode)
			{
#ifdef MAC
				PenPat (&drawSettings -> backPattern);
#else
				SetPenPat (drawSettings -> backPattern);
#endif
				PaintRect (&ObjectRect);
			}

			if (drawSettings -> frameCode == kPatFrameCode)
#ifdef MAC
				PenPat (&drawSettings -> forePattern);
#else
				SetPenPat (drawSettings -> forePattern);
#endif

			Our_PmForeColor (drawSettings -> foreColorInd);
//			DrawTextObject ((TextObjectRecHdl) theObjectHdl, Map, UpdateLRectPtr, drawSettings);
		}
		break;

		case kLineType:
		{
			LongRect			ObjectLRect;
			Rect				ObjectRect;
			LongPoint			LineStartLPoint, LineEndLPoint;
			Point				LineStartPoint, LineEndPoint;
			LineObjectRecHdl	lineObjectHdl;

			lineObjectHdl = (LineObjectRecHdl) theObjectHdl;

			LineStartLPoint = (**lineObjectHdl).lineStartLPoint;
			LineEndLPoint   = (**lineObjectHdl).lineEndLPoint;
			Map -> GetScrPoint (&LineStartLPoint, &LineStartPoint);
			Map -> GetScrPoint (&LineEndLPoint,   &LineEndPoint);
			LineStartPoint.h += drawSettings -> offsetDx;
			LineStartPoint.v += drawSettings -> offsetDy;
			LineEndPoint.h   += drawSettings -> offsetDx;
			LineEndPoint.v   += drawSettings -> offsetDy;

			if (drawSettings -> fillCode == kPatFillCode)
#ifdef MAC
				PenPat (&drawSettings -> backPattern);
#else
				SetPenPat (drawSettings -> backPattern);
#endif
			else if (drawSettings -> frameCode == kPatFrameCode)
#ifdef MAC
				PenPat (&drawSettings -> forePattern);
#else
				SetPenPat (drawSettings -> forePattern);
#endif

			Our_PmForeColor (drawSettings -> foreColorInd);
			MyMoveTo (LineStartPoint.h,  LineStartPoint.v);
			MyLineTo (LineEndPoint.h, LineEndPoint.v);
		}
		break;
	}

	// draw the object label if requested by local and universal flags
	if (drawSettings -> bDrawOLabels && IsOLabelVisible (theObjectHdl))
	{
		char		objectLabel [kObjectNameLen];
		Rect		labelRect;
		long		labelJustCode;	

		GetObjectLabel (theObjectHdl, objectLabel);
		if (strlen (objectLabel) > 0)
		{
			CalcSetOLabelRect (Map, theObjectHdl, &labelRect);
			labelJustCode = GetOLabelJust (theObjectHdl);

			TextFontSizeFace (kFontIDHelvetica,9,normal);

			labelRect.bottom -= 4;		// aesthetic tweek for symetry
//			MyTextBox (objectLabel, strlen (objectLabel), &labelRect, labelJustCode);

//			labelRect.bottom -= 4;		// aesthetic tweek for symetry
//			TextBox (objectLabel, strlen (objectLabel), &labelRect, labelJustCode);
		}
	}

	RGBForeColor (&SaveColor);

	return;
}
/**************************************************************************************************/
long GetObjectPart (CMap *Map, ObjectRecHdl theObjectHdl, LongPoint *MatrixLPtPtr)
{
	long		clickPart = kNoPart;
	LongRect	thisObjectLRect;
	
	GetObjectLRect (theObjectHdl, &thisObjectLRect);
	if (PtInLRect (MatrixLPtPtr, &thisObjectLRect))
		clickPart = kInObjectFrame;

	return (clickPart);
}
/**************************************************************************************************/
Boolean PtInObjectLRect (ObjectRecHdl theObjectHdl, LongPoint *MatrixLPtPtr)
{
	Boolean		bPtInObject = false;
	LongRect	thisObjectLRect;

	GetObjectLRect (theObjectHdl, &thisObjectLRect);
	if (PtInLRect (MatrixLPtPtr, &thisObjectLRect))
	{
		bPtInObject = true;
	}

	return (bPtInObject);
}
/**************************************************************************************************/
Boolean PointOnObject (CMap *Map, ObjectRecHdl theObjectHdl, Point LocalPt, long *clickedPart)
{
	Boolean		bPtOnObject = false;
	OSType		thisObjectType;
	LongPoint	MatrixPt;
	LongRect	thisObjectLRect;

	// convert screen click point to matrix point
	Map -> GetMatrixPoint (&LocalPt, &MatrixPt);

	GetObjectType (theObjectHdl, &thisObjectType);
	switch (thisObjectType)
	{
		case kSymbolType:
			Rect	SymbolScrRect;

			GetObjectScrRect (Map, theObjectHdl, &SymbolScrRect);
			if (MyPtInRect (LocalPt, &SymbolScrRect))
			{
				bPtOnObject = true;
//				Debug ("PtOnObject = true\n");
			}
		break;
		
		case kPolyType:
			GetObjectLRect (theObjectHdl, &thisObjectLRect);
			if (!PtInLRect (&MatrixPt, &thisObjectLRect))
				bPtOnObject = false;
			else
				bPtOnObject = PointInPoly (&MatrixPt, (PolyObjectHdl) theObjectHdl);
		break;

		case kLineType:
		{
			double				ptToSegDist;
			LongPoint			LineStartLPoint, LineEndLPoint;
			Point				LineStartPoint, LineEndPoint;
			LineObjectRecHdl	lineObjectHdl;
			long				pixDist = 0;
		
			lineObjectHdl = (LineObjectRecHdl) theObjectHdl;

			LineStartLPoint = (**lineObjectHdl).lineStartLPoint;
			LineEndLPoint   = (**lineObjectHdl).lineEndLPoint;
			
			Map -> GetScrPoint (&LineStartLPoint, &LineStartPoint);
			Map -> GetScrPoint (&LineEndLPoint,   &LineEndPoint);

//			pixDist = LineToPointDist (LocalPt, LineStartPoint, LineEndPoint);
			if (pixDist <= 2)
				bPtOnObject = true;
		}
		break;

		default:
			GetObjectLRect (theObjectHdl, &thisObjectLRect);
			if (PtInLRect (&MatrixPt, &thisObjectLRect))
				bPtOnObject = true;
		break;
	}

	return (bPtOnObject);
}
/**************************************************************************************************/
Boolean ObjectSectLRect (ObjectRecHdl theObjectHdl, LongRect *theLRectPtr)
{
	Boolean		bObjectIntersects = false;
	OSType		thisObjectType;

	GetObjectType (theObjectHdl, &thisObjectType);
	switch (thisObjectType)
	{
		case kSymbolType:
			Rect		SymbolScrRect;
			LongPoint	thisObjectLPoint;
			
			GetObjectLPoint (theObjectHdl, &thisObjectLPoint);
			if (PtInLRect (&thisObjectLPoint, theLRectPtr))
				bObjectIntersects = true;
		break;

		default:
			LongRect	thisObjectLRect;
			
			GetObjectLRect (theObjectHdl, &thisObjectLRect);
			if (IntersectLRect (theLRectPtr, &thisObjectLRect, nil))
				bObjectIntersects = true;
		break;
	}

	return (bObjectIntersects);
}
/**************************************************************************************************/
void InvalObjectRect (CMap *Map, ObjectRecHdl theObjectHdl)
{
	GrafPtr		SavePort;
	Rect		thisObjectRect;

	GetPort (&SavePort);

	GetObjectScrRect (Map, theObjectHdl, &thisObjectRect);
	MyInsetRect (&thisObjectRect, -1, -1);
	Map ->InvalRect (&thisObjectRect);
//	MyFrameRect (&thisObjectRect);

	SetPort (SavePort);

	return;
}
/**************************************************************************************************/
void OffsetObject (ObjectRecHdl ObjectHdl, CMap *Map, long LDx, long LDy, Boolean bInvalidate)
// this subroutine moves the given object by Dx and Dy values.  The values are 'added' to the
//	given object's coordinates.
{
	OSType			thisObjectType;
	PolyObjectHdl	thisPolyHdl;
	LongPoint		**thisPointsHdl, MatrixPt;
	LongRect		thisObjectLRect;
	long			PointIndex, PointCount;
	
	if (ObjectHdl != nil)
	{
		if (bInvalidate)
			InvalObjectRect (Map, ObjectHdl);
		
		// first offset the object's rectangle
		GetObjectLRect (ObjectHdl, &thisObjectLRect);
		OffsetLRect (&thisObjectLRect, LDx, LDy);
		SetObjectLRect (ObjectHdl, &thisObjectLRect);
		
		// then offset the object's point
		GetObjectLPoint (ObjectHdl, &MatrixPt);
		MatrixPt.h += LDx;
		MatrixPt.v += LDy;
		SetObjectLPoint (ObjectHdl, &MatrixPt);
		
		// offset object's label L-point
		GetOLabelLPoint (ObjectHdl, &MatrixPt);
		MatrixPt.h += LDx;
		MatrixPt.v += LDy;
		SetOLabelLPoint (ObjectHdl, &MatrixPt);

		// do any additional shifting of data base on object type
		GetObjectType (ObjectHdl, &thisObjectType);
		switch (thisObjectType)
		{
			case kPolyType:
				thisPolyHdl = (PolyObjectHdl) ObjectHdl;
				thisPointsHdl = (LongPoint**) (**thisPolyHdl).objectDataHdl;
				PointCount = (**thisPolyHdl).pointCount;

				if (thisPointsHdl != nil)
				{
					for (PointIndex = 0; PointIndex < PointCount; ++PointIndex)
					{
						MatrixPt = (*thisPointsHdl) [PointIndex];
						MatrixPt.h += LDx;
						MatrixPt.v += LDy;
						(*thisPointsHdl) [PointIndex] = MatrixPt;
					}
				}
			break;
			
			case kLineType:
			{
				LongPoint			LineStartLPoint, LineEndLPoint;
				LineObjectRecHdl	lineObjectHdl;
				
				lineObjectHdl = (LineObjectRecHdl) ObjectHdl;
				
				LineStartLPoint = (**lineObjectHdl).lineStartLPoint;
				LineEndLPoint   = (**lineObjectHdl).lineEndLPoint;
				LineStartLPoint.h += LDx;
				LineStartLPoint.v += LDy;
				LineEndLPoint.h   += LDx;
				LineEndLPoint.v   += LDy;
				(**lineObjectHdl).lineStartLPoint = LineStartLPoint;
				(**lineObjectHdl).lineEndLPoint   = LineEndLPoint;
			}
			break;
		}

		// calculate the object's new screen-label-rect
		CalcSetOLabelRect (Map, ObjectHdl, (Rect*) nil);

		if (bInvalidate)
			InvalObjectRect (Map, ObjectHdl);
	}
	
	return;
}
/**************************************************************************************************/
void ScaleObject (ObjectRecHdl ObjectHdl, CMap *Map, ScaleRecPtr ScaleInfoPtr, Boolean bInvalidate)
// this subroutine moves the given object by Dx and Dy values.  The values are 'added' to the
//	given object's coordinates.
{
	OSType			thisObjectType;
	PolyObjectHdl	thisPolyHdl;
	LongPoint		**thisPointsHdl, MatrixPt;
	LongRect		thisObjectLRect;
	long			PointIndex, PointCount;
	
	if (ObjectHdl != nil)
	{
		if (bInvalidate)
			InvalObjectRect (Map, ObjectHdl);

		// first scale the object's rectangle
		GetObjectLRect (ObjectHdl, &thisObjectLRect);
		thisObjectLRect.left   = thisObjectLRect.left   * ScaleInfoPtr -> XScale + ScaleInfoPtr -> XOffset;
		thisObjectLRect.right  = thisObjectLRect.right  * ScaleInfoPtr -> XScale + ScaleInfoPtr -> XOffset;
		thisObjectLRect.top    = thisObjectLRect.top    * ScaleInfoPtr -> YScale + ScaleInfoPtr -> YOffset;
		thisObjectLRect.bottom = thisObjectLRect.bottom * ScaleInfoPtr -> YScale + ScaleInfoPtr -> YOffset;
		SetObjectLRect (ObjectHdl, &thisObjectLRect);

		// then scale the object's point
		GetObjectLPoint (ObjectHdl, &MatrixPt);
		MatrixPt.h = MatrixPt.h * ScaleInfoPtr -> XScale + ScaleInfoPtr -> XOffset;
		MatrixPt.v = MatrixPt.v * ScaleInfoPtr -> YScale + ScaleInfoPtr -> YOffset;
		SetObjectLPoint (ObjectHdl, &MatrixPt);

		// do any additional shifting of data base on object type
		GetObjectType (ObjectHdl, &thisObjectType);
		switch (thisObjectType)
		{
			case kPolyType:
				thisPolyHdl = (PolyObjectHdl) ObjectHdl;
				thisPointsHdl = (LongPoint**) (**thisPolyHdl).objectDataHdl;
				PointCount = (**thisPolyHdl).pointCount;

				if (thisPointsHdl != nil)
				{
					for (PointIndex = 0; PointIndex < PointCount; ++PointIndex)
					{
						MatrixPt = (*thisPointsHdl) [PointIndex];
						MatrixPt.h = MatrixPt.h * ScaleInfoPtr -> XScale + ScaleInfoPtr -> XOffset;
						MatrixPt.v = MatrixPt.v * ScaleInfoPtr -> YScale + ScaleInfoPtr -> YOffset;
						(*thisPointsHdl) [PointIndex] = MatrixPt;
					}
				}
			break;

			case kLineType:
			{
				LongPoint			LineStartLPoint, LineEndLPoint;
				LineObjectRecHdl	lineObjectHdl;
				
				lineObjectHdl = (LineObjectRecHdl) ObjectHdl;
				
				LineStartLPoint = (**lineObjectHdl).lineStartLPoint;
				LineEndLPoint   = (**lineObjectHdl).lineEndLPoint;
				LineStartLPoint.h = LineStartLPoint.h * ScaleInfoPtr -> XScale + ScaleInfoPtr -> XOffset;
				LineStartLPoint.v = LineStartLPoint.v * ScaleInfoPtr -> YScale + ScaleInfoPtr -> YOffset;
				LineEndLPoint.h   = LineEndLPoint.h   * ScaleInfoPtr -> XScale + ScaleInfoPtr -> XOffset;
				LineEndLPoint.v   = LineEndLPoint.v   * ScaleInfoPtr -> YScale + ScaleInfoPtr -> YOffset;
				(**lineObjectHdl).lineStartLPoint = LineStartLPoint;
				(**lineObjectHdl).lineEndLPoint   = LineEndLPoint;
			}
			break;
		}

		if (bInvalidate)
			InvalObjectRect (Map, ObjectHdl);
	}
	
	return;
}
/**************************************************************************************************/
void InvalBelowObject (CMap *Map, CMyList *theObjectList, ObjectRecHdl theObjectHdl)
{
	long			ObjectCount, ObjectIndex, ObjectNum;
	ObjectRecHdl	thisObjectHdl;
	GrafPtr			SavePort;
	LongRect		ObjectLRect, thisObjectLRect;
	Rect			CommonRect, ObjectScrRect, thisObjectRect;
	
	if (theObjectList -> IsItemInList ((Ptr) &theObjectHdl, &ObjectNum))
	{
		GetPort (&SavePort);

		ObjectCount = theObjectList -> GetItemCount ();
		if (ObjectNum <= ObjectCount)
		{
			GetObjectScrRect (Map, theObjectHdl, &ObjectScrRect);
			MyInsetRect (&ObjectScrRect, -1, -1);					// extra margin

			// now inval any views above if they are opaque and there is an intersection
			if (ObjectNum < ObjectCount)
			{
				for (ObjectIndex = ObjectNum + 1; ObjectIndex <= ObjectCount; ++ObjectIndex)
				{
					theObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
					GetObjectScrRect (Map, thisObjectHdl, &thisObjectRect);
					MyInsetRect (&thisObjectRect, -1, -1);				// extra margin
					if (SectRect (&ObjectScrRect, &thisObjectRect, &CommonRect))
						Map ->InvalRect (&CommonRect);
				}
			}
		}
	}
	
	return;
}
/**************************************************************************************************/
void InvalAboveObject (CMap *Map, CMyList *theObjectList, ObjectRecHdl theObjectHdl)
{
	long			ObjectCount, ObjectIndex, ObjectNum;
	ObjectRecHdl	thisObjectHdl;
	GrafPtr			SavePort;
	LongRect		ObjectLRect, thisObjectLRect;
	Rect			CommonRect, ObjectScrRect, thisObjectRect;
	
	if (theObjectList -> IsItemInList ((Ptr) &theObjectHdl, &ObjectNum))
	{
		GetPort (&SavePort);

		ObjectCount = theObjectList -> GetItemCount ();
		if (ObjectNum <= ObjectCount)
		{
			GetObjectScrRect (Map, theObjectHdl, &ObjectScrRect);
			MyInsetRect (&ObjectScrRect, -1, -1);					// extra margin

			// now inval any views above if they are opaque and there is an intersection
			if (ObjectNum > 1)
			{
				for (ObjectIndex = ObjectNum - 1; ObjectIndex >= 1; --ObjectIndex)
				{
					theObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
					GetObjectScrRect (Map, thisObjectHdl, &thisObjectRect);
					MyInsetRect (&thisObjectRect, -1, -1);				// extra margin
					if (SectRect (&ObjectScrRect, &thisObjectRect, &CommonRect))
						Map ->InvalRect (&CommonRect);
				}
			}
		}
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
void GetObjectFillPat (ObjectRecHdl theObjectHdl, Pattern *theFillPat)
{
	*theFillPat = (**(theObjectHdl)).objectFillPat;
	
	return;
}
/**************************************************************************************************/
void SetObjectFillPat (ObjectRecHdl theObjectHdl, Pattern *theFillPat)
{
	(**(theObjectHdl)).objectFillPat = *theFillPat;
	
	return;
}

