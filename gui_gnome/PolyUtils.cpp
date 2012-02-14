
#include "Cross.h"
#include	"ObjectUtils.h"
#include	"Geometry.h"
#include	"GenDefs.h"

void DrawMapPoly (CMap* theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings)
{
	RGBColor	saveColor;
	drawSettings -> bClosed = IsPolyClosed (MapPolyHdl);
#ifdef MAC
	long		PointCount;

	//drawSettings -> bClosed = IsPolyClosed (MapPolyHdl);

	GetForeColor (&saveColor);		/* save original forecolor */

	if (drawSettings -> bClosed)
	{
		if (drawSettings -> mode == kPictMode)
		{
			DrawSectPoly (theMap, MapPolyHdl, drawSettings);	/* changed 11/21/95 due to postscript errors */
//			DrawNoSectPoly (theMap, MapPolyHdl, drawSettings);	/* changed back on 3/28/96
		}
		else
		{
			//PointCount = GetPolyPointCount (MapPolyHdl);
			//if (PointCount > 7000)
			//{
			//	/* draw polygon interior without any frame */
			//	drawSettings -> frameCode = kNoFrameCode;
			//	DrawNoSectPoly (theMap, MapPolyHdl, drawSettings);
			//
				/* then draw polygon outline without using poly-routines */
			//	drawSettings -> frameCode = kPaintFrameCode;
			//	drawSettings -> fillCode = kNoFillCode;
			//	DrawNoSectPoly (theMap, MapPolyHdl, drawSettings);
			//}
			//else
				DrawNoSectPoly (theMap, MapPolyHdl, drawSettings);
		}
	}
	else	/* hollow polygon, no fill of any kind */
	{
		DrawNoSectPoly (theMap, MapPolyHdl, drawSettings);
	}

	RGBForeColor (&saveColor);

	return;
#else
	long numPts = (**MapPolyHdl).pointCount;
	POINT **pointsH = (POINT**)_NewHandle(numPts *sizeof(POINT));
	LongPoint** thisPointsHdl=nil;
	Point pt;
	LongPoint wPt;
	long i, esiCode;
	long penWidth = 2, halfPenWidth = 0;
	Boolean bDrawBlackAndWhite = (sharedPrinting && settings.printMode != COLORMODE);
	Boolean offQuickDrawPlane = false, drawingESILayer = false;
	if(!pointsH) {SysBeep(5); return;}
	
	thisPointsHdl = (LongPoint**) (**MapPolyHdl).objectDataHdl;
	GetObjectESICode ((ObjectRecHdl) MapPolyHdl,&esiCode); 
	if (esiCode>0) 	// -500 is the default
	{
		//halfPenWidth = penWidth/2;
		PenStyle(BLACK,penWidth);
		drawingESILayer = true;
	}
	for(i = 0; i< numPts;i++)
	{
		wPt = INDEXH(thisPointsHdl,i);
		//pt.h = SameDifferenceX(wPt.h);
		//pt.v = (gRect.bottom + gRect.top) - SameDifferenceY(wPt.v);
		pt = GetQuickDrawPt(wPt.h,wPt.v,&gRect,&offQuickDrawPlane);
		//pt.h += drawSettings -> offsetDx;
		//pt.v += drawSettings -> offsetDy;
		INDEXH(pointsH,i) = MakePOINT(pt.h-halfPenWidth,pt.v-halfPenWidth);
		// code goes here, make sure this point does not equal previous point JLM
	}
	GetForeColor (&saveColor);		/* save original forecolor */

	//Our_PmForeColor (bDrawBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);//JLM
	// make sure the blackandwhite bitmaps come out right
	Our_PmForeColor (bDrawBlackAndWhite || gDrawBitmapInBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);//JLM
	if (drawSettings -> fillCode == kNoFillCode) 
		Our_PmForeColor (drawSettings -> foreColorInd);
	else
		if(bDrawBlackAndWhite) 
		{
			//SetPenPat(UPSTRIPES);
			// we want solid outline and a patterned inside
			FillPat(UPSTRIPES);
			PenStyle(BLACK,1);
		}
	
	//if(numPts > 2) Polygon(currentHDC,*pointsH,numPts);
	// 6/11/03 PC wasn't recognizing the flag for not filling a land polygon
	if (drawSettings -> bClosed)
	{
		if(numPts > 2) Polygon(currentHDC,*pointsH,numPts);
	}
	else
	{
		//POINT p[2];
		//p[0] = INDEXH(pointsH,numPts-1);
		//p[1] = INDEXH(pointsH,0);
		//RGBForeColor(&colors[BLACK]);
		if(numPts >= 2) 
		{
			Polyline(currentHDC,*pointsH,numPts);
			//if (!drawingESILayer)
				//Polyline(currentHDC,p,2);	// close the polygon
		}
	}

	if(bDrawBlackAndWhite) SetPenPat(BLACK);
	
	RGBForeColor (&saveColor);
	DisposeHandle((Handle)pointsH);

#endif
}

#ifdef IBM
void DrawMapBoundsPoly (CMap* theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings, Boolean erasePolygon)
{
	long numPts = (**MapPolyHdl).pointCount;
	POINT **pointsH = (POINT**)_NewHandle(numPts *sizeof(POINT));
	POINT *pointsPtr = (POINT*)_NewPtr(numPts *sizeof(POINT));
	LongPoint** thisPointsHdl=nil;
	Point pt;
	LongPoint wPt;
	long i;
	Boolean offQuickDrawPlane = false;
	RGBColor saveColor; // JLM ?? wouldn't compile without this
	if(!pointsH || !pointsPtr) {SysBeep(5); return;}
	
	thisPointsHdl = (LongPoint**) (**MapPolyHdl).objectDataHdl;
	for(i = 0; i< numPts;i++)
	{
		wPt = INDEXH(thisPointsHdl,i);
		pt = GetQuickDrawPt(wPt.h,wPt.v,&gRect,&offQuickDrawPlane);
		INDEXH(pointsH,i) = MakePOINT(pt.h,pt.v);
		(pointsPtr)[i] = MakePOINT(pt.h,pt.v);
		// code goes here, make sure this point does not equal previous point JLM
	}
	GetForeColor (&saveColor);		/* save original forecolor */

		if (erasePolygon)
		{
			RgnHandle newClip=0;
			HBRUSH whiteBrush;
			newClip = CreatePolygonRgn((const POINT*)pointsPtr,numPts,ALTERNATE);
			whiteBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
			//err = SelectClipRgn(currentHDC,savedClip);
			FillRgn(currentHDC, newClip, whiteBrush);
			//DeleteObject(newClip);
			DisposeRgn(newClip);
			//SelectClipRgn(currentHDC,0);
		}
		else
		{
			POINT p[2];
			p[0] = INDEXH(pointsH,numPts-1);
			p[1] = INDEXH(pointsH,0);
			RGBForeColor(&colors[BLACK]);
			if(numPts >= 2) 
			{
				Polyline(currentHDC,*pointsH,numPts);
				Polyline(currentHDC,p,2);	// close the polygon
			}
		}

	RGBForeColor (&saveColor);
	DisposeHandle((Handle)pointsH);
	if(pointsPtr) {_DisposePtr((Ptr)pointsPtr); pointsPtr = 0;}
}
#endif

/**************************************************************************************************/
LongPoint**	GetPolyPointsHdl (PolyObjectHdl thePolyHdl)
{
	LongPoint	**thePointsHdl;

	thePointsHdl = (LongPoint **) GetObjectDataHdl ((ObjectRecHdl) thePolyHdl);

	return (thePointsHdl);
}
/**************************************************************************************************/
void SetPolyPointsHdl (PolyObjectHdl thePolyHdl, LongPoint **thePointsHdl)
{
	SetObjectDataHdl ((ObjectRecHdl) thePolyHdl, (Handle) thePointsHdl);

	return;
}
/**************************************************************************************************/
void SetPolyPointCount (PolyObjectHdl thePolyHdl, long pointCount)
{
	if (thePolyHdl != nil)
		(**thePolyHdl).pointCount = pointCount;

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
void CalcSetPolyLRect (PolyObjectHdl thePolyHdl)
{
	LongRect	ObjectLRect;
	long		PointCount, PointIndex;
	LongPoint	MatrixPt;
	LongPoint	**thisPtsHdl = nil;
	
	if (thePolyHdl != nil)
	{
		thisPtsHdl = GetPolyPointsHdl (thePolyHdl);
		PointCount = GetPolyPointCount (thePolyHdl);
		
		/* set region rect to empty to start with */
		SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);
	
		for (PointIndex = 1; PointIndex <= PointCount; ++PointIndex)
		{
			MatrixPt = (*thisPtsHdl) [PointIndex - 1];
			
			if (MatrixPt.h < ObjectLRect.left)   ObjectLRect.left   = MatrixPt.h;
			if (MatrixPt.v > ObjectLRect.top)    ObjectLRect.top    = MatrixPt.v;
			if (MatrixPt.h > ObjectLRect.right)  ObjectLRect.right  = MatrixPt.h;
			if (MatrixPt.v < ObjectLRect.bottom) ObjectLRect.bottom = MatrixPt.v;
		}
	
		SetObjectLRect ((ObjectRecHdl) thePolyHdl, &ObjectLRect);
	}
	
	return;
}
/**************************************************************************************************/
void SetPolyClosed (PolyObjectHdl thePolyHdl, Boolean bClosed)
{
	(**thePolyHdl).bClosedPoly = bClosed;
	if (!bClosed)
		SetObjectBColor ((ObjectRecHdl) thePolyHdl, kNoColorInd);
	
	return;
}
/**************************************************************************************************/
Boolean	IsPolyClosed (PolyObjectHdl thePolyHdl)
{
	Boolean bClosed;

	bClosed = (**thePolyHdl).bClosedPoly;
	
	return (bClosed);
}
/**************************************************************************************************/
void SetPolyWaterFlag (PolyObjectHdl thePolyHdl, Boolean bWaterPoly)
{
	(**thePolyHdl).bWaterPoly = bWaterPoly;
	if (bWaterPoly)
		SetObjectBColor ((ObjectRecHdl) thePolyHdl, kWaterColorInd);

	return;
}
/**************************************************************************************************/
Boolean IsWaterPoly (PolyObjectHdl thePolyHdl)
{
	Boolean	bWaterPoly;
	
	bWaterPoly = (**thePolyHdl).bWaterPoly;

	return (bWaterPoly);
}
/**************************************************************************************************/
/**************************************************************************************************/
SegmentsHdl MakeSectPoly (LongRect *sectLRect, PolyObjectHdl MapPolyHdl)
/* map poly handle must be passed in but segment count is returned */
{
	SegmentsHdl		ScrSegHdl = nil, thisSegHdl = nil, SectSegHdl = nil;
			
	thisSegHdl = RgnToSegment (MapPolyHdl);
	ScrSegHdl = (SegmentsHdl) _NewHandleClear (sizeof (Segment) * 4);	/* will contain the screen rect */
	if (thisSegHdl != nil && ScrSegHdl != nil)
	{
		/* find the intersection between the screen and the polygon */
		(*ScrSegHdl) [0].fromLat  = sectLRect -> top;
		(*ScrSegHdl) [0].fromLong = sectLRect -> left;
		(*ScrSegHdl) [0].toLat    = sectLRect -> top;
		(*ScrSegHdl) [0].toLong   = sectLRect -> right;

		(*ScrSegHdl) [1].fromLat  = sectLRect -> top;
		(*ScrSegHdl) [1].fromLong = sectLRect -> right;
		(*ScrSegHdl) [1].toLat    = sectLRect -> bottom;
		(*ScrSegHdl) [1].toLong   = sectLRect -> right;

		(*ScrSegHdl) [2].fromLat  = sectLRect -> bottom;
		(*ScrSegHdl) [2].fromLong = sectLRect -> right;
		(*ScrSegHdl) [2].toLat    = sectLRect -> bottom;
		(*ScrSegHdl) [2].toLong   = sectLRect -> left;

		(*ScrSegHdl) [3].fromLat  = sectLRect -> bottom;
		(*ScrSegHdl) [3].fromLong = sectLRect -> left;
		(*ScrSegHdl) [3].toLat    = sectLRect -> top;
		(*ScrSegHdl) [3].toLong   = sectLRect -> left;
		
		SectSegHdl = IntersectPolygons (&ScrSegHdl, &thisSegHdl);
	}
	
	if (thisSegHdl != nil)
		DisposeHandle ((Handle) thisSegHdl);
	if (ScrSegHdl != nil)
		DisposeHandle ((Handle) ScrSegHdl);

	return (SectSegHdl);
}
/**************************************************************************************************/
void DrawBeachLEs (CMap *theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings)
{
	long			PointCount, PointIndex;
	LongPoint		MatrixPt;
	Point			LastScrPt, thisScrPt;
	LongPoint		**RgnPtsHdl;

	PointCount = GetPolyPointCount (MapPolyHdl);
	RgnPtsHdl = GetPolyPointsHdl (MapPolyHdl);

	if (RgnPtsHdl != nil)
	{
		for (PointIndex = 0; PointIndex < PointCount; ++PointIndex)
		{
			MatrixPt = (*RgnPtsHdl) [PointIndex];
			theMap -> GetScrPoint (&MatrixPt, &thisScrPt);
//			thisScrPt.h += drawSettings -> offsetDx;
//			thisScrPt.v += drawSettings -> offsetDy;
			
			MyMoveTo (thisScrPt.h - 1, thisScrPt.v - 1);
			MyLineTo (thisScrPt.h + 1, thisScrPt.v + 1);
			MyMoveTo (thisScrPt.h + 1, thisScrPt.v - 1);
			MyLineTo (thisScrPt.h - 1, thisScrPt.v + 1);
		}
	}

	return;
}
/**************************************************************************************************/
SegmentsHdl RgnToSegment (PolyObjectHdl MapPolyHdl)
// this subroutine converts the points in the given points handle into a handle containing from-to
//	type segments and returns it.
{
	long			PointCount, PointIndex, PointIndexPlus;
	LongPoint		MatrixPt;
	LongPoint		**RgnPtsHdl = nil;
	SegmentsHdl		thisSegHdl = nil;

	PointCount = (**MapPolyHdl).pointCount;
	RgnPtsHdl = (LongPoint**) (**MapPolyHdl).objectDataHdl;

	// create a new handle to contain region segments
	thisSegHdl = (SegmentsHdl) _NewHandleClear (sizeof (Segment) * PointCount);
	if (thisSegHdl != nil)
	{
		for (PointIndex = 0; PointIndex < PointCount; ++PointIndex)
		{
			PointIndexPlus = PointIndex + 1;
			if (PointIndexPlus == PointCount)
				PointIndexPlus = 0;

			(*thisSegHdl) [PointIndex].fromLat  = (*RgnPtsHdl) [PointIndex].v;
			(*thisSegHdl) [PointIndex].fromLong = (*RgnPtsHdl) [PointIndex].h;
			(*thisSegHdl) [PointIndex].toLat    = (*RgnPtsHdl) [PointIndexPlus].v;
			(*thisSegHdl) [PointIndex].toLong   = (*RgnPtsHdl) [PointIndexPlus].h;
		}
	}

	return thisSegHdl;
}
/**************************************************************************************************/
OSErr SegmentToRgn (SegmentsHdl theSegment, PolyObjectHdl newPoly)
// newPoly must be a polygon without any points added
{
	Boolean			DrawFromSegFlag;
	long			SegCount, SegIndex, SegIndexPlus;
	LongPoint		MatrixPt;
	PolyObjectHdl	newPolyHdl = nil;
	OSErr			err = noErr;
	LongRect		emptyLRect;

	// set region rect to empty to start with
	SetLRect (&emptyLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);

	SegCount = _GetHandleSize ((Handle) theSegment) / sizeof (Segment);
	if (SegCount > 0)
	{
		DrawFromSegFlag = true;

		for (SegIndex = 0; SegIndex < SegCount; ++SegIndex)
		{
			if (DrawFromSegFlag)	/* start of new poly */
			{
				/* create a new polygon object */
				CreateNewPtsHdl (newPolyHdl, 0);

				/* mark the newly added polygon */
				SetObjectMarked ((ObjectRecHdl) newPolyHdl, true);

				/* set bounds to empty rect */
				SetObjectLRect ((ObjectRecHdl) newPolyHdl, &emptyLRect);

				if (!err)
				{
					MatrixPt.h = (*theSegment) [SegIndex].fromLong;
					MatrixPt.v = (*theSegment) [SegIndex].fromLat;
					AddPointToPoly (newPolyHdl, &MatrixPt);

					DrawFromSegFlag = false;
				}
			}

			MatrixPt.h = (*theSegment) [SegIndex].toLong;
			MatrixPt.v = (*theSegment) [SegIndex].toLat;
			AddPointToPoly (newPolyHdl, &MatrixPt);

			SegIndexPlus = SegIndex + 1;
			if (SegIndexPlus == SegCount)
				SegIndexPlus = 0;

			/* check for end of current poly */
			if (((*theSegment) [SegIndex].toLat  != (*theSegment) [SegIndexPlus].fromLat ||
				 (*theSegment) [SegIndex].toLong != (*theSegment) [SegIndexPlus].fromLong))
			{
				DrawFromSegFlag = true;
			}
		}
	}

	return err;
}
/**************************************************************************************************/
Handle CreateNewPtsHdl (PolyObjectHdl thePolyHdl, long PointCount)
/* returns the new handle if succesful */
{
	Handle		newPtsHdl = nil;
	long		spaceBytes;
	LongRect	polyLRect;
	
	spaceBytes = sizeof (LongPoint) * PointCount;
	
	newPtsHdl = _NewHandleClear (spaceBytes);
	if (newPtsHdl != nil)
	{
		SetObjectDataHdl ((ObjectRecHdl) thePolyHdl, newPtsHdl);
		SetPolyPointCount (thePolyHdl, PointCount);
	}

	return newPtsHdl;
}
/**************************************************************************************************/
OSErr AddPointToPoly (PolyObjectHdl thePoly, LongPoint *newLPoint)
{
	LongPoint	**thisPointsHdl;
	OSErr		err = noErr;
	long		thisPolyPtCount;
	Size		NewHSize;
	LongRect	polyLRect;
	
	if (thePoly != nil)
	{
		thisPointsHdl = GetPolyPointsHdl (thePoly);
		if (thisPointsHdl != nil)
		{	
			NewHSize = _GetHandleSize ((Handle) thisPointsHdl) + sizeof (LongPoint);
			_SetHandleSize ((Handle) thisPointsHdl, NewHSize);
			if (_MemError ())
				err = memFullErr;
			else
			{
				thisPolyPtCount = GetPolyPointCount (thePoly);
				++thisPolyPtCount;
				
				(*thisPointsHdl) [thisPolyPtCount - 1] = *newLPoint;
				SetPolyPointCount (thePoly, thisPolyPtCount);
				
				GetObjectLRect ((ObjectRecHdl) thePoly, &polyLRect);
				
				if (newLPoint -> h < polyLRect.left)   polyLRect.left   = newLPoint -> h;
				if (newLPoint -> h > polyLRect.right)  polyLRect.right  = newLPoint -> h;
				if (newLPoint -> v > polyLRect.top)    polyLRect.top    = newLPoint -> v;
				if (newLPoint -> v < polyLRect.bottom) polyLRect.bottom = newLPoint -> v;

				SetObjectLRect ((ObjectRecHdl) thePoly, &polyLRect);
			}
		}
		else
			err = memFullErr;
	}
	else
		err = nilHandleErr;

	return	err;
}
/**************************************************************************************************/
OSErr IntersectPoly (PolyObjectHdl MainPolyHdl, LongRect *sectLRect, CMyList *theObjectList)
{
	SegmentsHdl		SectSegHdl = nil;
	CMyList			*sectPolyList = nil;
	PolyObjectHdl	newPolyHdl = nil;
	OSErr			err = noErr;
	
	SectSegHdl = MakeSectPoly (sectLRect, MainPolyHdl);
	if (SectSegHdl != nil)
	{
		Boolean		DrawFromSegFlag;
		long		SegCount, SegIndex, SegIndexPlus;
		LongPoint	MatrixPt;
		LongRect	emptyLRect;

		/* set region rect to empty to start with */
		SetLRect (&emptyLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);

		SegCount = _GetHandleSize ((Handle) SectSegHdl) / sizeof (Segment);
		if (SegCount > 0)
		{
			DrawFromSegFlag = true;

			for (SegIndex = 0; SegIndex < SegCount; ++SegIndex)
			{
				if (DrawFromSegFlag)	/* start of new poly */
				{
					/* create a new polygon object */
					newPolyHdl = MainPolyHdl;
					err = _HandToHand((Handle *) &newPolyHdl);
					if (!err)			/* create new points handle */
					{
						CreateNewPtsHdl (newPolyHdl, 0);

						/* mark the new polygon */
						SetObjectMarked ((ObjectRecHdl) newPolyHdl, true);

						/* set bounds to empty rect */
						SetObjectLRect ((ObjectRecHdl) newPolyHdl, &emptyLRect);

						/* add polygon to objects list */
						theObjectList -> AppendItem ((Ptr) &newPolyHdl);		
					}

					if (!err)
					{
						MatrixPt.h = (*SectSegHdl) [SegIndex].fromLong;
						MatrixPt.v = (*SectSegHdl) [SegIndex].fromLat;
						AddPointToPoly (newPolyHdl, &MatrixPt);
						
						DrawFromSegFlag = false;
					}
				}

				MatrixPt.h = (*SectSegHdl) [SegIndex].toLong;
				MatrixPt.v = (*SectSegHdl) [SegIndex].toLat;
				AddPointToPoly (newPolyHdl, &MatrixPt);

				SegIndexPlus = SegIndex + 1;
				if (SegIndexPlus == SegCount)
					SegIndexPlus = 0;
				
				/* check for end of current poly */
				if (((*SectSegHdl) [SegIndex].toLat  != (*SectSegHdl) [SegIndexPlus].fromLat ||
					 (*SectSegHdl) [SegIndex].toLong != (*SectSegHdl) [SegIndexPlus].fromLong))
				{
					DrawFromSegFlag = true;
				}
			}
		}
	}
	else
		err = memFullErr;
		
	if (err == memFullErr)
		TechError ("IntersectPoly()", "Out of memory while intersecting polgons!", err);
	else if (err)
		TechError ("IntersectPoly()", "Error while intersecting polygons!", err);

	return err;
}
/**************************************************************************************************/
Boolean PointInPoly (LongPoint *theLPoint, PolyObjectHdl mapPolyHdl)
{
	Boolean			bInPoly = false;
	SegmentsHdl		thisSegHdl = nil;
	long			numSegs;
	
	thisSegHdl = RgnToSegment (mapPolyHdl);
	if (thisSegHdl)
	{
		WorldPoint	wp;
		wp.pLat = theLPoint -> v;
		wp.pLong = theLPoint -> h;

		numSegs = _GetHandleSize ((Handle) thisSegHdl) / sizeof (Segment);
		bInPoly = PointInPolygon (wp, thisSegHdl, numSegs, false);
		DisposeHandle ((Handle) thisSegHdl);
	}
	
	return bInPoly;
}

#ifdef MAC

RGBColor esicolors[] = {
	// SEE THE clut 1000 RESOURCE FOR THE ACTUAL COLORS
	//red		green		blue
	{ 0,		0,			0 },		// dummy
	{ 120*120-1, 		1520, 			106*106-1 },		// dark purple
	{ 175*175-1, 	154*154-1, 		192*192-1 },	// light purple
	{ 0, 	152*152-1, 		213*213-1 },	// blue
	{ 147*147-1, 	210*210-1, 		242*242-1 },	// light blue
	{ 153*153-1, 	207*207-1, 		202*202-1 },	// light blue green
	{ 0, 	150*150-1, 		33*33-1 },	// green
	{ 222*222-1,	215*215-1,		0 },	// light green
	{ 215*215-1,	187*187-1,		0 },		// olive
	//{ 57343, 	41929, 		27043 },	// light brown (olive)
	{ 65535,	54288,		0 },	// yellow
	{ 255*255-1, 	190*190-1, 		171*171-1 },	// peach
	{ 248*248-1,	206*206-1,		76*76-1 },		// light orange
	{ 249*249-1,	164*164-1,		0 },	// orange
	{ 215*215-1,	0,		624 },		// red
	{ 246*246-1,	163*163-1,		189*189-1 },	// light magenta
	{ 210*210-1, 	78*78-1, 		81*81-1 },	// dark red
	{ 198*198-1,	115*115-1,		71*71-1 },	// brown
	{ 0,		0,			0 }			// most recent other
};
#else
RGBColor esicolors[] = {
	// SEE THE clut 1000 RESOURCE FOR THE ACTUAL COLORS
	//red		green		blue
	RGB( 0,		0,			0 ),		// dummy
	RGB( 119, 		38, 			105 ),		// dark purple
	RGB( 174, 	153, 		191 ),	// light purple
	RGB( 0, 	151, 		212 ),	// blue
	RGB( 146, 	209, 		241 ),	// light blue
	RGB( 152, 	206, 		201 ),	// light blue green
	RGB( 0, 	149, 		32 ),	// green
	RGB( 221,	214,		0 ),	// light green
	RGB( 214,	186,		0 ),		// olive
	//RGB( 57343, 	41929, 		27043 ),	// light brown (olive)
	RGB( 255,	232,		0 ),	// yellow
	RGB( 254, 	189, 		171 ),	// peach
	RGB( 247,	205,		75 ),		// light orange
	RGB( 248,	163,		0 ),	// orange
	RGB( 214,	0,		24 ),		// red
	RGB( 245,	162,		188 ),	// light magenta
	RGB( 209, 	77, 		80 ),	// dark red
	RGB( 197,	114,		70 ),	// brown
	RGB( 0,		0,			0 )			// most recent other
};
#endif
/**************************************************************************************************/

void Our_PmForeColor(long colorInd)
{
	switch(colorInd)
	{
		case kNoColorInd:
		default: // default needs to be black since the blacck and white bitmap does not set the colors right
			RGBForeColor (&colors[BLACK]);
			break;
		case kLandColorInd:
			RGBForeColor (&colors[LIGHTBROWN]);
			break;
		case kWaterColorInd: // make water white for black and white bitmap, etc, note this is also 
		// note kLgtBlueColorInd == kWaterColorInd
		case kWhiteColorInd:
			RGBForeColor (&colors[WHITE]);
			break;
		case kBlackColorInd:
			RGBForeColor (&colors[BLACK]);
			break;
		case kRedColorInd:
			RGBForeColor (&colors[RED]);
			break;
		case kBlueColorInd:
			//RGBForeColor (&colors[BLUE]);
			RGBForeColor (&colors[WHITE]); // BNA maps should not use blue for water, it messes up the black and white bitmap
			break;
		case kVLgtGrayColorInd:
		case kLgtGrayColorInd:
			RGBForeColor (&colors[LIGHTGRAY]);
			break;
		case kGrayColorInd:
			RGBForeColor (&colors[GRAY]);
			break;
		case kDkGrayColorInd:
			RGBForeColor (&colors[DARKGRAY]);
			break;
		case 100:
			RGBForeColor (&colors[PURPLE]);
			break;
		case 150:
			RGBForeColor (&colors[GREEN]);
			break;
		case 200:
			RGBForeColor (&colors[PINK]);
			break;
		case 225:
			RGBForeColor (&colors[YELLOW]);
			break;
/////////////////////////////////////////////////
		// ESI colors
		case kESIDkPurpleInd:
		case kESILgtPurpleInd:
		case kESIBlueInd:
		case kESILgtBlueInd:
		case kESILgtBlueGreenInd:
		case kESIGreenInd:
		case kESILgtGreenInd:
		case kESIOliveInd:
		case kESIYellowInd:
		case kESIPeachInd:
		case kESILgtOrangeInd:
		case kESIOrangeInd:
		case kESIRedInd:
		case kESILgtMagentaInd:
		case kESIDkRedInd:
		case kESIBrownInd:
			RGBForeColor (&esicolors[colorInd-300]);
			break;
	}
}



short SectionOfPlane(Rect *rect,Point pt)
{
	// sections  of the plane
	//    1 7 4
	//    2 0 5
	//    3 8 6
	
	
	if(pt.h < rect->left)
	{ // sections 1,2,3
		if(pt.v < rect->top) return 1;
		else if(pt.v > rect->bottom) return 3;
		else return 2;
	}
	else if(pt.h > rect->right)
	{ // sections 4,5,6
		if(pt.v < rect->top) return 4;
		else if(pt.v > rect->bottom) return 6;
		else return 5;
	}
	else if(pt.v < rect->top) return 7;
	else if(pt.v > rect->bottom) return 8;
	else return 0;
	
	return 0;
}


#ifdef MAC ///{

	
void DrawNoSectPolyRecursive (CMap *theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings,Rect subRect)
{
	long			PointCount, PlotCount = 0, PointIndex;
	LongPoint		MatrixPt;
	Point			LastScrPt, ThisScrPt, FirstScrPt;
	LongPoint		**RgnPtsHdl;
	PolyHandle		PolyHdl = nil;
	
	////////////
	Boolean  alwaysIn123 = true;
	Boolean  alwaysIn174 = true;
	Boolean  alwaysIn456 = true;
	Boolean  alwaysIn386 = true;
	Boolean canSkipDrawingPolygon = false;
	//    1 7 4
	//    2 0 5
	//    3 8 6
	////////////////

	PointCount = GetPolyPointCount (MapPolyHdl);
	RgnPtsHdl = GetPolyPointsHdl (MapPolyHdl);
	Boolean bDrawBlackAndWhite = (sharedPrinting && settings.printMode != COLORMODE);

	#define MAXNUMSEGMENTS  8000 // JLM, It seems the limit is 32K not 64K at the documentation says
	short thisSectionOfPlane,prevSectionOfPlane;
	Boolean canSkipThisPt,skippedAPt,offQuickDrawPlane=false;
	Point lastSkippedPt;
	Rect  fuzzyRect = subRect;
	long lineWidth = 1, esiCode;
	long outsetPixels = 2*lineWidth+2; // needs to be wider that the line width
	//long penWidth = 3;
	long penWidth = 2;
	long halfPenWidth = 0;
	InsetRect(&fuzzyRect,-outsetPixels,-outsetPixels);

	if (RgnPtsHdl != nil)
	{

		// must clip to this rect in addtion to the original clip
		Rect clippingRect = subRect;
		RgnHandle saveClip = NewRgn(), addition = NewRgn() , newClip = NewRgn();

		GetClip(saveClip);
		GetClip(newClip);
		RectRgn(addition, &clippingRect);
		SectRgn(newClip, addition, newClip);
		SetClip(newClip);
		if(newClip) DisposeRgn(newClip);
		if(addition) DisposeRgn(addition);


		if (drawSettings -> fillCode != kNoFillCode)
			PolyHdl = OpenPoly ();
		else
			Our_PmForeColor (gDrawBitmapInBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);

		GetObjectESICode ((ObjectRecHdl) MapPolyHdl,&esiCode); 
		if (esiCode>0) 	// -500 is the default
		{
			//halfPenWidth = penWidth/2;
#ifdef MAC
			PenSize(penWidth,penWidth);
#else
			PenStyle(BLACK,penWidth);
#endif
		}
		for (PointIndex = 0,skippedAPt = false,prevSectionOfPlane = -1; PointIndex < PointCount; ++PointIndex)
		{
			MatrixPt = (*RgnPtsHdl) [PointIndex];
//			theMap -> GetScrPoint (&MatrixPt, &ThisScrPt);

			//ThisScrPt.h = SameDifferenceX(MatrixPt.h);
			//ThisScrPt.v = (gRect.bottom + gRect.top) - SameDifferenceY(MatrixPt.v);
			ThisScrPt = GetQuickDrawPt(MatrixPt.h, MatrixPt.v, &gRect, &offQuickDrawPlane);

			// code goes here, what to do when point is off quickdraw plane?
			//if (offQuickDrawPlane) break;
			
			ThisScrPt.h += drawSettings -> offsetDx;
			ThisScrPt.v += drawSettings -> offsetDy;
			
			if(PolyHdl)
			{  //// JLM 2/18/99
				// for points outside the drawing area, it is not necessary to move to the correct point,
				// as long as we preserve the winding.  This allows us to ignore many of the points outside 
				// the drawing rectangle gRect
				thisSectionOfPlane = SectionOfPlane(&fuzzyRect,ThisScrPt);
				if( 	thisSectionOfPlane > 0 // outside of the rectangle
						&& thisSectionOfPlane == prevSectionOfPlane // we have not changed sections of the plane
						&& PointIndex < PointCount -1) // not the last point
					canSkipThisPt = true;
				else canSkipThisPt = false;
				prevSectionOfPlane = thisSectionOfPlane;
				if(canSkipThisPt) 
				{
					skippedAPt = true;
					lastSkippedPt =ThisScrPt;
					continue;
				}
				/// JLM 3/6/01
				switch(thisSectionOfPlane) {
					case 1: 															alwaysIn456 = false; alwaysIn386 = false; break;
					case 2: 								alwaysIn174 = false; alwaysIn456 = false; alwaysIn386 = false; break;
					case 3: 								alwaysIn174 = false; alwaysIn456 = false; 							break;
					case 4:	alwaysIn123 = false; 														alwaysIn386 = false; break;
					case 5:	alwaysIn123 = false;	alwaysIn174 = false; 							alwaysIn386 = false; break;
					case 6:	alwaysIn123 = false;	alwaysIn174 = false; 														break;
					case 7:	alwaysIn123 = false;								alwaysIn456 = false; alwaysIn386 = false; break;
					case 8:	alwaysIn123 = false;	alwaysIn174 = false; alwaysIn456 = false; 							break;
					default: alwaysIn123 = false;	alwaysIn174 = false; alwaysIn456 = false; alwaysIn386 = false; break;
				}
				//////
				if(skippedAPt)
				{	// then we have to draw to the previous point 
					// before we draw to the current point 
					PointIndex--; //so we do the previous point below
					ThisScrPt = lastSkippedPt; // restore the coordinates of the previous point
					prevSectionOfPlane = -1; // force the next point to not be skipped
				}
				skippedAPt = false;
				if(PlotCount > MAXNUMSEGMENTS)
				{	// there is a bug on the max when the number of points gets too large
					// try recusion
					ClosePoly();
					KillPoly(PolyHdl);
					SetClip(saveClip);// JLM 8/4/99
					goto recursion;
				}
				//////////////
			}


			if (PointIndex == 0)
			{
				MyMoveTo (ThisScrPt.h-halfPenWidth, ThisScrPt.v-halfPenWidth);
				FirstScrPt = ThisScrPt;
				LastScrPt = ThisScrPt;
				PlotCount = 0;
			}
			else
			{
				if (LastScrPt.h != ThisScrPt.h || LastScrPt.v != ThisScrPt.v)
				{
					MyLineTo (ThisScrPt.h-halfPenWidth, ThisScrPt.v-halfPenWidth);
					LastScrPt = ThisScrPt;
					++PlotCount;
				}
			}
		}

		if (drawSettings -> bClosed)	/* draw a line from last point to first point if requested */
		{
			MyLineTo (FirstScrPt.h-halfPenWidth, FirstScrPt.v-halfPenWidth);
			++PlotCount;
		}

		if (PolyHdl != nil)
		{
			ClosePoly ();
			////////////// JLM 3/6/01
			if(alwaysIn123 || alwaysIn174 || alwaysIn456 || alwaysIn386)
				canSkipDrawingPolygon = true;
			if(canSkipDrawingPolygon) PlotCount = 0; // so that we skip the code below
			////////////
			if (PlotCount > 0)
			{
				if (PlotCount > 2)			/* polygon must contain more than 2 line-to points */
				{
					if (drawSettings -> bErase)
						ErasePoly (PolyHdl);
	
					if (drawSettings -> fillCode == kPaintFillCode)
					{
						// this is the usual drawing code
						Our_PmForeColor (bDrawBlackAndWhite || gDrawBitmapInBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);//JLM
						if(bDrawBlackAndWhite) SetPenPat(UPSTRIPES);
						PaintPoly(PolyHdl);//JLM
						if(bDrawBlackAndWhite) SetPenPat(BLACK);
					}
					else if (drawSettings -> fillCode == kPatFillCode)
						FillPoly (PolyHdl, &(drawSettings -> backPattern));
				}

				if (drawSettings -> frameCode == kPaintFrameCode)
				{
					Our_PmForeColor (bDrawBlackAndWhite || gDrawBitmapInBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);
					FramePoly (PolyHdl);
				}
				else if (drawSettings -> frameCode == kPatFrameCode)
				{
					PenPat (&(drawSettings -> forePattern));
					FramePoly (PolyHdl);
				}
			}
			
			KillPoly (PolyHdl);
		}
		
		SetClip(saveClip);
		if(saveClip) DisposeRgn(saveClip);

	}

#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	return;
	
	////////////////////////////////
recursion:
	////////////////////////////////
	{
		#define MAXRECURSION 20 
		static short sRecursionValue = 0;
	
		if(sRecursionValue >= MAXRECURSION) 
		{
			printError("max recusion exceeded");
		}
		else
		{
			// use recursion
			Rect subBoundingRect;
			long middleH = (subRect.left+subRect.right)/2;
			long middleV = (subRect.top+subRect.bottom)/2;
			long index;
			
			sRecursionValue++;
		
			// divide the points up and Draw again
			for(index = 0; index < 4; index++)
			{
				subBoundingRect = subRect;
				switch(index)
				{
					case 0: 
						subBoundingRect.top  = middleV;
						subBoundingRect.left  = middleH;
						break;
					case 1: 
						subBoundingRect.top  = middleV;
						subBoundingRect.right  = middleH;
						break;
					case 2: 
						subBoundingRect.bottom  = middleV;
						subBoundingRect.left  = middleH;
						break;
					default: 
						subBoundingRect.bottom  = middleV;
						subBoundingRect.right  = middleH;
						break;
				}
				
				// the recursive call
				DrawNoSectPolyRecursive (theMap,MapPolyHdl,drawSettings,subBoundingRect);
				
			}
			// all done
			sRecursionValue--;
			return;
		}
	}
	
}

/////////////////////////////////////////////////

void DrawNoSectPoly (CMap *theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings)
{
 	DrawNoSectPolyRecursive (theMap, MapPolyHdl, drawSettings,gRect);
}

/**************************************************************************************************/

OSErr DrawSectPoly (CMap* theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings)
{
	long			SegCount, SegIndex, SegIndexPlus, PlotCount;
	LongPoint		MatrixPt;
	Point			ThisScrPt;
	OSErr			ErrCode = 0;
	Boolean			DrawFromSegFlag;
	PolyHandle		PolyHdl = nil;
	SegmentsHdl		SectSegHdl = nil;
	LongRect		sectLRect;
	Rect			ScrPolyRect;
	
	theMap -> GetMapDrawRect (&ScrPolyRect);
	InsetRect (&ScrPolyRect, -1, -1);
	theMap -> GetMatrixLRect (&ScrPolyRect, &sectLRect);

	SectSegHdl = MakeSectPoly (&sectLRect, MapPolyHdl);
	if (SectSegHdl != nil)
	{
		SegCount = _GetHandleSize ((Handle) SectSegHdl) / sizeof (Segment);

		PolyHdl = OpenPoly ();
			
		DrawFromSegFlag = true;
		PlotCount = 0;
		Boolean bDrawBlackAndWhite = (sharedPrinting && settings.printMode != COLORMODE);
		
		for (SegIndex = 0; SegIndex < SegCount; ++SegIndex)
		{
			if (DrawFromSegFlag)	/* start of new poly */
			{
				MatrixPt.h = (*SectSegHdl) [SegIndex].fromLong;
				MatrixPt.v = (*SectSegHdl) [SegIndex].fromLat;
				theMap -> GetScrPoint (&MatrixPt, &ThisScrPt);
				MyMoveTo (ThisScrPt.h, ThisScrPt.v);
				DrawFromSegFlag = false;
				PlotCount = 0;
			}
			
			MatrixPt.h = (*SectSegHdl) [SegIndex].toLong;
			MatrixPt.v = (*SectSegHdl) [SegIndex].toLat;
			theMap -> GetScrPoint (&MatrixPt, &ThisScrPt);
			MyLineTo (ThisScrPt.h, ThisScrPt.v);
			++PlotCount;

			SegIndexPlus = SegIndex + 1;
			if (SegIndexPlus == SegCount)
				SegIndexPlus = 0;
			
			/* check for end of current poly */
			if (((*SectSegHdl) [SegIndex].toLat  != (*SectSegHdl) [SegIndexPlus].fromLat ||
				 (*SectSegHdl) [SegIndex].toLong != (*SectSegHdl) [SegIndexPlus].fromLong))
			{
				ClosePoly ();
				
				if (PlotCount > 0)
				{
					if (PlotCount > 2)
					{
						if (drawSettings -> bErase)
							ErasePoly (PolyHdl);
		
						if (drawSettings -> fillCode == kPaintFillCode)
						{
							// this is the usual drawing code
							Our_PmForeColor (bDrawBlackAndWhite || gDrawBitmapInBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);//JLM
							if(bDrawBlackAndWhite) SetPenPat(UPSTRIPES);
							PaintPoly(PolyHdl);//JLM
							if(bDrawBlackAndWhite) SetPenPat(BLACK);
						}
						else if (drawSettings -> fillCode == kPatFillCode)
							FillPoly (PolyHdl, &(drawSettings -> backPattern));
					}
				
					if (drawSettings -> frameCode == kPaintFrameCode)
					{
						Our_PmForeColor (bDrawBlackAndWhite || gDrawBitmapInBlackAndWhite ? kBlackColorInd : drawSettings -> foreColorInd);
						FramePoly (PolyHdl);
					}
					else if (drawSettings -> frameCode == kPatFrameCode)
					{
						PenPat (&(drawSettings -> forePattern));
						FramePoly (PolyHdl);
					}
				}

				KillPoly (PolyHdl);

				/* now open a new poly for the next set of segments */
				PolyHdl = OpenPoly ();
				
				DrawFromSegFlag = true;
			}
		}

		ClosePoly ();

		if (PlotCount > 0)
		{
			if (PlotCount > 2)		/* must have at least 3 segments to fill-poly */
			{
				if (drawSettings -> bErase)
					ErasePoly (PolyHdl);

				if (drawSettings -> fillCode == kPaintFillCode)
				{
					Our_PmForeColor (drawSettings -> backColorInd);
					FillPoly (PolyHdl, &(drawSettings -> backPattern));
				}
				else if (drawSettings -> fillCode == kPatFillCode)
					FillPoly (PolyHdl, &(drawSettings -> backPattern));
			}

			if (drawSettings -> frameCode == kPaintFrameCode)
			{
				Our_PmForeColor (drawSettings -> foreColorInd);
				FramePoly (PolyHdl);
			}
			else if (drawSettings -> frameCode == kPatFrameCode)
			{
				PenPat (&(drawSettings -> forePattern));
				FramePoly (PolyHdl);
			}
		}

		KillPoly (PolyHdl);

		DisposeHandle ((Handle) SectSegHdl);
	}

	return (ErrCode);
}

#endif // end MAC only ///}

