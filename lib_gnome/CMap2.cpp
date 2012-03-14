
#include "Cross.h"
#include	"CMap2.h"
#include	"MapUtils.h"
#include	"Units.h"
/**************************************************************************************************/
OSErr CMap::IMap (WindowPtr ViewWPtr, Boolean LLVisFlag, Boolean LinesInBackFlag,
				  long mapProjCode, Boolean bInvalidate)
{
	OSErr		ErrCode = 0;
	LongRect	ViewBoundsLRect;

	wPtr = ViewWPtr;
	ViewStatusPtr = &MapViewStatus;

	// set lat / long grid defaults
	ViewStatusPtr -> bLatLongVisible = LLVisFlag;
	ViewStatusPtr -> prefLats = 2;
	ViewStatusPtr -> prefLongs = 2;
	ViewStatusPtr -> bLinesInBack = LinesInBackFlag;

	// initialize map to world rect coordinates
	SetLRect (&ViewBoundsLRect, kWorldLeft,  kWorldTop, kWorldRight, kWorldBottom);
	SetMapLBounds (&ViewBoundsLRect);

	ViewStatusPtr -> mapProjCode = mapProjCode;
	SetMapColorFlag (true, false);
	SetMapModified (false);

	ViewAllMap (bInvalidate);

	return (ErrCode);
}
/**************************************************************************************************/
void CMap::ViewAllMap (Boolean bInvalidate)
// subroutine to set scale for zero-magnification map drawing
{
	// set unscaled window-drawing-rectangle to be map window's content rect
	SetWDrawRect ();

	// set current map matrix rect to the bounds of the current map (only)
	ViewStatusPtr -> mapMatrixRect = ViewStatusPtr -> mapBoundsRect;

	// update the MapMatrixRatio and mapViewRatio fields of View-Status-Rect
	CalcMapRatios ();

	// now assign a scaled version of window's drawing rect to the mapWindowRect field
	CalcMapWindowRect ();

	SetMapScales ();					// set scales to reflect new window size

	DrawMapInLRect (&ViewStatusPtr -> mapMatrixRect, bInvalidate);
	SetZoomedOut (true);

//	if (bInvalidate)
//		InvalInfoBar (wPtr, false);		// invalidate info bar to update map scale

	return;
}
/**************************************************************************************************/
void CMap::SetWDrawRect ()
{
	Rect	WindowDrawRect;
	
	WindowDrawRect = GetWindowPortRect(wPtr);
	WindowDrawRect.right -= 15;
	WindowDrawRect.bottom -= 15;

	ViewStatusPtr -> mapWDrawRect = WindowDrawRect;
	
	return;
}
/**************************************************************************************************/
void CMap::CalcMapRatios ()
/* this subroutine calculates both the MapMatrixRatio and mapViewRatio fields in ViewStatusRec.
   Before calling, make sure the Map-Matrix-Rect field has been initialized. */
{
	double		exLat1, exLong1, exLat2, exLong2, exMapWidth, exMapHeight;
	LongRect	BoundsLRect, WorldRect;
	LongPoint	MatrixPt1, MatrixPt2;

	if (ViewStatusPtr -> mapProjCode == kLatLongProjCode)
	{
		BoundsLRect = ViewStatusPtr -> mapMatrixRect;
		
		// first calculate the XToY ratio of the map matrix bounds rectangle (MapMatrixRatio field)
		ViewStatusPtr -> mapMatrixRatio = (double) (labs (BoundsLRect.right  - BoundsLRect.left)) /
										  (double) (labs (BoundsLRect.bottom - BoundsLRect.top));
		
		// now convert coordinates to radians
		exLat1  = ((double) BoundsLRect.bottom / 1000000) * PI / 180.0;
		exLat2  = ((double) BoundsLRect.top    / 1000000) * PI / 180.0;
		exLong1 = ((double) BoundsLRect.left   / 1000000) * PI / 180.0;
		exLong2 = ((double) BoundsLRect.right  / 1000000) * PI / 180.0;
				
		// now calculate the height and width of the current map using a distance formula
		exMapHeight = exLat2 - exLat1;
		exMapWidth  = sqrt (cos ((exLat2 + exLat1) / 2.0) * (exLong2 - exLong1) * (exLong2 - exLong1));
	
		// set the map-view-ratio field accordingly
		ViewStatusPtr -> mapViewRatio = exMapWidth / exMapHeight;
	}
	
	return;
}
/**************************************************************************************************/
void CMap::CalcMapWindowRect ()
/* this subroutine calculates the mapWindowRect field of ViewStatusRec.  This is done by scaling
	the current screen drawing rectangle (WindowDrawRect) using the map view ratio (mapViewRatio) */
{
	Rect	ScaledWindowRect, WindowDrawRect;
	
	GetMapDrawRect (&WindowDrawRect);
	
	GetLargestRect (&WindowDrawRect, ViewStatusPtr -> mapViewRatio,
					 &ScaledWindowRect);
					 
	ViewStatusPtr -> mapWindowRect.top = (long) ScaledWindowRect.top;
	ViewStatusPtr -> mapWindowRect.left = (long) ScaledWindowRect.left;
	ViewStatusPtr -> mapWindowRect.bottom = (long) ScaledWindowRect.bottom;
	ViewStatusPtr -> mapWindowRect.right = (long) ScaledWindowRect.right;
	
	return;
}
/**************************************************************************************************/
OSErr CMap::SetMapScales ()
/* subroutine to set the screenToMatrix and matrixToScreen scales accoring to current
	rectangles contained in ViewStatusRec */
{
	LongRectPtr		SourceRectPtr, DestRectPtr;
	ScaleRecPtr		ScalePtr;
	OSErr			ErrCode;
	ViewStatusRec	SaveViewStatusRec;
	
	// save current ViewStatus to restore in case there is an error
	SaveViewStatusRec = *ViewStatusPtr;
	
	// get scale and offsets to convert from matrix coordinates to screen
	SourceRectPtr = &(ViewStatusPtr -> mapMatrixRect);
	DestRectPtr   = &(ViewStatusPtr -> mapWindowRect);
	ScalePtr = &(ViewStatusPtr -> matrixToScrScaleRec);	
	ErrCode = GetLScaleAndOffsets (SourceRectPtr, DestRectPtr, ScalePtr);
	
	if (!ErrCode)
	{
		// get scale and offsets to convert from screen to matrix coordinates
		SourceRectPtr = &(ViewStatusPtr -> mapWindowRect);
		DestRectPtr   = &(ViewStatusPtr -> mapMatrixRect);
		ScalePtr = &(ViewStatusPtr -> scrToMatrixScaleRec);	
		ErrCode = GetLScaleAndOffsets (SourceRectPtr, DestRectPtr, ScalePtr);
	}
	
	// if there was an error, restore view status to what it was before routine was called
	if (ErrCode)
		*ViewStatusPtr = SaveViewStatusRec;		// restore original rect

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMap::DrawMapInLRect (LongRect *MapLRectPtr, Boolean bInvalidate)
{
	double		SelRectRatio, RatioCoeff;
	OSErr		ErrCode = 0;
	Rect		WindowDrawRect;
	LongRect	ScaledZoomRect, WorldLRect, mapMatrixRect, MatrixSelRect, matrixViewRect;
	Boolean		MaxZoom;

	MatrixSelRect = *MapLRectPtr;	// duplicate LRect passed into subroutine so it may be changed
	
	// limits of the world in matrix coordinates
	SetLRect (&WorldLRect, kWorldLeft, kWorldTop, kWorldRight, kWorldBottom);

	GetMapDrawRect (&WindowDrawRect);		// map drawing rect for current window

	RatioCoeff = ViewStatusPtr -> mapViewRatio / ViewStatusPtr -> mapMatrixRatio;	
	
	// calculate the X to Y ratio of current window
	SelRectRatio = (double) (WindowDrawRect.right  - WindowDrawRect.left) /
				   (double) (WindowDrawRect.bottom - WindowDrawRect.top);
	SelRectRatio /= RatioCoeff;

	// assign the current window coordinates to the mapWindowRect field of ViewStatusRec
	ViewStatusPtr -> mapWindowRect.top    = WindowDrawRect.top;
	ViewStatusPtr -> mapWindowRect.bottom = WindowDrawRect.bottom;
	ViewStatusPtr -> mapWindowRect.left   = WindowDrawRect.left;
	ViewStatusPtr -> mapWindowRect.right  = WindowDrawRect.right;

	// make sure selection is contained within world coordinate limits
	MaxZoom = NudgeShiftLRect (&MatrixSelRect, &WorldLRect);
	if (MaxZoom)
		SetZoomedOut (true);
				
	// now scale the selection rectangle to conform to the current window ratio
	ScaledZoomRect = MatrixSelRect;
	GetSmallestLRect (&ScaledZoomRect, SelRectRatio, &ScaledZoomRect);
	
	// center the scaled rectangle within the original unscaled selection rectangle
	CenterLRectinLRect (&ScaledZoomRect, &MatrixSelRect);
	
	// set scales according to centered and scaled zoom rect
	ViewStatusPtr -> mapMatrixRect = ScaledZoomRect;

	ErrCode = SetMapScales ();

	if (!ErrCode)	// now center the zoomrect into the map view rectangle
	{
		InitViewMercator (&MatrixSelRect);
		CalcMatrixViewRect ();
						
		matrixViewRect = ViewStatusPtr -> matrixViewRect;
		mapMatrixRect  = ViewStatusPtr -> mapMatrixRect;
		
		// shift the map matrix rect in order to center it within the view rect
		CenterLRectinLRect (&mapMatrixRect, &matrixViewRect);
		ViewStatusPtr -> mapMatrixRect = mapMatrixRect;
		
		SetMapScales ();
	}
	
	if (!ErrCode && bInvalidate)
	{
		InvalWDrawRect ();
//		InvalInfoBar (wPtr, false);					// invalidate info bar to update map scale
	}
	
	if (ErrCode)
		TechError ("CMap::DrawMapInLRect()", "Zoom-in area too small!", ErrCode);

	return (ErrCode);
}
/**************************************************************************************************/
void CMap::CalcMatrixViewRect ()
/* this subroutine calculates the matrixViewRect field of ViewStatusRec by getting matrix point
	equivalents of each of the two corner points.  This is a long rect and is not scaled */
{
	Point			CornerPt;
	LongPoint		MatrixLPoint;
	LongRect		matrixViewRect;
	Rect			WindowDrawRect;
	
	GetMapDrawRect (&WindowDrawRect);

	// first calculate the matrix equivalent of the top-left corner of window
	CornerPt.h = WindowDrawRect.left;
	CornerPt.v = WindowDrawRect.top;
	GetMatrixPoint (&CornerPt, &MatrixLPoint);
	matrixViewRect.left = MatrixLPoint.h;
	matrixViewRect.top = MatrixLPoint.v;
	
	// now calculate the matrix equivalent of the bottom-right corner of the window
	CornerPt.h = WindowDrawRect.right;
	CornerPt.v = WindowDrawRect.bottom;
	GetMatrixPoint (&CornerPt, &MatrixLPoint);
	matrixViewRect.right = MatrixLPoint.h;
	matrixViewRect.bottom = MatrixLPoint.v;
	
	ViewStatusPtr -> matrixViewRect = matrixViewRect;

	return;
}
/**************************************************************************************************/
void CMap::GetMapLBounds (LongRect *MapBoundsLRectPtr)
{
	*MapBoundsLRectPtr = ViewStatusPtr -> mapBoundsRect;
	
	return;
}
/**************************************************************************************************/
void CMap::GetMapMatrixRect (LongRect *MapMatrixRectPtr)
{
	*MapMatrixRectPtr = ViewStatusPtr -> mapMatrixRect;
	
	return;
}
/**************************************************************************************************/
void CMap::GetMapViewLRect (LongRect *ViewLRectPtr)
{
	*ViewLRectPtr = ViewStatusPtr -> matrixViewRect;
	
	return;
}
/**************************************************************************************************/
void CMap::SetMapLBounds (LongRect *MapBoundsLRectPtr)
{
	ViewStatusPtr -> mapBoundsRect = *MapBoundsLRectPtr;

	return;
}
/**************************************************************************************************/
void CMap::GetMapDrawRect (Rect *WDrawRectPtr)
{
	*WDrawRectPtr = ViewStatusPtr -> mapWDrawRect;
	
	return;
}
/**************************************************************************************************/
void CMap::GetMatrixPoint (Point *ScrPointPtr, LongPointPtr LPointPtr)
/* this subroutine uses GetMatrixLPoint () to convert the given screen point into matrix coordinates */
{
	LongPoint	ScreenLPoint;
	
	ScreenLPoint.h = ScrPointPtr -> h;
	ScreenLPoint.v = ScrPointPtr -> v;
	GetMatrixLPoint (&ScreenLPoint, LPointPtr);
	
	return;
}
/**************************************************************************************************/
void CMap::GetMatrixLPoint (LongPoint *ScrLPointPtr, LongPointPtr LPointPtr)
{
	double	exLat, exLong;
	
	if (ViewStatusPtr -> mapProjCode == kLatLongProjCode)
	{
		LPointPtr -> h = ScrLPointPtr -> h * ViewStatusPtr -> scrToMatrixScaleRec.XScale +
							ViewStatusPtr -> scrToMatrixScaleRec.XOffset;
		LPointPtr -> v = ScrLPointPtr -> v * ViewStatusPtr -> scrToMatrixScaleRec.YScale +
							ViewStatusPtr -> scrToMatrixScaleRec.YOffset;
	}
	
	// now esure valid range
	if (LPointPtr -> h < kWorldLeft)   LPointPtr -> h = kWorldLeft;
	if (LPointPtr -> h > kWorldRight)  LPointPtr -> h = kWorldRight;
	if (LPointPtr -> v < kWorldBottom) LPointPtr -> v = kWorldBottom;
	if (LPointPtr -> v > kWorldTop)    LPointPtr -> v = kWorldTop;
	
	return;
}
/**************************************************************************************************/
OSErr CMap::GetMatrixLRect (Rect *ScreenRectPtr, LongRectPtr MatrixRectPtr)
// given a screen rectangle, this subroutine returns its matrix rectangle equivalent
{
	OSErr		ErrCode;
	LongRect	NormMatrixRect;
	LongPoint	ScreenLPoint, MatrixLPoint;
	
	// use GetMatrixLPoint () to convert top-left screen coordinates into matrix equivalents
	ScreenLPoint.h = ScreenRectPtr -> left;
	ScreenLPoint.v = ScreenRectPtr -> top;
	GetMatrixLPoint (&ScreenLPoint, &MatrixLPoint);
	MatrixRectPtr -> left = MatrixLPoint.h;
	MatrixRectPtr -> top = MatrixLPoint.v;
	
	// now use GetMatrixLPoint () to convert bottom-right screen coordinates into matrix equivalents
	ScreenLPoint.h = ScreenRectPtr -> right;
	ScreenLPoint.v = ScreenRectPtr -> bottom;
	GetMatrixLPoint (&ScreenLPoint, &MatrixLPoint);
	MatrixRectPtr -> right = MatrixLPoint.h;
	MatrixRectPtr -> bottom = MatrixLPoint.v;
	
	if (ViewStatusPtr -> mapProjCode == kLatLongProjCode)
	{
		MatrixRectPtr -> left = ScreenRectPtr -> left * ViewStatusPtr -> scrToMatrixScaleRec.XScale +
													    ViewStatusPtr -> scrToMatrixScaleRec.XOffset;
														
		MatrixRectPtr -> right = ScreenRectPtr -> right * ViewStatusPtr -> scrToMatrixScaleRec.XScale +
													      ViewStatusPtr -> scrToMatrixScaleRec.XOffset;
	
		MatrixRectPtr -> top = ScreenRectPtr -> top * ViewStatusPtr -> scrToMatrixScaleRec.YScale +
													  ViewStatusPtr -> scrToMatrixScaleRec.YOffset;
	
		MatrixRectPtr -> bottom = ScreenRectPtr -> bottom * ViewStatusPtr -> scrToMatrixScaleRec.YScale +
													        ViewStatusPtr -> scrToMatrixScaleRec.YOffset;
	}
														
	// normalize rect in case top is greater than bottom
	NormMatrixRect = *MatrixRectPtr;
	NormalizeLRect (&NormMatrixRect);

	ErrCode = 0;
	
	// check to see if zoom tollerence has been exceeded
	if (NormMatrixRect.right - NormMatrixRect.left < kZoomToller)
		ErrCode = 1;
	
	if (NormMatrixRect.bottom - NormMatrixRect.top < kZoomToller)
		ErrCode = 1;
		
	return (ErrCode);
}
/**************************************************************************************************/
void CMap::GetScrLPoint (LongPointPtr LPointPtr, LongPoint *ScrLPointPtr)
// subroutine to convert matrix point into a screen point
{
	LongPoint	ScreenLPoint;

	if (ViewStatusPtr -> mapProjCode == kLatLongProjCode)
	{
		ScrLPointPtr -> h = SameDifference (ViewStatusPtr -> mapMatrixRect.left,
										  	ViewStatusPtr -> mapMatrixRect.right,
										  	ViewStatusPtr -> mapWindowRect.left,
										  	ViewStatusPtr -> mapWindowRect.right,
										  	LPointPtr -> h) + 1;
										  
		ScrLPointPtr -> v = SameDifference (ViewStatusPtr -> mapMatrixRect.bottom,
										  	ViewStatusPtr -> mapMatrixRect.top,
										  	ViewStatusPtr -> mapWindowRect.bottom,
										  	ViewStatusPtr -> mapWindowRect.top,
										  	LPointPtr -> v) + 1;
	}
	
	return;
}
/**************************************************************************************************/
void CMap::GetScrPoint (LongPointPtr LPointPtr, Point *ScrPointPtr)
// subroutine to convert matrix point into a screen point
{
	LongPoint	ScreenLPoint;

	GetScrLPoint (LPointPtr, &ScreenLPoint);
	
	// now make sure our coordinates don't cause a short-integer to overflow
	if (ScreenLPoint.h < kMinMapInt)
		ScreenLPoint.h = kMinMapInt;
	else if (ScreenLPoint.h > kMaxMapInt)
		ScreenLPoint.h      = kMaxMapInt;
	
	if (ScreenLPoint.v < kMinMapInt)
		ScreenLPoint.v = kMinMapInt;
	else if (ScreenLPoint.v > kMaxMapInt)
		ScreenLPoint.v      = kMaxMapInt;

	ScrPointPtr -> h = (short) ScreenLPoint.h;
	ScrPointPtr -> v = (short) ScreenLPoint.v;

	return;
}
/**************************************************************************************************/
// given a long rectangle containing matrix coordinates, it returns the corresponding screen rect
void CMap::GetScrRect (LongRectPtr LRectPtr, Rect *ScrRectPtr)
{
	LongPoint	CornerLPoint;
	Point		ScrPoint;

	CornerLPoint.h = LRectPtr -> left;
	CornerLPoint.v = LRectPtr -> top;
	GetScrPoint (&CornerLPoint, &ScrPoint);
	
	ScrRectPtr -> left = ScrPoint.h;
	ScrRectPtr -> top = ScrPoint.v;
	
	CornerLPoint.h = LRectPtr -> right;
	CornerLPoint.v = LRectPtr -> bottom;
	GetScrPoint (&CornerLPoint, &ScrPoint);
	
	ScrRectPtr -> right = ScrPoint.h;
	ScrRectPtr -> bottom = ScrPoint.v;
	
	return;
}
/**************************************************************************************************/
/* given a long rectangle containing matrix coordinates, it returns the corresponding screen rect */
/* Note: this subroutine returns the rect in longRect coordinates */
void CMap::GetScrLRect (LongRectPtr LRectPtr, LongRect *ScrLRectPtr)
{
	LongPoint	CornerLPoint;
	LongPoint	ScrLPoint;
	
	CornerLPoint.h = LRectPtr -> left;
	CornerLPoint.v = LRectPtr -> top;
	GetScrLPoint (&CornerLPoint, &ScrLPoint);
	
	ScrLRectPtr -> left = ScrLPoint.h;
	ScrLRectPtr -> top = ScrLPoint.v;
	
	CornerLPoint.h = LRectPtr -> right;
	CornerLPoint.v = LRectPtr -> bottom;
	GetScrLPoint (&CornerLPoint, &ScrLPoint);
	
	ScrLRectPtr -> right = ScrLPoint.h;
	ScrLRectPtr -> bottom = ScrLPoint.v;
	
	return;
}
/**************************************************************************************************/
/* subroutine to zoom map in using a user selected area or using a zoom-in-point.			      */
/* if MarqueeFlag is true, the user selects the area, else user selects a point. 				  */
OSErr CMap::ZoomMapIn (Point LocalPt)
{
	Rect		ScrSelRect, WindowDrawRect;
	LongRect	MatrixSelRect, SaveMapWindowRect;
	OSErr		ErrCode = noErr;
	
	GetMapDrawRect (&WindowDrawRect);
		
	ErrCode = SelectZoomRect (LocalPt, &ScrSelRect);
	if (!ErrCode)
	{				
		SaveMapWindowRect = ViewStatusPtr -> mapWindowRect;

		// update the mapWindowRect field of ViewStatusRec in case the window has been resized
		CalcMapWindowRect ();

		// convert the above obtained screen rect into equivalent matrix rectangle
		ErrCode = GetMatrixLRect (&ScrSelRect, &MatrixSelRect);
		if (ErrCode)
		{
			TechError ("CMap::ZoomMapIn()", "Zoom-in area too small!", ErrCode);
			ViewStatusPtr -> mapWindowRect = SaveMapWindowRect;	// restore saved map window rect
		}
		else
		{
			// do zoom effect before map is redrawn
//			ZoomEffect (wPtr, &ScrSelRect, &WindowDrawRect, 8);
			
			ErrCode = DrawMapInLRect (&MatrixSelRect, true);
			if (!ErrCode)
			{
//				InvalInfoBar (wPtr, false);
				SetZoomedOut (false);
			}
		}
	}
	
	return (ErrCode);
}
/**************************************************************************************************/
// subroutine to zoom map out by 50% unless matrix boundary rect is exceeded
// note: this subroutine assumes the current port to be the main window
OSErr CMap::ZoomMapOut (Point LocalPt)
{
	LongRect	MatrixDrawRect, MaxMatrixRect;
	long		ReductionDX, ReductionDY;
	OSErr		ErrCode = 0;
	Boolean		bMaxZoom;
	Rect		WindowDrawRect, ScrZoomRect;

	GetMapDrawRect (&WindowDrawRect);
	bMaxZoom = false;

	if (MyPtInRect (LocalPt, &WindowDrawRect) && !IsZoomedOut ())
	{
		if (ViewStatusPtr -> mapProjCode == kLatLongProjCode)
			MatrixDrawRect = ViewStatusPtr -> mapMatrixRect;
		
		MaxMatrixRect = ViewStatusPtr -> mapBoundsRect;

		ReductionDX = labs (MatrixDrawRect.right  - MatrixDrawRect.left) / 4;
		ReductionDY = labs (MatrixDrawRect.top - MatrixDrawRect.bottom)  / 4;
	
		InsetLRect (&MatrixDrawRect, -ReductionDX, -ReductionDY);
	
		bMaxZoom = NudgeShiftLRect (&MatrixDrawRect, &MaxMatrixRect);
		if (bMaxZoom)
			SetZoomedOut (true);
		
		// do zoom effect before map is redrawn
		ScrZoomRect = WindowDrawRect;
		ReductionDX = (WindowDrawRect.right - WindowDrawRect.left) / 4;
		ReductionDY = (WindowDrawRect.bottom - WindowDrawRect.top) / 4;
		MyInsetRect (&ScrZoomRect, ReductionDX, ReductionDY);
//		PlaySound (kZoomOutSound);
//		ZoomEffect (wPtr, &WindowDrawRect, &ScrZoomRect, 7);
	
		ErrCode = DrawMapInLRect (&MatrixDrawRect, true);
	}
	else
		ErrCode = 1;
		
	return (ErrCode || bMaxZoom);
}
/**************************************************************************************************/
void CMap::LMoveTo (long MatrixX, long MatrixY)
{
	Point		ScreenPt;
	LongPoint	MatrixPt;

	MatrixPt.h = MatrixX;
	MatrixPt.v = MatrixY;
	
	GetScrPoint (&MatrixPt, &ScreenPt);
	MyMoveTo (ScreenPt.h, ScreenPt.v);

	return;
}
/**************************************************************************************************/
void CMap::LLineTo (long MatrixX, long MatrixY)
{
	Point		ScreenPt;
	LongPoint	MatrixPt;

	MatrixPt.h = MatrixX;
	MatrixPt.v = MatrixY;
	
	GetScrPoint (&MatrixPt, &ScreenPt);
	MyLineTo (ScreenPt.h, ScreenPt.v);

	return;
}
/**************************************************************************************************/
void CMap::SetZoomedOut (Boolean bZoomedOut)
{
	ViewStatusPtr -> bZoomedOut = bZoomedOut;

	return;
}
/**************************************************************************************************/
Boolean CMap::IsZoomedOut ()
{
	return (ViewStatusPtr -> bZoomedOut);
}
/**************************************************************************************************/
long CMap::GetMapProjCode ()
{
	return (ViewStatusPtr -> mapProjCode);
}
/**************************************************************************************************/
void CMap::SetMapProjCode (long ProjCode)
{
	LongRect	CurrViewLRect;
	Rect		WindowDrawRect;
	LongRect	WorldLRect;
	
	// limits of the world in matrix coordinates
	SetLRect (&WorldLRect, kWorldLeft, kWorldTop, kWorldRight, kWorldBottom);

	GetMapDrawRect (&WindowDrawRect);
	
	if (ViewStatusPtr -> mapProjCode != ProjCode)
	{
		// get the current geo-view rectangle
		if (ViewStatusPtr -> mapProjCode == kLatLongProjCode)
			CurrViewLRect = ViewStatusPtr -> matrixViewRect;
		
		// trim off excess borders beyond world limits
		TrimLRect (&CurrViewLRect, &WorldLRect);
		
		ViewStatusPtr -> mapProjCode = ProjCode;
		
		CalcMapRatios ();						// added 5/24/94, needed to switch projections
		DrawMapInLRect (&CurrViewLRect, true);
	}
}
/**************************************************************************************************/
void CMap::InitViewMercator (LongRect *ViewLRectPtr)
{	
	return;
}
/**************************************************************************************************/
void CMap::GetMapViewWidth (long UnitCode, double *ViewWidthPtr)
{
	LongRect	mapBoundsRect;
	char		ScaleLabelStr [255];
	LongRect	WorldLRect;
	LongPoint	LPt1, LPt2;
	Rect		WindowDrawRect;
	short		horizPPI;// vertPPI;
	double		DrawRectInches;

	GetMapDrawRect (&WindowDrawRect);
	mapBoundsRect = ViewStatusPtr -> matrixViewRect;
	SetLRect (&WorldLRect, kWorldLeft, kWorldTop, kWorldRight, kWorldBottom);
	TrimLRect (&mapBoundsRect, &WorldLRect);
	
	LPt1.h = mapBoundsRect.left;
	LPt1.v = (mapBoundsRect.top + mapBoundsRect.bottom) / 2;
	LPt2.h = mapBoundsRect.right;
	LPt2.v = (mapBoundsRect.top + mapBoundsRect.bottom) / 2;
	GetLPtsDist (&LPt2, &LPt1, kMilesCode, ViewWidthPtr);
	
	switch (UnitCode)
	{
		case kInchesCode:			// calculate 1 inch = n miles
			horizPPI =PixelsPerInchCurrent();
			//ScreenRes (&horizPPI, &vertPPI);
			DrawRectInches = ((double) (WindowDrawRect.right - WindowDrawRect.left) /
							  (double) horizPPI);
			*ViewWidthPtr = *ViewWidthPtr / DrawRectInches;
		break;
		
		case kRatioCode:				// calculate 1:N ratio
			horizPPI =PixelsPerInchCurrent();
			//ScreenRes (&horizPPI, &vertPPI);
			DrawRectInches = ((double) (WindowDrawRect.right - WindowDrawRect.left) /
							  (double) horizPPI);
			*ViewWidthPtr = (*ViewWidthPtr / DrawRectInches) / INCHESTOMILES;
		break;
	};

	return;
}
/**************************************************************************************************/
long CMap::GetMapScale ()
{
	double	exRatio;

	GetMapViewWidth (kRatioCode, &exRatio);

	return ((long) exRatio);
}
/**************************************************************************************************/
void CMap::InvalRect (Rect *r)
{
	InvalRectInWindow (*r,wPtr);
	return;
}
/**************************************************************************************************/
void CMap::InvalWDrawRect ()
{
	Rect	WindowDrawRect;
	
	GetMapDrawRect (&WindowDrawRect);		// map drawing rect for current window
	InvalRectInWindow (WindowDrawRect,wPtr);
	
	return;
}
/**************************************************************************************************/
void CMap::ClipWDrawRect ()
{
	GrafPtr SavePort;
	Rect	WindowDrawRect;
	
	GetPort (&SavePort);
	GetMapDrawRect (&WindowDrawRect);		// map drawing rect for current window
	MyClipRect (WindowDrawRect);
	
	SetPort (SavePort);
	
	return;
}
/**************************************************************************************************/
void CMap::GetPrefLines (long *prefLats, long *prefLongs)
{
	*prefLats  = ViewStatusPtr -> prefLats;
	*prefLongs = ViewStatusPtr -> prefLongs;
	
	return;
}
/**************************************************************************************************/
void CMap::SetPrefLines (long newPrefLats, long newPrefLongs)
{
	ViewStatusPtr -> prefLats  = newPrefLats;
	ViewStatusPtr -> prefLongs = newPrefLongs;
	
	return;
}
/**************************************************************************************************/
OSErr CMap::SaveMapData (short FRefNum)
{
	OSErr	ErrCode = 0;
	long	StructSize, WriteCount;
	OSType	DataType;
	
	// save a tag indicating type of data to follow
	DataType = 'Map ';
	WriteCount = sizeof (OSType);
	FSWrite (FRefNum, &WriteCount, &DataType);
	
	// save the current size of view status record
	WriteCount = sizeof (long);
	StructSize = sizeof (SavedMDataRec);
	FSWrite (FRefNum, &WriteCount, &StructSize);
	
	// now save the data in view status record
	WriteCount = sizeof (SavedMDataRec);
	FSWrite (FRefNum, &WriteCount, &(ViewStatusPtr -> bVFirstField));

	return (ErrCode);
}
/**************************************************************************************************/
OSErr CMap::ReadMapData (short FRefNum, Boolean bInvalidate)
{
	OSErr		ErrCode = 0;
	long		StructSize, ReadCount;
	LongRect	saveLRect;
	Boolean		bSaveZoom;

	// read the size of view status record when file was saved
	ReadCount = sizeof (long);
	FSRead (FRefNum, &ReadCount, &StructSize);
	
	// read the data into view status record
	FSRead (FRefNum, &StructSize, &(ViewStatusPtr -> bVFirstField));
	
	saveLRect = ViewStatusPtr -> matrixViewRect;	// previously saved view rect
	bSaveZoom = ViewStatusPtr -> bZoomedOut;
	
	ViewAllMap (false);								// view-all resets map params incl. zoomOut
	DrawMapInLRect (&saveLRect, bInvalidate);		// zoom to previous view rect
	SetZoomedOut (bSaveZoom);

	return (ErrCode);
}
/**************************************************************************************************/
long CMap::MatrixToPixels (long MatrixDist)
// pixel distance is calculated based on the latitude at the center of the current view rectangle
{
	LongPoint	LPt1, LPt2, ScrLPt1, ScrLPt2;
	LongRect	MapBoundsLRect;
	long		MidLat, ScrDist;
	
	GetMapLBounds (&MapBoundsLRect);
	MidLat = (MapBoundsLRect.top + MapBoundsLRect.bottom) / 2;
//	Debug ("MidLat = %ld\n", MidLat);

	LPt1.v = LPt2.v = MidLat;
	LPt1.h = 0;
	LPt2.h = MatrixDist;
	
	GetScrLPoint (&LPt1, &ScrLPt1);
	GetScrLPoint (&LPt2, &ScrLPt2);
	
//	Debug ("ScrLPt.h = %ld, ScrLPt.v = %ld, ScrLPt2.h = %ld, ScrLPt2.v = %ld\n",
//			ScrLPt1.h, ScrLPt1.v,			ScrLPt2.h, ScrLPt2.v);
	
	ScrDist = labs (ScrLPt2.h - ScrLPt1.h);

	return (ScrDist);
}
/**************************************************************************************************/
//CMap::~CMap ()
//{
/*	if (MapMercProj != nil)
	{
		delete (MapMercProj);
		MapMercProj = nil;
	}
*/
	//return;
//}
/**************************************************************************************************/
OSErr CMap::SelectZoomRect (Point LocalPt, Rect* ScrSelRectPtr)
// subroutine to select the area into which the user wants to zoom in to a map on the screen
{
	OSErr	ErrCode = 0;
	Rect	WindowDrawRect;
	
	GetMapDrawRect (&WindowDrawRect);
	
//JLM	ErrCode = GetDragRect (LocalPt, true, kZoomCursID, ScrSelRectPtr);
	if (ErrCode == 1)	// rect too small
		ErrCode = SelectZoomPoint (LocalPt, &WindowDrawRect, ScrSelRectPtr);

	return (ErrCode);
}
/**************************************************************************************************/
Boolean CMap::IsColorMap ()
// returns true if screen depth is able to support color & if map color flag is set to true
{
	Boolean	bColorFlag = true;

#ifdef IBM
	bColorFlag = true;
	// code goes shere
#else
	if (GetScrDepth () < 4)
		bColorFlag = false;
	else
		bColorFlag = true;
		
#endif

	if (bColorFlag)
		bColorFlag = ViewStatusPtr -> bColorMap;

	return (bColorFlag);
}
/**************************************************************************************************/
void CMap::SetMapColorFlag (Boolean bColorFlag, Boolean bInvalidate)
{
	ViewStatusPtr -> bColorMap = bColorFlag;

	if (bInvalidate)
		InvalWDrawRect ();

	return;
}
/**************************************************************************************************/
Boolean CMap::IsMapModified ()
{
	return (ViewStatusPtr -> bModified);
}
/**************************************************************************************************/
void CMap::SetMapModified (Boolean bModFlag)
{
	ViewStatusPtr -> bModified = bModFlag;
	
	return;
}
/**************************************************************************************************/
void CMap::CenterToLPoint (LongPoint centerLPoint, Boolean bInvalidate)
{
	long		dx, dy;
	LongRect	mapMatrixRect;
	LongPoint	currCenterPt;
	
	mapMatrixRect = ViewStatusPtr -> mapMatrixRect;
	GetLRectCenter (&mapMatrixRect, &currCenterPt);
	
	dx = centerLPoint.h - currCenterPt.h;
	dy = centerLPoint.v - currCenterPt.v;
	OffsetLRect (&mapMatrixRect, dx, dy);
	
	ViewStatusPtr -> mapMatrixRect = mapMatrixRect;
//	if (ViewStatusPtr -> mapProjCode == kMercatorProjCode)
//		MapMercProj -> OffsetMercProj (-dx, -dy);
	
	SetMapScales ();		// set scales to reflect new window size
	CalcMatrixViewRect ();	// update the matrix view rect
	
	if (bInvalidate)
		InvalWDrawRect ();

	return;
}
