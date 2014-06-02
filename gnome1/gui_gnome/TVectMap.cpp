
#include "Cross.h"
#include "Classes.h"
#include "ObjectUtilsPD.h"
#include "GenDefs.h"

/**************************************************************************************************/
Boolean RemoveQuotes (char *stringPtr)
// this routine removes any quote characters from the given string and returns true if it had any
{
	char	strNoQuotes [256];
	short	strNoQuotesLen, strIndex;
	Boolean	bHasQuotes;

	bHasQuotes = false;
	strNoQuotesLen = 0;
	for (strIndex = 0; strIndex < strlen (stringPtr); ++strIndex)
	{
		if (stringPtr [strIndex] != '"')
		{
			strNoQuotes [strNoQuotesLen] = stringPtr [strIndex];
			++strNoQuotesLen;
		}
		else
			bHasQuotes = true;	
	}
	strNoQuotes [strNoQuotesLen] = '\0';	// null terminate the quoteless string
	
	// now copy quoteless string to string that was passed in
	strcpy (stringPtr, strNoQuotes);
	
	return bHasQuotes;		// return true if string had quotes
}
/**************************************************************************************************/
Boolean GetStrExtension (char *labelStr, char *extStr)
/* this subroutine returns true if the given string has a suffix attached to it and false if it does
	not.  If the string does not have a suffix, the extension string is zeroed out.  Otherwise, the
	string's extension is copied to it starting with the first character followind the '.'
*/
{
	char	*extPtr;
	Boolean	bPeriodFound = false;
	short	strIndex;
	
	for (strIndex = 0; strIndex < strlen (labelStr); ++strIndex)
	{
		// if period is found and it is not the last character in the string
		if (labelStr [strIndex] == '.' && strlen (labelStr) > strIndex)
		{
			bPeriodFound = true;
			++strIndex;
			extPtr = labelStr + strIndex;	// make extptr pointer point to char after period
			break;
		}
	}
	
	if (bPeriodFound)
		strcpy (extStr, extPtr);
	else
		strcpy (extStr, "");

	return (bPeriodFound);
}
/**************************************************************************************************/
TVectorMap::TVectorMap (char* name, WorldRect bounds): TMap(name, bounds)
{
	thisMapLayer = nil;
	allowableSpillLayer = nil;
	mapBoundsLayer = nil;
	esiMapLayer = nil;
	map = nil;
#ifdef MAC
	memset(&fLandWaterBitmap,0,sizeof(fLandWaterBitmap)); //JLM
	memset(&fAllowableSpillBitmap,0,sizeof(fAllowableSpillBitmap)); //JLM
	memset(&fMapBoundsBitmap,0,sizeof(fMapBoundsBitmap)); //JLM
	memset(&fESIBitmap,0,sizeof(fESIBitmap)); //JLM
	//fESIBitmap = 0;	// memset the bitmap part
#else
	fLandWaterBitmap = 0;
	fAllowableSpillBitmap = 0;
	fMapBoundsBitmap = 0;
	fESIBitmap = 0;
#endif
	bDrawLandBitMap = false;
	bDrawAllowableSpillBitMap = false;
	bSpillableAreaActive = true;
	bSpillableAreaOpen = false;

	fExtendedMapBounds = bounds;
	fUseExtendedBounds = false;
	
	bShowLegend = false;
	bDrawESIBitMap = false;
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
	
	//fBitMapResMultiple = 1;

	return;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

void DrawCMapLayer(CMap *theMap,CMapLayer* mapLayer)
{
	DrawSpecRec drawSettings;
	
	long 				ObjectCount,ObjectIndex;
	CMyList			*thisObjectList;
	PolyObjectHdl	thisObjectHdl = 0;
	
	if(!mapLayer) return; // programmer error

	thisObjectList = mapLayer->GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();
	
	// you must call this once with nil, don't ask me who coded it this way
	mapLayer -> GetDrawSettings (&drawSettings, nil, kScreenMode);
	
	// to have lakes show up as water on bitmap, draw on top and use kNoFillCode (set in GetDrawSettings)
	//for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	for (ObjectIndex = 0; ObjectIndex < ObjectCount; ObjectIndex++)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		mapLayer -> GetDrawSettings (&drawSettings, (ObjectRecHdl)thisObjectHdl, kScreenMode);

		if (drawSettings.fillCode == kNoFillCode && drawSettings.backColorInd == kWaterColorInd)	
			DrawMapBoundsPoly (theMap, thisObjectHdl,  &drawSettings, true);
		else
			DrawMapPoly (theMap, thisObjectHdl,  &drawSettings);
	}
}

/////////////////////////////////////////////////

void DrawCMapLayerNoFill(CMap *theMap,CMapLayer* mapLayer, Boolean erasePolygons)
{
	DrawSpecRec drawSettings;
	
	long 				ObjectCount,ObjectIndex;
	CMyList			*thisObjectList;
	PolyObjectHdl	thisObjectHdl = 0;
	
	if(!mapLayer) return; // programmer error

	thisObjectList = mapLayer->GetLayerObjectList ();
	ObjectCount = thisObjectList -> GetItemCount ();
	
	// you must call this once with nil, don't ask me who coded it this way
	mapLayer -> GetDrawSettings (&drawSettings, nil, kScreenMode);
	drawSettings.fillCode  = kNoFillCode;
	
	for (ObjectIndex = ObjectCount - 1; ObjectIndex >= 0; --ObjectIndex)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, ObjectIndex);
		mapLayer -> GetDrawSettings (&drawSettings, (ObjectRecHdl)thisObjectHdl, kScreenMode);
		drawSettings.fillCode  = kNoFillCode;

#ifdef MAC
		DrawMapPoly (theMap, thisObjectHdl,  &drawSettings);
#else
		DrawMapBoundsPoly (theMap, thisObjectHdl,  &drawSettings, erasePolygons);
#endif

	}
}


/////////////////////////////////////////////////

void DrawLandLayer(void * object,WorldRect wRect,Rect r)
{
	TVectorMap* tvMap = (TVectorMap*)object; // typecast 
	CMap *theMap = (CMap*)tvMap;
	CMapLayer* mapLayer = tvMap->thisMapLayer;
	DrawCMapLayer(theMap,mapLayer);
}

void DrawAllowableSpillLayer(void * object,WorldRect wRect,Rect r)
{
	TVectorMap* tvMap = (TVectorMap*)object; // typecast 
	CMap *theMap = (CMap*)tvMap;
	CMapLayer* mapLayer = tvMap->allowableSpillLayer;
	DrawCMapLayer(theMap,mapLayer);
}

void DrawMapBoundsLayer(void * object,WorldRect wRect,Rect r)
{
	TVectorMap* tvMap = (TVectorMap*)object; // typecast 
	CMap *theMap = (CMap*)tvMap;
	CMapLayer* mapLayer = tvMap->mapBoundsLayer;
	DrawCMapLayer(theMap,mapLayer);
}

void DrawESIMapLayer(void * object,WorldRect wRect,Rect r)
{
	TVectorMap* tvMap = (TVectorMap*)object; // typecast 
	CMap *theMap = (CMap*)tvMap;
	CMapLayer* mapLayer = tvMap->esiMapLayer;
	//gDrawBitmapInBlackAndWhite = false;	// this case is a color bitmap
	DrawCMapLayer(theMap,mapLayer);
	//DrawCMapLayerNoFill(theMap,mapLayer,false);
	//gDrawBitmapInBlackAndWhite = true;	// this case is a color bitmap
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

/**************************************************************************************************/
OSErr TVectorMap::InitMap () // JLM,7/31/98 this function was missing so base class got called 
{
	OSErr		err = noErr;
	
	TMap::InitMap ();

	map = new CMap;
	if (map)
		map -> IMap (mapWindow, false, false, kLatLongProjCode, false);
	else
	{
		delete (map);
		TechError ("TVectorMap ()", "map -> IMap", memFullErr);
		err = memFullErr;
	}

	if (!err)
	{
		thisMapLayer = AddNewLayer (map, nil, true, false);	/* error alert built in */
		if (thisMapLayer != nil)
		{
			thisMapLayer -> SetLayerOLabelVisible (false);
			thisMapLayer -> SetLayerOLMaxScale (15000000);
		}
		else
			err = nilHandleErr;
	}
	
	if (!err)
	{
		allowableSpillLayer = AddNewLayer (map, nil, true, false);	/* error alert built in */
		if (allowableSpillLayer != nil)
		{
			allowableSpillLayer -> SetLayerOLabelVisible (false);
			allowableSpillLayer -> SetLayerOLMaxScale (15000000);
		}
		else
			err = nilHandleErr;
	}
	
	if (!err)
	{
		mapBoundsLayer = AddNewLayer (map, nil, true, false);	/* error alert built in */
		if (mapBoundsLayer != nil)
		{
			mapBoundsLayer -> SetLayerOLabelVisible (false);
			mapBoundsLayer -> SetLayerOLMaxScale (15000000);
		}
		else
			err = nilHandleErr;
	}
	
	if (!err)
	{
		esiMapLayer = AddNewLayer (map, nil, true, false);	/* error alert built in */
		if (esiMapLayer != nil)
		{
			esiMapLayer -> SetLayerOLabelVisible (false);
			esiMapLayer -> SetLayerOLMaxScale (15000000);
		}
		else
			err = nilHandleErr;
	}
	
	return err;
}
Boolean gHighRes = true;
//Boolean gHighRes = false;
long gDesiredNumBits = 40000000L;
OSErr LandBitMapWidthHeight(WorldRect wRect, long *width, long* height)
{	// we can use the aspect ratio to get better fits
	// but make sure the width is a multiple of 32 !!!!
	//#define BWMAPWIDTH 3200 // pixels
	//#define BWMAPHEIGHT 3200  //pixels
	WorldPoint center;
	//long desiredNumBits =  10000000L;// use approx 10 million bits
	// note : increasing this can cause problems reading in SAV files, might have to increase application memory
	//long desiredNumBits =  40000000L;// use approx 40 million bits
	long desiredNumBits =  10000000L;// use approx 40 million bits
	double fraction,latDist,lngDist;
	long w,h;
	OSErr err = 0;
	
	//if (gHighRes) desiredNumBits = 40000000L;
	if (gHighRes) desiredNumBits = 100000000L;	// try something bigger 2/14/13
	center.pLat = (wRect.loLat + wRect.hiLat)/2; 
	center.pLong = (wRect.loLong + wRect.hiLong)/2; 
	
	latDist = fabs(LatToDistance(wRect.loLat - wRect.hiLat)); // in kilometers
	lngDist = fabs(LongToDistance(wRect.loLong - wRect.hiLong,center));// in kilometers
	
	// return error if either latDist or lngDist = 0
	if(lngDist > 0 ) fraction = latDist/lngDist;
	//else fraction = 0.5; // this should not happen
	//else fraction = 1.; // this should not happen
	else return -1;
	if(latDist == 0) return -1;
	
	//if(fraction < 0.01) fraction = 0.01; // minimum aspect ratios
	//if(fraction < 0.99) fraction = 0.99;
	
	if(fraction < 0.1) fraction = 0.1; // minimum aspect ratios - probably even bigger value since .25 caused a problem...
	if(fraction > 10.) fraction = 10.;
	
	// fraction = latDist/lngDist = height/width;
	// height*width = 1 meg
	// substituting yields
	// (height/width)*width^2 = 1 meg 
	
	w = sqrt(desiredNumBits/fraction);
	w += 32 - (w%32); // bump it up to the next multiple of 32
	
	h = desiredNumBits/w;
	
	// there is a hard limit on height and width of 16384 (2^14)
	if (w > 16384) w = 16384;
	if (h > 16384) h = 16384;
	
	*width = w;
	*height = h; 
	
	return err;
}

OSErr TVectorMap::InitMap (char *path)
{
	OSErr	err = noErr;
	LongRect	LayerLBounds;
	WorldRect	wRect;
	Rect bitMapRect;
	long bmWidth, bmHeight;
	
	err = this->InitMap();
	if (err) return err;

	if (path)
		err = ImportMap (path);
	else
		err = SelectMapBox(emptyWorldRect);	// CDOG option to create a water map rectangle
	if (err) return err;

	//err = ImportESIData("MacintoshHD:Desktop Folder:sanesi.imp");
	//if (err) {printNote("messed up ESI file");}
	thisMapLayer -> GetLayerScope (&LayerLBounds, true);
	wRect.hiLat = LayerLBounds.top;
	wRect.loLat = LayerLBounds.bottom;
	wRect.loLong = LayerLBounds.left;
	wRect.hiLong = LayerLBounds.right;
	
	/////////////////////////////////////////////////
	// JLM 6/10/99 include the spillable area when finding the map bounds
	// this allows the spillable area to extend out into the ocean when along the coast 
	if(this->HaveAllowableSpillLayer()){ 
		WorldRect spillableBounds;
		this->allowableSpillLayer -> GetLayerScope (&LayerLBounds, true);
		spillableBounds.hiLat = LayerLBounds.top;
		spillableBounds.loLat = LayerLBounds.bottom;
		spillableBounds.loLong = LayerLBounds.left;
		spillableBounds.hiLong = LayerLBounds.right;
		wRect = UnionWRect(spillableBounds,wRect);
	}
	/////////////////////////////////////////////////
	
	if(this->HaveMapBoundsLayer()){
		WorldRect mapBounds;
		this->mapBoundsLayer -> GetLayerScope (&LayerLBounds, true);
		mapBounds.hiLat = LayerLBounds.top;
		mapBounds.loLat = LayerLBounds.bottom;
		mapBounds.loLong = LayerLBounds.left;
		mapBounds.hiLong = LayerLBounds.right;
		wRect = UnionWRect(mapBounds,wRect);
	}

	if(this->HaveESIMapLayer()){
		WorldRect esiMapBounds;
		this->esiMapLayer -> GetLayerScope (&LayerLBounds, true);
		esiMapBounds.hiLat = LayerLBounds.top;
		esiMapBounds.loLat = LayerLBounds.bottom;
		esiMapBounds.loLong = LayerLBounds.left;
		esiMapBounds.hiLong = LayerLBounds.right;
		wRect = UnionWRect(esiMapBounds,wRect);
	}

	SetMapBounds (wRect);
	SetExtendedBounds (wRect);
	err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
	if (err){printError("Unable to create bitmap in TVectorMap::InitMap"); return err;}
	MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);

	fLandWaterBitmap = GetBlackAndWhiteBitmap(DrawLandLayer,this,wRect,bitMapRect,&err);
	
	if(!err && this->HaveAllowableSpillLayer())
		fAllowableSpillBitmap = GetBlackAndWhiteBitmap(DrawAllowableSpillLayer,this,wRect,bitMapRect,&err);
	
	if(!err && this->HaveMapBoundsLayer())
		fMapBoundsBitmap = GetBlackAndWhiteBitmap(DrawMapBoundsLayer,this,wRect,bitMapRect,&err);
	
	if(!err && this->HaveESIMapLayer())
		fESIBitmap = GetBlackAndWhiteBitmap(DrawESIMapLayer,this,wRect,bitMapRect,&err);
		/*#ifdef MAC
		{
			Rect r = MapDrawingRect();
			bitMapRect.bottom /= 1;	// this color bitmap uses a lot of memory, is there a better way?
			bitMapRect.right /= 1;
		fESIBitmap = GetColorImage(DrawESIMapLayer,this,wRect,bitMapRect,&err);
	}
		#else
			fESIBitmap = GetColorImageBitmap(DrawESIMapLayer,this,wRect,bitMapRect,&err);
	//mapImage = GetColorImageDIB(DrawBaseMap,model,view,r,&err); 
		#endif*/
	
	switch(err) 
	{
		case noErr: break;
		case memFullErr: printError("Out of memory in TVectorMap::InitMap"); break;
		default: TechError("TVectorMap::InitMap", "GetBlackAndWhiteBitmap", err); break;
	}

	return err;
}
/////////////////////////////////////////////////
// This subroutine is for CDOG map bounds 
/**************************************************************************************************/
OSErr CreateMapBox()
{
	char nameStr [256], shortFileName[64];
	char *path=nil;	// no path indicates points will be input at the dialog
	WorldRect	theRect = emptyWorldRect;
	OSErr err = 0;
	TVectorMap	*vMap;
	{
		strcpy (nameStr, "Vector Map: ");
		//strcat (nameStr, shortFileName);
		strcat (nameStr, "Map Box");
	
		vMap = (TVectorMap*) new TVectorMap (nameStr, theRect);
		if (!vMap)
			{ TechError("CreateMapBox()", "new TVectorMap()", 0); return -1; }

		if (err = vMap -> InitMap(path)) { delete vMap; return err; }
	
		if (err = model->AddMap(vMap, 0))
		{
			vMap -> Dispose ();
			delete vMap;
			err = -1;
		}
	}
	if (!err)
	{
		model->NewDirtNotification();
	}

	return err;
}
/**************************************************************************************************/
OSErr TVectorMap::ImportMap (char *path)
{
	OSErr			ReadErrCode = noErr, err = noErr;
	Size			NewHSize;
	long			FileSize, TotalBytesRead, PointCount, PointIndex, AddPointCount, ObjectCount;
	long			colorIndex;
	long			line = -1;
	WindowPtr		ProgressWPtr;
	Boolean			InterruptFlag = false, bClosed, GroupFlag, PointsAddedFlag;
	double			ExLat, ExLong;
	LongRect		ObjectLRect;
	LongPoint		MatrixPt, **thisPointsHdl;
	PolyObjectHdl	thisPolyHdl = nil;
	ObjectIDRec		FirstObjectID, LastObjectID;
	CHARH			f =0;
	char			strLine [255], ObjectName [128], ObjectName2 [128], KeyStr [255],
					*LineStartPtr;
	long numLinesText;
	CMapLayer * theLayer = nil;
	
	err = 0;
	TotalBytesRead = 0;
	ProgressWPtr = nil;

//	err = FSpOpenDF (BNAFSpecPtr, fsRdPerm, &FRefNum);
	err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f);
	if (err)
	{
		printError ("Error opening BNA map file for reading!");
		return err;
	}
	// get the total number of bytes in the file we are about to read
//	GetEOF (FRefNum, &FileSize);

//	InterruptFlag = Progress ((long) 0, "Importing BNA mapÉ", &ProgressWPtr);
	_HLock((Handle)f); // JLM 8/4/99

	numLinesText = NumLinesInText (*f);
	while (err == 0 && !InterruptFlag)
	{
		MySpinCursor(); // JLM 8/4/99
		if (++line >= numLinesText - 1)
			ReadErrCode = 1;
		else
		{
			NthLineInTextOptimized(*f, line, strLine, 255); // 10 u values
			if (strlen (strLine) <= 2)
				ReadErrCode = -1;
		}
//		ReadErrCode = FSReadLine (FRefNum, strLine, &TotalBytesRead);
		if (ReadErrCode)
			//goto done;
			break;

		if (strLine [0] == LINEFEED)
			LineStartPtr = &(strLine [1]);	// if line-feed char exists, advance pointer by one
		else
			LineStartPtr = &(strLine [0]);	// else keep it at the start of line

		err = GetHeaderLineInfo (LineStartPtr, ObjectName, ObjectName2, &PointCount);
		if (err==1)
		{
			// in case of error object-name still contains the name of the last polygon read
			sprintf (strLine, "Unexpected data encountered after reading polygon %s", ObjectName);
			printError(strLine);
		}
		if (err) goto done;
		
		// check for "Map Box" header used in digitized maps and ignore the following polygon
		if(!strcmpnocase(ObjectName,"Map Box")) // digitized map header, ignore the following polygon
		{
			for (PointIndex = 1; PointIndex <= labs(PointCount); ++PointIndex)
			{
				if (++line >= numLinesText - 1)
					ReadErrCode = 1;
				else
				{
					NthLineInTextOptimized(*f, line, strLine, 255); 
					if (strlen (strLine) <= 2)
						ReadErrCode = -1;
				}
			}
			if (ReadErrCode) 
				//goto done;
				break;
			else 
				continue;
		}
		///////////}

		///////////////{
		// JLM 12/24/98
		if(!strcmpnocase(ObjectName,"SpillableArea"))// special polygon defining spillable area
		{
			theLayer = allowableSpillLayer;
			strcpy(ObjectName2,"1");	// spillable area lakes mess up bitmap
		}
		else if(!strcmpnocase(ObjectName,"Map Bounds"))// special polygon defining the map bounds
			theLayer = mapBoundsLayer;
		else if(!strcmpnocase(ObjectName,"Ice Map"))// special polygon defining the ice shoreline (should match bna)
			theLayer = esiMapLayer;	// now using this for ESI shoreline, but leave in for now
		else 
			theLayer = thisMapLayer;
			
		///////////}
					
		if (PointCount < 0)		// negative number indicates unfilled polygon
		{
			PointCount = labs (PointCount);
			bClosed = false;
		}
		else
			bClosed = true;
	
		// set region rect to empty to start with
		SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);

		GroupFlag = false;			// clear flag indicating grouping
		PointsAddedFlag  = false;	// will be set to true if a point is appended
		AddPointCount = 0;			// number of points added
		
		// now read in the points for the next region in the file
		for (PointIndex = 1; PointIndex <= PointCount && err == 0; ++PointIndex)
		{
//				InterruptFlag = Progress ((long) (((double) TotalBytesRead / (double) FileSize) * 100.0 + 1), "", &ProgressWPtr);
			if (++line >= numLinesText - 1)
				ReadErrCode = 1;
			else
			{
				NthLineInTextOptimized(*f, line, strLine, 255); // 10 u values
				if (strlen (strLine) <= 2)
					ReadErrCode = -1;
			}
//				ReadErrCode = FSReadLine (FRefNum, strLine, &TotalBytesRead);
			if (ReadErrCode)
				//goto done;
				break;
			
			if (strLine [0] == LINEFEED)
				LineStartPtr = &(strLine [1]);	// if line-feed char exists, advance pointer by one
			else
				LineStartPtr = &(strLine [0]);	// else keep it at the start of line
			
			if (err = ScanMatrixPt (LineStartPtr, &MatrixPt)) goto done;
			if (MatrixPt.h < kWorldLeft)   MatrixPt.h = kWorldLeft;
			if (MatrixPt.h > kWorldRight)  MatrixPt.h = kWorldRight;
			if (MatrixPt.v > kWorldTop)    MatrixPt.v = kWorldTop;
			if (MatrixPt.v < kWorldBottom) MatrixPt.v = kWorldBottom;
			// code goes here, if user wants longitudes switched to Eastern hemisphere add 360, but make sure no values > 0
			//if (MatrixPt.h < 0 && gNoaaVersion)  MatrixPt.h += kWorldRight;	//  maybe set up a temp set in case of error?
			//if (MatrixPt.h < 0 && gNoaaVersion)  MatrixPt.h += kWorldRight;
			//if (PointCount <= 2)	// two point polygon not valid, but doesn't seem to cause a problem
				//continue;
			if (PointIndex == 1)	// save the key string for comparing later
				strcpy (KeyStr, LineStartPtr);
			
			if (bClosed && PointIndex > 1  && !strcmp (KeyStr, LineStartPtr))
			{
				if (PointIndex < PointCount) continue;	// skip any duplicates of the start point since they cause Windows GNOME to crash on exit
				thisPolyHdl = (PolyObjectHdl) theLayer -> AddNewObject (kPolyType, &ObjectLRect, false);
				if (thisPolyHdl != nil)
				{
					SetObjectLabel ((ObjectRecHdl) thisPolyHdl, ObjectName);
					
					// set the object label point at the center of polygon
					GetObjectCenter ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
					SetOLabelLPoint ((ObjectRecHdl) thisPolyHdl, &MatrixPt);

					// set the polygon points handle field
					SetPolyPointsHdl (thisPolyHdl, thisPointsHdl);
					SetPolyPointCount (thisPolyHdl, AddPointCount);
											
					if (!GroupFlag)
						GroupFlag = true;

					if (!strcmp (ObjectName2, "2"))			// if name indicates water polygon
					{
						colorIndex = kLgtBlueColorInd;		// use water color for this poly
						SetPolyWaterFlag (thisPolyHdl, true);
					}
					//else if (!strcmp (ObjectName2, "3"))	// ice color
						//colorIndex = kLgtGrayColorInd;	
					else
						colorIndex = kLandColorInd;

					SetObjectBColor ((ObjectRecHdl) thisPolyHdl, colorIndex);
					SetObjectColor ((ObjectRecHdl) thisPolyHdl, colorIndex);// added JLM 7/1/99

					// poke the filled flag into poly handle!!
					SetPolyClosed (thisPolyHdl, bClosed);

//						if (PointIndex == PointCount)		// last point of polygon
//							SetObjectGroupID ((ObjectRecHdl) thisPolyHdl, &FirstObjectID);
//						else
//							SetObjectGroupID ((ObjectRecHdl) thisPolyHdl, &LastObjectID);

					// set the next region's rect to empty
					SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);

					PointsAddedFlag = false;
					AddPointCount = 0;
				}
				else
				{
					err = memFullErr;
					goto done;
				}
			}
			else
			{
				if (AddPointCount == 0)
				{
					thisPointsHdl = (LongPoint**) _NewHandleClear (0);
					if (thisPointsHdl == nil)
					{
						err = memFullErr;
						goto done;
					}
				}

				NewHSize = _GetHandleSize ((Handle) thisPointsHdl) + sizeof (LongPoint);
				_SetHandleSize ((Handle) thisPointsHdl, NewHSize);
				if (_MemError ())
					err = memFullErr;

				if (err)
					goto done;
				else
				{
					++AddPointCount;
					PointsAddedFlag = true;
					
					// store the new point in the points handle
					(*thisPointsHdl) [AddPointCount - 1] = MatrixPt;
					
					// update region boundary rectangle
					if (MatrixPt.h < ObjectLRect.left)   ObjectLRect.left   = MatrixPt.h;
					if (MatrixPt.v > ObjectLRect.top)    ObjectLRect.top    = MatrixPt.v;
					if (MatrixPt.h > ObjectLRect.right)  ObjectLRect.right  = MatrixPt.h;
					if (MatrixPt.v < ObjectLRect.bottom) ObjectLRect.bottom = MatrixPt.v;
				}
			}
		}

		if (PointsAddedFlag)
		{
			thisPolyHdl = (PolyObjectHdl) theLayer -> AddNewObject (kPolyType, &ObjectLRect, false);
			if (thisPolyHdl != nil)
			{
				SetObjectLabel ((ObjectRecHdl) thisPolyHdl, ObjectName);
				
				// set the object label point at the center of polygon
				GetObjectCenter ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
				SetOLabelLPoint ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
				
				// set the polygon points handle field
				SetPolyPointsHdl (thisPolyHdl, thisPointsHdl);
				SetPolyPointCount (thisPolyHdl, AddPointCount);
														
				if (!strcmp (ObjectName2, "2"))			// if name indicates water polygon
				{
					colorIndex = kLgtBlueColorInd;		// use water color for this poly
					SetPolyWaterFlag (thisPolyHdl, true);
				}
				//else if (!strcmp (ObjectName2, "3"))	// ice color
					//colorIndex = kLgtGrayColorInd;
				else
					colorIndex = kLandColorInd;			// default land color index

				SetObjectBColor ((ObjectRecHdl) thisPolyHdl, colorIndex);
				SetObjectColor ((ObjectRecHdl) thisPolyHdl, colorIndex);// added JLM 7/1/99
				SetPolyClosed (thisPolyHdl, bClosed);
			}	
			else
			{
				err = memFullErr;
				goto done;
			}
		}
	}

done:

	_HUnlock((Handle)f); // JLM 8/4/99

	if(f) {DisposeHandle((Handle) f); f = 0; }
	if (err == memFullErr)
		printError ("Out of application memory! Try increasing application's memory partition.");

	return err;
}
// Color bitmap information from Quickdraw.h
//struct ColorTable {
	//long							ctSeed;						/*unique identifier for table*/
	//short							ctFlags;					/*high bit: 0 = PixMap; 1 = device*/
	//short							ctSize;						/*number of entries in CTTable*/
	//CSpecArray						ctTable;					/*array [0..0] of ColorSpec*/
//};
//typedef struct ColorTable ColorTable, *CTabPtr, **CTabHandle;

//struct PixMap {
	//Ptr								baseAddr;					/*pointer to pixels*/
	//short							rowBytes;					/*offset to next line*/
	//Rect							bounds;						/*encloses bitmap*/
	//short							pmVersion;					/*pixMap version number*/
	//short							packType;					/*defines packing format*/
	//long							packSize;					/*length of pixel data*/
	//Fixed							hRes;						/*horiz. resolution (ppi)*/
	//Fixed							vRes;						/*vert. resolution (ppi)*/
	//short							pixelType;					/*defines pixel type*/
	//short							pixelSize;					/*# bits in pixel*/
	//short							cmpCount;					/*# components in pixel*/
	//short							cmpSize;					/*# bits per component*/
	//long							planeBytes;					/*offset to next plane*/
	//CTabHandle						pmTable;					/*color map for this pixMap*/
	//long							pmReserved;					/*for future use. MUST BE 0*/
//};
//typedef struct PixMap PixMap, *PixMapPtr, **PixMapHandle;

long GetColorFromESICode(long code)
{
	long colorIndex;
	switch(code) 
	{
		case 1: // 1A/1B	Dark Purple
			return kESIDkPurpleInd;
			break;
		case 2: // 2A/2B	Light Purple
			return kESILgtPurpleInd;
			break;
		case 3: // 3A/3B	Blue
			return kESIBlueInd;
			break;
		case 4: // 3C/4	Light Blue
			return kESILgtBlueInd;
			break;
		case 5: // 5	Light Blue Green
			return kESILgtBlueGreenInd;
			break;
		case 6: // 6A	Green
			return kESIGreenInd;
			break;
		case 7: // 6B	Light Green
			return kESILgtGreenInd;
			break;
		case 8: // 7	Olive
			return kESIOliveInd;
			break;
		case 9: // 8A	Yellow
			return kESIYellowInd;
			break;
		case 10: // 8B	Peach
			return kESIPeachInd;
			break;
		case 11: //  8C/8D/8E/8F	Light Orange
			return kESILgtOrangeInd;
			break;
		case 12: //  9A/9B/9C	Orange
			return kESIOrangeInd;
			break;
		case 13: // 10A	Red
			return kESIRedInd;
			break;
		case 14: // 10B/10E	Light Magenta
			return kESILgtMagentaInd;
			break;
		case 15: // 10C	Dark Red
			return kESIDkRedInd;
			break;
		case 16: // 10D	Brown
			return kESIBrownInd;
			break;
		return -1;
	}
	return -1; // should have a default
}

/*long GetColorFromESICode(long code)
{
	long colorIndex;
	switch(code) 
	{
		// will need to add new colors since all blues are redefined as white in Our_PmForeColor
		case 1: return kBlackColorInd;
		//case 1: return 200;
		case 2: return 200;
		//case 2: return kVLgtGrayColorInd;
		//case 3: return kRedColorInd;
		case 3: return 100;
		//case 4: return kBlackColorInd;
		//case 4: return kRedColorInd;
		case 4: return 150;
		//case 5: return kRedColorInd;
		case 5: return 200;
		//case 6: return kRedColorInd;
		case 6: return 225;
		case 7: return kRedColorInd;
		//case 7: return 150;
		case 8: return kVLgtGrayColorInd;
		case 9: return kGrayColorInd;
		//case 10: return kDkGrayColorInd;
		case 10: return 150;
	}
	return -1; // should have a default
}*/


long GetESICode(char *ESICodeStr)
{	// will also have to handle 6B, 6B/2, ...
	// look for a slash, count characters, check alpha vs numeric
	// there could be up to 3 esi types associated with a segment, listed inland to water 
	long i,j,k,codeIndex,numCodes = 1;
	char	c, code1[64], code2[64];
	Boolean keepGoing = true;
	OSErr	ErrCode = 0;
	
	j=0;
	for (codeIndex=1; codeIndex<=2 && !ErrCode; ++codeIndex)
	{
		k=0;
		c = ESICodeStr[j];
		keepGoing = true;
		while(keepGoing)
		{
			if (ESICodeStr[j]=='/') 
			{
				j++; 
				keepGoing = false; 
				code1[k++]=0; 
				numCodes++;
			}
			else if (ESICodeStr[j]==0) 
			{
				keepGoing = false; 
				if (numCodes==2)  
				code2[k++]=0;
				else {code1[k++]=0; ErrCode = 1;}
			}
			//else if (isalnum(ESICodeStr[j]))
			else if ((c>='0' && c<='9') || (c>='A' && c<='Z') || (c>='a' && c<='z')) // for code warrior
			{
				if (codeIndex == 1)
					code1[k++] = ESICodeStr[j];
				else if (codeIndex == 2)
					code2[k++] = ESICodeStr[j];
				j++;
			}
			else ErrCode = -1;
		}
	}
	// break into 16 types, but what to do with multiple codes?
	if (!strcmp(code1, "1") || !strcmp(code1, "1A") || !strcmp(code1, "1B")) return 1;
	if (!strcmp(code1, "2") || !strcmp(code1, "2A") || !strcmp(code1, "2B")) return 2;
	if (!strcmp(code1, "3") || !strcmp(code1, "3A") || !strcmp(code1, "3B")) return 3;
	if (!strcmp(code1, "3C") || !strcmp(code1, "4")) return 4;
	if (!strcmp(code1, "5")) return 5;
	if (!strcmp(code1, "6") || !strcmp(code1, "6A")) return 6;
	if (!strcmp(code1, "6B")) return 7;
	if (!strcmp(code1, "7")) return 8;
	if (!strcmp(code1, "8") || !strcmp(code1, "8A")) return 9;
	if (!strcmp(code1, "8B")) return 10;
	if (!strcmp(code1, "8C") || !strcmp(code1, "8D") ||!strcmp(code1, "8E") ||!strcmp(code1, "8F")) return 11;
	if (!strcmp(code1, "9") || !strcmp(code1, "9A") || !strcmp(code1, "9B") ||!strcmp(code1, "9C")) return 12;
	if (!strcmp(code1, "10") || !strcmp(code1, "10A")) return 13;
	if (!strcmp(code1, "10B") || !strcmp(code1, "10E")) return 14;
	if (!strcmp(code1, "10C")) return 15;
	if (!strcmp(code1, "10D")) return 16;

	/*if (!strncmp(ESICodeStr, "1",1)) return 1;
	if (!strncmp(ESICodeStr, "2",1)) return 2;
	if (!strncmp(ESICodeStr, "3",1)) return 3;
	if (!strncmp(ESICodeStr, "4",1)) return 4;
	if (!strncmp(ESICodeStr, "5",1)) return 5;
	if (!strncmp(ESICodeStr, "6",1)) return 6;
	if (!strncmp(ESICodeStr, "7",1)) return 7;
	if (!strncmp(ESICodeStr, "8",1)) return 8;
	if (!strncmp(ESICodeStr, "9",1)) return 9;
	if (!strncmp(ESICodeStr, "10",2)) return 10;*/
	return 0; // should have a default
}
/**************************************************************************************************/
OSErr TVectorMap::ImportESIData (char *path)
{
	OSErr			ReadErrCode = noErr, err = noErr;
	Size			NewHSize;
	long			FileSize, TotalBytesRead, PointCount, PointIndex, AddPointCount, ObjectCount;
	long			colorIndex;
	long			line = -1;
	WindowPtr		ProgressWPtr;
	Boolean			InterruptFlag = false, bClosed, GroupFlag, PointsAddedFlag;
	double			ExLat, ExLong;
	LongRect		ObjectLRect;
	LongPoint		MatrixPt, **thisPointsHdl;
	PolyObjectHdl	thisPolyHdl = nil;
	ObjectIDRec		FirstObjectID, LastObjectID;
	CHARH			f =0;
	char			strLine [255], ObjectName [128], ObjectName2 [128], KeyStr [255],
					*LineStartPtr;
	long numLinesText, esiCode;
	CMapLayer * theLayer = nil;
	
	err = 0;
	TotalBytesRead = 0;
	ProgressWPtr = nil;

	err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f);
	if (err)
	{
		printError ("Error opening ESI map file for reading!");
		return err;
	}

	_HLock((Handle)f); // JLM 8/4/99

	numLinesText = NumLinesInText (*f);
	while (err == 0 && !InterruptFlag)
	{
		MySpinCursor(); // JLM 8/4/99
		if (++line >= numLinesText - 1)
			ReadErrCode = 1;
		else
		{
			NthLineInTextOptimized(*f, line, strLine, 255); 
			if (strlen (strLine) <= 2)
				ReadErrCode = -1;
		}

		if (ReadErrCode)
			break;

		if (strLine [0] == LINEFEED)
			LineStartPtr = &(strLine [1]);	// if line-feed char exists, advance pointer by one
		else
			LineStartPtr = &(strLine [0]);	// else keep it at the start of line

		err = GetHeaderLineInfo (LineStartPtr, ObjectName, ObjectName2, &PointCount);
		if (err==1)
		{
			// in case of error object-name still contains the name of the last polygon read
			sprintf (strLine, "Unexpected data encountered after reading polygon %s", ObjectName);
			printError(strLine);
		}
		if (err) goto done;
		
		esiCode = GetESICode(ObjectName2);
		// for now use the iceMapLayer to hold the esi segments, since ice is not used
		// will decide which layer to use based on the esiCode, will only need as many 
		// layers/bitmaps as different land types.
		theLayer = esiMapLayer;
			
		///////////}
					
		if (PointCount < 0)		// negative number indicates unfilled polygon - should all be negative...
		{
			PointCount = labs (PointCount);
			bClosed = false;
		}
		else
			bClosed = true;
	
		// set region rect to empty to start with
		SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);

		GroupFlag = false;			// clear flag indicating grouping
		PointsAddedFlag  = false;	// will be set to true if a point is appended
		AddPointCount = 0;			// number of points added
		
		// now read in the points for the next region in the file
		for (PointIndex = 1; PointIndex <= PointCount && err == 0; ++PointIndex)
		{
//				InterruptFlag = Progress ((long) (((double) TotalBytesRead / (double) FileSize) * 100.0 + 1), "", &ProgressWPtr);
			if (++line >= numLinesText - 1)
				ReadErrCode = 1;
			else
			{
				NthLineInTextOptimized(*f, line, strLine, 255); // 10 u values
				if (strlen (strLine) <= 2)
					ReadErrCode = -1;
			}
//				ReadErrCode = FSReadLine (FRefNum, strLine, &TotalBytesRead);
			if (ReadErrCode)
				//goto done;
				break;
			
			if (strLine [0] == LINEFEED)
				LineStartPtr = &(strLine [1]);	// if line-feed char exists, advance pointer by one
			else
				LineStartPtr = &(strLine [0]);	// else keep it at the start of line
			
			if (err = ScanMatrixPt (LineStartPtr, &MatrixPt)) goto done;
			if (MatrixPt.h < kWorldLeft)   MatrixPt.h = kWorldLeft;
			if (MatrixPt.h > kWorldRight)  MatrixPt.h = kWorldRight;
			if (MatrixPt.v > kWorldTop)    MatrixPt.v = kWorldTop;
			if (MatrixPt.v < kWorldBottom) MatrixPt.v = kWorldBottom;

			if (PointIndex == 1)	// save the key string for comparing later
				strcpy (KeyStr, LineStartPtr);
			
			if (AddPointCount == 0)
			{
				thisPointsHdl = (LongPoint**) _NewHandleClear (0);
				if (thisPointsHdl == nil)
				{
					err = memFullErr;
					goto done;
				}
			}

			NewHSize = _GetHandleSize ((Handle) thisPointsHdl) + sizeof (LongPoint);
			_SetHandleSize ((Handle) thisPointsHdl, NewHSize);
			if (_MemError ())
				err = memFullErr;

			if (err)
				goto done;
			else
			{
				++AddPointCount;
				PointsAddedFlag = true;
				
				// store the new point in the points handle
				(*thisPointsHdl) [AddPointCount - 1] = MatrixPt;
				
				// update region boundary rectangle
				if (MatrixPt.h < ObjectLRect.left)   ObjectLRect.left   = MatrixPt.h;
				if (MatrixPt.v > ObjectLRect.top)    ObjectLRect.top    = MatrixPt.v;
				if (MatrixPt.h > ObjectLRect.right)  ObjectLRect.right  = MatrixPt.h;
				if (MatrixPt.v < ObjectLRect.bottom) ObjectLRect.bottom = MatrixPt.v;
			}
		}

		if (PointsAddedFlag)
		{
			thisPolyHdl = (PolyObjectHdl) theLayer -> AddNewObject (kPolyType, &ObjectLRect, false);
			if (thisPolyHdl != nil)
			{
				SetObjectLabel ((ObjectRecHdl) thisPolyHdl, ObjectName);
				
				// set the object label point at the center of polygon
				GetObjectCenter ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
				SetOLabelLPoint ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
				
				// set the polygon points handle field
				SetPolyPointsHdl (thisPolyHdl, thisPointsHdl);
				SetPolyPointCount (thisPolyHdl, AddPointCount);
														
				colorIndex = GetColorFromESICode(esiCode);	// should have a default value
				SetPolyWaterFlag (thisPolyHdl, true);

				SetObjectBColor ((ObjectRecHdl) thisPolyHdl, colorIndex);
				SetObjectColor ((ObjectRecHdl) thisPolyHdl, colorIndex);// added JLM 7/1/99
				SetPolyClosed (thisPolyHdl, bClosed);
				SetObjectESICode ((ObjectRecHdl) thisPolyHdl,esiCode); 
			}	
			else
			{
				err = memFullErr;
				goto done;
			}
		}
	}

done:

	_HUnlock((Handle)f); // JLM 8/4/99

	if(f) {DisposeHandle((Handle) f); f = 0; }
	if (err == memFullErr)
		printError ("Out of application memory! Try increasing application's memory partition.");

	return err;
}
/**************************************************************************************************/
void TVectorMap::Dispose ()
/* once the view is saved etc., this subroutine should be called to dispose any data belonging to
	the view */
{
	if (thisMapLayer != nil)
	{
		thisMapLayer -> Dispose ();
		delete (thisMapLayer);
		thisMapLayer = nil;
	}
	
	if (allowableSpillLayer != nil)
	{
		allowableSpillLayer -> Dispose ();
		delete (allowableSpillLayer);
		allowableSpillLayer = nil;
	}
	
	if (mapBoundsLayer != nil)
	{
		mapBoundsLayer -> Dispose ();
		delete (mapBoundsLayer);
		mapBoundsLayer = nil;
	}
	
	if (esiMapLayer != nil)
	{
		esiMapLayer -> Dispose ();
		delete (esiMapLayer);
		esiMapLayer = nil;
	}
	
#ifdef MAC
	DisposeBlackAndWhiteBitMap (&fLandWaterBitmap);
	DisposeBlackAndWhiteBitMap (&fAllowableSpillBitmap);
	DisposeBlackAndWhiteBitMap (&fMapBoundsBitmap);
	DisposeBlackAndWhiteBitMap (&fESIBitmap);
	/*if (fESIBitmap)
	{
		KillGWorld (fESIBitmap);
		fESIBitmap = nil;
	}*/
#else
	if(fLandWaterBitmap) DestroyDIB(fLandWaterBitmap);
	fLandWaterBitmap = 0;
	if(fAllowableSpillBitmap) DestroyDIB(fAllowableSpillBitmap);
	fAllowableSpillBitmap = 0;
	if(fMapBoundsBitmap) DestroyDIB(fMapBoundsBitmap);
	fMapBoundsBitmap = 0;
	if(fESIBitmap) DestroyDIB(fESIBitmap);
	fESIBitmap = 0;
#endif

	TMap::Dispose();
	
	return;
}
/**************************************************************************************************/
OSErr GetHeaderLineInfo (char *strLine, char *Str1, char *Str2, long *NumOfPtsPtr)
/* given a line of the format {"Name 1","Name 2",1234}, this subroutine returns "Name 1" in Str1,
	"Name 2" in Str2 and 1234 in *NumOfPts.  The strings may also be separated by tabs instead of
	commas. */
{
	char	Delimiters [32], *TokenPtr, NumString [255];
	OSErr	err = noErr;

	if (strLine [0] != '"')
		err = 1;
	else
	{
		strcpy (Delimiters, ",\t");					/* tab and comma are the delimiters */
		TokenPtr = strtok (strLine, Delimiters);	/* get a pointer to str1 */
		if(!TokenPtr) {err=-1; goto done;}
		strcpy (Str1, TokenPtr);
		RemoveQuotes (Str1);						/* get rid of quote marks if any */
		TokenPtr = strtok (nil, Delimiters);		/* get a pointer to str2 */
		if(!TokenPtr) {err=-1; goto done;}
		strcpy (Str2, TokenPtr);
		RemoveQuotes (Str2);						/* get rid of quote marks if any */
		TokenPtr = strtok (nil, Delimiters);		/* get a pointer to count string */
		if(!TokenPtr) {err=-1; goto done;}
		strcpy (NumString, TokenPtr);
		sscanf (NumString, "%ld\n", NumOfPtsPtr);	/* convert string to number */
	}

done:
	if (err==-1)
		printError("Error reading map header line");
	return err;
}
/**************************************************************************************************/
#define TVectorMapREADWRITEVERSION 2 //spillable area active
//#define TVectorMapREADWRITEVERSION 1 //JLM
OSErr TVectorMap::Read(BFPB *bfpb)
{
	long version;
	ClassID id;
	long 	numObjs,numPolyPts,i,ptIndex;
	OSType thisObjectType;
	WorldRect	wRect;
	char c;
	LongRect	LayerLBounds;
	OSErr err =0;
	Rect bitMapRect;
	
	if (err = TMap::Read(bfpb)) return err;

	StartReadWriteSequence("TVectorMap::Read()");

	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != this->GetClassID()) goto cantReadFile;

	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TVectorMapREADWRITEVERSION || version < 1) goto cantReadFile;
	
	// read in the  map-specfic fields
	if(err = thisMapLayer->Read(bfpb)) return err;
	if(err = allowableSpillLayer->Read(bfpb)) return err;
	if(err = mapBoundsLayer->Read(bfpb)) return err;
	if(err = esiMapLayer->Read(bfpb)) return err;
	
	
	//////// code goes here ---- do we need this ???
	/////////{
	//CMap				*map;
	// we don't write out the map field
	// since we only need it's bounding rect and we find that from the layer
	// all we need do is set the bounding rect of the newly created CMap object 
	// Code goes here -- JLM, am I right about the "map" field ??
	thisMapLayer -> GetLayerScope (&LayerLBounds, true);
	wRect.hiLat = LayerLBounds.top;
	wRect.loLat = LayerLBounds.bottom;
	wRect.loLong = LayerLBounds.left;
	wRect.hiLong = LayerLBounds.right;
	
	/////////////////////////////////////////////////
	// JLM 6/10/99 include the spillable area when finding the map bounds
	// this allows the spillable area to extend out into the ocean when along the coast 
	if(this->HaveAllowableSpillLayer()){ 
		WorldRect spillableBounds;
		this->allowableSpillLayer -> GetLayerScope (&LayerLBounds, true);
		spillableBounds.hiLat = LayerLBounds.top;
		spillableBounds.loLat = LayerLBounds.bottom;
		spillableBounds.loLong = LayerLBounds.left;
		spillableBounds.hiLong = LayerLBounds.right;
		wRect = UnionWRect(spillableBounds,wRect);
	}
	/////////////////////////////////////////////////
	
	if(this->HaveMapBoundsLayer()){
		WorldRect mapBounds;
		this->mapBoundsLayer -> GetLayerScope (&LayerLBounds, true);
		mapBounds.hiLat = LayerLBounds.top;
		mapBounds.loLat = LayerLBounds.bottom;
		mapBounds.loLong = LayerLBounds.left;
		mapBounds.hiLong = LayerLBounds.right;
		wRect = UnionWRect(mapBounds,wRect);
	}

	if(this->HaveESIMapLayer()){
		WorldRect esiMapBounds;
		this->esiMapLayer -> GetLayerScope (&LayerLBounds, true);
		esiMapBounds.hiLat = LayerLBounds.top;
		esiMapBounds.loLat = LayerLBounds.bottom;
		esiMapBounds.loLong = LayerLBounds.left;
		esiMapBounds.hiLong = LayerLBounds.right;
		wRect = UnionWRect(esiMapBounds,wRect);
	}

	/////////////////////////////////////////////////
	if(!EqualWRects(fMapBounds,wRect)) printError("Programmer error in TVectorMap::Read"); // code goes here
	fMapBounds = wRect;
	/////}
	
	
	//BitMap				fLandWaterBitmap; 
	//BitMap				fAllowableSpillBitmap; 
	// these fields were not written out since they can be reconstructed
	// we reconstruct them below after everything is read in

	if (err = ReadMacValue(bfpb, &bDrawLandBitMap)) return err;
	if (err = ReadMacValue(bfpb, &bDrawAllowableSpillBitMap)) return err;
	
	//////////////////
	// now reconstruct the offscreen Land/Water bitmap
	///////////////////
	
	long bmWidth, bmHeight;
	//if (err = ReadMacValue(bfpb, &fBitMapResMultiple)) return err;
	err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
	if (err) {printError("Unable to recreate bitmap in TVectorMap::Read"); return err;}
	MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);

	fLandWaterBitmap = GetBlackAndWhiteBitmap(DrawLandLayer,this,wRect,bitMapRect,&err);

	if(!err && this->HaveAllowableSpillLayer())
		fAllowableSpillBitmap = GetBlackAndWhiteBitmap(DrawAllowableSpillLayer,this,wRect,bitMapRect,&err);
 		
	if(!err && this->HaveMapBoundsLayer())
		fMapBoundsBitmap = GetBlackAndWhiteBitmap(DrawMapBoundsLayer,this,wRect,bitMapRect,&err);
 		
	if(!err && this->HaveESIMapLayer())
		fESIBitmap = GetBlackAndWhiteBitmap(DrawESIMapLayer,this,wRect,bitMapRect,&err);
		//fESIBitmap = GetColorImageBitmap(DrawESIMapLayer,this,wRect,bitMapRect,&err);
 		
	switch(err) 
	{
		case noErr: break;
		case memFullErr: printError("Out of memory in TVectorMap::Read"); break;
		default: TechError("TVectorMap::Read", "GetBlackAndWhiteBitmap", err); break;
	}
	
	if (version>1 && HaveAllowableSpillLayer())
	{
		if (err = ReadMacValue(bfpb,&bSpillableAreaActive)) return err;
		if (err = ReadMacValue(bfpb,&bSpillableAreaOpen)) return err;
	}
	if (err = ReadMacValue(bfpb, &fExtendedMapBounds)) return err;
	if (err = ReadMacValue(bfpb, &fUseExtendedBounds)) return err;
	
	return err;
	
cantReadFile:
		printSaveFileVersionError(); 
		return -1; 
	
}
/**************************************************************************************************/
OSErr TVectorMap::Write (BFPB *bfpb)
{
	long 			version = TVectorMapREADWRITEVERSION;
	ClassID 		id = this->GetClassID();
	OSErr			err = noErr;

	if (err = TMap::Write(bfpb)) return err;

	if (!thisMapLayer || ! allowableSpillLayer /* || !mapBoundsLayer || !esiMapLayer */)
		{ TechError("TVectorMap::Write()", "map", 0); return -1; }

	StartReadWriteSequence("TVectorMap::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;

	// begin writing map-specfic fields
	if(err = thisMapLayer->Write(bfpb)) return err;
	if(err = allowableSpillLayer->Write(bfpb)) return err;
	if(err = mapBoundsLayer->Write(bfpb)) return err;
	if(err = esiMapLayer->Write(bfpb)) return err;

	//CMap				*map;
	// we don't write out the map field
	// since we only need its bounding rect and we find that from the layer
	
	//BitMap				fLandWaterBitmap; 
	//BitMap				fAllowableSpillBitmap; 
	// these fields are not written out since they can be reconstructed

	if (err = WriteMacValue(bfpb,bDrawLandBitMap)) return err;
	if (err = WriteMacValue(bfpb,bDrawAllowableSpillBitMap)) return err;
	//if (err = WriteMacValue(bfpb, fBitMapResMultiple)) return err;
	if (HaveAllowableSpillLayer())
	{
		if (err = WriteMacValue(bfpb,bSpillableAreaActive)) return err;
		if (err = WriteMacValue(bfpb,bSpillableAreaOpen)) return err;
	}
	if (err = WriteMacValue(bfpb, fExtendedMapBounds)) return err;
	if (err = WriteMacValue(bfpb, fUseExtendedBounds)) return err;

	return err;
}

/////////////////////////////////////////////////

OSErr TVectorMap::CheckAndPassOnMessage(TModelMessage *message)
{	
	char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		//long bitMapRes;
		OSErr err =0;
		//////////
		// code goes here, don't allow extended bounds with a map bounds layer
		err = message->GetParameterAsDouble("ExtendedLoLat",&val);
		if(!err)
		{	
			val *= 1000000; 
			if(val < this -> fExtendedMapBounds.loLat && !HaveMapBoundsLayer()) {
				this -> fUseExtendedBounds = true;
				this -> fExtendedMapBounds.loLat = val; 
				model -> NewDirtNotification();
			}
		}
		//////////
		err = message->GetParameterAsDouble("ExtendedHiLat",&val);
		if(!err)
		{	
			val *= 1000000; 
			if(val > this -> fExtendedMapBounds.hiLat && !HaveMapBoundsLayer()) {
				this -> fUseExtendedBounds = true;
				this -> fExtendedMapBounds.hiLat = val; 
				model -> NewDirtNotification();
			}
		}
		//////////
		err = message->GetParameterAsDouble("ExtendedLoLong",&val);
		if(!err)
		{	
		val *= 1000000; 
			if(val < this -> fExtendedMapBounds.loLong && !HaveMapBoundsLayer()) {
				this -> fUseExtendedBounds = true;
				this -> fExtendedMapBounds.loLong = val; 
				model -> NewDirtNotification();
			}
		}
		//////////
		err = message->GetParameterAsDouble("ExtendedLeftLong",&val);
		if(!err)
		{	
		val *= 1000000; 
			if(val < this -> fExtendedMapBounds.loLong && !HaveMapBoundsLayer()) {
				this -> fUseExtendedBounds = true;
				this -> fExtendedMapBounds.loLong = val; 
				model -> NewDirtNotification();
			}
		}
		//////////
		err = message->GetParameterAsDouble("ExtendedHiLong",&val);
		if(!err)
		{	
		val *= 1000000; 
			if(val > this -> fExtendedMapBounds.hiLong && !HaveMapBoundsLayer()) {
				this -> fUseExtendedBounds = true;
				this -> fExtendedMapBounds.hiLong = val; 
				model -> NewDirtNotification();
			}
		}
		//////////
		err = message->GetParameterAsDouble("ExtendedRightLong",&val);
		if(!err)
		{	
		val *= 1000000; 
			if(val > this -> fExtendedMapBounds.hiLong && !HaveMapBoundsLayer()) {
				this -> fUseExtendedBounds = true;
				this -> fExtendedMapBounds.hiLong = val; 
				model -> NewDirtNotification();
			}
		}
		//////////
		/*err = message->GetParameterAsLong("BitMapResolution",&bitMapRes);
		if(!err)
		{	
			if(bitMapRes > 0 && bitMapRes < 5) {
				this -> fBitMapResMultiple = bitMapRes;
				model -> NewDirtNotification();
			}
		}*/
	}
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TMap::CheckAndPassOnMessage(message);
}



/**************************************************************************************************/
OSErr AddVectorMapDialog()
{
	char 		path[256], nameStr [256];
	OSErr		err = noErr;
	long 		n;
	Point 		where = CenteredDialogUpLeft(M38b);
	TVectorMap 	*map = nil;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;

#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
			   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return USERCANCEL;
	strcpy(path, reply.fullPath);

	//strcpy(tempStr,path);
	//SplitPathFile(tempStr,shortFileName);
	strcpy (nameStr, "VectorMap: ");
	strcat (nameStr, path);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38b,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	#else
		strcpy(path, reply.fName);
	#endif

	strcpy (nameStr, "VectorMap: ");
	strcat (nameStr, (char*) reply.fName);	// should use short name here?
#endif	
	map = new TVectorMap (nameStr, voidWorldRect);
	if (!map)
		{ TechError("AddVectorMapDialog()", "new TVectorMap()", 0); return -1; }

	if (err = map->InitMap(path)) { delete map; return err; }

	if (err = model->AddMap(map, 0))
		{ map->Dispose(); delete map; return -1; }
	else {
		model->NewDirtNotification();
	}

	return err;

}

OSErr TVectorMap::ReplaceMap()
{
	char 		path[256], nameStr [256];
	OSErr		err = noErr;
	Point 		where = CenteredDialogUpLeft(M38b);
	TVectorMap 	*map = nil;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;
	Boolean isESI = false;

#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
			   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return USERCANCEL;
	strcpy(path, reply.fullPath);
	strcpy (nameStr, "VectorMap: ");
	strcat (nameStr, path);
#else
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, M38b,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	#else
		strcpy(path, reply.fName);
	#endif
#endif
	if (!IsVectorMap (path, &isESI))
	{
		printError("New map must be of the same type.");
		return USERCANCEL;	// to return to the dialog
	}
	if (isESI)
	{
		// code goes here, test out the ESI stuff here
		LongRect	LayerLBounds;
		long bmWidth, bmHeight;
		Rect bitMapRect;
		ImportESIData(path);
		if(this->HaveESIMapLayer())
		{
			WorldRect esiMapBounds,wRect;
			this->esiMapLayer -> GetLayerScope (&LayerLBounds, true);
			esiMapBounds.hiLat = LayerLBounds.top;
			esiMapBounds.loLat = LayerLBounds.bottom;
			esiMapBounds.loLong = LayerLBounds.left;
			esiMapBounds.hiLong = LayerLBounds.right;
			wRect = UnionWRect(esiMapBounds,fMapBounds);
			err = LandBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
			if (err) {printError("Unable to create bitmap in TVectorMap::ReplaceMap"); return err;}
			MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
			if(!err && this->HaveESIMapLayer())
			fESIBitmap = GetBlackAndWhiteBitmap(DrawESIMapLayer,this,wRect,bitMapRect,&err);
			return err;
		}

	}
#if !TARGET_API_MAC_CARBON
	strcpy (nameStr, "VectorMap: ");
	strcat (nameStr, (char*) reply.fName);
#endif	
	map = new TVectorMap (nameStr, voidWorldRect);
	if (!map)
		{ TechError("ReplaceMap()", "new TVectorMap()", 0); return -1; }

	if (err = map->InitMap(path)) { delete map; return err; }

	if (err = model->AddMap(map, 0))
		{ map->Dispose(); delete map; return -1; } 
	else 
	{
		// put movers on the new map and activate
		TMover *thisMover = nil;
		Boolean	timeFileChanged = false;
		long k, d = this -> moverList -> GetItemCount ();
		for (k = 0; k < d; k++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (err = AddMoverToMap(map, timeFileChanged, thisMover)) return err; 
			thisMover->SetMoverMap(map);	
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		if (err = model->DropMap(this)) return err;
		model->NewDirtNotification();
	}

	return err;
	
}

#ifdef MAC
void DrawBitmapImage(short colorIndex, BitMap * bitmap,Rect m)
{
	RGBColor	saveColor;
	
	GetForeColor (&saveColor);		// save original forecolor
	RGBForeColor(&colors[colorIndex]); // Use same color as the IBM !!!
	DrawBlackAndWhiteBitMap (bitmap, m);
	RGBForeColor(&saveColor);
}
#else
void DrawDIBImage(short colorIndex, HDIB *hBitmapPtr,Rect m)
{
	HDIB hDib = *hBitmapPtr; // NOTE: the MAC wanted to use the address
	RGBColor	saveColor;
	if(hDib)
	{
		RECT Rdst,Rsrc;
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(hDib);
		if (lpDIBHdr)
		{
			Rsrc.left = Rsrc.top = 0;
			Rsrc.right = lpDIBHdr->biWidth;
			Rsrc.bottom = lpDIBHdr->biHeight;
			MakeWindowsRect(&m,&Rdst);
			GetForeColor (&saveColor);		
			RGBForeColor(&colors[colorIndex]);
			PaintDIB(currentHDC,&Rdst,hDib,&Rsrc,0);
			RGBForeColor(&saveColor);
		}
		GlobalUnlock(hDib);
	}
}
void DrawBitmapImage(short colorIndex, HBITMAP *hBitmapPtr,Rect m)
{
	HBITMAP hBitmap = *hBitmapPtr; // NOTE: the MAC wanted to use the address
	RGBColor	saveColor;
	if(hBitmap)
	{
		RECT Rdst,Rsrc;
		BITMAP b;
		GetObject(hBitmap,sizeof(BITMAP),(LPSTR)&b);
		
		Rsrc.left = Rsrc.top = 0;
		Rsrc.right = b.bmWidth;
		Rsrc.bottom = b.bmHeight;
		MakeWindowsRect(&m,&Rdst);
		GetForeColor (&saveColor);		
		RGBForeColor(&colors[colorIndex]);
		PaintBitmap(currentHDC,&Rdst,hBitmap,&Rsrc,0);
		RGBForeColor(&saveColor);
	}
}
#endif


void TVectorMap::Draw(Rect r, WorldRect view)
{
	LongRect	updateLRect;
	LongRect	mapLongRect;
	Rect		m;
	Boolean  onQuickDrawPlane;
	RgnHandle saveClip=0, newClip=0;
	
	Boolean drawDiagnosticOnBottom  = ControlKeyDown();
	
	//WorldRect ourBounds =  this -> GetMapBounds();
	WorldRect ourBounds = fMapBounds;  // don't want extended bounds for Bitmaps
	
	mapLongRect.left = SameDifferenceX(ourBounds.loLong);
	mapLongRect.top = (r.bottom + r.top) - SameDifferenceY(ourBounds.hiLat);
	mapLongRect.right = SameDifferenceX(ourBounds.hiLong);
	mapLongRect.bottom = (r.bottom + r.top) - SameDifferenceY(ourBounds.loLat);
	
	onQuickDrawPlane = IntersectToQuickDrawPlane(mapLongRect,&m);
	
	// erase non-rectangular map
	if(this -> HaveMapBoundsLayer() && onQuickDrawPlane)
	{
#ifdef MAC
		saveClip = NewRgn();//
		GetClip(saveClip);//
		newClip = NewRgn();
		OpenRgn();
#endif
		TVectorMap* tvMap = (TVectorMap*)this; // typecast 
		CMap *theMap = (CMap*)tvMap;
		CMapLayer* mapLayer = tvMap->mapBoundsLayer;
#ifdef MAC
		DrawCMapLayerNoFill(theMap,mapLayer,false);	
		CloseRgn(newClip);
		EraseRgn(newClip);		
		SetClip(saveClip);//
		DisposeRgn(saveClip);//
		DisposeRgn(newClip);
#else
		DrawCMapLayerNoFill(theMap,mapLayer,true);	// erase polygons
#endif
	}
	else
	{
		/////////////////////////////////////////////////
		// JLM 6/10/99 maps must erase their rectangles in case a lower priority map drew in our rectangle
		EraseRect(&m); 
		/////////////////////////////////////////////////
	}

	
	if(this ->HaveAllowableSpillLayer() && this -> bDrawAllowableSpillBitMap && onQuickDrawPlane)
		DrawDIBImage(LIGHTBLUE,&fAllowableSpillBitmap,m);
	
	if(this -> bDrawLandBitMap && drawDiagnosticOnBottom && onQuickDrawPlane)
		DrawDIBImage(DARKGREEN,&fLandWaterBitmap,m);
	
	//////
	thisMapLayer -> GetLayerScope (&updateLRect, true);
	map -> ViewAllMap (false);
	gRect = r;
	gDrawBitmapInBlackAndWhite = false;	// this case is drawing land to the screen
	thisMapLayer -> DrawLayer (&updateLRect, kScreenMode);
	gDrawBitmapInBlackAndWhite = true;
	PenNormal();
	//MyFrameRect(&m); done in base class
	////////

	if(this ->HaveESIMapLayer() && this -> bDrawESIBitMap && onQuickDrawPlane)
	{
		//////
		esiMapLayer -> GetLayerScope (&updateLRect, true);
		map -> ViewAllMap (false);
		gRect = r;
		gDrawBitmapInBlackAndWhite = false;	// this case is drawing land to the screen
		esiMapLayer -> DrawLayer (&updateLRect, kScreenMode);
		gDrawBitmapInBlackAndWhite = true;
		PenNormal();
		////////
		//DrawDIBImage(PURPLE,&fESIBitmap,m);
		//DrawDIBImage(LIGHTGRAY,&fESIBitmap,m);
//#ifdef MAC
		//DrawBitmapImage (PURPLE, (BitMap*) (*(fESIBitmap -> portPixMap)), m); // actually copies to current GWorld, not screen
//#else 
		//DrawBitmapImage(PURPLE,&fESIBitmap,m); // NOTE: color is ignored since it is a color bitmap
//#endif
	}
	
	if(this -> bDrawLandBitMap && !drawDiagnosticOnBottom && onQuickDrawPlane)
		DrawDIBImage(DARKGREEN,&fLandWaterBitmap,m);

	// Draw alternative map bounds if they exist ...
	if(this -> HaveMapBoundsLayer() && onQuickDrawPlane)
	{
		TVectorMap* tvMap = (TVectorMap*)this; // typecast 
		CMap *theMap = (CMap*)tvMap;
		CMapLayer* mapLayer = tvMap->mapBoundsLayer;
		DrawCMapLayerNoFill(theMap,mapLayer,false);
	}
	
	TMap::Draw(r, view); // this draws the movers
}

void GetESICodeStr(long esiCode,char *esiCodeStr)
{
	//char esiCodeStr[64];
	esiCodeStr[0]=0;
	switch(esiCode)
	{
		case 1:	strcpy(esiCodeStr,"1A/1B"); break;
		case 2:	strcpy(esiCodeStr,"2A/2B"); break;
		case 3:	strcpy(esiCodeStr,"3A/3B"); break;
		case 4:	strcpy(esiCodeStr,"3C/4"); break;
		case 5:	strcpy(esiCodeStr,"5"); break;
		case 6:	strcpy(esiCodeStr,"6A"); break;
		case 7:	strcpy(esiCodeStr,"6B"); break;
		case 8:	strcpy(esiCodeStr,"7"); break;
		case 9:	strcpy(esiCodeStr,"8A"); break;
		case 10:	strcpy(esiCodeStr,"8B"); break;
		case 11:	strcpy(esiCodeStr,"8C/8D/8E/8F"); break;
		case 12:	strcpy(esiCodeStr,"9A/9B/9C"); break;
		case 13:	strcpy(esiCodeStr,"10A"); break;
		case 14:	strcpy(esiCodeStr,"10B/10E"); break;
		case 15:	strcpy(esiCodeStr,"10C"); break;
		case 16:	strcpy(esiCodeStr,"10D"); break;
	}
	return;
}

void TVectorMap::DrawESILegend(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	Rect		rgbrect;
	char 		codestr[64],text[30];
	long 		i,numLevels;
	double 	value;
	
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	TextFont(kFontIDGeneva); TextSize(6);
#endif

	if (EmptyRect(&fLegendRect)||!RectInRect2(&fLegendRect,&r))	// don't let legend go out of view
	{
		fLegendRect.top = r.top;
		fLegendRect.left = r.left;
		fLegendRect.bottom = r.top + 180;	// reset after contour levels drawn
		fLegendRect.right = r.left + 90;	// reset if values go beyond default width
	}
	rgbrect = fLegendRect;
	EraseRect(&rgbrect);
	
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10;
	y = rgbrect.top + dY / 2;
	MyMoveTo(x - stringwidth("ESI rank") / 2, y + dY);
	drawstring("ESI rank");
	v = rgbrect.top+30;
	//v = rgbrect.top+60;
	h = rgbrect.left;
	//for (i=0;i<numLevels;i++)
	for (i=0;i<16;i++)
	{
	
		MySetRect(&rgbrect,h+4,v-9,h+14,v+1);

		RGBForeColor(&esicolors[i+1]);
		PaintRect(&rgbrect);
		MyFrameRect(&rgbrect);
	
		MyMoveTo(h+20,v+.5);
	
		RGBForeColor(&colors[BLACK]);
		
		GetESICodeStr(i+1,codestr);

		drawstring(codestr);
		if (stringwidth(codestr)>widestNum) widestNum = stringwidth(codestr);
		v = v+9;
	}
	fLegendRect.bottom = v+3;
	if (fLegendRect.right<h+20+widestNum+4) fLegendRect.right = h+20+widestNum+4;
	else if (fLegendRect.right>fLegendRect.left+80 && h+20+widestNum+4<=fLegendRect.left+80)
		fLegendRect.right = fLegendRect.left+80;	// may want to redraw to recenter the header
 	MyFrameRect(&fLegendRect);
	return;
}

#ifdef MAC
Boolean IsBlackPixel(WorldPoint p,WorldRect mapBounds,BitMap bm)
#else
Boolean IsBlackPixel(WorldPoint p,WorldRect mapBounds,HDIB bm)
#endif
{	// forces the point to the closest point in the bounds
	// or if bitmap does not exist
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point pt;
	Boolean isBlack = false;
	
#ifdef MAC
	bounds = bm.bounds;
	rowBytes = bm.rowBytes;
	baseAddr = bm.baseAddr;
#else //IBM
	if(bm)
	{
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(bm);
		baseAddr = (char*) FindDIBBits((LPSTR)lpDIBHdr);
		#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
		rowBytes = WIDTHBYTES(lpDIBHdr->biBitCount * lpDIBHdr->biWidth);
		MySetRect(&bounds,0,0,lpDIBHdr->biWidth,lpDIBHdr->biHeight);
	}
#endif
	
	if(baseAddr)
	{
		// find the point in question in the bitmap
		// determine the pixel in the bitmap we need to look at
		// think of the bitmap as an array of pixels 
		pt = WorldToScreenPoint(p, mapBounds,bounds);
		
		// force into range and then check the bitmap
		pt.h = _max(pt.h,bounds.left); 
		pt.h = _min(pt.h,bounds.right-1); 
		pt.v = _max(pt.v,bounds.top); 
		pt.v = _min(pt.v,bounds.bottom-1); 

		#ifdef IBM
			/// on the IBM,  the rows of pixels are "upsidedown"
			offset = rowBytes*(long)(bounds.bottom -1 - pt.v);
			/// on the IBM ,for a mono map, 1 is background color,
			isBlack = !BitTst(baseAddr + offset, pt.h);
		#else
			offset = (rowBytes*(long)pt.v);
			isBlack = BitTst(baseAddr + offset, pt.h);
		#endif
	}
	
#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif

	return isBlack;
}

/////////////////////////////////////////////////

/**************************************************************************************************/
Boolean TVectorMap::InMap(WorldPoint p)
{
	WorldRect ourBounds = this -> GetMapBounds(); 
	if (!WPointInWRect(p.pLong, p.pLat, &ourBounds)) return false;
	if (HaveMapBoundsLayer())
	{
		return IsBlackPixel(p,ourBounds,fMapBoundsBitmap);
	}
	else
		return TMap::InMap(p);
}

/////////////////////////////////////////////////
Boolean TVectorMap::OnLand(WorldPoint p)
{
	Boolean onLand = false;
	
	if(!this->InMap(p)) return false; // off map is not on land
	
	if (HaveESIMapLayer())	// esi is the shoreline
	{	
		//PixMapHandle myPixMap = fESIBitmap -> portPixMap;
		//CTabHandle myColorTable = (*myPixMap) -> pmTable;
		//short size =(* myColorTable)->ctSize;
		//CSpecArray
		//ColorSpec myCSpec = (*myColorTable) -> ctTable[0];
		if (IsBlackPixel(p,fMapBounds,fESIBitmap) )
		//if (IsBlackPixel(p,fMapBounds,*(BitMap*) (*(fESIBitmap -> portPixMap))) )
		{
			return true;
		}

	}
	
	if(fUseExtendedBounds)
	{// if the point is outside the real bounds of the map say it is water
		if(!WPointInWRect(p.pLong,p.pLat,&fMapBounds))
			return false; // in water
	}
	onLand = IsBlackPixel(p,fMapBounds,fLandWaterBitmap);
	
	return onLand;
}


WorldPoint3D	TVectorMap::MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed)
{
	// check every pixel along the line it makes on the land water bitmap
	// forces the point to the closest point in the bounds
	// or if bitmap does not exist
	#ifdef MAC
		BitMap bm = fLandWaterBitmap;
	#else
		HDIB bm = fLandWaterBitmap;
	#endif
	
	// this code is similar to IsBlackPixel
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point fromPt,toPt;
	Boolean isBlack = false;
	
#ifdef MAC
	bounds = bm.bounds;
	rowBytes = bm.rowBytes;
	baseAddr = bm.baseAddr;
#else //IBM
	if(bm)
	{
		LPBITMAPINFOHEADER lpDIBHdr  = (LPBITMAPINFOHEADER)GlobalLock(bm);
		baseAddr = (char*) FindDIBBits((LPSTR)lpDIBHdr);
		#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
		rowBytes = WIDTHBYTES(lpDIBHdr->biBitCount * lpDIBHdr->biWidth);
		MySetRect(&bounds,0,0,lpDIBHdr->biWidth,lpDIBHdr->biHeight);
	}
#endif
	
	if(baseAddr)
	{
		// find the point in question in the bitmap
		// determine the pixel in the bitmap we need to look at
		// think of the bitmap as an array of pixels 
		long maxChange,esiCode;
		WorldPoint3D wp = {0,0,0.};
		//WorldRect mapBounds = this->GetMapBounds();
		WorldRect mapBounds = fMapBounds;  // don't want extended bounds for Bitmaps, 3/10/00
		
		fromPt = WorldToScreenPoint(fromWPt.p,mapBounds,bounds);
		toPt = WorldToScreenPoint(toWPt.p,mapBounds,bounds);
		
		// check the bitmap for each pixel when in range
		// so find the number of pixels change hori and vertically
		maxChange = _max(abs(toPt.h - fromPt.h),abs(toPt.v - fromPt.v));

		if(maxChange == 0) {
			// it's the same pixel, there is nothing to do
		}
		//else if (maxChange == 1) {
			// no need to check at the new pixel
			// the calling code will check to see it is land or water
		//}
		else { // maxChange > 1
			long i;
			double fraction;
			Point pt, prevPt = fromPt;
			WorldPoint3D prevWPt = fromWPt;
			// note: there is no need to check the first or final pixel, so i only has to go to maxChange-1 
			for(i = 0; i < maxChange; i++) 
			{
				fraction = (i+1)/(double)(maxChange); 
				wp.p.pLat = (1-fraction)*fromWPt.p.pLat + fraction*toWPt.p.pLat;
				wp.p.pLong = (1-fraction)*fromWPt.p.pLong + fraction*toWPt.p.pLong;
				wp.z = (1-fraction)*fromWPt.z + fraction*toWPt.z;
				
				pt = WorldToScreenPoint(wp.p,mapBounds,bounds);

				// only check this pixel if it is in range
				// otherwise it is not on our map, hence not our problem
				// so assume it is water and OK

				if(bounds.left <= pt.h && pt.h < bounds.right
					&& bounds.top <= pt.v && pt.v < bounds.bottom)
				{
		
					#ifdef IBM
						/// on the IBM, the rows of pixels are "upsidedown"
						offset = rowBytes*(long)(bounds.bottom - 1 - pt.v);
						/// on the IBM, for a mono map, 1 is background color,
						isBlack = !BitTst(baseAddr + offset, pt.h);
					#else
						offset = (rowBytes*(long)pt.v);
						isBlack = BitTst(baseAddr + offset, pt.h);
					#endif
					
					if (isBlack) { 
						return wp;// this is a land point, restrict the movement to this point
					}
					if (abs(pt.h - prevPt.h) == 1 && abs(pt.v - prevPt.v) == 1)
					{	// figure out which pixel was crossed

						float xRatio = (float)(wp.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds),
							  yRatio = (float)(wp.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float ptL = bounds.left + RectWidth(bounds) * xRatio;
						float ptB = bounds.bottom - RectHeight(bounds) * yRatio;
						xRatio = (float)(prevWPt.p.pLong - mapBounds.loLong) / (float)WRectWidth(mapBounds);
						yRatio = (float)(prevWPt.p.pLat - mapBounds.loLat) / (float)WRectHeight(mapBounds);
						float prevPtL = bounds.left + RectWidth(bounds) * xRatio;
						float prevPtB = bounds.bottom - RectHeight(bounds) * yRatio;
						float dir = (ptB - prevPtB)/(ptL - prevPtL);
						float testv; 
							
						testv = dir*(_max(prevPt.h,pt.h) - prevPtL) + prevPtB;

						if (prevPt.v < pt.v)
						{
							if (ceil(testv) == pt.v)
								prevPt.h = pt.h;
							else if (floor(testv) == pt.v)
								prevPt.v = pt.v;
						}
						else if (prevPt.v > pt.v)
						{
							if (ceil(testv) == prevPt.v)
								prevPt.v = pt.v;
							else if (floor(testv) == prevPt.v)
								prevPt.h = pt.h;
						}
						
						if(bounds.left <= prevPt.h && prevPt.h < bounds.right
							&& bounds.top <= prevPt.v && prevPt.v < bounds.bottom)
						{
				
							#ifdef IBM
								/// on the IBM, the rows of pixels are "upsidedown"
								offset = rowBytes*(long)(bounds.bottom -1 - prevPt.v);
								/// on the IBM, for a mono map, 1 is background color,
								isBlack = !BitTst(baseAddr + offset, prevPt.h);
							#else
								offset = (rowBytes*(long)prevPt.v);
								isBlack = BitTst(baseAddr + offset, prevPt.h);
							#endif
							
							if (isBlack) { 
								// this is a land point, restrict the movement to this point
								//return ScreenToWorldPoint(prevPt, bounds, mapBounds);
								wp.p = ScreenToWorldPoint(prevPt, bounds, mapBounds);
								return wp;	// bnas are only 2D so don't need to worry about fixing z

							}
						}
					}
				}
				prevPt = pt;
				prevWPt = wp;
			}
		}
	}

done:

#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif

	return toWPt;
}

/////////////////////////////////////////////////

Boolean TVectorMap::IsAllowableSpillPoint(WorldPoint p)
{
	if(!this->InMap(p)) return false;// not on this map
	if(this->OnLand(p)) return false;// on land
	// check allowable spill polygon
	//if(this->HaveAllowableSpillLayer())	// look for noaa.ver in Diagnostic Mode
	//if(this->HaveAllowableSpillLayer() && !(gNoaaVersion &&  model->GetModelMode() == ADVANCEDMODE))
	if(this->HaveAllowableSpillLayer() && bSpillableAreaActive)
	{ // make sure it is on the allowable spill bitmap
		//WorldRect mapBounds = this->GetMapBounds();
		WorldRect mapBounds = fMapBounds;	// don't want to use extended bounds for bitmaps
		Boolean isAllowable = IsBlackPixel(p,mapBounds,fAllowableSpillBitmap);
		return isAllowable;
	}
	return true; // a water point and no spillable polygon layer is defined
}

/////////////////////////////////////////////////

/**************************************************************************************************/
Boolean IsVectorMap (char *path, Boolean *isESI)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line, numPts, numScanned;
	char	strLine [256];
	char	firstPartOfFile [256], segNumStr[32], codeStr[32];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (CountSetInString (strLine, "\042") == 4 &&		// four quote marks
			 CountSetInString (strLine, ",") == 2)				// two commas
			bIsValid = true;
		else
			{bIsValid = false; return bIsValid;}
	}
	else
		{bIsValid = false; return bIsValid;}
	// 11/10/03 need to distinguish between map files and esi files
	*isESI = false;
#ifdef MPW	// code goes here for code warrior
	StringSubstitute(strLine, ' ', '');
#endif
	StringSubstitute(strLine, ',', ' ');
	numScanned=sscanf(strLine, "%s %s %ld",
				  segNumStr, codeStr, &numPts) ;
	if (numScanned!=3)	
		{ /*err = -1; TechError("IsVectorMap()", "sscanf() == 3", 0);*/ return bIsValid; }	// shouldn't happen, just let it go as bna
	if (numPts<0) *isESI = true;	// in theory map files always have closed polygons, esi's are open segments
	
	return bIsValid;
}
/**************************************************************************************************/
long TVectorMap::GetListLength()
{
	long i, n, count = 1;
	TMover *mover;

	if (bOpen) {
		count++;// name

 		count++; // REFLOATHALFLIFE

		//if(this->HaveAllowableSpillLayer()) count++; // show spillable area box
		if(this->HaveAllowableSpillLayer()) 
		{
			count++;	//  name
			if (bSpillableAreaOpen)
			{
				count+=2; // spillable area active, show spillable area box
			}
		}
		count ++;// bitmap-visible-box

		if (this->HaveESIMapLayer())	count++; // show ESI layer
		if (this->HaveESIMapLayer())	count++; // show ESI legend

		if (bMoversOpen)
			for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count += mover->GetListLength();
			}

	}

	return count;
}
/**************************************************************************************************/
ListItem TVectorMap::GetNthListItem(long n, short indent, short *style, char *text)
{
	long i, m, count;
	TMover *mover;
	ListItem item = { this, 0, indent, 0 };
		
	if (n == 0) {
		item.index = I_VMAPNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text, className);
		
		return item;
	}
	n -= 1;

	if (n == 0) {
		//item = TMap::GetNthListItem(I_REFLOATHALFLIFE, indent, style, text);
		item.index = I_VREFLOATHALFLIFE; // override the index
		item.indent = indent; // override the indent
			sprintf(text, "Refloat half life: %g hr",fRefloatHalfLifeInHrs);
		return item;
	}
	n -= 1;

	if(this ->HaveAllowableSpillLayer())
	{
		if (n == 0) {
			item.indent++;
			item.index = I_VSPILLBITMAP;
			item.bullet = bSpillableAreaOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Spillable Area");
			
			return item;
		}
		n -= 1;
		if (bSpillableAreaOpen)
		{
			if (n == 0) {
				//item.indent++;
				item.indent+=2;
				item.index = I_VSPILLBITMAPACTIVE;
				item.bullet = bSpillableAreaActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Spillable Area Active");
				
				return item;
			}
			n -= 1;
			if (n == 0) {
				//item.indent++;
				item.indent+=2;
				item.index = I_VDRAWALLOWABLESPILLBITMAP;
				item.bullet = bDrawAllowableSpillBitMap ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Show Allowable Spill Area");
				
				return item;
			}
			n -= 1;
		}
	}
	if (n == 0) {
		item.indent++;
		item.index = I_VDRAWLANDBITMAP;
		item.bullet = bDrawLandBitMap ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
		strcpy(text, "Show Land / Water Map");
		
		return item;
	}
	n -= 1;
	
	if (this->HaveESIMapLayer())
	{
		if (n == 0) {
			item.indent++;
			item.index = I_VDRAWESIBITMAP;
			item.bullet = bDrawESIBitMap ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show ESI Shoreline");
			
			return item;
		}
		n -= 1;
		if (n == 0) {
			item.indent++;
			item.index = I_VSHOWESILEGEND;
			item.bullet = bShowLegend ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show ESI Legend");
			
			return item;
		}
		n -= 1;
	}
	
	
	if (bOpen) {
		indent++;
		if (n == 0) {
			item.index = I_VMOVERS;
			item.indent = indent;
			item.bullet = bMoversOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Movers");
			
			return item;
		}
		
		n -= 1;
		
		if (bMoversOpen)
			for (i = 0, m = moverList->GetItemCount() ; i < m ; i++) {
				moverList->GetListItem((Ptr)&mover, i);
				count = mover->GetListLength();
				if (count > n)
					return mover->GetNthListItem(n, indent + 1, style, text);
				
				n -= count;
			}
	}
	
	item.owner = 0;
	
	return item;
}
/**************************************************************************************************/
Boolean TVectorMap::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet) {
		switch (item.index) {
			case I_VMAPNAME: bOpen = !bOpen; return TRUE;
			case I_VDRAWLANDBITMAP:
				bDrawLandBitMap = !bDrawLandBitMap;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_VSPILLBITMAP: bSpillableAreaOpen = !bSpillableAreaOpen; return TRUE;
			case I_VDRAWALLOWABLESPILLBITMAP:
				bDrawAllowableSpillBitMap = !bDrawAllowableSpillBitMap;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_VSPILLBITMAPACTIVE:
				bSpillableAreaActive = !bSpillableAreaActive;
				/*model->NewDirtNotification(DIRTY_MAPDRAWINGRECT);*/ return TRUE;
			case I_VDRAWESIBITMAP:
				bDrawESIBitMap = !bDrawESIBitMap;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_VSHOWESILEGEND:
				bShowLegend = !bShowLegend;
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_VMOVERS: bMoversOpen = !bMoversOpen; return TRUE;
		}
	}
	
	if (doubleClick) {
		if (this -> FunctionEnabled(item, SETTINGSBUTTON)) {
			item.index = I_MAPNAME;
			this -> SettingsItem(item);
			return TRUE;
		}

		if (item.index == I_VMOVERS)
		{
			item.owner -> AddItem (item);
			return TRUE;
		}
		
	}

	return false;
}
/**************************************************************************************************/
Boolean TVectorMap::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	
	switch (item.index) {
		case I_VMAPNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE; 
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!model->mapList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (model->mapList->GetItemCount() - 1);
					}
			}
			break;
		case I_VMOVERS:
			switch (buttonID) {
				case ADDBUTTON: return TRUE;
				case SETTINGSBUTTON: return FALSE;
				case DELETEBUTTON: return FALSE;
			}
			break;
		case I_VDRAWLANDBITMAP:
		case I_VDRAWALLOWABLESPILLBITMAP:
		case I_VSPILLBITMAP:
		case I_VSPILLBITMAPACTIVE:
		case I_VDRAWESIBITMAP:
		case I_VSHOWESILEGEND:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
		case I_VREFLOATHALFLIFE:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return FALSE;
}

/////////////////////////////////////////////////

void	TVectorMap::GetSuggestedFileName(char* suggestedFileName,char* extension)
{		
	char s[256] ="";
	char copyOfExtension[32]="";
	long nameLen;
	long idLen = strlen("Vector Map:");
	this -> GetClassName(s);

	if (extension && extension[0])
	{
		if (extension[0]!='.')
			strcat(copyOfExtension,".");
		
		strncat(copyOfExtension,extension,4);
		copyOfExtension[4]=0;
	}

	// Remove leading Vector Map: tag
	if (!strncmp(s,"Vector Map:",idLen))
	{
		strcpy(s,s+idLen);
	}
	
	nameLen=strlen(s);	


	// Chop off extension
	if(nameLen >= 4) 
	{
		long i;
		for(i=1; i<=4; i++)
		{
			if(s[nameLen-i] == '.')
			{	
				s[nameLen-i]=0;
				break; // only take off last extension
			}
		}
	}
		
	// Get rid of leading and trailing spaces
	strtrimcpy(s, s);				

	// Ensure file name not over 27 characters
	s[27]=0;
	strtrimcpy(s, s);				
		
	if (s[0])
		strcpy(suggestedFileName, s);
	else
		strcpy(suggestedFileName, "untitled");
	
	strcat(suggestedFileName, copyOfExtension);		
}

OSErr TVectorMap::ExportAsBNAFileForGnomeAnalyst(char* path)
{
	// because this is for GNOME analyst, we leave off the spillable area layer and mapbounds layer
	char buffer[512];
	long numChar;
	long objectIndex;
	OSType thisObjectType;
	CMyList * thisObjectList =0;
	long objectCount = 0;
	ObjectRecHdl thisObjectHdl = 0;
	long numPts;
	Boolean isWaterPoly;
	LongPoint** thisPointsHdl =0;
	LongPoint longPt;
	long i;
	PolyObjectHdl thisObjectHdlAsPolyObjectHdl =0;
	char theTypeChar;
	BFPB bfpb;
	OSErr err = 0;
	double lat,lng;
	char latStr[32],lngStr[32];
	char *p;
	
	//if(this -> thisMapLayer)
	if(this -> thisMapLayer)
	{
		thisObjectList = this -> thisMapLayer -> GetLayerObjectList ();
		//thisObjectList = this -> mapBoundsLayer -> GetLayerObjectList ();
		objectCount = thisObjectList -> GetItemCount ();
	}
	
//"2068","1",50
//-119.918419,34.075127
//-119.888138,34.074093

	if (objectCount <= 0)
	{
		printError("This map has no polygons to export.");
		return -1;
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	for (objectIndex = 0; objectIndex < objectCount; objectIndex++)
	{
		thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, objectIndex);
		
//void DrawObject (ObjectRecHdl theObjectHdl, CMap *Map, LongRect *UpdateLRectPtr,
//				 DrawSpecRecPtr drawSettings)

		GetObjectType (thisObjectHdl, &thisObjectType);
		if(thisObjectType != kPolyType)
		{
			printError("Unexpected thisObjectType found.");
			err = -1; goto done;
		}
			
//			DrawMapPoly (Map, (PolyObjectHdl) thisObjectHdl, drawSettings);
//void DrawMapPoly (CMap* theMap, PolyObjectHdl MapPolyHdl, DrawSpecRecPtr drawSettings)
			
		thisObjectHdlAsPolyObjectHdl = (PolyObjectHdl) thisObjectHdl;
		numPts = (**thisObjectHdlAsPolyObjectHdl).pointCount;
		thisPointsHdl = (LongPoint**) (**thisObjectHdlAsPolyObjectHdl).objectDataHdl;
		isWaterPoly = (**thisObjectHdlAsPolyObjectHdl).bWaterPoly;
			
		if(numPts > 0 && thisPointsHdl)
		{
			LongPoint firstPt,lastPt;
			long numPtsToWrite = numPts;
			
			if(isWaterPoly) 
				theTypeChar = '2';
			else 
				theTypeChar = '1';
				
			firstPt = INDEXH(thisPointsHdl,0);
			lastPt = INDEXH(thisPointsHdl,numPts-1);
			
			#define ADDCLOSINGPOINT TRUE // specifies we want to add a closing point equal to the first point in the BNA file
			if(ADDCLOSINGPOINT)
			{
				if(firstPt.h != lastPt.h || firstPt.v != lastPt.v)
				{	// add a closing point equal to the first point
					numPtsToWrite += 1;
				}
			}
			
			numChar = sprintf(buffer,"\"%ld\",\"%c\",%ld%s",objectIndex,theTypeChar,numPtsToWrite,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, numChar)) goto done;

			for(i = 0; i< numPtsToWrite;i++)
			{
				if(i < numPts)
					longPt = INDEXH(thisPointsHdl,i);
				else
					longPt = firstPt; // used when numPtsToWrite > numPts (i.e to close off the polygon)
					
				lat = longPt.h/1000000.;
				lng = longPt.v/1000000.;
				StringWithoutTrailingZeros(latStr,lat,6);
				StringWithoutTrailingZeros(lngStr,lng,6);
				// some programs may expect the decimal place to be there
				// so make sure we have the decimal point and at least one decimal place
				p = strchr(latStr,'.');
				if(!p) strcat(latStr,".0");
				p = strchr(lngStr,'.');
				if(!p) strcat(lngStr,".0");
				/////
				strcpy(buffer,latStr);
				strcat(buffer,",");
				strcat(buffer,lngStr);
				strcat(buffer,NEWLINESTRING);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	}

done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}

OSErr TVectorMap::ExportAsBNAFile(char* path)
{
	// include the spillable area layer and mapbounds layer
	char buffer[512];
	long numChar;
	long objectIndex;
	OSType thisObjectType;
	CMyList * thisObjectList =0;
	long objectCount = 0;
	ObjectRecHdl thisObjectHdl = 0;
	long numPts;
	Boolean isWaterPoly;
	LongPoint** thisPointsHdl =0;
	LongPoint longPt;
	long i, j, numLayers = 3;	// probably won't add any more but leave the option
	PolyObjectHdl thisObjectHdlAsPolyObjectHdl =0;
	char theTypeChar;
	BFPB bfpb;
	OSErr err = 0;
	double lat,lng;
	char latStr[32],lngStr[32];
	char *p;
	
	/*if (objectCount <= 0)
	{
		printError("This map has no polygons to export.");
		return -1;
	}*/
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	for (j=0; j<numLayers; j++)
	{
		if (j==0)
		{
			if(this -> mapBoundsLayer)
			{
				thisObjectList = this -> mapBoundsLayer -> GetLayerObjectList ();
				objectCount = thisObjectList -> GetItemCount ();
			}
		}
		else if (j==1)
		{
			if(this -> thisMapLayer)
			{
				thisObjectList = this -> thisMapLayer -> GetLayerObjectList ();
				objectCount = thisObjectList -> GetItemCount ();
			}
		}
		else if (j==2)
		{
			if(this -> allowableSpillLayer)
			{
				thisObjectList = this -> allowableSpillLayer -> GetLayerObjectList ();
				objectCount = thisObjectList -> GetItemCount ();
			}
		}	
		if (objectCount <= 0) continue;
		for (objectIndex = 0; objectIndex < objectCount; objectIndex++)
		{
			thisObjectList -> GetListItem ((Ptr) &thisObjectHdl, objectIndex);
			
			GetObjectType (thisObjectHdl, &thisObjectType);
			if(thisObjectType != kPolyType)
			{
				printError("Unexpected thisObjectType found.");
				err = -1; goto done;
			}
							
			thisObjectHdlAsPolyObjectHdl = (PolyObjectHdl) thisObjectHdl;
			numPts = (**thisObjectHdlAsPolyObjectHdl).pointCount;
			thisPointsHdl = (LongPoint**) (**thisObjectHdlAsPolyObjectHdl).objectDataHdl;
			isWaterPoly = (**thisObjectHdlAsPolyObjectHdl).bWaterPoly;
				
			if(numPts > 0 && thisPointsHdl)
			{
				LongPoint firstPt,lastPt;
				long numPtsToWrite = numPts;
				
				if(isWaterPoly) 
					theTypeChar = '2';
				else 
					theTypeChar = '1';
					
				firstPt = INDEXH(thisPointsHdl,0);
				lastPt = INDEXH(thisPointsHdl,numPts-1);
				
				#define ADDCLOSINGPOINT TRUE // specifies we want to add a closing point equal to the first point in the BNA file
				if(ADDCLOSINGPOINT)
				{
					if(firstPt.h != lastPt.h || firstPt.v != lastPt.v)
					{	// add a closing point equal to the first point
						numPtsToWrite += 1;
					}
				}
				if (j==0)
					numChar = sprintf(buffer,"\"Map Bounds\",\"%c\",%ld%s",theTypeChar,numPtsToWrite,NEWLINESTRING);
				else if (j==1)
					numChar = sprintf(buffer,"\"%ld\",\"%c\",%ld%s",objectIndex,theTypeChar,numPtsToWrite,NEWLINESTRING);
				else if (j==2)
					numChar = sprintf(buffer,"\"SpillableArea\",\"%c\",%ld%s",theTypeChar,numPtsToWrite,NEWLINESTRING);
				
				if (err = WriteMacValue(&bfpb, buffer, numChar)) goto done;
	
				for(i = 0; i< numPtsToWrite;i++)
				{
					if(i < numPts)
						longPt = INDEXH(thisPointsHdl,i);
					else
						longPt = firstPt; // used when numPtsToWrite > numPts (i.e to close off the polygon)
						
					lat = longPt.h/1000000.;
					lng = longPt.v/1000000.;
					StringWithoutTrailingZeros(latStr,lat,6);
					StringWithoutTrailingZeros(lngStr,lng,6);
					// some programs may expect the decimal place to be there
					// so make sure we have the decimal point and at least one decimal place
					p = strchr(latStr,'.');
					if(!p) strcat(latStr,".0");
					p = strchr(lngStr,'.');
					if(!p) strcat(lngStr,".0");
					/////
					strcpy(buffer,latStr);
					strcat(buffer,",");
					strcat(buffer,lngStr);
					strcat(buffer,NEWLINESTRING);
					if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
				}
			}
		}
	}
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}


/////////////////////////////////////////////////
// This subroutine is for CDOG map bounds 
/**************************************************************************************************/
OSErr TVectorMap::ChangeMapBox(WorldPoint p, WorldPoint p2)
{
	char nameStr [256];
	OSErr	err = noErr;
	TVectorMap 	*map = nil;
	LongRect	LayerLBounds;
	WorldRect mapBounds, newBounds;
	Rect bitMapRect;
	long bmWidth, bmHeight;
	
	strcpy (nameStr, "VectorMap: ");
	strcat (nameStr, "Map Box");
	
	// check that the bounds are ok
	if (p.pLat<=p2.pLat || p.pLong>=p2.pLong) {printError("Bounds are not consistent"); return -1;}
	map = new TVectorMap (nameStr, voidWorldRect);
	if (!map)
	{ TechError("ChangeMapBox()", "new TVectorMap()", 0); return -1; }
	
	if (err = map->InitMap()) return err;
	
	newBounds.hiLat = p.pLat;
	newBounds.loLong = p.pLong;
	newBounds.loLat = p2.pLat;
	newBounds.hiLong = p2.pLong;
	if (err = map->SelectMapBox(newBounds)) return err;	// this sets up the map bounds layer
	
	if(map->HaveMapBoundsLayer()){
		(map->mapBoundsLayer) -> GetLayerScope (&LayerLBounds, true);
		mapBounds.hiLat = LayerLBounds.top;
		mapBounds.loLat = LayerLBounds.bottom;
		mapBounds.loLong = LayerLBounds.left;
		mapBounds.hiLong = LayerLBounds.right;
	}
	
	map->SetMapBounds (mapBounds);
	map->SetExtendedBounds (mapBounds);
	err = LandBitMapWidthHeight(mapBounds,&bmWidth,&bmHeight);
	if (err) {printError("Unable to make bitmap in TVectorMap::ChangeMapBox");return err;}
	MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
	
	if(!err && map->HaveMapBoundsLayer())
		(map->fMapBoundsBitmap) = GetBlackAndWhiteBitmap(DrawMapBoundsLayer,map,mapBounds,bitMapRect,&err);
	
	switch(err) 
	{
		case noErr: break;
		case memFullErr: printError("Out of memory in TVectorMap::ChangeMapBox"); break;
		default: TechError("TVectorMap::ChangeMapBox", "GetBlackAndWhiteBitmap", err); break;
	}
	
	if (err) return err;
	
	if (err = model->AddMap(map, 0))
	{ map->Dispose(); delete map; return -1; } 
	else 
	{
		// put movers on the new map and activate
		TMover *thisMover = nil;
		Boolean	timeFileChanged = false;
		long k, d = this -> moverList -> GetItemCount ();
		for (k = 0; k < d; k++)
		{
			this -> moverList -> GetListItem ((Ptr) &thisMover, 0); // will always want the first item in the list
			if (err = AddMoverToMap(map, timeFileChanged, thisMover)) return err; 
			thisMover->SetMoverMap(map);	
			if (err = this->DropMover(thisMover)) return err; // gets rid of first mover, moves rest up
		}
		if (err = model->DropMap(this)) return err;
		model->NewDirtNotification();
	}
	
	return err;
	
}
/////////////////////////////////////////////////
OSErr TVectorMap::SelectMapBox (WorldRect mapBoxBounds)
{
	OSErr			err = noErr;
	Size			NewHSize;
	long			PointCount = 5, PointIndex, AddPointCount, ObjectCount;
	long			colorIndex;
	Boolean			bClosed, GroupFlag, PointsAddedFlag;
	LongRect		ObjectLRect;
	LongPoint		MatrixPt, MatrixPoints[5],**thisPointsHdl;
	PolyObjectHdl	thisPolyHdl = nil;
	ObjectIDRec		FirstObjectID, LastObjectID;
	CMapLayer * theLayer = nil;
	char ObjectName [128] = "Map Bounds";
	
	// call dialog to select points
	if (mapBoxBounds.loLong == 0 && mapBoxBounds.hiLong == 0) err = MapBoxDialog(&mapBoxBounds);
	if (err) return err;
	
	MatrixPoints[0].h = mapBoxBounds.loLong;
	MatrixPoints[0].v = mapBoxBounds.loLat;
	MatrixPoints[1].h = mapBoxBounds.hiLong;
	MatrixPoints[1].v = mapBoxBounds.loLat;
	MatrixPoints[2].h = mapBoxBounds.hiLong;
	MatrixPoints[2].v = mapBoxBounds.hiLat;
	MatrixPoints[3].h = mapBoxBounds.loLong;
	MatrixPoints[3].v = mapBoxBounds.hiLat;
	MatrixPoints[4].h = mapBoxBounds.loLong;
	MatrixPoints[4].v = mapBoxBounds.loLat;
	
	theLayer = mapBoundsLayer;
	
	bClosed = true;
	
	// set region rect to empty to start with
	SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);
	
	GroupFlag = false;			// clear flag indicating grouping
	PointsAddedFlag  = false;	// will be set to true if a point is appended
	AddPointCount = 0;			// number of points added
	
	// now read in the points for the next region in the file
	// should be able to reduce this code...
	for (PointIndex = 1; PointIndex <= PointCount && err == 0; ++PointIndex)
	{			
		MatrixPt = MatrixPoints[PointIndex-1];
		if (MatrixPt.h < kWorldLeft)   MatrixPt.h = kWorldLeft;
		if (MatrixPt.h > kWorldRight)  MatrixPt.h = kWorldRight;
		if (MatrixPt.v > kWorldTop)    MatrixPt.v = kWorldTop;
		if (MatrixPt.v < kWorldBottom) MatrixPt.v = kWorldBottom;
		
		if (bClosed && PointIndex > 1  && PointIndex == PointCount)
		{
			thisPolyHdl = (PolyObjectHdl) theLayer -> AddNewObject (kPolyType, &ObjectLRect, false);
			if (thisPolyHdl != nil)
			{
				SetObjectLabel ((ObjectRecHdl) thisPolyHdl, ObjectName);
				
				// set the object label point at the center of polygon
				GetObjectCenter ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
				SetOLabelLPoint ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
				
				// set the polygon points handle field
				SetPolyPointsHdl (thisPolyHdl, thisPointsHdl);
				SetPolyPointCount (thisPolyHdl, AddPointCount);
				
				if (!GroupFlag) GroupFlag = true;
				
				colorIndex = kLandColorInd;
				
				SetObjectBColor ((ObjectRecHdl) thisPolyHdl, colorIndex);
				SetObjectColor ((ObjectRecHdl) thisPolyHdl, colorIndex);// added JLM 7/1/99
				
				// poke the filled flag into poly handle!!
				SetPolyClosed (thisPolyHdl, bClosed);
				
				// set the next region's rect to empty
				SetLRect (&ObjectLRect, kWorldRight, kWorldBottom, kWorldLeft, kWorldTop);
				
				PointsAddedFlag = false;
				AddPointCount = 0;
			}
			else
			{
				err = memFullErr;
				goto done;
			}
		}
		else
		{
			if (AddPointCount == 0)
			{
				thisPointsHdl = (LongPoint**) _NewHandleClear (0);
				if (thisPointsHdl == nil)
				{
					err = memFullErr;
					goto done;
				}
			}
			
			NewHSize = _GetHandleSize ((Handle) thisPointsHdl) + sizeof (LongPoint);
			_SetHandleSize ((Handle) thisPointsHdl, NewHSize);
			if (_MemError ())
				err = memFullErr;
			
			if (err)
				goto done;
			else
			{
				++AddPointCount;
				PointsAddedFlag = true;
				
				// store the new point in the points handle
				(*thisPointsHdl) [AddPointCount - 1] = MatrixPt;
				
				// update region boundary rectangle
				if (MatrixPt.h < ObjectLRect.left)   ObjectLRect.left   = MatrixPt.h;
				if (MatrixPt.v > ObjectLRect.top)    ObjectLRect.top    = MatrixPt.v;
				if (MatrixPt.h > ObjectLRect.right)  ObjectLRect.right  = MatrixPt.h;
				if (MatrixPt.v < ObjectLRect.bottom) ObjectLRect.bottom = MatrixPt.v;
			}
		}
	}
	
	/////////////////////////////////////////////////
	
	if (PointsAddedFlag)
	{
		thisPolyHdl = (PolyObjectHdl) theLayer -> AddNewObject (kPolyType, &ObjectLRect, false);
		if (thisPolyHdl != nil)
		{
			SetObjectLabel ((ObjectRecHdl) thisPolyHdl, ObjectName);
			
			// set the object label point at the center of polygon
			GetObjectCenter ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
			SetOLabelLPoint ((ObjectRecHdl) thisPolyHdl, &MatrixPt);
			
			// set the polygon points handle field
			SetPolyPointsHdl (thisPolyHdl, thisPointsHdl);
			SetPolyPointCount (thisPolyHdl, AddPointCount);
			
			colorIndex = kLandColorInd;			// default land color index
			
			SetObjectBColor ((ObjectRecHdl) thisPolyHdl, colorIndex);
			SetObjectColor ((ObjectRecHdl) thisPolyHdl, colorIndex);// added JLM 7/1/99
			SetPolyClosed (thisPolyHdl, bClosed);
		}	
		else
		{
			err = memFullErr;
			goto done;
		}
	}
	
done:
	
	if (err == memFullErr)
		printError ("Out of application memory! Try increasing application's memory partition.");
	
	return err;
}

