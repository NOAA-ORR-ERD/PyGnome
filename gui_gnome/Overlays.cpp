#include "cross.h"
#include "OUtils.h"
#include "Overlays.h"
#include "ObjectUtilsPD.h"

/////////////////////////////////////////////////
/////////////////////////////////////////////////


/////////////////////////////////////////////////
/////////////////////////////////////////////////

TOverlay::TOverlay ()
{
	// constructor 
	bShowOverlay = true;
	fFilePath[0] = 0;
	fColor = colors[RED]; // set the default overlay color to red
	fBounds = voidWorldRect;
}



void TOverlay::Dispose()
{

}

		
void TOverlay::GetFileName(char *name)
{
	char *p;
	name[0] = 0;
	p = strrchr(this->fFilePath,DIRDELIMITER);
	if(p){
		strcpy(name,p+1); // the short file name
	}
}

void TOverlay::SetClassNameToFileName()  // for command files
{
	char *p;
	char fileName[kMaxNameLen];
	this->GetFileName(fileName);
	this->SetClassName(fileName);
}


OSErr TOverlay::ReadFromFile(char *path)
{
	// do nothing.. this could be a virtual function
	return noErr;
}



void TOverlay::Draw (Rect r, WorldRect view)
{	
	// do nothing.. this could be a virtual function
}


long TOverlay::GetListLength ()
{
	return 1;
}


ListItem TOverlay::GetNthListItem (long n, short indent, short *style, char *text)
{
	ListItem item = { this, 0, indent, 0 };
	char *p;
	if (n == 0) {
		item.index = I_OVERLAYNAME;
		item.bullet = bShowOverlay ? BULLET_FILLEDBOX :BULLET_EMPTYBOX ;
		p = strrchr(this->fFilePath,DIRDELIMITER);
		if(p){
			sprintf(text,"File: %s",p+1); // the short file name
		}
		else {
			text[0] = 0;
		}
		
		return item;
	}
	n -= 1;
	
	/////////////
	item.owner = 0;
	return item;
}


Boolean TOverlay::ListClick (ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet && item.index == I_OVERLAYNAME){ 
		bShowOverlay = !bShowOverlay;
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT); 
		return TRUE; 
	}
	if (doubleClick && item.index == I_OVERLAYNAME)  {
		SettingsItem(item);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT); 
		return true;
	}
		
	return false;
}


Boolean TOverlay::FunctionEnabled (ListItem item, short buttonID)
{
	long i;
	
	switch (item.index) {
		case I_OVERLAYNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE; 
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!model->fOverlayList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (model->fOverlayList->GetItemCount() - 1);
					}
			}
			break;
	}
	return false;
}

OSErr TOverlay::UpItem(ListItem item)
{	
	long i;
	OSErr err = 0;
	
	if (item.index == I_OVERLAYNAME)
		if (model->fOverlayList->IsItemInList((Ptr)&item.owner, &i))
			if (i > 0) {
				if (err = model->fOverlayList->SwapItems(i, i - 1))
					{ TechError("TOverlay::UpItem()", "model->fOverlayList->SwapItems()", err); return err; }
				SelectListItem(item);
				UpdateListLength(true);
				InvalidateMapImage();
				InvalMapDrawingRect();
			}
	
	return 0;
}

OSErr TOverlay::DownItem(ListItem item)
{
	long i;
	OSErr err = 0;
	
	if (item.index == I_OVERLAYNAME)
		if (model->fOverlayList->IsItemInList((Ptr)&item.owner, &i))
			if (i < (model->fOverlayList->GetItemCount() - 1)) {
				if (err = model->fOverlayList->SwapItems(i, i + 1))
					{ TechError("TOverlay::UpItem()", "model->fOverlayList->SwapItems()", err); return err; }
				SelectListItem(item);
				UpdateListLength(true);
				InvalidateMapImage();
				InvalMapDrawingRect();
			}
	
	return 0;
}
OSErr TOverlay::SettingsItem (ListItem item)
{
	fColor =  MyPickColor(fColor,mapWindow);
	model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT); 
	return noErr;
}

OSErr TOverlay::DeleteItem (ListItem item)
{
	if (item.index == I_OVERLAYNAME)
		return model->DropOverlay(this);
	return noErr;
}

OSErr TOverlay::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		Boolean val;
		OSErr err = 0;
		
		err = message->GetParameterAsBoolean("bShowOverlay",&val);
		if(!err)
		{	
			this->bShowOverlay  = val; 
			model->NewDirtNotification();// tell model about dirt
		}
	}
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TClassID::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////
////////////////////////////////////////////////


/////////////////////////////////////////////////
/////////////////////////////////////////////////

Boolean IsShapeFile(char * path) 
{
	return false;
}
/////////////////////////////
// JLM 5/15/10


Boolean IsNesdisPolygonFile(char * path) // JLM 5/15/10
{
	// read the top of the file to see if it is a Nesdis file
	// first line should be "Polygon"
	// next line should be "0 0"
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long lineNum;
	char	s [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	char *key;
	double lat,lng;
	int ptNum,polyNum,flag;
	int numScanned;
	char extraStuff[256];
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	if(err) return false;
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	lineNum = 0;
	NthLineInTextOptimized (firstPartOfFile, lineNum++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	key = "Polygon";
	if (strncmpnocase (s, key, strlen(key))) 
		return false; //did not match the key

	// next line should be polygon number and a flag
	NthLineInTextOptimized (firstPartOfFile, lineNum++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	numScanned = sscanf(s,"%d %d %s",&polyNum,&flag,extraStuff);
	if(numScanned != 2) { // there should be no extraStuff	
		return false; 
	}

	// next line should be point number  lng  lat
	NthLineInTextOptimized (firstPartOfFile, lineNum++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	numScanned = sscanf(s,"%d %lf %lf",&ptNum,&lng,&lat);
	if(numScanned == 3 && -180 <= lat && lat <= 180 && -180 <= lng && lat <= 180) {
		return true; // paassed our test
	}


	return false; // failed the test
}



TNesdisOverlay::TNesdisOverlay ()
{
	// constructor
	memset(&fNesdisPoints,0,sizeof(fNesdisPoints));

#ifdef MAC
	memset(&fBitmap,0,sizeof(fBitmap)); 
#else
	fBitmap = 0;
#endif
}



void TNesdisOverlay::Dispose()
{
	// dispose of the allocated memory
	this->DisposeNesdisPoints();
#ifdef MAC
	DisposeBlackAndWhiteBitMap (&fBitmap);
#else
	if(fBitmap) DestroyDIB(fBitmap);
	fBitmap = 0;
#endif
	TOverlay::Dispose ();
}



void TNesdisOverlay::DisposeNesdisPoints(void)
{
	if(fNesdisPoints.pts) free(fNesdisPoints.pts); fNesdisPoints.pts = 0;
	memset(&fNesdisPoints,0,sizeof(fNesdisPoints));
	fFilePath[0] = 0; // make sure the file path is cleared 
}


OSErr TNesdisOverlay::AllocateNesdisPoints(long numToAllocate)
{
	DisposeNesdisPoints();	// in case they are allocated
	if(numToAllocate > 0) { 
		fNesdisPoints.pts = (NesdisPoint*)calloc(numToAllocate,sizeof(*fNesdisPoints.pts));
		if(fNesdisPoints.pts) {
			fNesdisPoints.numAllocated = numToAllocate;
			return noErr;
		}
		// else
		return memFullErr;
	}
	return noErr;
}


// code goes here, read directly from shape file
// generalize to overlay generic shape file or bna
OSErr TNesdisOverlay::ReadFromBNAFile(char * path)
{
	OSErr			ReadErrCode = noErr, err = noErr;
	long			PointCount, PointIndex, AddPointCount, ObjectCount;
	long			colorIndex;
	long			line = -1;
	Boolean			bClosed, GroupFlag, PointsAddedFlag;
	CHARH			f =0;
	char			strLine [255], ObjectName [128], ObjectName2 [128], KeyStr [255],
					*LineStartPtr;
	long numLinesText;
	int thisPolyNum = -1; 
	int thisPolyFlag = -1; 
	int thisPolyInteriorRingFlag = -1;
	int i, numScanned;
	double lat,lng;
	NesdisPoint np;

	memset(&np,0,sizeof(np));
	
	err = 0;

	err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f);
	if (err)
	{
		printError ("Error opening BNA map file for reading!");
		return err;
	}

	_HLock((Handle)f); // JLM 8/4/99

	numLinesText = NumLinesInText (*f);

	err = AllocateNesdisPoints(numLinesText);
	if(err) goto done;
	
	while (err == 0)
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
				break;
			else 
				continue;
		}
		///////////}

		///////////////{
		// JLM 12/24/98
		/*if(!strcmpnocase(ObjectName,"SpillableArea"))// special polygon defining spillable area
		{	// do anything differently here
		}
		else if(!strcmpnocase(ObjectName,"Map Bounds"))// special polygon defining the map bounds
		else */
			
		///////////}
					
		if (PointCount < 0)		// negative number indicates unfilled polygon
		{
			PointCount = labs (PointCount);
			bClosed = false;
		}
		else
			bClosed = true;
	
		if(!strcmpnocase(ObjectName2,"2"))// lake polygon
			thisPolyInteriorRingFlag = 1;
		else
		{
			thisPolyInteriorRingFlag = 0;
			thisPolyNum++;
		}
		GroupFlag = false;			// clear flag indicating grouping
		PointsAddedFlag  = false;	// will be set to true if a point is appended
		AddPointCount = 0;			// number of points added
		
		// now read in the points for the next region in the file
		for (PointIndex = 1; PointIndex <= PointCount && err == 0; ++PointIndex)
		{
			if (++line >= numLinesText)
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
			
			if (PointIndex == 1)	// save the key string for comparing later
				strcpy (KeyStr, LineStartPtr);
		
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine,"%lf %lf",&lng,&lat);
		if(numScanned == 2 && -90 <= lat && lat <= 90 && -360 <= lng && lng <= 360) {
			// it is a point
			// Note: we expect the first point to be the same as the last point 
			// Note: we could verify the point is in sequence here
			if(fNesdisPoints.numFilledIn < fNesdisPoints.numAllocated){ // safety check
				np.lat = lat;
				np.lng = lng;
				np.polyNum = thisPolyNum;
				np.flag = thisPolyFlag;
				np.interiorRingFlag = thisPolyInteriorRingFlag;
				fNesdisPoints.pts[fNesdisPoints.numFilledIn++] = np;	
			}
		}
	}
	//thisPolyNum++;
	}

	strcpy(fFilePath,path); // record the path now that we have been successful
	this->SetClassNameToFileName(); // for the wizard

	// figure out the bounds
	fBounds = voidWorldRect;
	for(i = 0; i < fNesdisPoints.numFilledIn; i++) {
		WorldPoint wp;
		np = fNesdisPoints.pts[i];
		wp.pLat = 1000000.0*np.lat;
		wp.pLong = 1000000.0*np.lng;
		AddWPointToWRect(wp.pLat, wp.pLong, &fBounds);
	}
	this->MakeBitmap(); //Amy wants to be able to auto fill a spray can with points, perhaps we could use a bitmap to do this for her, 6/6/10
	
	// make sure the overlay is shown in the current view
	if(!EqualWRects(voidWorldRect,fBounds))
		ChangeCurrentView(UnionWRect(settings.currentView,fBounds), TRUE, TRUE);

done:
	// cause a screen refresh
	InvalMapDrawingRect();
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	return err;
}

OSErr TNesdisOverlay::ReadFromFile(char * path)
{
	OSErr err = 0;
	long polygonNum, zeroOrOne;
	long numLinesInFile = 0;
	long numPoints = 0;
	CHARH f = 0;
	char *fileContents;
	long lineNum;
	char s[256];
	char extraStuff[256];
	int numScanned;
	double lat,lng;
	int ptNum,polyNum,flag;
	// for keeping track where we are in the file
	int thisPolyNum = -1; 
	int thisPolyFlag = -1; 
	int thisPolyInteriorRingFlag = -1;
	NesdisPoint np;
	int i;

	memset(&np,0,sizeof(np));


	SetWatchCursor();

	// these files are not huge, read entire file into memory
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TNesdisOverlay::ReadFromFile()", "ReadFileContents()", err);
		goto done;
	}
	_HLock((Handle)f); 
	fileContents = *f;

	// then create a list of points we can draw from
	numLinesInFile = NumLinesInText(fileContents);
	err = AllocateNesdisPoints(numLinesInFile);
	if(err) goto done;


	lineNum = 0;

	// first line is "Polygon"
	NthLineInTextOptimized(fileContents, lineNum++, s, 256);

	for(;lineNum < numLinesInFile;) {
		// read the next line
		NthLineInTextOptimized (fileContents, lineNum++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);

		// determine if it is :
		//		the end of the file
		//		new polygon
		//		point in the current polygon

		if (strncmpnocase (s, "END", strlen("END")) == 0) 
			break; //end of the file, we are done

		if(s[0] == 0) // blank line (possibly due to hand editing)
			continue;


		// is it a header line or a point?
		// first see if it is point
		numScanned = sscanf(s,"%d %lf %lf",&ptNum,&lng,&lat);
		if(numScanned == 3 && -180 <= lat && lat <= 180 && -180 <= lng && lat <= 180) {
			// it is a point
			// Note: we expect the first point to be the same as the last point 
			// Note: we could verify the point is in sequence here
			if(fNesdisPoints.numFilledIn < fNesdisPoints.numAllocated){ // safety check
				np.lat = lat;
				np.lng = lng;
				np.polyNum = thisPolyNum;
				np.flag = thisPolyFlag;
				np.interiorRingFlag = thisPolyInteriorRingFlag;
				fNesdisPoints.pts[fNesdisPoints.numFilledIn++] = np;	
			}
		}
		else { // see if it is the next polygon line
			// Note: header line of each polygon is "<polygonNum> <zeroOrOne>"
			if(strstr(s,"InteriorRing")){ // start of an interior ring
				thisPolyInteriorRingFlag++; // increase the flag so we know it is another piece	
			}
			else {
				numScanned = sscanf(s,"%d %d %s",&polyNum,&flag,extraStuff);
				if(numScanned == 2) { // there should be no extraStuff
					// new polygon
					// it looks like sometimes the same polygon number as the last one with the flag set to 1, not sure what that signifies
					thisPolyNum = polyNum;
					thisPolyFlag = flag;
					thisPolyInteriorRingFlag = 0; // it is an outer ring
				}
				else {
						char msg[512];
						err = true; 
						sprintf(msg,"Unable to parse line %d:%s%s",lineNum,NEWLINESTRING,s);
						printError(msg);
						goto done; // we didn't recognize the lines
				}
			}
		}
	}

	strcpy(fFilePath,path); // record the path now that we have been successful
	this->SetClassNameToFileName(); // for the wizard

	// figure out the bounds
	fBounds = voidWorldRect;
	for(i = 0; i < fNesdisPoints.numFilledIn; i++) {
		WorldPoint wp;
		np = fNesdisPoints.pts[i];
		wp.pLat = 1000000.0*np.lat;
		wp.pLong = 1000000.0*np.lng;
		AddWPointToWRect(wp.pLat, wp.pLong, &fBounds);
	}
	this->MakeBitmap(); //Amy wants to be able to auto fill a spray can with points, perhaps we could use a bitmap to do this for her, 6/6/10
	
	// make sure the overlay is shown in the current view
	if(!EqualWRects(voidWorldRect,fBounds))
		ChangeCurrentView(UnionWRect(settings.currentView,fBounds), TRUE, TRUE);

done:
	// cause a screen refresh
	InvalMapDrawingRect();
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	return err;
}




/////////////////////////////////////////////////

//settings.currentView,MapDrawingRect()
void DrawNesdisOutline(NesdisPointsInfo nesdisPoints, Rect r, WorldRect view)
{
	long i;
	long prevPolyNum;
	Boolean useMoveTo;
	NesdisPoint lastPt,thisPt;
	Point p;
	WorldPoint wp;
	WorldPointF wpf;

	memset(&lastPt,0,sizeof(lastPt));
	for(i = 0; i < nesdisPoints.numFilledIn; i++) {
		thisPt = nesdisPoints.pts[i];
		useMoveTo =  false ; // set to use lineto
		if(i == 0) 
			useMoveTo = true; // first polygon
		else if(lastPt.polyNum != thisPt.polyNum)
			useMoveTo = true; // new polygon
		else if(lastPt.flag != thisPt.flag)
			useMoveTo = true; // new sub-polygon
		else if(lastPt.interiorRingFlag != thisPt.interiorRingFlag)
			useMoveTo = true; // new interior ring

		wp.pLat = 1000000.0*thisPt.lat;
		wp.pLong = 1000000.0*thisPt.lng;

		p = WorldToScreenPointRound(wp,view,r); 

		if(useMoveTo)MyMoveTo(p.h,p.v);
		else MyLineTo(p.h,p.v);

		lastPt = thisPt;
	}
}
#ifdef MAC
void DrawNesdisPolygonsForBitmap(NesdisPointsInfo nesdisPoints, Rect r, WorldRect view)
{
	long i,j;
	long prevPolyNum;
	Boolean useMoveTo;
	NesdisPoint lastPt,thisPt;
	Point p;
	WorldPoint wp;
	WorldPointF wpf;
	Boolean drawPolygonAsHole;

	int startIndex,endIndex;
	#ifdef IBM
	POINT P;
	POINT *pts = (POINT*)calloc(nesdisPoints.numFilledIn,sizeof(POINT));
	#else
	Point *pts = (Point*)calloc(nesdisPoints.numFilledIn,sizeof(Point));
	PolyHandle poly;
	#endif
	if(pts == NULL) return;


	memset(&lastPt,0,sizeof(lastPt));
	startIndex = endIndex = 0;
	for(i = 0; i < nesdisPoints.numFilledIn; i++) {
		thisPt = nesdisPoints.pts[i];
		useMoveTo =  false ; // set to use lineto
		if(i == 0) {
			// first polygon
		}
		else if(lastPt.polyNum != thisPt.polyNum){
			// new polygon
			endIndex = i-1;
		}
		else if(lastPt.flag != thisPt.flag) {
			// new sub-polygon
			endIndex = i-1;
		}
		else if(lastPt.interiorRingFlag != thisPt.interiorRingFlag) {
			// new interior ring
			endIndex = i-1;
		}
		else if (i == nesdisPoints.numFilledIn-1) {
			// last point
			endIndex = i;
		}

		wp.pLat = 1000000.0*thisPt.lat;
		wp.pLong = 1000000.0*thisPt.lng;

		p = WorldToScreenPointRound(wp,view,r); 
#ifdef IBM
		MakeWindowsPoint(&p,&P);
		pts[i] = P; // fill in array as we go
#else
		pts[i] = p; // fill in array as we go
#endif
		if(endIndex > startIndex) { // we are ready to draw the polygon
			drawPolygonAsHole = (nesdisPoints.pts[startIndex].interiorRingFlag > 0);
			if(drawPolygonAsHole) {
				// draw as white, so that it erases
				RGBForeColor(&colors[WHITE]);
			}
			else {
				// draw as black
				RGBForeColor(&colors[BLACK]);
			}
#ifdef MAC
		poly = OpenPoly();
		for (j=startIndex;j<endIndex;j++)
		{
			MyMoveTo(pts[j].h,pts[j].v);
			MyLineTo(pts[j+1].h,pts[j+1].v);
		}
		MyLineTo(pts[startIndex].h,pts[startIndex].v);
		
		ClosePoly();
	
		//if(fillTriangle)
			//if (drawPolygonAsHole) ErasePoly(poly);

			/*else*/ PaintPoly(poly);	// want bitmap filled in
		
		FramePoly(poly);
		
		KillPoly(poly);
#else
			Polygon(currentHDC,pts+startIndex,1 + endIndex-startIndex);
#endif
			startIndex = endIndex+1;
			endIndex = startIndex;
		}

		lastPt = thisPt;
	}
	RGBForeColor(&colors[BLACK]);

	free(pts); pts = NULL;
}
#else
void DrawNesdisPolygonsForBitmap(NesdisPointsInfo nesdisPoints, Rect r, WorldRect view)
{
	long i;
	long prevPolyNum;
	Boolean useMoveTo;
	NesdisPoint lastPt,thisPt;
	Point p;
	WorldPoint wp;
	WorldPointF wpf;
	Boolean drawPolygonAsHole;

	int startIndex,endIndex;
	POINT P;
	POINT *pts = (POINT*)calloc(nesdisPoints.numFilledIn,sizeof(POINT));
	if(pts == NULL) return;


	memset(&lastPt,0,sizeof(lastPt));
	startIndex = endIndex = 0;
	for(i = 0; i < nesdisPoints.numFilledIn; i++) {
		thisPt = nesdisPoints.pts[i];
		useMoveTo =  false ; // set to use lineto
		if(i == 0) {
			// first polygon
		}
		else if(lastPt.polyNum != thisPt.polyNum){
			// new polygon
			endIndex = i-1;
		}
		else if(lastPt.flag != thisPt.flag) {
			// new sub-polygon
			endIndex = i-1;
		}
		else if(lastPt.interiorRingFlag != thisPt.interiorRingFlag) {
			// new interior ring
			endIndex = i-1;
		}
		else if (i == nesdisPoints.numFilledIn-1) {
			// last point
			endIndex = i;
		}

		wp.pLat = 1000000.0*thisPt.lat;
		wp.pLong = 1000000.0*thisPt.lng;

		p = WorldToScreenPointRound(wp,view,r); 
		MakeWindowsPoint(&p,&P);
		pts[i] = P; // fill in array as we go

		if(endIndex > startIndex) { // we are ready to draw the polygon
			drawPolygonAsHole = (nesdisPoints.pts[startIndex].interiorRingFlag > 0);
			if(drawPolygonAsHole) {
				// draw as white, so that it erases
				RGBForeColor(&colors[WHITE]);
			}
			else {
				// draw as black
				RGBForeColor(&colors[BLACK]);
			}
			Polygon(currentHDC,pts+startIndex,1 + endIndex-startIndex);
			startIndex = endIndex+1;
			endIndex = startIndex;
		}

		lastPt = thisPt;
	}

	free(pts); pts = NULL;
}
#endif
void TNesdisOverlay::Draw (Rect r, WorldRect view)
{

	if(fNesdisPoints.numFilledIn == 0)
		return; // nothing to draw

	if(!bShowOverlay) 
		return; //nothing to draw

//#ifdef IBM
	//if(fBitmap){	// need to check for fBitmap differently for Mac
		// code goes here... eventually remove this (i.e. no need to draw this bitmap)
		//if(ShiftKeyDown() && ControlKeyDown()) 
			//this->DrawBitmap(r,view); 
	//}
//#endif
	
	RGBForeColor(&fColor);
	PenSize(3,3);
	PenSize(1,1);
	DrawNesdisOutline(fNesdisPoints,r,view);
	RGBForeColor(&colors[BLACK]);
	PenNormal();
	


}

ListItem TNesdisOverlay::GetNthListItem (long n, short indent, short *style, char *text)
{
	ListItem item = { this, 0, indent, 0 };
	char *p;
	if (n == 0) {
		item.index = I_OVERLAYNAME;
		item.bullet = bShowOverlay ? BULLET_FILLEDBOX :BULLET_EMPTYBOX ;
		p = strrchr(this->fFilePath,DIRDELIMITER);
		if(p){
			sprintf(text,"NESDIS: %s",p+1); // the short file name
		}
		else {
			text[0] = 0;
		}
		
		return item;
	}
	n -= 1;
	
	/////////////
	item.owner = 0;
	return item;
}


/////////////////////////////////////////////////

void DrawFilledNesdisPolygons(void * object,WorldRect wRect,Rect r)
{
	TNesdisOverlay* nesdisOverlay = (TNesdisOverlay*)object; // typecast
	
	PenSize(1,1);
	DrawNesdisPolygonsForBitmap(nesdisOverlay->fNesdisPoints, r, wRect);
	PenNormal();
	return;
}


void NesdisCourseBitMapWidthHeight(WorldRect wRect, long *width, long* height)
{	// we can use the aspect ratio to get better fits
	// but make sure the width is a multiple of 32 !!!!
	WorldPoint center;
	long desiredNumBits =  30000000L;
	double fraction,latDist,lngDist;
	long w,h;
	
	center.pLat = (wRect.loLat + wRect.hiLat)/2; 
	center.pLong = (wRect.loLong + wRect.hiLong)/2; 
	
	latDist = fabs(LatToDistance(wRect.loLat - wRect.hiLat)); // in kilometers
	lngDist = fabs(LongToDistance(wRect.loLong - wRect.hiLong,center));// in kilometers
	
	if(lngDist > 0 ) fraction = latDist/lngDist;
	else fraction = 0.5; // this should not happen
	
	if(fraction < 0.01) fraction = 0.01; // minimum aspect ratios
	if(fraction < 0.99) fraction = 0.99;
	
	// fraction = latDist/lngDist = height/width;
	// height*width = 1 meg
	// substituting yields
	// (height/width)*width^2 = 1 meg 
	
	w = sqrt(desiredNumBits/fraction);
	w += 32 - (w%32); // bump it up to the next multiple of 32
	
	// JLM try to be better about preserving the aspect ration here
	// when the number of bits is small
	h = desiredNumBits/w;
	////
	{
		double pixelsPerPerKm = w/lngDist;
		// we want same # pixels h and vertically
		h = pixelsPerPerKm*latDist;
	}
	
	*width = w;
	*height = h; 
}


OSErr TNesdisOverlay::MakeBitmap(void)
{
	OSErr err = 0;
		
	{ // make the bitmap etc
		Rect bitMapRect;
		long bmWidth, bmHeight;
		WorldRect wRect = this -> fBounds;
		NesdisCourseBitMapWidthHeight(wRect,&bmWidth,&bmHeight);
		MySetRect (&bitMapRect, 0, 0, bmWidth, bmHeight);
		fBitmap = GetBlackAndWhiteBitmap(DrawFilledNesdisPolygons,this,wRect,bitMapRect,&err); 
		if(err) goto done;
	
	}
done:	
	if(err)
	{
#ifdef MAC
		DisposeBlackAndWhiteBitMap (&fBitmap);
#else
		if(fBitmap) DestroyDIB(fBitmap);
		fBitmap = 0;
#endif
	}
	return err;
}

void TNesdisOverlay::DrawBitmap(Rect r, WorldRect view)
{
	LongRect	mapLongRect;
	Rect m;
	Boolean  onQuickDrawPlane;
	WorldRect bounds = this -> fBounds;
	RgnHandle saveClip=0, newClip=0;
	
	mapLongRect.left = SameDifferenceX(bounds.loLong);
	mapLongRect.top = (r.bottom + r.top) - SameDifferenceY(bounds.hiLat);
	mapLongRect.right = SameDifferenceX(bounds.hiLong);
	mapLongRect.bottom = (r.bottom + r.top) - SameDifferenceY(bounds.loLat);
	onQuickDrawPlane = IntersectToQuickDrawPlane(mapLongRect,&m);

	if (onQuickDrawPlane)
		DrawDIBImage(LIGHTBLUE,&fBitmap,m);
}

typedef struct {
	WorldPoint wp;
	Boolean filledIn;
} WorldPointWithFlag;

#ifdef MAC
long GetNesdisLEs(WorldPoint *wpArray, long maxNumWorldPts , WorldRect wrBounds,BitMap bm)
#else
long GetNesdisLEs(WorldPoint *wpArray, long maxNumWorldPts , WorldRect wrBounds,HDIB bm)
#endif
{	// returns number of points filled in (up to maxNumWorldPts) 
	Rect bounds;
	char* baseAddr= 0;
	long rowBytes;
	long rowByte,bitNum,byteNumber,offset;
	Point pt;
	Boolean isBlack = false;
	long numFilledIn = 0;
	WorldPoint wp;
	long desiredNumBoxes = maxNumWorldPts;
	long boxSize;
	long numBoxesPerRow, numBoxesPerCol;
	long row, col;
	double numPixelsPerBox;
	WorldPointWithFlag *wpfArray = NULL;
	WorldPointWithFlag wpf;
	long numAllocated = 0;
	long i,index;
	
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
		// look through the bitmap for black pixels

		// Because we don't know how sparse the black pixels are
		// we will need to increase the number of boxes tillwe get approximately the number of pixels we are looking for.
		// 

		numPixelsPerBox = _max(1.0,sqrt( (bounds.right*bounds.bottom)/(double)desiredNumBoxes) );

		for(;;) { // forever

			// figure out the size of the box want to use per LE 
			numBoxesPerRow = (long)(bounds.right/numPixelsPerBox); // truncate
			numBoxesPerCol = (long)(bounds.bottom/numPixelsPerBox); // truncate

			if(wpfArray) free(wpfArray);
			numAllocated = numBoxesPerRow*numBoxesPerCol;
			wpfArray = (WorldPointWithFlag*)calloc(numAllocated,sizeof(*wpfArray));
			if(wpfArray == NULL) goto done; // memory error


			for(pt.h = bounds.left; pt.h < bounds.right; pt.h++) {
				for(pt.v = bounds.top; pt.v < bounds.bottom; pt.v++) {

					// figure out which box we are in

					row = _min((long)(pt.h/numPixelsPerBox),numBoxesPerRow-1);
					col = _min((long)(pt.v/numPixelsPerBox),numBoxesPerCol-1);
					index = (row*numBoxesPerCol) + col;
					if(wpfArray[index].filledIn)
						continue; // this box is already filled in 
						//(or we could go for the closest point to the center

					#ifdef IBM
						/// on the IBM,  the rows of pixels are "upsidedown"
						offset = rowBytes*(long)(bounds.bottom -1 - pt.v);
						/// on the IBM ,for a mono map, 1 is background color,
						isBlack = !BitTst(baseAddr + offset, pt.h);
					#else
						offset = (rowBytes*(long)pt.v);
						isBlack = BitTst(baseAddr + offset, pt.h);
					#endif
					
					if(isBlack) {
						wpfArray[index].filledIn = true;
						wpfArray[index].wp = ScreenToWorldPoint(pt,bounds,wrBounds);
					}
				}
			}

			if(numPixelsPerBox < 1.01) {
				break; // this is the best we can do
			}

			// now check to see how many of the boxes were filled in
			for(i = 0; i < numAllocated ; i++) {
				if(wpfArray[i].filledIn) {
					numFilledIn++;
				}
			}
			if(numFilledIn > 0.5 *maxNumWorldPts) {
				break; // close enough
			}
			// otherwise, try again with more boxes//
			// We'd expect that 1/2 the numPixelsPerBox would give us about 4 times as many hits
			if(numFilledIn == 0)
				break; // this should never happen (it means all pixels were white)

			numPixelsPerBox = numPixelsPerBox / sqrt(maxNumWorldPts/(double)numFilledIn);
			if(numPixelsPerBox < 1) 
				numPixelsPerBox = 1.0; // this is the best we can do

		}
		// now fill in the array to be returned
		numFilledIn = 0;
		if(wpfArray) {
			for(i = 0; i < numAllocated ; i++) {
				wpf = wpfArray[i];
				if(wpf.filledIn) {
					if(numFilledIn < maxNumWorldPts)
						wpArray[numFilledIn++] = wpf.wp;
				}
			}
		}
	}
	
done:

#ifdef IBM
	if(bm) GlobalUnlock(bm);
#endif

	if(wpfArray) free(wpfArray);
	return numFilledIn;
}


void TNesdisOverlay::AddSprayedLEsToLEList(void)
{
	long i,j;
	SprayDialogInfo	sprayDialogInfo;
	WorldPoint wp;
	long totalNumLEs = 0;
	char msg[256];
	OSErr err = 0;
	TSprayLEList *list=nil,*uncertaintyLEList = nil;
	//int bitmapWidth,bitmapHeight;
	#define kMaxNumLEs 2000
	WorldPoint wpArray[kMaxNumLEs];
	int numFilledIn = 0;
	char *p;
	
	
	WorldRect currentView = settings.currentView;  
	Rect currentMapDrawingRect = MapDrawingRect();

	numFilledIn = sizeof(wpArray); // just testing
	memset(wpArray,0,sizeof(wpArray)); 

	if(model->mapList->GetItemCount() == 0) {
		printNote("You need to have a map before generating LEs because LEs will only be placed in the water." );
		return;
	}
	

	SetWatchCursor();

	numFilledIn = GetNesdisLEs(wpArray,kMaxNumLEs,fBounds,fBitmap);

	// create the spray sets
	list = new TSprayLEList();
	if (!list) { OutOffMemoryAlert(); err= -1; goto done;}
	
	memset(&sprayDialogInfo,0,sizeof(sprayDialogInfo));

	sprayDialogInfo.numOfLEs = 0;
	sprayDialogInfo.pollutantType = OIL_CONSERVATIVE;
	sprayDialogInfo.totalMass = 0;
	sprayDialogInfo.massUnits = BARRELS;
	sprayDialogInfo.density = 1;
	sprayDialogInfo.overflightTime = model -> GetStartTime ();
	sprayDialogInfo.spillTime = model->GetStartTime(); 
	sprayDialogInfo.ageHours = 0; 
	sprayDialogInfo.whenAmountIsSpecified = ATOVERFLIGHTTIME;

	strcpy(sprayDialogInfo.spillName,"NESDIS ");
	p = strrchr(this->fFilePath,DIRDELIMITER);
	if(p){
		char nameWithoutExtension[kMaxNameLen];
		strcat(nameWithoutExtension,p+1); // the short file name
		p = strrchr(nameWithoutExtension,'.');
		if(p && strlen(p) <= 4) {
			*p = 0; // chop this file extension
		}
	}

	
	err = list->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;
		
	uncertaintyLEList = new TSprayLEList ();
	if (!uncertaintyLEList) { OutOffMemoryAlert(); err= -1; goto done; }
	uncertaintyLEList->fLeType = UNCERTAINTY_LE;
	uncertaintyLEList->fOwnersUniqueID = list->GetUniqueID();
	
	err = uncertaintyLEList->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;
	
	// put an LE for each black pixel in the coarse bitmap
	totalNumLEs = 0;
	for (i = 0; i < numFilledIn; i++)
	{
		wp = wpArray[i];
		if(model->IsWaterPoint(wp)) {
			err = list ->AddSprayPoint(wp);
			if(err) goto done;
			totalNumLEs++;
		}
		else  {
			Boolean notWater = model->IsWaterPoint(wp);
		}
	}

	sprayDialogInfo.numOfLEs = totalNumLEs;
	sprayDialogInfo.totalMass = totalNumLEs;
	
	model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
	// since the model might be in negative time, in which case the LE's would be deleted and
	// readded from the files, which do not contain these new LE sets
	// Note: we want to do this before Initialize() is called
	// because it uses GetModelTime() -- JLM 10/18/00

	err = list->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;
	err = uncertaintyLEList->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;

	model->NewDirtNotification();

	err = model->AddLEList(list, 0);
	if(err) goto done;
	err = model->AddLEList(uncertaintyLEList, 0);
	if(err) goto done;

	// bring up the settings for the LE's
	{
		ListItem item = { list, 0, 0, 0 };
		list->SettingsItem(item);
	}
	

done:
	if(err)
	{
		if(list) {list -> Dispose(); delete list; list = 0;}
		if(uncertaintyLEList) {uncertaintyLEList -> Dispose(); delete uncertaintyLEList; uncertaintyLEList = 0;}
	}
	
}


OSErr TNesdisOverlay::SettingsItem (ListItem item)
{
	if(ControlKeyDown()){
		this->AddSprayedLEsToLEList();
	}
	else 
	{
		TOverlay::SettingsItem (item); 
	}
	return noErr;
}




//////////////////////////////
///////////////////////////////



TBuoyOverlay::TBuoyOverlay ()
{
	// constructor
	memset(&fBuoyPoints,0,sizeof(fBuoyPoints));
	fColor = colors[BLUE];
}



void TBuoyOverlay::Dispose()
{
	// dispose of the allocated memory
	this->DisposeBuoyPoints();
	TOverlay::Dispose ();
}



void TBuoyOverlay::DisposeBuoyPoints(void)
{
	if(fBuoyPoints.pts) free(fBuoyPoints.pts); fBuoyPoints.pts = 0;
	memset(&fBuoyPoints,0,sizeof(fBuoyPoints));
	fFilePath[0] = 0; // make sure the file path is cleared 
}


OSErr TBuoyOverlay::AllocateBuoyPoints(long numToAllocate)
{
	DisposeBuoyPoints();	// in case they are allocated
	if(numToAllocate > 0) { 
		fBuoyPoints.pts = (BuoyPoint*)calloc(numToAllocate,sizeof(*fBuoyPoints.pts));
		if(fBuoyPoints.pts) {
			fBuoyPoints.numAllocated = numToAllocate;
			return noErr;
		}
		// else
		return memFullErr;
	}
	return noErr;
}

long TBuoyOverlay::NumBuoys ()
{
	return 1; 
}


OSErr TBuoyOverlay::ReadFromFile(char * path)
{
	return noErr;
}

void DrawBuoyTracks(BuoyPointsInfo buoyPoints, Rect r, WorldRect view)
{
	long i;
	long prevPolyNum;
	Boolean useMoveTo;
	BuoyPoint lastPt,thisPt;
	Point p;
	WorldPoint wp;
	WorldPointF wpf;

	memset(&lastPt,0,sizeof(lastPt));
	for(i = 0; i < buoyPoints.numFilledIn; i++) {
		thisPt = buoyPoints.pts[i];
		useMoveTo =  false ; // set to use lineto
		if(i == 0) 
			useMoveTo = true; // first buoy
		else if(lastPt.buoyNum != thisPt.buoyNum)
			useMoveTo = true; // new buoy

		wp.pLat = 1000000.0*thisPt.lat;
		wp.pLong = 1000000.0*thisPt.lng;

		p = WorldToScreenPointRound(wp,view,r); 

		if(useMoveTo)MyMoveTo(p.h,p.v);
		else MyLineTo(p.h,p.v);

		lastPt = thisPt;
	}
}

void TBuoyOverlay::Draw (Rect r, WorldRect view)
{
	// we may eventually want to draw different buoys in different colors
	if(fBuoyPoints.numFilledIn == 0)
		return; // nothing to draw

	if(!bShowOverlay) 
		return; //nothing to draw

	
	RGBForeColor(&fColor);

	DrawBuoyTracks(fBuoyPoints,r,view);
	RGBForeColor(&colors[BLACK]);
	PenNormal();
	
}

long TBuoyOverlay::GetListLength ()
{
	// add support for more than one buoy
	return 1;
}



ListItem TBuoyOverlay::GetNthListItem (long n, short indent, short *style, char *text)
{
	// add support for more than one buoy
	ListItem item = { this, 0, indent, 0 };
	char *p;
	if (n == 0) {
		item.index = I_OVERLAYNAME;
		item.bullet = bShowOverlay ? BULLET_FILLEDBOX :BULLET_EMPTYBOX ;
		p = strrchr(this->fFilePath,DIRDELIMITER);
		if(p){
			sprintf(text,"Buoy: %s",p+1); // the short file name
		}
		else {
			text[0] = 0;
		}
		
		return item;
	}
	n -= 1;
	
	/////////////
	item.owner = 0;
	return item;
}

Boolean TBuoyOverlay::ListClick (ListItem item, Boolean inBullet, Boolean doubleClick)
{
	// add support for more than one buoy
	if (inBullet && item.index == I_OVERLAYNAME){ 
		bShowOverlay = !bShowOverlay;
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT); 
		return TRUE; 
	}
	if (doubleClick && item.index == I_OVERLAYNAME)  {
		SettingsItem(item);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT); 
		return true;
	}
		
	return false;
}


Boolean TBuoyOverlay::FunctionEnabled (ListItem item, short buttonID)
{
	// add support for toggle when more than one buoy
	long i;
	
	switch (item.index) {
		case I_OVERLAYNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE; 
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!model->fOverlayList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (model->fOverlayList->GetItemCount() - 1);
					}
			}
			break;
	}
	return false;
}

OSErr TBuoyOverlay::SettingsItem (ListItem item)
{
	// add support to allow each buoy to be a differnet color ?
	fColor =  MyPickColor(fColor,mapWindow);
	model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT); 
	return noErr;
}



////////////////////////////
///////////////////////////


/////////////
////////////////


Boolean IsBpBuoyFile(char * path)
{
	// read the top of the file to see if it is a buoy file
	// First lines start with #
	// last header line should start with 
	//#FHD_ID, UTC Date, UTC Time, Lat, Lon,
	OSErr	err = noErr;
	#define kMaxHeaderLen 5000
	char	firstPartOfFile [kMaxHeaderLen];
	long lenToRead,fileLength;
	char *key;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(kMaxHeaderLen,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	if(err) return false;
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	key = "#FHD_ID, UTC Date, UTC Time, Lat, Lon,";
	if (strstr (firstPartOfFile, key)) 
		return true; //found the key

	return false; // failed the test
}


TBpBuoyOverlay::TBpBuoyOverlay ()
{
	// constructor
}




OSErr TBpBuoyOverlay::ReadFromFile(char * path)
{
	OSErr err = 0;
	long numLinesInFile = 0;
	long numPoints = 0;
	CHARH f = 0;
	char *fileContents;
	long lineNum;
	char s[256];
	double lat,lng;
	int buoyNum;
	// for keeping track where we are in the file
	int thisBuoyNum = -1; 
	int i;
	BuoyPoint bp;
	char *p;

	memset(&bp,0,sizeof(bp));

	SetWatchCursor();

	// these files are not huge, read entire file into memory
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("BpBuoyOverlay::ReadFromFile()", "ReadFileContents()", err);
		goto done;
	}
	_HLock((Handle)f); 
	fileContents = *f;

	// then create a list of points we can draw from
	numLinesInFile = NumLinesInText(fileContents);
	err = AllocateBuoyPoints(numLinesInFile);
	if(err) goto done;


	for(lineNum = 0;lineNum <numLinesInFile;) {
		// read the next line
		NthLineInTextOptimized (fileContents, lineNum++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);

		if(s[0] == '#')// it is a header line
			continue; 

		if(s[0] == 0)// it is a blank line separating the buoys
			continue; 


		// example line
		//02521, 2010/05/27, 22:55,   28.562,  -86.237,   0.9,   3.0,  0.09, -43.3
		p = strtok(s, ","); // buoy number
		if(p) thisBuoyNum = atol(p);
		p = strtok(0, ","); // date
		p = strtok(0, ","); // time
		p = strtok(0, ","); // lat
		if(p) lat = atof(p);
		p = strtok(0, ","); // lng
		if(p) lng = atof(p);
		if(p && -180 <= lat && lat <= 180 && -180 <= lng && lat <= 180) {
			// it is a point
			if(fBuoyPoints.numFilledIn < fBuoyPoints.numAllocated){ // safety check
				bp.lat = lat;
				bp.lng = lng;
				bp.buoyNum = thisBuoyNum;
				fBuoyPoints.pts[fBuoyPoints.numFilledIn++] = bp;	
			}

		}
		else {
			char msg[512];
			sprintf(msg,"Error parsing line %d of file %s",lineNum,path);
			printError(msg);
			err = true; 
			goto done;
			// error in the file
		}

	}

	strcpy(fFilePath,path); // record the path now that we have been successful
	this->SetClassNameToFileName(); // for the wizard

	// figure out the bounds
	fBounds = voidWorldRect;
	for(i = 0; i < fBuoyPoints.numFilledIn; i++) {
		WorldPoint wp;
		wp.pLat = 1000000.0*fBuoyPoints.pts[i].lat;
		wp.pLong = 1000000.0*fBuoyPoints.pts[i].lng;
		AddWPointToWRect(wp.pLat, wp.pLong, &fBounds);
	}
	
	// make sure the overlay is shown in the current view
	if(!EqualWRects(voidWorldRect,fBounds))
		ChangeCurrentView(UnionWRect(settings.currentView,fBounds), TRUE, TRUE);

done:
	// cause a screen refresh
	InvalMapDrawingRect();
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	return err;
}


///////////////////////////
////////////////////////////

// SLDMB Buoy

/*
"BuoyNum","DateTimeZULU","Latitude","Longitude","SST","Dir","Speed","Source"
"43442","06/01/2010 11:00:00","26.894814","-86.07466","28.2","129","2","GPS"
*/


Boolean IsSLDMBBuoyFile(char * path)
{
	// read the top of the file to see if it is a buoy file
	// first line should be "Polygon"
	// next line should be "0 0"
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	s [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	char *key;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	if(err) return false;
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	NthLineInTextOptimized (firstPartOfFile, line = 0, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	key = "\"BuoyNum\",\"DateTimeZULU\",\"Latitude\",\"Longitude\",";
	if (strncmpnocase (s, key, strlen(key))) 
		return false; //did not match the key

	return true; // passed the test
}



TSLDMBBuoyOverlay::TSLDMBBuoyOverlay ()
{
	// constructor
}

void RemoveDoubleQuotes(char *s)
{
	// should we remove all doouble quotes or just the outer ones ?
	// Perhaps inner ones would have escape chars or be pairs of double quotes
	int i;
	char* p;
	if(s == NULL) return;

	if(s[0] == '"') {
		// move the rest of the string over
		for(p = s+1;true;p++) {
			*(p-1) = *p;
			if((*p) == 0) {// end of the string 
				// remove the last char if it is a double quote
				// note that p-1 in the new terminator and the last char of the new string in p-2
				if(*(p-2) == '"')
					*(p-2) = 0;
				break; 
			}
		}
	}
	

}


OSErr TSLDMBBuoyOverlay::ReadFromFile(char * path)
{
	OSErr err = 0;
	long numLinesInFile = 0;
	long numPoints = 0;
	CHARH f = 0;
	char *fileContents;
	long lineNum;
	char s[256];
	double lat,lng;
	int buoyNum;
	// for keeping track where we are in the file
	int thisBuoyNum = -1; 
	int i;
	BuoyPoint bp;
	char *p;

	memset(&bp,0,sizeof(bp));

	SetWatchCursor();

	// these files are not huge, read entire file into memory
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TSLDMBBuoyOverlay::ReadFromFile()", "ReadFileContents()", err);
		goto done;
	}
	_HLock((Handle)f); 
	fileContents = *f;

	// then create a list of points we can draw from
	numLinesInFile = NumLinesInText(fileContents);
	err = AllocateBuoyPoints(numLinesInFile);
	if(err) goto done;


	for(lineNum = 0;lineNum <numLinesInFile;) {
		// read the next line
		NthLineInTextNonOptimized (fileContents, lineNum++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);

		if(lineNum == 1)// it is a header line
			continue; 

		if(s[0] == 0)// it is a blank line separating the buoys
			continue; 


		// example line
		//"43166","06/09/2010 11:00:00","24.925599","-85.173609","29.4","236","2.2","GPS"
		p = strtok(s, ","); // buoy number
		if(p) RemoveDoubleQuotes(p);
		if(p) thisBuoyNum = atol(p);

		p = strtok(0, ","); // date time

		p = strtok(0, ","); // lat
		if(p) RemoveDoubleQuotes(p);
		if(p) lat = atof(p);

		p = strtok(0, ","); // lng
		if(p) RemoveDoubleQuotes(p);
		if(p) lng = atof(p);

		if(p && -180 <= lat && lat <= 180 && -180 <= lng && lat <= 180) {
			// it is a point
			if(fBuoyPoints.numFilledIn < fBuoyPoints.numAllocated){ // safety check
				bp.lat = lat;
				bp.lng = lng;
				bp.buoyNum = thisBuoyNum;
				fBuoyPoints.pts[fBuoyPoints.numFilledIn++] = bp;	
			}

		}
		else {
			char msg[512];
			sprintf(msg,"Error parsing line %d of file %s",lineNum,path);
			printError(msg);
			err = true; 
			goto done;
			// error in the file
		}

	}

	strcpy(fFilePath,path); // record the path now that we have been successful
	this->SetClassNameToFileName(); // for the wizard

	// figure out the bounds
	fBounds = voidWorldRect;
	for(i = 0; i < fBuoyPoints.numFilledIn; i++) {
		WorldPoint wp;
		wp.pLat = 1000000.0*fBuoyPoints.pts[i].lat;
		wp.pLong = 1000000.0*fBuoyPoints.pts[i].lng;
		AddWPointToWRect(wp.pLat, wp.pLong, &fBounds);
	}
	
	// make sure the overlay is shown in the current view
	if(!EqualWRects(voidWorldRect,fBounds))
		ChangeCurrentView(UnionWRect(settings.currentView,fBounds), TRUE, TRUE);

done:
	// cause a screen refresh
	InvalMapDrawingRect();
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	return err;
}



//////////////////////
///////////////////////



Boolean IsOverflightFile(char * path) // JLM 5/15/10
{
	// read the top of the file to see if it is in the recognized format
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long lineNum;
	#define kMaxSizeToRead 1024
	char	s [kMaxSizeToRead];
	char	firstPartOfFile [kMaxSizeToRead];
	long lenToRead,fileLength;
	char *key;
	double lat,lng;
	int numScanned;
	int numLinesInThisPart;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(kMaxSizeToRead,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	if(err) return false;
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString

	numLinesInThisPart = NumLinesInText(firstPartOfFile);

	lineNum = 0;

	// first line should be start with "Grid"
	key = "Grid";
	NthLineInTextOptimized (firstPartOfFile, lineNum++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	if (strncmpnocase (s, key, strlen(key))) 
		return false; //did not match the key

	// next line should be "Datum"
	key = "Datum";
	NthLineInTextOptimized (firstPartOfFile, lineNum++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	if (strncmpnocase (s, key, strlen(key))) 
		return false; //did not match the key

	// one of the next few lines should start with "Waypoint" or "Trackpoint"
	for(;lineNum <numLinesInThisPart;) {
		NthLineInTextOptimized (firstPartOfFile, lineNum++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);
		key = "Waypoint";
		if (strncmpnocase (s, key, strlen(key)) == 0) 
			return true; //matched the key
		key = "Trackpoint";
		if (strncmpnocase (s, key, strlen(key)) == 0) 
			return true; //matched the key
	}


	return false; // failed the test
}



TOverflightOverlay::TOverflightOverlay ()
{
	// constructor
	memset(&fTrackPoints,0,sizeof(fTrackPoints));

}



void TOverflightOverlay::Dispose()
{
	// dispose of the allocated memory
	this->DisposeTrackPoints();
	TOverlay::Dispose ();
}



void TOverflightOverlay::DisposeTrackPoints(void)
{
	if(fTrackPoints.pts) free(fTrackPoints.pts); fTrackPoints.pts = 0;
	memset(&fTrackPoints,0,sizeof(fTrackPoints));
	fFilePath[0] = 0; // make sure the file path is cleared 
}


OSErr TOverflightOverlay::AllocateTrackPoints(long numToAllocate)
{
	DisposeTrackPoints();	// in case they are allocated
	if(numToAllocate > 0) { 
		fTrackPoints.pts = (TrackPoint*)calloc(numToAllocate,sizeof(*fTrackPoints.pts));
		if(fTrackPoints.pts) {
			fTrackPoints.numAllocated = numToAllocate;
			return noErr;
		}
		// else
		return memFullErr;
	}
	return noErr;
}



OSErr TOverflightOverlay::ReadFromFile(char * path)
{
	OSErr err = 0;
	long polygonNum, zeroOrOne;
	long numLinesInFile = 0;
	long numPoints = 0;
	CHARH f = 0;
	char *fileContents;
	long lineNum;
	char s[256];
	char extraStuff[256];
	int numScanned;
	double lat,lng;
	int ptNum,polyNum,flag;
	// for keeping track where we are in the file
	int thisTrackNum = -1; 
	int thisPolyFlag = -1; 
	int thisPolyInteriorRingFlag = -1;
	TrackPoint tp;
	int i;
	char *key = 0;
	char *p;

	memset(&tp,0,sizeof(tp));


	SetWatchCursor();

	// these files are not huge, read entire file into memory
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TOverflightOverlay::ReadFromFile()", "ReadFileContents()", err);
		goto done;
	}
	_HLock((Handle)f); 
	fileContents = *f;

	// then create a list of points we can draw from
	numLinesInFile = NumLinesInText(fileContents);
	err = AllocateTrackPoints(numLinesInFile);
	if(err) goto done;


	for(lineNum = 0;lineNum < numLinesInFile;) {
		// read the next line
		NthLineInTextOptimized (fileContents, lineNum++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);

		if(s[0] == 0) // blank line (possibly due to hand editing)
			continue;

		// determine if it is :
		//		the end of the file
		//		new track
		//		point in the current track

		key = "Track\t";
		if(strncmpnocase(s,key,strlen(key)) == 0){
			// it is a header line for a new track
			// code goes here could read the actual track here, e.g. "ACTIVE LOG 003"
			//strtok(s,"\t");
			thisTrackNum++; 
			continue;
		}

		key = "Trackpoint";
		if (strncmpnocase(s,key,strlen(key)) == 0){
			// parse the trackpoint
			lat = lng =  -200;  // initialize to bad values

			// example line (tab delimiters)
			//"Trackpoint<tab>N29 17.766 W89 22.323<tab>6/20/2010 10:03:13 AM<tab>-44 ft<tab>12 ft<tab>0:00:01<tab>8 mph<tab>219° true"
			p = strtok(s, "\t"); // key

			p = strtok(0, "\t"); // lat lng
			if(p) {
				// e.g. "N29 17.766 W89 22.323"
				#define kMaxLatLngStrLen 64
				if(strlen(p) < kMaxLatLngStrLen) {
					char s1[kMaxLatLngStrLen],s2[kMaxLatLngStrLen],s3[kMaxLatLngStrLen],s4[kMaxLatLngStrLen];
					numScanned  = sscanf(p,"%s %s %s %s",s1,s2,s3,s4);
					if (numScanned == 4 && strlen(s1) > 1 && strlen(s2) > 1 && strlen(s3) > 1 && strlen(s4) > 1 ) {
						int sign;
						if(s1[0] == 'S') sign = -1;
						else sign = 1;
						lat = sign*(atof(s1+1) + atof(s2)/60.0);
						if(s3[0] == 'W') sign = -1;
						else sign = 1;
						lng = sign*(atof(s3+1) + atof(s4)/60.0);
					}
				}
			}
			// check the data and report any parsing errors
			if(p && -180 <= lat && lat <= 180 && -180 <= lng && lat <= 180) {
				// it is a point
				if(fTrackPoints.numFilledIn < fTrackPoints.numAllocated){ // safety check
					tp.lat = lat;
					tp.lng = lng;
					tp.trackNum = thisTrackNum;
					fTrackPoints.pts[fTrackPoints.numFilledIn++] = tp;	
				}

			}
			else {
				char msg[512];
				sprintf(msg,"Error parsing line %d of file %s",lineNum,path);
				printError(msg);
				err = true; 
				goto done;
				// error in the file
			}

		}
	}

	strcpy(fFilePath,path); // record the path now that we have been successful
	this->SetClassNameToFileName(); // for the wizard

	// figure out the bounds
	fBounds = voidWorldRect;
	for(i = 0; i < fTrackPoints.numFilledIn; i++) {
		WorldPoint wp;
		tp = fTrackPoints.pts[i];
		wp.pLat = 1000000.0*tp.lat;
		wp.pLong = 1000000.0*tp.lng;
		AddWPointToWRect(wp.pLat, wp.pLong, &fBounds);
	}
	
	// make sure the overlay is shown in the current view
	if(!EqualWRects(voidWorldRect,fBounds))
		ChangeCurrentView(UnionWRect(settings.currentView,fBounds), TRUE, TRUE);

done:
	// cause a screen refresh
	InvalMapDrawingRect();
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}
	return err;
}


ListItem TOverflightOverlay::GetNthListItem (long n, short indent, short *style, char *text)
{
	ListItem item = { this, 0, indent, 0 };
	char *p;
	if (n == 0) {
		item.index = I_OVERLAYNAME;
		item.bullet = bShowOverlay ? BULLET_FILLEDBOX :BULLET_EMPTYBOX ;
		p = strrchr(this->fFilePath,DIRDELIMITER);
		if(p){
			sprintf(text,"Overflight: %s",p+1); // the short file name
		}
		else {
			text[0] = 0;
		}
		
		return item;
	}
	n -= 1;
	
	/////////////
	item.owner = 0;
	return item;
}



//settings.currentView,MapDrawingRect()
void DrawTrackLine(TrackPointsInfo trackPoints, Rect r, WorldRect view)
{
	long i;
	long prevPolyNum;
	Boolean useMoveTo;
	TrackPoint lastPt,thisPt;
	Point p;
	WorldPoint wp;
	WorldPointF wpf;

	memset(&lastPt,0,sizeof(lastPt));
	for(i = 0; i < trackPoints.numFilledIn; i++) {
		thisPt = trackPoints.pts[i];
		useMoveTo =  false ; // set to use lineto
		if(i == 0) 
			useMoveTo = true; // first track
		else if(lastPt.trackNum != thisPt.trackNum)
			useMoveTo = true; // new track

		wp.pLat = 1000000.0*thisPt.lat;
		wp.pLong = 1000000.0*thisPt.lng;

		p = WorldToScreenPointRound(wp,view,r); 

		if(useMoveTo)MyMoveTo(p.h,p.v);
		else MyLineTo(p.h,p.v);

		lastPt = thisPt;
	}
}


void TOverflightOverlay::Draw (Rect r, WorldRect view)
{

	if(fTrackPoints.numFilledIn == 0)
		return; // nothing to draw

	if(!bShowOverlay) 
		return; //nothing to draw

	
	RGBForeColor(&fColor);
	PenSize(3,3);
	PenSize(1,1);
	DrawTrackLine(fTrackPoints,r,view);
	RGBForeColor(&colors[BLACK]);
	PenNormal();
	


}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

Boolean IsOverlayFile(char* path)
{
	Boolean isESI = false;
	if(IsNesdisPolygonFile(path)) return true;
	if(IsBpBuoyFile(path)) return true;
	if(IsSLDMBBuoyFile(path)) return true;
	if(IsOverflightFile(path)) return true;
	if(IsVectorMap(path, &isESI)) return true;
	return false; // not a recognized type of overlay file
}




OSErr AddOverlayFromFile(char *path) 
{
	OSErr err = noErr;
	char tempStr[256];
	TOverlay *thisOverlay = NULL;
	Boolean isESI = false;

	if(!IsOverlayFile(path)) {
		err = true;
		strcpy(tempStr,"This file is not a recognizable overlay file.");
		printError(tempStr);
	}
	else if (IsNesdisPolygonFile (path)){
		thisOverlay = new TNesdisOverlay ();
	}
	else if(IsBpBuoyFile(path)) {
		thisOverlay = new TBpBuoyOverlay ();
	}
	else if(IsSLDMBBuoyFile(path)) {
		thisOverlay = new TSLDMBBuoyOverlay ();
	}
	else if(IsOverflightFile(path)) {
		thisOverlay = new TOverflightOverlay ();
	}
	else if (IsVectorMap (path, &isESI)){
		thisOverlay = new TNesdisOverlay ();
	}
	/// read the file and add the overlay if we got one
	if(thisOverlay) {
		if (IsVectorMap (path, &isESI)){
			err = ((TNesdisOverlay*)thisOverlay)->ReadFromBNAFile(path);
		}
		else err = thisOverlay->ReadFromFile(path);
		//err = thisOverlay->ReadFromFile(path);
		if(!err) err = model->AddOverlay(thisOverlay, 0);
	
		if (err)
		{
			thisOverlay -> Dispose ();
			delete thisOverlay;
			err = -1;
		}		
	}

	model->NewDirtNotification();

	return err;

}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
OSErr AddOverlayDialog(void)
{
	char 		path[256], nameStr [256], shortFileName[256];
	char tempStr[256];
	OSErr		err = noErr;
	short dialogID = -1;// an undefined dialog
	Point 		where = CenteredDialogUpLeft(dialogID); 
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;
	TOverlay *thisOverlay = 0;

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38g, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
		strcpy(tempStr,path);
		SplitPathFile(tempStr,shortFileName);
#else
	paramtext("","","","");
	#ifdef MAC
		dialogID = M38g;
	#endif
	sfpgetfile(&where, "",
			   (FileFilterUPP)0,
			   -1, typeList,
			   (DlgHookUPP)0,
			   &reply, dialogID,
			   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) return USERCANCEL;

	my_p2cstr(reply.fName);
	#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
		strcpy(shortFileName,(char*) reply.fName);
	#else
		strcpy(path, reply.fName);
		strcpy(tempStr,path);
		SplitPathFile(tempStr,shortFileName);
	#endif
#endif	

	err = AddOverlayFromFile(path);
	if(err) goto done;
	if(NumFilesSelected() > 1) { 
		long i;
		long n = NumFilesSelected();
		for(i=2; i <= n; i++) {
			strcpyNthFile(path,i); // let's assume they are all of the correct type
			err = AddOverlayFromFile(path);
			if(err) goto done;
		}
	}

done:
	return err;

}


