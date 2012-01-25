
#include "Cross.h"
#include "MapUtils.h"
#include "GenDefs.h"


/**************************************************************************************************/
OSErr AddOMapDialog()
{
	char 		path[256];
	OSErr		err = noErr;
	long 		n;
	Point 		where = CenteredDialogUpLeft(M38b);
	TOSSMMap 	*map;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;

#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(path, reply.fullPath);
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
	map = new TOSSMMap("", voidWorldRect);
	if (!map)
		{ TechError("AddOMapDialog()", "new TOSSMMap()", 0); return -1; }

	if (err = map->InitMap(path)) { delete map; return err; }

	if (err = model->AddMap(map, 0))
		{ map->Dispose(); delete map; return -1; }
	else {
		model->NewDirtNotification();
	}

	return err;
}

/////////////////////////////////////////////////
long NumColsInOssmGrid (char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line,row,col,n,i;
	char	strLine [256], num[10];
	char	firstPartOfFile [256];
	long lenToRead,fileLength,numCols=0;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	if (err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0)) return 0;
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString

	// take the first data line and count the number of cells
	NthLineInTextNonOptimized (firstPartOfFile, line = 2, strLine, 256);	

	for (col = 0, i = 0 ; /*col < fNumCols &&*/ strLine[i] ; col += n) {
		strnzcpy(num, &strLine[i], 2); i += 2;
		sscanf(num, "%ld", &n);
		numCols += n;
		i += 2;	// the land type code
	}

	return numCols;
}

///////////////////////////////////////////////////////////////////////////

TOSSMMap::TOSSMMap(char* name, WorldRect bounds)
		: TMap(name, bounds)
{
	fGridHdl = 0;
	fRefloatTimesHdl = 0;
}

void TOSSMMap::Dispose()
{
	if (fGridHdl) {
		DisposeHandle((Handle)fGridHdl);
		fGridHdl = 0;
	}
	if (fRefloatTimesHdl) {
		DisposeHandle((Handle)fRefloatTimesHdl);
		fRefloatTimesHdl = 0;
	}
	
	TMap::Dispose();
}

OSErr TOSSMMap::InitMap()
{
	fGridHdl = nil;
	fRefloatTimesHdl = nil;

	return InitMap(0);
}

OSErr TOSSMMap::InitMap(char *path)
{	// updated to allow non-standard sized OSSM Maps 10/27/03
	char s[500], name[kMaxNameLen], num[10], code[10];
	long i, j, n, col, row, line = 0;
	long numLines,numHeaderAndFooterLines=4;
	float highLatDegrees, highLatMinutes, lowLatDegrees, lowLatMinutes,
		  lowLongDegrees, lowLongMinutes, highLongDegrees, highLongMinutes;
	CHARH f = 0;
	WorldRect bounds;
	OSErr err = noErr;
	long totalPts;

	//fNumRows = kOMapHeight;
	//fNumCols = kOMapWidth;
	//totalPts = fNumRows*fNumCols;

	err = TMap::InitMap();

	/*if (!err && !fGridHdl)
	{
		fGridHdl = (CHARH)_NewHandleClear(totalPts);
		if (!fGridHdl) { TechError("TOSSMMap::InitMap()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
	}*/

	if (!err)
	{
		//for (row = 0 ; row < fNumRows ; row++)
			//for (col = 0 ; col < fNumCols ; col++)
				//INDEXH(fGridHdl, row * fNumCols + col) = LT_WATER;
				//INDEXH(fGridHdl, row * fNumCols + col) = OSSM_WW; // use default water type

		if (!path) return 0;
		
		fNumCols = NumColsInOssmGrid (path);

		if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
			{ TechError("TOSSMMap::InitMap()", "ReadFileContents()", err); goto done; }
		
		numLines = NumLinesInText(*f);
		fNumRows = numLines - numHeaderAndFooterLines;	// might need to check fNumRows since could have multiple override lines?

		totalPts = fNumRows * fNumCols;
		if (!err && !fGridHdl)
		{
			fGridHdl = (CHARH)_NewHandleClear(totalPts);
			if (!fGridHdl) { TechError("TOSSMMap::InitMap()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
		}

		NthLineInTextOptimized(*f, line++, s, 500); // map name
		strnztrimcpy(name, s, kMaxNameLen - 1);
		SetMapName(name);
		
		NthLineInTextOptimized(*f, line++, s, 500); // bounding rect
		if (sscanf(s, "%f, %f, %f, %f, %f, %f, %f, %f",
			&highLatDegrees, &highLatMinutes,
			&lowLatDegrees, &lowLatMinutes,
			&lowLongDegrees, &lowLongMinutes,
			&highLongDegrees, &highLongMinutes) != 8)
			{ printError("OSSM map file invalid."); err = -1; goto done; }
		else
		{
			SetWorldRect(&bounds,
						 (lowLatDegrees + lowLatMinutes / 60.0) * 1000000,
						 (lowLongDegrees + lowLongMinutes / 60.0) * 1000000,
						 (highLatDegrees + highLatMinutes / 60.0) * 1000000,
						 (highLongDegrees + highLongMinutes / 60.0) * 1000000);
			///////////////////////////////////////////
			// in old OSSM style, you determined the hemisphere by whether hiLong < loLong
			if (bounds.hiLat < bounds.loLat)
			{
				bounds.hiLat = -bounds.hiLat;
				bounds.loLat = -bounds.loLat;
			}

			if (bounds.hiLong < bounds.loLong)
			{
				bounds.hiLong = -bounds.hiLong;
				bounds.loLong = -bounds.loLong;
			}
			///////////////////////////////////////////

			SetMapBounds(bounds);
		}

		if (!fRefloatTimesHdl)
		{
			fRefloatTimesHdl = (FLOATH)_NewHandleClear(sizeof(float)*(kNumOSSMLandTypes+kNumOSSMWaterTypes));
			if (!fRefloatTimesHdl) { TechError("TOSSMMap::InitMap()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
		}

		SetDefaultRefloatTimes();

		for (row = 0 ; row < fNumRows ; row++) {
			NthLineInTextOptimized(*f, line++, s, 500); // grid row
			if (s[0] == 0) {line--; break;}	// if there are override values fNumRows will be too small
			for (col = 0, i = 0 ; col < fNumCols && s[i] ; col += n) {
				strnzcpy(num, &s[i], 2); i += 2;
				sscanf(num, "%ld", &n);
				strnzcpy(code, &s[i], 2); i += 2;
				//if (code[0] == 'L')
					for (j = 0 ; j < n && (col + j) < fNumCols ; j++)
					{
						// code goes here, keep the land type
						// get land type from code[1] and put into fGridHdl
						short type = ConvertToLandType(code);
						if (type==LT_UNDEFINED) {err=-1; printError("Undefined land type in OSSM map file"); goto done;}
						//INDEXH(fGridHdl, row * fNumCols + col + j) = LT_LAND;
						INDEXH(fGridHdl, row * fNumCols + col + j) = type;
					}
			}
		}
		if (fNumRows != (line - 2)) // have to reset fGridHdl size and fNumRows
		{
			fNumRows = (line - 2);	// should only be greater...
			_SetHandleSize((Handle)fGridHdl,fNumRows*fNumCols);
			err = _MemError();
		}

		// File ends with 2 blank lines, which may be split by override values for land and water defaults
		NthLineInTextOptimized(*f, line++, s, 500); // first blank line
		while (1)
		{	
			NthLineInTextOptimized(*f, line++, s, 500); 
			if(s[0] == 0) break; // it's the second blank line so we are done
			// else scan line and set new refloat values
			err = OverrideRefloatTimes(s);
			if (err) goto done;
		}
		
	}

done:
	if (err)
	{
		if (fGridHdl != nil)
		{
			DisposeHandle ((Handle) fGridHdl);
			fGridHdl = nil;
		}
		if (fRefloatTimesHdl != nil)
		{
			DisposeHandle ((Handle) fRefloatTimesHdl);
			fRefloatTimesHdl = nil;
		}
	}
	
	if(f) DisposeHandle((Handle)f); // JLM 12/16/98

	return err;
}

OSErr TOSSMMap::ReplaceMap()
{
	char 		path[256], nameStr [256];
	OSErr		err = noErr;
	Point 		where = CenteredDialogUpLeft(M38b);
	TOSSMMap 	*map = nil;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;

#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
			   (MyDlgHookUPP)0, &reply, M38b, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return USERCANCEL;
	strcpy(path, reply.fullPath);
	strcpy (nameStr, "Grid Map: ");
	strcat (nameStr, path);	// code goes here, use short name instead
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
	strcpy (nameStr, "Grid Map: ");
	strcat (nameStr, (char*) reply.fName);
#endif
	if (!IsGridMap (path)) 
	{
		printError("New map must be of the same type.");
		return USERCANCEL;	// to return to the dialog
	}
	
	map = new TOSSMMap (nameStr, voidWorldRect);
	if (!map)
		{ TechError("ReplaceMap()", "new TOSSMMap()", 0); return -1; }

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

short TOSSMMap::ConvertToLandType(char *code)
{
	if (code[0]=='L')
	{
		if (code[1]=='0') 	// Exposed headland
			return OSSM_L0;
		else if (code[1]=='1')	// Wave-cut platform
			return OSSM_L1;
		else if (code[1]=='2')	// Pocket beach
			return OSSM_L2;
		else if (code[1]=='3')	// Sand beach
			return OSSM_L3;
		else if (code[1]=='4')	// Sand and gravel beach
			return OSSM_L4;
		else if (code[1]=='5')	// Sand and cobble beach
			return OSSM_L5;
		else if (code[1]=='6')	// Exposed tide flats
			return OSSM_L6;
		else if (code[1]=='7')	// Sheltered rock shore
			return OSSM_L7;
		else if (code[1]=='8')	// Sheltered tide flat
			return OSSM_L8;
		else if (code[1]=='9')	// Sheltered marsh
			return OSSM_L9;
		else if (code[1]=='L')	// Default land type
			return OSSM_LL;
		else
			return LT_UNDEFINED;
	}
	else if (code[0]=='W')
	{
		if (code[1]=='0')	// Exposed tide flats
			return OSSM_W0;	
		else if (code[1]=='1')	// Sheltered marsh
			return OSSM_W1;
		else if (code[1]=='2')	// Water
			return OSSM_W2;
		else if (code[1]=='3')	// Water
			return OSSM_W3;
		else if (code[1]=='W')	// Default water type
			return OSSM_WW;
		else
			return LT_UNDEFINED;
	}
	else
		return LT_UNDEFINED;
}

void TOSSMMap::SetDefaultRefloatTimes()
{	// values from On-Scene Spill Model reference manual
	(*fRefloatTimesHdl)[0]=(*fRefloatTimesHdl)[1]=(*fRefloatTimesHdl)[6]=1;		// 1 hour
	(*fRefloatTimesHdl)[2]=(*fRefloatTimesHdl)[3]=(*fRefloatTimesHdl)[4]=24;	// 1 day
	(*fRefloatTimesHdl)[5]=(*fRefloatTimesHdl)[7]=(*fRefloatTimesHdl)[8]=(*fRefloatTimesHdl)[9]=8760;	// 1 year
	(*fRefloatTimesHdl)[10] = 1;	// old default value was 1 hour, though book has this as 1 year ??
}

OSErr TOSSMMap::OverrideRefloatTimes(char *s)
{	
	long len=2, numScanned, type, vulnerability, crossRef;
	float height, newRefloatTime, speedFactor;
	char descriptor[64],code[10];
	short index;
	OSErr err = 0;

	strnzcpy(code, &s[0], len);

	if (code[0]=='L')
	{
		numScanned = sscanf(s+len,lfFix("%f%ld%ld%f%s"),&height,&type,&vulnerability,&newRefloatTime,descriptor);
		if (numScanned != 5)
			{ err = -1; TechError("TOSSMMap::OverrideRefloatTimes()", "sscanf() == 5", 0); return err; }
			
		index  = ConvertToLandType(code) - 1;
		if (index >= 0) 
			(*fRefloatTimesHdl)[index] = newRefloatTime;
		else { err = -1; printError("Unrecognized land type in TOSSMMap::OverrideRefloatTimes()"); return err; }
	
	}
	else if (code[0]=='W')
	{
		numScanned = sscanf(s+len,lfFix("%f%ld%ld%f%s"),&speedFactor,&crossRef,&vulnerability);
		if (numScanned != 3)
			{ err = -1; TechError("TOSSMMap::OverrideRefloatTimes()", "sscanf() == 3", 0); return err; }
			
		index  = ConvertToLandType(code) - 1;
		if (index >= 0) 
			(*fRefloatTimesHdl)[index] = speedFactor;
		else { err = -1; printError("Unrecognized land type in TOSSMMap::OverrideRefloatTimes()"); return err; }
	}
	else 
	{
		err = -1;
		printError("Unrecognized land type in TOSSMMap::OverrideRefloatTimes()");
	}
	return err;
}

LongPoint TOSSMMap::GetGridRowCol(WorldPoint p)
{	// returns neg values if the point is not in the bounding rect
	LongPoint ossmBox;
	WorldRect bounds;
	
	if (!InMap(p)) {
		ossmBox.h = -1;
		ossmBox.v = -1;
	}
	else
	{
		WorldRect bounds = GetMapBounds();
		long dX = WRectWidth(bounds) / fNumCols;
		long dY = WRectHeight(bounds) / fNumRows;
		
		long col = (p.pLong - bounds.loLong) / dX;
		long row = (bounds.hiLat - p.pLat) / dY;
		// enforce bounds bdue to round off etc
		if (col > fNumCols  - 1) col = fNumCols  - 1;
		if (col < 0) col = 0;
		if (row > fNumRows - 1) row = fNumRows - 1;
		if (row < 0) row = 0;
		ossmBox.h = col;
		ossmBox.v = row;
	}
	return ossmBox;
}

long TOSSMMap::GetLandType(WorldPoint p)
{
	long col, row;
	LongPoint pt = this -> GetGridRowCol(p);
	if (fGridHdl && pt.h >= 0 && pt.v >= 0) {
		row = pt.v;
		col = pt.h;
		return INDEXH(fGridHdl, row * fNumCols + col);
	}
	return LT_UNDEFINED;
	
}

float TOSSMMap::RefloatHalfLifeInHrs(WorldPoint p) 
{ 
	float refloatHalfLifeInHrs;
	short type = GetLandType(p); 
	if (type <= kNumOSSMLandTypes) 
		refloatHalfLifeInHrs = (*fRefloatTimesHdl)[type-1];
	else // water types, should this ever happen?
		refloatHalfLifeInHrs = 1;
	return refloatHalfLifeInHrs;
}


Boolean TOSSMMap::OnLand(WorldPoint p)
{
	short	type = GetLandType(p);

	if (type <= kNumOSSMLandTypes) 
		return TRUE; // there are 11 OSSM land types, the larger numbers are water types
	return FALSE;
	//return type == LT_LAND;
}


long TOSSMMap::NumPtsInGridHdl(void)
{
	long numInHdl = 0;
	if (fGridHdl) numInHdl = _GetHandleSize((Handle)fGridHdl)/sizeof(**fGridHdl);
	
	return numInHdl;
}

#define TOSSMMapREADWRITEVERSION  2	// updated to allow fNumRows and fNumCols to vary 10/27/03
OSErr TOSSMMap::Write(BFPB *bfpb)
{
	char r;
	long i;
	long version = TOSSMMapREADWRITEVERSION;
	ClassID id = GetClassID ();
	long totalPts = this -> NumPtsInGridHdl();
	long numRefloatTimes = kNumOSSMLandTypes + kNumOSSMWaterTypes;
	float val;
	OSErr err = 0;
	
	if (err = TMap::Write(bfpb)) return err;
		
	StartReadWriteSequence("TOSSMMap::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	/////
	if (totalPts != fNumRows * fNumCols) printError("Total points not equal to numRows * numCols in OSSM Map");
	if (err = WriteMacValue(bfpb, fNumRows)) return err;
	if (err = WriteMacValue(bfpb, fNumCols)) return err;
	//if (err = WriteMacValue(bfpb, totalPts)) return err;
	for (i = 0 ; i < totalPts ; i++) {
		r = INDEXH(fGridHdl, i);
		if (err = WriteMacValue(bfpb, r)) return err;
	}
	
	if (err = WriteMacValue(bfpb, numRefloatTimes)) return err;
	for (i = 0 ; i < numRefloatTimes ; i++) {
		val = INDEXH(fRefloatTimesHdl, i);
		if (err = WriteMacValue(bfpb, val)) return err;
	}
	
	return 0;
}

OSErr TOSSMMap::Read(BFPB *bfpb)
{
	char r;
	long i,version;
	ClassID id;
	long 	totalPts, numRefloatTimes;
	float val;
	OSErr err = 0;
	
	if (err = TMap::Read(bfpb)) return err;

	StartReadWriteSequence("TOSSMMap::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TOSSMMap::Read()", "id == TYPE_OSSMMAP", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TOSSMMapREADWRITEVERSION || version < 1) { printSaveFileVersionError(); return -1; }
	
	if (version == 1)
	{
		if (err = ReadMacValue(bfpb, &totalPts)) return err;
		fNumRows = kOMapHeight;
		fNumCols = kOMapWidth;
	}
	else
	{
		if (err = ReadMacValue(bfpb, &fNumRows)) return err;
		if (err = ReadMacValue(bfpb, &fNumCols)) return err;
		totalPts = fNumRows * fNumCols;
	}
	// if allow numRow, numCols to be different from 48 x 80 will need to write out the values
	if (totalPts != fNumRows*fNumCols){ printSaveFileVersionError(); return -1; }
	if (!err)
	{
		fGridHdl = (CHARH)_NewHandleClear(totalPts);
		if (!fGridHdl)
			{ TechError("TOSSMMap::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < totalPts ; i++) {
			if (err = ReadMacValue(bfpb, &r)) { printSaveFileVersionError(); return -1; }
			INDEXH(fGridHdl, i) = r;
		}
	}

	if (err = ReadMacValue(bfpb, &numRefloatTimes)) return err;	
	if (numRefloatTimes != kNumOSSMLandTypes + kNumOSSMWaterTypes){ printSaveFileVersionError(); return -1; }
	if (!err)
	{
		fRefloatTimesHdl = (FLOATH)_NewHandleClear(sizeof(float)*numRefloatTimes);
		if (!fRefloatTimesHdl)
			{ TechError("TOSSMMap::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < numRefloatTimes ; i++) {
			if (err = ReadMacValue(bfpb, &val)) { printSaveFileVersionError(); return -1; }
			INDEXH(fRefloatTimesHdl, i) = val;
		}
	}

	return 0;
}

void TOSSMMap::Draw(Rect r, WorldRect view)
{
	short col, row, color;
	double dX, dY;
	Rect c;
	WorldRect cell;
	
	
	/////////////////////////////////////////////////
	// JLM 6/10/99 maps must erase their rectangles in case a lower priority map drew in our rectangle
	LongRect	mapLongRect;
	Rect m;
	Point topLeftPt, bottomRightPt;
	Boolean  onQuickDrawPlane, offQuickDrawPlane;
	WorldRect bounds = this -> GetMapBounds();
	mapLongRect.left = SameDifferenceX(bounds.loLong);
	mapLongRect.top = (r.bottom + r.top) - SameDifferenceY(bounds.hiLat);
	mapLongRect.right = SameDifferenceX(bounds.hiLong);
	mapLongRect.bottom = (r.bottom + r.top) - SameDifferenceY(bounds.loLat);
	onQuickDrawPlane = IntersectToQuickDrawPlane(mapLongRect,&m);
	EraseRect(&m); 
	/////////////////////////////////////////////////

	
	dX = (float)WRectWidth(bounds) / (float)fNumCols;
	dY = (float)WRectHeight(bounds) / (float)fNumRows;
	
	PenNormal();
	
	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {
			cell.loLong = round(bounds.loLong + dX * col);//JLM round so there are not occasional missing lines
			cell.hiLat =  round(bounds.hiLat - dY * row);
			cell.hiLong =  round(cell.loLong + dX);
			cell.loLat =  round(cell.hiLat - dY);

			topLeftPt = GetQuickDrawPt(cell.loLong,cell.hiLat,&r,&offQuickDrawPlane);
			bottomRightPt = GetQuickDrawPt(cell.hiLong,cell.loLat,&r,&offQuickDrawPlane);
			MySetRect(&c, topLeftPt.h, topLeftPt.v, bottomRightPt.h, bottomRightPt.v);
			//MySetRect(&c, SameDifferenceX(cell.loLong),
						  //(r.bottom + r.top) - SameDifferenceY(cell.hiLat),
						 // SameDifferenceX(cell.hiLong),
						  //(r.bottom + r.top) - SameDifferenceY(cell.loLat));
			
			//color = (INDEXH(fGridHdl, row * fNumCols + col) == LT_LAND) ? LIGHTBROWN : LIGHTBLUE;
			color = (INDEXH(fGridHdl, row * fNumCols + col) <= kNumOSSMLandTypes) ? LIGHTBROWN : LIGHTBLUE;
//			RGBForeColor(&colors[color]);
			DrawCell (&c, kScreenMode, true, col, row);
		}
	
//	RGBForeColor(&colors[BLACK]);
	
	TMap::Draw(r, view);
}
/**************************************************************************************************/
void TOSSMMap::DrawCell (Rect *cellRect, long mode, Boolean bColor, long cellCol, long cellRow)
{
	char		thisBoxType, leftBoxType, topBoxType;
	Pattern		GrayPattern;

	// now erase / fill the rect if it is a land-type box
	//if (INDEXH(fGridHdl, cellRow * fNumCols + cellCol) == LT_LAND)	// 'W' indicates water
	if (INDEXH(fGridHdl, cellRow * fNumCols + cellCol) <= kNumOSSMLandTypes)	// 'W' indicates water
	{
		if ((mode == kScreenMode || mode == kPictMode) && bColor)
		{
			RGBColor	SaveBColor;

			GetForeColor (&SaveBColor);
//			Our_PmForeColor (kOMapColorInd);
//			ForeColor (BROWN);						// codewarrior mod
			RGBForeColor (&colors[LIGHTBROWN]);
			
			PaintRect (cellRect);

			RGBForeColor (&SaveBColor);
		}
		else if ((mode == kScreenMode || mode == kPictMode) && !bColor)
		{
			// initialize gray pattern bits	
//			StuffHex (&GrayPattern, kLgtGrayPat);
//			MyFillRect (cellRect, GrayPattern);
			MyFillRect (cellRect, LTGRAY_BRUSH);		// codewarrior mod
		}
		else if (mode == kPrintMode)
		{
//			StuffHex (&GrayPattern, kVeryLgtGrayPat);
//			MyFillRect (cellRect, GrayPattern);
			MyFillRect (cellRect, LTGRAY_BRUSH);		// codewarrior mod
		}
	}

	// check to see if current type matches box at right and to the bottom
	thisBoxType = leftBoxType = topBoxType = INDEXH(fGridHdl, cellRow * fNumCols + cellCol);

	if (cellCol != 0)
		leftBoxType = INDEXH(fGridHdl, cellRow * fNumCols + (cellCol - 1));

	if (cellRow != 0)
		topBoxType = INDEXH(fGridHdl, (cellRow - 1) * fNumCols + cellCol);

	if (cellCol == fNumCols - 1)
		cellRect -> right -= 1;
	if (cellRow == fNumRows - 1)
		cellRect -> bottom -= 1;

	// draw the right vertical edge if type at right is different
	if (leftBoxType != thisBoxType)
	{
		MyMoveTo (cellRect -> left, cellRect -> top);
		MyLineTo (cellRect -> left, cellRect -> bottom);
	}

	// draw top horizontal edge if type at top is different
	if (topBoxType != thisBoxType)
	{
		MyMoveTo (cellRect -> left, cellRect -> top);
		MyLineTo (cellRect -> right, cellRect -> top);
	}

	return;
}
/**************************************************************************************************/
Boolean IsOSSMMap (char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 1, strLine, 256);
		if (CountSetInString (strLine, ",") == 7 && CountSetInString (strLine, ".") == 4)
			bIsValid = true;
	}
	
	return bIsValid;
}

Boolean IsGridMap (char *path)
{
	if (IsOSSMMap(path)) 
		return true;
	
	return false;
}
