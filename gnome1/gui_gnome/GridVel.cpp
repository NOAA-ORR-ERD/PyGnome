
#include "Cross.h"
#include "GridVel.h"
#include "MapUtils.h"
#include "DagTreePD.h"
#include "DagTreeIO.h"
#include "PtCurMover.h"
#include "Contdlg.h"
//#include "NetCDFMover.h"
//#include "netcdf.h"
#include "Outils.h"

void DrawArrowHead (Point p1, Point p2, VelocityRec velocity);
void MyDrawArrow(short h0,short v0,short h1,short v1);

/////////////////////////////////////////////////
Boolean IsOilMapFile(char *path)
{

	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (strstr(strLine,"= IMAX"))
			bIsValid = true;
	}
	
	return bIsValid;
}

Boolean IsGridCurFile(char *path)
{

	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (strstr(strLine,"[GRIDCUR]"))
			bIsValid = true;
	}
	
	return bIsValid;
}

/*Boolean IsGridWindFile(char *path,short *selectedUnitsP)
{

	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined, numScanned;
	char unitsStr[64], gridwindStr[64];
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (strstr(strLine,"[GRIDWIND"))
		{
			bIsValid = true;
			*selectedUnitsP = selectedUnits;
			numScanned = sscanf(strLine,"%s%s",gridwindStr,unitsStr);
			if(numScanned != 2) { selectedUnits = kUndefined; goto done; }
			RemoveLeadingAndTrailingWhiteSpace(unitsStr);
			selectedUnits = StrToSpeedUnits(unitsStr);// note we are not supporting cm/sec in gnome
		}
	}
	
done:
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
	}
	return bIsValid;
}*/

Boolean IsOssmCurFile(char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (CountSetInString (strLine, ".") == kVelsPerLine)
			bIsValid = true;
	}
	
	return bIsValid;
}


Boolean IsRectGridFile (char *path)
{
	if(IsOilMapFile(path)) return true;
	if(IsOssmCurFile(path)) return true;
	if(IsGridCurFile(path)) return true;
	return false;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

Boolean IsTriGridFile (char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with "DAG
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (!strncmp (firstPartOfFile, "DAG", 3))
			bIsValid = true;
	}
	
	return bIsValid;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

TGridVel::TGridVel() 
{
	fGridBounds = emptyWorldRect;
}
		


/////////////////////////////////////////////////////////////////////////////
// RectGridVel 
// Velocities are defined on a rectangular grid 
/////////////////////////////////////////////////////////////////////////////

TRectGridVel::TRectGridVel(void)
{
	fGridHdl = 0;
	fNumRows = 0;
	fNumCols = 0;
}



/////////////////////////////////////////////////////////////////////////////
#define TRectGridVelREADWRITEVERSION 1
OSErr TRectGridVel::Write(BFPB *bfpb)
{
	VelocityRec velocity;
	OSErr 		err=noErr;
	long 		i, version = TRectGridVelREADWRITEVERSION;
	ClassID 	id = GetClassID ();
	long totalVels = this -> NumVelsInGridHdl();
	
	StartReadWriteSequence("TRectGridVel::Write()");

	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	///
	if (err = WriteMacValue(bfpb, fGridBounds)) return err;
	
	if (err = WriteMacValue(bfpb, fNumRows)) return err;
	if (err = WriteMacValue(bfpb, fNumCols)) return err;

	/////
	if (err = WriteMacValue(bfpb, totalVels)) return err;
	for (i = 0 ; i < totalVels ; i++) {
		velocity = INDEXH(fGridHdl, i);
		if (err = WriteMacValue(bfpb, velocity.u)) return err;
		if (err = WriteMacValue(bfpb, velocity.v)) return err;
	}

	return err;
}
/////////////////////////////////////////////////////////////////////////////
OSErr TRectGridVel::Read(BFPB *bfpb)
{
	long 		i, version;
	OSErr		err = noErr;
	VelocityRec velocity;
	ClassID 	id;
	long 		totalVels;
		
	StartReadWriteSequence("TRectGridVel::Read()");

	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { printError("Bad id in TRectGridVel::Read"); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version != TRectGridVelREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	///
	if (err = ReadMacValue(bfpb, &fGridBounds)) return err;

	if (err = ReadMacValue(bfpb, &fNumRows)) return err;
	if (err = ReadMacValue(bfpb, &fNumCols)) return err;
	///
	if (err = ReadMacValue(bfpb, &totalVels)) return err;	
	//if (fNumRows<=0 || fNumCols <= 0 || totalVels != fNumRows*fNumCols){ printSaveFileVersionError(); return -1; }
	if (fNumRows<0 || fNumCols < 0 || totalVels != fNumRows*fNumCols){ printSaveFileVersionError(); return -1; }
	//if (!err)
	if (!err && totalVels>0)	// GridCurTime stores values differently
	{
		fGridHdl = (VelocityH)_NewHandleClear(totalVels * sizeof(VelocityRec));
		if (!fGridHdl)
			{ TechError("TRectGridVel::Read()", "_NewHandleClear()", 0); return -1; }
		
		for (i = 0 ; i < totalVels ; i++) {
			if (err = ReadMacValue(bfpb, &velocity.u)) { printSaveFileVersionError(); return -1; }
			if (err = ReadMacValue(bfpb, &velocity.v)) { printSaveFileVersionError(); return -1; }
			INDEXH(fGridHdl, i) = velocity;
		}
	}
	
	return err;
}
/////////////////////////////////////////////////////////////////////////////
OSErr TRectGridVel::TextRead(char *path)
{
	if(IsOilMapFile(path))
		return this->ReadOilMapFile(path);
	else if(IsOssmCurFile(path))
		return this -> ReadOssmCurFile(path);
	else if(IsGridCurFile(path))
		return this -> ReadGridCurFile(path);
	
	return -1;
}
/////////////////////////////////////////////////

OSErr TRectGridVel::ReadOilMapFile(char *path)
{
	char s[256];
	long i, j, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	long totalVels; 
	long numLines,numScanned;
	long nSeq,nRec,nTide,nonTide;
	double dLon,dLat,oLon,oLat;
	
	if (!path) return -1;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) goto done;
	////
	// read the header
	///////////////////////
    //       100 = IMAX
	//       100 = JMAX
	// 10000 = NSEQ
	//    .01001 = DLON
	//    .01200 = DLAT
	// -89.00117 = OLON
	//  25.79745 = OLAT
	//         1 = NREC
	//         0 = NTIDE
	//         1 = NONTIDE
	//Galttst1   = NONTIDE  1
	/////////////////////////////////////////////////
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= IMAX")) { err = -2; goto done; }
	numScanned = sscanf(s,"%ld",&fNumRows);
	if(numScanned != 1 || fNumRows <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= JMAX")) { err = -2; goto done; }
	numScanned = sscanf(s,"%ld",&fNumCols);
	if(numScanned != 1 || fNumCols <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= NSEQ")) { err = -2; goto done; }
	numScanned = sscanf(s,"%ld",&nSeq);
	if(numScanned != 1 || nSeq <= 0 /*|| nSeq != fNumRows*fNumCols*/) { err = -2; goto done; } // there may not be a full set of data
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= DLON")) { err = -2; goto done; }
	numScanned = sscanf(s,lfFix("%lf"),&dLon);
	if(numScanned != 1 || dLon <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= DLAT")) { err = -2; goto done; }
	numScanned = sscanf(s,lfFix("%lf"),&dLat);
	if(numScanned != 1 || dLat <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= OLON")) { err = -2; goto done; }
	numScanned = sscanf(s,lfFix("%lf"),&oLon);
	if(numScanned != 1 ) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= OLAT")) { err = -2; goto done; }
	numScanned = sscanf(s,lfFix("%lf"),&oLat);
	if(numScanned != 1 ) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= NREC")) { err = -2; goto done; }
	numScanned = sscanf(s,"%ld",&nRec);
	if(numScanned != 1 ) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= NTIDE")) { err = -2; goto done; }
	numScanned = sscanf(s,"%ld",&nTide);
	if(numScanned != 1 ) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"= NONTIDE")) { err = -2; goto done; }
	numScanned = sscanf(s,"%ld",&nonTide);
	if(numScanned != 1 ) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); // mystery line
	//
	
	
	// check hemisphere stuff here , code goes here
	fGridBounds.loLat = round((oLat - dLat/2.0)*1000000);
	fGridBounds.hiLat = round((oLat + (fNumRows-1)*dLat + dLat/2.0)*1000000);
	fGridBounds.loLong = round((oLon - dLon/2.0)*1000000);
	fGridBounds.hiLong = round((oLon + (fNumCols-1)*dLon + dLon/2.0)*1000000);
	
	totalVels = fNumRows*fNumCols;
	//numLines = fNumRows*fNumCols;
	numLines = nSeq; // there may not be a full set of data

	fGridHdl = (VelocityH)_NewHandleClear(totalVels * sizeof(VelocityRec));
	if (!fGridHdl) { err = memFullErr; goto done; }
	////
	// read the lines
	for (i = 0 ; i < numLines ; i++) {
		long lineIndex,rowNum,colNum,mysteryItem;
		double u,v;
		long index;
		NthLineInTextOptimized(*f, line++, s, 256); 
		numScanned = sscanf(s,lfFix("%ld %ld %ld %ld %lf %lf"),&lineIndex,&rowNum,&colNum,&mysteryItem,&u,&v);
		if(numScanned != 6 
			|| rowNum <= 0 || rowNum > fNumRows
			|| colNum <= 0 || colNum > fNumCols
			)
			{ err = -1;  goto done; }
		index = (rowNum -1) * fNumCols + colNum-1;
		INDEXH(fGridHdl, index).u = u/1000.; // units ??? convert to m/s
		INDEXH(fGridHdl, index).v = v/1000.;
	}
done:
	if(f) { DisposeHandle((Handle)f); f = 0;}
	if(err)
	{
		if(fGridHdl) {DisposeHandle((Handle)fGridHdl); fGridHdl = 0;}
		if(err==memFullErr)
			 TechError("TRectGridVel::ReadOilMapFile()", "_NewHandleClear()", 0); 
		else
			printError("Unable to read OilMap file.");
	}
	return err;
}

/////////////////////////////////////////////////

OSErr TRectGridVel::ReadGridCurFile(char *path)
{
	char s[256];
	long i, j, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	long totalVels; 
	long numLines,numScanned;
	double dLon,dLat,oLon,oLat;
	double lowLon,lowLat,highLon,highLat;
	Boolean velAtCenter = 0;
	Boolean velAtCorners = 0;
	long numLinesInText,headerLines = 8;
	
	if (!path) return -1;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) goto done;
	numLinesInText = NumLinesInText(*f);
	////
	// read the header
	///////////////////////
	/////////////////////////////////////////////////
	NthLineInTextOptimized(*f, line++, s, 256); // gridcur header
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"NUMROWS")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("NUMROWS"),"%ld",&fNumRows);
	if(numScanned != 1 || fNumRows <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"NUMCOLS")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("NUMCOLS"),"%ld",&fNumCols);
	if(numScanned != 1 || fNumCols <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 

	if(s[0]=='S') // check if lat/long given as corner point and increment, and read in
	{
		if(!strstr(s,"STARTLAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("STARTLAT"),lfFix("%lf"),&oLat);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"STARTLONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("STARTLONG"),lfFix("%lf"),&oLon);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"DLAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("DLAT"),lfFix("%lf"),&dLat);
		if(numScanned != 1 || dLat <= 0) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"DLONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("DLONG"),lfFix("%lf"),&dLon);
		if(numScanned != 1 || dLon <= 0) { err = -2; goto done; }
	
		velAtCorners=true;
		//
	}
	else if(s[0]=='L') // check if lat/long bounds given, and read in
	{
		//NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"LOLAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("LOLAT"),lfFix("%lf"),&lowLat);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"HILAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("HILAT"),lfFix("%lf"),&highLat);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"LOLONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("LOLONG"),lfFix("%lf"),&lowLon);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"HILONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("HILONG"),lfFix("%lf"),&highLon);
		if(numScanned != 1 ) { err = -2; goto done; }
		
		velAtCenter=true;
	}
	else {err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); // row col u v header
	//
	
	
	// check hemisphere stuff here , code goes here
	if(velAtCenter)
	{
		fGridBounds.loLat = lowLat*1000000;
		fGridBounds.hiLat = highLat*1000000;
		fGridBounds.loLong = lowLon*1000000;
		fGridBounds.hiLong = highLon*1000000;
	}
	else if(velAtCorners)
	{
		fGridBounds.loLat = round((oLat - dLat/2.0)*1000000);
		fGridBounds.hiLat = round((oLat + (fNumRows-1)*dLat + dLat/2.0)*1000000);
		fGridBounds.loLong = round((oLon - dLon/2.0)*1000000);
		fGridBounds.hiLong = round((oLon + (fNumCols-1)*dLon + dLon/2.0)*1000000);
	}
	totalVels = fNumRows*fNumCols;
	//numLines = fNumRows*fNumCols;
	numLines = numLinesInText - headerLines;	// allows user to leave out land points within grid (or standing water)

	fGridHdl = (VelocityH)_NewHandleClear(totalVels * sizeof(VelocityRec));
	if (!fGridHdl) { err = memFullErr; goto done; }
	////
	// read the lines
	for (i = 0 ; i < numLines ; i++) {
		long rowNum,colNum;
		double u,v;
		long index;
		NthLineInTextOptimized(*f, line++, s, 256); 
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		numScanned = sscanf(s,lfFix("%ld %ld %lf %lf"),&rowNum,&colNum,&u,&v);
		if(numScanned != 4 
			|| rowNum <= 0 || rowNum > fNumRows
			|| colNum <= 0 || colNum > fNumCols
			)
			{ err = -1;  goto done; }
		index = (rowNum -1) * fNumCols + colNum-1;
		INDEXH(fGridHdl, index).u = u; // units ??? assumed m/s
		INDEXH(fGridHdl, index).v = v;
	}
done:
	if(f) { DisposeHandle((Handle)f); f = 0;}
	if(err)
	{
		if(fGridHdl) {DisposeHandle((Handle)fGridHdl); fGridHdl = 0;}
		if(err==memFullErr)
			 TechError("TRectGridVel::ReadGridCurFile()", "_NewHandleClear()", 0); 
		else
			printError("Unable to read GridCur file.");
	}
	return err;
}

/////////////////////////////////////////////////
OSErr TRectGridVel::ReadOssmCurFile(char *path)
{
	char s[256];
	long i, j, line = 0;
	float component; // loLong, loLat, hiLong, hiLat
	CHARH f = 0;
	OSErr err = 0;
	long totalVels; 
	long numLines;
	
	fNumRows = kOCurHeight;
	fNumCols = kOCurWidth;
	totalVels = fNumRows*fNumCols;
	
	if (!path) return -1;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) goto done;
	
	fGridHdl = (VelocityH)_NewHandleClear(totalVels * sizeof(VelocityRec));
	if (!fGridHdl) { err = memFullErr; goto done; }
	
	// read the first half of the file containing the u components
	numLines = totalVels/kVelsPerLine;
	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, line++, s, 256); // 10 u values
		for (j = 0 ; j < kVelsPerLine ; j++) {
			if (sscanf(&s[j * 8], "%f", &component) != 1) { err = -1;  goto done; }
			INDEXH(fGridHdl, i * kVelsPerLine + j).u = component;
		}
	}

	// read the second half of the file containing the v components

	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, line++, s, 256); // 10 u values
		for (j = 0 ; j < kVelsPerLine ; j++) {
			if (sscanf(&s[j * 8], "%f", &component) != 1) { err = -1; goto done; }
			INDEXH(fGridHdl, i * kVelsPerLine + j).v = component;
		}
	}
	
done:
	if(f) { DisposeHandle((Handle)f); f =0;}
	if(err)
	{
		if(fGridHdl) {DisposeHandle((Handle)fGridHdl); fGridHdl = 0;}
		if(err==memFullErr)
			 TechError("TRectGridVel::ReadOssmCurFile()", "_NewHandleClear()", 0); 
		else
			printError("Unable to read OSSM CUR file.");
	}
	return err;
}

void TRectGridVel::Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,
						double arrowScale, double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor) 
{
	short row, col, pixX, pixY;
	long dLong, dLat;
	float inchesX, inchesY;
	Point p, p2;
	Rect c;
	WorldPoint wp;
	WorldRect  boundsRect;
	VelocityRec velocity;
	Rect	newCATSgridRect = {0, 0, fNumRows - 1, fNumCols - 1};
	Boolean offQuickDrawPlane = false;
	
	if (!bDrawArrows && !bDrawGrid) return;
	
	//p.h = SameDifferenceX(refP.pLong);
	//p.v = (r.bottom + r.top) - SameDifferenceY(refP.pLat);
	p = GetQuickDrawPt(refP.pLong, refP.pLat, &r, &offQuickDrawPlane);
	
	// draw the reference point
	RGBForeColor(&colors[BLUE]);
	MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
	PaintRect(&c);
	RGBForeColor(&colors[BLACK]);
		
	dLong = (WRectWidth(fGridBounds) / fNumCols) / 2;
	dLat = (WRectHeight(fGridBounds) / fNumRows) / 2;
	RGBForeColor(&colors[PURPLE]);

	boundsRect = fGridBounds;
	InsetWRect (&boundsRect, dLong, dLat);

	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {
			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newCATSgridRect, boundsRect);
			velocity = GetPatValue(wp);
			//p.h = SameDifferenceX(wp.pLong);
			//p.v = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bDrawGrid) PaintRect(&c);
				
			if (bDrawArrows && (velocity.u != 0 || velocity.v != 0))
			{
				RGBForeColor(&arrowColor);
				inchesX = (velocity.u * refScale) / arrowScale;
				inchesY = (velocity.v * refScale) / arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
				//DrawArrowHead (p, p2, velocity);
				//DrawArrowHead(p2, velocity);
				RGBForeColor(&colors[PURPLE]);
			}
		}
		
	RGBForeColor(&colors[BLACK]);
}


/////////////////////////////////////////////////


OSErr TTriGridVel::TextRead(char *path)
{
	OSErr err=-1;
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	char s[256],errmsg[256];
	long i, line = 0;
	CHARH fileBufH = 0;
	LongPointHdl ptsH;	
	FLOATH depthsH = 0;
	tree.treeHdl = 0;
	if (!path) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &fileBufH))
	{ 
		printError("Invalid Triangle Grid file.");
		goto done; 
	}
	
	_HLock((Handle)fileBufH); // JLM 8/4/99
	
	// Read header
	line = 0;
	NthLineInTextOptimized(*fileBufH,line++, s, 256);

	if(strncmp(s,"DAG 1.0",strlen("DAG 1.0")) != 0)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTVertices(fileBufH,&line,&pts,&depthsH,errmsg))goto done;
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTTopology(fileBufH,&line,&topo,&velH,errmsg))goto done;
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTIndexedDagTree(fileBufH,&line,&tree,errmsg))
	{
		// allow user to leave out the dagtree
		char errmsg[256];
		errmsg[0]=0;
		err = 0;
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Dag Tree");
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		DisplayMessage(0);
		if (errmsg[0])	
		err = -1; // for now we require TIndexedDagTree
		// code goes here, support Galt style ??
		if(err) goto done;
	}
	//goto done;
	MySpinCursor(); // JLM 8/4/99
	
	fDagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches);
	if(!fDagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	SetBathymetry(depthsH);
	//if(depthsH) {DisposeHandle((Handle)depthsH); depthsH=0;}	
	/////////////////////////////////////////////////
	/// figure out the bounds
	ptsH = fDagTree->GetPointsHdl();
	if(ptsH) 
	{
		long numPoints, i;
		LongPoint	thisLPoint;
	
		numPoints = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
		if(numPoints > 0)
		{
			WorldPoint  wp;
			WorldRect bounds = voidWorldRect;
			for(i=0;i<numPoints;i++)
			{
				thisLPoint = (*ptsH)[i];
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
			fGridBounds = bounds;
		}
	}
	/////////////////////////////////////////////////
	
	
	err = noErr;

done:

	if(fileBufH) 
	{
		_HUnlock((Handle)fileBufH); // JLM 8/4/99
		DisposeHandle((Handle)fileBufH); 
		fileBufH = 0;
	}

	if(err)
	{
		TechError("TTriGridVel::TextRead(char* path)", errmsg, 0); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(depthsH) {DisposeHandle((Handle)depthsH); depthsH=0;}	// shouldn't exist
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(fDagTree)
		{
			delete fDagTree;
			fDagTree = 0;
		}
	}
	return err;
}
////////////////////////////////////////////////////////////////////////////////////
//#define TTriGridVelREADWRITEVERSION 2  // updated to 2 for Read/WriteTopology haveVelocityHdl variable
#define TTriGridVelREADWRITEVERSION 3  // updated to 3 for bathymetry
OSErr TTriGridVel::Write(BFPB *bfpb)
{
	VelocityRec velocity;
	OSErr 		err = noErr;
	long 		i, version = TTriGridVelREADWRITEVERSION, numDepths = 0;
	ClassID 	id = GetClassID ();
	float 		val;
	char 		errStr[256] = "";
	
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;

	if (err = WriteMacValue(bfpb, fGridBounds)) return err;
	
	if(err = WriteVertices(bfpb,fDagTree -> fPtsH,errStr))goto done;
	if(err = WriteTopology(bfpb,fDagTree -> fTopH,fDagTree -> fVelH, errStr))goto done;
	if(err = WriteIndexedDagTree(bfpb,fDagTree -> fTreeH,errStr))goto done;

	if (fBathymetryH) numDepths = _GetHandleSize((Handle)fBathymetryH)/sizeof(**fBathymetryH);
	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fBathymetryH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
done:

	if(err)
		TechError("TTriGridVel::Write(char* path)", errStr, 0); 

	return err;
}
////////////////////////////////////////////////////////////////////////////////////
OSErr TTriGridVel::Read(BFPB *bfpb)
{
	OSErr err = noErr;
	char errmsg[256];
	long numBranches, numDepths;
	float val;
	TopologyHdl topH=0;
	LongPointHdl ptsH=0;
	VelocityFH velH = 0;
	DAGHdl treeH = 0;
	long i, version;
	ClassID id;
	
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { printError("Bad id in TTriGridVel::Read"); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != TTriGridVelREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (version < 2 || version > TTriGridVelREADWRITEVERSION) { printSaveFileVersionError(); return -1; } // broke save files on version 2

	if (err = ReadMacValue(bfpb, &fGridBounds)) return err;

	if(err = ReadVertices(bfpb,&ptsH,errmsg))goto done;
	if(err = ReadTopology(bfpb,&topH,&velH,errmsg))goto done;
	if(err = ReadIndexedDagTree(bfpb,&treeH,errmsg)) goto done;

	numBranches = _GetHandleSize ((Handle) treeH) / sizeof (DAG);
	fDagTree = new TDagTree(ptsH,topH,treeH,velH,numBranches);
	if(!fDagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	if (version > 2)
	{
		if (err = ReadMacValue(bfpb, &numDepths)) goto done;
		if (numDepths > 0)
		{
			fBathymetryH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
			if (!fBathymetryH)
				{ TechError("TTriGridVel::Read()", "_NewHandleClear()", 0); goto done; }
		}
		
		for (i = 0 ; i < numDepths ; i++) {
			if (err = ReadMacValue(bfpb, &val)) goto done;
			INDEXH(fBathymetryH, i) = val;
		}
	}
	// fDagTree is now responsible for these handles
	ptsH = 0;
	topH = 0;
	velH = 0;
	treeH = 0;

done:
	if(err)
	{
		TechError("TTriGridVel::Read(char* path)", errmsg, 0); 
		if(ptsH) {DisposeHandle((Handle)ptsH); ptsH=0;}
		if(topH) {DisposeHandle((Handle)topH); topH=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(treeH) {DisposeHandle((Handle)treeH); treeH=0;}
		if(fDagTree)
		{
			delete fDagTree;
			fDagTree = 0;
		}
		if(fBathymetryH) {DisposeHandle((Handle)fBathymetryH); fBathymetryH=0;}
	}
	return err;
}

void TTriGridVel::Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,double arrowScale,
					   double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor)
{
	short row, col, pixX, pixY;
	float inchesX, inchesY;
	Point p, p2;
	Rect c;
	WorldPoint wp;
	VelocityRec velocity;
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	LongPoint wp1,wp2,wp3;
	long i,numTri;
	Boolean offQuickDrawPlane = false;
	

	if(fDagTree == 0)return;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);

	//p.h = SameDifferenceX(refP.pLong);
	//p.v = (r.bottom + r.top) - SameDifferenceY(refP.pLat);
	p = GetQuickDrawPt(refP.pLong, refP.pLat, &r, &offQuickDrawPlane);
	
	// draw the reference point
	if (!offQuickDrawPlane)
	{
		RGBForeColor(&colors[BLUE]);
		MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
		PaintRect(&c);
	}
	//RGBForeColor(&colors[BLACK]);
		
	RGBForeColor(&colors[PURPLE]);


	for (i = 0 ; i< numTri; i++)
	{
		if (bDrawArrows)
		{
			RGBForeColor(&arrowColor);
			wp1 = (*ptsH)[(*topH)[i].vertex1];
			wp2 = (*ptsH)[(*topH)[i].vertex2];
			wp3 = (*ptsH)[(*topH)[i].vertex3];
	
			wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
			wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
			velocity = GetPatValue(wp);

			// if want to see currents below surface
			/*if (arrowDepth > 1 && bApplyLogProfile)	// start the profile after the first meter
			{
				double scaleFactor=1., depthAtPoint;
				//depthAtPoint = ((TTriGridVel*)fGrid)->GetDepthAtPoint(p.p);
				depthAtPoint = this->GetDepthAtPoint(wp);
				if (arrowDepth >= depthAtPoint)scaleFactor = 0.;
				else if (depthAtPoint > 0) scaleFactor = 1. - log(arrowDepth)/log(depthAtPoint);
				velocity.u *= scaleFactor;
				velocity.v *= scaleFactor;
			}*/
			
			//p.h = SameDifferenceX(wp.pLong);
			//p.v = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
//			PaintRect(&c);
							
			if (velocity.u != 0 || velocity.v != 0) 
			{
				inchesX = (velocity.u * refScale) / arrowScale;
				inchesY = (velocity.v * refScale) / arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
	
				//DrawArrowHead (p, p2, velocity);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
		}
		
		if (bDrawGrid) 
		{
			RGBForeColor(&colors[PURPLE]);
			DrawTriangle(&r,i,FALSE);	// don't fill triangles
		}
	}
	RGBForeColor(&colors[BLACK]);

	return;
}

void TTriGridVel::DrawCurvGridPts(Rect r, WorldRect view)
{
	Point p;
	Rect c;
	LongPointHdl ptsH ;
	long i,numPts;
	Boolean offQuickDrawPlane = false;

	if(fDagTree == 0)return;

	ptsH = fDagTree->GetPointsHdl();
	numPts = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);

	RGBForeColor(&colors[PURPLE]);

	for (i = 0 ; i< numPts; i++)
	{
		p = GetQuickDrawPt((*ptsH)[i].h,(*ptsH)[i].v,&r,&offQuickDrawPlane);
		MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
		PaintRect(&c);
	}

	RGBForeColor(&colors[BLACK]);

	return;
}

void TTriGridVel::DrawBitMapTriangles(Rect r)
{
	TopologyHdl topH ;
	long i,numTri;

	if(fDagTree == 0)return;

	topH = fDagTree->GetTopologyHdl();
	numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);

	RGBForeColor(&colors[BLACK]);
	for (i = 0 ; i< numTri; i++)
	{		
		DrawTriangle(&r,i,TRUE);	// fill triangles	
	}

	return;
}

void TTriGridVel::DrawTriangle(Rect *r,long triNum,Boolean fillTriangle)
{
#ifdef IBM
	POINT points[4];
#else
	PolyHandle poly;
#endif
	long v1,v2,v3;
	Point pt1,pt2,pt3;
	TopologyHdl topH = fDagTree->GetTopologyHdl();
	LongPointHdl ptsH = fDagTree->GetPointsHdl();
	Boolean offQuickDrawPlane;
	
	v1 = (*topH)[triNum].vertex1;
	v2 = (*topH)[triNum].vertex2;
	v3 = (*topH)[triNum].vertex3;

	
	pt1 = GetQuickDrawPt((*ptsH)[v1].h,(*ptsH)[v1].v,r,&offQuickDrawPlane);
	pt2 = GetQuickDrawPt((*ptsH)[v2].h,(*ptsH)[v2].v,r,&offQuickDrawPlane);
	pt3 = GetQuickDrawPt((*ptsH)[v3].h,(*ptsH)[v3].v,r,&offQuickDrawPlane);
	
	PenMode(patCopy);
#ifdef MAC
		poly = OpenPoly();
		MyMoveTo(pt1.h,pt1.v);
		MyLineTo(pt2.h,pt2.v);
		MyLineTo(pt3.h,pt3.v);
		MyLineTo(pt1.h,pt1.v);
		ClosePoly();
	
		if(fillTriangle)
			PaintPoly(poly);
		
		FramePoly(poly);
		
		KillPoly(poly);
#else
		points[0] = MakePOINT(pt1.h,pt1.v);
		points[1] = MakePOINT(pt2.h,pt2.v);
		points[2] = MakePOINT(pt3.h,pt3.v);
		points[3] = MakePOINT(pt1.h,pt1.v);
	
	
		if(fillTriangle)
			Polygon(currentHDC,points,4); // code goes here

		Polyline(currentHDC,points,4);
		
#endif
}

////////////////////////////////////////////////////////////////////////////////////
TTriGridVel3D::TTriGridVel3D() 
{
	fDepthsH=0; 
	fDepthContoursH=0;
	fTriSelected = 0;
	fPtsSelected = 0;
	fOilConcHdl = 0;
	fMaxLayerDataHdl = 0;
	fTriAreaHdl = 0;
	fDosageHdl = 0;
	bShowSelectedTriangles = true;
	//fPercentileForMaxConcentration = .9;
	fPercentileForMaxConcentration = 1.;	// make user decide if they want to fudge this
	bCalculateDosage = false;
	bShowDosage = false;
	fDosageThreshold = .2;
	fMaxTri = -1;
	bShowMaxTri = false;
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
}


//#define TTriGridVel3DREADWRITEVERSION 1  
#define TTriGridVel3DREADWRITEVERSION 2  // added depth contours 7/21/03
OSErr TTriGridVel3D::Write(BFPB *bfpb)
{
	OSErr 		err=noErr;
	long 		i, version = TTriGridVel3DREADWRITEVERSION;
	ClassID 	id = GetClassID ();
	char 		errStr[256]="";
	long 		numDepths = GetNumDepths(), ntri = GetNumTriangles(), triSel;
	//long 		npts = GetNumPoints(), ptSel;
	long		numDepthContours = GetNumDepthContours();
	long		numOutputDataValues = GetNumOutputDataValues();
	float 		val;
	double 		val2;
	
	if (err = TTriGridVel::Write(bfpb)) return err;
	
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;

	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fDepthsH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
	if (err = WriteMacValue(bfpb, numOutputDataValues)) goto done;
	for (i=0;i<numOutputDataValues;i++)
	{
		if (err = WriteMacValue(bfpb, (*fOilConcHdl)[i].avOilConcOverSelectedTri)) goto done;
		if (err = WriteMacValue(bfpb, (*fOilConcHdl)[i].maxOilConcOverSelectedTri)) goto done;
		if (err = WriteMacValue(bfpb, (*fOilConcHdl)[i].time)) goto done;
	}
	// fMaxLayerDataHdl
	if (err = WriteMacValue(bfpb, numDepthContours)) goto done;
	for (i=0;i<numDepthContours;i++)
	{
		val2 = INDEXH(fDepthContoursH,i);
		if (err = WriteMacValue(bfpb, val2)) goto done;
	}
	if (fTriSelected)
	{
		if (err = WriteMacValue(bfpb, ntri)) goto done;
		for (i=0;i<ntri;i++)
		{
			triSel = INDEXH(fTriSelected,i);
			if (err = WriteMacValue(bfpb, triSel)) goto done;
		}
	}
	else
	{
		ntri = 0;
		if (err = WriteMacValue(bfpb, ntri)) goto done;
	}
	/*if (fPtsSelected)
	{
		if (err = WriteMacValue(bfpb, ntri)) goto done;
		for (i=0;i<npts;i++)
		{
			ptSel = INDEXH(fPtsSelected,i);
			if (err = WriteMacValue(bfpb, ptSel)) goto done;
		}
	}
	else
	{
		npts = 0;
		if (err = WriteMacValue(bfpb, npts)) goto done;
	}*/
	// code goes here fDosageHdl, bShowSelectedTriangles

done:

	if(err)
		TechError("TTriGridVel3D::Write(char* path)", errStr, 0); 

	return err;
}
////////////////////////////////////////////////////////////////////////////////////
OSErr TTriGridVel3D::Read(BFPB *bfpb)
{
	OSErr err=noErr;
	char errmsg[256];
	long	numDepths, numDepthContours, ntri, triSel, numOutputValues;
	//long	npts, ptSel;
	float val;
	double val2,avConc,maxConc;
	Seconds time;
	long 		version, i;
	ClassID 	id;

	
	if (err = TTriGridVel::Read(bfpb)) return err;
	
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { printError("Bad id in TTriGridVel3D::Read"); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != TTriGridVel3DREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (version <1 || version > TTriGridVel3DREADWRITEVERSION) { printSaveFileVersionError(); return -1; }

	if (err = ReadMacValue(bfpb, &numDepths)) goto done;
	
	if (numDepths>0)
	{	
		fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
		if (!fDepthsH)
			{ TechError("TTriGridVel3D::Read()", "_NewHandleClear()", 0); goto done; }
		
		for (i = 0 ; i < numDepths ; i++) {
			if (err = ReadMacValue(bfpb, &val)) goto done;
			INDEXH(fDepthsH, i) = val;
		}
	}
	if (version>1)
	{
		if (err = ReadMacValue(bfpb, &numOutputValues)) goto done;	
		if (numOutputValues>0)
		{
			fOilConcHdl = (outputDataHdl)_NewHandleClear(sizeof(outputData)*numOutputValues);
			if (!fOilConcHdl)
				{ TechError("TTriGridVel3D::Read()", "_NewHandleClear()", 0); goto done; }
		}
		
		for (i = 0 ; i < numOutputValues ; i++) {
			if (err = ReadMacValue(bfpb, &avConc)) goto done;
			if (err = ReadMacValue(bfpb, &maxConc)) goto done;
			if (err = ReadMacValue(bfpb, &time)) goto done;
			INDEXH(fOilConcHdl,i).avOilConcOverSelectedTri = avConc;
			INDEXH(fOilConcHdl,i).maxOilConcOverSelectedTri = maxConc;
			INDEXH(fOilConcHdl,i).time = time;
		}
		
		if (err = ReadMacValue(bfpb, &numDepthContours)) goto done;	
		if (numDepthContours>0)
		{
			fDepthContoursH = (DOUBLEH)_NewHandleClear(sizeof(double)*numDepthContours);
			if (!fDepthContoursH)
				{ TechError("TTriGridVel3D::Read()", "_NewHandleClear()", 0); goto done; }
		}
		
		for (i = 0 ; i < numDepthContours ; i++) {
			if (err = ReadMacValue(bfpb, &val2)) goto done;
			INDEXH(fDepthContoursH, i) = val2;
		}
	}
	// fMaxLayerDataHdl
	if (err = ReadMacValue(bfpb, &ntri)) goto done;	
	if (ntri==0)
	{
		fTriSelected = nil;
		//goto done;
	}
	else
	{
		fTriSelected = (Boolean**)_NewHandleClear(sizeof(Boolean)*ntri);
		if (!fTriSelected)
			{ TechError("TTriGridVel3D::Read()", "_NewHandleClear()", 0); goto done; }
		
		for (i = 0 ; i < ntri ; i++) {
			if (err = ReadMacValue(bfpb, &triSel)) goto done;
			INDEXH(fTriSelected, i) = triSel;
		}
	}
	/*if (err = ReadMacValue(bfpb, &npts)) goto done;	
	if (npts==0)
	{
		fPtsSelected = nil;
		//goto done;
	}
	else
	{
		fPtsSelected = (Boolean**)_NewHandleClear(sizeof(Boolean)*npts);
		if (!fPtsSelected)
			{ TechError("TTriGridVel3D::Read()", "_NewHandleClear()", 0); goto done; }
		
		for (i = 0 ; i < npts ; i++) {
			if (err = ReadMacValue(bfpb, &ptSel)) goto done;
			INDEXH(fPtsSelected, i) = ptSel;
		}
	}*/
	// code goes here fDosageHdl
	//SetShowSelectedTriangles(bShowSelectedTriangles);

done:
	if(err)
	{
		TechError("TTriGridVel3D::Read(char* path)", errmsg, 0); 
		if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
		if(fTriSelected) {DisposeHandle((Handle)fTriSelected); fTriSelected=0;}
		if(fPtsSelected) {DisposeHandle((Handle)fPtsSelected); fPtsSelected=0;}
	}
	return err;
}


long TTriGridVel3D::GetNumDepthContours(void)
 {
	 long numContourValues = 0;
	 if (fDepthContoursH) numContourValues = _GetHandleSize((Handle)fDepthContoursH)/sizeof(**fDepthContoursH);
	 
	 return numContourValues;
 }

Boolean **TTriGridVel3D::GetPtsSelection(Boolean initHdl) 
{
	if (fPtsSelected)	
		return fPtsSelected;
	else if (initHdl)
	{
		long i;
		long npts = GetNumPoints();
		fPtsSelected =(Boolean **)_NewHandle(sizeof(Boolean)*npts);
		if(fPtsSelected)
		{
			for(i=0; i < npts; i++)
			{
				(*fPtsSelected)[i] = false;
			}
			return fPtsSelected;
		}
	}
	return nil;
}

void TTriGridVel3D::ClearTriSelection()
{
	if(fTriSelected) 
	{
		DisposeHandle((Handle)fTriSelected); 
		fTriSelected = 0;
	}
}

void TTriGridVel3D::ClearPtsSelection()
{
	if(fPtsSelected) 
	{
		DisposeHandle((Handle)fPtsSelected); 
		fPtsSelected = 0;
	}
}



OSErr TTriGridVel3D::ExportAllDataAtSetTimes(char* path)
{	// av, max concentration, area at each level, budget table, at 3,12,24,48,72 hours after dispersion or after spill
	OSErr err = 0;
	DateTimeRec dateTime;
	Seconds time;
	double avConc,maxConc;
	double triArea;
	outputData data;
	char buffer[512],concStr1[64],concStr2[64],timeStr[128],headerStr[128];
	long i,j,numTriAreaTimes,k=0,n,numLevels,numConcOutputValues = 0,numTriAreaOutputValues = 0;
	long numZeros, numBudgetTableOutputValues = 0, numTimeStepsPerOutput, numSpills = 0;
	char triAreaStr[64],indexStr[24],concStr[64];
	TLEList *thisLEList;
	double amttotal,amtevap,amtbeached,amtoffmap,amtfloating,amtreleased,amtdispersed,amtremoved=0;
	char unitsStr[64],massStr[64];
	char amtEvapStr[64],amtDispStr[64],amtBeachedStr[64],amtRelStr[64],amtFloatStr[64],amtOffStr[64],amtRemStr[64];
	double totalMass;
	BudgetTableDataH budgetTableH=0, totalBudgetTableH=0;
	BudgetTableData budgetTable;
	DispersionRec dispInfo;
	Seconds disperseTime, timeStep = model->GetTimeStep(), startTime = model->GetStartTime(), spillStartTime;
	short massUnits, totalBudgetMassUnits;
	Boolean someSpillIsSubsurface = false;
	BFPB bfpb;
	AdiosInfoRecH adiosBudgetTable;
	PtCurMap *map = GetPtCurMap();
	if (!map) return -1;


	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }

	// get the totals
	totalBudgetMassUnits = model->GetMassUnitsForTotals();	// use first LE set
	GetLeUnitsStr(unitsStr,totalBudgetMassUnits);
	model->GetTotalAmountStatistics(totalBudgetMassUnits,&amttotal,&amtreleased,&amtevap,&amtdispersed,&amtbeached,&amtoffmap,&amtfloating,&amtremoved);
	StringWithoutTrailingZeros(massStr,amttotal,3);
	strcpy(buffer,"Total Amount Spilled - ");
	strcat(buffer,massStr);
	strcat(buffer," ");
	strcat(buffer,unitsStr);
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;

	model->GetTotalBudgetTableHdl(totalBudgetMassUnits,&totalBudgetTableH);
	someSpillIsSubsurface = map->ThereIsADispersedSpill();

	// need to get disperse time, model time step
	for (i = 0, n = model->LESetsList->GetItemCount() ; i < n ; i++) {
		model->LESetsList->GetListItem((Ptr)&thisLEList, i);
		if(thisLEList -> GetLEType() == UNCERTAINTY_LE ) continue;
		numSpills++;

		dispInfo = ((TOLEList *)thisLEList) -> GetDispersionInfo();
		adiosBudgetTable = ((TOLEList *)thisLEList) -> GetAdiosInfo();
		spillStartTime = ((TOLEList *)thisLEList) -> GetSpillStartTime();
		disperseTime = spillStartTime + dispInfo.timeToDisperse;	// use first disperse time for each spill
		totalMass = ((TOLEList *)thisLEList)->fSetSummary.totalMass;
		GetLeUnitsStr(unitsStr,((TOLEList *)thisLEList)->fSetSummary.massUnits);

		StringWithoutTrailingZeros(massStr,totalMass,3);
		sprintf(headerStr,"Spill %ld : Amount Spilled - ",numSpills);
		strcpy(buffer,headerStr);
		strcat(buffer,massStr);
		strcat(buffer," ");
		strcat(buffer,unitsStr);
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		strcpy(buffer,"Spill start time - ");
		Secs2DateString2(spillStartTime,timeStr);
		strcat(buffer,timeStr);
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		if (dispInfo.bDisperseOil)
		{
			strcpy(buffer,"Dispersion time - ");
			Secs2DateString2(disperseTime,timeStr);
			strcat(buffer,timeStr);
			if (adiosBudgetTable) {strcat(buffer,NEWLINESTRING); strcat(buffer,"Natural Dispersion from ADIOS");}
		}
		else if (adiosBudgetTable)
			strcpy(buffer,"Natural Dispersion from ADIOS");
		else if ((*(TOLEList*)thisLEList).fSetSummary.z > 0)
			strcpy(buffer,"Bottom release");
		else
			strcpy(buffer,"Not Dispersed");
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;

	}
	numLevels = map->GetNumContourLevels();
	numConcOutputValues = GetNumOutputDataValues();

	// Write out the times and values, add header line
	//strcpy(buffer,"Hr\t\tRel\t   Float\t   Evap\t   Disp\t   Beach\t   OffMap");
	//strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap");
	strcpy(buffer,"Hr\tRel\tFloat\tEvap\tDisp\tBeach\tOffMap\tRemoved");

	if (someSpillIsSubsurface)
	//if (dispInfo.bDisperseOil || (*(TOLEList*)thisLEList).fSetSummary.z > 0)
	{
		strcat(buffer,"\t");
		strcat(buffer,"AvConc\tMaxConc");
	
		for (j=0; j < numLevels; j++)
		{
			strcpy(concStr,"C");
			sprintf(indexStr,"ld",j+1);
			strcat(concStr,indexStr);
			//strcat(buffer,"	  ");
			strcat(buffer,"\t");
			strcat(buffer,concStr);
		}
	}
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	if (fTriAreaHdl)
		numTriAreaOutputValues = _GetHandleSize((Handle)fTriAreaHdl)/sizeof(double);
	//numBudgetTableOutputValues = _GetHandleSize((Handle)budgetTableH)/sizeof(BudgetTableData);
	if (totalBudgetTableH)
		numBudgetTableOutputValues = _GetHandleSize((Handle)totalBudgetTableH)/sizeof(BudgetTableData);
	numTriAreaTimes = numTriAreaOutputValues / numLevels;
	numZeros = numBudgetTableOutputValues - numTriAreaTimes;
	numTimeStepsPerOutput = model->LEDumpInterval / timeStep;
	if (model->LEDumpInterval % timeStep)
	{
		printError("Output times not divisible by time step. Can't combine concentration data with budget and area.");
		err = -1;
		goto done;
	}
	if (numZeros < 0) // something went wrong
	{
		printError("Triangle area times don't match budget table times");
		err = -1;
		goto done;
	}
	for(i = 0; i< numBudgetTableOutputValues;i++)
	{
		long triAreaIndex = 0;
		//budgetTable = INDEXH(budgetTableH,i);
		budgetTable = INDEXH(totalBudgetTableH,i);
		time = budgetTable.timeAfterSpill;
		amtreleased = budgetTable.amountReleased;
		amtfloating = budgetTable.amountFloating;
		amtdispersed = budgetTable.amountDispersed;
		amtevap = budgetTable.amountEvaporated;
		amtbeached = budgetTable.amountBeached;
		amtoffmap = budgetTable.amountOffMap;
		amtremoved = budgetTable.amountRemoved;

		StringWithoutTrailingZeros(timeStr,time/model->LEDumpInterval,3);
		StringWithoutTrailingZeros(amtEvapStr,amtevap,3);
		StringWithoutTrailingZeros(amtDispStr,amtdispersed,3);
		StringWithoutTrailingZeros(amtFloatStr,amtfloating,3);
		StringWithoutTrailingZeros(amtRelStr,amtreleased,3);
		StringWithoutTrailingZeros(amtBeachedStr,amtbeached,3);
		StringWithoutTrailingZeros(amtOffStr,amtoffmap,3);
		StringWithoutTrailingZeros(amtRemStr,amtremoved,3);
		/////
		strcpy(buffer,timeStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRelStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtFloatStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtEvapStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtDispStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtBeachedStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtOffStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,amtRemStr);
		//strcat(buffer,NEWLINESTRING);

		if (!someSpillIsSubsurface)
		//if (!dispInfo.bDisperseOil && !(*(TOLEList*)thisLEList).fSetSummary.z > 0)
		//if (numConcOutputValues==0 && numTriAreaOutputValues==0)	// non-dispersed oil
		{
			strcat(buffer,NEWLINESTRING);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			continue;
		}
		avConc = 0;
		maxConc = 0;
		if (i >= numZeros)
		{
			long index = ((i-numZeros+1)*numTimeStepsPerOutput) - 1;
			if (index < numConcOutputValues) 
			{
				data = INDEXH(fOilConcHdl,index);
				//time = data.time;
				avConc = data.avOilConcOverSelectedTri;
				maxConc = data.maxOilConcOverSelectedTri;
				//SecondsToDate(time,&dateTime); // convert to 2 digit year?
				//sprintf(timeStr, "%02hd,%02hd,%02hd,%02hd,%02hd",
					  // dateTime.day, dateTime.month, dateTime.year,
					  // dateTime.hour, dateTime.minute);
			}
		}
		StringWithoutTrailingZeros(concStr1,avConc,3);
		StringWithoutTrailingZeros(concStr2,maxConc,3);
		/////
		//strcpy(buffer,timeStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,concStr1);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,concStr2);

		if(k>=numTriAreaOutputValues) break;
		for (j=0; j < numLevels; j++)
		{
			if (i < numZeros) triArea = 0;
			else
			{
				triArea = INDEXH(fTriAreaHdl,k);
				k++;
			}
			MyNumToStr(triArea,triAreaStr);
			//strcat(buffer,"		");
			strcat(buffer,"\t");
			strcat(buffer,triAreaStr);
		}
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

done:
	// 
	FSCloseBuf(&bfpb);
	if (totalBudgetTableH) {DisposeHandle((Handle)totalBudgetTableH); totalBudgetTableH = 0;}
	if(err) {	
		// the user has already been told there was a problem
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
// code for selecting triangles using arrow tool and lasso tool

void TTriGridVel3D::DeselectAll()
{  
	long i,ntri = GetNumTriangles();
	GrafPtr		savePort;
	Rect saveClip,r=MapDrawingRect();
	
	if (!fTriSelected) return;
	
	GetPortGrafPtr(&savePort);
	//MySetPort(mapWindow);
	SetPortWindowPort(mapWindow);

	saveClip = MyClipRect(MapDrawingRect()); 
	//PrepareToDraw(MapDrawingRect(),settings.currentView,0,0);
	
	for(i = 0; i < ntri; i++)
	{
		if((*fTriSelected)[i])
		{
			DrawTriangle3D(&r,i,false,true);
			(*fTriSelected)[i] = false;
		}
	}
	MyClipRect(saveClip); 
	//MySetPort(savePort);
	SetPortGrafPort(savePort);
	//ClearTriSelection();
}

void TTriGridVel3D::DeselectAllPoints()
{  
	long i,npts = GetNumPoints();
	GrafPtr		savePort;
	Rect saveClip,r=MapDrawingRect();
	
	if (!fPtsSelected) return;
	
	GetPortGrafPtr(&savePort);
	//MySetPort(mapWindow);
	SetPortWindowPort(mapWindow);

	saveClip = MyClipRect(MapDrawingRect()); 
	//PrepareToDraw(MapDrawingRect(),settings.currentView,0,0);
	
	for(i=npts-1; i >=0; i--)
	{
		if((*fPtsSelected)[i])
		{
			//DrawPointAt(&r,i,POINTDESELECTFLAG);
			//DrawPointAt(&r,i,0);
			(*fPtsSelected)[i] = false;
		}
	}
	
	MyClipRect(saveClip); 
	//MySetPort(savePort);
	SetPortGrafPort(savePort);
	//ClearPtsSelection();
}

void TTriGridVel3D::ToggleTriSelection(long i)
{
	(*fTriSelected)[i] = !(*fTriSelected)[i];
}

void TTriGridVel3D::TogglePointSelection(long i)
{
	(*fPtsSelected)[i] = !(*fPtsSelected)[i];
}

long TTriGridVel3D::FindTriNearClick(Point where)
{
	WorldPoint wp = ScreenToWorldPoint(where, MapDrawingRect(), settings.currentView);

	LongPoint lp;
	lp.h = wp.pLong;
	lp.v = wp.pLat;

	if(!fDagTree) return -1;
	long trinum = fDagTree->WhatTriAmIIn(lp);
	return trinum;
}
Boolean TTriGridVel3D::PointsSelected()
{
	long n = GetNumPoints();
	long i;
	Boolean retval = false;
	if(fPtsSelected)
	{
		for(i=0;i<n;i++)
		{
			if((*fPtsSelected)[i])
			{
				retval =true;
				break;
			}
		}
	}
	if(retval == false)
	{
		printError("No points are selected.");
	}
	return retval;
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////

void GetStringRect(char text[],short h, short v, Rect *r)
{
	short lineHt;
	FontInfo Finfo;

	GetFontInfo(&Finfo);

	lineHt = Finfo.ascent + Finfo.descent + Finfo.leading;
	r->bottom = v+2;
	r->left = h-1;
#ifdef IBM
	r->top = v - lineHt - 4;
#else
	r->top = v - lineHt - 1;
#endif
	r->right = h + stringwidth(text)+2;
}

void GetTextOffsets(char s[], short *h, short *v)
{	// original CATS code has direction parameter too
	FontInfo Finfo;
	short width;
	
	GetFontInfo(&Finfo);
	width = stringwidth(s);
	*h -= width/2;
	*v += Finfo.ascent/2;
}

void TTriGridVel::DrawContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	Rect legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],text[30];
	long 		i,numLevels;
	double	minLevel, maxLevel;
	double 	value;
	
	SetRGBColor(&rgb,0,0,0);
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	TextFont(kFontIDGeneva); TextSize(6);
#endif
	if (!fDepthContoursH) 
	{
		OSErr err = 0;
		fDepthContoursH = (DOUBLEH)_NewHandleClear(0);
		if(!fDepthContoursH){TechError("TTriGridVel3D::DrawContourScale()", "_NewHandle()", 0); err = memFullErr; return;}
		if (err = SetDefaultContours(fDepthContoursH,1)) 
		{
			return;
		}
		//return;	// might not want to allow legend without contours
	}

	if (gSavingOrPrintingPictFile)
	{
		Rect mapRect;
#ifdef MAC
		mapRect = DrawingRect(settings.listWidth + 1, RIGHTBARWIDTH);
#else
		mapRect = DrawingRect(settings.listWidth, RIGHTBARWIDTH);
#endif
		if (!EqualRects(r,mapRect))
		{
			Boolean bCloserToTop = (legendRect.top - mapRect.top) <= (mapRect.bottom - legendRect.bottom);
			Boolean bCloserToLeft = (legendRect.left - mapRect.left) <= (mapRect.right - legendRect.right);
			if (bCloserToTop)
			{
				legendRect.top = legendRect.top - mapRect.top + r.top;
				legendRect.bottom = legendRect.bottom - mapRect.top + r.top;
			}
			else
			{
				legendRect.top = r.bottom - (mapRect.bottom - legendRect.top);
				legendRect.bottom = r.bottom - (mapRect.bottom - legendRect.bottom);
			}
			if (bCloserToLeft)
			{
				legendRect.left = legendRect.left - mapRect.left + r.left;
				legendRect.right = legendRect.right - mapRect.left + r.left;
			}
			else
			{
				legendRect.left = r.right - (mapRect.right - legendRect.left);
				legendRect.right = r.right - (mapRect.right - legendRect.right);
			}
		}
	}
	else
	{
		if (EmptyRect(&fLegendRect)||!RectInRect2(&legendRect,&r))
		{
			legendRect.top = r.top;
			legendRect.left = r.right - 80;
			//legendRect.bottom = r.top + 120;	// reset after contour levels drawn
			legendRect.bottom = r.top + 90;	// reset after contour levels drawn
			legendRect.right = r.right;	// reset if values go beyond default width
		}
	}
	rgbrect = legendRect;
	EraseRect(&rgbrect);
	
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10;
	y = rgbrect.top + dY / 2;
	MyMoveTo(x - stringwidth("Depth Contours") / 2, y + dY);
	drawstring("Depth Contours");
	//MyMoveTo(x - stringwidth("Units: m") / 2, y + 2 * dY);
	//drawstring("Units: m");
	numLevels = GetNumDoubleHdlItems(fDepthContoursH);
	v = rgbrect.top+30;
	//v = rgbrect.top+60;
	h = rgbrect.left;
	for (i=0;i<numLevels;i++)
	{
		//float colorLevel = .8*float(i)/float(numLevels-1);
		float colorLevel = float(i)/float(numLevels-1);
		value = (*fDepthContoursH)[i];
	
		MySetRect(&rgbrect,h+4,v-9,h+14,v+1);
		
#ifdef IBM		
		rgb = GetRGBColor(colorLevel);
#else
		rgb = GetRGBColor(1.-colorLevel);
#endif
		//rgb = GetRGBColor(0.8-colorLevel);
		RGBForeColor(&rgb);
		PaintRect(&rgbrect);
		MyFrameRect(&rgbrect);
	
		MyMoveTo(h+20,v+.5);
	
		MyNumToStr(value,numstr);
		strcat(numstr," m");
		drawstring(numstr);
		if (stringwidth(numstr)>widestNum) widestNum = stringwidth(numstr);
		v = v+9;
	}
	legendRect.bottom = v+3;
	if (legendRect.right<h+20+widestNum+4) legendRect.right = h+20+widestNum+4;
	else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
		legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
	RGBForeColor(&colors[BLACK]);
 	MyFrameRect(&legendRect);

	if (!gSavingOrPrintingPictFile)
		fLegendRect = legendRect;
	return;
}

void TTriGridVel3D::DrawTriangleStr(Rect *r,long triNum, double value)
{
	long v1,v2,v3;
	Point pt1,pt2,pt3;
	TopologyHdl topH = fDagTree->GetTopologyHdl();
	LongPointHdl ptsH = fDagTree->GetPointsHdl();
	Boolean offQuickDrawPlane;
	char triangleStr[64];
	//Rect strRect;
	short x,y;
	
	if (value<0) 
	{
		//triangleStr[0]='X';
		//triangleStr[1]=0;
		strcpy(triangleStr,"X");
		TextFace(bold);
	}
	else
		MyNumToStr(value,triangleStr);
#ifdef MAC		
	short fontnum;
	//getfnum("System",&fontnum);
	//TextFont(fontnum);  // JLM 5/6/08
	TextFont(kFontIDGeneva);
	TextSize(8);
#else
	//TextFont(smallFonts);
	TextFont(kFontIDGeneva);
	TextSize(6);
#endif
	RGBForeColor(&colors[BLACK]);
	PenNormal();
	
	v1 = (*topH)[triNum].vertex1;
	v2 = (*topH)[triNum].vertex2;
	v3 = (*topH)[triNum].vertex3;

	pt1 = GetQuickDrawPt((*ptsH)[v1].h,(*ptsH)[v1].v,r,&offQuickDrawPlane);
	pt2 = GetQuickDrawPt((*ptsH)[v2].h,(*ptsH)[v2].v,r,&offQuickDrawPlane);
	pt3 = GetQuickDrawPt((*ptsH)[v3].h,(*ptsH)[v3].v,r,&offQuickDrawPlane);

	x = (pt1.h+pt2.h+pt3.h)/3;
	y = (pt1.v+pt2.v+pt3.v)/3;
	GetTextOffsets(triangleStr,&x,&y);
	//GetStringRect(triangleStr,x,y,&strRect);
	//EraseRect(&strRect);
	MyMoveTo(x,y);
	drawstring(triangleStr);
	TextFace(normal);
}

////////////////////////////////////////////////////////////////////////////////////
void TTriGridVel3D::Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,double arrowScale,
					   double arrowDepth,Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor)
{
	short row, col, pixX, pixY;
	float inchesX, inchesY;
	Point p, p2;
	Rect c;
	WorldPoint wp;
	VelocityRec velocity;
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	LongPoint wp1,wp2,wp3;
	long i,numTri,n;
	Boolean offQuickDrawPlane = false;
	PenState	pensave;
	

	if(fDagTree == 0)return;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();
	numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);

	//p.h = SameDifferenceX(refP.pLong);
	//p.v = (r.bottom + r.top) - SameDifferenceY(refP.pLat);
	p = GetQuickDrawPt(refP.pLong, refP.pLat, &r, &offQuickDrawPlane);
	
	// draw the reference point
	// for now don't draw
	/*if (!offQuickDrawPlane)
	{
		RGBForeColor(&colors[BLUE]);
		MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
		PaintRect(&c);
	}*/
	RGBForeColor(&colors[BLACK]);
		
	RGBForeColor(&colors[PURPLE]);


	for (i = 0 ; i< numTri; i++)
	{
		if (bDrawArrows) 
		{
			RGBForeColor(&arrowColor);
			wp1 = (*ptsH)[(*topH)[i].vertex1];
			wp2 = (*ptsH)[(*topH)[i].vertex2];
			wp3 = (*ptsH)[(*topH)[i].vertex3];
	
			wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
			wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
			velocity = GetPatValue(wp);
			
			//p.h = SameDifferenceX(wp.pLong);
			//p.v = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
//			PaintRect(&c);
							
			if (velocity.u != 0 || velocity.v != 0) 
			{
				inchesX = (velocity.u * refScale) / arrowScale;
				inchesY = (velocity.v * refScale) / arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
	
				//DrawArrowHead (p, p2, velocity);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
			RGBForeColor(&colors[PURPLE]);
		}
		
		//if (bDrawGrid) DrawTriangle3D(&r,i,FALSE,FALSE);	// don't fill triangles
		if (bDrawGrid || (bShowSelectedTriangles && fTriSelected && (*fTriSelected)[i])) 
			DrawTriangle3D(&r,i,FALSE,FALSE);	// don't fill triangles
		if (bShowSelectedTriangles && fTriSelected && (*fTriSelected)[i]) 
		{
			RGBForeColor(&colors[BLACK]);
			DrawTriangle3D(&r,i,FALSE,TRUE);	// don't fill triangles, highlight selected triangles
			RGBForeColor(&colors[PURPLE]);
		}
	}
	RGBForeColor(&colors[BLACK]);
	GetPenState(&pensave);
	n = GetNumPoints();
	if(fPtsSelected)
	{
		for(i=0;i<n;i++)
		{
			if((*fPtsSelected)[i])
			{
				this->DrawPointAt(&r,i,0);
			}
		}
	}
	SetPenState(&pensave);

	return;
}

void TTriGridVel3D::DrawTriangle3D(Rect *r,long triNum,Boolean fillTriangle,Boolean selected)
{
#ifdef IBM
	POINT points[4];
#else
	PolyHandle poly;
#endif
	long v1,v2,v3;
	Point pt1,pt2,pt3;
	TopologyHdl topH = fDagTree->GetTopologyHdl();
	LongPointHdl ptsH = fDagTree->GetPointsHdl();
	Boolean offQuickDrawPlane;
	
	v1 = (*topH)[triNum].vertex1;
	v2 = (*topH)[triNum].vertex2;
	v3 = (*topH)[triNum].vertex3;

	
	pt1 = GetQuickDrawPt((*ptsH)[v1].h,(*ptsH)[v1].v,r,&offQuickDrawPlane);
	pt2 = GetQuickDrawPt((*ptsH)[v2].h,(*ptsH)[v2].v,r,&offQuickDrawPlane);
	pt3 = GetQuickDrawPt((*ptsH)[v3].h,(*ptsH)[v3].v,r,&offQuickDrawPlane);
	
	PenMode(patCopy);
#ifdef MAC
		poly = OpenPoly();
		MyMoveTo(pt1.h,pt1.v);
		MyLineTo(pt2.h,pt2.v);
		MyLineTo(pt3.h,pt3.v);
		MyLineTo(pt1.h,pt1.v);
		ClosePoly();
	
		//if(fillTriangle)
			//PaintPoly(poly);
		
		//FramePoly(poly);
		
		if(selected )
		{	
			RGBForeColor(&colors[BLACK]);
			PenMode(patXor);	// inverts, so will deselect a selected triangle
			PaintPoly(poly);
			PenMode(patCopy);
		}
		else
		{
			//RGBForeColor(&colors[BLACK]);
			if(fillTriangle)
				PaintPoly(poly);
			FramePoly(poly);
		}
	
		KillPoly(poly);
#else
		points[0] = MakePOINT(pt1.h,pt1.v);
		points[1] = MakePOINT(pt2.h,pt2.v);
		points[2] = MakePOINT(pt3.h,pt3.v);
		points[3] = MakePOINT(pt1.h,pt1.v);
	
	
		//if(fillTriangle)
			//Polygon(currentHDC,points,4); // code goes here

		if(selected )
		{
			//RGBForeColor(&colors[BLACK]);
			PenMode(patXor);
			Polygon(currentHDC, points, 4);
			PenMode(patCopy);
		}
		else
		{
			if(fillTriangle)
				Polygon(currentHDC,points,4); // code goes here
			Polyline(currentHDC,points,4);
		}

		//Polyline(currentHDC,points,4);
#endif
}

void TTriGridVel3D::DrawPointAt(Rect *r,long verIndex,short selectMode )
{
	Rect ovr;
	Point pt;
	Boolean offQuickDrawPlane;
	short x,y,h=3;
	LongPointHdl ptsH = fDagTree->GetPointsHdl();
	pt = GetQuickDrawPt((*ptsH)[verIndex].h,(*ptsH)[verIndex].v,r,&offQuickDrawPlane);

	//if(gWeArePrinting)selectMode = POINTDRAWFLAG;
	//if(!DrawPointNumberAt(r,verIndex,selectMode))
	{
		//if(selectMode != POINTDRAWFLAG || (buttonSettings[VERTICESINDEX] && selectMode == POINTDRAWFLAG))
		{
			/*x = SameDifferenceX((*gVertices)[verIndex].pLong);
			y = (r->bottom + r->top) - SameDifferenceY((*gVertices)[verIndex].pLat);
			switch(selectMode)
			{
				case POINTDRAWFLAG:
					h = 1;
					break;
				case BADPOINTDRAWFLAG:
					h = 2;
					selectMode = POINTDRAWFLAG;
					break;
				default:
					h = 3;
					break;
			}*/
			//ovr.left = x- h; ovr.right = x+h;
			//ovr.top = y- h; ovr.bottom = y + h;
			ovr.left = pt.h- h; ovr.right = pt.h+h;
			ovr.top = pt.v- h; ovr.bottom = pt.v + h;
			//PenMode(selectMode != POINTDRAWFLAG ? patXor: patCopy);
			PenMode(patXor);
			//if(selectMode == POINTDRAWFLAG)
			{
				EraseRect(&ovr);
			}
			PaintRect(&ovr);
		}
	}
}
Boolean TTriGridVel3D::ThereAreTrianglesSelected2() 
{
	long i, numTri;
	if (!fTriSelected) return false;
	numTri = GetNumTriangles();
	for (i=0;i<numTri;i++)
	{
		if ((*fTriSelected)[i]) return true;
	}
	return false;
}

Boolean TTriGridVel3D::SelectTriInPolygon(WORLDPOINTH wh, Boolean *needToRefresh)
{	// code extended from original cats to deal with refreshing
	long ntri = GetNumTriangles(),i,numsegs;
	WorldPoint w[3];
	Boolean triSelected = false, someTrisInPolygonAlreadySelected = false;
	Boolean someTrisOutsidePolygonStillSelected = false;
	SEGMENTH poly = WPointsToSegments(wh,_GetHandleSize((Handle)(wh))/sizeof(WorldPoint),&numsegs);
	*needToRefresh = false;
	if(poly != 0)
	{
		for(i=0; i < ntri; i++)
		{
			GetTriangleVerticesWP(i, w);
			if( PointInPolygon(w[0],poly,numsegs,true) &&
			   PointInPolygon(w[1],poly,numsegs,true) &&
			   PointInPolygon(w[2],poly,numsegs,true)
			   )
			{
				*needToRefresh = true;
				if ((*fTriSelected)[i] == true)
				{
					someTrisInPolygonAlreadySelected = true;
					break;
				}
				(*fTriSelected)[i]=true;		
				triSelected = true;
			}
		}
		if (someTrisInPolygonAlreadySelected)
		{
			for(i=0; i < ntri; i++)
			{
				GetTriangleVerticesWP(i, w);
				if( PointInPolygon(w[0],poly,numsegs,true) &&
				   PointInPolygon(w[1],poly,numsegs,true) &&
				   PointInPolygon(w[2],poly,numsegs,true)
				   )
				{
					(*fTriSelected)[i]=false;		
					triSelected = true;	// marks a change
				}
				else
				{
					if ((*fTriSelected)[i])
						someTrisOutsidePolygonStillSelected = true;
				}
			}
			// need to draw if deselected but not if didn't select at all
			if (!someTrisOutsidePolygonStillSelected)	{ClearTriSelection(); triSelected = false;}
		}
	}
	return triSelected;
}



OSErr TTriGridVel3D::ExportOilConcHdl(char* path)
{
	OSErr err = 0;
	DateTimeRec dateTime;
	Seconds time;
	double avConc,maxConc;
	outputData data;
	long numOutputValues,i;
	char buffer[512],concStr1[64],concStr2[64],timeStr[128];
	BFPB bfpb;
	
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out the times and values
	// add header line
	//strcpy(buffer,"Day Mo Yr Hr Min\t\tAv Conc\tMax Conc");
	strcpy(buffer,"Day\tMo\tYr\tHr\tMin\tAv Conc\tMax Conc");
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	numOutputValues = _GetHandleSize((Handle)fOilConcHdl)/sizeof(outputData);
	for(i = 0; i< numOutputValues;i++)
	{
		data = INDEXH(fOilConcHdl,i);
		time = data.time;
		avConc = data.avOilConcOverSelectedTri;
		maxConc = data.maxOilConcOverSelectedTri;
		SecondsToDate(time,&dateTime); // convert to 2 digit year?
		//if(dateTime.year>=2000) dateTime.year-=2000;
		//if(dateTime.year>=1900) dateTime.year-=1900;
		/*sprintf(timeStr, "%02hd,%02hd,%02hd,%02hd,%02hd",
		 dateTime.day, dateTime.month, dateTime.year,
		 dateTime.hour, dateTime.minute);*/
		
		sprintf(timeStr, "%02hd\t%02hd\t%02hd\t%02hd\t%02hd",
				dateTime.day, dateTime.month, dateTime.year,
				dateTime.hour, dateTime.minute);
		
		StringWithoutTrailingZeros(concStr1,avConc,3);
		StringWithoutTrailingZeros(concStr2,maxConc,3);
		/////
		strcpy(buffer,timeStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,concStr1);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		strcat(buffer,concStr2);
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
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

OSErr TTriGridVel3D::ExportTriAreaHdl(char* path, long numLevels)
{
	OSErr err = 0;
	double triArea;
	long numOutputValues,i,j,numTimes,k=0;
	char buffer[512],triAreaStr[64],indexStr[24],concStr[64];
	BFPB bfpb;
	
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	// Write out the times and values
	numOutputValues = _GetHandleSize((Handle)fTriAreaHdl)/sizeof(double);
	numTimes = numOutputValues / numLevels;
	
	// add header line
	strcpy(buffer,"Hr");
	for (j=0; j < numLevels; j++)
	{
		strcpy(concStr,"C");
		sprintf(indexStr,"%ld",j+1);
		strcat(concStr,indexStr);
		//strcat(buffer,"	  ");
		strcat(buffer,"\t");
		strcat(buffer,concStr);
	}
	strcat(buffer,NEWLINESTRING);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i = 0; i< numTimes;i++)
	{
		float outputTime;
		if(k>=numOutputValues) break;
		outputTime = model->LEDumpInterval / 3600.;
		sprintf(indexStr,"%ld",(i+1) * (long)outputTime);	// should multiply by LEDumpInterval since that is the output time step
		strcpy(buffer,indexStr);
		//strcat(buffer,"		");
		strcat(buffer,"\t");
		for (j=0; j < numLevels; j++)
		{
			triArea = INDEXH(fTriAreaHdl,k);
			MyNumToStr(triArea,triAreaStr);
			//sprintf(triAreaStr,"%.1g",triArea);
			strcat(buffer,triAreaStr);
			//strcat(buffer,"		");
			strcat(buffer,"\t");
			k++;
		}
		strcat(buffer,NEWLINESTRING);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
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

