
#include "Cross.h"
//#include "Classes.h"
#include "GridVel.h"
#include "MapUtils.h"
#include "DagTreeIO.h"
//#include "RectUtils.h"
#include "PtCurMover/PtCurMover.h"
#include "Contdlg.h"
#include "NetCDFMover/NetCDFMover.h"
#include "netcdf.h"
#include "Outils.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT GRIDVEL
#endif
#endif

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

Boolean IsGridWindFile(char *path,short *selectedUnitsP)
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
}

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

Boolean IsNetCDFPathsFile (char *path, Boolean *isNetCDFPathsFile, char *fileNamesPath, short *gridType)
{
	// NOTE!! if the input variable path does point to a NetCDFPaths file, 
	// the input variable is overwritten with the path to the first NetCDF file.
	// The original input value of path is copied to fileNamesPath in such a case.
	// If the input vatiable does not point to a NetCDFPaths file, the input path is left unchanged.
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line = 0;
	char	strLine [512];
	char	firstPartOfFile [512], classicPath[256];
	long lenToRead,fileLength;
	char *key;
	
	*isNetCDFPathsFile = false;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString

	if(err) {
		// should we report the file i/o err to the user here ?
		return false;
	}

	// must start with "NetCDF Files"
	NthLineInTextNonOptimized (firstPartOfFile, line++, strLine, 512);
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	key = "NetCDF Files";
	if (strncmpnocase (strLine, key, strlen(key)) != 0)
		return false;

	// next line must be "[FILE] <path>"
	NthLineInTextNonOptimized(firstPartOfFile, line++, strLine, 512); 
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	key = "[FILE]";
	if (strncmpnocase (strLine, key, strlen(key)) != 0)
		return false;

	strcpy(fileNamesPath,path); // transfer the input path to this output variable

	strcpy(path,strLine+strlen(key)); // this is overwriting the input variable (see NOTE above)
	RemoveLeadingAndTrailingWhiteSpace(path);
	ResolvePathFromInputFile(fileNamesPath,path); // JLM 6/8/10

	if(!FileExists(0,0,path)){
		// tell the user the file does not exist
		printError("FileExists returned false for the first path listed in the IsNetCDFPathsFile.");
		return false;
	}
	
	bIsValid = IsNetCDFFile (path, gridType);
	if (bIsValid) *isNetCDFPathsFile = true;
	else{
		// tell the user this is not a NetCDF file
		printError("IsNetCDFFile returned false for the first path listed in the IsNetCDFPathsFile.");
		return false;
	}

	return bIsValid;
}

/////////////////////////////////////////////////

Boolean IsNetCDFFile (char *path, short *gridType)	
{
	// separate into IsNetCDFFile and GetGridType
	Boolean	bIsValid = false;
	OSErr err = noErr;
	long line;
	char strLine [512], outPath[256];
	char firstPartOfFile [512], *modelTypeStr=0, *gridTypeStr=0, *sourceStr=0/*, *historyStr=0*/;
	long lenToRead,fileLength;
	int status, ncid;
	size_t t_len, t_len2;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with "CDF
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (!strncmp (firstPartOfFile, "CDF", 3))
			bIsValid = true;
	}
	
	if (!bIsValid) return false;

	// need a global attribute to identify grid type - this won't work for non Navy regular grid
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{*gridType = CURVILINEAR; goto done;}*/	// this should probably be an error
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		//if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;}	// this should probably be an error
		if (status != NC_NOERR) {*gridType = REGULAR; goto done;}	// this should probably be an error - change default to regular 1/29/09
	}
	//OSStatus strcpyFileSystemRepresentationFromClassicPath(char *nativePath, char * classicPath, long nativePathMaxLength )
	//if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;}	// this should probably be an error

	status = nc_inq_attlen(ncid,NC_GLOBAL,"grid_type",&t_len2);
	if (status == NC_NOERR) /*{*gridType = CURVILINEAR; goto done;}*/
	{
		gridTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "grid_type", gridTypeStr);
		//if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;} 
		if (status != NC_NOERR) {*gridType = REGULAR; goto done;} 
		gridTypeStr[t_len2] = '\0';
		
		//if (!strncmpnocase (gridTypeStr, "REGULAR", 7) || !strncmpnocase (gridTypeStr, "UNIFORM", 7) || !strncmpnocase (gridTypeStr, "RECTANGULAR", 11))
		if (!strncmpnocase (gridTypeStr, "REGULAR", 7) || !strncmpnocase (gridTypeStr, "UNIFORM", 7) || !strncmpnocase (gridTypeStr, "RECTANGULAR", 11) /*|| !strncmpnocase (gridTypeStr, "RECTILINEAR", 11)*/)
		// note CO-OPS uses rectilinear but they have all the data for curvilinear so don't add the grid type
		{
			 *gridType = REGULAR;
			 goto done;
		}
		if (!strncmpnocase (gridTypeStr, "CURVILINEAR", 11) || !strncmpnocase (gridTypeStr, "RECTILINEAR", 11) || strstrnocase(gridTypeStr,"curv"))// "Rectilinear" is what CO-OPS uses, not one of our keywords. Their data is in curvilinear format. NYHOPS uses "Orthogonal Curv Grid"
		{
			 *gridType = CURVILINEAR;
			 goto done;
		}
		if (!strncmpnocase (gridTypeStr, "TRIANGULAR", 10))
		{
			 *gridType = TRIANGULAR;
			 goto done;
		}
	}
	else	// for now don't require global grid identifier since LAS files don't have it
	{
		status = nc_inq_attlen(ncid,NC_GLOBAL,"source",&t_len2);	// for HF Radar use source since no grid_type global
		if (status == NC_NOERR) 
		{
			sourceStr = new char[t_len2+1];
			status = nc_get_att_text(ncid, NC_GLOBAL, "source", sourceStr);
			if (status != NC_NOERR) { } 
			else
			{
				sourceStr[t_len2] = '\0';			
				if (!strncmpnocase (sourceStr, "Surface Ocean HF-Radar", 22)) { *gridType = REGULAR; goto done;}
			}
		}
		/*status = nc_inq_attlen(ncid,NC_GLOBAL,"history",&t_len2);	// LAS uses ferret, would also need to check for coordinate variable...
		if (status == NC_NOERR) 
		{
			historyStr = new char[t_len2+1];
			status = nc_get_att_text(ncid, NC_GLOBAL, "history", historyStr);
			if (status != NC_NOERR) { } 
			else
			{
				sourceStr[t_len2] = '\0';			
				if (strstrnocase (historyStr, "ferret") { *gridType = REGULAR; goto done;}	// could be curvilinear - maybe a ferret flag??
			}
		}*/
	}
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len);
	if (status != NC_NOERR) {
		status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len);
		//if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;}}
		if (status != NC_NOERR) {*gridType = REGULAR; goto done;}}	// changed default to REGULAR 1/29/09
	modelTypeStr = new char[t_len+1];
	status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
	if (status != NC_NOERR) {
		status = nc_get_att_text(ncid, NC_GLOBAL, "generator", modelTypeStr);
		//if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;} }
		if (status != NC_NOERR) {*gridType = REGULAR; goto done;} }	// changed default to REGULAR 1/29/09
	modelTypeStr[t_len] = '\0';
	
	if (!strncmp (modelTypeStr, "SWAFS", 5))
		 *gridType = REGULAR_SWAFS;
	//else if (!strncmp (modelTypeStr, "NCOM", 4))
	else if (strstr (modelTypeStr, "NCOM"))	// Global NCOM
		 *gridType = REGULAR;
	//else if (!strncmp (modelTypeStr, "fictitious test data", strlen("fictitious test data")))
		//*gridType = CURVILINEAR;	// for now, should have overall Navy identifier
	else
		 //*gridType = CURVILINEAR;
		 *gridType = REGULAR; // change default to REGULAR - 1/29/09

done:
	if (modelTypeStr) delete [] modelTypeStr;	
	if (gridTypeStr) delete [] gridTypeStr;	
	if (sourceStr) delete [] sourceStr;	
	//if (historyStr) delete [] historyStr;	
	return bIsValid;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

TGridVel::TGridVel() 
{
	fGridBounds = emptyWorldRect;
}
		
		
void TGridVel::Dispose()
{ 
	return; 
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

void TRectGridVel::Dispose ()
{
	if (fGridHdl)
	{
		DisposeHandle((Handle)fGridHdl);
		fGridHdl = nil;
	}
	TGridVel::Dispose ();
}

long TRectGridVel::NumVelsInGridHdl(void)
{
	long numInHdl = 0;
	if (fGridHdl) numInHdl = _GetHandleSize((Handle)fGridHdl)/sizeof(**fGridHdl);
	
	return numInHdl;
}

void TRectGridVel::SetBounds(WorldRect bounds)
{
	// if we read on old style OSSM cur file, we take the bounds from the map
	// (The map calls SetBounds with its bounds)
	// BUT if we read a new style grid file, we already know the lat long and don't want the map overriding it 
	// so ignore the call to this function in that case
	if(EqualWRects(fGridBounds,emptyWorldRect))
	{
		fGridBounds = bounds; // we haven't set the bounds, take their value
	}
	else
	{
		// ignore their value, we already know the bounds
	}
}


VelocityRec TRectGridVel::GetPatValue(WorldPoint p)
{

	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;

	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, fGridBounds.loLong, fGridBounds.loLat, fGridBounds.hiLong, fGridBounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
//	gridP = WorldToScreenPoint(p, bounds, CATSgridRect);
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;


	if (!fGridHdl || colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)

		{ velocity.u = 0.0; velocity.v = 0.0; return velocity; }
		
	return INDEXH (fGridHdl, rowNum * fNumCols + colNum);

}

VelocityRec TRectGridVel::GetSmoothVelocity(WorldPoint p)
{
	Point gridP;
	long rowNum, colNum;
	VelocityRec	velocity, velNew;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;

	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, fGridBounds.loLong, fGridBounds.loLat, fGridBounds.hiLong, fGridBounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);

	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	velocity = GetPatValue (p);

	if (colNum > 0 && colNum < fNumCols - 1 &&
		rowNum > 0 && rowNum < fNumRows - 1)
	{
		VelocityRec		topV, leftV, bottomV, rightV;
		
		topV    = INDEXH (fGridHdl, rowNum + 1 * fNumCols + colNum);
		bottomV = INDEXH (fGridHdl, rowNum - 1 * fNumCols + colNum);
		leftV   = INDEXH (fGridHdl, rowNum     * fNumCols + colNum - 1);
		rightV  = INDEXH (fGridHdl, rowNum     * fNumCols + colNum + 1);
		
		velNew.u = .5 * velocity.u + .125 * (topV.u + bottomV.u + leftV.u + rightV.u);
		velNew.v = .5 * velocity.v + .125 * (topV.v + bottomV.v + leftV.v + rightV.v);
	}
	else
		velNew = velocity;

	return velNew;
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
						double arrowScale, Boolean bDrawArrows, Boolean bDrawGrid) 
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
			}
		}
		
	RGBForeColor(&colors[BLACK]);
}


/////////////////////////////////////////////////

void TTriGridVel::Dispose ()
{
	if (fDagTree)
	{
		fDagTree->Dispose();
		delete fDagTree;
		fDagTree = nil;
	}
	if (fBathymetryH)
	{
		DisposeHandle((Handle)fBathymetryH);
		fBathymetryH = 0;
	}
	TGridVel::Dispose ();
}

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

LongPointHdl TTriGridVel::GetPointsHdl(void)
{
	if(!fDagTree) return nil;
	
	return fDagTree->GetPointsHdl();
}

TopologyHdl TTriGridVel::GetTopologyHdl(void)
{
	if(!fDagTree) return nil;
	
	return fDagTree->GetTopologyHdl();
}

/*DAGHdl TTriGridVel::GetDagTreeHdl(void)
{
	if(!fDagTree) return nil;
	
	return fDagTree->GetDagTreeHdl();
}*/

long TTriGridVel::GetNumTriangles(void)
{
	long numTriangles = 0;
	TopologyHdl topoH = fDagTree->GetTopologyHdl();
	if (topoH) numTriangles = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
	
	return numTriangles;
}

InterpolationVal TTriGridVel::GetInterpolationValues(WorldPoint refPoint)
{
	InterpolationVal interpolationVal;
	LongPoint lp;
	long ntri;
	ExPoint vertex1,vertex2,vertex3;
	double denom,refLon,refLat;
	double num1,num2,num3;

	TopologyHdl topH ;
	LongPointHdl ptsH ;

	memset(&interpolationVal,0,sizeof(interpolationVal));
	
	if(!fDagTree) return interpolationVal;

	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		interpolationVal.ptIndex1 = ntri; // flag it
		return interpolationVal;
	}

	refLon = lp.h/1000000.;
	refLat = lp.v/1000000.;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();

	if(!topH || !ptsH) return interpolationVal;

	// get the index into the pts handle for each vertex
	
	interpolationVal.ptIndex1 = (*topH)[ntri].vertex1;
	interpolationVal.ptIndex2 = (*topH)[ntri].vertex2;
	interpolationVal.ptIndex3 = (*topH)[ntri].vertex3;
	
	// get the vertices from fPtsH and figure out the interpolation coefficients

	vertex1.h = (*ptsH)[interpolationVal.ptIndex1].h/1000000.;
	vertex1.v = (*ptsH)[interpolationVal.ptIndex1].v/1000000.;
	vertex2.h = (*ptsH)[interpolationVal.ptIndex2].h/1000000.;
	vertex2.v = (*ptsH)[interpolationVal.ptIndex2].v/1000000.;
	vertex3.h = (*ptsH)[interpolationVal.ptIndex3].h/1000000.;
	vertex3.v = (*ptsH)[interpolationVal.ptIndex3].v/1000000.;


	// use a1*x1+a2*x2+a3*x3=x_ref, a1*y1+a2*y2+a3*y3=y_ref, and a1+a2+a3=1
	
	denom = (vertex3.v-vertex1.v)*(vertex2.h-vertex1.h)-(vertex3.h-vertex1.h)*(vertex2.v-vertex1.v);
	
	num1 = ((refLat-vertex3.v)*(vertex3.h-vertex2.h)-(refLon-vertex3.h)*(vertex3.v-vertex2.v));
	num2 = ((refLon-vertex1.h)*(vertex3.v-vertex1.v)-(refLat-vertex1.v)*(vertex3.h-vertex1.h));
	num3 = ((refLat-vertex1.v)*(vertex2.h-vertex1.h)-(refLon-vertex1.h)*(vertex2.v-vertex1.v));

	interpolationVal.alpha1 = num1/denom;
	interpolationVal.alpha2 = num2/denom;
	interpolationVal.alpha3 = num3/denom;

	return interpolationVal;
}

long TTriGridVel::GetRectIndexFromTriIndex2(long triIndex,LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp;
	long i, n, ntri = triIndex, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ;
	double refLon,refLat;

	TopologyHdl topH ;

	if(!fDagTree) return -1;

	if (ntri < 0) return ntri;

	topH = fDagTree->GetTopologyHdl();

	if(!topH) return -1;

	// get the index into the pts handle for each vertex
	
	//ptIndex1 = (*topH)[ntri].vertex1;
	//ptIndex2 = (*topH)[ntri].vertex2;
	//ptIndex3 = (*topH)[ntri].vertex3;
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;

	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	return index;
}

long TTriGridVel::GetRectIndexFromTriIndex(WorldPoint refPoint,LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp;
	long i, n, ntri, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ;
	double refLon,refLat;

	TopologyHdl topH ;

	if(!fDagTree) return -1;

	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		index = ntri; // flag it
		return index;
	}

	topH = fDagTree->GetTopologyHdl();

	if(!topH) return -1;

	// get the index into the pts handle for each vertex
	
	//ptIndex1 = (*topH)[ntri].vertex1;
	//ptIndex2 = (*topH)[ntri].vertex2;
	//ptIndex3 = (*topH)[ntri].vertex3;
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;

	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	return index;
}

OSErr TTriGridVel::GetRectCornersFromTriIndexOrPoint(long *index1, long *index2, long *index3, long *index4, WorldPoint refPoint,long triNum, Boolean useTriNum, LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp;
	long i, n, ntri, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long debug_ptIndex1, debug_ptIndex2, debug_ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ, numCols = numCols_ext-1;
	double refLon,refLat;

	TopologyHdl topH ;

	if(!fDagTree) return -1;

	if (!useTriNum)
	{
		lp.h = refPoint.pLong;
		lp.v = refPoint.pLat;
		ntri = fDagTree->WhatTriAmIIn(lp);
	}
	else ntri = triNum;
	if (ntri < 0) 
	{
		index = ntri; // flag it
		return index;
	}

	topH = fDagTree->GetTopologyHdl();

	if(!topH) return -1;

	// get the index into the pts handle for each vertex
	
	debug_ptIndex1 = (*topH)[ntri].vertex1;
	debug_ptIndex2 = (*topH)[ntri].vertex2;
	debug_ptIndex3 = (*topH)[ntri].vertex3;
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;

	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	//index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	
	/**index1 = (largestI-1)*(numCols_ext-1)+smallestJ;
	*index2 = (largestI)*(numCols_ext-1)+smallestJ;
	*index3 = (largestI-1)*(numCols_ext-1)+smallestJ+1;
	*index4 = (largestI)*(numCols_ext-1)+smallestJ+1;*/

if (smallestJ>=numCols-1)
{
	if (smallestJ==numCols)
		*index1=0;
}
if (largestI<=1) 
{
	if (largestI==0) 
		*index2 = -1;
}
	*index1 = (largestI-2)*(numCols_ext-1)+smallestJ;
	*index2 = (largestI-1)*(numCols_ext-1)+smallestJ;
	*index3 = (largestI-2)*(numCols_ext-1)+smallestJ+1;
	*index4 = (largestI-1)*(numCols_ext-1)+smallestJ+1;

	if (largestI==1) {*index1=-1; *index3=-1;}
	if (smallestJ==numCols-1) {*index3=-1;*index4=-1;}

	return 0;
}

LongPoint TTriGridVel::GetRectIndicesFromTriIndex(WorldPoint refPoint,LONGH ptrVerdatToNetCDFH,long numCols_ext)
{
	// code goes here, may eventually want to get interpolation indices around point
	LongPoint lp={-1,-1}, indices;
	long i, n, ntri, index=-1, ptIndex1,ptIndex2,ptIndex3;
	long iIndex[3],jIndex[3],largestI,smallestJ;
	double refLon,refLat;

	TopologyHdl topH ;

	if(!fDagTree) return lp;

	lp.h = refPoint.pLong;
	lp.v = refPoint.pLat;
	ntri = fDagTree->WhatTriAmIIn(lp);
	if (ntri < 0) 
	{
		index = ntri; // flag it
		return lp;
	}

	topH = fDagTree->GetTopologyHdl();

	if(!topH) return lp;

	// get the index into the pts handle for each vertex
	
	ptIndex1 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex1);
	ptIndex2 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex2);
	ptIndex3 = INDEXH(ptrVerdatToNetCDFH,(*topH)[ntri].vertex3);
	//n = INDEXH(verdatPtsH,i);
	// need to translate back to original index via verdatPtsH
	
	iIndex[0] = ptIndex1/numCols_ext;
	jIndex[0] = ptIndex1%numCols_ext;
	iIndex[1] = ptIndex2/numCols_ext;
	jIndex[1] = ptIndex2%numCols_ext;
	iIndex[2] = ptIndex3/numCols_ext;
	jIndex[2] = ptIndex3%numCols_ext;

	largestI = iIndex[0];
	smallestJ = jIndex[0];
	for(i=0;i<2;i++)
	{
		if (iIndex[i+1]>largestI) largestI = iIndex[i+1];
		if (jIndex[i+1]<smallestJ) smallestJ = jIndex[i+1];
	}
	//index = (largestI-1)*(numCols_ext-1)+smallestJ;	// velocity for grid box is lower left hand corner 
	indices.h = smallestJ;
	indices.v = largestI-1;
	return indices;
}

VelocityRec TTriGridVel::GetSmoothVelocity(WorldPoint p)
{
	VelocityRec r;
	LongPoint lp;

	lp.h = p.pLong;
	lp.v = p.pLat;

	fDagTree->GetVelocity(lp,&r);

	return r;
}

VelocityRec TTriGridVel::GetPatValue(WorldPoint p)
{
	VelocityRec r;
	LongPoint lp;

	lp.h = p.pLong;
	lp.v = p.pLat;

	fDagTree->GetVelocity(lp,&r);

	return r;
}

void TTriGridVel::Draw(Rect r, WorldRect view,WorldPoint refP,double refScale,double arrowScale,
					   Boolean bDrawArrows, Boolean bDrawGrid)
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
		}
		
		if (bDrawGrid) DrawTriangle(&r,i,FALSE);	// don't fill triangles
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
	//gCoord = 0;	// is this used??
	bCalculateDosage = false;
	bShowDosage = false;
	fDosageThreshold = .2;
	fMaxTri = -1;
	bShowMaxTri = false;
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
}
		
void TTriGridVel3D::Dispose ()
{
	if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
	if(fDepthContoursH) {DisposeHandle((Handle)fDepthContoursH); fDepthContoursH=0;}
	if(fTriSelected) {DisposeHandle((Handle)fTriSelected); fTriSelected=0;}
	if(fPtsSelected) {DisposeHandle((Handle)fPtsSelected); fPtsSelected=0;}
	if(fOilConcHdl) {DisposeHandle((Handle)fOilConcHdl); fOilConcHdl=0;}
	if(fMaxLayerDataHdl) {DisposeHandle((Handle)fMaxLayerDataHdl); fMaxLayerDataHdl=0;}
	if(fTriAreaHdl) {DisposeHandle((Handle)fTriAreaHdl); fTriAreaHdl=0;}
	if(fDosageHdl) {DisposeHandle((Handle)fDosageHdl); fDosageHdl=0;}
	//if(gCoord) {DisposeHandle((Handle)gCoord); gCoord=0;}
	
	TTriGridVel::Dispose ();
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
	fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
	if (!fDepthsH)
		{ TechError("TTriGridVel3D::Read()", "_NewHandleClear()", 0); goto done; }
	
	for (i = 0 ; i < numDepths ; i++) {
		if (err = ReadMacValue(bfpb, &val)) goto done;
		INDEXH(fDepthsH, i) = val;
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

long TTriGridVel3D::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

void TTriGridVel3D::ScaleDepths(double scaleFactor)
{
	long i, numDepths;
	if (!fDepthsH) return;
	numDepths = GetNumDepths();
	for (i=0;i<numDepths;i++)
	{
		(*fDepthsH)[i] *= scaleFactor;
	}
	return;
}

long TTriGridVel3D::GetNumDepthContours(void)
{
	long numContourValues = 0;
	if (fDepthContoursH) numContourValues = _GetHandleSize((Handle)fDepthContoursH)/sizeof(**fDepthContoursH);
	
	return numContourValues;
}

long TTriGridVel3D::GetNumOutputDataValues(void)
{
	long numOutputDataValues = 0;
	if (fOilConcHdl) numOutputDataValues = _GetHandleSize((Handle)fOilConcHdl)/sizeof(**fOilConcHdl);
	
	return numOutputDataValues;
}

/*long TTriGridVel3D::GetNumTriangles(void)
{
	long numTriangles = 0;
	TopologyHdl topoH = fDagTree->GetTopologyHdl();
	if (topoH) numTriangles = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
	
	return numTriangles;
}*/

long TTriGridVel3D::GetNumPoints(void)
{
	long numPts = 0;
	LongPointHdl ptsH ;

	ptsH = fDagTree->GetPointsHdl();
	if (ptsH) numPts = _GetHandleSize((Handle)ptsH)/sizeof(LongPoint);
	
	return numPts;
}

OSErr TTriGridVel3D::GetTriangleVertices(long i, long *x, long *y)
{
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	long ptIndex1, ptIndex2, ptIndex3;

	if(!fDagTree) return -1;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();

	if(!topH || !ptsH) return -1;

	ptIndex1 = (*topH)[i].vertex1;
	ptIndex2 = (*topH)[i].vertex2;
	ptIndex3 = (*topH)[i].vertex3;
	
	x[0] = (*ptsH)[ptIndex1].h;
	y[0] = (*ptsH)[ptIndex1].v;
	x[1] = (*ptsH)[ptIndex2].h;
	y[1] = (*ptsH)[ptIndex2].v;
	x[2] = (*ptsH)[ptIndex3].h;
	y[2] = (*ptsH)[ptIndex3].v;
	
	return noErr;
}	

OSErr TTriGridVel3D::GetTriangleVertices3D(long i, long *x, long *y, long *z)
{
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	long ptIndex1, ptIndex2, ptIndex3;

	if(!fDagTree) return -1;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();

	if(!topH || !ptsH) return -1;

	ptIndex1 = (*topH)[i].vertex1;
	ptIndex2 = (*topH)[i].vertex2;
	ptIndex3 = (*topH)[i].vertex3;
	
	x[0] = (*ptsH)[ptIndex1].h;
	y[0] = (*ptsH)[ptIndex1].v;
	x[1] = (*ptsH)[ptIndex2].h;
	y[1] = (*ptsH)[ptIndex2].v;
	x[2] = (*ptsH)[ptIndex3].h;
	y[2] = (*ptsH)[ptIndex3].v;
	
	z[0] = (*fDepthsH)[ptIndex1];
	z[1] = (*fDepthsH)[ptIndex2];
	z[2] = (*fDepthsH)[ptIndex3];
	
	return noErr;
}	

double GetTriangleArea(WorldPoint pt1, WorldPoint pt2, WorldPoint pt3)
{
	double sideA, sideB, sideC, angle3;
	double cp, triArea;
	WorldPoint center1,center2;
	// flat earth or spherical earth?
	sideC = DistanceBetweenWorldPoints(pt1,pt2);	// kilometers
	sideB = DistanceBetweenWorldPoints(pt1,pt3);
	sideA = DistanceBetweenWorldPoints(pt2,pt3);

	center1.pLat = (pt1.pLat + pt2.pLat) / 2.;	// center of map or center of line?
	center1.pLong = (pt1.pLong + pt2.pLong) / 2.;
	center2.pLat = (pt1.pLat + pt3.pLat) / 2.;
	center2.pLong = (pt1.pLong + pt3.pLong) / 2.;
	cp = LongToDistance(pt2.pLong - pt1.pLong, center1) * LatToDistance(pt3.pLat - pt1.pLat) - LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt2.pLat - pt1.pLat);
	//angle3 = acos((sideA*sideA + sideB*sideB - sideC*sideC)/(2*sideA*sideB));
	//triArea = sin(angle3)*sideA*sideB/2.;
	triArea = fabs(cp)/2.;

	return triArea;
}

// combine with GetTriangleArea
double GetQuadArea(WorldPoint pt1, WorldPoint pt2, WorldPoint pt3, WorldPoint pt4)
{
	double cp, quadArea;
	WorldPoint center1,center2,center3;

	center1.pLat = (pt1.pLat + pt2.pLat) / 2.;
	center1.pLong = (pt1.pLong + pt2.pLong) / 2.;
	center2.pLat = (pt1.pLat + pt3.pLat) / 2.;
	center2.pLong = (pt1.pLong + pt3.pLong) / 2.;
	center3.pLat = (pt1.pLat + pt4.pLat) / 2.;
	center3.pLong = (pt1.pLong + pt4.pLong) / 2.;
	cp =  LongToDistance(pt2.pLong - pt1.pLong, center1) * LatToDistance(pt3.pLat - pt1.pLat) - LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt2.pLat - pt1.pLat)
		+ LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt4.pLat - pt1.pLat) - LongToDistance(pt4.pLong - pt1.pLong, center3) * LatToDistance(pt3.pLat - pt1.pLat);
	
	quadArea = fabs(cp)/2.;

	return quadArea;
}

int WorldPoint3DCompare(void const *x1, void const *x2)
{
	WorldPoint3D *p1,*p2;	
	p1 = (WorldPoint3D*)x1;
	p2 = (WorldPoint3D*)x2;
	
	if ((*p1).z < (*p2).z) 
		return -1;  // first less than second
	else if ((*p1).z > (*p2).z)
		return 1;
	else return 0;// equivalent	
}

void FindPolygonPoints(short polygonType, WorldPoint3D *pts, double upperDepth, double lowerDepth, double *midPtArea, double *bottomArea)
{
	long k;
	double offset, dist, len;
	WorldPoint3D ptOnT1B3,ptOnT3B3,ptOnB2B3,ptOnT1B2,ptOnT2B2,center1;
	WorldPoint3D T1,T2,T3,B1,B2,B3;
	double h = lowerDepth -  upperDepth;

	T1 = pts[0]; T2 = pts[1]; T3 = pts[2];
	B1 = pts[0]; B2 = pts[1]; B3 = pts[2];
	T2.z = T1.z; T3.z = T1.z;
	
	*bottomArea = 0;
	*midPtArea = 0;
	for (k = 0; k<2; k++)
	{
		if (k==0) offset = h/2;
		else offset = 0;
		if (k==1 && lowerDepth == B3.z) break;
		dist = lowerDepth - offset - T1.z;
		len = B3.z - T1.z;
		ptOnT1B3.z = lowerDepth - offset;
		ptOnT1B3.p.pLat = T1.p.pLat + dist/len * (B3.p.pLat - T1.p.pLat);
		ptOnT1B3.p.pLong = T1.p.pLong + dist/len * (B3.p.pLong - T1.p.pLong);
		dist = lowerDepth - offset - T3.z;
		len = B3.z - T3.z;
		//  here lat/lon same at both points
		ptOnT3B3.p.pLat = T3.p.pLat;
		ptOnT3B3.p.pLong = T3.p.pLong;
		ptOnT3B3.z =  lowerDepth - offset;
		
		if (polygonType==0)
		{	// triangle
			dist = lowerDepth - offset - B2.z;
			len = B3.z - B2.z;
			//ptOnB2B3.p.pLat = B2.p.pLat + DistanceToLat(dist/len * LatToDistance(B3.p.pLat - B2.p.pLat, center),center);
			ptOnB2B3.p.pLat = B2.p.pLat + dist/len * (B3.p.pLat - B2.p.pLat);
			ptOnB2B3.p.pLong = B2.p.pLong + dist/len * (B3.p.pLong - B2.p.pLong);
			ptOnB2B3.z = lowerDepth - offset;
			if (k==1) *bottomArea = GetTriangleArea(ptOnT1B3.p,ptOnT3B3.p,ptOnB2B3.p);
			if (k==0) *midPtArea = GetTriangleArea(ptOnT1B3.p,ptOnT3B3.p,ptOnB2B3.p);
		}
		else
		{	// quadrilateral
			dist = lowerDepth - offset - T1.z;
			len = B2.z - T1.z;
			//ptOnT1B2.p.pLat = T1.p.pLat + DistanceToLat(dist/len * LatToDistance(B2.p.pLat - T1.p.pLat));
			ptOnT1B2.p.pLat = T1.p.pLat + dist/len * (B2.p.pLat - T1.p.pLat);
			//center1.pLong = (B2.p.pLong + T1.p.pLong) / 2.; center1.pLat = (B2.p.pLat + T1.p.pLat)/2.;
			//ptOnT1B2.p.pLong = T1.p.pLong + DistanceToLong(dist/len * LongToDistance(B2.p.pLong - T1.p.pLong,center1),center1);
			ptOnT1B2.p.pLong = T1.p.pLong + dist/len * (B2.p.pLong - T1.p.pLong);
			ptOnT1B2.z = lowerDepth - offset;
			dist = lowerDepth - offset - T2.z;
			len = B2.z - T2.z;
			// here lat/lon same at both points
			ptOnT2B2.p.pLat = T2.p.pLat;
			ptOnT2B2.p.pLong = T2.p.pLong;
			ptOnT2B2.z = lowerDepth - offset;
			if (k==0) *midPtArea = GetQuadArea(ptOnT1B2.p, ptOnT2B2.p, ptOnT3B3.p, ptOnT1B3.p);
			if (k==1) *bottomArea = GetQuadArea(ptOnT1B2.p, ptOnT2B2.p, ptOnT3B3.p, ptOnT1B3.p);
		}
	}
	return;
}

OSErr TTriGridVel3D::CalculateDepthSliceVolume(double *triVol, long triNum,float origUpperDepth, float origLowerDepth)
{
	double h, dist, len, debugTriVol;
	WorldPoint3D ptOnT1B2, ptOnT2B2, ptOnT1B3, ptOnT3B3, ptOnB2B3;
	long i,j,k, shallowIndex, midLevelIndex, deepIndex;
	double triArea, topTriArea, botTriArea, midTriArea, lastTriArea, offset = 0;
	if (triNum < 0) return -1;
	WorldPoint center1,center2;
	float upperDepth = origUpperDepth, lowerDepth = origLowerDepth; 
	OSErr err = 0;

	WorldPoint3D T1,T2,T3,B1,B2,B3, wp[3];

	err = GetTriangleVerticesWP3D(triNum, wp);
	qsort(wp,3,sizeof(WorldPoint3D),WorldPoint3DCompare);
	
	T1 = wp[0]; T2 = wp[1]; T3 = wp[2];
	B1 = wp[0]; B2 = wp[1]; B3 = wp[2];
	T2.z = T1.z; T3.z = T1.z;
	
	triArea = GetTriArea(triNum);	// kilometers
	lastTriArea = triArea;

	h = lowerDepth - upperDepth;	// usually 1 for depth profile
	if (h<=0) {*triVol = 0; return -1;}
	if (upperDepth > B3.z) {*triVol = 0; return noErr;}	// range is below bottom
	if (lowerDepth <= B1.z)	// shallowest depth
	{ 
		double theTriArea = GetTriangleArea(T1.p,T2.p,T3.p);
		*triVol = triArea * h * 1000 * 1000;	//	convert to meters 
		return noErr;
	}
	// need to deal with non-uniform volume once depth of shallowest vertex is reached
	else 
	{
		double firstPart = 0, secondPart = 0, thirdPart = 0;
		topTriArea = lastTriArea;
		if (lowerDepth <= B2.z)
		{
			// here bottom shape will be quadrilateral
			// get points where depth line intersects prism
			// get points where half depth line intersects prism - check if upperDepth > B1.z, otherwise need pieces
			// also check if j - B1.z < 0, then part of the region is above, ...
			if (upperDepth < B1.z)
			{
				firstPart = triArea * (B1.z - upperDepth);
				h = lowerDepth - B1.z;
			}
			FindPolygonPoints(1, wp, upperDepth, lowerDepth, &midTriArea, &botTriArea);
			secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
		}
		//else if ((j+1) <= B3.z)
		else	// bottom depth below second depth
		{	// special cases first
			if (lowerDepth > B3.z) 
			{
				// don't go below bottom
				h = B3.z - upperDepth;
				lowerDepth = B3.z;
			}
			if (upperDepth < B2.z)	// check B2 == B3 too
			{
				if (B2.z == B1.z)
				{
					firstPart = triArea * (B1.z - upperDepth);
					h = lowerDepth - B1.z;	
					upperDepth = B1.z;
					// fall to triArea calculation
				}
				else if (B2.z == B3.z)
				{
					if (upperDepth < B1.z)
					{
						firstPart = triArea * (B1.z - upperDepth);
						h = lowerDepth - B1.z;	// lower depth must be B2.z = B3.z
					}
					else
					{
						firstPart = 0;
						h = lowerDepth - upperDepth;
					}
					FindPolygonPoints(1, wp, upperDepth, lowerDepth, &midTriArea, &botTriArea);
					secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
					// no third part
				}
				else
				{
					// calculate the quad area, check if all points fall inside region
					if (upperDepth < B1.z)
					{	// three pieces
						firstPart = triArea * (B1.z - upperDepth);
						h = B2.z - B1.z;
						// calculate quad area
						FindPolygonPoints(1, wp, B1.z, B2.z, &midTriArea, &botTriArea);
						secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
						h = lowerDepth - B2.z;
						upperDepth = B2.z;
						topTriArea = botTriArea;
						// calculate tri area
					}
					else
					{
						h = B2.z - upperDepth;
						//lowerDepth = B2.z;
						FindPolygonPoints(1, wp, upperDepth, B2.z, &midTriArea, &botTriArea);
						secondPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);
						// Calculate quad stuff
						// then calcuate tri stuff with
						topTriArea = botTriArea;
						h = lowerDepth - B2.z;
						upperDepth = B2.z;
					}
				}
			}
			if (B2.z != B3.z) {FindPolygonPoints(0, wp, upperDepth, B2.z, &midTriArea, &botTriArea);
			thirdPart =  h/6. * (topTriArea + 4.*midTriArea + botTriArea);}

		}
		lastTriArea = botTriArea;
		//debugTriVol = (firstPart + h/6. * (topTriArea + 4.*midTriArea + botTriArea)) * 1000 * 1000;	
		//*triVol = (firstPart +  h/6. * (topTriArea + 4.*midTriArea + botTriArea)) * 1000 * 1000;	// convert to meters
		debugTriVol = (firstPart + secondPart + thirdPart) * 1000 * 1000;	
		*triVol = (firstPart + secondPart + thirdPart) * 1000 * 1000;	// convert to meters
	}

	return noErr;
}

OSErr TTriGridVel3D::GetTriangleDepths(long i, float *z)
{
	TopologyHdl topH ;
	LongPointHdl ptsH ;
	long ptIndex1, ptIndex2, ptIndex3;

	if(!fDagTree) return -1;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();

	if(!topH || !ptsH) return -1;

	ptIndex1 = (*topH)[i].vertex1;
	ptIndex2 = (*topH)[i].vertex2;
	ptIndex3 = (*topH)[i].vertex3;
	
	z[0] = (*fDepthsH)[ptIndex1];
	z[1] = (*fDepthsH)[ptIndex2];
	z[2] = (*fDepthsH)[ptIndex3];
	
	return noErr;
}	

OSErr TTriGridVel3D::GetMaxDepthForTriangle(long triNum, double *maxDepth)
{
	TopologyHdl topH ;
	long i, ptIndex[3];
	double z;

	if(!fDagTree) return -1;

	topH = fDagTree->GetTopologyHdl();

	if(!topH) return -1;

	*maxDepth = 0;
	ptIndex[0] = (*topH)[triNum].vertex1;
	ptIndex[1] = (*topH)[triNum].vertex2;
	ptIndex[2] = (*topH)[triNum].vertex3;
	
	for (i=0;i<3;i++)
	{
		z = (*fDepthsH)[ptIndex[i]];
		if (z > *maxDepth) *maxDepth = z;
	}	
	
	return noErr;
}	

OSErr TTriGridVel3D::GetTriangleCentroidWC(long trinum, WorldPoint *p)
{	
	long x[3],y[3];
	OSErr err = GetTriangleVertices(trinum,x,y);
	p->pLat = (y[0]+y[1]+y[2])/3;
	p->pLong =(x[0]+x[1]+x[2])/3;
	return err;
}

double TTriGridVel3D::GetTriArea(long triNum)
{
	WorldPoint pt1,pt2,pt3,center1,center2;
	long ptIndex1, ptIndex2, ptIndex3;
	//double sideA, sideB, sideC, angle3;
	double cp, triArea;

	TopologyHdl topH ;
	LongPointHdl ptsH ;

	if(!fDagTree) return -1;

	topH = fDagTree->GetTopologyHdl();
	ptsH = fDagTree->GetPointsHdl();

	if(!topH || !ptsH) return -1;
	if (triNum < 0) return -1;
	
	// get the index into the pts handle for each vertex
	ptIndex1 = (*topH)[triNum].vertex1;
	ptIndex2 = (*topH)[triNum].vertex2;
	ptIndex3 = (*topH)[triNum].vertex3;
	
	// get the vertices from the points handle

	pt1.pLong = (*ptsH)[ptIndex1].h;
	pt1.pLat = (*ptsH)[ptIndex1].v;
	pt2.pLong = (*ptsH)[ptIndex2].h;
	pt2.pLat = (*ptsH)[ptIndex2].v;
	pt3.pLong = (*ptsH)[ptIndex3].h;
	pt3.pLat = (*ptsH)[ptIndex3].v;

	center1.pLong = (pt2.pLong+pt1.pLong) / 2;
	center1.pLat = (pt2.pLat+pt1.pLat) / 2;
	center2.pLong = (pt3.pLong+pt1.pLong) / 2;
	center2.pLat = (pt3.pLat+pt1.pLat) / 2;
	//sideC = DistanceBetweenWorldPoints(pt1,pt2);	// kilometers
	//sideB = DistanceBetweenWorldPoints(pt1,pt3);
	//sideA = DistanceBetweenWorldPoints(pt2,pt3);

	cp = LongToDistance(pt2.pLong - pt1.pLong, center1) * LatToDistance(pt3.pLat - pt1.pLat) - LongToDistance(pt3.pLong - pt1.pLong, center2) * LatToDistance(pt2.pLat - pt1.pLat);
	//angle3 = acos((sideA*sideA + sideB*sideB - sideC*sideC)/(2.*sideA*sideB));
	//triArea = sin(angle3)*sideA*sideB/2.;
	triArea = fabs(cp)/2.;
	return triArea;
}

double **TTriGridVel3D::GetDosageHdl(Boolean initHdl)
{
	if (fDosageHdl) return fDosageHdl;
	else if (initHdl)
	{
		long i;
		long ntri = GetNumTriangles();
		fDosageHdl =(double **)_NewHandle(sizeof(double)*ntri);
		if(fDosageHdl)
		{
			for(i=0; i < ntri; i++)
			{
				(*fDosageHdl)[i] = 0.;
			}
			return fDosageHdl;
		}
		else {printError("Not enough memory to create dosage handle"); return nil;}
	}
	return nil;
	
}
Boolean **TTriGridVel3D::GetTriSelection(Boolean initHdl) 
{
	if (fTriSelected)	
		return fTriSelected;
	else if (initHdl)
	{
		long i;
		long ntri = GetNumTriangles();
		fTriSelected =(Boolean **)_NewHandle(sizeof(Boolean)*ntri);
		if(fTriSelected)
		{
			for(i=0; i < ntri; i++)
			{
				(*fTriSelected)[i] = false;
			}
			return fTriSelected;
		}
	}
	return nil;
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

double TTriGridVel3D::GetMaxAtPreviousTimeStep(Seconds time)
{
	long sizeOfHdl;
	float prevMax = -1;
	OSErr err = 0;
	outputData data;
	if(!fOilConcHdl)
	{
		return -1;
	}
	sizeOfHdl = _GetHandleSize((Handle)fOilConcHdl)/sizeof(outputData);
	if (sizeOfHdl>0) data = (*fOilConcHdl)[sizeOfHdl-1];
	if (sizeOfHdl>1 && time==(*fOilConcHdl)[sizeOfHdl-1].time)	
		prevMax = (*fOilConcHdl)[sizeOfHdl-2].maxOilConcOverSelectedTri;
	return prevMax;
}

	
void TTriGridVel3D::AddToOutputHdl(double avConcOverSelectedTriangles, double maxConcOverSelectedTriangles, Seconds time)
{
	long sizeOfHdl;
	OSErr err = 0;
	if(!fOilConcHdl)
	{
		fOilConcHdl = (outputDataHdl)_NewHandle(0);
		if(!fOilConcHdl) {TechError("TTriGridVel3D::AddToOutputHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fOilConcHdl)/sizeof(outputData);
	//if (sizeOfHdl>0 && time==(*fOilConcHdl)[sizeOfHdl-1].time) return;	// code goes here, check all times
	_SetHandleSize((Handle) fOilConcHdl, (sizeOfHdl+1)*sizeof(outputData));
	if (_MemError()) { TechError("TTriGridVel3D::AddToOutputHdl()", "_SetHandleSize()", 0); return; }
	//(*fOilConcHdl)[sizeOfHdl].oilConcAtSelectedTri = concentrationInSelectedTriangles;	// should add to old value??			
	(*fOilConcHdl)[sizeOfHdl].avOilConcOverSelectedTri = avConcOverSelectedTriangles;	// should add to old value??			
	(*fOilConcHdl)[sizeOfHdl].maxOilConcOverSelectedTri = maxConcOverSelectedTriangles;	// should add to old value??			
	(*fOilConcHdl)[sizeOfHdl].time = time;				
}

/*void TTriGridVel3D::AddToMaxLayerHdl(long maxLayer, long maxTri, Seconds time)
{	// want top/bottom ? 
	long sizeOfHdl;
	OSErr err = 0;
	if(!fMaxLayerDataHdl)
	{
		fMaxLayerDataHdl = (maxLayerDataHdl)_NewHandle(0);
		if(!fMaxLayerDataHdl) {TechError("TTriGridVel3D::AddToMaxLayerHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fMaxLayerDataHdl)/sizeof(maxLayerData);
	if (sizeOfHdl>0 && time==(*fMaxLayerDataHdl)[sizeOfHdl-1].time) return;	// code goes here, check all times
	_SetHandleSize((Handle) fMaxLayerDataHdl, (sizeOfHdl+1)*sizeof(maxLayerData));
	if (_MemError()) { TechError("TTriGridVel3D::AddToMaxLayerHdl()", "_SetHandleSize()", 0); return; }
	(*fMaxLayerDataHdl)[sizeOfHdl].maxLayer = maxLayer;	// should add to old value??			
	(*fMaxLayerDataHdl)[sizeOfHdl].maxTri = maxTri;	// should add to old value??			
	(*fMaxLayerDataHdl)[sizeOfHdl].time = time;				
}
Boolean TTriGridVel3D::GetMaxLayerInfo(long *maxLayer, long *maxTri, Seconds time)
{
	long i, sizeOfHdl = _GetHandleSize((Handle)fMaxLayerDataHdl)/sizeof(maxLayerData);
	*maxLayer = -1; *maxTri = -1;
	if (time > (*fMaxLayerDataHdl)[sizeOfHdl-1].time) return false;	// will need to calculate the info

	for (i=0;i<sizeOfHdl;i++)
	{
		if (time==(*fMaxLayerDataHdl)[i].time)
		{
			*maxLayer = (*fMaxLayerDataHdl)[i].maxLayer;
			*maxTri = (*fMaxLayerDataHdl)[i].maxTri;
			return true;
		}
	}
	return false;	// error message? if time step has changed either need to rerun
}
*/
void TTriGridVel3D::AddToTriAreaHdl(double *triAreaArray, long numValues)
{
	long i,sizeOfHdl;
	OSErr err = 0;
	if(!fTriAreaHdl)
	{
		fTriAreaHdl = (double**)_NewHandle(0);
		if(!fTriAreaHdl) {TechError("TTriGridVel3D::AddToTriAreaHdl()", "_NewHandle()", 0); err = memFullErr; return;}
	}
	sizeOfHdl = _GetHandleSize((Handle)fTriAreaHdl)/sizeof(double);
	//if (sizeOfHdl>0 && time==(*fTriAreaHdl)[sizeOfHdl-1].time) return;	// code goes here, check all times
	_SetHandleSize((Handle) fTriAreaHdl, (sizeOfHdl+numValues)*sizeof(double));
	if (_MemError()) { TechError("TTriGridVel3D::AddToTriAreaHdl()", "_SetHandleSize()", 0); return; }
	for (i=0;i<numValues;i++)
	{
		(*fTriAreaHdl)[sizeOfHdl+i] = triAreaArray[i];				
	}
}

void TTriGridVel3D::ClearOutputHandles()
{
	if(fOilConcHdl) 
	{
		DisposeHandle((Handle)fOilConcHdl); 
		fOilConcHdl = 0;
	}
	if(fTriAreaHdl) 
	{
		DisposeHandle((Handle)fTriAreaHdl); 
		fTriAreaHdl = 0;
	}
	if(fDosageHdl) 
	{
		DisposeHandle((Handle)fDosageHdl); 
		fDosageHdl = 0;
	}
	if(fMaxLayerDataHdl)
	{
		DisposeHandle((Handle)fMaxLayerDataHdl); 
		fMaxLayerDataHdl = 0;
	}
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

void TTriGridVel3D::GetTriangleVerticesWP(long i, WorldPoint *w)
{
	TopologyHdl topH;
	LongPointHdl ptsH;
	topH = fDagTree->GetTopologyHdl();	
	ptsH = fDagTree->GetPointsHdl();
	if(!topH || !ptsH) return;
	w[0].pLong = (*ptsH)[(*topH)[i].vertex1].h;
	w[0].pLat = (*ptsH)[(*topH)[i].vertex1].v;
	w[1].pLong = (*ptsH)[(*topH)[i].vertex2].h;
	w[1].pLat = (*ptsH)[(*topH)[i].vertex2].v;
	w[2].pLong = (*ptsH)[(*topH)[i].vertex3].h;
	w[2].pLat = (*ptsH)[(*topH)[i].vertex3].v;
	return;
}

OSErr TTriGridVel3D::GetTriangleVerticesWP3D(long i, WorldPoint3D *w)
{
	TopologyHdl topH;
	LongPointHdl ptsH;
	topH = fDagTree->GetTopologyHdl();	
	ptsH = fDagTree->GetPointsHdl();
	if(!topH || !ptsH) return -1;
	w[0].p.pLong = (*ptsH)[(*topH)[i].vertex1].h;
	w[0].p.pLat = (*ptsH)[(*topH)[i].vertex1].v;
	w[1].p.pLong = (*ptsH)[(*topH)[i].vertex2].h;
	w[1].p.pLat = (*ptsH)[(*topH)[i].vertex2].v;
	w[2].p.pLong = (*ptsH)[(*topH)[i].vertex3].h;
	w[2].p.pLat = (*ptsH)[(*topH)[i].vertex3].v;
	w[0].z = (*fDepthsH)[(*topH)[i].vertex1];
	w[1].z = (*fDepthsH)[(*topH)[i].vertex2];
	w[2].z = (*fDepthsH)[(*topH)[i].vertex3];
	return 0;
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

void TTriGridVel3D::DrawContourScale(Rect r, WorldRect view)
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
					   Boolean bDrawArrows, Boolean bDrawGrid)
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


