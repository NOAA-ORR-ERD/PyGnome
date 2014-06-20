#include "Cross.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "CurrentCycleMover.h"
#include "OUtils.h"
#include "DagTreeIO.h"
#include "netcdf.h"
#include "NetCDFMover.h"
#include "TShioTimeValue.h"
#include "PtCurMover.h"

enum {
	I_GRIDCURNAME = 0 ,
	I_GRIDCURACTIVE, 
	I_GRIDCURGRID, 
	I_GRIDCURARROWS,
	//I_GRIDCURSCALE,
	I_GRIDCURUNCERTAINTY,
	I_GRIDCURSTARTTIME,
	I_GRIDCURDURATION, 
	I_GRIDCURALONGCUR,
	I_GRIDCURCROSSCUR,
	I_GRIDCURMINCURRENT
};


static PopInfoRec cs2PopTable[] = {
	{ M16, nil, M16LATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
	{ M16, nil, M16LONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
	{ M16, nil, M16TIMEFILETYPES, 0, pTIMEFILETYPES, 0, 1, FALSE, nil }
};

static CurrentCycleMover	*sharedCCMover = 0;
//static CMyList 		*sharedMoverList = 0;
//static char 		sharedCMFileName[256];
static Boolean		sharedCCMChangedTimeFile;
static TOSSMTimeValue *sharedCCMDialogTimeDep = 0;
static CurrentUncertainyInfo sSharedCurrentCycleUncertainyInfo; // used to hold the uncertainty dialog box info in case of a user cancel

/////////////////////////////////////////////////
// JLM 11/25/98
// structure to help reset stuff when the user cancels from the uncertainty dialog box


static CurrentCycleDialogNonPtrFields sharedCurrentCycleDialogNonPtrFields;
CurrentCycleMover::CurrentCycleMover (TMap *owner, char *name) : GridCurrentMover(owner, name)
{
	TOSSMTimeValue *timeDep = 0;
	WorldPoint p = {0,0};
	bTimeFileActive = true;
	fPatternStartPoint = MaxFlood;	// this should be user input

	fEddyDiffusion = 0; // JLM 5/20/991e6; // cm^2/sec
	fEddyV0 = 0.1; // JLM 5/20/99

	bApplyLogProfile = false;

	refP = p;
	refZ = 0;
	scaleType = SCALE_NONE;
	scaleValue = 1.0;
	scaleOtherFile[0] = 0;
	bRefPointOpen = FALSE;
	//bUncertaintyPointOpen = FALSE;
	bTimeFileOpen = FALSE;
	/*fTimeHdl = 0;

	fUserUnits = kUndefined;
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -.5; 
	fUpCurUncertainty = .5; 	
	fRightCurUncertainty = .25;  
	fLeftCurUncertainty = -.25; 
	fDuration=24*3600.; //24 hrs as seconds 
	fUncertainStartTime = 0.; // seconds
	fEddyV0 = 0.0;	// fVar.uncertMinimumInMPS
	//SetClassName (name); // short file name

	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	fNumNodes = 0;
	
	//fPatternStartPoint = 2;	// some default
	fPatternStartPoint = MaxFlood;	// this should be user input
	fTimeAlpha = -1;

	fFillValue = -1e+34;
	fDryValue = -1e+34;
	//fDryValue = 999;
	
	fTopFilePath[0] = 0;*/	// don't seem to need this
}


/*void CurrentCycleMover::Dispose ()
{
	//if (fGrid)
	//{
		//fGrid -> Dispose();
		//delete fGrid;
		//fGrid = nil;
	//}

	if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}


	TCATSMover::Dispose ();
}*/


Boolean IsCurrentCycleFile (char *path, short *gridType)
{
	Boolean bIsValid = false;
	OSErr	err = noErr;
	long line;
	char strLine [512], outPath[256];
	char firstPartOfFile [512], *modelTypeStr=0, *gridTypeStr=0;
	long lenToRead,fileLength;
	int status, ncid;
	size_t t_len, t_len2;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with CDF
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (!strncmp (firstPartOfFile, "CDF", 3))
			bIsValid = true;
	}
	if (!bIsValid) return false;
	
	// need global attribute to identify grid and model type
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {bIsValid=false; goto done;}	// this should probably be an error
	}
	//if (status != NC_NOERR) 
	//{
	// check if path is resource which contains a path
	/*CHARH f;
	 long lenOfHdl;
	 if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
	 { TechError("IsCurrentCycleMover::ReadTimeValues()", "ReadFileContents()", 0); bIsValid=false; goto done; }
	 
	 lenOfHdl = _GetHandleSize((Handle)f);
	 //err = WriteFileContents(TATvRefNum, TATdirID, "GnomeTideFile.nc", '????', 'TEXT',
	 //				 		*f, lenOfHdl, 0);
	 err = WriteFileContents(TATvRefNum, TATdirID, "GnomeTideFile.nc", 'ttxt', 'TEXT',
	 0, lenOfHdl, f);
	 if(f) {DisposeHandle((Handle)f); f = 0;}
	 // replace file path with new file path*/
	//bIsValid=false; 
	//goto done;
	//}	// this should probably be an error
	
	status = nc_inq_attlen(ncid,NC_GLOBAL,"model_type",&t_len);
	if (status == NC_NOERR) /*{*gridType = CURVILINEAR; goto done;}*/
	{
		modelTypeStr = new char[t_len+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "model_type", modelTypeStr);
		if (status != NC_NOERR) {bIsValid=false; goto done;} 
		modelTypeStr[t_len] = '\0';
		
		if (!strncmpnocase (modelTypeStr, "TidalCurrentCycle", 17))
		{
			bIsValid = true;
		}
		else
		{bIsValid=false; goto done;}
	}
	else
	{
		bIsValid=false;
		goto done;
		// for now require global model identifier
	}
	status = nc_inq_attlen(ncid,NC_GLOBAL,"grid_type",&t_len2);
	if (status == NC_NOERR) /*{*gridType = CURVILINEAR; goto done;}*/
	{
		gridTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "grid_type", gridTypeStr);
		if (status != NC_NOERR) {*gridType = CURVILINEAR; goto done;} 
		gridTypeStr[t_len2] = '\0';
		
		if (!strncmpnocase (gridTypeStr, "REGULAR", 7))
		{
			*gridType = REGULAR;
			goto done;
		}
		if (!strncmpnocase (gridTypeStr, "CURVILINEAR", 11))
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
	else
	{
		// for now don't require global grid identifier
	}
	
done:
	if (modelTypeStr) delete [] modelTypeStr;	
	if (gridTypeStr) delete [] gridTypeStr;	
	return bIsValid;
}



/*Boolean CurrentCycleMover::CheckInterval(long &timeDataInterval, const Seconds& start_time, const Seconds& model_time)
 {

 // Seconds time =  model->GetModelTime();	// minus AH 07/17/2012
 Seconds time =  model_time;	// AH 07/17/2012
 long i,numTimes;
 
 
 numTimes = this -> GetNumTimesInFile(); 
 if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists
 
 // check for constant current
 if (numTimes==1) 
 {
 timeDataInterval = -1; // some flag here
 if(fStartData.timeIndex==0 && fStartData.dataHdl)
 return true;
 else
 return false;
 }
 
 if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
 {
 if (time>=(*fTimeHdl)[fStartData.timeIndex] && time<=(*fTimeHdl)[fEndData.timeIndex])
 {	// we already have the right interval loaded
 timeDataInterval = fEndData.timeIndex;
 return true;
 }
 }
 
 for (i=0;i<numTimes;i++) 
 {	// find the time interval
 if (time>=(*fTimeHdl)[i] && time<=(*fTimeHdl)[i+1])
 {
 timeDataInterval = i+1; // first interval is between 0 and 1, and so on
 return false;
 }
 }	
 // don't allow time before first or after last
 if (time<(*fTimeHdl)[0]) 
 timeDataInterval = 0;
 if (time>(*fTimeHdl)[numTimes-1]) 
 timeDataInterval = numTimes;
 return false;
 
 }*/
/*OSErr CurrentCycleMover::SetInterval(char *errmsg, const Seconds& start_time, const Seconds& model_time)
 {
 long timeDataInterval;
 Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
 long indexOfStart = timeDataInterval-1;
 long indexOfEnd = timeDataInterval;
 long numTimesInFile = this -> GetNumTimesInFile();
 OSErr err = 0;
 
 strcpy(errmsg,"");
 
 if(intervalLoaded) 
 return 0;
 
 // check for constant current 
 if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
 {
 indexOfStart = 0;
 indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
 }
 
 if(timeDataInterval == 0 || timeDataInterval == numTimesInFile)
 {	// before the first step in the file
 
 if (err==0)
 {
 // this will be handled differently here
 }
 else
 {
 err = -1;
 strcpy(errmsg,"Time outside of interval being modeled");
 goto done;
 }
 }
 // load the two intervals
 {
 DisposeLoadedData(&fStartData);
 
 if(indexOfStart == fEndData.timeIndex) // passing into next interval
 {
 fStartData = fEndData;
 ClearLoadedData(&fEndData);
 }
 else
 {
 DisposeLoadedData(&fEndData);
 }
 
 //////////////////
 
 if(fStartData.dataHdl == 0 && indexOfStart >= 0) 
 { // start data is not loaded
 err = this -> ReadTimeData(indexOfStart,&fStartData.dataHdl,errmsg);
 if(err) goto done;
 fStartData.timeIndex = indexOfStart;
 }	
 
 if(indexOfEnd < numTimesInFile && indexOfEnd != UNASSIGNEDINDEX)  // not past the last interval and not constant current
 {
 err = this -> ReadTimeData(indexOfEnd,&fEndData.dataHdl,errmsg);
 if(err) goto done;
 fEndData.timeIndex = indexOfEnd;
 }
 }
 
 done:	
 if(err)
 {
 if(!errmsg[0])strcpy(errmsg,"Error in CurrentCycleMover::SetInterval()");
 DisposeLoadedData(&fStartData);
 DisposeLoadedData(&fEndData);
 }
 return err;
 
 }*/


#define CurrentCycleMoverREADWRITEVERSION 1 

OSErr CurrentCycleMover::Write (BFPB *bfpb)
{
	long i, version = CurrentCycleMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints, /*numTimes = GetNumTimesInFile(),*/ index;
	Seconds time;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = GridCurrentMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("CurrentCycleMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	//if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	//if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	//if (err = WriteMacValue(bfpb, fNumNodes)) goto done;
	if (err = WriteMacValue(bfpb, fPatternStartPoint)) goto done;
	/*if (err = WriteMacValue(bfpb, fTimeAlpha)) goto done;
	if (err = WriteMacValue(bfpb, fFillValue)) goto done;
	//if (err = WriteMacValue(bfpb, fDryValue)) goto done;
	if (err = WriteMacValue(bfpb, fUserUnits)) goto done;
	if (err = WriteMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	if (err = WriteMacValue(bfpb, fFileName, kMaxNameLen)) goto done;	
	
	//
	if (err = WriteMacValue(bfpb, numTimes)) goto done;
	for (i=0;i<numTimes;i++)
	{
		time = INDEXH(fTimeHdl,i);
		if (err = WriteMacValue(bfpb, time)) goto done;
	}
	
	numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}
	
	numPoints = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}
	*/
done:
	if(err)
		TechError("CurrentCycleMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr CurrentCycleMover::Read(BFPB *bfpb)
{
	char msg[256], fileName[64];
	long i, version, index, numPoints, numTimes;
	ClassID id;
	WorldPointF vertex;
	Seconds time;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = GridCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("CurrentCycleMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("CurrentCycleMover::Read()", "id != TYPE_CURRENTCYCLEMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > CurrentCycleMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the GridCur mover (should always be rectgrid...)
	//if (err = ReadMacValue(bfpb, &fNumNodes)) goto done;	
	if (err = ReadMacValue(bfpb, &fPatternStartPoint)) goto done;
	/*if (err = ReadMacValue(bfpb, &fTimeAlpha)) goto done;
	if (err = ReadMacValue(bfpb, &fFillValue)) goto done;
	//if (err = ReadMacValue(bfpb, &DryValue)) goto done;
	if (err = ReadMacValue(bfpb, &fUserUnits)) goto done;
	if (err = ReadMacValue(bfpb, fPathName, kMaxNameLen)) goto done;	
	ResolvePath(fPathName); // JLM 6/3/10
	//if (!FileExists(0,0,fPathName)) {err=-1; sprintf(msg,"The file path %s is no longer valid.",fPathName); printError(msg); goto done;}
	if (!FileExists(0,0,fPathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],*p;
		strcpy(fileName,"");
		strcpy(newPath,fPathName);
		p = strrchr(newPath,DIRDELIMITER);
		if (p)
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
			//if (!FileExists(0,0,fileName)) 
		{bPathIsValid = false;}
		else
			strcpy(fPathName,fileName);
	}
	if (err = ReadMacValue(bfpb, fFileName, kMaxNameLen)) goto done;	
	//
	if (!bPathIsValid)
	{	// try other platform
		char delimStr[32] ={DIRDELIMITER,0};		
		strcpy(fileName,delimStr);
		strcat(fileName,fFileName);
		ResolvePath(fileName);
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{}
		else
		{
			strcpy(fPathName,fileName);
			bPathIsValid = true;
		}
	}
	// otherwise ask the user
	if(!bPathIsValid)
	{
		Point where;
		OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
		MySFReply reply;
		where = CenteredDialogUpLeft(M38c);
		char newPath[kMaxNameLen], s[kMaxNameLen];
		sprintf(msg,"This save file references a netCDF file which cannot be found.  Please find the file \"%s\".",fPathName);printNote(msg);
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(newPath, reply.fullPath);
		strcpy (s, newPath);
		SplitPathFile (s, fileName);
		strcpy (fPathName, newPath);
		strcpy (fFileName, fileName);
#else
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		//if (!reply.good) return 0;	// just keep going...
		if (reply.good)
		{
			my_p2cstr(reply.fName);
#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, newPath);
#else
			strcpy(newPath, reply.fName);
#endif
			
			strcpy (s, newPath);
			SplitPathFile (s, fileName);
			strcpy (fPathName, newPath);
			strcpy (fFileName, fileName);
		}
#endif
	}
	
	//
	if (err = ReadMacValue(bfpb, &numTimes)) goto done;	
	fTimeHdl = (Seconds**)_NewHandleClear(sizeof(Seconds)*numTimes);
	if(!fTimeHdl)
	{TechError("NetCDFMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numTimes ; i++) {
		if (err = ReadMacValue(bfpb, &time)) goto done;
		INDEXH(fTimeHdl, i) = time;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	// for curvilinear
	if(!fVerdatToNetCDFH)
	{TechError("NetCDFMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	// for curvilinear
	if(!fVertexPtsH)
	{TechError("NetCDFMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}
	
	
	//fUserUnits = kKnots;	// code goes here, implement using units
	*/
done:
	if(err)
	{
		TechError("CurrentCycleMover::Read(char* path)", " ", 0); 
		//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr CurrentCycleMover::CheckAndPassOnMessage(TModelMessage *message)
{
	// no good to do this here since topology needs to be read in before these messages are received
	/*char ourName[kMaxNameLen];
	 OSErr err = 0;
	 
	 // see if the message is of concern to us
	 this->GetClassName(ourName);
	 
	 if(message->IsMessage(M_SETFIELD,ourName))
	 {
	 char str[256];
	 message->GetParameterString("topFile",str,256);
	 ResolvePath(str);
	 if(str[0])
	 {
	 strcpy(fTopFilePath,str);
	 }
	 }*/
	return GridCurrentMover::CheckAndPassOnMessage(message); 
}

/////////////////////////////////////////////////
/*long CurrentCycleMover::GetListLength()
 {
 long n;
 short indent = 0;
 short style;
 char text[256];
 ListItem item;
 
 for(n = 0;TRUE;n++) {
 item = this -> GetNthListItem(n,indent,&style,text);
 if(!item.owner)
 return n;
 }
 //return n;
 }*/

/*ListItem CurrentCycleMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64];
	ListItem item = { dynamic_cast<CurrentCycleMover *>(this), 0, indent, 0 };
	
	
	if (n == 0) {
		item.index = I_GRIDCURNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		//sprintf(text, "Tide Pattern Time Series: \"%s\"", timeGrid->fFileName);
		sprintf(text, "Tide Pattern Time Series: \"%s\"", timeGrid->fVar.userName);
		if(!bActive)*style = italic; // JLM 6/14/10 -- check this
		
		return item;
	}
	else return GridCurrentMover::GetNthListItem(n,indent,style,text);
	
	if (bOpen) {
	 
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURACTIVE;
	 item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
	 strcpy(text, "Active");
	 item.indent++;
	 return item;
	 }
	 
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURGRID;
	 item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
	 sprintf(text, "Show Grid");
	 item.indent++;
	 return item;
	 }
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURARROWS;
	 item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
	 StringWithoutTrailingZeros(valStr,arrowScale,6);
	 sprintf(text, "Show Velocities (@ 1 in = %s m/s) ", valStr);
	 
	 item.indent++;
	 return item;
	 }
	 
	 //if (--n == 0) {
	 //item.index = I_GRIDCURSCALE;
	 //StringWithoutTrailingZeros(valStr,fVar.curScale,6);
	 //sprintf(text, "Multiplicative Scalar: %s", valStr);
	 //return item;
	 //}
	 
	 
	 if(model->IsUncertain())
	 {
	 if (--n == 0) {
	 item.index = I_GRIDCURUNCERTAINTY;
	 item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
	 strcpy(text, "Uncertainty");
	 item.indent++;
	 return item;
	 }
	 
	 if (bUncertaintyPointOpen) {
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURALONGCUR;
	 item.indent++;
	 StringWithoutTrailingZeros(valStr,fUpCurUncertainty*100,6);
	 sprintf(text, "Along Current: %s %%",valStr);
	 return item;
	 }
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURCROSSCUR;
	 item.indent++;
	 StringWithoutTrailingZeros(valStr,fRightCurUncertainty*100,6);
	 sprintf(text, "Cross Current: %s %%",valStr);
	 return item;
	 }
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURMINCURRENT;
	 item.indent++;
	 StringWithoutTrailingZeros(valStr,fEddyV0,6);	// reusing the TCATSMover variable for now
	 sprintf(text, "Current Minimum: %s m/s",valStr);
	 return item;
	 }
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURSTARTTIME;
	 item.indent++;
	 StringWithoutTrailingZeros(valStr,fUncertainStartTime/3600,6);
	 sprintf(text, "Start Time: %s hours",valStr);
	 return item;
	 }
	 
	 if (--n == 0) {
	 item.index = I_GRIDCURDURATION;
	 //item.bullet = BULLET_DASH;
	 item.indent++;
	 StringWithoutTrailingZeros(valStr,fDuration/3600,6);
	 sprintf(text, "Duration: %s hours",valStr);
	 return item;
	 }
	 
	 
	 }
	 
	 }  // uncertainty is on
	 
	 } // bOpen
	
	item.owner = 0;
	
	return item;
}*/

/*Boolean CurrentCycleMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
 {
 if (inBullet)
 switch (item.index) {
 case I_GRIDCURNAME: bOpen = !bOpen; return TRUE;
 case I_GRIDCURGRID: bShowGrid = !bShowGrid; 
 model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
 case I_GRIDCURARROWS: bShowArrows = !bShowArrows; 
 model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
 case I_GRIDCURUNCERTAINTY: bUncertaintyPointOpen = !bUncertaintyPointOpen; return TRUE;
 case I_GRIDCURACTIVE:
 bActive = !bActive;
 model->NewDirtNotification(); 
 return TRUE;
 }
 
 if (doubleClick && !inBullet)
 {
 Boolean userCanceledOrErr,timeFileChanged ;
 //(void) this -> SettingsDialog();
 CATSSettingsDialog (this, this -> moverMap, &timeFileChanged);
 return TRUE;
 }
 
 // do other click operations...
 
 return FALSE;
 }*/

/*Boolean CurrentCycleMover::FunctionEnabled(ListItem item, short buttonID)
{
	switch (item.index) {
		case I_GRIDCURNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
			}
			break;
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return GridCurrentMover::FunctionEnabled(item, buttonID);
}*/

/*OSErr CurrentCycleMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}*/

/*OSErr CurrentCycleMover::AddItem(ListItem item)
 {
 if (item.index == I_GRIDCURNAME)
 return TMover::AddItem(item);
 
 return 0;
 }*/

OSErr CurrentCycleMover::DeleteItem(ListItem item)
{
	if (item.index == I_GRIDCURNAME)
		return moverMap -> DropMover(dynamic_cast<CurrentCycleMover *>(this));
	
	return 0;
}

Boolean CurrentCycleMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant current, we can say "no"
	//if (model->GetModelMode()==ADVANCEDMODE && bShowGrid) depends = true;
	if(timeGrid->GetNumTimesInFile()==1 && !(timeGrid->GetNumFiles()>1)) depends = false;
	return depends;
}

void CurrentCycleMover::Draw(Rect r, WorldRect view) 
{	// will need to update once triangle format is set
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	
	short ebbFloodType;
	long offset;
	float fraction;
	fraction = 0; offset = 0;	// for now
	
	if (timeDep && bTimeFileActive) 
	{
		dynamic_cast<TShioTimeValue*> (timeDep) -> GetLocationInTideCycle(time,&ebbFloodType,&fraction);
		if (ebbFloodType>=fPatternStartPoint)
		{
			offset = ebbFloodType - fPatternStartPoint; // pass the fraction and offset to the timegrid
		}
		else
		{
			offset = ebbFloodType+4 - fPatternStartPoint;
		}
	}
	timeGrid -> SetTimeCycleInfo(fraction,offset);
	
	timeGrid->Draw(r,view,fCurScale,fArrowScale,fArrowDepth,bShowArrows,bShowGrid,fColor);
	/*if(fGrid && (bShowArrows || bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		TTriGridVel* triGrid = (TTriGridVel*)fGrid;
		//TopologyHdl topH = triGrid->GetTopologyHdl();
		long j,numTri;
		//Boolean offQuickDrawPlane = false;
		//numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
		numTri = triGrid->GetNumTriangles();
		
		err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
		
		if (err) {fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,arrowDepth,overrideDrawArrows,bShowGrid,fColor); return;}
		//if(err) return;

	 if (bShowGrid && !bShowArrows) fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,arrowDepth,overrideDrawArrows,bShowGrid,fColor);
		//fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,overrideDrawArrows,bShowGrid);
		
		if(bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;			
			
			if (bShowGrid)
			{	
				RGBForeColor(&colors[PURPLE]);
				
				for (j = 0 ; j< numTri; j++)
				{
					// get vertices, check if dry triangle, then draw gray
					Boolean isDryTri = dynamic_cast<CurrentCycleMover *>(this)->IsDryTri(j);
					if (model->GetModelMode()==ADVANCEDMODE && isDryTri)
					{
						RGBForeColor(&colors[LIGHTGRAY]);
						//RGBForeColor(&colors[RED]);
						triGrid->DrawTriangle(&r,j,TRUE);	// fill triangles					
						RGBForeColor(&colors[PURPLE]);
					}
					else
						triGrid->DrawTriangle(&r,j,FALSE);	// don't fill triangles
				}
				RGBForeColor(&colors[BLACK]);
			}
			//TTriGridVel* triGrid = (TTriGridVel*)fGrid;
			
			//err = this -> SetInterval(errmsg);
			//if(err) return;
			
			//loaded = this -> CheckInterval(timeDataInterval);
			//if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (fTimeAlpha==-1)
				{
					//Seconds relTime = time - model->GetStartTime();
					Seconds relTime = time - fModelStartTime;
					startTime = (*fTimeHdl)[fStartData.timeIndex];
					endTime = (*fTimeHdl)[fEndData.timeIndex];
					//timeAlpha = (endTime - time)/(double)(endTime - startTime);
					timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
				}
				else
					timeAlpha = fTimeAlpha;
			}
			
			for(i = 0; i < numVertices; i++)
			{
				RGBForeColor(&fColor);
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex = INDEXH(fVerdatToNetCDFH,i);
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.}, startVelocity = {0.,0.}, endVelocity = {0.,0.};
				Boolean offQuickDrawPlane = false, isDryPt1 = false, isDryPt2 = false;
				//long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case
				
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if(GetNumTimesInFile()==1)
				{
					//if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					//{
					//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
					//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
					velocity = GetStartVelocity(ptIndex,&isDryPt1);
					isDryPt2 = true;
					//}
				}
				else // time varying current
				{
					//if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					//{
					//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
					//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
					startVelocity = GetStartVelocity(ptIndex,&isDryPt1);
					endVelocity = GetEndVelocity(ptIndex,&isDryPt2);
					//if (timeAlpha==1) isDryPt1 = true;
					//if (timeAlpha==0) isDryPt2 = true;
					if (timeAlpha==1) isDryPt2 = true;
					if (timeAlpha==0) isDryPt1 = true;
					velocity.u = timeAlpha*startVelocity.u + (1-timeAlpha)*endVelocity.u;
					velocity.v = timeAlpha*startVelocity.v + (1-timeAlpha)*endVelocity.v;
					//}
				}
				if ((velocity.u != 0 || velocity.v != 0))
				{
					float inchesX = (velocity.u * refScale) / arrowScale;
					float inchesY = (velocity.v * refScale) / arrowScale;
					short pixX = inchesX * PixelsPerInchCurrent();
					short pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
				//else if (isDryPt1 && isDryPt2)
				if (bShowGrid && isDryPt1 && isDryPt2 && model->GetModelMode()==ADVANCEDMODE)
				{
					RGBForeColor(&colors[RED]);
					short offset = round(4*PixelsPerPoint()); // 2 points each way
					MyMoveTo(p.h-offset, p.v);MyLineTo(p.h+offset,p.v);
					MyMoveTo(p.h,p.v-offset);MyLineTo(p.h,p.v+offset);
					RGBForeColor(&colors[BLACK]);
				}
			}
			RGBForeColor(&colors[BLACK]);
		}
	}*/
}

/*void CurrentCycleMover::Draw(Rect r, WorldRect view) 
 {	// will need to update once triangle format is set
 WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
 double timeAlpha,depthAlpha;
 float topDepth,bottomDepth;
 Point p;
 Rect c;
 Seconds startTime,endTime;
 Seconds time = model->GetModelTime();
 OSErr err = 0;
 char errmsg[256];
 
 if(fGrid && (bShowArrows || bShowGrid))
 {
 Boolean overrideDrawArrows = FALSE;
 fGrid->Draw(r,view,wayOffMapPt,refScale,arrowScale,overrideDrawArrows,bShowGrid);
 if(bShowArrows)
 { // we have to draw the arrows
 long numVertices,i;
 LongPointHdl ptsHdl = 0;
 long timeDataInterval;
 Boolean loaded;
 TTriGridVel* triGrid = (TTriGridVel*)fGrid;
 
 err = this -> SetInterval(errmsg);
 if(err) return;
 
 //loaded = this -> CheckInterval(timeDataInterval);
 //if(!loaded) return;
 
 ptsHdl = triGrid -> GetPointsHdl();
 if(ptsHdl)
 numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
 else 
 numVertices = 0;
 
 // Check for time varying current 
 if(GetNumTimesInFile()>1)
 {
 // Calculate the time weight factor
 if (fTimeAlpha==-1)
 {
 Seconds relTime = time - model->GetStartTime();
 startTime = (*fTimeHdl)[fStartData.timeIndex];
 endTime = (*fTimeHdl)[fEndData.timeIndex];
 //timeAlpha = (endTime - time)/(double)(endTime - startTime);
 timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
 }
 else
 timeAlpha = fTimeAlpha;
 }
 
 for(i = 0; i < numVertices; i++)
 {
 // get the value at each vertex and draw an arrow
 LongPoint pt = INDEXH(ptsHdl,i);
 long ptIndex = INDEXH(fVerdatToNetCDFH,i);
 WorldPoint wp;
 Point p,p2;
 VelocityRec velocity = {0.,0.}, startVelocity = {0.,0.}, endVelocity = {0.,0.};
 Boolean offQuickDrawPlane = false, isDryPt1 = false, isDryPt2 = false;
 //long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case
 
 //GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
 
 //if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
 //continue;	// no value for this point at chosen depth
 
 //if (depthIndex2!=UNASSIGNEDINDEX)
 //{
 // Calculate the depth weight factor
 //topDepth = INDEXH(fDepthsH,depthIndex1);
 //bottomDepth = INDEXH(fDepthsH,depthIndex2);
 //depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
 //}
 
 wp.pLat = pt.v;
 wp.pLong = pt.h;
 
 p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
 
 // Check for constant current 
 if(GetNumTimesInFile()==1)
 {
 //if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
 //{
 //velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
 //velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
 velocity = GetStartVelocity(ptIndex,&isDryPt1);
 isDryPt2 = true;
 //}
 //else 	// below surface velocity
 //{
 //velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
 //velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
 //}
 }
 else // time varying current
 {
 //if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
 //{
 //velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
 //velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
 startVelocity = GetStartVelocity(ptIndex,&isDryPt1);
 endVelocity = GetEndVelocity(ptIndex,&isDryPt2);
 if (timeAlpha==1) isDryPt1 = true;
 if (timeAlpha==0) isDryPt2 = true;
 velocity.u = timeAlpha*startVelocity.u + (1-timeAlpha)*endVelocity.u;
 velocity.v = timeAlpha*startVelocity.v + (1-timeAlpha)*endVelocity.v;
 //}
 //else	// below surface velocity
 //{
 //velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
 //velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
 //velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
 //velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
 //}
 }
 if ((velocity.u != 0 || velocity.v != 0))
 {
 float inchesX = (velocity.u * refScale) / arrowScale;
 float inchesY = (velocity.v * refScale) / arrowScale;
 short pixX = inchesX * PixelsPerInchCurrent();
 short pixY = inchesY * PixelsPerInchCurrent();
 p2.h = p.h + pixX;
 p2.v = p.v - pixY;
 MyMoveTo(p.h, p.v);
 MyLineTo(p2.h, p2.v);
 MyDrawArrow(p.h,p.v,p2.h,p2.v);
 }
 //else if (isDryPt1 && isDryPt2)
 if (isDryPt1 && isDryPt2 && model->GetModelMode()==ADVANCED_MODE)
 {
 RGBForeColor(&colors[RED]);
 short offset = round(2*PixelsPerPoint()); // 2 points each way
 MyMoveTo(p.h-offset, p.v);MyLineTo(p.h+offset,p.v);
 MyMoveTo(p.h,p.v-offset);MyLineTo(p.h,p.v+offset);
 RGBForeColor(&colors[BLACK]);
 }
 }
 }
 }
 }*/
/////////////////////////////////////////////////////////////////


// this assumes a triangle mover
//OSErr CurrentCycleMover::TextRead(char *path, TMap **newMap) 
OSErr CurrentCycleMover::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// needs to be updated once triangle grid format is set
	
	OSErr err = 0;
	/*long i, numScanned;
	int status, ncid, nodeid, nbndid, bndid, neleid, latid, lonid, recid, timeid;
	size_t nodeLength, nbndLength, neleLength, recs, t_len;
	float timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0, month[10];	
	WORLDPOINTFH vertexPtsH=0;
	float *lat_vals=0,*lon_vals=0;
	short *bndry_indices=0, *bndry_nums=0, *bndry_type=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0,bndIndex[2]={0,0};
	static size_t pt_count, bnd_count[2];
	Seconds startTime, startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	char fileName[64],s[256],topPath[256],outPath[256];
	
	char *modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false;
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	tree.numBranches = 0;
	TDagTree *dagTree = 0;
	
	if (!path || !path[0]) return 0;
	strcpy(fPathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fFileName, fileName); // maybe use a name from the file
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) 
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) {err = -1; goto done;} 
	
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {err = -1; goto done;} 
	
	//status = nc_inq_attlen(ncid, recid, "units", &t_len);
	// code goes here, there will be a different time scale for tidal current cycle patterns
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		err = -1; goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10], junk2[10], junk3[10];
		
		timeUnits = new char[t_len+1];
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);// recid is the dimension id not the variable id
		status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		// now should be 'days since model start' - or hours, minutes,...
		// just store the seconds into the cycle and add model start time back in
		// when showing just the patterns
		numScanned=sscanf(timeUnits, "%s %s %s %s",
						  unitStr, junk, &junk2, &junk3) ;
		if (numScanned<4) // really only care about 1	
		{ err = -1; TechError("CurrentCycleMover::TextRead()", "sscanf() == 4", 0); goto done; }
		if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
			timeConversion = 3600.;
		else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
			timeConversion = 60.;
		else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
			timeConversion = 1.;
		else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
			timeConversion = 24*3600.;
	} 
	
	status = nc_inq_dimid(ncid, "node", &nodeid); 
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nodeid, &nodeLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimid(ncid, "nbnd", &nbndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "bnd", &bndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nbndid, &nbndLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	bnd_count[0] = nbndLength;
	bnd_count[1] = 1;
	bndry_indices = new short[nbndLength]; 
	bndry_nums = new short[nbndLength]; 
	bndry_type = new short[nbndLength]; 
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//bndIndex[1] = 0;
	bndIndex[1] = 1;	// take second point of boundary segments instead, so that water boundaries work out
	status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 2;
	status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 3;
	status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_type);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_dimid(ncid, "nele", &neleid);	
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required
	status = nc_inq_dimlen(ncid, neleid, &neleLength);
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required
	
	// option to use index values?
	status = nc_inq_varid(ncid, "lat", &latid);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "lon", &lonid);
	if (status != NC_NOERR) {err = -1; goto done;}
	pt_count = nodeLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(nodeLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	pts = (LongPointHdl)_NewHandleClear(nodeLength*sizeof(LongPoint));
	if (!pts) {err = memFullErr; goto done;}
	lat_vals = new float[nodeLength]; 
	lon_vals = new float[nodeLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_float(ncid, latid, &ptIndex, &pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, lonid, &ptIndex, &pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<nodeLength;i++)
	{
		INDEXH(vertexPtsH,i).pLat = lat_vals[i];	
		INDEXH(vertexPtsH,i).pLong = lon_vals[i];
		INDEXH(pts,i).v = (long)(lat_vals[i]*1e6);
		INDEXH(pts,i).h = (long)(lon_vals[i]*1e6);
	}
	fVertexPtsH	 = vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(nodeLength > 0)
		{
			WorldPoint  wp;
			for(i=0;i<nodeLength;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	startTime = model->GetStartTime();	// shouldn't need this, the time values are relative to whatever the model time is
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}

		
		newTime = (Seconds)(round(timeVal*timeConversion));
		INDEXH(fTimeHdl,i) = newTime;
		
	}
	
	fNumNodes = nodeLength;
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//err = this -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
	//if(err) goto done;
	
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
	
	
	// if location file don't bring up dialog ?
	//if (fTopFilePath[0]) {err = ReadTopology(fTopFilePath,newMap); goto done;}	// if location file specified path don't ask
	if (model->fWizard->HaveOpenWizardFile())
		//code goes here, come up with a graceful way to pass the message of the topology file path
		//{strcpy(fTopFilePath,"resnum 10009"); err = ReadTopology(fTopFilePath,newMap); goto done;}
	{if (topFilePath[0]) {strcpy(fTopFilePath,topFilePath); err = ReadTopology(fTopFilePath,newMap); goto done;}}
	if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto done;}
	//if (model->fWizard->HaveOpenWizardFile())
	//{	// there is a location file
	//if (fTopPath[0]) err = ReadTopology(topPath,newMap);	// if location file specified path don't ask
	//else err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	
	//goto done;
	//}
	
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	if (true)
	{
		short buttonSelected;
		buttonSelected  = MULTICHOICEALERT(1688,"Do you have an extended topology file to load?",FALSE);
		switch(buttonSelected){
			case 1: // there is an extended top file
				bTopFile = true;
				break;  
			case 3: // no extended top file
				bTopFile = false;
				break;
			case 4: // cancel
				err=-1;// stay at this dialog
				goto done;
		}
	}
	if(bTopFile)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) 
		{
			err = dynamic_cast<CurrentCycleMover *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			//err = dynamic_cast<CurrentCycleMover *>(this)->ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}
		else
			strcpy(topPath, reply.fullPath);
#else
		where = CenteredDialogUpLeft(M38c);
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) 
		{
			err = dynamic_cast<CurrentCycleMover *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			goto done;	
			//return 0;
		}
		
		my_p2cstr(reply.fName);
		
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
#else
		strcpy(topPath, reply.fName);
#endif
#endif		
		strcpy (s, topPath);
		err = ReadTopology(topPath,newMap);	// newMap here
		goto done;
		//SplitPathFile (s, fileName);
	}
	
	err = dynamic_cast<CurrentCycleMover *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
	
	
	
done:
	if (err)
	{
		if (!errmsg[0]) 
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;	fVertexPtsH	 = 0;}
	}
	//printNote("NetCDF triangular grid model current mover is not yet implemented");
	
	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (bndry_indices) delete [] bndry_indices;
	if (bndry_nums) delete [] bndry_nums;
	if (bndry_type) delete [] bndry_type;
	*/
	return err;
}


/////////////////////////////////////////////////
//OSErr CurrentCycleMover::ReadTopology(char* path, TMap **newMap)
/*OSErr CurrentCycleMover::ReadTopology(vector<string> &linesInFile, TMap **newMap)
{
	// import NetCDF triangle info so don't have to regenerate
	// this is same as curvilinear mover so may want to combine later
	char s[1024], errmsg[256];
	long i, numPoints, numTopoPoints, line = 0, numPts;
	string currentLine;
	//CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	currentLine = linesInFile[line++];
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	//if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	if(IsTransposeArrayHeaderLine(currentLine,numPts)) // 
	{
		//if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		if (err = ReadTransposeArray(linesInFile,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	
	if(err = ReadTVertices(linesInFile,&line,&pts,&depths,errmsg)) goto done;
	//if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;
	
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	MySpinCursor();
	
	currentLine = linesInFile[line++];
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	//if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	if(IsBoundarySegmentHeaderLine(currentLine,numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundarySegs>0)
			//err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
			err = ReadBoundarySegs(linesInFile,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		currentLine = linesInFile[line++];
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	
	//if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	if(IsWaterBoundaryHeaderLine(currentLine,numWaterBoundaries,numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		//err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		err = ReadWaterBoundaries(linesInFile,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		currentLine = linesInFile[line++];
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Water boundaries header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	//if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	if(IsTTopologyHeaderLine(currentLine,numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		//err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		err = ReadTTopologyBody(linesInFile,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		currentLine = linesInFile[line++];
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	//if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	if(IsTIndexedDagTreeHeaderLine(currentLine,numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		//err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		err = ReadTIndexedDagTreeBody(linesInFile,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg,"Error in dag tree header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	if (waterBoundaries && (this -> moverMap == model -> uMap))
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		//PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap(fPathName,bounds); // the map bounds are the same as the grid bounds, could use fFileName here
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetWaterBoundaries(waterBoundaries);
		
		*newMap = map;
	}
	
	//if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	else	// maybe assume rectangle grids will have map?
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//depths = 0;
	
done:
	
	if(depths) {DisposeHandle((Handle)depths); depths=0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (*newMap) 
		{
			(*newMap)->Dispose();
			delete *newMap;
			*newMap=0;
		}
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
	}
	return err;
}*/


/////////////////////////////////////////////////
/////////////////////////////////////////////////

CurrentCycleDialogNonPtrFields GetCurrentCycleDialogNonPtrFields(CurrentCycleMover	* cm)
{
	CurrentCycleDialogNonPtrFields f;
	
	f.fUncertainStartTime  = cm->fUncertainStartTime; 	
	f.fDuration  = cm->fDuration; 
	//
	f.refP  = cm->refP; 	
	f.refZ  = cm->refZ; 	
	f.scaleType = cm->scaleType; 
	f.scaleValue = cm->scaleValue;
	strcpy(f.scaleOtherFile,cm->scaleOtherFile);
	f.refScale = cm->refScale; 
	f.bTimeFileActive = cm->bTimeFileActive; 
	f.bShowGrid = cm->bShowGrid; 
	f.bShowArrows = cm->bShowArrows; 
	f.arrowScale = cm->fArrowScale; 
	f.fEddyDiffusion = cm->fEddyDiffusion; 
	f.fEddyV0 = cm->fEddyV0; 
	f.fDownCurUncertainty = cm->fDownCurUncertainty; 
	f.fUpCurUncertainty = cm->fUpCurUncertainty; 
	f.fRightCurUncertainty = cm->fRightCurUncertainty; 
	f.fLeftCurUncertainty = cm->fLeftCurUncertainty; 
	return f;
}

void SetCurrentCycleDialogNonPtrFields(CurrentCycleMover	* cm,CurrentCycleDialogNonPtrFields * f)
{
	cm->fUncertainStartTime = f->fUncertainStartTime; 	
	cm->fDuration  = f->fDuration; 
	//
	cm->refP = f->refP; 	
	cm->refZ  = f->refZ; 	
	cm->scaleType = f->scaleType; 
	cm->scaleValue = f->scaleValue;
	strcpy(cm->scaleOtherFile,f->scaleOtherFile);
	cm->refScale = f->refScale; 
	cm->bTimeFileActive = f->bTimeFileActive; 
	cm->bShowGrid = f->bShowGrid; 
	cm->bShowArrows = f->bShowArrows; 
	cm->fArrowScale = f->arrowScale; 
	cm->fEddyDiffusion = f->fEddyDiffusion;
	cm->fEddyV0 = f->fEddyV0;
	cm->fDownCurUncertainty = f->fDownCurUncertainty; 
	cm->fUpCurUncertainty = f->fUpCurUncertainty; 
	cm->fRightCurUncertainty = f->fRightCurUncertainty; 
	cm->fLeftCurUncertainty = f->fLeftCurUncertainty; 
}

///////////////////////////////////////////////////////////////////////////

void ShowUnscaledValue2(DialogPtr dialog)
{
	double length;
	WorldPoint p;
	WorldPoint3D p3D = {0,0,0.};
	VelocityRec velocity = {0.,0.};
	
	(void)EditTexts2LL(dialog, M16LATDEGREES, &p, FALSE);
	p3D.p = p;
	//velocity = sharedCCMover->timeGrid->GetPatValue(p3D);
	//velocity = sharedCCMover->timeGrid->GetScaledPatValue(model->GetModelTime(),p3D);
	length = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	Float2EditText(dialog, M16UNSCALEDVALUE, length, 4);
}

Boolean CurrentCycleMover::OkToAddToUniversalMap()
{
	// only allow this if we have grid with valid bounds
	WorldRect gridBounds;
	if (!timeGrid->fGrid) {
		printError("Error in CurrentCycleMover::OkToAddToUniversalMap.");
		return false;
	}
	gridBounds = timeGrid -> fGrid -> GetBounds();
	if(EqualWRects(gridBounds,emptyWorldRect)) {
		printError("You cannot create a universal mover from a current file which does not specify the grid's bounds.");
		return false;
	}
	return true;
}



OSErr CurrentCycleMover::InitMover(TimeGridVel *grid)
{
	OSErr	err = noErr;
	//timeGrid = grid;
	timeDep = 0;
	refP.pLat = 0;
	refP.pLong = 0;
	refZ = 0;
	scaleType = SCALE_NONE;
	scaleValue = 1.0;
	scaleOtherFile[0] = 0;
	bRefPointOpen = FALSE;
	bUncertaintyPointOpen = FALSE;
	bTimeFileOpen = FALSE;
	bShowArrows = FALSE;
	bShowGrid = FALSE;
	fArrowScale = 1;// debra wanted 10, CJ wanted 5, JLM likes 5 too (was 1)
	// CJ wants it back to 1, 4/11/00

	bApplyLogProfile = false;
	fArrowDepth = 0.;	// if want to show subsurface velocity for log profile
	
	//dynamic_cast<TCATSMover *>(this)->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
	
	err = GridCurrentMover::InitMover (grid);
	return err;
}

OSErr CurrentCycleMover::ReplaceMover()
{
	OSErr err = 0;
/*	CurrentCycleMover* mover = CreateAndInitCatsCurrentsMover (this -> moverMap,true,0,0); // only allow to replace with same type of mover
	if (mover)
	{
		// save original fields
		CurrentCycleDialogNonPtrFields fields = GetCurrentCycleDialogNonPtrFields(dynamic_cast<CurrentCycleMover *>(this));
		SetCurrentCycleDialogNonPtrFields(mover,&fields);
		if(this->timeDep)
		{
			err = this->timeDep->MakeClone(&mover->timeDep);
			if (err) { delete mover; mover=0; return err; }
			// check if shio or hydrology, save ref point 
			//if (!(this->timeDep->GetFileType() == OSSMTIMEFILE)) 
			//mover->refP = this->refP;
			//mover->bTimeFileActive=true;
			// code goes here , should replace all the fields?
			//mover->scaleType = this->scaleType;
			//mover->scaleValue = this->scaleValue;
			//mover->refScale = this->refScale;
			//strcpy(mover->scaleOtherFile,this->scaleOtherFile);
		}
		if (err = this->moverMap->AddMover(mover,0))
		{mover->Dispose(); delete mover; mover = 0; return err;}
		if (err = this->moverMap->DropMover(dynamic_cast<TCATSMover *>(this)))
		{mover->Dispose(); delete mover; mover = 0; return err;}
	}
	else 
	{err = -1; return err;}
*/	
	model->NewDirtNotification();
	return err;
}

/////////////////////////////////////////////////
long CurrentCycleMover::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 4;		// minimum CATS mover lines
		if (gNoaaVersion) count++;	// apply log profile
		if (timeDep)count++;
		if (bRefPointOpen) count += 3;
		if(model->IsUncertain())count++;
		if(bUncertaintyPointOpen && model->IsUncertain())count +=6;
		// add 1 to # of time-values for active / inactive
		
		// JLM if (bTimeFileOpen) count += timeDep ? timeDep -> GetNumValues () : 0;
		if (bTimeFileOpen) count += timeDep ? (1 + timeDep -> GetListLength ()) : 0; //JLM, add 1 for the active flag
	}
	
	return count;
}

ListItem CurrentCycleMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char *p, latS[20], longS[20], valStr[32];
	ListItem item = { this, 0, indent, 0 };
	
	if (n == 0) {
		item.index = I_CATSNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		//		sprintf(text, "CATS: \"%s\"", className);
		sprintf(text, "Currents: \"%s\"", className);
		*style = bActive ? italic : normal;
		
		return item;
	}
	
	item.indent++;
	
	if (bOpen) {
		
		
		if (--n == 0) {
			item.index = I_CATSACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		
		if (--n == 0) {
			item.index = I_CATSGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			sprintf(text, "Show Grid");
			
			return item;
		}
		
		if (--n == 0) {
			item.index = I_CATSARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,fArrowScale,6);
			sprintf(text, "Show Velocities (@ 1 in = %s m/s)", valStr);
			
			return item;
		}
		
		if (gNoaaVersion)
		{
		if (--n == 0) {
			item.index = I_CATSLOGPROFILE;
			item.bullet = bApplyLogProfile ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			//StringWithoutTrailingZeros(valStr,arrowScale,6);
			sprintf(text, "Apply log profile");
			
			return item;
		}
		}
		
		if (--n == 0) {
			item.index = I_CATSREFERENCE;
			item.bullet = bRefPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Reference Point");
			
			return item;
		}
		
		
		if (bRefPointOpen) {
			if (--n == 0) {
				item.index = I_CATSSCALING;
				//item.bullet = BULLET_DASH;
				item.indent++;
				switch (scaleType) {
					case SCALE_NONE:
						strcpy(text, "No reference point scaling");
						break;
					case SCALE_CONSTANT:
						if (timeDep && timeDep->GetFileType()==HYDROLOGYFILE)
							StringWithoutTrailingZeros(valStr,refScale,6);
						//StringWithoutTrailingZeros(valStr,timeDep->fScaleFactor,6);
						else
							StringWithoutTrailingZeros(valStr,scaleValue,6);
						sprintf(text, "Scale to: %s ", valStr);
						// units
						if (timeDep)
							strcat(text,"* file value");
						else
							strcat(text,"m/s");
						break;
					case SCALE_OTHERGRID:
						sprintf(text, "Scale to grid: %s", scaleOtherFile);
						break;
				}
				
				return item;
			}
			
			n--;
			
			if (n < 2) {
				item.indent++;
				item.index = (n == 0) ? I_CATSLAT : I_CATSLONG;
				//item.bullet = BULLET_DASH;
				WorldPointToStrings(refP, latS, longS);
				strcpy(text, (n == 0) ? latS : longS);
				
				return item;
			}
			
			n--;
		}
		
		
		if(timeDep)
		{
			if (--n == 0)
			{
				char	timeFileName [kMaxNameLen];
				
				item.index = I_CATSTIMEFILE;
				item.bullet = bTimeFileOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				timeDep -> GetTimeFileName (timeFileName);
				if (timeDep -> GetFileType() == HYDROLOGYFILE)
					sprintf(text, "Hydrology File: %s", timeFileName);
				else
					sprintf(text, "Tide File: %s", timeFileName);
				if(!bTimeFileActive)*style = italic; // JLM 6/14/10
				return item;
			}
		}
		
		if (bTimeFileOpen && timeDep) {
			
			if (--n == 0)
			{
				item.indent++;
				item.index = I_CATSTIMEFILEACTIVE;
				item.bullet = bTimeFileActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Active");
				
				return item;
			}
			
			///JLM  ///{
			// Note: n is one higher than JLM expected it to be 
			// (the CATS mover code is pre-decrementing when checking)
			if(timeDep -> GetListLength () > 0)
			{	// only check against the entries if we have some 
				n--; // pre-decrement
				if (n < timeDep -> GetListLength ()) {
					item.indent++;
					item = timeDep -> GetNthListItem(n,item.indent,style,text);
					// over-ride the objects answer ??  JLM
					// no 10/23/00 
					//item.owner = this; // so the clicks come to me
					//item.index = I_CATSTIMEENTRIES + n;
					//////////////////////////////////////
					//item.bullet = BULLET_DASH;
					return item;
				}
				n -= timeDep -> GetListLength ()-1; // the -1 is to leave the count one higher so they can pre-decrement
			}
			////}
			
		}
		
		if(model->IsUncertain())
		{
			if (--n == 0) {
				item.index = I_CATSUNCERTAINTY;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				
				return item;
			}
			
			if (bUncertaintyPointOpen) {
				
				if (--n == 0) {
					item.index = I_CATSSTARTTIME;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainParams.startTimeInHrs/3600.));
					return item;
				}
				
				if (--n == 0) {
					item.index = I_CATSDURATION;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Duration: %.2f hours",fUncertainParams.durationInHrs/3600);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_CATSDOWNCUR;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Down Current: %.2f to %.2f %%",fUncertainParams.alongCurUncertainty*-100, fUncertainParams.alongCurUncertainty*100);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_CATSCROSSCUR;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Cross Current: %.2f to %.2f %%",fUncertainParams.crossCurUncertainty*-100,fUncertainParams.crossCurUncertainty*100);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_CATSDIFFUSIONCOEFFICIENT;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Eddy Diffusion: %.2e cm^2/sec",fEddyDiffusion);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_CATSEDDYV0;
					//item.bullet = BULLET_DASH;
					item.indent++;
					sprintf(text, "Eddy V0: %.2e m/sec",fEddyV0);
					return item;
				}
				
			}
			
		}
	}
	
	item.owner = 0;
	
	return item;
}

Boolean CurrentCycleMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	Boolean timeFileChanged = false;
	if (inBullet)
		switch (item.index) {
			case I_CATSNAME: bOpen = !bOpen; return TRUE;
			case I_CATSGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_CATSARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_CATSREFERENCE: bRefPointOpen = !bRefPointOpen; return TRUE;
			case I_CATSUNCERTAINTY: bUncertaintyPointOpen = !bUncertaintyPointOpen; return TRUE;
			case I_CATSTIMEFILE: bTimeFileOpen = !bTimeFileOpen; return TRUE;
			case I_CATSTIMEFILEACTIVE: bTimeFileActive = !bTimeFileActive; 
				model->NewDirtNotification(); return TRUE;
			case I_CATSACTIVE:
				bActive = !bActive;
				model->NewDirtNotification(); 
				if (!bActive && bTimeFileActive)
				{
					// deactivate time file if main mover is deactivated
					//					bTimeFileActive = false;
					//					VLUpdate (&objects);
				}
				return TRUE;
			case I_CATSLOGPROFILE:
				bApplyLogProfile = !bApplyLogProfile; return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_CATSNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
	}
	
	if (doubleClick && !inBullet)
	{
		switch(item.index)
		{
			case I_CATSSTARTTIME:
			case I_CATSDURATION:
			case I_CATSDOWNCUR:
			case I_CATSCROSSCUR:
			case I_CATSDIFFUSIONCOEFFICIENT:
			case I_CATSEDDYV0:
			case I_CATSUNCERTAINTY:
			{
				Boolean userCanceledOrErr, uncertaintyValuesChanged=false ;
				CurrentUncertainyInfo info  = this -> GetCurrentUncertaintyInfo();
				userCanceledOrErr = CurrentUncertaintyDialog(&info,mapWindow,&uncertaintyValuesChanged);
				if(!userCanceledOrErr) 
				{
					if (uncertaintyValuesChanged)
					{
						this->SetCurrentUncertaintyInfo(info);
						// code goes here, if values have changed needToReInit in UpdateUncertainty
						dynamic_cast<TCATSMover *>(this)->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
					}
				}
				return TRUE;
				break;
			}
			default:
				CurrentCycleSettingsDialog (dynamic_cast<CurrentCycleMover *>(this), this -> moverMap, &timeFileChanged);
				return TRUE;
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean CurrentCycleMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i,n,j,num;
	//TMover* mover,mover2;
	switch (item.index) {
		case I_CATSNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					// need a way to check if mover is part of a Compound Mover - thinks it's just a currentmover
					
					//if (bIAmPartOfACompoundMover)
						//return TCurrentMover::FunctionEnabled(item, buttonID);
					
					
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
					break;
			}
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return GridCurrentMover::FunctionEnabled(item, buttonID);
}

OSErr CurrentCycleMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

TOSSMTimeValue *sTimeValue2;

static PopInfoRec HydrologyPopTable[] = {
	{ M32, nil, M32INFOTYPEPOPUP, 0, pHYDROLOGYINFO, 0, 1, FALSE, nil },
	{ M32, nil, M32TRANSPORT1UNITS, 0, pTRANSPORTUNITS, 0, 1, FALSE, nil },
	//{ M32, nil, M32TRANSPORT2UNITS, 0, pSPEEDUNITS, 0, 1, FALSE, nil },
	{ M32, nil, M32VELOCITYUNITS, 0, pSPEEDUNITS2, 0, 1, FALSE, nil }
};

double ConvertTransportUnitsToCMS2(short transportUnits)
{
	double conversionFactor = 1.;
	switch(transportUnits)
	{
		case 1: conversionFactor = 1.0; break;	// CMS
		case 2: conversionFactor = 1000.; break;	// KCMS
		case 3: conversionFactor = .3048*.3048*.3048; break;	// CFS
		case 4: conversionFactor = .3048*.3048*.3048 * 1000.; break; // KCFS
			//default: err = -1; goto done;
	}
	return conversionFactor;
}

void ShowHideHydrologyDialogItems2(DialogPtr dialog)
{
	Boolean showTransport1Items, showTransport2Items;
	short typeOfInfoSpecified = GetPopSelection(dialog, M32INFOTYPEPOPUP);
	
	switch (typeOfInfoSpecified)
	{
		default:
			//case HAVETRANSPORT:
		case 1:
			showTransport1Items=TRUE;
			showTransport2Items=FALSE;
			break;
			//case HAVETRANSPORTANDVELOCITY:
		case 2:
			showTransport2Items=TRUE;
			showTransport1Items=FALSE;
			break;
	}
	ShowHideDialogItem(dialog, M32TRANSPORT1LABELA, showTransport1Items ); 
	ShowHideDialogItem(dialog, M32TRANSPORT1, true); 
	ShowHideDialogItem(dialog, M32TRANSPORT1UNITS, true); 
	ShowHideDialogItem(dialog, M32TRANSPORT1LABELB, showTransport1Items); 
	
	ShowHideDialogItem(dialog, M32TRANSPORT2LABELA, showTransport2Items); 
	//ShowHideDialogItem(dialog, M32TRANSPORT2, showTransport2Items); 
	//ShowHideDialogItem(dialog, M32TRANSPORT2UNITS, showTransport2Items); 
	ShowHideDialogItem(dialog, M32TRANSPORT2LABELB, showTransport2Items); 
	ShowHideDialogItem(dialog, M32VELOCITY, showTransport2Items); 
	ShowHideDialogItem(dialog, M32VELOCITYUNITS, showTransport2Items); 
}

short HydrologyClick2(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	
	VelocityRec refVel;
	long menuID_menuItem;
	switch(itemNum)
	{
		case M32OK:
		{						
			double userVelocity = 0, userTransport = 0, transportConversionFactor = 1, scaleFactor = 0, conversionFactor/*, origScaleFactor*/;
			short typeOfInfoSpecified = GetPopSelection(dialog, M32INFOTYPEPOPUP);
			short transportUnits = GetPopSelection(dialog, M32TRANSPORT1UNITS);
			WorldPoint3D refPoint3D = {0,0,0.};
			// code goes here, calculations to determine scaling factor based on inputs
			
			userTransport = EditText2Float(dialog, M32TRANSPORT1);
			if (userTransport == 0)
			{
				printError("You must enter a value for the transport");
				break;
			}
			if (typeOfInfoSpecified == 2)
			{
				userVelocity = EditText2Float(dialog, M32VELOCITY);
				if (userVelocity == 0)
				{
					printError("You must enter a value for the velocity");
					break;
				}
			}
			//sTimeValue->fUserUnits is the file units
			transportConversionFactor = ConvertTransportUnitsToCMS2(sTimeValue2->fUserUnits) / ConvertTransportUnitsToCMS2(transportUnits);
			// get value at reference point and calculate scale factor
			// need units conversion for transport and velocity
			refPoint3D.p = sTimeValue2->fStationPosition;
			//refVel = ((TCATSMover*)(TTimeValue*)sTimeValue2->owner)->GetPatValue(sTimeValue2->fStationPosition);
			refVel = ((TCATSMover*)(TTimeValue*)sTimeValue2->owner)->GetPatValue(refPoint3D);
			//origScaleFactor = sTimeValue2->fScaleFactor;
			if (typeOfInfoSpecified == 1)
			{
				//scaleFactor = 1./ (userTransport * transportConversionFactor);
				scaleFactor = transportConversionFactor / userTransport;
				//sTimeValue2->fScaleFactor = 1./ (userTransport * transportConversionFactor);
			}
			else if (typeOfInfoSpecified == 2)
			{
				double refSpeed = sqrt(refVel.u*refVel.u + refVel.v*refVel.v);
				short velUnits = GetPopSelection(dialog, M32VELOCITYUNITS);
				switch(velUnits)
				{
					case kKnots: conversionFactor = KNOTSTOMETERSPERSEC; break;
					case kMilesPerHour: conversionFactor = MILESTOMETERSPERSEC; break;
					case kMetersPerSec: conversionFactor = 1.0; break;
						//default: err = -1; goto done;
				}
				if (refSpeed > 1e-6) // any error if not? default = 0? 1? ...
					scaleFactor = (userVelocity * conversionFactor)/(userTransport * transportConversionFactor * refSpeed); // maybe an error if refSpeed too small
				//sTimeValue2->fScaleFactor = (userVelocity * conversionFactor)/(userTransport * transportConversionFactor * refSpeed); // maybe an error if refSpeed too small
			}
			//sTimeValue2->RescaleTimeValues(origScaleFactor, scaleFactor);
			sTimeValue2->fScaleFactor = scaleFactor;
			//sTimeValue2->fTransport = userTransport * transportConversionFactor;
			sTimeValue2->fTransport = userTransport * ConvertTransportUnitsToCMS2(transportUnits);
			sTimeValue2->fVelAtRefPt = userVelocity * conversionFactor;
			
			return M32OK;
		}
		case M32CANCEL:
			return M32CANCEL;
			break;
			
		case M32TRANSPORT1:
			//case M32TRANSPORT2:
		case M32VELOCITY:		
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;
			
		case M32INFOTYPEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideHydrologyDialogItems2(dialog);
			break;
			
		case M32TRANSPORT1UNITS:
			//case M32TRANSPORT2UNITS:
		case M32VELOCITYUNITS:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
	}
	return 0;
}

OSErr HydrologyInit2(DialogPtr dialog, VOIDPTR data)
{
#pragma unused (data)
	char roundLat,roundLong;
	char posStr[64], latStr[64], longStr[64], unitStr[64];
	
	SetDialogItemHandle(dialog, M32HILITEDEFAULT, (Handle)FrameDefault);
	//SetDialogItemHandle(dialog, M32FROST, (Handle)FrameEmbossed);
	
	//RegisterPopTable (HydrologyPopTable, sizeof (HydrologyPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M32, dialog);
	
	SetPopSelection (dialog, M32INFOTYPEPOPUP, 1);
	
	SetPopSelection (dialog, M32TRANSPORT1UNITS, 1);
	//SetPopSelection (dialog, M32TRANSPORT2UNITS, 1);
	SetPopSelection (dialog, M32VELOCITYUNITS, 1);
	
	ShowHideHydrologyDialogItems2(dialog);
	
	mysetitext(dialog, M32FILENAME, sTimeValue2->fStationName);
	settings.latLongFormat = DEGREES;
	WorldPointToStrings2(sTimeValue2->fStationPosition, latStr, &roundLat, longStr, &roundLong);	
	SimplifyLLString(longStr, 3, roundLong);
	SimplifyLLString(latStr, 3, roundLat);
	sprintf(posStr, "%s, %s", latStr,longStr);
	mysetitext(dialog, M32POSITION, posStr);
	
	ConvertToTransportUnits(sTimeValue2->fUserUnits,unitStr);
	mysetitext(dialog, M32UNITS, unitStr);
	MySelectDialogItemText(dialog, M32TRANSPORT1, 0, 255);
	
	return 0;
}

OSErr HydrologyDialog2(TOSSMTimeValue *dialogTimeFileData, WindowPtr parentWindow)
{
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[20];
	
	if(parentWindow == nil) parentWindow = mapWindow; // we need the parent on the IBM
	sTimeValue2 = dialogTimeFileData;
	
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(HydrologyPopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = HydrologyPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);
	
	item = MyModalDialog(M32, parentWindow, 0, HydrologyInit2, HydrologyClick2);
	RestorePopTableInfo(saveTable);
	if (item == M32OK) {
		dialogTimeFileData = sTimeValue2;
		if(parentWindow == mapWindow) {
			model->NewDirtNotification(); // when a dialog is the parent, we rely on that dialog to notify about Dirt 
			// that way we don't get the map redrawing behind the parent dialog on the IBM
		}
	}
	if (item == M32CANCEL) {return USERCANCEL;}
	return item == M32OK? 0 : -1;
}

void DisposeDialogTimeDep2(void)
{
	if (sharedCCMDialogTimeDep) {
		if(sharedCCMDialogTimeDep != sharedCCMover->timeDep)
		{	// only dispose of this if it is different
			sharedCCMDialogTimeDep->Dispose();
			delete sharedCCMDialogTimeDep;
		}
		sharedCCMDialogTimeDep = nil;
	}
}

void ShowCurrentCycleDialogUnitLabels(DialogPtr dialog)
{
	char scaleToUnitsStr[64] = "";
	char fileUnitsStr[64] = "";
	double scaleFactor,transport,velAtRefPt;
	short fileType;
	
	if (GetPopSelection (dialog, M16TIMEFILETYPES) == NOTIMEFILE) // scaling to a velocity
		strcpy(scaleToUnitsStr,"m/s at reference point");
	else // scaling to multiple of the file value
		strcpy(scaleToUnitsStr,"* file value at reference point");
	mysetitext(dialog, M16SCALETOVALUEUNITS, scaleToUnitsStr);
	
	if(sharedCCMDialogTimeDep) 
	{
		scaleFactor = sharedCCMDialogTimeDep->fScaleFactor;
		transport = sharedCCMDialogTimeDep->fTransport;
		velAtRefPt = sharedCCMDialogTimeDep->fVelAtRefPt;
		fileType = sharedCCMDialogTimeDep->GetFileType();
		if (fileType == SHIOHEIGHTSFILE || fileType == PROGRESSIVETIDEFILE)
		{
			Float2EditText(dialog, M16TIMEFILESCALEFACTOR, scaleFactor, 4);
		}
		else if (fileType == HYDROLOGYFILE)
		{
			Float2EditText(dialog, M16HYDROLOGYSCALEFACTOR, scaleFactor, 4);
			Float2EditText(dialog, M16TRANSPORT, transport, 4);
			Float2EditText(dialog, M16VELATREFPT, velAtRefPt, 4);
			
		}
		else 
		{
			ConvertToUnitsShort (sharedCCMDialogTimeDep->GetUserUnits(), fileUnitsStr);
			mysetitext(dialog, M16TIMEFILEUNITS, fileUnitsStr);
		}
	}
}

void ShowHideCurrentCycleDialogItems(DialogPtr dialog)
{
	short fileType = OSSMTIMEFILE;
	
	if ((sharedCCMover->timeGrid->fGrid->GetClassID()==TYPE_TRIGRIDVEL))	// only allow for catsmovers on triangle grid for now
		//if (!(sharedCCMover->IAm(TYPE_GRIDCURMOVER)))
		ShowHideDialogItem(dialog, M16REPLACEMOVER, true); 
	else 
		ShowHideDialogItem(dialog, M16REPLACEMOVER, false); 
	
	if(!sharedCCMDialogTimeDep)
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, false); 
		ShowHideDialogItem(dialog, M16TIMEFILENAMELABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILENAME, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
		ShowHideDialogItem(dialog, M16TRANSPORT, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
		ShowHideDialogItem(dialog, M16VELLABEL, false); 
		ShowHideDialogItem(dialog, M16VELATREFPT, false); 
		ShowHideDialogItem(dialog, M16VELUNITS, false); 
		return;
	}
	else
	{
		ShowHideDialogItem(dialog, M16TIMEFILENAMELABEL, true); 
		ShowHideDialogItem(dialog, M16TIMEFILENAME, true); 	
	}
	
	if(sharedCCMDialogTimeDep)
		fileType = sharedCCMDialogTimeDep->GetFileType();
	
	if(fileType==SHIOHEIGHTSFILE || fileType==PROGRESSIVETIDEFILE)
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, true); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, true); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
		ShowHideDialogItem(dialog, M16TRANSPORT, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
		ShowHideDialogItem(dialog, M16VELLABEL, false); 
		ShowHideDialogItem(dialog, M16VELATREFPT, false); 
		ShowHideDialogItem(dialog, M16VELUNITS, false); 
	}
	else if(fileType==HYDROLOGYFILE)
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, false); 
		if(sharedCCMDialogTimeDep->bOSSMStyle)
		{
			ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
			ShowHideDialogItem(dialog, M16TRANSPORT, false); 
			ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
			ShowHideDialogItem(dialog, M16VELLABEL, false); 
			ShowHideDialogItem(dialog, M16VELATREFPT, false); 
			ShowHideDialogItem(dialog, M16VELUNITS, false); 
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, true); 
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, true); 
		}
		else
		{
			ShowHideDialogItem(dialog, M16TRANSPORTLABEL, true); 
			ShowHideDialogItem(dialog, M16TRANSPORT, true); 
			ShowHideDialogItem(dialog, M16TRANSPORTUNITS, true); 
			if(sharedCCMDialogTimeDep->fVelAtRefPt!=0)
			{
				ShowHideDialogItem(dialog, M16VELLABEL, true); 
				ShowHideDialogItem(dialog, M16VELATREFPT, true);
				ShowHideDialogItem(dialog, M16VELUNITS, true); 
			}
			else
			{
				ShowHideDialogItem(dialog, M16VELLABEL, false); 
				ShowHideDialogItem(dialog, M16VELATREFPT, false); 
				ShowHideDialogItem(dialog, M16VELUNITS, false); 
			}
			
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
			ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		}
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, false); 
	}
	else 
	{
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16TIMEFILESCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTORLABEL, false); 
		ShowHideDialogItem(dialog, M16HYDROLOGYSCALEFACTOR, false); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITSLABEL, true); 
		ShowHideDialogItem(dialog, M16TIMEFILEUNITS, true); 
		ShowHideDialogItem(dialog, M16TRANSPORTLABEL, false); 
		ShowHideDialogItem(dialog, M16TRANSPORT, false); 
		ShowHideDialogItem(dialog, M16TRANSPORTUNITS, false); 
		ShowHideDialogItem(dialog, M16VELLABEL, false); 
		ShowHideDialogItem(dialog, M16VELATREFPT, false); 
		ShowHideDialogItem(dialog, M16VELUNITS, false); 
	}
}

void ShowHideScaleFactorItems2(DialogPtr dialog)
{
	short fileType = OSSMTIMEFILE;
	
	if(sharedCCMDialogTimeDep)
		fileType = sharedCCMDialogTimeDep->GetFileType();
	
	if (sharedCCMDialogTimeDep && fileType == HYDROLOGYFILE)
	{
		ShowHideDialogItem(dialog, M16NOSCALING, false); 
		ShowHideDialogItem(dialog, M16SCALETOCONSTANT, false); 
		ShowHideDialogItem(dialog, M16SCALEVALUE, false); 
		ShowHideDialogItem(dialog, M16SCALETOGRID, false); 
		ShowHideDialogItem(dialog, M16SCALEGRIDNAME, false); 
		ShowHideDialogItem(dialog, M16SCALETOVALUEUNITS, false);
	}
	else
	{
		ShowHideDialogItem(dialog, M16NOSCALING, true); 
		ShowHideDialogItem(dialog, M16SCALETOCONSTANT, true); 
		ShowHideDialogItem(dialog, M16SCALEVALUE, true); 
		ShowHideDialogItem(dialog, M16SCALETOGRID, true); 
		ShowHideDialogItem(dialog, M16SCALEGRIDNAME, true); 
		ShowHideDialogItem(dialog, M16SCALETOVALUEUNITS, true); 
	}
	
}

short CurrentCycleClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Boolean changed;
	short item;
	WorldPoint p;
	TOSSMTimeValue *timeFile;
	OSErr err = 0;
	double scaleValue;
	
	StandardLLClick(dialog, itemNum, M16LATDEGREES, M16DEGREES, &p, &changed);
	ShowUnscaledValue2(dialog);
	
	switch (itemNum) {
		case M16OK:
		{
			// this is tricky , we have saved the NonPtrFields so we are free to mess with them since
			// they get restored if the user cancels.
			// We just have to be careful not to change sharedCCMover -> timeDep.
			// To accomplish this we use sharedCCMDialogTimeDep until the point of no return.
			///////////////////
			if (GetButton(dialog, M16NOSCALING)) sharedCCMover->scaleType = SCALE_NONE;
			if (GetButton(dialog, M16SCALETOCONSTANT)) sharedCCMover->scaleType = SCALE_CONSTANT;
			if (GetButton(dialog, M16SCALETOGRID)) sharedCCMover->scaleType = SCALE_OTHERGRID;
			sharedCCMover->scaleValue = EditText2Float(dialog, M16SCALEVALUE);
			mygetitext(dialog, M16SCALEGRIDNAME, sharedCCMover->scaleOtherFile, 31);
			
			err = EditTexts2LL(dialog, M16LATDEGREES, &sharedCCMover->refP,TRUE);
			if(err) break;
			
			if(!(sharedCCMDialogTimeDep && (sharedCCMDialogTimeDep->GetFileType() == HYDROLOGYFILE) && sharedCCMDialogTimeDep->bOSSMStyle))
			{	// old OSSM style files may have refP on land, but this point is not used in calculation
				
				//err = sharedCCMover->ComputeVelocityScale(model->GetModelTime());	// AH 07/10/2012
				
				if(err) 
				{	// restore values and report error to user
					printError("The unscaled value is too small at the chosen reference point.");
					break;
				}
			}
			if(sharedCCMDialogTimeDep && (sharedCCMDialogTimeDep->GetFileType() == SHIOHEIGHTSFILE || sharedCCMDialogTimeDep->GetFileType() == PROGRESSIVETIDEFILE || 
										 (sharedCCMDialogTimeDep->GetFileType() == HYDROLOGYFILE && sharedCCMDialogTimeDep->bOSSMStyle)) )
			{
				double newScaleFactor = EditText2Float(dialog, (sharedCCMDialogTimeDep->GetFileType() == 
																SHIOHEIGHTSFILE || sharedCCMDialogTimeDep->GetFileType() ==PROGRESSIVETIDEFILE) ? M16TIMEFILESCALEFACTOR : M16HYDROLOGYSCALEFACTOR);
				if (newScaleFactor == 0)
				{
					printError("The scale factor must be positive.");
					return 0;
				}
			}
			////////////////////
			// point of no return
			///////////////////
			sharedCCMover->bActive = GetButton(dialog, M16ACTIVE);
			
			// deal with the timeDep guy, JLM /11/25/98
			///////////////
			if(sharedCCMDialogTimeDep != sharedCCMover -> timeDep)
			{
				if(sharedCCMover -> timeDep)
				{	// dispose of the one we are replacing
					sharedCCMover -> timeDep->Dispose();
					delete sharedCCMover -> timeDep;
				}
				sharedCCMover -> timeDep = sharedCCMDialogTimeDep;
				sharedCCMChangedTimeFile = TRUE; 
			}
			if(sharedCCMover->timeDep && (sharedCCMover->timeDep->GetFileType() == SHIOHEIGHTSFILE || sharedCCMover->timeDep->GetFileType() == PROGRESSIVETIDEFILE))
			{
				double newScaleFactor = EditText2Float(dialog, M16TIMEFILESCALEFACTOR);
				sharedCCMover->timeDep->fScaleFactor = newScaleFactor;
			}
			if(sharedCCMover->timeDep && sharedCCMover->timeDep->GetFileType() == HYDROLOGYFILE)
			{
				if (sharedCCMover->timeDep->bOSSMStyle)
				{
					double newScaleFactor = EditText2Float(dialog, M16HYDROLOGYSCALEFACTOR);
					//sharedCCMover->timeDep->RescaleTimeValues(sharedCCMover->timeDep->fScaleFactor, newScaleFactor);
					sharedCCMover->timeDep->fScaleFactor = newScaleFactor; // code goes here, also refScale, scaleValue
					sharedCCMover->refScale = newScaleFactor; // code goes here, also refScale, scaleValue
				}
				else 
					sharedCCMover->refScale = sharedCCMover->timeDep->fScaleFactor;	// redundant ?
			}
			sharedCCMover -> bTimeFileActive = (sharedCCMover -> timeDep != 0); // active if we have one
			//err = sharedCCMover->ComputeVelocityScale();	// need to set refScale if hydrology
			DisposeDialogTimeDep2();
			////////////////////
			
			sharedCCMover->bShowArrows = GetButton(dialog, M16SHOWARROWS);
			sharedCCMover->fArrowScale = EditText2Float(dialog, M16ARROWSCALE);
			
			if (!sharedCCMover->CurrentUncertaintySame(sSharedCurrentCycleUncertainyInfo))
			{
				sharedCCMover -> SetCurrentUncertaintyInfo(sSharedCurrentCycleUncertainyInfo);
				sharedCCMover->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
			}
			return M16OK;
		}
			
		case M16CANCEL: 
			DisposeDialogTimeDep2();
			SetCurrentCycleDialogNonPtrFields(sharedCCMover,&sharedCurrentCycleDialogNonPtrFields);
			return M16CANCEL;
			
		case M16ACTIVE:
		case M16SHOWARROWS:
			ToggleButton(dialog, itemNum);
			break;
			
		case M16SETUNCERTAINTY:
		{
			Boolean userCanceledOrErr, uncertaintyValuesChanged=false;
			//CurrentUncertainyInfo info  = sharedCCMover -> GetCurrentUncertaintyInfo();
			CurrentUncertainyInfo info  = sSharedCurrentCycleUncertainyInfo;
			userCanceledOrErr = CurrentUncertaintyDialog(&info,GetDialogWindow(dialog),&uncertaintyValuesChanged);
			if(!userCanceledOrErr) 
			{
				if (uncertaintyValuesChanged)
				{
					sSharedCurrentCycleUncertainyInfo = info;
					//sharedCCMover->SetCurrentUncertaintyInfo(info);	// only want to update uncertainty if something has been changed and ok button hit
				}
			}
			break;
		}
			
			
		case M16NOSCALING:
		case M16SCALETOCONSTANT:
		case M16SCALETOGRID:
			SetButton(dialog, M16NOSCALING, itemNum == M16NOSCALING);
			SetButton(dialog, M16SCALETOCONSTANT, itemNum == M16SCALETOCONSTANT);
			SetButton(dialog, M16SCALETOGRID, itemNum == M16SCALETOGRID);
			if (itemNum == M16SCALETOGRID) {
			/*	char classNameOfSelectedGrid[32];
				ActivateParentDialog(FALSE);
				mygetitext(dialog, M16SCALEGRIDNAME, classNameOfSelectedGrid, 31);
				item = ChooseOtherGridDialog(sharedCCMover,classNameOfSelectedGrid);
				ActivateParentDialog(TRUE);
				if (item == M17OK)
					mysetitext(dialog, M16SCALEGRIDNAME, classNameOfSelectedGrid);
				else {
					SetButton(dialog, M16SCALETOGRID, FALSE);
					SetButton(dialog, M16NOSCALING, TRUE);
				}*/
			}
			break;
			
		case M16SCALEVALUE:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);
			CurrentCycleClick(dialog, M16SCALETOCONSTANT, 0, 0);
			break;
			
		case M16ARROWSCALE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
		case M16TIMEFILETYPES:
			
			short	theType;
			long	menuID_menuItem;
			PopClick(dialog, itemNum, &menuID_menuItem);
			theType = GetPopSelection (dialog, M16TIMEFILETYPES);
			if (theType==1)
			{
				// selected No Time Series
				if (sharedCCMDialogTimeDep == nil) break;	// already selected
				if (!CHOICEALERT(M79, 0, TRUE)) goto donetimefile;/*break;*/		// user canceled
				DisposeDialogTimeDep2();	
			}
			else 
			{
				short flag = kUndefined;
				if (theType==PROGRESSIVETIDEFILE) flag = kFudgeFlag;
				if (sharedCCMDialogTimeDep && !CHOICEALERT(M79, 0, TRUE)) goto donetimefile;/*break*/;	// user canceled
				// code goes here, need to know what type of file it is to decide if user wants standing wave or progressive wave...
				timeFile = LoadTOSSMTimeValue(sharedCCMover,flag); 
				// if user chose to cancel?
				if(!timeFile) goto donetimefile;/*break*/; // user canceled or an error 
				
				if (timeFile->GetFileType() != theType)	// file type doesn't match selected popup item
				{
					char fileTypeName[64], msg[128], fileTypeName2[64];
					
					switch(theType)
					{
						case OSSMTIMEFILE:
							strcpy(fileTypeName, "Tidal Current Time Series ");
							break;
						case SHIOCURRENTSFILE:
							strcpy(fileTypeName, "Shio Currents Coefficients ");
							break;
						case SHIOHEIGHTSFILE:
							strcpy(fileTypeName, "Shio Heights Coefficients ");
							break;
						case HYDROLOGYFILE:
							strcpy(fileTypeName, "Hydrology Time Series ");
							break; 
						case PROGRESSIVETIDEFILE:
							strcpy(fileTypeName, "Progressive Wave Coefficients ");
							break;
					}
					switch(timeFile->GetFileType())
					{
						case OSSMTIMEFILE:
							strcpy(fileTypeName2, "Tidal Current Time Series ");
							break;
						case SHIOCURRENTSFILE:
							strcpy(fileTypeName2, "Shio Currents Coefficients ");
							break;
						case SHIOHEIGHTSFILE:
							strcpy(fileTypeName2, "Shio Heights Coefficients ");
							break; 
						case HYDROLOGYFILE:
							strcpy(fileTypeName2, "Hydrology Time Series ");
							break;
						case PROGRESSIVETIDEFILE:
							strcpy(fileTypeName2, "Progressive Wave Coefficients ");
							break;
					}
					//if (timeFile->GetFileType()==SHIOHEIGHTSFILE && theType==PROGRESSIVETIDEFILE)
					if (timeFile->GetFileType()==PROGRESSIVETIDEFILE && theType==SHIOHEIGHTSFILE)
					{	// let it go, backwards for some reason...
						sprintf(msg,"The selected shio heights file will be treated as a progressive wave");
						//printNote(msg);
						//timeFile->SetFileType(PROGRESSIVETIDEFILE);
						timeFile->SetFileType(SHIOHEIGHTSFILE);
					}
					else
					{
						sprintf(msg,"The selected file is a %s file not a %s file. ",fileTypeName2,fileTypeName);
						printError(msg);
						goto donetimefile;
						break;
					}
				}
				if (timeFile->GetFileType() == HYDROLOGYFILE)	// bring up hydrology popup, unless file is in old OSSM format
				{
					Boolean userCanceledOrErr = false;
					{
						WorldRect gridBounds;
						char msg[256], latS[20], longS[20];
						WorldPointToStrings(timeFile->fStationPosition, latS, longS);
						if(sharedCCMover -> timeGrid->fGrid == 0)
						{ printError("Programmer error: sharedCCMover ->timeGrid-> fGrid is nil"); break;}
						gridBounds = sharedCCMover -> timeGrid->fGrid -> GetBounds();
						
						if(!WPointInWRect(timeFile->fStationPosition.pLong,timeFile->fStationPosition.pLat,&gridBounds))
						{
							sprintf(msg,"Check that this is the right file.%sThe reference point in this file is not within the grid bounds.%sLat: %s%sLng: %s",NEWLINESTRING,NEWLINESTRING,latS,NEWLINESTRING,longS);
							printWarning(msg);
							goto donetimefile;
						}
					}
					
					if (!timeFile->bOSSMStyle) userCanceledOrErr = HydrologyDialog2(timeFile,GetDialogWindow(dialog));
					//RegisterPopTable (csPopTable, sizeof (csPopTable) / sizeof (PopInfoRec));
					if(userCanceledOrErr)
					{	// leave in previous state
						goto donetimefile;
					}
					else
					{
						sharedCCMover->refP = timeFile->fStationPosition;
						if (!timeFile->bOSSMStyle) 
						{
							sharedCCMover->refScale = timeFile->fScaleFactor;
							//sharedCCMover->scaleValue = timeFile->fScaleFactor;
							Float2EditText(dialog, M16SCALEVALUE, timeFile->fScaleFactor, 4);
						}
						SwitchLLFormat(dialog, M16LATDEGREES, M16DEGREES);
						LL2EditTexts(dialog, M16LATDEGREES, &sharedCCMover->refP);
						(void)CurrentCycleClick(dialog,M16SCALETOCONSTANT,lParam,data);
					}
				}
				// JLM 7/13/99, CJ would like the interface to set the ref pt
				if(timeFile->GetClassID () == TYPE_SHIOTIMEVALUES)
				{	// it is a SHIO mover
					TShioTimeValue * shioTimeValue = (TShioTimeValue*)timeFile; // typecast
					WorldPoint wp = shioTimeValue -> GetStationLocation();
					VelocityRec vel;
					short btnHit;
					char msg[256], latS[20], longS[20];
					WorldPointToStrings(wp, latS, longS);
					WorldRect gridBounds;
					if(sharedCCMover -> timeGrid->fGrid == 0)
					{ printError("Programmer error: sharedCCMover -> timeGrid->fGrid is nil"); break;}
					gridBounds = sharedCCMover -> timeGrid->fGrid -> GetBounds();
					
					//if(WPointInWRect(wp.pLong,wp.pLat,&sharedCCMover -> bounds))
					if(WPointInWRect(wp.pLong,wp.pLat,&gridBounds))
					{
						btnHit = MULTICHOICEALERT(1670, 0, FALSE);
						switch(btnHit)
						{
							case 1:  // Yes, default button
								// user want to use the ref point info from the file
								// set the lat,long 
								LL2EditTexts(dialog, M16LATDEGREES, &wp);
								// set the scale to either 1 or -1 
								scaleValue = EditText2Float(dialog, M16SCALEVALUE);
								if(scaleValue < 0.0) Float2EditText(dialog, M16SCALEVALUE,-1.0, 4);//preserve the sign, i.e. preserve the direction the user set
								else Float2EditText(dialog, M16SCALEVALUE,1.0, 4);
								// turn on the scale button
								(void)CurrentCycleClick(dialog,M16SCALETOCONSTANT,lParam,data);
								break; 
								//case 3: return USERCANCEL; //NO button
							case 3: break; //NO button, still may want to set by hand, not a user cancel
						}
					}
					else
					{
						// hmmmm, this is most likely an error
						// should we alert the user ??
						sprintf(msg,"Check that this is the right file.%sThe reference point in this file is not within the map bounds.%sLat:%s%sLng:%s",NEWLINESTRING,NEWLINESTRING,latS,NEWLINESTRING,longS);
						printWarning(msg);
					}
					// if file contains height coefficients force derivative to be calculated and scale to be input
					if (timeFile->GetFileType() == SHIOHEIGHTSFILE || timeFile->GetFileType() == SHIOCURRENTSFILE || timeFile->GetFileType() == PROGRESSIVETIDEFILE)
					{
						if (err = timeFile->GetTimeValue(model->GetStartTime(),&vel))	// minus AH 07/10/2012
							goto donetimefile;	// user cancel or error
					}
				}
				
				sharedCCMDialogTimeDep = timeFile;
			}
			
		donetimefile:
			SetPopSelection (dialog, M16TIMEFILETYPES, sharedCCMDialogTimeDep ? sharedCCMDialogTimeDep->GetFileType() : NOTIMEFILE);
			PopDraw(dialog, M16TIMEFILETYPES);
		{	char itextstr[256] = "<none>";	// code warrior didn't like the ?: expression
			if (sharedCCMDialogTimeDep) strcpy (itextstr,sharedCCMDialogTimeDep->fileName);
			mysetitext(dialog, M16TIMEFILENAME, itextstr);
		}
			//mysetitext(dialog, M16TIMEFILENAME, sharedCCMDialogTimeDep ? sharedCCMDialogTimeDep->fileName : "<none>");
			ShowHideCurrentCycleDialogItems(dialog);
			ShowHideScaleFactorItems2(dialog);
			ShowCurrentCycleDialogUnitLabels(dialog);
			break;
			
		case M16TIMEFILESCALEFACTOR:
		case M16HYDROLOGYSCALEFACTOR:
			//case M16TRANSPORT:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
			
		case M16REPLACEMOVER:
			err = sharedCCMover -> ReplaceMover();
			if (err == USERCANCEL) break;	// stay at dialog
			return itemNum;	// what to do on error?
			break;
			
		case M16DEGREES:
		case M16DEGMIN:
		case M16DMS:
			err = EditTexts2LL(dialog, M16LATDEGREES, &p,TRUE);
			if(err) break;
			if (itemNum == M16DEGREES) settings.latLongFormat = DEGREES;
			if (itemNum == M16DEGMIN) settings.latLongFormat = DEGMIN;
			if (itemNum == M16DMS) settings.latLongFormat = DMS;
			SwitchLLFormat(dialog, M16LATDEGREES, M16DEGREES);
			LL2EditTexts(dialog, M16LATDEGREES, &p);
			break;
	}
	
	return 0;
}


OSErr CurrentCycleInit(DialogPtr dialog, VOIDPTR data)
{
	char itextstr[256] = "<none>";
	sharedCCMDialogTimeDep = sharedCCMover->timeDep;
	sharedCurrentCycleDialogNonPtrFields = GetCurrentCycleDialogNonPtrFields(sharedCCMover);
	sSharedCurrentCycleUncertainyInfo = sharedCCMover -> GetCurrentUncertaintyInfo();
	
	SetDialogItemHandle(dialog, M16HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M16FROST, (Handle)FrameEmbossed);
	
	RegisterPopTable(cs2PopTable, sizeof(cs2PopTable) / sizeof(PopInfoRec));
	RegisterPopUpDialog(M16, dialog);
	
	mysetitext(dialog, M16FILENAME, sharedCCMover->className);
	SetButton(dialog, M16ACTIVE, sharedCCMover->bActive);
	
	SetButton(dialog, M16SHOWARROWS, sharedCCMover->bShowArrows);
	Float2EditText(dialog, M16ARROWSCALE, sharedCCMover->fArrowScale, 4);
	
	SetPopSelection(dialog, M16TIMEFILETYPES, sharedCCMover->timeDep ? sharedCCMover->timeDep->GetFileType() : NOTIMEFILE);
	if (sharedCCMover->timeDep) strcpy(itextstr,sharedCCMover->timeDep->fileName);
	//mysetitext(dialog, M16TIMEFILENAME, (sharedCCMover->timeDep ? sharedCCMover->timeDep->fileName : "<none>"));
	mysetitext(dialog, M16TIMEFILENAME, itextstr);	// code warrior doesn't like the ?: expression
	
	//if (sharedCCMover->IAm(TYPE_TIDECURCYCLEMOVER)) setwtitle(dialog,"Tide Pattern Mover Settings");
	
	SetButton(dialog, M16NOSCALING, sharedCCMover->scaleType == SCALE_NONE);
	SetButton(dialog, M16SCALETOCONSTANT, sharedCCMover->scaleType == SCALE_CONSTANT);
	SetButton(dialog, M16SCALETOGRID, sharedCCMover->scaleType == SCALE_OTHERGRID);
	Float2EditText(dialog, M16SCALEVALUE, sharedCCMover->scaleValue, 4);
	mysetitext(dialog, M16SCALEGRIDNAME, sharedCCMover->scaleOtherFile);
	SwitchLLFormat(dialog, M16LATDEGREES, M16DEGREES);
	LL2EditTexts(dialog, M16LATDEGREES, &sharedCCMover->refP);
	
	ShowHideCurrentCycleDialogItems(dialog);
  	ShowHideScaleFactorItems2(dialog);
	
	ShowUnscaledValue2(dialog);
	ShowCurrentCycleDialogUnitLabels(dialog);
	
	return 0;
}



OSErr CurrentCycleSettingsDialog(CurrentCycleMover *newMover, TMap *owner, Boolean *timeFileChanged)
{
	short item;
	
	if (!newMover)return -1;
	
	sharedCCMover = newMover;
	
	sharedCCMChangedTimeFile = FALSE;
	item = MyModalDialog(M16, mapWindow, newMover, CurrentCycleInit, CurrentCycleClick);
	*timeFileChanged = sharedCCMChangedTimeFile;
	
	if(M16OK == item)	model->NewDirtNotification();// tell model about dirt
	return M16OK == item ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////

// for now just use the CATS current dialog
/*static CurrentCycleMover *sTideCurCycleDialogMover;
 
 short CurrentCycleMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
 {
 
 switch (itemNum) {
 case M33OK:
 {
 //mygetitext(dialog, M33NAME, sTideCurCycleDialogMover->fPathName, kPtCurUserNameLen-1);
 sTideCurCycleDialogMover->bActive = GetButton(dialog, M33ACTIVE);
 sTideCurCycleDialogMover->bShowArrows = GetButton(dialog, M33SHOWARROWS);
 sTideCurCycleDialogMover->arrowScale = EditText2Float(dialog, M33ARROWSCALE);
 //sTideCurCycleDialogMover->arrowDepth = arrowDepth;
 //sTideCurCycleDialogMover->fVar.curScale = EditText2Float(dialog, M33SCALE);
 sTideCurCycleDialogMover->fUpCurUncertainty = EditText2Float(dialog, M33ALONG)/100;
 sTideCurCycleDialogMover->fDownCurUncertainty = - EditText2Float(dialog, M33ALONG)/100;
 sTideCurCycleDialogMover->fRightCurUncertainty = EditText2Float(dialog, M33CROSS)/100;
 sTideCurCycleDialogMover->fLeftCurUncertainty = - EditText2Float(dialog, M33CROSS)/100;
 sTideCurCycleDialogMover->fEddyV0 = EditText2Float(dialog, M33MINCURRENT);
 sTideCurCycleDialogMover->fUncertainStartTime = (long)(round(EditText2Float(dialog, M33STARTTIME)*3600));
 sTideCurCycleDialogMover->fDuration = EditText2Float(dialog, M33DURATION)*3600;
 
 return M33OK;
 }
 
 case M33CANCEL: 
 return M33CANCEL;
 
 case M33ACTIVE:
 case M33SHOWARROWS:
 ToggleButton(dialog, itemNum);
 break;
 
 case M33ARROWSCALE:
 //case M33ARROWDEPTH:
 case M33SCALE:
 case M33ALONG:
 case M33CROSS:
 case M33MINCURRENT:
 case M33STARTTIME:
 case M33DURATION:
 CheckNumberTextItem(dialog, itemNum, TRUE);
 break;
 
 }
 
 return 0;
 }
 
 
 OSErr CurrentCycleMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
 {
 char pathName[256],fileName[64];
 SetDialogItemHandle(dialog, M33HILITEDEFAULT, (Handle)FrameDefault);
 SetDialogItemHandle(dialog, M33UNCERTAINTYBOX, (Handle)FrameEmbossed);
 
 strcpy(pathName,sTideCurCycleDialogMover->fPathName); // SplitPathFile changes the original path name
 SplitPathFile(pathName, fileName);
 mysetitext(dialog, M33NAME, fileName); // use short file name for now
 SetButton(dialog, M33ACTIVE, sTideCurCycleDialogMover->bActive);
 
 SetButton(dialog, M33SHOWARROWS, sTideCurCycleDialogMover->bShowArrows);
 Float2EditText(dialog, M33ARROWSCALE, sTideCurCycleDialogMover->arrowScale, 6);
 //Float2EditText(dialog, M33ARROWDEPTH, sTideCurCycleDialogMover->arrowDepth, 6);
 
 //Float2EditText(dialog, M33SCALE, sTideCurCycleDialogMover->fVar.curScale, 6);
 ShowHideDialogItem(dialog, M33SCALE, false); 
 ShowHideDialogItem(dialog, M33SCALELABEL, false); 
 Float2EditText(dialog, M33ALONG, sTideCurCycleDialogMover->fUpCurUncertainty*100, 6);
 Float2EditText(dialog, M33CROSS, sTideCurCycleDialogMover->fRightCurUncertainty*100, 6);
 Float2EditText(dialog, M33MINCURRENT, sTideCurCycleDialogMover->fEddyV0, 6);	// uncertainty min in mps ?
 Float2EditText(dialog, M33STARTTIME, sTideCurCycleDialogMover->fUncertainStartTime/3600., 2);
 Float2EditText(dialog, M33DURATION, sTideCurCycleDialogMover->fDuration/3600., 2);
 
 ShowHideDialogItem(dialog, M33TIMEZONEPOPUP, false); 
 ShowHideDialogItem(dialog, M33TIMESHIFTLABEL, false); 
 ShowHideDialogItem(dialog, M33TIMESHIFT, false); 
 ShowHideDialogItem(dialog, M33GMTOFFSETS, false); 
 
 MySelectDialogItemText(dialog, M33ALONG, 0, 100);
 
 return 0;
 }
 
 
 
 OSErr CurrentCycleMover::SettingsDialog()
 {
 short item;
 //Point where = CenteredDialogUpLeft(M33);
 
 //OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
 //MySFReply reply;
 
 sTideCurCycleDialogMover = this; // should pass in what is needed only
 item = MyModalDialog(M33, mapWindow, 0, CurrentCycleMoverSettingsInit, CurrentCycleMoverSettingsClick);
 sTideCurCycleDialogMover = 0;
 
 if(M33OK == item)	model->NewDirtNotification();// tell model about dirt
 return M33OK == item ? 0 : -1;
 }*/

/////////////////////////////////////////////////

/*OSErr CurrentCycleMover::InitMover()
 {	
 OSErr	err = noErr;
 
 err = TCATSMover::InitMover ();
 return err;
 }*/


/**************************************************************************************************/
CurrentUncertainyInfo CurrentCycleMover::GetCurrentUncertaintyInfo ()
{
	CurrentUncertainyInfo	info;
	
	memset(&info,0,sizeof(info));
	info.setEddyValues = TRUE;
	info.fUncertainStartTime	= this -> fUncertainParams.startTimeInHrs;
	info.fDuration					= this -> fUncertainParams.durationInHrs;
	info.fEddyDiffusion			= this -> fEddyDiffusion;		
	info.fEddyV0					= this -> fEddyV0;			
	info.fDownCurUncertainty	= this -> fUncertainParams.alongCurUncertainty * -1;
	info.fUpCurUncertainty		= this -> fUncertainParams.alongCurUncertainty;	
	info.fRightCurUncertainty	= this -> fUncertainParams.crossCurUncertainty;
	info.fLeftCurUncertainty	= this -> fUncertainParams.crossCurUncertainty * -1;	
	
	return info;
}
/**************************************************************************************************/
void CurrentCycleMover::SetCurrentUncertaintyInfo (CurrentUncertainyInfo info)
{
	this -> fUncertainParams.startTimeInHrs	= info.fUncertainStartTime;
	this -> fUncertainParams.durationInHrs 	= info.fDuration;
	this -> fEddyDiffusion 			= info.fEddyDiffusion;		
	this -> fEddyV0 					= info.fEddyV0;			
	//this -> fDownCurUncertainty 	= info.fDownCurUncertainty;	
	this -> fUncertainParams.alongCurUncertainty 		= info.fUpCurUncertainty;	
	this -> fUncertainParams.crossCurUncertainty 	= info.fRightCurUncertainty;	
	//this -> fLeftCurUncertainty 	= info.fLeftCurUncertainty;	
	
	return;
}
Boolean CurrentCycleMover::CurrentUncertaintySame (CurrentUncertainyInfo info)
{
	if (this -> fUncertainStartTime	== info.fUncertainStartTime 
		&&	this -> fUncertainParams.durationInHrs == info.fDuration
		&&	this -> fEddyDiffusion 			== info.fEddyDiffusion		
		&&	this -> fEddyV0 				== info.fEddyV0			
		&&	this -> fUncertainParams.alongCurUncertainty 	== -1*info.fDownCurUncertainty	
		&&	this -> fUncertainParams.alongCurUncertainty 	== info.fUpCurUncertainty	
		&&	this -> fUncertainParams.crossCurUncertainty	== info.fRightCurUncertainty	
		&&	this -> fUncertainParams.crossCurUncertainty 	== -1 * info.fLeftCurUncertainty	)
		return true;
	else return false;
}
