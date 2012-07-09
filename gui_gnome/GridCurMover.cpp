#include "GridCurMover.h"
#include "MemUtils.h"
#include "CROSS.H"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT GRIDCURMOVER
#endif
#endif

extern TModel *model;

GridCurMover::GridCurMover (TMap *owner, char *name) : TCATSMover(owner, name)
{
	fTimeDataHdl = 0;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;

	fUserUnits = kUndefined;
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fInputFilesHdl = 0;	// for multiple files case
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -.5; 
	fUpCurUncertainty = .5; 	
	fRightCurUncertainty = .25;  
	fLeftCurUncertainty = -.25; 
	fDuration=24*3600.; //24 hrs as seconds 
	fUncertainStartTime = 0.; // seconds
	fEddyV0 = 0.0;	// fVar.uncertMinimumInMPS
	//SetClassName (name); // short file name
}



void GridCurMover::Dispose ()
{
	/*if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}*/

	if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}

	TCATSMover::Dispose ();
}


Boolean IsGridCurTimeFile (char *path, short *selectedUnitsP)
{
	Boolean bIsValid = false;
	OSErr	err = noErr;
	long line;
	char strLine [256];
	char firstPartOfFile [256];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined, numScanned;
	char unitsStr[64], gridcurStr[64];
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with [GRIDCURTIME]
		char * strToMatch = "[GRIDCURTIME]";
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (!strncmp (strLine,strToMatch,strlen(strToMatch)))
		{
			bIsValid = true;
			*selectedUnitsP = selectedUnits;
			numScanned = sscanf(strLine,"%s%s",gridcurStr,unitsStr);
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

/////////////////////////////////////////////////
/////////////////////////////////////////////////

static GridCurMover *sGridCurDialogMover;
static Boolean sDialogUncertaintyChanged;

short GridCurMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	
	switch (itemNum) {
		case M33OK:
		{
			double tempAlong, tempCross/*, tempUncertMin*/;
			long tempStartTime;
			Seconds tempDuration;
			//mygetitext(dialog, M33NAME, sGridCurDialogMover->fPathName, kPtCurUserNameLen-1);
			sGridCurDialogMover->bActive = GetButton(dialog, M33ACTIVE);
			sGridCurDialogMover->bShowArrows = GetButton(dialog, M33SHOWARROWS);
			sGridCurDialogMover->arrowScale = EditText2Float(dialog, M33ARROWSCALE);
			//sGridCurDialogMover->arrowDepth = arrowDepth;
			//sGridCurDialogMover->fVar.curScale = EditText2Float(dialog, M33SCALE);
			
			
			tempAlong = EditText2Float(dialog, M33ALONG)/100;
			tempCross = EditText2Float(dialog, M33CROSS)/100;
			//tempUncertMin = EditText2Float(dialog, M33MINCURRENT);
			tempStartTime = (long)(round(EditText2Float(dialog, M33STARTTIME)*3600));
			tempDuration = EditText2Float(dialog, M33DURATION)*3600;
			if (sGridCurDialogMover->fUpCurUncertainty != tempAlong || sGridCurDialogMover->fRightCurUncertainty != tempCross
				|| sGridCurDialogMover->fUncertainStartTime != tempStartTime || sGridCurDialogMover->fDuration != tempDuration
				/*|| sGridCurDialogMover->fEddyV0 != tempUncertMin*/) sDialogUncertaintyChanged = true;
			sGridCurDialogMover->fUpCurUncertainty = EditText2Float(dialog, M33ALONG)/100;
			sGridCurDialogMover->fDownCurUncertainty = - EditText2Float(dialog, M33ALONG)/100;
			sGridCurDialogMover->fRightCurUncertainty = EditText2Float(dialog, M33CROSS)/100;
			sGridCurDialogMover->fLeftCurUncertainty = - EditText2Float(dialog, M33CROSS)/100;
			//sGridCurDialogMover->fEddyV0 = EditText2Float(dialog, M33MINCURRENT);
			sGridCurDialogMover->fUncertainStartTime = (long)(round(EditText2Float(dialog, M33STARTTIME)*3600));
			sGridCurDialogMover->fDuration = EditText2Float(dialog, M33DURATION)*3600;
			
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
			//case M33MINCURRENT:
		case M33STARTTIME:
		case M33DURATION:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
	}
	
	return 0;
}


OSErr GridCurMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
{
	char pathName[256],fileName[64];
	SetDialogItemHandle(dialog, M33HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M33UNCERTAINTYBOX, (Handle)FrameEmbossed);
	
	strcpy(pathName,sGridCurDialogMover->fPathName); // SplitPathFile changes the original path name
	SplitPathFile(pathName, fileName);
	mysetitext(dialog, M33NAME, fileName); // use short file name for now
	SetButton(dialog, M33ACTIVE, sGridCurDialogMover->bActive);
	
	SetButton(dialog, M33SHOWARROWS, sGridCurDialogMover->bShowArrows);
	Float2EditText(dialog, M33ARROWSCALE, sGridCurDialogMover->arrowScale, 6);
	//Float2EditText(dialog, M33ARROWDEPTH, sGridCurDialogMover->arrowDepth, 6);
	
	ShowHideDialogItem(dialog, M33ARROWDEPTHAT, false); 	// this mover is always 2D
	ShowHideDialogItem(dialog, M33ARROWDEPTH, false); 
	ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, false); 
	
	//Float2EditText(dialog, M33SCALE, sGridCurDialogMover->fVar.curScale, 6);
	ShowHideDialogItem(dialog, M33SCALE, false); 
	ShowHideDialogItem(dialog, M33SCALELABEL, false); 
	Float2EditText(dialog, M33ALONG, sGridCurDialogMover->fUpCurUncertainty*100, 6);
	Float2EditText(dialog, M33CROSS, sGridCurDialogMover->fRightCurUncertainty*100, 6);
	//Float2EditText(dialog, M33MINCURRENT, sGridCurDialogMover->fEddyV0, 6);	// uncertainty min in mps ?
	Float2EditText(dialog, M33STARTTIME, sGridCurDialogMover->fUncertainStartTime/3600., 2);
	Float2EditText(dialog, M33DURATION, sGridCurDialogMover->fDuration/3600., 2);
	
	ShowHideDialogItem(dialog, M33TIMEZONEPOPUP, false); 
	ShowHideDialogItem(dialog, M33TIMESHIFTLABEL, false); 
	ShowHideDialogItem(dialog, M33TIMESHIFT, false); 
	ShowHideDialogItem(dialog, M33GMTOFFSETS, false); 
	ShowHideDialogItem(dialog, M33TIMEZONELABEL, false); 
	
	ShowHideDialogItem(dialog, M33MINCURRENTLABEL, false); 
	ShowHideDialogItem(dialog, M33MINCURRENT, false); 
	ShowHideDialogItem(dialog, M33MINCURRENTUNITS, false); 
	
	ShowHideDialogItem(dialog, M33VELOCITYATBOTTOMCHECKBOX, false); // this mover is always 2D
	ShowHideDialogItem(dialog, M33REPLACEMOVER, false); // this mover is always 2D
	ShowHideDialogItem(dialog, M33EXTRAPOLATETOVALUE, false);
	ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, false);		
	ShowHideDialogItem(dialog, M33EXTRAPOLATETOLABEL, false); 
	ShowHideDialogItem(dialog, M33EXTRAPOLATETOVALUE, false); 
	ShowHideDialogItem(dialog, M33EXTRAPOLATETOUNITSLABEL, false); 
	
	ShowHideDialogItem(dialog, M33EXTRAPOLATECHECKBOX, false); // probably want to allow this
	
	MySelectDialogItemText(dialog, M33ALONG, 0, 100);
	
	return 0;
}



OSErr GridCurMover::SettingsDialog()
{
	short item;
	
	sGridCurDialogMover = dynamic_cast<GridCurMover *>(this); // should pass in what is needed only
	sDialogUncertaintyChanged = false;
	item = MyModalDialog(M33, mapWindow, 0, GridCurMoverSettingsInit, GridCurMoverSettingsClick);
	sGridCurDialogMover = 0;
	
	if(M33OK == item)	
	{
		if (sDialogUncertaintyChanged) dynamic_cast<GridCurMover *>(this)->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
		model->NewDirtNotification();// tell model about dirt
	}
	return M33OK == item ? 0 : -1;
}


/*OSErr GridCurMover::InitMover()
 {	
 OSErr	err = noErr;
 err = TCATSMover::InitMover ();
 return err;
 }*/



OSErr GridCurMover::CheckAndScanFile(char *errmsg)
{
	Seconds time = model->GetModelTime(), startTime, endTime, lastEndTime, testTime;
	long i,numFiles = GetNumFiles();
	OSErr err = 0;
	
	errmsg[0]=0;
	if (fEndData.timeIndex!=UNASSIGNEDINDEX)
		testTime = (*fTimeDataHdl)[fEndData.timeIndex].time;	// currently loaded end time
	
	for (i=0;i<numFiles;i++)
	{
		startTime = (*fInputFilesHdl)[i].startTime;
		endTime = (*fInputFilesHdl)[i].endTime;
		if (startTime<=time&&time<=endTime && !(startTime==endTime))
		{
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeDataHdl,false);
			// code goes here, check that start/end times match
			strcpy(fPathName,(*fInputFilesHdl)[i].pathName);
			fOverLap = false;
			return err;
		}
		if (startTime==endTime && startTime==time)	// one time in file, need to overlap
		{
			long fileNum;
			if (i<numFiles-1)
				fileNum = i+1;
			else
				fileNum = i;
			fOverLapStartTime = (*fInputFilesHdl)[fileNum-1].endTime;	// last entry in previous file
			DisposeLoadedData(&fStartData);
			/*if (fOverLapStartTime==testTime)	// shift end time data to start time data
			 {
			 fStartData = fEndData;
			 ClearLoadedData(&fEndData);
			 }
			 else*/
			{
				if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeDataHdl,false);
				DisposeLoadedData(&fEndData);
				strcpy(fPathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeDataHdl,false);
			strcpy(fPathName,(*fInputFilesHdl)[fileNum].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		if (i>0 && (lastEndTime<time && time<startTime))
		{
			fOverLapStartTime = (*fInputFilesHdl)[i-1].endTime;	// last entry in previous file
			DisposeLoadedData(&fStartData);
			if (fOverLapStartTime==testTime)	// shift end time data to start time data
			{
				fStartData = fEndData;
				ClearLoadedData(&fEndData);
			}
			else
			{
				if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[i-1].pathName,&fTimeDataHdl,false);
				DisposeLoadedData(&fEndData);
				strcpy(fPathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeDataHdl,false);
			if (err) return err;
			strcpy(fPathName,(*fInputFilesHdl)[i].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		lastEndTime = endTime;
	}
	strcpy(errmsg,"Time outside of interval being modeled");
	return -1;	
	//return err;
}

Boolean GridCurMover::CheckInterval(long &timeDataInterval)
{
	Seconds time =  model->GetModelTime();
	long i,numTimes;
	
	
	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists
	
	// check for constant current
	if (numTimes==1 && !(GetNumFiles()>1)) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
			return true;
		else
			return false;
	}
	
	if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
	{
		if (time>=(*fTimeDataHdl)[fStartData.timeIndex].time && time<=(*fTimeDataHdl)[fEndData.timeIndex].time)
		{	// we already have the right interval loaded
			timeDataInterval = fEndData.timeIndex;
			return true;
		}
	}
	
	if (GetNumFiles()>1 && fOverLap)
	{	
		if (time>=fOverLapStartTime && time<=(*fTimeDataHdl)[fEndData.timeIndex].time)
			return true;	// we already have the right interval loaded, time is in between two files
		else fOverLap = false;
	}
	
	for (i=0;i<numTimes;i++) 
	{	// find the time interval
		if (time>=(*fTimeDataHdl)[i].time && time<=(*fTimeDataHdl)[i+1].time)
		{
			timeDataInterval = i+1; // first interval is between 0 and 1, and so on
			return false;
		}
	}	
	// don't allow time before first or after last
	if (time<(*fTimeDataHdl)[0].time) 
		timeDataInterval = 0;
	if (time>(*fTimeDataHdl)[numTimes-1].time) 
		timeDataInterval = numTimes;
	return false;
	
}

void GridCurMover::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void GridCurMover::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long GridCurMover::GetNumTimesInFile()
{
	long numTimes = 0;
	
	if (fTimeDataHdl) numTimes = _GetHandleSize((Handle)fTimeDataHdl)/sizeof(**fTimeDataHdl);
	return numTimes;     
}

long GridCurMover::GetNumFiles()
{
	long numFiles = 0;
	
	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

OSErr GridCurMover::SetInterval(char *errmsg)
{
	long timeDataInterval=0;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
	
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
	
	// check for constant current 
	if(numTimesInFile==1 && !(GetNumFiles()>1))	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile)
	{	// before the first step in the file
		
		if (GetNumFiles()>1)
		{
			if ((err = CheckAndScanFile(errmsg)) || fOverLap) goto done;	// overlap is special case
			intervalLoaded = this -> CheckInterval(timeDataInterval);
			indexOfStart = timeDataInterval-1;
			indexOfEnd = timeDataInterval;
			numTimesInFile = this -> GetNumTimesInFile();
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
		if(!errmsg[0])strcpy(errmsg,"Error in GridCurMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;
	
}


//#define GridCurMoverREADWRITEVERSION 1 //JLM
#define GridCurMoverREADWRITEVERSION 2 //CMO added multiple files option

OSErr GridCurMover::Write (BFPB *bfpb)
{
	char c;
	long i, version = GridCurMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	VelocityRec velocity;
	long amtTimeData = GetNumTimesInFile();
	long numPoints, numFiles;
	float val;
	PtCurTimeData timeData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TCATSMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("GridCurMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	/*id = fGrid -> GetClassID (); //JLM
	 if (err = WriteMacValue(bfpb, id)) return err; //JLM
	 if (err = fGrid -> Write (bfpb)) goto done;
	 */
	if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	if (err = WriteMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	
	if (err = WriteMacValue(bfpb, amtTimeData)) goto done;
	for (i=0;i<amtTimeData;i++)
	{
		timeData = INDEXH(fTimeDataHdl,i);
		if (err = WriteMacValue(bfpb, timeData.fileOffsetToStartOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.lengthOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.time)) goto done;
	}
	
	numFiles = GetNumFiles();
	if (err = WriteMacValue(bfpb, numFiles)) goto done;
	if (numFiles > 0)
	{
		for (i = 0 ; i < numFiles ; i++) {
			fileInfo = INDEXH(fInputFilesHdl,i);
			if (err = WriteMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
			if (err = WriteMacValue(bfpb, fileInfo.startTime)) goto done;
			if (err = WriteMacValue(bfpb, fileInfo.endTime)) goto done;
		}
		if (err = WriteMacValue(bfpb, fOverLap)) return err;
		if (err = WriteMacValue(bfpb, fOverLapStartTime)) return err;
	}
	
done:
	if(err)
		TechError("GridCurMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr GridCurMover::Read(BFPB *bfpb)
{
	char c, pathName[256], fileName[64], msg[256];
	long i, version, amtTimeData, numPoint, numFiles;
	ClassID id;
	float val;
	PtCurTimeData timeData;
	PtCurFileInfo fileInfo;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TCATSMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("GridCurMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("GridCurMover::Read()", "id != TYPE_GRIDCURMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > GridCurMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the GridCur mover (should always be rectgrid...)
	/*if (err = ReadMacValue(bfpb,&id)) return err;
	 switch(id)
	 {
	 case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
	 //case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
	 //case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
	 default: printError("Unrecognized Grid type in GridCurMover::Read()."); return -1;
	 }
	 
	 if (err = fGrid -> Read (bfpb)) goto done;
	 */
	if (err = ReadMacValue(bfpb, &fNumRows)) goto done;	
	if (err = ReadMacValue(bfpb, &fNumCols)) goto done;	
	if (err = ReadMacValue(bfpb, fPathName, kMaxNameLen)) goto done;	
	ResolvePath(fPathName); // JLM 6/3/10
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
		{ 
			bPathIsValid = false;
		}
		else
		{
			strcpy(fPathName,fileName);
			strcpy(pathName,fPathName);
			SplitPathFile(pathName,fFileName);
		}
		
	}
	else
	{
		strcpy(pathName,fPathName);
		SplitPathFile(pathName,fFileName);
	}
	
	if (!bPathIsValid)
	{	// try other platform
		char delimStr[32] ={DIRDELIMITER,0}, delimStrOpp[32] = {OPPOSITEDIRDELIMITER,0}, origPathName[kMaxNameLen], *p;	
		strcpy(origPathName,fPathName);
		p = strrchr(origPathName,OPPOSITEDIRDELIMITER);
		if (p) 
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{
		}
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
		sprintf(msg,"This save file references a gridCur file which cannot be found.  Please find the file \"%s\".",fPathName);printNote(msg);
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
	
	if (err = ReadMacValue(bfpb, &amtTimeData)) goto done;	
	fTimeDataHdl = (PtCurTimeDataHdl)_NewHandleClear(sizeof(PtCurTimeData)*amtTimeData);
	if(!fTimeDataHdl)
	{TechError("GridCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < amtTimeData ; i++) {
		if (err = ReadMacValue(bfpb, &timeData.fileOffsetToStartOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.lengthOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.time)) goto done;
		INDEXH(fTimeDataHdl, i) = timeData;
	}
	
	if (version>1)
	{
		if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
		if (numFiles > 0)
		{
			fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
			if(!fInputFilesHdl)
			{TechError("GridCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
			for (i = 0 ; i < numFiles ; i++) {
				if (err = ReadMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
				ResolvePath(fileInfo.pathName); // JLM 6/3/10
				if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
				if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
				INDEXH(fInputFilesHdl,i) = fileInfo;
			}
			if (err = ReadMacValue(bfpb, &fOverLap)) return err;
			if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
		}
	}
	
	fUserUnits = kKnots;	// code goes here, implement using units
	
done:
	if(err)
	{
		TechError("GridCurMover::Read(char* path)", " ", 0); 
		if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
		if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr GridCurMover::CheckAndPassOnMessage(TModelMessage *message)
{
	return TCATSMover::CheckAndPassOnMessage(message); 
}

/////////////////////////////////////////////////
long GridCurMover::GetListLength()
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
}

ListItem GridCurMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64];
	ListItem item = { dynamic_cast<GridCurMover *>(this), 0, indent, 0 };
	
	
	if (n == 0) {
		item.index = I_GRIDCURNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Currents: \"%s\"", fFileName);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	
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
		
		/*if (--n == 0) {
		 item.index = I_GRIDCURSCALE;
		 StringWithoutTrailingZeros(valStr,fVar.curScale,6);
		 sprintf(text, "Multiplicative Scalar: %s", valStr);
		 //item.indent++;
		 return item;
		 }*/
		
		
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
				
				/*if (--n == 0) {
				 item.index = I_GRIDCURMINCURRENT;
				 item.indent++;
				 StringWithoutTrailingZeros(valStr,fEddyV0,6);	// reusing the TCATSMover variable for now
				 sprintf(text, "Current Minimum: %s m/s",valStr);
				 return item;
				 }*/
				
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
}

Boolean GridCurMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
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
		Boolean userCanceledOrErr ;
		(void) this -> SettingsDialog();
		return TRUE;
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean GridCurMover::FunctionEnabled(ListItem item, short buttonID)
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
	
	return TCATSMover::FunctionEnabled(item, buttonID);
}

OSErr GridCurMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

/*OSErr GridCurMover::AddItem(ListItem item)
 {
 if (item.index == I_GRIDCURNAME)
 return TMover::AddItem(item);
 
 return 0;
 }*/

OSErr GridCurMover::DeleteItem(ListItem item)
{
	if (item.index == I_GRIDCURNAME)
		return moverMap -> DropMover(dynamic_cast<GridCurMover *>(this));
	
	return 0;
}

Boolean GridCurMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant current, we can say "no"
	if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	return depends;
}

void GridCurMover::Draw(Rect r, WorldRect view) 
{
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newCATSgridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of GridCurMover
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	OSErr err = 0;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	
	
	if (!bShowArrows && !bShowGrid) return;
	
	p = GetQuickDrawPt(refP.pLong, refP.pLat, &r, &offQuickDrawPlane);
	bounds = rectGrid->GetBounds();
	
	// draw the reference point
	RGBForeColor(&colors[BLUE]);
	MySetRect(&c, p.h - 2, p.v - 2, p.h + 2, p.v + 2);
	PaintRect(&c);
	RGBForeColor(&colors[BLACK]);
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / fNumCols) / 2;
	dLat = (WRectHeight(bounds) / fNumRows) / 2;
	RGBForeColor(&colors[PURPLE]);
	
	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);
	
	if (bShowArrows)
	{
		err = this -> SetInterval(errmsg);
		if(err && !bShowGrid) return;	// want to show grid even if there's no current data
		
		loaded = this -> CheckInterval(timeDataInterval);
		if(!loaded && !bShowGrid) return;
		
		// Check for time varying current 
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime;
			else
				startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
			//startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
			endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
		}
	}	
	
	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {
			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newCATSgridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = /*OK*/dynamic_cast<GridCurMover *>(this)->GetVelocityIndex(wp);
				if (bShowArrows && index >= 0)
				{
					// Check for constant current 
					if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
					{
						velocity.u = INDEXH(fStartData.dataHdl,index).u;
						velocity.v = INDEXH(fStartData.dataHdl,index).v;
					}
					else // time varying current
					{
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
					}
				}
			}
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bShowGrid) PaintRect(&c);
			
			if (bShowArrows && (velocity.u != 0 || velocity.v != 0))
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
			}
		}
	
	RGBForeColor(&colors[BLACK]);
}

OSErr GridCurMover::ReadHeaderLines(char *path, WorldRect *bounds)
{
	char s[256], classicPath[256];
	long line = 0;
	CHARH f = 0;
	OSErr err = 0;
	long /*numLines,*/numScanned;
	double dLon,dLat,oLon,oLat;
	double lowLon,lowLat,highLon,highLat;
	Boolean velAtCenter = 0;
	Boolean velAtCorners = 0;
	//long numLinesInText, headerLines = 8;
	
	if (!path) return -1;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) goto done;
	//numLinesInText = NumLinesInText(*f);
	////
	// read the header
	///////////////////////
	/////////////////////////////////////////////////
	NthLineInTextOptimized(*f, line++, s, 256); // gridcur header
	if(fUserUnits == kUndefined)
	{	
		// we have to ask the user for units...
		Boolean userCancel=false;
		short selectedUnits = kKnots; // knots will be default
		err = AskUserForUnits(&selectedUnits,&userCancel);
		if(err || userCancel) { err = -1; goto done;}
		fUserUnits = selectedUnits;
	}
	
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
	//NthLineInTextOptimized(*f, line++, s, 256); // row col u v header
	//
	
	// check hemisphere stuff here , code goes here
	if(velAtCenter)
	{
		(*bounds).loLat = lowLat*1000000;
		(*bounds).hiLat = highLat*1000000;
		(*bounds).loLong = lowLon*1000000;
		(*bounds).hiLong = highLon*1000000;
	}
	else if(velAtCorners)
	{
		(*bounds).loLat = round((oLat - dLat/2.0)*1000000);
		(*bounds).hiLat = round((oLat + (fNumRows-1)*dLat + dLat/2.0)*1000000);
		(*bounds).loLong = round((oLon - dLon/2.0)*1000000);
		(*bounds).hiLong = round((oLon + (fNumCols-1)*dLon + dLon/2.0)*1000000);
	}
	//numLines = numLinesInText - headerLines;	// allows user to leave out land points within grid (or standing water)
	
	NthLineInTextOptimized(*f, (line)++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	while ((s[0]=='[' && s[1]=='U') || s[0]==0)
	{	// [USERDATA] lines, and blank lines, generalize to anything but [FILE] ?
		NthLineInTextOptimized(*f, (line)++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);
	}
	if(!strstr(s,"[FILE]")) 
	{	// single file
		err = ScanFileForTimes(path,&fTimeDataHdl,true);
		if (err) goto done;
	}
	else
	{	// multiple files
		long numLinesInText = NumLinesInText(*f);
		long numFiles = (numLinesInText - (line - 1))/3;	// 3 lines for each file - filename, starttime, endtime
		//strcpy(fPathName,s+strlen("[FILE]\t"));
		strcpy(fPathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace(fPathName);
		ResolvePathFromInputFile(path,fPathName); // JLM 6/8/10
		if(fPathName[0] && FileExists(0,0,fPathName))
		{
			err = ScanFileForTimes(fPathName,&fTimeDataHdl,true);
			if (err) goto done;
			// code goes here, maybe do something different if constant current
			line--;
			err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to GridCur data File does not exist.%s%s",NEWLINESTRING,fPathName);
			printError(msg);
			err = true;
		}
		
		/*err = ScanFileForTimes(fPathName,&fTimeDataHdl,true);
		 if (err) goto done;
		 line--;
		 err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);*/
	}
	
done:
	if(f) { DisposeHandle((Handle)f); f = 0;}
	if(err)
	{
		if(err==memFullErr)
			TechError("TRectGridVel::ReadGridCurFile()", "_NewHandleClear()", 0); 
		else
			printError("Unable to read GridCur file.");
	}
	return err;
}


/////////////////////////////////////////////////////////////////


OSErr GridCurMover::TextRead(char *path) 
{
	WorldRect bounds;
	OSErr err = 0;
	char pathName[256];
	
	TRectGridVel *rectGrid = nil;
	
	if (!path || !path[0]) return 0;
	
	strcpy(fPathName,path);
	
	strcpy(pathName,fPathName);
	SplitPathFile(pathName,fFileName);
	
	// code goes here, we need to worry about really big files
	
	// do the readgridcur file stuff, store numrows, numcols, return the bounds
	err = this -> ReadHeaderLines(path,&bounds);
	if(err)
		goto done;
	
	/////////////////////////////////////////////////
	
	rectGrid = new TRectGridVel;
	if (!rectGrid)
	{		
		err = true;
		TechError("Error in GridCurMover::TextRead()","new TRectGridVel" ,err);
		goto done;
	}
	
	fGrid = (TGridVel*)rectGrid;
	
	rectGrid -> SetBounds(bounds); 
	
	// scan through the file looking for "[TIME ", then read and record the time, filePosition, and length of data
	// consider the possibility of multiple files
	/*NthLineInTextOptimized(*f, (line)++, s, 256); 
	 if(!strstr(s,"[FILE]")) 
	 {	// single file
	 err = ScanFileForTimes(path,&fTimeDataHdl,true);
	 if (err) goto done;
	 }
	 else
	 {	// multiple files
	 long numLinesInText = NumLinesInText(*f);
	 long numFiles = (numLinesInText - (line - 1))/3;	// 3 lines for each file - filename, starttime, endtime
	 strcpy(fPathName,s+strlen("[FILE]\t"));
	 err = ScanFileForTimes(fPathName,&fTimeDataHdl,true);
	 if (err) goto done;
	 line--;
	 err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);
	 }*/
	
	//err = ScanFileForTimes(path,&fTimeDataHdl);
	//if (err) goto done;
	
	
done:
	
	if(err)
	{
		printError("An error occurred in GridCurMover::TextRead"); 
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
	}
	return err;
	
	// rest of file (i.e. velocity data) is read as needed
}


OSErr GridCurMover::ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile)
{
	long i,numScanned;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], classicPath[256];
	
	PtCurFileInfoH inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("GridCurMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i=0;i<numFiles;i++)	// should count files as go along, and check that they exist ?
	{
		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 	// check it is a [FILE] line
		//strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE]\t"));
		RemoveLeadingAndTrailingWhiteSpace(s);
		strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace((*inputFilesHdl)[i].pathName);
		// allow for a path relative to the GNOME directory
		ResolvePathFromInputFile(pathOfInputfile,(*inputFilesHdl)[i].pathName); // JLM 6/8/10, we need to pass in the input file path so we can use it here
		
		if((*inputFilesHdl)[i].pathName[0] && FileExists(0,0,(*inputFilesHdl)[i].pathName))
		{
			//
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to GridCur data File does not exist.%s%s",NEWLINESTRING,(*inputFilesHdl)[i].pathName);
			printError(msg);
			err = true;
			goto done;
		}
		
		
		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); // check it is a [STARTTIME] line
		RemoveLeadingAndTrailingWhiteSpace(s);
		
		numScanned=sscanf(s+strlen("[STARTTIME]"), "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("GridCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		// not allowing constant current in separate file
		//if (time.day == time.month == time.year == time.hour == time.minute == -1)
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
		{
			timeSeconds = CONSTANTCURRENT;
		}
		else // time varying current
		{
			CheckYear(&time.year);
			
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}
		(*inputFilesHdl)[i].startTime = timeSeconds;
		
		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); // check it is an [ENDTIME] line
		RemoveLeadingAndTrailingWhiteSpace(s);
		
		/*strToMatch = "[ENDTIME]";
		 len = strlen(strToMatch);
		 //NthLineInTextOptimized (sectionOfFile, line = 0, s, 1024);
		 if(!strncmp(s,strToMatch,len)) 
		 {
		 numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
		 &time.day, &time.month, &time.year,
		 &time.hour, &time.minute) ;
		 if (numScanned!= 5)
		 { err = -1; TechError("GridCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		 }
		 */
		numScanned=sscanf(s+strlen("[ENDTIME]"), "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("GridCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
			//if (time.day == time.month == time.year == time.hour == time.minute == -1)
		{
			timeSeconds = CONSTANTCURRENT;
		}
		else // time varying current
		{
			CheckYear(&time.year);
			
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}
		(*inputFilesHdl)[i].endTime = timeSeconds;
	}
	*inputFilesH = inputFilesHdl;
	
done:
	if (err)
	{
		if(inputFilesHdl) {DisposeHandle((Handle)inputFilesHdl); inputFilesHdl=0;}
	}
	return err;
}

OSErr GridCurMover::ScanFileForTimes(char *path, PtCurTimeDataHdl *timeDataH,Boolean setStartTime)
{
	// scan through the file looking for times "[TIME "  (close file if necessary...)
	
	OSErr err = 0;
	CHARH h = 0;
	char *sectionOfFile = 0;
	
	long fileLength,lengthRemainingToScan,offset;
	long lengthToRead,lengthOfPartToScan,numTimeBlocks=0;
	long i, numScanned;
	DateTimeRec time;
	Seconds timeSeconds;	
	
	// allocate an empty handle
	PtCurTimeDataHdl timeDataHdl;
	timeDataHdl = (PtCurTimeDataHdl)_NewHandle(0);
	if(!timeDataHdl) {TechError("GridCurMover::ScanFileForTimes()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	// think in terms of 100K blocks, allocate 101K, read 101K, scan 100K
	
#define kGridCurFileBufferSize  100000 // code goes here, increase to 100K or more
#define kGridCurFileBufferExtraCharSize  256
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) goto done;
	
	offset = 0;
	lengthRemainingToScan = fileLength - 5;
	
	// loop until whole file is read 
	
	h = (CHARH)_NewHandle(2* kGridCurFileBufferSize+1);
	if(!h){TechError("GridCurMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	_HLock((Handle)h);
	sectionOfFile = *h;
	
	while (lengthRemainingToScan>0)
	{
		if(lengthRemainingToScan > 2* kGridCurFileBufferSize)
		{
			lengthToRead = kGridCurFileBufferSize + kGridCurFileBufferExtraCharSize; 
			lengthOfPartToScan = kGridCurFileBufferSize; 		
		}
		else
		{
			// deal with it in one piece
			// just read the rest of the file
			lengthToRead = fileLength - offset;
			lengthOfPartToScan = lengthToRead - 5; 
		}
		
		err = ReadSectionOfFile(0,0,path,offset,lengthToRead,sectionOfFile,0);
		if(err || !h) goto done;
		sectionOfFile[lengthToRead] = 0; // make it a C string
		
		lengthRemainingToScan -= lengthOfPartToScan;
		
		
		// scan 100K chars of the buffer for '['
		for(i = 0; i < lengthOfPartToScan; i++)
		{
			if(	sectionOfFile[i] == '[' 
			   && sectionOfFile[i+1] == 'T'
			   && sectionOfFile[i+2] == 'I'
			   && sectionOfFile[i+3] == 'M'
			   && sectionOfFile[i+4] == 'E')
			{
				// read and record the time and filePosition
				PtCurTimeData timeData;
				memset(&timeData,0,sizeof(timeData));
				timeData.fileOffsetToStartOfData = i + offset;
				
				if (numTimeBlocks > 0) 
				{
					(*timeDataHdl)[numTimeBlocks-1].lengthOfData = i+offset - (*timeDataHdl)[numTimeBlocks-1].fileOffsetToStartOfData;					
				}
				// some sort of a scan
				numScanned=sscanf(sectionOfFile+i+6, "%hd %hd %hd %hd %hd",
								  &time.day, &time.month, &time.year,
								  &time.hour, &time.minute) ;
				if (numScanned != 5)
				{ err = -1; TechError("GridCurMover::TextRead()", "sscanf() == 5", 0); goto done; }
				// check for constant current
				if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
					//if (time.day == time.month == time.year == time.hour == time.minute == -1)
				{
					timeSeconds = CONSTANTCURRENT;
					setStartTime = false;
				}
				else // time varying current
				{
					if (time.year < 1900)					// two digit date, so fix it
					{
						if (time.year >= 40 && time.year <= 99)	
							time.year += 1900;
						else
							time.year += 2000;					// correct for year 2000 (00 to 40)
					}
					
					time.second = 0;
					DateToSeconds (&time, &timeSeconds);
				}
				
				timeData.time = timeSeconds;
				
				// if we don't know the number of times ahead of time
				_SetHandleSize((Handle) timeDataHdl, (numTimeBlocks+1)*sizeof(timeData));
				if (_MemError()) { TechError("GridCurMover::TextRead()", "_SetHandleSize()", 0); goto done; }
				if (numTimeBlocks==0 && setStartTime) 
				{	// set the default times to match the file
					model->SetModelTime(timeSeconds);
					model->SetStartTime(timeSeconds);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
				}
				(*timeDataHdl)[numTimeBlocks++] = timeData;				
			}
		}
		offset += lengthOfPartToScan;
	}
	if (numTimeBlocks > 0)  // last block goes to end of file
	{
		(*timeDataHdl)[numTimeBlocks-1].lengthOfData = fileLength - (*timeDataHdl)[numTimeBlocks-1].fileOffsetToStartOfData;				
	}
	*timeDataH = timeDataHdl;
	
	
	
done:
	
	if(h) {
		_HUnlock((Handle)h); 
		DisposeHandle((Handle)h); 
		h = 0;
	}
	if (err)
	{
		if(timeDataHdl) {DisposeHandle((Handle)timeDataHdl); timeDataHdl=0;}
	}
	return err;
}

/////////////////////////////////////////////////

OSErr GridCurMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	char s[256], path[256]; 
	long i,line = 0;
	long offset,lengthToRead;
	CHARH h = 0;
	char *sectionOfFile = 0;
	char *strToMatch = 0;
	long len,numScanned;
	VelocityFH velH = 0;
	long totalNumberOfVels = fNumRows * fNumCols;
	long numLinesInBlock;
	
	OSErr err = 0;
	DateTimeRec time;
	Seconds timeSeconds;
	errmsg[0]=0;
	
	strcpy(path,fPathName);
	if (!path || !path[0]) return -1;
	
	lengthToRead = (*fTimeDataHdl)[index].lengthOfData;
	offset = (*fTimeDataHdl)[index].fileOffsetToStartOfData;
	
	h = (CHARH)_NewHandle(lengthToRead+1);
	if(!h){TechError("GridCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	_HLock((Handle)h);
	sectionOfFile = *h;			
	
	err = ReadSectionOfFile(0,0,path,offset,lengthToRead,sectionOfFile,0);
	if(err || !h) 
	{
		char firstPartOfLine[128];
		sprintf(errmsg,"Unable to open data file:%s",NEWLINESTRING);
		strncpy(firstPartOfLine,path,120);
		strcpy(firstPartOfLine+120,"...");
		strcat(errmsg,firstPartOfLine);
		goto done;
	}
	sectionOfFile[lengthToRead] = 0; // make it a C string
	numLinesInBlock = NumLinesInText(sectionOfFile);
	
	// some other way to calculate
	velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*totalNumberOfVels);
	if(!velH){TechError("GridCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	strToMatch = "[TIME]";
	len = strlen(strToMatch);
	NthLineInTextOptimized (sectionOfFile, line = 0, s, 256);
	if(!strncmp(s,strToMatch,len)) 
	{
		numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("GridCurMover::ReadTimeData()", "sscanf() == 5", 0); goto done; }
		// check for constant current
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
			//if (time.year == time.month == time.day == time.hour == time.minute == -1) 
		{
			timeSeconds = CONSTANTCURRENT;
		}
		else // time varying current
		{
			if (time.year < 1900)					// two digit date, so fix it
			{
				if (time.year >= 40 && time.year <= 99)	
					time.year += 1900;
				else
					time.year += 2000;					// correct for year 2000 (00 to 40)
			}
			
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}
		
		// check time is correct
		if (timeSeconds!=(*fTimeDataHdl)[index].time)
		{ err = -1;  strcpy(errmsg,"Can't read data - times in the file have changed."); goto done; }
		line++;
	}
	
	// allow to omit areas of the grid with zero velocity, use length of data info
	//for(i=0;i<totalNumberOfVels;i++) // interior points
	for(i=0;i<numLinesInBlock-1;i++) // lines of data
	{
		VelocityRec vel;
		long rowNum,colNum;
		long index;
		
		NthLineInTextOptimized(sectionOfFile, line++, s, 256); 	// in theory should run out of lines eventually
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		numScanned = sscanf(s,lfFix("%ld %ld %lf %lf"),&rowNum,&colNum,&vel.u,&vel.v);
		if(numScanned != 4 
		   || rowNum <= 0 || rowNum > fNumRows
		   || colNum <= 0 || colNum > fNumCols
		   )
		{ 
			err = -1;  
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read velocity data from line %ld:%s",line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done; 
		}
		index = (rowNum -1) * fNumCols + colNum-1;
		(*velH)[index].u = vel.u; // units ??? assumed m/s
		(*velH)[index].v = vel.v; 
	}
	*velocityH = velH;
	
done:
	
	if(h) {
		_HUnlock((Handle)h); 
		DisposeHandle((Handle)h); 
		h = 0;
	}
	
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridCurMover::ReadTimeData");
		//printError(errmsg); // This alert causes a freeze up...
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	return err;
	
}

