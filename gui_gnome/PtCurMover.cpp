#include "Earl.h"
#include "TypeDefs.h"
#include "Cross.h"
#include "OUtils.h"
#include "Uncertainty.h"
#include "DagTreeIO.h"
#include "GridVel.h"
#include "my_build_list.h"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT PTCURMOVER
#endif
#endif

/*enum {
		I_PTCURNAME = 0 ,
		I_PTCURACTIVE, 
		I_PTCURGRID, 
		I_PTCURARROWS,
	   I_PTCURSCALE,
		I_PTCURUNCERTAINTY,
		I_PTCURSTARTTIME,
		I_PTCURDURATION, 
		I_PTCURALONGCUR,
		I_PTCURCROSSCUR,
		I_PTCURMINCURRENT
		};
*/
Boolean IsPtCurFile (char *path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{	// must start with [FILETYPE] PTCUR
		char * strToMatch = "[FILETYPE]\tPTCUR";
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (!strncmp (strLine,strToMatch,strlen(strToMatch)))
			bIsValid = true;
	}
	
	return bIsValid;
}

/////////////////////////////////////////////////

static PtCurMover *sPtCurDialogMover;
static Boolean sDialogUncertaintyChanged;


short PtCurMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	
	switch (itemNum) {
		case M30OK:
		{
			float arrowDepth = 0, maxDepth;
			double tempAlong, tempCross, tempDuration, tempStartTime, tempUncertMin;
			TMap *map = sPtCurDialogMover -> GetMoverMap();
			if (map && map->IAm(TYPE_PTCURMAP))
			{	// can't set depth on first viewing of dialog since map doesn't exist yet so can't check value
				maxDepth = /*CHECK*/(dynamic_cast<PtCurMap *>(map)) -> GetMaxDepth2();
				arrowDepth = EditText2Float(dialog, M30ARROWDEPTH);
				if (arrowDepth > maxDepth)
				{
					char errStr[64];
					sprintf(errStr,"The maximum depth of the region is %g meters.",maxDepth);
					printError(errStr);
					break;
				}
			}
			mygetitext(dialog, M30NAME, sPtCurDialogMover->fVar.userName, kPtCurUserNameLen-1);
			sPtCurDialogMover -> bActive = GetButton(dialog, M30ACTIVE);
			sPtCurDialogMover->fVar.bShowArrows = GetButton(dialog, M30SHOWARROWS);
			sPtCurDialogMover->fVar.arrowScale = EditText2Float(dialog, M30ARROWSCALE);
			sPtCurDialogMover->fVar.arrowDepth = arrowDepth;
			sPtCurDialogMover->fVar.curScale = EditText2Float(dialog, M30SCALE);


			tempAlong = EditText2Float(dialog, M30ALONG)/100;
			tempCross = EditText2Float(dialog, M30CROSS)/100;
			tempUncertMin = EditText2Float(dialog, M30MINCURRENT);
			tempStartTime = EditText2Float(dialog, M30STARTTIME);
			tempDuration = EditText2Float(dialog, M30DURATION);
			if (sPtCurDialogMover->fVar.alongCurUncertainty != tempAlong || sPtCurDialogMover->fVar.crossCurUncertainty != tempCross
				|| sPtCurDialogMover->fVar.startTimeInHrs != tempStartTime || sPtCurDialogMover->fVar.durationInHrs != tempDuration
				|| sPtCurDialogMover->fVar.uncertMinimumInMPS != tempUncertMin) sDialogUncertaintyChanged = true;
			sPtCurDialogMover->fVar.alongCurUncertainty = EditText2Float(dialog, M30ALONG)/100;
			sPtCurDialogMover->fVar.crossCurUncertainty = EditText2Float(dialog, M30CROSS)/100;
			sPtCurDialogMover->fVar.uncertMinimumInMPS = EditText2Float(dialog, M30MINCURRENT);
			sPtCurDialogMover->fVar.startTimeInHrs = EditText2Float(dialog, M30STARTTIME);
			sPtCurDialogMover->fVar.durationInHrs = EditText2Float(dialog, M30DURATION);
	
			sPtCurDialogMover->fDownCurUncertainty = -sPtCurDialogMover->fVar.alongCurUncertainty; 
			sPtCurDialogMover->fUpCurUncertainty = sPtCurDialogMover->fVar.alongCurUncertainty; 	
			sPtCurDialogMover->fRightCurUncertainty = sPtCurDialogMover->fVar.crossCurUncertainty;  
			sPtCurDialogMover->fLeftCurUncertainty = -sPtCurDialogMover->fVar.crossCurUncertainty; 
			sPtCurDialogMover->fDuration = sPtCurDialogMover->fVar.durationInHrs * 3600.;  
			sPtCurDialogMover->fUncertainStartTime = (long) (sPtCurDialogMover->fVar.startTimeInHrs * 3600.); 

			return M30OK;
		}

		case M30CANCEL: 
			return M30CANCEL;
		
		case M30ACTIVE:
		case M30SHOWARROWS:
			ToggleButton(dialog, itemNum);
			break;
		
		case M30ARROWSCALE:
		case M30ARROWDEPTH:
		//case M30SCALE:
		case M30ALONG:
		case M30CROSS:
		case M30MINCURRENT:
		case M30STARTTIME:
		case M30DURATION:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M30SCALE:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);	// decide whether to allow half hours
	}
	
	return 0;
}


OSErr PtCurMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
{
	
	SetDialogItemHandle(dialog, M30HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M30UNCERTAINTYBOX, (Handle)FrameEmbossed);
	
	mysetitext(dialog, M30NAME, sPtCurDialogMover->fVar.userName);
	SetButton(dialog, M30ACTIVE, sPtCurDialogMover->bActive);
	
	SetButton(dialog, M30SHOWARROWS, sPtCurDialogMover->fVar.bShowArrows);
	Float2EditText(dialog, M30ARROWSCALE, sPtCurDialogMover->fVar.arrowScale, 6);
	Float2EditText(dialog, M30ARROWDEPTH, sPtCurDialogMover->fVar.arrowDepth, 6);
	
	ShowHideDialogItem(dialog, M30ARROWDEPTHAT, sPtCurDialogMover->fVar.gridType != TWO_D); 
	ShowHideDialogItem(dialog, M30ARROWDEPTH, sPtCurDialogMover->fVar.gridType != TWO_D); 
	ShowHideDialogItem(dialog, M30ARROWDEPTHUNITS, sPtCurDialogMover->fVar.gridType != TWO_D); 
	
	ShowHideDialogItem(dialog, M30BAROMODESINPUT, false); 

	Float2EditText(dialog, M30SCALE, sPtCurDialogMover->fVar.curScale, 6);
	Float2EditText(dialog, M30ALONG, sPtCurDialogMover->fVar.alongCurUncertainty*100, 6);
	Float2EditText(dialog, M30CROSS, sPtCurDialogMover->fVar.crossCurUncertainty*100, 6);
	Float2EditText(dialog, M30MINCURRENT, sPtCurDialogMover->fVar.uncertMinimumInMPS, 6);
	Float2EditText(dialog, M30STARTTIME, sPtCurDialogMover->fVar.startTimeInHrs, 6);
	Float2EditText(dialog, M30DURATION, sPtCurDialogMover->fVar.durationInHrs, 6);
	

	//ShowHideDialogItem(dialog, M30TIMEZONEPOPUP, false); 
	//ShowHideDialogItem(dialog, M30TIMESHIFTLABEL, false); 
	//ShowHideDialogItem(dialog, M30TIMESHIFT, false); 
	//ShowHideDialogItem(dialog, M30GMTOFFSETS, false); 

	MySelectDialogItemText(dialog, M30ALONG, 0, 100);
	
	return 0;
}



OSErr PtCurMover::SettingsDialog()
{
	short item;
	
	sPtCurDialogMover = this;
	sDialogUncertaintyChanged = false;
	item = MyModalDialog(M30, mapWindow, 0, PtCurMoverSettingsInit, PtCurMoverSettingsClick);
	sPtCurDialogMover = 0;

	if(M30OK == item)	
	{
		if (sDialogUncertaintyChanged) this->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
		model->NewDirtNotification();// tell model about dirt
	}
	return M30OK == item ? 0 : -1;
}




/////////////////////////////////////////////////
/////////////////////////////////////////////////

PtCurMover::PtCurMover (TMap *owner, char *name) : TCurrentMover(owner, name)
{
	memset(&fVar,0,sizeof(fVar));
	fVar.arrowScale = 5;
	fVar.arrowDepth = 0;
	fVar.alongCurUncertainty = .5;
	fVar.crossCurUncertainty = .25;
	//fVar.uncertMinimumInMPS = .05;
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	fVar.durationInHrs = 24.0;
	fVar.numLandPts = 0; // default that boundary velocities are given
	fVar.maxNumDepths = 1; // 2D default
	fVar.gridType = TWO_D; // 2D default
	fVar.bLayerThickness = 0.; // FREESLIP default
	//
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);
	//
	fGrid = 0;
	fTimeDataHdl = 0;
	fIsOptimizedForStep = false;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;

	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;

	fDepthsH = 0;
	fDepthDataInfo = 0;
	fInputFilesHdl = 0;	// for multiple files case
	
	SetClassName (name); // short file name
}

OSErr PtCurMover::InitMover()
{	
	OSErr	err = noErr;
	err = TCurrentMover::InitMover ();
	return err;
}

OSErr PtCurMover::CheckAndScanFile(char *errmsg)
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
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
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
				strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeDataHdl,false);
			strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum].pathName);
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
				strcpy(fVar.pathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeDataHdl,false);
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
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

Boolean PtCurMover::CheckInterval(long &timeDataInterval)
{
	Seconds time = model->GetModelTime();
	long i,numTimes;


	numTimes = this -> GetNumTimesInFile(); 

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



OSErr PtCurMover::PrepareForModelStep()
{
	long timeDataInterval;
	//Boolean intervalLoaded;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	//this is done first thing in SetInterval
	//check to see that the time interval is loaded
	/*intervalLoaded = this -> CheckInterval(timeDataInterval);
	 
	 if(timeDataInterval<=0||timeDataInterval>=this->GetNumTimesInFile())
	 {
	 err = -1;
	 strcpy(errmsg,"Time outside of interval being modeled");
	 goto done;
	 }
	 
	 if(!intervalLoaded)*/
	
	if (model->GetModelTime() == model->GetStartTime())	// first step, save depth range here?
	{
		if (moverMap->IAm(TYPE_PTCURMAP))
		{
			/*OK*/(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
			/*OK*/(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
			if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)	// I think this will always be 3D, but maybe old SAV files...
				((TTriGridVel3D*)fGrid)->ClearOutputHandles();
		}
	}
	if (!bActive) return 0; 
	err = this -> SetInterval(errmsg);
	if(err) goto done;
	
	fIsOptimizedForStep = true;
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in PtCurMover::PrepareForModelStep");
		printError(errmsg); 
	}
	return err;
}

OSErr PtCurMover::SetInterval(char *errmsg)
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
		if(!errmsg[0])strcpy(errmsg,"Error in PtCurMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;

}



void PtCurMover::Dispose ()
{
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}

	if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
	if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}

	TCurrentMover::Dispose ();
}


#define PtCurMoverREADWRITEVERSION 1 //JLM

OSErr PtCurMover::Write (BFPB *bfpb)
{
	char c;
	long i, version = PtCurMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	VelocityRec velocity;
	long 	numDepths = GetNumDepths(), amtTimeData = GetNumTimesInFile();
	long numPoints, numFiles;
	float val;
	PtCurTimeData timeData;
	DepthDataInfo depthData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;

	if (err = TCurrentMover::Write (bfpb)) return err;

	StartReadWriteSequence("PtCurMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	if (err = WriteMacValue(bfpb, fVar.pathName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fVar.userName, kPtCurUserNameLen)) return err;
	if (err = WriteMacValue(bfpb, fVar.alongCurUncertainty)) return err;
	if (err = WriteMacValue(bfpb, fVar.crossCurUncertainty)) return err;
	if (err = WriteMacValue(bfpb, fVar.uncertMinimumInMPS)) return err;
	if (err = WriteMacValue(bfpb, fVar.curScale)) return err;
	if (err = WriteMacValue(bfpb, fVar.startTimeInHrs)) return err;
	if (err = WriteMacValue(bfpb, fVar.durationInHrs)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.numLandPts)) return err;
	if (err = WriteMacValue(bfpb, fVar.maxNumDepths)) return err;
	if (err = WriteMacValue(bfpb, fVar.gridType)) return err;
	if (err = WriteMacValue(bfpb, fVar.bLayerThickness)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.bShowGrid)) return err;
	if (err = WriteMacValue(bfpb, fVar.bShowArrows)) return err;
	if (err = WriteMacValue(bfpb, fVar.bUncertaintyPointOpen)) return err;
	if (err = WriteMacValue(bfpb, fVar.arrowScale)) return err;
	if (err = WriteMacValue(bfpb, fVar.arrowDepth)) return err;
	
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	if (err = fGrid -> Write (bfpb)) goto done;
	
	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fDepthsH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
	
	if (err = WriteMacValue(bfpb, amtTimeData)) goto done;
	for (i=0;i<amtTimeData;i++)
	{
		timeData = INDEXH(fTimeDataHdl,i);
		if (err = WriteMacValue(bfpb, timeData.fileOffsetToStartOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.lengthOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.time)) goto done;
	}

	numPoints = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i = 0 ; i < numPoints ; i++) {
		depthData = INDEXH(fDepthDataInfo,i);
		if (err = WriteMacValue(bfpb, depthData.totalDepth)) goto done;
		if (err = WriteMacValue(bfpb, depthData.indexToDepthData)) goto done;
		if (err = WriteMacValue(bfpb, depthData.numDepths)) goto done;
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
		TechError("PtCurMover::Write(char* path)", " ", 0); 

	return err;
}

OSErr PtCurMover::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[64];
	long i, version, numDepths, amtTimeData, numPoints, numFiles = 0;
	ClassID id;
	float val;
	PtCurTimeData timeData;
	DepthDataInfo depthData;
	PtCurFileInfo fileInfo;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("PtCurMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("PtCurMover::Read()", "id != TYPE_PTCURMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != PtCurMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, fVar.pathName, kMaxNameLen)) return err;
	ResolvePath(fVar.pathName); // JLM 6/3/10
	//if (!FileExists(0,0,fVar.pathName)) {/*err=-1;*/ sprintf(msg,"The file path %s is no longer valid.",fVar.pathName); printNote(msg); /*goto done;*/}
	if (!FileExists(0,0,fVar.pathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],*p;
		strcpy(fileName,"");
		strcpy(newPath,fVar.pathName);
		p = strrchr(newPath,DIRDELIMITER);
		if (p) 
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{bPathIsValid = false;}
		else
			strcpy(fVar.pathName,fileName);

	}
	if (err = ReadMacValue(bfpb, fVar.userName, kPtCurUserNameLen)) return err;

	if (!bPathIsValid)
	{	// try other platform
		char delimStr[32] ={DIRDELIMITER,0};		
		strcpy(fileName,delimStr);
		strcat(fileName,fVar.userName);
		ResolvePath(fileName);
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{/*err=-1;*/ /*sprintf(msg,"The file path %s is no longer valid.",fVar.pathName); printNote(msg);*/ /*goto done;*/}
		else
		{
			strcpy(fVar.pathName,fileName);
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
		sprintf(msg,"This save file references a ptCUR file which cannot be found.  Please find the file \"%s\".",fVar.pathName);printNote(msg);
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(newPath, reply.fullPath);
		strcpy (s, newPath);
		SplitPathFile (s, fileName);
		strcpy (fVar.pathName, newPath);
		strcpy (fVar.userName, fileName);
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
			strcpy (fVar.pathName, newPath);
			strcpy (fVar.userName, fileName);
		}
#endif			
	}
	
	if (err = ReadMacValue(bfpb, &fVar.alongCurUncertainty)) return err;
	if (err = ReadMacValue(bfpb, &fVar.crossCurUncertainty)) return err;
	if (err = ReadMacValue(bfpb, &fVar.uncertMinimumInMPS)) return err;
	if (err = ReadMacValue(bfpb, &fVar.curScale)) return err;
	if (err = ReadMacValue(bfpb, &fVar.startTimeInHrs)) return err;
	if (err = ReadMacValue(bfpb, &fVar.durationInHrs)) return err;
	//
	if (err = ReadMacValue(bfpb, &fVar.numLandPts)) return err;
	if (err = ReadMacValue(bfpb, &fVar.maxNumDepths)) return err;
	if (err = ReadMacValue(bfpb, &fVar.gridType)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bLayerThickness)) return err;
	//
	if (err = ReadMacValue(bfpb, &fVar.bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bShowArrows)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bUncertaintyPointOpen)) return err;
	if (err = ReadMacValue(bfpb, &fVar.arrowScale)) return err;
	if (err = ReadMacValue(bfpb, &fVar.arrowDepth)) return err;

	// read the type of grid used for the PtCur mover (should always be trigrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in PtCurMover::Read()."); return -1;
	}

	if (err = fGrid -> Read (bfpb)) goto done;
	
	if (err = ReadMacValue(bfpb, &numDepths)) goto done;	
	if (numDepths>0)
	{
		fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
		if (!fDepthsH)
			{ TechError("PtCurMover::Read()", "_NewHandleClear()", 0); goto done; }
		
		for (i = 0 ; i < numDepths ; i++) {
			if (err = ReadMacValue(bfpb, &val)) goto done;
			INDEXH(fDepthsH, i) = val;
		}
	}

	if (err = ReadMacValue(bfpb, &amtTimeData)) goto done;	
	fTimeDataHdl = (PtCurTimeDataHdl)_NewHandleClear(sizeof(PtCurTimeData)*amtTimeData);
	if(!fTimeDataHdl)
		{TechError("PtCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < amtTimeData ; i++) {
		if (err = ReadMacValue(bfpb, &timeData.fileOffsetToStartOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.lengthOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.time)) goto done;
		INDEXH(fTimeDataHdl, i) = timeData;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(DepthDataInfo)*numPoints);
	if(!fDepthDataInfo)
		{TechError("PtCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &depthData.totalDepth)) goto done;
		if (err = ReadMacValue(bfpb, &depthData.indexToDepthData)) goto done;
		if (err = ReadMacValue(bfpb, &depthData.numDepths)) goto done;
		INDEXH(fDepthDataInfo, i) = depthData;
	}

	if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
	if (numFiles > 0)	// should have increased version with this, broke old save files
	{
		fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
		if(!fInputFilesHdl)
			{TechError("PtCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
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
	
done:
	if(err)
	{
		TechError("PtCurMover::Read(char* path)", " ", 0); 
		if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
		if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
		if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
		if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr PtCurMover::CheckAndPassOnMessage(TModelMessage *message)
{
	return TCurrentMover::CheckAndPassOnMessage(message); 
}

/////////////////////////////////////////////////
long PtCurMover::GetListLength()
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

ListItem PtCurMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64], dateStr[64];
	long numTimesInFile = GetNumTimesInFile();
	ListItem item = { this, 0, indent, 0 };
	

	if (n == 0) {
		item.index = I_PTCURNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Currents: \"%s\"", fVar.userName);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	

	if (bOpen) {
	

		if (--n == 0) {
			item.index = I_PTCURACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			item.indent++;
			return item;
		}
		
		
		if (--n == 0) {
			item.index = I_PTCURGRID;
			item.bullet = fVar.bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			sprintf(text, "Show Grid");
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_PTCURARROWS;
			item.bullet = fVar.bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,fVar.arrowScale,6);
			if (fVar.gridType==TWO_D)
				sprintf(text, "Show Velocities (@ 1 in = %s m/s)", valStr);
			else
				sprintf(text, "Show Velocities (@ 1 in = %s m/s) at %g m", valStr, fVar.arrowDepth);
			
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_PTCURSCALE;
			StringWithoutTrailingZeros(valStr,fVar.curScale,6);
			sprintf(text, "Multiplicative Scalar: %s", valStr);
			//item.indent++;
			return item;
		}
		
		
		// release time
		if (numTimesInFile>0)
		{
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fTimeDataHdl)[0].time /*+ fTimeShift*/;				
				Secs2DateString2 (time, dateStr);
				/*if(numTimesInFile>0)*/ sprintf (text, "Start Time: %s", dateStr);
				//else sprintf (text, "Time: %s", dateStr);
				return item;
			}
		}
		
		if (numTimesInFile>0)
		{
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fTimeDataHdl)[numTimesInFile-1].time /*+ fTimeShift*/;				
				Secs2DateString2 (time, dateStr);
				sprintf (text, "End Time: %s", dateStr);
				return item;
			}
		}


		if(model->IsUncertain())
		{
			if (--n == 0) {
				item.index = I_PTCURUNCERTAINTY;
				item.bullet = fVar.bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				item.indent++;
				return item;
			}

			if (fVar.bUncertaintyPointOpen) {
			
				if (--n == 0) {
					item.index = I_PTCURALONGCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.alongCurUncertainty*100,6);
					sprintf(text, "Along Current: %s %%",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_PTCURCROSSCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.crossCurUncertainty*100,6);
					sprintf(text, "Cross Current: %s %%",valStr);
					return item;
				}
			
				if (--n == 0) {
					item.index = I_PTCURMINCURRENT;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.uncertMinimumInMPS,6);
					sprintf(text, "Current Minimum: %s m/s",valStr);
					return item;
				}

				if (--n == 0) {
					item.index = I_PTCURSTARTTIME;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.startTimeInHrs,6);
					sprintf(text, "Start Time: %s hours",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_PTCURDURATION;
					//item.bullet = BULLET_DASH;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.durationInHrs,6);
					sprintf(text, "Duration: %s hours",valStr);
					return item;
				}
			
								
			}
			
		}  // uncertainty is on
		
	} // bOpen
	
	item.owner = 0;
	
	return item;
}

Boolean PtCurMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_PTCURNAME: bOpen = !bOpen; return TRUE;
			case I_PTCURGRID: fVar.bShowGrid = !fVar.bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PTCURARROWS: fVar.bShowArrows = !fVar.bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_PTCURUNCERTAINTY: fVar.bUncertaintyPointOpen = !fVar.bUncertaintyPointOpen; return TRUE;
			case I_PTCURACTIVE:
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

Boolean PtCurMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_PTCURNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
			}
			break;
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return TCurrentMover::FunctionEnabled(item, buttonID);
}

OSErr PtCurMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

/*OSErr PtCurMover::AddItem(ListItem item)
{
	if (item.index == I_PTCURNAME)
		return TMover::AddItem(item);
	
	return 0;
}*/

OSErr PtCurMover::DeleteItem(ListItem item)
{
	if (item.index == I_PTCURNAME)
		return moverMap -> DropMover(this);
	
	return 0;
}

Boolean PtCurMover::DrawingDependsOnTime(void)
{
	Boolean depends = fVar.bShowArrows;
	// if this is a constant current, we can say "no"
	if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	return depends;
}


void PtCurMover::Draw(Rect r, WorldRect view)
{
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	
	if(fGrid && (fVar.bShowArrows || fVar.bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,fVar.curScale,fVar.arrowScale,overrideDrawArrows,fVar.bShowGrid);
		if(fVar.bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't think need 3D here

			err = this -> SetInterval(errmsg);
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval);
			if(!loaded) return;

			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime;
				else
					startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
				endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
				timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			 
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				//long ptIndex = (*fDepthDataInfo)[i].indexToDepthData;
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?

				GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth

				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}

				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				//p.h = SameDifferenceX(wp.pLong);
				//p.v = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
						velocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u;
						velocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
						velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
					}
				}
				else // time varying current
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1/*ptIndex*/).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1/*ptIndex*/).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					}
				}
				if ((velocity.u != 0 || velocity.v != 0))
				{
					float inchesX = (velocity.u * fVar.curScale) / fVar.arrowScale;
					float inchesY = (velocity.v * fVar.curScale) / fVar.arrowScale;
					short pixX = inchesX * PixelsPerInchCurrent();
					short pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
	}
}


OSErr PtCurMover::ReadHeaderLine(char *s)
{
	char msg[512],str[256];
	char gridType[24],boundary[24];
	char *strToMatch = 0;
	long len,numScanned,longVal;
	double val=0.;
	if(s[0] != '[')
		return -1; // programmer error
	
	switch(s[1]) {
		case 'C':
			strToMatch = "[CURSCALE]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				numScanned = sscanf(s+len,lfFix("%lf"),&val);
				if (numScanned != 1 || val <= 0.0)
					goto BadValue; 
				fVar.curScale = val;
				return 0; // no error
			}
			break;
	
		case 'F':
			strToMatch = "[FILETYPE]";
			if(!strncmp(s,strToMatch,strlen(strToMatch))) {
				return 0; // no error, already dealt with this
			}
			break;

		case 'G':
			strToMatch = "[GRIDTYPE]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				numScanned = sscanf(s+len,"%s",gridType);
				if (numScanned != 1)
					goto BadValue; 
				if (!strncmp(gridType,"2D",strlen("2D")))
					fVar.gridType = TWO_D;
				else
				{
					// code goes here, deal with bottom boundary condition
					if (!strncmp(gridType,"BAROTROPIC",strlen("BAROTROPIC")))
						fVar.gridType = BAROTROPIC;
					else if (!strncmp(gridType,"SIGMA",strlen("SIGMA")))
						fVar.gridType = SIGMA;
					else if (!strncmp(gridType,"MULTILAYER",strlen("MULTILAYER")))
						fVar.gridType = MULTILAYER;
					numScanned = sscanf(s+len+strlen(gridType),lfFix("%s%lf"),boundary,&val);
					if (numScanned < 1 || val < 0.)
						goto BadValue; 	
					// check on FREESLIP vs NOSLIP
					fVar.bLayerThickness = val;
				}
				return 0; // no error
			}
			break;

		case 'N':
			strToMatch = "[NAME]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				strncpy(fVar.userName,s+len,kPtCurUserNameLen);
				fVar.userName[kPtCurUserNameLen-1] = 0;
				return 0; // no error
			}
			break;

		case 'M':
			strToMatch = "[MAXNUMDEPTHS]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				numScanned = sscanf(s+len,"%ld",&longVal);
				//if (numScanned != 1 || longVal <= 0.0)
				if (numScanned != 1 || longVal < 0.0)
					goto BadValue; 
				fVar.maxNumDepths = longVal;
				return 0; // no error
			}
			break;
			
		case 'U':
			///
			strToMatch = "[UNCERTALONG]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				numScanned = sscanf(s+len,lfFix("%lf"),&val);
				if (numScanned != 1 || val <= 0.0)
					goto BadValue; 
				fVar.alongCurUncertainty = val;
				return 0; // no error
			}
			///
			strToMatch = "[UNCERTCROSS]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				numScanned = sscanf(s+len,lfFix("%lf"),&val);
				if (numScanned != 1 || val <= 0.0)
					goto BadValue; 
				fVar.crossCurUncertainty = val;
				return 0; // no error
			}
			///
			strToMatch = "[UNCERTMIN]\t";
			len = strlen(strToMatch);
			if(!strncmp(s,strToMatch,len)) {
				numScanned = sscanf(s+len,lfFix("%lf"),&val);
				if (numScanned != 1 || val <= 0.0)
					goto BadValue; 
				fVar.uncertMinimumInMPS = val;
				return 0; // no error
			}
			///
			strToMatch = "[USERDATA]";
			if(!strncmp(s,strToMatch,strlen(strToMatch))) {
				return 0; // no error, but nothing to do
			}
			break;
	
	}
	// if we get here, we did not recognize the string
	strncpy(str,s,255);
	strcpy(str+250,"..."); // cute trick
	sprintf(msg,"Unrecognized line:%s%s",NEWLINESTRING,str);
	printError(msg);
	
	return -1;
	
BadValue:
	strncpy(str,s,255);
	strcpy(str+250,"..."); // cute trick
	sprintf(msg,"Bad value:%s%s",NEWLINESTRING,str);
	printError(msg);
	return -1;

}


/////////////////////////////////////////////////
#define PTCUR_DELIM_STR " \t"

Boolean IsPtCurVerticesHeaderLine(char *s, long* numPts, long* numLandPts)
{
	char* token = strtok(s,PTCUR_DELIM_STR);
	*numPts = 0;
	*numLandPts = 0;	
	if(!token || strncmpnocase(token,"VERTICES",strlen("VERTICES")) != 0)
	{
		return FALSE;
	}

	token = strtok(NULL,PTCUR_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numPts) != 1)
	{
		return FALSE;
	}

	token = strtok(NULL,PTCUR_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numLandPts) != 1)
	{
		//return FALSE;
		*numLandPts = 0;	// don't require
	}
	return TRUE;
	
	/*char* strToMatch = "VERTICES";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numPts);
		if (numScanned != 1 || *numPts <= 0.0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; */
}

/////////////////////////////////////////////////////////////////
OSErr PtCurMover::ReadPtCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH,FLOATH *bathymetryH,char* errmsg,long numPoints)
// Note: '*line' must contain the line# at which the vertex data begins
{
	LongPointHdl ptsH = nil;
	FLOATH depthsH = 0, bathymetryHdl = 0;
	DepthDataInfoH depthDataInfo = 0;
	OSErr err=-1;
	char *s;
	long i,index = 0;

	strcpy(errmsg,""); // clear it
	*pointsH = 0;

	ptsH = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numPoints));
	if(ptsH == nil)
	{
		strcpy(errmsg,"Not enough memory to read PtCur file.");
		return -1;
	}
	
	bathymetryHdl = (FLOATH)_NewHandle(sizeof(float)*(numPoints));
	if(bathymetryHdl == nil)
	{
		strcpy(errmsg,"Not enough memory to read PtCur file.");
		return -1;
	}
	
	if (fVar.gridType != TWO_D) // have depth info
	{	
		depthsH = (FLOATH)_NewHandle(0);
		if(!depthsH) {TechError("PtCurMover::ReadPtCurVertices()", "_NewHandle()", 0); err = memFullErr; goto done;}

	}
	
	depthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**depthDataInfo)*numPoints);
	if(!depthDataInfo){TechError("PtCurMover::ReadPtCurVertices()", "_NewHandle()", 0); err = memFullErr; goto done;}

	s = new char[(fVar.maxNumDepths+4)*64]; // large enough to hold ptNum, vertex, total depth, and all depths
	if(!s) {TechError("PtCurMover::ReadPtCurVertices()", "new[]", 0); err = memFullErr; goto done;}

	for(i=0;i<numPoints;i++)
	{
		LongPoint vertex;
		NthLineInTextOptimized(*fileBufH, (*line)++, s, (fVar.maxNumDepths+4)*64); 

		char* token = strtok(s,PTCUR_DELIM_STR); // points to ptNum	 - skip over (maybe check...)
		token = strtok(NULL,PTCUR_DELIM_STR); // points to x
		
		err = ScanMatrixPt(token,&vertex);
		if(err)
		{
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read vertex data from line %ld:%s",*line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done;
		}

		// should be (*ptsH)[ptNum-1] or track the original indices 
		(*ptsH)[i].h = vertex.h;
		(*ptsH)[i].v = vertex.v;

		if (fVar.gridType != TWO_D) // have depth info
		{
			double depth;
			long numDepths = 0;
			(*depthDataInfo)[i].indexToDepthData = index;

			token = strtok(NULL,PTCUR_DELIM_STR); // points to y
			
			while (numDepths!=fVar.maxNumDepths+1)
			{
				token = strtok(NULL,PTCUR_DELIM_STR); // points to a depth
				err = ScanDepth(token,&depth);
				if(err)
				{
					char firstPartOfLine[128];
					sprintf(errmsg,"Unable to read depth data from line %ld:%s",*line,NEWLINESTRING);
					strncpy(firstPartOfLine,s,120);
					strcpy(firstPartOfLine+120,"...");
					strcat(errmsg,firstPartOfLine);
					goto done;
				}

				if (depth==-1) break; // no more depths
				if (numDepths==0) // first one is actual depth at the location
				{
					(*depthDataInfo)[i].totalDepth = depth;
					(*bathymetryHdl)[i] = depth;
				}
				else
				{
					// since we don't know the number of depths ahead of time
					_SetHandleSize((Handle) depthsH, (index+numDepths)*sizeof(**depthsH));
					if (_MemError()) { TechError("PtCurMover::ReadPtCurVertices()", "_SetHandleSize()", 0); goto done; }
					(*depthsH)[index+numDepths-1] = depth; 
				}
				numDepths++;
			}
			if (numDepths==1) // first one is actual depth at the location
			{
				(*depthDataInfo)[i].numDepths = numDepths;
				//(*depthDataInfo)[i].indexToDepthData = i;			
				index+=numDepths;
			}
			else
			{
				numDepths--; // don't count the actual depth
				(*depthDataInfo)[i].numDepths = numDepths;
				index+=numDepths;
			}
		}
		else // 2D, no depth info
		{
			(*depthDataInfo)[i].indexToDepthData = i;			
			(*depthDataInfo)[i].numDepths = 1;	// surface velocity only
			(*depthDataInfo)[i].totalDepth = -1;	// unknown
			(*bathymetryHdl)[i] = -1;	// don't we always have bathymetry?
		}
	}

	*pointsH = ptsH;
	fDepthsH = depthsH;
	fDepthDataInfo = depthDataInfo;
	*bathymetryH = bathymetryHdl;
	err = noErr;

	
done:
	
	if(s) {delete[] s;  s = 0;}
	if(err) 
	{
		if(ptsH) {DisposeHandle((Handle)ptsH); ptsH = 0;}
		if(depthsH) {DisposeHandle((Handle)depthsH); depthsH = 0;}
		if(depthDataInfo) {DisposeHandle((Handle)depthDataInfo); depthDataInfo = 0;}
		if(bathymetryHdl) {DisposeHandle((Handle)bathymetryHdl); bathymetryHdl = 0;}
	}
	return err;		
}

/////////////////////////////////////////////////////////////////
OSErr PtCurMover::TextRead(char *path, TMap **newMap) 
{
	char s[1024], errmsg[256], classicPath[256];
	long i, numPoints, numTopoPoints, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH bathymetryH = 0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds;

	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;

	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	Boolean haveBoundaryData = false;

	errmsg[0]=0;
		

	if (!path || !path[0]) return 0;
	
	strcpy(fVar.pathName,path);
	
	// code goes here, we need to worry about really big files

	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("PtCurMover::TextRead()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// code goes here, worry about really long lines in the file

	// read header here
	for (i = 0 ; TRUE ; i++) {
		NthLineInTextOptimized(*f, line++, s, 1024); 
		if(s[0] != '[')
			break;
		err = this -> ReadHeaderLine(s);
		if(err)
			goto done;
	}

	// option to read in exported topology or just require cut and paste into file	
	// read triangle/topology info if included in file, otherwise calculate

	if(IsPtCurVerticesHeaderLine(s,&numPoints,&fVar.numLandPts))	// Points in Galt format
	{
		MySpinCursor();
		err = ReadPtCurVertices(f,&line,&pts,&bathymetryH,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		printError("Unable to read PtCur Triangle Velocity file."); 
		goto done;
	}

	// figure out the bounds
	bounds = voidWorldRect;
	long numPts;
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
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
		haveBoundaryData = true;
	}
	else
	{
		haveBoundaryData = false;
		// not needed for 2D files, unless there is no topo - store a flag
	}
	MySpinCursor(); // JLM 8/4/99

	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		// not needed for 2D files
	}
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		if (!haveBoundaryData) {err=-1; strcpy(errmsg,"File must have boundary data to create topology"); goto done;}
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Triangles");
		if (err = maketriangles(&topo,pts,numPoints,boundarySegs,numBoundarySegs))  // use maketriangles.cpp
			err = -1; // for now we require TTopology
		// code goes here, support Galt style ??
		DisplayMessage(0);
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99

	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Dag Tree");
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		DisplayMessage(0);
		if (errmsg[0])	
		err = -1; // for now we require TIndexedDagTree
		// code goes here, support Galt style ??
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	if (waterBoundaries && (this -> moverMap == model -> uMap || fVar.gridType != TWO_D))
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,bounds); // the map bounds are the same as the grid bounds
		if (!map) goto done;
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetWaterBoundaries(waterBoundaries);

		*newMap = map;
	}
	else
	{
		if (boundarySegs){DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (waterBoundaries){DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
	}

	/////////////////////////////////////////////////


	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in PtCurMover::TextRead()","new TTriGridVel3D" ,err);
		goto done;
	}

	fGrid = (TGridVel*)triGrid;

	triGrid -> SetBounds(bounds); 

	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//if (fDepthsH) triGrid->SetBathymetry(fDepthsH);	// maybe set both?
	if (bathymetryH) triGrid->SetDepths(bathymetryH);	// want just the bottom depths not all levels, so not fDepthsH

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	bathymetryH = 0; // because fGrid is now responsible for it
	

	// scan through the file looking for "[TIME ", then read and record the time, filePosition, and length of data
	// consider the possibility of multiple files
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(!strstr(s,"[FILE]")) 
	{	// single file
		err = ScanFileForTimes(path,&fTimeDataHdl,true);
		if (err) goto done;
	}
	else
	{	// multiple files
		long numLinesInText = NumLinesInText(*f);
		long numFiles = (numLinesInText - (line - 1))/3;	// 3 lines for each file - filename, starttime, endtime
		strcpy(fVar.pathName,s+strlen("[FILE]\t"));
		ResolvePathFromInputFile(path,fVar.pathName); // JLM 6/8/10
		err = ScanFileForTimes(fVar.pathName,&fTimeDataHdl,true);
		if (err) goto done;
		// code goes here, maybe do something different if constant current
		line--;
		err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);
	}
	//err = ScanFileForTimes(path,&fTimeDataHdl);
	//if (err) goto done;
	
		

done:

	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in PtCurMover::TextRead");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(bathymetryH) {DisposeHandle((Handle)bathymetryH); bathymetryH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
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
		if (boundarySegs){DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (waterBoundaries){DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
	}
	return err;

	// rest of file (i.e. velocity data) is read as needed
}

void CheckYear(short *year)
{
	if (*year < 1900)					// two digit date, so fix it
	{
		if (*year >= 40 && *year <= 99)	
			*year += 1900;
		else
			*year += 2000;					// correct for year 2000 (00 to 40)
	}

}

OSErr PtCurMover::ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile)
{
	long i,numScanned;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], classicPath[256];
	
	PtCurFileInfoH inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("PtCurMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i=0;i<numFiles;i++)
	{
		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 
		strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE]\t"));
		ResolvePathFromInputFile(pathOfInputfile,(*inputFilesHdl)[i].pathName); // JLM 6/8/10 , need to have path here to use this function

		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 

		numScanned=sscanf(s+strlen("[STARTTIME]"), "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			{ err = -1; TechError("PtCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		// not allowing constant current in separate file
		/*if (time.year < 1900)					// two digit date, so fix it
		{
			if (time.year >= 40 && time.year <= 99)	
				time.year += 1900;
			else
				time.year += 2000;					// correct for year 2000 (00 to 40)
		}*/
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
		//if (time.year == time.month == time.day == time.hour == time.minute == -1) 
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

		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 

		numScanned=sscanf(s+strlen("[ENDTIME]"), "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			{ err = -1; TechError("PtCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		/*if (time.year < 1900)					// two digit date, so fix it
		{
			if (time.year >= 40 && time.year <= 99)	
				time.year += 1900;
			else
				time.year += 2000;					// correct for year 2000 (00 to 40)
		}*/
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

OSErr PtCurMover::ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataH,Boolean setStartTime)
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
	if(!timeDataHdl) {TechError("PtCurMover::ScanFileForTimes()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	// think in terms of 100K blocks, allocate 101K, read 101K, scan 100K

	#define kPtCurFileBufferSize  100000 // code goes here, increase to 100K or more
	#define kPtCurFileBufferExtraCharSize  256

	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) goto done;

	offset = 0;
	lengthRemainingToScan = fileLength - 5;

	// loop until whole file is read 
	
	h = (CHARH)_NewHandle(2* kPtCurFileBufferSize+1);
	if(!h){TechError("PtCurMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}

	_HLock((Handle)h);
	sectionOfFile = *h;
	
	while (lengthRemainingToScan>0)
	{
		if(lengthRemainingToScan > 2* kPtCurFileBufferSize)
		{
			lengthToRead = kPtCurFileBufferSize + kPtCurFileBufferExtraCharSize; 
			lengthOfPartToScan = kPtCurFileBufferSize; 		
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
					{ err = -1; TechError("PtCurMover::TextRead()", "sscanf() == 5", 0); goto done; }
				// check for constant current
				//if (time.day == time.month == time.year == time.hour == time.minute == -1)
				if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
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
				if (_MemError()) { TechError("PtCurMover::TextRead()", "_SetHandleSize()", 0); goto done; }
				if (numTimeBlocks==0 && setStartTime) 
				{	// set the default times to match the file (only if this is the first time...)
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

OSErr PtCurMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	char s[256], path[256]; 
	long i,j,line = 0;
	long offset,lengthToRead;
	CHARH h = 0;
	char *sectionOfFile = 0;
	char *strToMatch = 0;
	long len,numScanned;
	VelocityFH velH = 0;
	long totalNumberOfVels = 0;
	long numDepths = 1;

	LongPointHdl ptsHdl = 0;
	TTriGridVel* triGrid = (TTriGridVel*)fGrid; // don't think need 3D here

	OSErr err = 0;
	DateTimeRec time;
	Seconds timeSeconds;
	long numPoints; 
	errmsg[0]=0;

	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;

	lengthToRead = (*fTimeDataHdl)[index].lengthOfData;
	offset = (*fTimeDataHdl)[index].fileOffsetToStartOfData;

	ptsHdl = triGrid -> GetPointsHdl();
	if(ptsHdl)
		numPoints = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
	else 
		{err=-1; goto done;} // no data
			

	h = (CHARH)_NewHandle(lengthToRead+1);
	if(!h){TechError("PtCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}

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

	totalNumberOfVels = (*fDepthDataInfo)[numPoints-1].indexToDepthData+(*fDepthDataInfo)[numPoints-1].numDepths;
	if(totalNumberOfVels<numPoints) {err=-1; goto done;} // must have at least full set of 2D velocity data
	velH = (VelocityFH)_NewHandle(sizeof(**velH)*totalNumberOfVels);
	if(!velH){TechError("PtCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}

	strToMatch = "[TIME]";
	len = strlen(strToMatch);
	NthLineInTextOptimized (sectionOfFile, line = 0, s, 256);
	if(!strncmp(s,strToMatch,len)) 
	{
		numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			{ err = -1; TechError("PtCurMover::ReadTimeData()", "sscanf() == 5", 0); goto done; }
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


	for(i=0;i<fVar.numLandPts;i++)	// zero out boundary velocity
	{
		numDepths = (*fDepthDataInfo)[i].numDepths;
		for(j=0;j<numDepths;j++) 
		{
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u = 0.0;
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v = 0.0;
		}
	}
	
	for(i=fVar.numLandPts;i<numPoints;i++) // interior points
	{
		VelocityRec vel;
		char *startScan;
		long scanLength,stringIndex=0;
		numDepths = (*fDepthDataInfo)[i].numDepths;

		char *s1 = new char[numDepths*64];
		if(!s1) {TechError("PtCurMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}

		NthLineInTextOptimized (sectionOfFile, line, s1, numDepths*64);
		startScan = &s1[stringIndex];

		for(j=0;j<numDepths;j++) 
		{
			err = ScanVelocity(startScan,&vel,&scanLength); 
			// ScanVelocity is faster than scanf, but doesn't handle scientific notation. Try a scanf on error.
			if (err)
			{
				if(sscanf(&s1[stringIndex],lfFix("%lf%lf"),&vel.u,&vel.v) < 2)
				{
					char firstPartOfLine[128];
					sprintf(errmsg,"Unable to read velocity data from line %ld:%s",line,NEWLINESTRING);
					strncpy(firstPartOfLine,s1,120);
					strcpy(firstPartOfLine+120,"...");
					strcat(errmsg,firstPartOfLine);
					delete[] s1; s1=0;
					goto done;
				}
				err = 0;
			}
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u = vel.u; 
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v = vel.v; 
			stringIndex += scanLength;
			startScan = &s1[stringIndex];
		}
		line++;
		delete[] s1; s1=0;
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
			strcpy(errmsg,"An error occurred in PtCurMover::ReadTimeData");
		//printError(errmsg); // This alert causes a freeze up...
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	return err;

}


/**************************************************************************************************/
OSErr ScanVelocity (char *startChar, VelocityRec *VelocityPtr, long *scanLength)
{	// expects a number of the form 
	// <number><comma><number>
	//e.g.  "-120.2345,40.345"
	// JLM, 2/26/99 extended to handle
	// <number><whiteSpace><number>
	long	j, k, pairIndex;
	char	num [64];
	OSErr	err = 0;
	char delimiterChar = ',';
	Boolean scientificNotation = false;
	
	j = 0;	/* index into supplied string */

	for (pairIndex = 1; pairIndex <= 2 && !err; ++pairIndex)
	{
	   /* scan u, then v */
	   Boolean keepGoing = true;
	   for (k = 0 ; keepGoing; j++)
	   {	   			
			switch(startChar[j])
			{
				case ',': // delimiter
				case 0: // end of string
					keepGoing = false;
					break;
				case '.':
					num[k++] = startChar[j];
					break;
				case '+': // number
					if (!scientificNotation) 
					{
						err=-1; 
						return err;
					}
					// else number
				case '-': // number
				case '0': case '1': case '2': case '3': case '4':
				case '5': case '6': case '7': case '8': case '9':
						num[k++] = startChar[j];
						if(k>=32) // no space or comma found to signal end of number
						{
							err = -1;
							return err;
						}
					break;
				case 'e': 
						num[k++] = startChar[j];
						if(k>=32) // no space or comma found to signal end of number
						{
							err = -1;
							return err;
						}
						scientificNotation = true;
					break;
				case ' ':
				case '\t': // white space
					if(k == 0) continue; // ignore leading white space
					while(startChar[j+1] == ' ' || startChar[j+1] == '\t') j++; // move past any additional whitespace chars
					if(startChar[j+1] == ',') j++; // it was <whitespace><comma>, use the comma as the delimiter
					// we have either found a comma or will use the white space as a delimiter
					// in either case we stop this loop
					keepGoing = false;
					break;
				default:
					err = -1;
					return err;
					break;
			}
		}
		
		if(err) break; // so we break out of the main loop, shouldn't happen
		
		num[k++] = 0;									/* terminate the number-string */
		
		if (pairIndex == 1)
		{
			if (!scientificNotation) VelocityPtr -> u = atof(num);
			
			if (startChar[j] == ',')					/* increment j past the comma to next coordinate */
			{
				++j;
				delimiterChar = ','; // JLM reset the delimiter char
			}
		}
		else
		{
			if (!scientificNotation) VelocityPtr -> v = atof(num);
			*scanLength = j; // amount of input string that was read
		}
	}
	///////////////
	
	if (scientificNotation) return -2;
	return err;

}
/**************************************************************************************************/

/**************************************************************************************************/
OSErr ScanDepth (char *startChar, double *DepthPtr)
{	// expects a single depth value 
	long	j, k;
	char	num [64];
	OSErr	err = 0;
	
	j = 0;	/* index into supplied string */

	Boolean keepGoing = true;
	for (k = 0 ; keepGoing; j++)
	{	   			
		switch(startChar[j])
		{
			case 0: // end of string
				keepGoing = false;
				break;
			case '.':
				num[k++] = startChar[j];
				break;
			case '-': // depths can't be negative but -1 is end of data flag
			case '0': case '1': case '2': case '3': case '4':
			case '5': case '6': case '7': case '8': case '9':
				num[k++] = startChar[j];
				if(k>=32) // end of number not found
				{
					err = -1;
					return err;
				}
				break;
			case ' ':
			case '\t': // white space
				if(k == 0) continue; // ignore leading white space
				while(startChar[j+1] == ' ' || startChar[j+1] == '\t') j++; // move past any additional whitespace chars
				keepGoing = false;
				break;
			default:
				err = -1;
				return err;
				break;
		}
	}
		
	num[k++] = 0;									/* terminate the number-string */
	
	*DepthPtr = atof(num);
	///////////////
	
	return err;

}
/**************************************************************************************************/
OSErr PtCurMover::ReadTopology(char* path, TMap **newMap)
{
	// import PtCur triangle info so don't have to regenerate
	char s[1024], errmsg[256];
	long i, numPoints, numTopoPoints, line = 0, numPts;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;

	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;

	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;

	errmsg[0]=0;
		
	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("PtCurMover::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	MySpinCursor(); // JLM 8/4/99
	if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;

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

	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundarySegs>0)
			err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99

	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
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
	
	if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
			err = -1; // for now we require TTopology
			strcpy(errmsg,"Error in topology header line");
			if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99


	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
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
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
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


	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in PtCurMover::ReadTopology()","new TTriGridVel" ,err);
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
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in PtCurMover::ReadTopology");
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
}

Boolean PtCurMover::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	err = this -> SetInterval(errmsg);
	if(err) return false;
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	// at this point this is only showing the surface velocity values
	interpolationVal = fGrid -> GetInterpolationValues(wp.p);
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		ptIndex1 =  (*fDepthDataInfo)[interpolationVal.ptIndex1].indexToDepthData;
		ptIndex2 =  (*fDepthDataInfo)[interpolationVal.ptIndex2].indexToDepthData;
		ptIndex3 =  (*fDepthDataInfo)[interpolationVal.ptIndex3].indexToDepthData;
	}
	
	
	// Check for constant current 
	if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
	{
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			velocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
			+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
			+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			velocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
			+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
			+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			velocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
			+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
			+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			velocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
			+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
			+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	//velocity.u *= fVar.curScale; 
	//velocity.v *= fVar.curScale; 
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fVar.curScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);
	
	return true;
}


OSErr PtCurMover::ExportTopology(char* path)
{
	// export triangle info so don't have to regenerate each time
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	DAGHdl treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0;
	BFPB bfpb;

	triGrid = (TTriGridVel*)(this->fGrid);
	if (!triGrid) {printError("There is no topology to export"); return -1;}
	dagTree = triGrid->GetDagTree();
	if (dagTree) 
	{
		ptsH = dagTree->GetPointsHdl();
		topH = dagTree->GetTopologyHdl();
		treeH = dagTree->GetDagTreeHdl();
	}
	else 
	{
		printError("There is no topology to export");
		return -1;
	}
	if(!ptsH || !topH || !treeH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	
	if (moverMap->IAm(TYPE_PTCURMAP))
	{
		boundaryTypeH = /*CHECK*/(dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		boundarySegmentsH = /*CHECK*/(dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	// Write out values
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
	sprintf(hdrStr,"Vertices\t%ld\n",nver);	// total vertices
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	sprintf(hdrStr,"%ld\t%ld\n",nver,nver);	// junk line
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<nver;i++)
	{	
		x = (*ptsH)[i].h/1000000.0;
		y =(*ptsH)[i].v/1000000.0;
		//sprintf(topoStr,"%ld\t%lf\t%lf\t%lf\n",i+1,x,y,(*gDepths)[i]);	
		//sprintf(topoStr,"%ld\t%lf\t%lf\n",i+1,x,y);
		sprintf(topoStr,"%lf\t%lf\n",x,y);	// add depths 
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

	if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]); // when reading in subtracts 1
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	if (boundaryTypeH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundaryTypeH)/sizeof(long);	// should be same size as previous handle
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2) nWaterBoundaries++;
		}
		sprintf(hdrStr,"WaterBoundaries\t%ld\t%ld\n",nWaterBoundaries,nBoundarySegs);	
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2)
			//sprintf(topoStr,"%ld\n",(*boundaryTypeH)[i]);
			{
				sprintf(topoStr,"%ld\n",i);
				strcpy(buffer,topoStr);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	}
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	sprintf(hdrStr,"Topology\t%ld\n",numTriangles);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i = 0; i< numTriangles;i++)
	{
		v1 = (*topH)[i].vertex1;
		v2 = (*topH)[i].vertex2;
		v3 = (*topH)[i].vertex3;
		n1 = (*topH)[i].adjTri1;
		n2 = (*topH)[i].adjTri2;
		n3 = (*topH)[i].adjTri3;
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
			   v1, v2, v3, n1, n2, n3);

		/////
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

	numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	sprintf(hdrStr,"DAGTree\t%ld\n",dagTree->fNumBranches);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;

	for(i = 0; i<dagTree->fNumBranches; i++)
	{
		sprintf(topoStr,"%ld\t%ld\t%ld\n",(*treeH)[i].topoIndex,(*treeH)[i].branchLeft,(*treeH)[i].branchRight);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		printError("Error writing topology");
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}

WorldPoint3D PtCurMover::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depth = (*theLE).z;
	long ptIndex1,ptIndex2,ptIndex3; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	memset(&interpolationVal,0,sizeof(interpolationVal));
	
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	interpolationVal = fGrid -> GetInterpolationValues(refPoint);
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		ptIndex1 =  (*fDepthDataInfo)[interpolationVal.ptIndex1].indexToDepthData;
		ptIndex2 =  (*fDepthDataInfo)[interpolationVal.ptIndex2].indexToDepthData;
		ptIndex3 =  (*fDepthDataInfo)[interpolationVal.ptIndex3].indexToDepthData;
	}
	
	// code goes here, need interpolation in z if LE is below surface
	// what kind of weird things can triangles do below the surface ??
	if (depth>0 && interpolationVal.ptIndex1 >= 0) 
	{
		scaledPatVelocity = GetMove3D(interpolationVal,depth);
		goto scale;
	}						
	
	// Check for constant current 
	if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
	{
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			scaledPatVelocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
			+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
			+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			scaledPatVelocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
			+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
			+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
			+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
			+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
			+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
			+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	
scale:
	
	scaledPatVelocity.u *= fVar.curScale; 
	scaledPatVelocity.v *= fVar.curScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

VelocityRec PtCurMover::GetMove3D(InterpolationVal interpolationVal,float depth)
{
	// figure out which depth values the LE falls between
	// will have to interpolate in lat/long for both levels first
	// and some sort of check on the returned indices, what to do if one is below bottom?
	// for sigma model might have different depth values at each point
	// for multilayer they should be the same, so only one interpolation would be needed
	// others don't have different velocities at different depths so no interpolation is needed
	// in theory the surface case should be a subset of this case, may eventually combine
	
	long pt1depthIndex1, pt1depthIndex2, pt2depthIndex1, pt2depthIndex2, pt3depthIndex1, pt3depthIndex2;
	double topDepth, bottomDepth, depthAlpha, timeAlpha;
	VelocityRec pt1interp = {0.,0.}, pt2interp = {0.,0.}, pt3interp = {0.,0.};
	VelocityRec scaledPatVelocity = {0.,0.};
	Seconds startTime, endTime, time = model->GetModelTime();
	
	GetDepthIndices(interpolationVal.ptIndex1,depth,&pt1depthIndex1,&pt1depthIndex2);	
	GetDepthIndices(interpolationVal.ptIndex2,depth,&pt2depthIndex1,&pt2depthIndex2);	
	GetDepthIndices(interpolationVal.ptIndex3,depth,&pt3depthIndex1,&pt3depthIndex2);	
	
 	// the contributions from each point will default to zero if the depth indicies
	// come back negative (ie the LE depth is out of bounds at the grid point)
	if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
	{
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v);
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v); 
			}
		}
	}
	
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v); 
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1)
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v); 
			}
		}
	}
	scaledPatVelocity.u = pt1interp.u + pt2interp.u + pt3interp.u;
	scaledPatVelocity.v = pt1interp.v + pt2interp.v + pt3interp.v;
	
	return scaledPatVelocity;
}

