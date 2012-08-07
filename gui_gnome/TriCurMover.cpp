#include "Cross.h"
#include "OUtils.h"
#include "Uncertainty.h"
#include "DagTreeIO.h"
#include "GridVel.h"
#include "my_build_list.h"
#include "TriCurMover.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT TRICURMOVER
#endif
#endif

enum {
	I_TRICURNAME = 0 ,
	I_TRICURACTIVE, 
	I_TRICURGRID, 
	I_TRICURARROWS,
	I_TRICURSCALE,
	I_TRICURUNCERTAINTY,
	I_TRICURSTARTTIME,
	I_TRICURDURATION, 
	I_TRICURALONGCUR,
	I_TRICURCROSSCUR,
	I_TRICURMINCURRENT
};


TriCurMover::TriCurMover (TMap *owner, char *name) : TCurrentMover(owner, name)
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
	//fOverLap = false;		// for multiple files case
	//fOverLapStartTime = 0;

	memset(&fInputValues,0,sizeof(fInputValues));

	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;

	fDepthsH = 0;
	fDepthDataInfo = 0;
	//fInputFilesHdl = 0;	// for multiple files case
	
	bShowDepthContourLabels = false;
	bShowDepthContours = false;

	memset(&fLegendRect,0,sizeof(fLegendRect)); 

	SetClassName (name); // short file name
	
}

void TriCurMover::Dispose ()
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
	//if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	
	TCurrentMover::Dispose ();
}


Boolean IsTriCurFile (char *path)
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
	{	// must start with [FILETYPE] TRICUR
		char * strToMatch = "[FILETYPE]\tTRICUR";
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		if (!strncmp (strLine,strToMatch,strlen(strToMatch)))
			bIsValid = true;
	}
	
	return bIsValid;
}

/////////////////////////////////////////////////

static TriCurMover *sTriCurDialogMover;


short TriCurMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	
	switch (itemNum) {
		case M30OK:
		{
			float arrowDepth = 0, maxDepth;
			//TMap *map = sTriCurDialogMover -> GetMoverMap();
			//if (map && map->IAm(TYPE_PTCURMAP))
			{
				//maxDepth = ((PtCurMap*)map) -> GetMaxDepth();
				maxDepth = sTriCurDialogMover->GetMaxDepth();	// this finds largest centroid depth, draw uses centroid
				arrowDepth = EditText2Float(dialog, M30ARROWDEPTH);
				if (arrowDepth > maxDepth)
				{
					char errStr[64];
					sprintf(errStr,"The maximum depth of the region is %g meters.",maxDepth);
					printError(errStr);
					break;
				}
			}
			mygetitext(dialog, M30NAME, sTriCurDialogMover->fVar.userName, kPtCurUserNameLen-1);
			sTriCurDialogMover -> bActive = GetButton(dialog, M30ACTIVE);
			sTriCurDialogMover->fVar.bShowArrows = GetButton(dialog, M30SHOWARROWS);
			sTriCurDialogMover->fVar.arrowScale = EditText2Float(dialog, M30ARROWSCALE);
			sTriCurDialogMover->fVar.arrowDepth = arrowDepth;
			sTriCurDialogMover->fVar.curScale = EditText2Float(dialog, M30SCALE);
			sTriCurDialogMover->fVar.alongCurUncertainty = EditText2Float(dialog, M30ALONG)/100;
			sTriCurDialogMover->fVar.crossCurUncertainty = EditText2Float(dialog, M30CROSS)/100;
			sTriCurDialogMover->fVar.uncertMinimumInMPS = EditText2Float(dialog, M30MINCURRENT);
			sTriCurDialogMover->fVar.startTimeInHrs = EditText2Float(dialog, M30STARTTIME);
			sTriCurDialogMover->fVar.durationInHrs = EditText2Float(dialog, M30DURATION);
			
			sTriCurDialogMover->fDownCurUncertainty = -sTriCurDialogMover->fVar.alongCurUncertainty; 
			sTriCurDialogMover->fUpCurUncertainty = sTriCurDialogMover->fVar.alongCurUncertainty; 	
			sTriCurDialogMover->fRightCurUncertainty = sTriCurDialogMover->fVar.crossCurUncertainty;  
			sTriCurDialogMover->fLeftCurUncertainty = -sTriCurDialogMover->fVar.crossCurUncertainty; 
			sTriCurDialogMover->fDuration = sTriCurDialogMover->fVar.durationInHrs * 3600.;  
			sTriCurDialogMover->fUncertainStartTime = (long) (sTriCurDialogMover->fVar.startTimeInHrs * 3600.); 
			
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
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);
			break;
		case M30BAROMODESINPUT:
			sTriCurDialogMover->InputValuesDialog();
			break;
			
	}
	
	return 0;
}


OSErr TriCurMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
{
	
	SetDialogItemHandle(dialog, M30HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M30UNCERTAINTYBOX, (Handle)FrameEmbossed);
	
	mysetitext(dialog, M30NAME, sTriCurDialogMover->fVar.userName);
	SetButton(dialog, M30ACTIVE, sTriCurDialogMover->bActive);
	
	SetButton(dialog, M30SHOWARROWS, sTriCurDialogMover->fVar.bShowArrows);
	Float2EditText(dialog, M30ARROWSCALE, sTriCurDialogMover->fVar.arrowScale, 6);
	Float2EditText(dialog, M30ARROWDEPTH, sTriCurDialogMover->fVar.arrowDepth, 6);
	
	ShowHideDialogItem(dialog, M30ARROWDEPTHAT, sTriCurDialogMover->fVar.gridType != TWO_D); 
	ShowHideDialogItem(dialog, M30ARROWDEPTH, sTriCurDialogMover->fVar.gridType != TWO_D); 
	ShowHideDialogItem(dialog, M30ARROWDEPTHUNITS, sTriCurDialogMover->fVar.gridType != TWO_D); 
	
	Float2EditText(dialog, M30SCALE, sTriCurDialogMover->fVar.curScale, 6);
	Float2EditText(dialog, M30ALONG, sTriCurDialogMover->fVar.alongCurUncertainty*100, 6);
	Float2EditText(dialog, M30CROSS, sTriCurDialogMover->fVar.crossCurUncertainty*100, 6);
	Float2EditText(dialog, M30MINCURRENT, sTriCurDialogMover->fVar.uncertMinimumInMPS, 6);
	Float2EditText(dialog, M30STARTTIME, sTriCurDialogMover->fVar.startTimeInHrs, 6);
	Float2EditText(dialog, M30DURATION, sTriCurDialogMover->fVar.durationInHrs, 6);
	
	
	if (sTriCurDialogMover->fInputValues.modelType<1 || sTriCurDialogMover->fInputValues.modelType>4) ShowHideDialogItem(dialog, M30BAROMODESINPUT, false); 
	
	//ShowHideDialogItem(dialog, M30TIMEZONEPOPUP, false); 
	//ShowHideDialogItem(dialog, M30TIMESHIFTLABEL, false); 
	//ShowHideDialogItem(dialog, M30TIMESHIFT, false); 
	//ShowHideDialogItem(dialog, M30GMTOFFSETS, false); 
	
	MySelectDialogItemText(dialog, M30ALONG, 0, 100);
	
	return 0;
}



OSErr TriCurMover::SettingsDialog()
{
	short item;
	Point where = CenteredDialogUpLeft(M30);
	
	sTriCurDialogMover = dynamic_cast<TriCurMover *>(this);
	item = MyModalDialog(M30, mapWindow, 0, TriCurMoverSettingsInit, TriCurMoverSettingsClick);
	sTriCurDialogMover = 0;
	
	if(M30OK == item)	model->NewDirtNotification();// tell model about dirt
	return M30OK == item ? 0 : -1;
}


/////////////////////////////////////////////////

static BaromodesParameters *sDialogInputValues;


short BaromodesInputValuesClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	
	switch (itemNum) {
		case M34OK:
		{
			// at this point nothing can be changed 
			return M34OK;
		}
			
		case M30CANCEL: 
			return M30CANCEL;
			
	}
	
	return 0;
}


OSErr BaromodesInputValuesInit(DialogPtr dialog, VOIDPTR data)
{
	short modelType = sDialogInputValues->modelType;
	char modelTypeStr[128];
	
	SetDialogItemHandle(dialog, M34HILITEDEFAULT, (Handle)FrameDefault);
	
	mysetitext(dialog, M34CURFILEPATH, sDialogInputValues->curFilePathName);
	mysetitext(dialog, M34SSHFILEPATH, sDialogInputValues->sshFilePathName);
	mysetitext(dialog, M34PYCFILEPATH, sDialogInputValues->pycFilePathName);
	mysetitext(dialog, M34LLDFILEPATH, sDialogInputValues->lldFilePathName);
	mysetitext(dialog, M34ULDFILEPATH, sDialogInputValues->uldFilePathName);
	
	switch(modelType)
	{
		case ONELAYER_CONSTDENS: strcpy(modelTypeStr,"One Layer Constant Density Model");break;
		case ONELAYER_VARDENS: strcpy(modelTypeStr,"One Layer Variable Density Model");break;
		case TWOLAYER_CONSTDENS: strcpy(modelTypeStr,"Two Layer Constant Density Model");break;
		case TWOLAYER_VARDENS: strcpy(modelTypeStr,"Two Layer Variable Density Model");break;
		default: printError("Unrecognized Model type in BaromodesInputValuesInit()."); return -1;
	}
	mysetitext(dialog, M34MODELTYPE, modelTypeStr);
	
	Float2EditText(dialog, M34SCALEVEL, sDialogInputValues->scaleVel, 6);
	Float2EditText(dialog, M34BLTHICKNESS, sDialogInputValues->bottomBLThickness, 6);
	Float2EditText(dialog, M34UPPEREDDYVISC, sDialogInputValues->upperEddyViscosity, 6);
	Float2EditText(dialog, M34LOWEREDDYVISC, sDialogInputValues->upperEddyViscosity, 6);
	Float2EditText(dialog, M34UPPERDENS, sDialogInputValues->upperLevelDensity, 6);
	Float2EditText(dialog, M34LOWERDENS, sDialogInputValues->lowerLevelDensity, 6);
	
	ShowHideDialogItem(dialog, M34LOWEREDDYVISC, modelType == TWOLAYER_CONSTDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34UPPERDENS, modelType == ONELAYER_CONSTDENS || modelType == TWOLAYER_CONSTDENS); 
	ShowHideDialogItem(dialog, M34LOWERDENS, modelType == TWOLAYER_CONSTDENS); 
	ShowHideDialogItem(dialog, M34LOWEREDDYLABEL, modelType == TWOLAYER_CONSTDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34UPPERDENSLABEL, modelType == ONELAYER_CONSTDENS || modelType == TWOLAYER_CONSTDENS); 
	ShowHideDialogItem(dialog, M34LOWERDENSLABEL, modelType == TWOLAYER_CONSTDENS); 
	ShowHideDialogItem(dialog, M34LOWEREDDYUNITS, modelType == TWOLAYER_CONSTDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34UPPERDENSUNITS, modelType == ONELAYER_CONSTDENS || modelType == TWOLAYER_CONSTDENS); 
	ShowHideDialogItem(dialog, M34LOWERDENSUNITS, modelType == TWOLAYER_CONSTDENS); 
	ShowHideDialogItem(dialog, M34PYCFILEPATH, modelType == TWOLAYER_CONSTDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34ULDFILEPATH, modelType == ONELAYER_VARDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34LLDFILEPATH, modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34PYCNAMELABEL, modelType == TWOLAYER_CONSTDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34ULDNAMELABEL, modelType == ONELAYER_VARDENS || modelType == TWOLAYER_VARDENS); 
	ShowHideDialogItem(dialog, M34LLDNAMELABEL, modelType == TWOLAYER_VARDENS); 
	
	//MySelectDialogItemText(dialog, M34SCALEVEL, 0, 100);
	
	return 0;
}



OSErr TriCurMover::InputValuesDialog()
{
	short item;
	Point where = CenteredDialogUpLeft(M34);
	
	sDialogInputValues = &fInputValues;
	item = MyModalDialog(M34, mapWindow, 0, BaromodesInputValuesInit, BaromodesInputValuesClick);
	sDialogInputValues = 0;	//? not a handle so probably don't need to do anything here
	
	if(M34OK == item)	model->NewDirtNotification();// tell model about dirt
	return M34OK == item ? 0 : -1;
}

OSErr TriCurMover::InitMover()
{	
	OSErr	err = noErr;
	err = TCurrentMover::InitMover ();
	return err;
}


/*OSErr TriCurMover::CheckAndScanFile(char *errmsg, const Seconds& start_time, const Seconds& model_time)
 {
//  Seconds time = model->GetModelTime(), startTime, endTime, lastEndTime, testTime;	// minus AH 07/17/2012
 Seconds time = model_time, startTime, endTime, lastEndTime, testTime; // AH 07/17/2012
 
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
 //if (fOverLapStartTime==testTime)	// shift end time data to start time data
 //{
 //fStartData = fEndData;
 //ClearLoadedData(&fEndData);
 //}
 //else
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
 }*/

#define TriCurMoverREADWRITEVERSION 1 //JLM
//#define TriCurMoverREADWRITEVERSION 2 //JLM

OSErr TriCurMover::Write (BFPB *bfpb)
{
	char c;
	long i, j, version = TriCurMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	VelocityRec velocity;
	long 	numDepths = dynamic_cast<TriCurMover *>(this)->GetNumDepths(), amtTimeData = GetNumTimesInFile();
	long numPoints, numFiles, numTris;
	float val;
	PtCurTimeData timeData;
	DepthDataInfo depthData;
	//PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TCurrentMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("TriCurMover::Write()");
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
	//
	if (err = WriteMacValue(bfpb, fInputValues.curFilePathName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.sshFilePathName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.pycFilePathName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.uldFilePathName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.lldFilePathName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.modelType)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.scaleVel)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.bottomBLThickness)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.upperEddyViscosity)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.lowerEddyViscosity)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.upperLevelDensity)) return err;
	if (err = WriteMacValue(bfpb, fInputValues.lowerLevelDensity)) return err;
	//
	
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
	
	numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	if (err = WriteMacValue(bfpb, numTris)) goto done;
	for (i = 0 ; i < numTris ; i++) {
		depthData = INDEXH(fDepthDataInfo,i);
		if (err = WriteMacValue(bfpb, depthData.totalDepth)) goto done;
		if (err = WriteMacValue(bfpb, depthData.indexToDepthData)) goto done;
		if (err = WriteMacValue(bfpb, depthData.numDepths)) goto done;
	}
	
	// write out the data if constant current, don't require path on reading in
	if (amtTimeData==1)
	{
		for(i=0;i<numTris;i++) // interior points
		{
			VelocityRec vel;
			//long numDepths = (*fDepthDataInfo)[i].numDepths;
			long numDepths = fVar.maxNumDepths;
			//might want to check that the number of lines matches the number of triangles (ie there is data at every triangle)
			
			for(j=0;j<numDepths;j++) 
			{
				vel.u = INDEXH(fStartData.dataHdl,(*fDepthDataInfo)[i].indexToDepthData+j).u;
				vel.v = INDEXH(fStartData.dataHdl,(*fDepthDataInfo)[i].indexToDepthData+j).v;
				//vel.u = (*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u; 
				//vel.v = (*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v; 
				if (err = WriteMacValue(bfpb, vel.u)) goto done;
				if (err = WriteMacValue(bfpb, vel.v)) goto done;
			}
		}
	}
	/*numFiles = GetNumFiles();
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
	 }*/
	
done:
	if(err)
		TechError("TriCurMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr TriCurMover::Read(BFPB *bfpb)
{
	char c, msg[256];
	long i, j, version, numDepths, amtTimeData, numPoints, numFiles, numTris;
	ClassID id;
	float val;
	PtCurTimeData timeData;
	DepthDataInfo depthData;
	//PtCurFileInfo fileInfo;
	Boolean badPath = false;
	VelocityFH velH = 0;
	OSErr err = 0;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TriCurMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TriCurMover::Read()", "id != TYPE_TRICURMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > TriCurMoverREADWRITEVERSION || version < 1) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, fVar.pathName, kMaxNameLen)) return err;
	ResolvePath(fVar.pathName); // JLM 6/3/10
	//if (!FileExists(0,0,fVar.pathName)) {err=-1; sprintf(msg,"The file path %s is no longer valid.",fVar.pathName); printError(msg); goto done;}
	if (!FileExists(0,0,fVar.pathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],fileName[64],*p;
		strcpy(newPath,fVar.pathName);
		p = strrchr(newPath,DIRDELIMITER);
		strcpy(fileName,p);
		ResolvePath(fileName);
		if (!FileExists(0,0,fileName)) // flag this and report later if necessary
		{/*err=-1;*/ sprintf(msg,"The file path %s is no longer valid.",fVar.pathName); badPath = true; /*printNote(msg);*/ /*goto done;*/}
		else
		{strcpy(fVar.pathName,fileName); badPath = false;}
	}
	if (err = ReadMacValue(bfpb, fVar.userName, kPtCurUserNameLen)) return err;
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
	//
	if (err = ReadMacValue(bfpb, fInputValues.curFilePathName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, fInputValues.sshFilePathName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, fInputValues.pycFilePathName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, fInputValues.uldFilePathName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, fInputValues.lldFilePathName, kMaxNameLen)) return err;
	
	ResolvePath(fInputValues.curFilePathName); // JLM 6/3/10
	ResolvePath(fInputValues.sshFilePathName); // JLM 6/3/10
	ResolvePath(fInputValues.pycFilePathName); // JLM 6/3/10
	ResolvePath(fInputValues.uldFilePathName); // JLM 6/3/10
	ResolvePath(fInputValues.lldFilePathName); // JLM 6/3/10
	
	if (err = ReadMacValue(bfpb, &fInputValues.modelType)) return err;
	if (err = ReadMacValue(bfpb, &fInputValues.scaleVel)) return err;
	if (err = ReadMacValue(bfpb, &fInputValues.bottomBLThickness)) return err;
	if (err = ReadMacValue(bfpb, &fInputValues.upperEddyViscosity)) return err;
	if (err = ReadMacValue(bfpb, &fInputValues.lowerEddyViscosity)) return err;
	if (err = ReadMacValue(bfpb, &fInputValues.upperLevelDensity)) return err;
	if (err = ReadMacValue(bfpb, &fInputValues.lowerLevelDensity)) return err;
	//
	// read the type of grid used for the PtCur mover (should always be trigrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in TriCurMover::Read()."); return -1;
	}
	
	if (err = fGrid -> Read (bfpb)) goto done;
	
	if (err = ReadMacValue(bfpb, &numDepths)) goto done;	
	fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
	if (!fDepthsH)
	{ TechError("TriCurMover::Read()", "_NewHandleClear()", 0); goto done; }
	
	for (i = 0 ; i < numDepths ; i++) {
		if (err = ReadMacValue(bfpb, &val)) goto done;
		INDEXH(fDepthsH, i) = val;
	}
	
	if (err = ReadMacValue(bfpb, &amtTimeData)) goto done;	
	fTimeDataHdl = (PtCurTimeDataHdl)_NewHandleClear(sizeof(PtCurTimeData)*amtTimeData);
	if(!fTimeDataHdl)
	{TechError("TriCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < amtTimeData ; i++) {
		if (err = ReadMacValue(bfpb, &timeData.fileOffsetToStartOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.lengthOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.time)) goto done;
		INDEXH(fTimeDataHdl, i) = timeData;
	}
	
	if (err = ReadMacValue(bfpb, &numTris)) goto done;	
	fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(DepthDataInfo)*numTris);
	if(!fDepthDataInfo)
	{TechError("TriCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numTris ; i++) {
		if (err = ReadMacValue(bfpb, &depthData.totalDepth)) goto done;
		if (err = ReadMacValue(bfpb, &depthData.indexToDepthData)) goto done;
		if (err = ReadMacValue(bfpb, &depthData.numDepths)) goto done;
		INDEXH(fDepthDataInfo, i) = depthData;
	}
	
	// write out the data if constant current, don't require path on reading in
	if (amtTimeData==1 /*&& version > 1*/)
	{
		VelocityRec vel;
		//long numDepths = (*fDepthDataInfo)[i].numDepths;
		long numDepths = fVar.maxNumDepths;
		long totalNumberOfVels = (*fDepthDataInfo)[numTris-1].indexToDepthData+(*fDepthDataInfo)[numTris-1].numDepths;
		//totalNumberOfVels = numTris*numDepths;
		if(totalNumberOfVels<numTris) {err=-1; goto done;} // must have at least full set of 2D velocity data
		velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*totalNumberOfVels);
		if(!velH){TechError("TriCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for(i=0;i<numTris;i++) // interior points
		{	
			for(j=0;j<numDepths;j++) 
			{
				if (err = ReadMacValue(bfpb, &vel.u)) goto done;
				if (err = ReadMacValue(bfpb, &vel.v)) goto done;
				(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u = vel.u; 
				(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v = vel.v; 
			}
		}
		fStartData.timeIndex = 0;
		fStartData.dataHdl = velH;
	}
	else
	{	
		if (badPath) printNote(msg);
	}
	/*if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
	 if (numFiles > 0)
	 {
	 fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	 if(!fInputFilesHdl)
	 {TechError("PtCurMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	 for (i = 0 ; i < numFiles ; i++) {
	 if (err = ReadMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
	 ResolvePath(fileInfo.pathName, kMaxNameLen); // JLM 6/3/10
	 if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
	 if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
	 INDEXH(fInputFilesHdl,i) = fileInfo;
	 }
	 if (err = ReadMacValue(bfpb, &fOverLap)) return err;
	 if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
	 }*/
	
done:
	if(err)
	{
		TechError("TriCurMover::Read(char* path)", " ", 0); 
		if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
		if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
		if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
		//if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr TriCurMover::CheckAndPassOnMessage(TModelMessage *message)
{
	return TCurrentMover::CheckAndPassOnMessage(message); 
}

/////////////////////////////////////////////////
long TriCurMover::GetListLength()
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

ListItem TriCurMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64];
	ListItem item = { dynamic_cast<TriCurMover *>(this), 0, indent, 0 };
	
	
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

Boolean TriCurMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
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

Boolean TriCurMover::FunctionEnabled(ListItem item, short buttonID)
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

OSErr TriCurMover::SettingsItem(ListItem item)
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

OSErr TriCurMover::DeleteItem(ListItem item)
{
	if (item.index == I_PTCURNAME)
		return moverMap -> DropMover(dynamic_cast<TriCurMover *>(this));
	
	return 0;
}

Boolean TriCurMover::DrawingDependsOnTime(void)
{
	Boolean depends = fVar.bShowArrows;
	// if this is a constant current, we can say "no"
	if(GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/) depends = false;
	return depends;
}

void TriCurMover::DrawContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	Rect legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],text[30],errmsg[256];
	long 		i,numLevels,istep=1;
	double	minLevel, maxLevel;
	double 	value;
	float totalDepth = 0;
	long numDepths = 0, numTris = 0, triNum = 0;
	OSErr err = 0;
	PtCurMap *map = GetPtCurMap();
	TTriGridVel3D *triGrid = (TTriGridVel3D*) map->GetGrid3D(false);
	Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
	
	long timeDataInterval;
	Boolean loaded;
	
//	err = this -> SetInterval(errmsg);	// minus AH 07/17/2012
	//err = this -> SetInterval(errmsg, model->GetStartTime(), model->GetModelTime());	// AH 07/17/2012
	err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
	
	if(err) return;
	
//	loaded = this -> CheckInterval(timeDataInterval);	// minus AH 07/17/2012
	//loaded = this -> CheckInterval(timeDataInterval, model->GetStartTime(), model->GetModelTime());	// AH 07/17/2012
	loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
	
	if(!loaded) return;
	
	
	if (!fDepthDataInfo) return;
	numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);	// depth from input file (?) at triangle center
	
	//list which triNum, use selected triangle, scale arrows, list values ??? 
	if (triSelected)
	{
		for (i=0;i<numTris; i++)
		{
			if ((*triSelected)[i]) 
			{
				triNum = i;
				break;
			}
		}
	}
	else
		triNum = GetRandom(0,numTris-1);
	
	numDepths = INDEXH(fDepthDataInfo,triNum).numDepths;
	totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	
	//SetRGBColor(&rgb,0,0,0);
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	TextFont(kFontIDGeneva); TextSize(6);
#endif
	
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
	MyMoveTo(x - stringwidth("Depth Barbs") / 2, y + dY);
	drawstring("Depth Barbs");
	numtostring(triNum+1,numstr);
	strcpy(numstr2,"Tri Num = ");
	strcat(numstr2,numstr);
	MyMoveTo(x-stringwidth(numstr2) / 2, y + 2*dY);
	drawstring(numstr2);
	widestNum = stringwidth(numstr2);
	
	v = rgbrect.top+45;
	h = rgbrect.left;
	//if (numDepths>20) istep = (long)numDepths/20.;
	//for (i=0;i<numDepths;i++)
	for (i=0;i<numDepths;i+=istep)
	{
		WorldPoint wp;
		Point p,p2;
		VelocityRec velocity = {0.,0.};
		Boolean offQuickDrawPlane = false;
		
		long velDepthIndex1 = (*fDepthDataInfo)[triNum].indexToDepthData+i;
		
		velocity.u = INDEXH(fStartData.dataHdl,velDepthIndex1).u;
		velocity.v = INDEXH(fStartData.dataHdl,velDepthIndex1).v;
		
		MyMoveTo(h+40,v+.5);
		
		if ((velocity.u != 0 || velocity.v != 0))
		{
			float inchesX = (velocity.u * fVar.curScale) / fVar.arrowScale;
			float inchesY = (velocity.v * fVar.curScale) / fVar.arrowScale;
			short pixX = inchesX * PixelsPerInchCurrent();
			short pixY = inchesY * PixelsPerInchCurrent();
			//p.h = h+20;
			p.h = h+40;
			p.v = v+.5;
			p2.h = p.h + pixX;
			p2.v = p.v - pixY;
			//MyMoveTo(p.h, p.v);
			MyLineTo(p2.h, p2.v);
			MyDrawArrow(p.h,p.v,p2.h,p2.v);
		}
		if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
		v = v+9;
	}
	sprintf(text, "Depth: %g m",totalDepth);
	//MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);
	MyMoveTo(h+20, v+5);
	drawstring(text);
	if (stringwidth(text)+20 > widestNum) widestNum = stringwidth(text)+20;
	v = v + 9;
	legendRect.bottom = v+3;
	if (legendRect.right<h+20+widestNum+4) legendRect.right = h+20+widestNum+4;
	else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
		legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
	RGBForeColor(&colors[BLACK]);
 	//MyFrameRect(&legendRect);
	
	if (!gSavingOrPrintingPictFile)
		fLegendRect = legendRect;
	return;
}

void TriCurMover::Draw(Rect r, WorldRect view)
{
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth/*,totalDepth=0*/;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	long i, numTris = 0, numDepths = 0;
	if (fDepthDataInfo) numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	
	
	if(fGrid && (fVar.bShowArrows || fVar.bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,fVar.curScale,fVar.arrowScale,fVar.arrowDepth,overrideDrawArrows,fVar.bShowGrid,fColor);
		if(fVar.bShowArrows)
		{ // we have to draw the arrows
			RGBForeColor(&fColor);
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			
			err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
			
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 /*|| GetNumFiles()>1*/)
			{
				// Calculate the time weight factor
				//if (GetNumFiles()>1 && fOverLap)
				//startTime = fOverLapStartTime;
				//else
				startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
				endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
				timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			
			for(i = 0; i < numTris; i++)
			{
			 	// get the value at each triangle center and draw an arrow
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?
				err = ((TTriGridVel3D*)fGrid)->GetTriangleCentroidWC(i,&wp);
				
				dynamic_cast<TriCurMover *>(this)->GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					//if (fDepthDataInfo) totalDepth = INDEXH(fDepthDataInfo,i).totalDepth;	// depth from input file (?) at triangle center
					//else {printError("Problem with depth data in TriCurMover::Draw"); return;}
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if(GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/)
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
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
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
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
		RGBForeColor(&colors[BLACK]);
	}
	if (bShowDepthContours) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
	
}


OSErr TriCurMover::ReadHeaderLine(char *s)
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
				if (numScanned != 1 || longVal <= 0.0)
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
#define TRICUR_DELIM_STR " \t"

Boolean IsTriCurVerticesHeaderLine(char *s, long* numPts)
{
	char* token = strtok(s,TRICUR_DELIM_STR);
	*numPts = 0;
	if(!token || strncmpnocase(token,"VERTICES",strlen("VERTICES")) != 0)
	{
		return FALSE;
	}
	
	token = strtok(NULL,TRICUR_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numPts) != 1)
	{
		return FALSE;
	}
	
	return TRUE;
}

/////////////////////////////////////////////////////////////////
OSErr TriCurMover::ReadTriCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH, FLOATH *totalDepthH,char* errmsg,long numPoints)
// Note: '*line' must contain the line# at which the vertex data begins
{
	LongPointHdl ptsH = nil;
	FLOATH depthsH = 0;
	DepthDataInfoH depthDataInfo = 0;
	OSErr err=-1;
	char s[256];
	long i,index = 0;
	double depth;
	
	strcpy(errmsg,""); // clear it
	*pointsH = 0;
	
	ptsH = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numPoints));
	if(ptsH == nil)
	{
		strcpy(errmsg,"Not enough memory to read TriCur file.");
		return -1;
	}
	
	depthsH = (FLOATH)_NewHandle(sizeof(float)*numPoints);
	if(!depthsH) {TechError("TriCurMover::ReadTriCurVertices()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	for(i=0;i<numPoints;i++)
	{
		LongPoint vertex;
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		
		char* token = strtok(s,TRICUR_DELIM_STR); // points to ptNum	 - skip over (maybe check...)
		token = strtok(NULL,TRICUR_DELIM_STR); // points to x
		
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
		
		token = strtok(NULL,TRICUR_DELIM_STR); // points to y		
		token = strtok(NULL,TRICUR_DELIM_STR); // points to a depth
		
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
		
		(*depthsH)[i] = depth; 	// may want to calculate all the sigma levels once here
	}
	
	*pointsH = ptsH;
	*totalDepthH = depthsH;
	err = noErr;
	
	
done:
	
	if(err) 
	{
		if(ptsH) {DisposeHandle((Handle)ptsH); ptsH = 0;}
		if(depthsH) {DisposeHandle((Handle)depthsH); depthsH = 0;}
	}
	return err;		
}

/////////////////////////////////////////////////////////////////
Boolean IsBaromodesInputValuesHeaderLine(char *s, short* modelType)
{
	// note this method requires a dummy line in ptcur file or else the next line is garbled or skipped
	/*char* token = strtok(s,TRICUR_DELIM_STR);
	 *modelType = 0;
	 if(!token || strncmpnocase(token,"Input Values",strlen("Input Values")) != 0)
	 {
	 return FALSE;
	 }
	 
	 token = strtok(NULL,TRICUR_DELIM_STR);
	 
	 if(!token || sscanf(token,"%hd",modelType) != 1)
	 {
	 return FALSE;
	 }
	 
	 return TRUE;*/
	char* strToMatch = "Input Values";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%hd",modelType);
		if (numScanned != 1 || *modelType <= 0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}

OSErr TriCurMover::ReadBaromodesInputValues(CHARH fileBufH,long *line,BaromodesParameters *inputValues,char* errmsg,short modelType)
{
	
	BaromodesParameters inputParameters;
	OSErr err=0;
	char s[256];
	char sshFilePath[256],pycFilePath[256],curFilePath[256],lldFilePath[256],uldFilePath[256];
	double scaleVel,bottomBL,upperEddyVisc,lowerEddyVisc,upperDens,lowerDens;
	long i, numScanned;
	
	strcpy(errmsg,""); // clear it
	if (modelType<1 || modelType>4) return -1;
	
	memset(&inputParameters,0,sizeof(inputParameters));
	
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!strstr(s,"Scale velocity:")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("Scale velocity:"),lfFix("%lf"),&scaleVel);
	if(numScanned != 1 ) { err = -2; goto done; }
	inputParameters.scaleVel = scaleVel;
	
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!strstr(s,"Bottom BL thickness:")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("Bottom BL thickness:"),lfFix("%lf"),&bottomBL);
	if(numScanned != 1 ) { err = -2; goto done; }
	inputParameters.bottomBLThickness = bottomBL;
	
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!strstr(s,"Upper (or only) eddy viscosity:")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("Upper (or only) eddy viscosity:"),lfFix("%lf"),&upperEddyVisc);
	if(numScanned != 1 ) { err = -2; goto done; }
	inputParameters.upperEddyViscosity = upperEddyVisc;
	
	(*inputValues).scaleVel = inputParameters.scaleVel;
	(*inputValues).bottomBLThickness = inputParameters.bottomBLThickness;
	(*inputValues).upperEddyViscosity = inputParameters.upperEddyViscosity;
	
	if (modelType==TWOLAYER_CONSTDENS || modelType==TWOLAYER_VARDENS)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		if(!strstr(s,"Lower eddy viscosity:")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("Lower eddy viscosity:"),lfFix("%lf"),&lowerEddyVisc);
		if(numScanned != 1 ) { err = -2; goto done; }
		inputParameters.lowerEddyViscosity = lowerEddyVisc;
		
		(*inputValues).lowerEddyViscosity = inputParameters.lowerEddyViscosity;
	}
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!strstr(s,"Current file:")) { err = -2; goto done; }
	//numScanned = sscanf(s+strlen("Current file:"),lfFix("%lf"),&lowerEddyVisc);
	//if(numScanned != 1 ) { err = -2; goto done; }
	strcpy(curFilePath,s+strlen("Current file:"));
	RemoveLeadingAndTrailingWhiteSpace(curFilePath);
	strcpy(inputParameters.curFilePathName,curFilePath);
	
	NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
	if(!strstr(s,"Sea surface heights file:")) { err = -2; goto done; }
	//numScanned = sscanf(s+strlen("Sea surface heights file:"),lfFix("%lf"),&lowerEddyVisc);
	//if(numScanned != 1 ) { err = -2; goto done; }
	strcpy(sshFilePath,s+strlen("Sea surface heights file:"));
	RemoveLeadingAndTrailingWhiteSpace(sshFilePath);
	strcpy(inputParameters.sshFilePathName,sshFilePath);
	
	strcpy(inputValues->curFilePathName,inputParameters.curFilePathName);
	strcpy(inputValues->sshFilePathName,inputParameters.sshFilePathName);
	
	if (modelType==ONELAYER_CONSTDENS)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		if(!strstr(s,"Density:")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("Density:"),lfFix("%lf"),&upperDens);
		if(numScanned != 1 ) { err = -2; goto done; }
		inputParameters.upperLevelDensity = upperDens;
		
		(*inputValues).upperLevelDensity = inputParameters.upperLevelDensity;
	}
	else if (modelType==ONELAYER_VARDENS)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		if(!strstr(s,"Single layer density file:")) { err = -2; goto done; }
		//numScanned = sscanf(s+strlen("Sea surface heights file:"),lfFix("%lf"),&lowerEddyVisc);
		//if(numScanned != 1 ) { err = -2; goto done; }
		strcpy(uldFilePath,s+strlen("Single layer density file:"));
		RemoveLeadingAndTrailingWhiteSpace(uldFilePath);
		strcpy(inputParameters.uldFilePathName,uldFilePath);
		
		strcpy(inputValues->uldFilePathName,inputParameters.uldFilePathName);
	}
	else 	// 2 layer models
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		if(!strstr(s,"Pycnocline file:")) { err = -2; goto done; }
		//numScanned = sscanf(s+strlen("Sea surface heights file:"),lfFix("%lf"),&lowerEddyVisc);
		//if(numScanned != 1 ) { err = -2; goto done; }
		strcpy(pycFilePath,s+strlen("Pycnocline file:"));
		RemoveLeadingAndTrailingWhiteSpace(pycFilePath);
		strcpy(inputParameters.pycFilePathName,pycFilePath);
		
		strcpy(inputValues->pycFilePathName,inputParameters.pycFilePathName);
		
		if (modelType==TWOLAYER_CONSTDENS)
		{
			NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
			if(!strstr(s,"Upper layer density:")) { err = -2; goto done; }
			numScanned = sscanf(s+strlen("Upper layer density:"),lfFix("%lf"),&upperDens);
			if(numScanned != 1 ) { err = -2; goto done; }
			inputParameters.upperLevelDensity = upperDens;
			
			NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
			if(!strstr(s,"Lower layer density:")) { err = -2; goto done; }
			numScanned = sscanf(s+strlen("Lower layer density:"),lfFix("%lf"),&lowerDens);
			if(numScanned != 1 ) { err = -2; goto done; }
			inputParameters.lowerLevelDensity = lowerDens;
			
			(*inputValues).upperLevelDensity = inputParameters.upperLevelDensity;
			(*inputValues).lowerLevelDensity = inputParameters.lowerLevelDensity;
		}
		else if (modelType==TWOLAYER_VARDENS)
		{
			NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
			if(!strstr(s,"Upper layer density file:")) { err = -2; goto done; }
			//numScanned = sscanf(s+strlen("Sea surface heights file:"),lfFix("%lf"),&lowerEddyVisc);
			//if(numScanned != 1 ) { err = -2; goto done; }
			strcpy(uldFilePath,s+strlen("Upper layer density file:"));
			RemoveLeadingAndTrailingWhiteSpace(uldFilePath);
			strcpy(inputParameters.uldFilePathName,uldFilePath);
			
			NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
			if(!strstr(s,"Lower layer density file:")) { err = -2; goto done; }
			//numScanned = sscanf(s+strlen("Sea surface heights file:"),lfFix("%lf"),&lowerEddyVisc);
			//if(numScanned != 1 ) { err = -2; goto done; }
			strcpy(lldFilePath,s+strlen("Lower layer density file:"));
			RemoveLeadingAndTrailingWhiteSpace(lldFilePath);
			strcpy(inputParameters.lldFilePathName,lldFilePath);
			
			strcpy(inputValues->uldFilePathName,inputParameters.uldFilePathName);
			strcpy(inputValues->lldFilePathName,inputParameters.lldFilePathName);
		}
	}
	
	(*inputValues).modelType = modelType;
	//SetInputValues(inputParameters);
	
	//err = noErr;
	
done:
	
	if(err) 
	{
		
	}
	return err;		
}
/////////////////////////////////////////////////////////////////
Boolean IsCentroidDepthsHeaderLine(char *s, long* numPts)
{
	// note this method requires a dummy line in ptcur file or else the next line is garbled or skipped
	/*char* token = strtok(s,TRICUR_DELIM_STR);
	 *numPts = 0;
	 if(!token || strncmpnocase(token,"CentroidDepths",strlen("CentroidDepths")) != 0)
	 {
	 return FALSE;
	 }
	 
	 token = strtok(NULL,TRICUR_DELIM_STR);
	 
	 if(!token || sscanf(token,"%ld",numPts) != 1)
	 {
	 return FALSE;
	 }
	 
	 return TRUE;*/
	char* strToMatch = "CentroidDepths";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numPts);
		if (numScanned != 1 || *numPts <= 0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}

OSErr TriCurMover::ReadCentroidDepths(CHARH fileBufH,long *line,long numTris,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{
	FLOATH depthsH = 0;
	DepthDataInfoH depthDataInfo = 0;
	OSErr err=-1;
	//char s[256];
	char *s;
	long i,index = 0;
	double depth;
	//long numDepths = fVar.maxNumDepths;	//for now
	
	strcpy(errmsg,""); // clear it
	
	if (fVar.gridType != TWO_D) // have depth info
	{	
		depthsH = (FLOATH)_NewHandle(0);
		if(!depthsH) {TechError("TriCurMover::ReadCentroidDepths()", "_NewHandle()", 0); err = memFullErr; goto done;}
		
	}
	
	depthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**depthDataInfo)*numTris);
	if(!depthDataInfo){TechError("TriCurMover::ReadCentroidDepths()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	//s = new char[(fVar.maxNumDepths+4)*64]; // large enough to hold ptNum, vertex, total depth, and all depths
	s = new char[(fVar.maxNumDepths+1)*64]; // large enough to hold total depth and all depths
	if(!s) {TechError("TriCurMover::ReadCentroidDepths()", "new[]", 0); err = memFullErr; goto done;}
	
	for(i=0;i<numTris;i++)
	{
		long numDepths = 0;
		NthLineInTextOptimized(*fileBufH, (*line)++, s, (fVar.maxNumDepths+1)*64); 
		//NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		//if (sscanf(s,lfFix("%lf"),&depth) < 1) err = true;
		
		char* token = strtok(s,TRICUR_DELIM_STR); 	// points to a depth	
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
		
		(*depthDataInfo)[i].indexToDepthData = index;
		(*depthDataInfo)[i].totalDepth = depth;
		//(*depthDataInfo)[i].numDepths = numDepths;
		//index+=numDepths;
		
		//while (numDepths!=fVar.maxNumDepths+1)
		if (fVar.maxNumDepths == 1) numDepths = 1;
		while (numDepths!=fVar.maxNumDepths)
		{
			token = strtok(NULL,TRICUR_DELIM_STR); // points to a depth
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
			/*if (numDepths==0) // first one is actual depth at the location
			 {
			 (*depthDataInfo)[i].totalDepth = depth;
			 (*totalDepthHdl)[i] = depth;
			 }*/
			//else
			{
				// since we don't know the number of depths ahead of time
				//_SetHandleSize((Handle) depthsH, (index+numDepths)*sizeof(**depthsH));
				_SetHandleSize((Handle) depthsH, (index+numDepths+1)*sizeof(**depthsH));
				if (_MemError()) { TechError("TriCurMover::ReadCentroidDepths()", "_SetHandleSize()", 0); goto done; }
				//(*depthsH)[index+numDepths-1] = depth; 
				(*depthsH)[index+numDepths] = depth; 
			}
			numDepths++;
		}
		//numDepths--; // don't count the actual depth
		(*depthDataInfo)[i].numDepths = numDepths;
		index+=numDepths;
	}
	
	fDepthDataInfo = depthDataInfo;
	fDepthsH = depthsH;
	err = noErr;
	
	
done:
	
	if(s) {delete[] s;  s = 0;}
	if(err) 
	{
		if(depthDataInfo) {DisposeHandle((Handle)depthDataInfo); depthDataInfo = 0;}
		if(depthsH) {DisposeHandle((Handle)depthsH); depthsH = 0;}
	}
	return err;		
}

/////////////////////////////////////////////////////////////////
Boolean IsSigmaLevelsHeaderLine(char *s, long* numPts)
{
	char* token = strtok(s,TRICUR_DELIM_STR);
	*numPts = 0;
	if(!token || strncmpnocase(token,"SigmaLevels",strlen("SigmaLevels")) != 0)
	{
		return FALSE;
	}
	
	token = strtok(NULL,TRICUR_DELIM_STR);
	
	if(!token || sscanf(token,"%ld",numPts) != 1)
	{
		return FALSE;
	}
	
	return TRUE;
}

OSErr TriCurMover::ReadSigmaLevels(CHARH fileBufH,long *line,FLOATH *sigmaLevels,long numLevels,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{
	FLOATH sigmaLevelsH = 0;
	OSErr err=-1;
	char s[256];
	long i;
	double sigmaLevel;
	//long numDepths = fVar.maxNumDepths;	//for now
	
	strcpy(errmsg,""); // clear it
	
	sigmaLevelsH = (FLOATH)_NewHandle(sizeof(float)*numLevels);
	if(!sigmaLevelsH) {TechError("TriCurMover::ReadSigmaLevels()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	for(i=0;i<numLevels;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 256); 
		//if (sscanf(s,lfFix("%lf"),&sigmaLevel) < 1) err = true;
		
		char* token = strtok(s,TRICUR_DELIM_STR); 		
		//token = strtok(NULL,PTCUR_DELIM_STR); // points to a depth
		err = ScanDepth(token,&sigmaLevel);
		
		if(err)
		{
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read depth data from line %ld:%s",*line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done;
		}
		
		(*sigmaLevelsH)[i] = sigmaLevel;
	}
	
	*sigmaLevels = sigmaLevelsH;
	err = noErr;
	
	
done:
	
	if(err) 
	{
		if(sigmaLevelsH) {DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH = 0;}
	}
	return err;		
}


OSErr TriCurMover::TextRead(char *path, TMap **newMap) 
{
	char s[1024], errmsg[256], copyPath[256];
	long i, numPoints, numTopoPoints = 0, line = 0;
	CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH totalDepthH = 0;
	FLOATH sigmaLevelsH = 0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds;
	
	BaromodesParameters inputValues;
	memset(&inputValues,0,sizeof(inputValues));
	short modelType = 0;
	
	//TTriGridVel *triGrid = nil;	
	TTriGridVel3D *triGrid = nil;	// may not need this if not using PtCurMap
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs, numTris, numLevels=0;
	LONGH boundarySegs=0, waterBoundaries=0;
	Boolean haveBoundaryData = false, haveCentroidDepths = false;
	
	errmsg[0]=0;
	
	
	if (!path || !path[0]) return 0;
	
	strcpy(fVar.pathName,path);
	
	// code goes here, we need to worry about really big files
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("TriCurMover::TextRead()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// code goes here, worry about really long lines in the file
	
	// read header here
	strcpy(copyPath,path);
	SplitPathFile(copyPath,fVar.userName);
	//fVar.userName[kPtCurUserNameLen-1] = 0;
	for (i = 0 ; TRUE ; i++) {
		NthLineInTextOptimized(*f, line++, s, 1024); 
		if(s[0] != '[')
			break;
		err = this -> ReadHeaderLine(s);
		if(err)
			goto done;
	}
	
	if (fVar.gridType==TWO_D && fVar.maxNumDepths>1)
	{
		printError("Please check your input file. 2D grids cannot have multiple depth levels.");
		err = -1;
		goto done;
	}
	// option to read in exported topology or just require cut and paste into file	
	// read triangle/topology info if included in file, otherwise calculate
	
	if(IsBaromodesInputValuesHeaderLine(s,&modelType))	// Additional header info about what parameters were used in Baromodes
	{
		MySpinCursor();
		err = ReadBaromodesInputValues(f,&line,&inputValues,errmsg,modelType);
		if(err) goto done;
		inputValues.modelType = modelType;	// 
		SetInputValues(inputValues);
		MySpinCursor();
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{	// backward compatibility, allow old format
		inputValues.modelType = 0;	// 
		SetInputValues(inputValues);
	}
	
	if(IsTriCurVerticesHeaderLine(s,&numPoints))	// Points in Galt format - can probably use old format now
	{
		MySpinCursor();
		err = ReadTriCurVertices(f,&line,&pts,&totalDepthH,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		printError("Unable to read TriCur Triangle Velocity file."); 
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
		// for now not using for 3D files either
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
		{err = -1;  goto done;} // for now we require TTopology
		// code goes here, support Galt style ??
		DisplayMessage(0);
		//if(err) goto done;
		if (topo) numTopoPoints = _GetHandleSize((Handle)topo)/sizeof(**topo);
	}
	MySpinCursor(); // JLM 8/4/99
	
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//DisplayMessage("NEXTMESSAGETEMP");
		DisplayMessage("Making Dag Tree");
		MySpinCursor(); // JLM 8/4/99
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		MySpinCursor(); // JLM 8/4/99
		DisplayMessage(0);
		if (errmsg[0])	
			err = -1; // for now we require TIndexedDagTree
		// code goes here, support Galt style ??
		if(err) goto done;
	}
	
	MySpinCursor();
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsCentroidDepthsHeaderLine(s,&numTris)) // Boundary data from CATs
	{
		// check numTris matches numTopoPoints
		MySpinCursor();
		//err = ReadCentroidDepths(f,&line,&centroidDepths,numTris,errmsg);
		err = ReadCentroidDepths(f,&line,numTris,errmsg);
		if(err) {/*printError("Error reading centroid data");*/ goto done;}
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
		haveCentroidDepths = true;
	}
	else
	{
		//will have to calculate centroid depths by hand after reading topology
		haveCentroidDepths = false;
		if (fVar.gridType == TWO_D) 
		{
			err = dynamic_cast<TriCurMover *>(this)->CalculateVerticalGrid(pts,totalDepthH,topo,numTopoPoints,sigmaLevelsH,numLevels);
			if (err) goto done;
		}
		//err = -1; // for now we require Centroid Depths
		//if(err) {printError("Error reading centroid header line"); goto done;}
	}
	
	MySpinCursor();
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsSigmaLevelsHeaderLine(s,&numLevels)) // If the same everywhere don't need all the depth levels above
	{
		// maybe this is an option if centroid block is missing, pieces can be calculated
		MySpinCursor();
		err = ReadSigmaLevels(f,&line,&sigmaLevelsH,numLevels,errmsg);
		//err = ReadSigmaLevels(f,&line,numLevels,errmsg);
		if(err) {/*printError("Error reading sigma level header line");*/ goto done;}
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		if (!haveCentroidDepths && !(fVar.gridType == TWO_D))
		{
			// use sigma levels to set fDepthDataInfo and fDepthsH, will need to interpolate total centroid depths from vertex depths
			err = dynamic_cast<TriCurMover *>(this)->CalculateVerticalGrid(pts,totalDepthH,topo,numTopoPoints,sigmaLevelsH,numLevels);
			if (err) goto done;
		}
	}
	else
	{
		if (haveCentroidDepths || fVar.gridType == TWO_D)
		{
			// don't require sigma levels
		}
		else
		{
			// maybe a different option or is this required?
			err = -1; // for now we require TIndexedDagTree
			// code goes here, support Galt style ??
			if(err) {printError("Either sigma levels or vertical grid points are required"); goto done;}
		}
		//fVar.gridType = TWO_D;
		//fVar.maxNumDepths = 1;
	}
	
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	if (waterBoundaries && (this -> moverMap == model -> uMap /*|| fVar.gridType != TWO_D*/))
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
	
	
	//triGrid = new TTriGridVel;
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TriCurMover::TextRead()","new TTriGridVel" ,err);
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
	//if (fDepthsH) triGrid->SetBathymetry(fDepthsH);
	//if (fDepthsH) triGrid -> SetDepths(fDepthsH);	// used by PtCurMap to check vertical movement
	if (totalDepthH) triGrid -> SetDepths(totalDepthH);	// the grid should use the depths at vertices, not at centers
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	totalDepthH = 0; // because fGrid is now responsible for it
	
	
	// scan through the file looking for "[TIME ", then read and record the time, filePosition, and length of data
	// consider the possibility of multiple files
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	//if(!strstr(s,"[FILE]")) 
	//{	// single file
	err = ScanFileForTimes(path,&fTimeDataHdl,true);	// minus AH 07/17/2012
	
	if (err) goto done;
	//}
	/*	else
	 {	// multiple files
	 long numLinesInText = NumLinesInText(*f);
	 long numFiles = (numLinesInText - (line - 1))/3;	// 3 lines for each file - filename, starttime, endtime
	 strcpy(fVar.pathName,s+strlen("[FILE]\t"));
	 ResolvePath(fVar.pathName);
	 err = ScanFileForTimes(fVar.pathName,&fTimeDataHdl,true);
	 if (err) goto done;
	 // code goes here, maybe do something different if constant current
	 line--;
	 err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);
	 }*/
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
			strcpy(errmsg,"An error occurred in TriCurMover::TextRead");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
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
	if (sigmaLevelsH){DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH=0;}	
	
	return err;
	
	// rest of file (i.e. velocity data) is read as needed
}

/*OSErr TriCurMover::ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH)
 {
 long i,numScanned;
 DateTimeRec time;
 Seconds timeSeconds;
 OSErr err = 0;
 char s[1024];
 
 PtCurFileInfoH inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
 if(!inputFilesHdl) {TechError("TriCurMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
 for (i=0;i<numFiles;i++)
 {
 NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 
 strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE]\t"));
 // allow for a path relative to the GNOME directory
 ResolvePath((*inputFilesHdl)[i].pathName);
 
 NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 
 
 numScanned=sscanf(s+strlen("[STARTTIME]"), "%hd %hd %hd %hd %hd",
 &time.day, &time.month, &time.year,
 &time.hour, &time.minute) ;
 if (numScanned!= 5)
 { err = -1; TechError("TriCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
 // not allowing constant current in separate file
 //if (time.year < 1900)					// two digit date, so fix it
 //{
 //if (time.year >= 40 && time.year <= 99)	
 //time.year += 1900;
 //else
 //time.year += 2000;					// correct for year 2000 (00 to 40)
 //}
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
 { err = -1; TechError("TriCurMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
 //if (time.year < 1900)					// two digit date, so fix it
 //{
 //if (time.year >= 40 && time.year <= 99)	
 //time.year += 1900;
 //else
 //time.year += 2000;					// correct for year 2000 (00 to 40)
 //}
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
 */
OSErr TriCurMover::ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataH,Boolean setStartTime)
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
	if(!timeDataHdl) {TechError("TriCurMover::ScanFileForTimes()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	// think in terms of 100K blocks, allocate 101K, read 101K, scan 100K
	
#define kTriCurFileBufferSize  100000 // code goes here, increase to 100K or more
#define kTriCurFileBufferExtraCharSize  256
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) goto done;
	
	offset = 0;
	lengthRemainingToScan = fileLength - 5;
	
	// loop until whole file is read 
	
	h = (CHARH)_NewHandle(2* kTriCurFileBufferSize+1);
	if(!h){TechError("TriCurMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	_HLock((Handle)h);
	sectionOfFile = *h;
	
	while (lengthRemainingToScan>0)
	{
		if(lengthRemainingToScan > 2* kTriCurFileBufferSize)
		{
			lengthToRead = kTriCurFileBufferSize + kTriCurFileBufferExtraCharSize; 
			lengthOfPartToScan = kTriCurFileBufferSize; 		
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
				{ err = -1; TechError("TriCurMover::TextRead()", "sscanf() == 5", 0); goto done; }
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
				if (_MemError()) { TechError("TriCurMover::TextRead()", "_SetHandleSize()", 0); goto done; }
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
void TriCurMover::SetInputValues(BaromodesParameters inputValues) 
{
	strcpy(fInputValues.curFilePathName,inputValues.curFilePathName);
	strcpy(fInputValues.sshFilePathName,inputValues.sshFilePathName);
	strcpy(fInputValues.uldFilePathName,inputValues.uldFilePathName);
	strcpy(fInputValues.lldFilePathName,inputValues.lldFilePathName);
	strcpy(fInputValues.pycFilePathName,inputValues.pycFilePathName);
	
	fInputValues.modelType = inputValues.modelType;
	
	fInputValues.scaleVel = inputValues.scaleVel;
	fInputValues.bottomBLThickness = inputValues.bottomBLThickness;
	fInputValues.upperEddyViscosity = inputValues.upperEddyViscosity;
	fInputValues.lowerEddyViscosity = inputValues.lowerEddyViscosity;
	fInputValues.upperLevelDensity = inputValues.upperLevelDensity;
	fInputValues.lowerLevelDensity = inputValues.lowerLevelDensity;
	
}



/**************************************************************************************************/
/*OSErr TriCurMover::ReadTopology(char* path, TMap **newMap)
 {
 // import TriCur triangle info so don't have to regenerate
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
 
 TTriGridVel *triGrid = nil;
 tree.treeHdl = 0;
 TDagTree *dagTree = 0;
 
 long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
 LONGH boundarySegs=0, waterBoundaries=0;
 
 errmsg[0]=0;
 
 if (!path || !path[0]) return 0;
 
 if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
 TechError("TriCurMover::ReadTopology()", "ReadFileContents()", err);
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
 if(f) 
 {
 _HUnlock((Handle)f); 
 DisposeHandle((Handle)f); 
 f = 0;
 }
 
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
 }
 
 OSErr TriCurMover::ExportTopology(char* path)
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
 boundaryTypeH = ((PtCurMap*)moverMap)->GetWaterBoundaries();
 boundarySegmentsH = ((PtCurMap*)moverMap)->GetBoundarySegs();
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
 sprintf(topoStr,"%lf\t%lf\n",x,y);
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
 }*/

