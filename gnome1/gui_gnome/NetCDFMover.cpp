#include "Earl.h"
#include "TypeDefs.h"
#include "Cross.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "NetCDFMover.h"
#include "netcdf.h"
#include "DagTreeIO.h"
#include "netcdf.h"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT NETCDFMOVER
#endif
#endif

static PopInfoRec NetCDFMoverPopTable[] = {
	{ M33, nil, M33TIMEZONEPOPUP, 0, pTIMEZONES, 0, 1, FALSE, nil }
};

static NetCDFMover *sNetCDFDialogMover;
static Boolean sDialogUncertaintyChanged;


NetCDFMover::NetCDFMover (TMap *owner, char *name) : TCurrentMover(owner, name)
{
	memset(&fVar,0,sizeof(fVar));
	fVar.arrowScale = 1.;
	fVar.arrowDepth = 0;
	if (gNoaaVersion)
	{
		fVar.alongCurUncertainty = .5;
		fVar.crossCurUncertainty = .25;
		fVar.durationInHrs = 24.0;
	}
	else
	{
		fVar.alongCurUncertainty = 0.;
		fVar.crossCurUncertainty = 0.;
		fVar.durationInHrs = 0.;
	}
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	fVar.gridType = TWO_D; // 2D default
	fVar.maxNumDepths = 1;	// 2D default - may always be constant for netCDF files
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);
	//
	fGrid = 0;
	fTimeHdl = 0;
	fDepthLevelsHdl = 0;	// depth level, sigma, or sc_r
	fDepthLevelsHdl2 = 0;	// Cs_r
	hc = 1.;	// what default?
	
	bShowDepthContours = false;
	bShowDepthContourLabels = false;
	
	fTimeShift = 0;	// assume file is in local time
	fIsOptimizedForStep = false;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;
	
	fFillValue = -1e+34;
	fIsNavy = false;	
	
	fFileScaleFactor = 1.;	// let user set a scale factor in addition to what is in the file
	
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
	
	fNumDepthLevels = 1;	// default surface current only
	
	fAllowExtrapolationOfCurrentsInTime = false;
	fAllowVerticalExtrapolationOfCurrents = false;
	fMaxDepthForExtrapolation = 0.;	// assume 2D is just surface
	
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
}

void NetCDFMover::Dispose ()
{
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}

	if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	if(fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
	if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}

	TCurrentMover::Dispose ();
}


void ShowNetCDFMoverDialogItems(DialogPtr dialog)
{
	Boolean bShowGMTItems = true;
	short timeZone = GetPopSelection(dialog, M33TIMEZONEPOPUP);
	if (timeZone == 1) bShowGMTItems = false;
	
	ShowHideDialogItem(dialog, M33TIMESHIFTLABEL, bShowGMTItems); 
	ShowHideDialogItem(dialog, M33TIMESHIFT, bShowGMTItems); 
	ShowHideDialogItem(dialog, M33GMTOFFSETS, bShowGMTItems); 
}

void ShowHideVerticalExtrapolationDialogItems(DialogPtr dialog)
{
	Boolean extrapolateVertically, okToExtrapolate = false, showVelAtBottom = GetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX);
	TMap *map = sNetCDFDialogMover -> GetMoverMap();
	
	if (map && map->IAm(TYPE_PTCURMAP))
	{
		if ((dynamic_cast<PtCurMap *>(map))->GetMaxDepth2() > 0 || sNetCDFDialogMover -> GetMaxDepth() > 0) okToExtrapolate = true;
	} 
	
	if (sNetCDFDialogMover->fVar.gridType!=TWO_D || !okToExtrapolate)	// if model has depth data assume that is what user wants to use
	{
		extrapolateVertically = false;
		ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, extrapolateVertically); 
		ShowHideDialogItem(dialog, M33EXTRAPOLATETOLABEL, extrapolateVertically); 
		ShowHideDialogItem(dialog, M33EXTRAPOLATETOVALUE, extrapolateVertically); 
		ShowHideDialogItem(dialog, M33EXTRAPOLATETOUNITSLABEL, extrapolateVertically); 
	}
	else
	{
		extrapolateVertically = GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX);
		ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, true); 
		ShowHideDialogItem(dialog, M33EXTRAPOLATETOLABEL, extrapolateVertically); 
		ShowHideDialogItem(dialog, M33EXTRAPOLATETOVALUE, extrapolateVertically); 
		ShowHideDialogItem(dialog, M33EXTRAPOLATETOUNITSLABEL, extrapolateVertically); 
	}
	ShowHideDialogItem(dialog, M33ARROWDEPTHAT, (sNetCDFDialogMover->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate))); 
	ShowHideDialogItem(dialog, M33ARROWDEPTH, (sNetCDFDialogMover->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate)) && !showVelAtBottom); 
	ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, (sNetCDFDialogMover->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate)) && !showVelAtBottom); 
	ShowHideDialogItem(dialog, M33VELOCITYATBOTTOMCHECKBOX, (sNetCDFDialogMover->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate))); 
}

short NetCDFMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	long menuID_menuItem;
	switch (itemNum) {
		case M33OK:
		{
			char errmsg[256];
			short timeZone = GetPopSelection(dialog, M33TIMEZONEPOPUP);
			Seconds timeShift = sNetCDFDialogMover->fTimeShift;
			float arrowDepth = EditText2Float(dialog, M33ARROWDEPTH), maxDepth=0;
			float maxDepthForExtrapolation = EditText2Float(dialog, M33EXTRAPOLATETOVALUE);
			double tempAlong, tempCross, tempDuration, tempStart;
			long timeShiftInHrs;
			Boolean extrapolateVertically = GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX);
			Boolean showBottomVel = GetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX);
			TMap *map = sNetCDFDialogMover -> GetMoverMap();
			
			if (showBottomVel) arrowDepth = -2;	//check maxDepth>0
			if (map)
			{
				maxDepth = map -> GetMaxDepth2();	// 2D vs 3D ?
				//arrowDepth = EditText2Float(dialog, M33ARROWDEPTH);
				if (arrowDepth > maxDepth)
				{
					char errStr[64];
					sprintf(errStr,"The maximum depth of the region is %g meters.",maxDepth);
					printError(errStr);
					break;
				}
			}
			else
			{	// only need this if doing something 3D
				maxDepth = sNetCDFDialogMover -> GetMaxDepth();
				if (arrowDepth > maxDepth)
				{
					char errStr[64];
					sprintf(errStr,"The maximum depth of the region is %g meters.",maxDepth);
					printError(errStr);
					break;
				}
			}
			
			strcpy(errmsg,"");
			tempAlong = EditText2Float(dialog, M33ALONG);
			tempCross = EditText2Float(dialog, M33CROSS);
			tempDuration = EditText2Float(dialog, M33DURATION);
			tempStart = EditText2Float(dialog, M33STARTTIME);
			if(tempAlong <= 0 || tempCross <= 0) strcpy(errmsg,"The uncertainty must be greater than zero.");	
			else if(tempAlong > 100 || tempCross > 100)	strcpy(errmsg,"The uncertainty cannot exceed 100%.");
			
			if(errmsg[0])
			{
				printError(errmsg);
				if (tempAlong <= 0 || (tempAlong > 100 && tempCross > 0)) MySelectDialogItemText(dialog, M33ALONG,0,100);
				else MySelectDialogItemText(dialog, M33CROSS,0,100);
				break;
			}
			
			if(tempDuration < 1) strcpy(errmsg,"The uncertainty duration must be at least 1 hour.");	// maximum?
			if(errmsg[0])
			{
				printError(errmsg);
				MySelectDialogItemText(dialog, M33DURATION,0,100);
				break;
			}
			
			timeShiftInHrs = EditText2Long(dialog, M33TIMESHIFT);
			if (timeShiftInHrs < -12 || timeShiftInHrs > 14)	// what should limits be?
			{
				printError("Time offsets must be in the range -12 : 14");
				MySelectDialogItemText(dialog, M33TIMESHIFT,0,100);
				break;
			}
			
			if (extrapolateVertically)
			{
				if (maxDepthForExtrapolation==0)
				{
					if (maxDepth>0) 
						sprintf(errmsg,"Either set a max depth for extrapolation or turn off option. The maximum depth of the region is %g meters.",maxDepth);
					else
						sprintf(errmsg,"Either set a max depth for extrapolation or turn off option");
					printNote(errmsg);
					break;
				}
				if (maxDepth>0 && maxDepthForExtrapolation>maxDepth)
				{
					sprintf(errmsg,"The maximum depth of the region is %g meters.",maxDepth);
					printNote(errmsg);
					break;
				}
			}
			mygetitext(dialog, M33NAME, sNetCDFDialogMover->fVar.userName, kPtCurUserNameLen-1);
			sNetCDFDialogMover->bActive = GetButton(dialog, M33ACTIVE);
			sNetCDFDialogMover->fVar.bShowArrows = GetButton(dialog, M33SHOWARROWS);
			sNetCDFDialogMover->fVar.arrowScale = EditText2Float(dialog, M33ARROWSCALE);
			sNetCDFDialogMover->fVar.arrowDepth = arrowDepth;
			sNetCDFDialogMover->fVar.curScale = EditText2Float(dialog, M33SCALE);
			
			if (sNetCDFDialogMover->fVar.alongCurUncertainty != tempAlong || sNetCDFDialogMover->fVar.crossCurUncertainty != tempCross
				|| sNetCDFDialogMover->fVar.startTimeInHrs != tempStart || sNetCDFDialogMover->fVar.durationInHrs != tempDuration)
				sDialogUncertaintyChanged = true;
			sNetCDFDialogMover->fVar.alongCurUncertainty = EditText2Float(dialog, M33ALONG)/100;
			sNetCDFDialogMover->fVar.crossCurUncertainty = EditText2Float(dialog, M33CROSS)/100;
			//sNetCDFDialogMover->fVar.uncertMinimumInMPS = EditText2Float(dialog, M33MINCURRENT);
			sNetCDFDialogMover->fVar.startTimeInHrs = EditText2Float(dialog, M33STARTTIME);
			sNetCDFDialogMover->fVar.durationInHrs = EditText2Float(dialog, M33DURATION);
			
			sNetCDFDialogMover->fDownCurUncertainty = -sNetCDFDialogMover->fVar.alongCurUncertainty; 
			sNetCDFDialogMover->fUpCurUncertainty = sNetCDFDialogMover->fVar.alongCurUncertainty; 	
			sNetCDFDialogMover->fRightCurUncertainty = sNetCDFDialogMover->fVar.crossCurUncertainty;  
			sNetCDFDialogMover->fLeftCurUncertainty = -sNetCDFDialogMover->fVar.crossCurUncertainty; 
			sNetCDFDialogMover->fDuration = sNetCDFDialogMover->fVar.durationInHrs * 3600.;  
			sNetCDFDialogMover->fUncertainStartTime = (long) (sNetCDFDialogMover->fVar.startTimeInHrs * 3600.); 
			//if (timeZone>1) sNetCDFDialogMover->fTimeShift = EditText2Long(dialog, M33TIMESHIFT)*3600*-1;
			if (timeZone>1) sNetCDFDialogMover->fTimeShift =(long)( EditText2Float(dialog, M33TIMESHIFT)*3600);
			else sNetCDFDialogMover->fTimeShift = 0;	// file is in local time
			
			//if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
			// code goes here, should also check if start time has been shifted correctly already (if there is another current)
			//if (timeShift != sNetCDFDialogMover->fTimeShift || sNetCDFDialogMover->GetTimeValue(0) != model->GetStartTime())
			// GetTImeValue adds in the time shift
			// What if time zone hasn't changed but start time has from loading in a different file??
			if ((timeShift != sNetCDFDialogMover->fTimeShift && sNetCDFDialogMover->GetTimeValue(0) != model->GetStartTime()))
			{
				//model->SetModelTime(model->GetModelTime() - (sNetCDFDialogMover->fTimeShift-timeShift));
				if (model->GetStartTime() + sNetCDFDialogMover->fTimeShift - timeShift == sNetCDFDialogMover->GetTimeValue(0))
					model->SetStartTime(model->GetStartTime() + (sNetCDFDialogMover->fTimeShift-timeShift));
				//model->SetStartTime(sNetCDFDialogMover->GetTimeValue(0));	// just in case a different file has been loaded in the meantime, but what if user doesn't want start time to change???
				//sNetCDFDialogMover->SetInterval(errmsg);
				model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
			}
			sNetCDFDialogMover->fAllowExtrapolationOfCurrentsInTime = GetButton(dialog, M33EXTRAPOLATECHECKBOX);
			sNetCDFDialogMover->fAllowVerticalExtrapolationOfCurrents = GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX);
			// code goes here, add box to dialog to input desired depth to extrapolate to 
			if (sNetCDFDialogMover->fAllowVerticalExtrapolationOfCurrents) sNetCDFDialogMover->fMaxDepthForExtrapolation = EditText2Float(dialog, M33EXTRAPOLATETOVALUE); 
			else sNetCDFDialogMover->fMaxDepthForExtrapolation = 0.;
			
			return M33OK;
		}
			
		case M33CANCEL: 
			return M33CANCEL;
			
		case M33ACTIVE:
		case M33SHOWARROWS:
		case M33EXTRAPOLATECHECKBOX:
			ToggleButton(dialog, itemNum);
			break;
			
		case M33VELOCITYATBOTTOMCHECKBOX:	// code goes here, showhide the M33ARROWDEPTH, M33ARROWDEPTHUNITS
			ToggleButton(dialog, itemNum);
			ShowHideVerticalExtrapolationDialogItems(dialog);
			if (!GetButton(dialog,M33VELOCITYATBOTTOMCHECKBOX))
			/*sNetCDFDialogMover->fVar.arrowDepth=0*/Float2EditText(dialog, M33ARROWDEPTH, 0, 6);
			//ShowHideDialogItem(dialog, M33ARROWDEPTH, GetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX)); 
			//ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, GetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX)); 
			break;
			
		case M33EXTRAPOLATEVERTCHECKBOX:
			ToggleButton(dialog, itemNum);
			ShowHideVerticalExtrapolationDialogItems(dialog);
			/*ShowHideDialogItem(dialog, M33ARROWDEPTHAT, sNetCDFDialogMover->fVar.gridType!=TWO_D || GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX)); 
			 ShowHideDialogItem(dialog, M33ARROWDEPTH, sNetCDFDialogMover->fVar.gridType!=TWO_D || GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX)); 
			 ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, sNetCDFDialogMover->fVar.gridType!=TWO_D || GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX)); 
			 ShowHideDialogItem(dialog, M33EXTRAPOLATETOLABEL, GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX)); 
			 ShowHideDialogItem(dialog, M33EXTRAPOLATETOVALUE, GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX)); 
			 ShowHideDialogItem(dialog, M33EXTRAPOLATETOUNITSLABEL, GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX)); */
			break;
			
		case M33ARROWSCALE:
		case M33ARROWDEPTH:
			//case M33SCALE:
		case M33ALONG:
		case M33CROSS:
			//case M33MINCURRENT:
		case M33STARTTIME:
		case M33DURATION:
			//case M33TIMESHIFT:
		case M33EXTRAPOLATETOVALUE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
		case M33SCALE:
		case M33TIMESHIFT:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);	// decide whether to allow half hours
			break;
			
		case M33TIMEZONEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowNetCDFMoverDialogItems(dialog);
			if (GetPopSelection(dialog, M33TIMEZONEPOPUP) == 2) MySelectDialogItemText(dialog, M33TIMESHIFT, 0, 100);
			break;
			
		case M33GMTOFFSETS:
		{
			char gmtStr[512], gmtStr2[512];
			strcpy(gmtStr,"Standard time offsets from GMT\n\nHawaiian Standard -10 hrs\nAlaska Standard -9 hrs\n");
			strcat(gmtStr,"Pacific Standard -8 hrs\nMountain Standard -7 hrs\nCentral Standard -6 hrs\nEastern Standard -5 hrs\nAtlantic Standard -4 hrs\n");
			strcpy(gmtStr2,"Daylight time offsets from GMT\n\nHawaii always in standard time\nAlaska Daylight -8 hrs\n");
			strcat(gmtStr2,"Pacific Daylight -7 hrs\nMountain Daylight -6 hrs\nCentral Daylight -5 hrs\nEastern Daylight -4 hrs\nAtlantic Daylight -3 hrs");
			//printNote(gmtStr);
			short buttonSelected  = MULTICHOICEALERT2(1691,gmtStr,gmtStr2,TRUE);
			switch(buttonSelected){
				case 1:// ok
					//return -1;// 
					break;  
				case 3: // cancel
					//return -1;// 
					break;
			}
			break;
		}
			
		case M33REPLACEMOVER:
		{
			sNetCDFDialogMover->ReplaceMover();
			mysetitext(dialog, M33NAME, sNetCDFDialogMover->fVar.userName); // use short file name for now		
		}
	}
	
	return 0;
}

OSErr NetCDFMover::ReplaceMover()	// code goes here, maybe not for NetCDF?
{
	char 		path[256], nameStr [256], fileNamesPath[256];
	short 		item, gridType;
	OSErr		err = noErr;
	Point 		where = CenteredDialogUpLeft(M38b);
	PtCurMap 	*map = nil;
	OSType 	typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply 	reply;
	Boolean isNetCDFPathsFile = false;
	
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
	
	IsNetCDFPathsFile(path, &isNetCDFPathsFile, fileNamesPath, &gridType);
	
	if (isNetCDFPathsFile)
	{
		char errmsg[256], fileName[64], s[256];
		err = this->ReadInputFileNames(fileNamesPath);
		if (err) return err;
		this->DisposeAllLoadedData();
		// need to reset the current file name since old one may still be accessible
		strcpy(fVar.pathName,(*fInputFilesHdl)[0].pathName);
		strcpy(s,(*fInputFilesHdl)[0].pathName);
		SplitPathFile (s, fileName);
		strcpy(fVar.userName, fileName);	// maybe use a name from the file
		err = ScanFileForTimes((*fInputFilesHdl)[0].pathName,&fTimeHdl,false);	// AH 07/17/2012
		
		err = this->SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
		
		if(err) return err;
	}
	/*	else if (IsNetCDFFile (path, &gridType))
	 {	// here want to keep map and put this current on it - trust user that it matches?
	 TMap *newMap = 0;
	 char s[256],fileName[256];
	 strcpy(s,path);
	 SplitPathFile (s, fileName);
	 strcat (nameStr, fileName);
	 TCurrentMover *newMover = CreateAndInitCurrentsMover (model->uMap,false,path,fileName,&newMap);	// already have path
	 
	 if (newMover && newMap)
	 {
	 NetCDFMover *netCDFMover = (NetCDFMover*)newMover;
	 err = netCDFMover -> SettingsDialog();
	 if(err)	
	 { 
	 newMover->Dispose(); delete newMover; newMover = 0;
	 if (newMap) {newMap->Dispose(); delete newMap; newMap = 0;} 
	 }
	 
	 if(newMover && !err)
	 {
	 Boolean timeFileChanged = false;
	 if (!newMap) 
	 {
	 err = AddMoverToMap (model->uMap, timeFileChanged, newMover);
	 }
	 else
	 {
	 err = model -> AddMap(newMap, 0);
	 if (err) 
	 {
	 newMap->Dispose(); delete newMap; newMap =0; 
	 newMover->Dispose(); delete newMover; newMover = 0;
	 return -1; 
	 }
	 err = AddMoverToMap(newMap, timeFileChanged, newMover);
	 if(err) 
	 {
	 newMap->Dispose(); delete newMap; newMap =0; 
	 newMover->Dispose(); delete newMover; newMover = 0;
	 return -1; 
	 }
	 newMover->SetMoverMap(newMap);
	 }
	 }
	 }*/
	return err;
	
}

OSErr NetCDFMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
{
	char curStr[64], blankStr[32];
	strcpy(blankStr,"");
	
	RegisterPopTable (NetCDFMoverPopTable, sizeof (NetCDFMoverPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog(M33, dialog);
	
	SetDialogItemHandle(dialog, M33HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M33UNCERTAINTYBOX, (Handle)FrameEmbossed);
	
	mysetitext(dialog, M33NAME, sNetCDFDialogMover->fVar.userName); // use short file name for now
	SetButton(dialog, M33ACTIVE, sNetCDFDialogMover->bActive);
	
	if (sNetCDFDialogMover->fTimeShift == 0) SetPopSelection (dialog, M33TIMEZONEPOPUP, 1);
	else SetPopSelection (dialog, M33TIMEZONEPOPUP, 2);
	//Long2EditText(dialog, M33TIMESHIFT, (long) (-1.*sNetCDFDialogMover->fTimeShift/3600.));
	Float2EditText(dialog, M33TIMESHIFT, (float)(sNetCDFDialogMover->fTimeShift)/3600.,1);
	
	SetButton(dialog, M33SHOWARROWS, sNetCDFDialogMover->fVar.bShowArrows);
	Float2EditText(dialog, M33ARROWSCALE, sNetCDFDialogMover->fVar.arrowScale, 6);
	Float2EditText(dialog, M33ARROWDEPTH, sNetCDFDialogMover->fVar.arrowDepth, 6);
	
	ShowHideDialogItem(dialog, M33MINCURRENTLABEL, false); 
	ShowHideDialogItem(dialog, M33MINCURRENT, false); 
	ShowHideDialogItem(dialog, M33MINCURRENTUNITS, false); 
	
	//ShowHideDialogItem(dialog, M33SCALE, false); 
	//ShowHideDialogItem(dialog, M33SCALELABEL, false); 
	
	ShowHideDialogItem(dialog, M33ARROWDEPTHAT, sNetCDFDialogMover->fNumDepthLevels > 1); 
	ShowHideDialogItem(dialog, M33ARROWDEPTH, sNetCDFDialogMover->fNumDepthLevels > 1 && sNetCDFDialogMover->fVar.arrowDepth >= 0); 
	ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, sNetCDFDialogMover->fNumDepthLevels > 1 && sNetCDFDialogMover->fVar.arrowDepth >= 0); 
	
	if (!gNoaaVersion)
		ShowHideDialogItem(dialog, M33REPLACEMOVER, false);	// code goes here, eventually allow this for outside
	
	SetButton(dialog, M33EXTRAPOLATECHECKBOX, sNetCDFDialogMover->fAllowExtrapolationOfCurrentsInTime);
	SetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX, sNetCDFDialogMover->fAllowVerticalExtrapolationOfCurrents);
	SetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX, sNetCDFDialogMover->fVar.arrowDepth < 0);
	
	ShowHideDialogItem(dialog, M33VELOCITYATBOTTOMCHECKBOX, sNetCDFDialogMover->fNumDepthLevels > 1); 
	
	Float2EditText(dialog, M33SCALE, sNetCDFDialogMover->fVar.curScale, 6);
	if(sNetCDFDialogMover->fVar.alongCurUncertainty == 0) 
	{	// assume if one is not set, none are
		mysetitext(dialog, M33ALONG, blankStr); // force user to set uncertainty
		mysetitext(dialog, M33CROSS, blankStr); // force user to set uncertainty
		mysetitext(dialog, M33STARTTIME, blankStr); // force user to set
		mysetitext(dialog, M33DURATION, blankStr); // force user to set 
	}
	else
	{
		Float2EditText(dialog, M33ALONG, sNetCDFDialogMover->fVar.alongCurUncertainty*100, 6);
		Float2EditText(dialog, M33CROSS, sNetCDFDialogMover->fVar.crossCurUncertainty*100, 6);
		Float2EditText(dialog, M33STARTTIME, sNetCDFDialogMover->fVar.startTimeInHrs, 6);
		Float2EditText(dialog, M33DURATION, sNetCDFDialogMover->fVar.durationInHrs, 6);
	}
	//if(sNetCDFDialogMover->fVar.crossCurUncertainty == 0) mysetitext(dialog, M33CROSS, blankStr); // force user to set uncertainty
	/*else
	 Float2EditText(dialog, M33CROSS, sNetCDFDialogMover->fVar.crossCurUncertainty*100, 6);
	 //Float2EditText(dialog, M33MINCURRENT, sNetCDFDialogMover->fVar.uncertMinimumInMPS, 6);	// this is not implemented, hide?
	 if(sNetCDFDialogMover->fVar.startTimeInHrs < 0) mysetitext(dialog, M33STARTTIME, blankStr); // force user to set 
	 else
	 Float2EditText(dialog, M33STARTTIME, sNetCDFDialogMover->fVar.startTimeInHrs, 6);
	 if(sNetCDFDialogMover->fVar.durationInHrs == 0) mysetitext(dialog, M33DURATION, blankStr); // force user to set 
	 else
	 Float2EditText(dialog, M33DURATION, sNetCDFDialogMover->fVar.durationInHrs, 6);*/
	
	Float2EditText(dialog, M33EXTRAPOLATETOVALUE, sNetCDFDialogMover->fMaxDepthForExtrapolation, 6);
	
	ShowHideVerticalExtrapolationDialogItems(dialog);
	ShowNetCDFMoverDialogItems(dialog);
	if (sNetCDFDialogMover->fTimeShift == 0) MySelectDialogItemText(dialog, M33ALONG, 0, 100);
	else MySelectDialogItemText(dialog, M33TIMESHIFT, 0, 100);
	
	//ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, sNetCDFDialogMover->fVar.gridType==TWO_D); 	// if current already has 3D info don't confuse with extrapolation option
	if (!gNoaaVersion) ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, false); 	// for now hide until decide how to handle 3D currents...
	
	return 0;
}



OSErr NetCDFMover::SettingsDialog()
{
	short item;
	
	sNetCDFDialogMover = dynamic_cast<NetCDFMover *>(this); // should pass in what is needed only
	sDialogUncertaintyChanged = false;
	item = MyModalDialog(M33, mapWindow, 0, NetCDFMoverSettingsInit, NetCDFMoverSettingsClick);
	sNetCDFDialogMover = 0;
	
	if(M33OK == item)	
	{
		if (sDialogUncertaintyChanged) dynamic_cast<NetCDFMover *>(this)->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
		model->NewDirtNotification();// tell model about dirt
	}
	return M33OK == item ? 0 : -1;
}

OSErr NetCDFMover::InitMover()
{	
	OSErr	err = noErr;
	err = TCurrentMover::InitMover ();
	return err;
}


//#define NetCDFMoverREADWRITEVERSION 1 //JLM
//#define NetCDFMoverREADWRITEVERSION 2 //JLM
//#define NetCDFMoverREADWRITEVERSION 3 //JLM
//#define NetCDFMoverREADWRITEVERSION 4 //JLM
//#define NetCDFMoverREADWRITEVERSION 5 //JLM
#define NetCDFMoverREADWRITEVERSION 6 //added file scale factor to allow user scale factor to be separate

OSErr NetCDFMover::Write (BFPB *bfpb)
{
	long i, version = NetCDFMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numTimes = GetNumTimesInFile(), numPoints = 0, numPts = 0, numFiles = 0;
	long 	numDepths = dynamic_cast<NetCDFMover *>(this)->GetNumDepths();
	Seconds time;
	float val, depthLevel;
	DepthDataInfo depthData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TCurrentMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	if (err = fGrid -> Write (bfpb)) goto done;
	
	if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	if (err = WriteMacValue(bfpb, fVar.pathName, kMaxNameLen)) goto done;
	if (err = WriteMacValue(bfpb, fVar.userName, kPtCurUserNameLen)) return err;
	if (err = WriteMacValue(bfpb, fVar.alongCurUncertainty)) return err;
	if (err = WriteMacValue(bfpb, fVar.crossCurUncertainty)) return err;
	if (err = WriteMacValue(bfpb, fVar.uncertMinimumInMPS)) return err;
	if (err = WriteMacValue(bfpb, fVar.curScale)) return err;
	if (err = WriteMacValue(bfpb, fVar.startTimeInHrs)) return err;
	if (err = WriteMacValue(bfpb, fVar.durationInHrs)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.maxNumDepths)) return err;
	if (err = WriteMacValue(bfpb, fVar.gridType)) return err;
	//
	if (err = WriteMacValue(bfpb, fVar.bShowGrid)) return err;
	if (err = WriteMacValue(bfpb, fVar.bShowArrows)) return err;
	if (err = WriteMacValue(bfpb, fVar.bUncertaintyPointOpen)) return err;
	if (err = WriteMacValue(bfpb, fVar.arrowScale)) return err;
	if (err = WriteMacValue(bfpb, fVar.arrowDepth)) return err;
	//
	if (err = WriteMacValue(bfpb, fFillValue)) return err;
	if (err = WriteMacValue(bfpb, fIsNavy)) return err;
	//
	if (err = WriteMacValue(bfpb, numDepths)) goto done;
	for (i=0;i<numDepths;i++)
	{
		val = INDEXH(fDepthsH,i);
		if (err = WriteMacValue(bfpb, val)) goto done;
	}
	
	if (err = WriteMacValue(bfpb, numTimes)) goto done;
	for (i=0;i<numTimes;i++)
	{
		time = INDEXH(fTimeHdl,i);
		if (err = WriteMacValue(bfpb, time)) goto done;
	}
	
	if (fDepthDataInfo) numPoints = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i = 0 ; i < numPoints ; i++) {
		depthData = INDEXH(fDepthDataInfo,i);
		if (err = WriteMacValue(bfpb, depthData.totalDepth)) goto done;
		if (err = WriteMacValue(bfpb, depthData.indexToDepthData)) goto done;
		if (err = WriteMacValue(bfpb, depthData.numDepths)) goto done;
	}
	/*if (version > 1)*/ {if (err = WriteMacValue(bfpb, fTimeShift)) goto done;}
	if (err = WriteMacValue(bfpb, fAllowExtrapolationOfCurrentsInTime)) goto done;
	if (err = WriteMacValue(bfpb, fAllowVerticalExtrapolationOfCurrents)) goto done;
	if (err = WriteMacValue(bfpb, fMaxDepthForExtrapolation)) goto done;
	//fNumDepthLevels,fDepthLevelsHdl
	
	if (fDepthLevelsHdl) numPts = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	if (err = WriteMacValue(bfpb, numPts)) goto done;
	for (i = 0 ; i < numPts ; i++) {
		depthLevel = INDEXH(fDepthLevelsHdl,i);
		if (err = WriteMacValue(bfpb, depthLevel)) goto done;
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
	
	if (fDepthLevelsHdl2) 
	{
		numPts = _GetHandleSize((Handle)fDepthLevelsHdl2)/sizeof(**fDepthLevelsHdl2);
		if (err = WriteMacValue(bfpb, numPts)) goto done;
		for (i = 0 ; i < numPts ; i++) {
			depthLevel = INDEXH(fDepthLevelsHdl2,i);
			if (err = WriteMacValue(bfpb, depthLevel)) goto done;
		}
	}
	else
	{
		numPts = 0;
		if (err = WriteMacValue(bfpb, numPts)) goto done;
	}
	if (err = WriteMacValue(bfpb, hc)) goto done;
	if (err = WriteMacValue(bfpb, fFileScaleFactor)) goto done;
	
done:
	if(err)
		TechError("NetCDFMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr NetCDFMover::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[256], newFileName[64];
	long i, version, numDepths, numTimes, numPoints, numFiles;
	ClassID id;
	float val, depthLevel;
	Seconds time;
	DepthDataInfo depthData;
	PtCurFileInfo fileInfo;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFMover::Read()", "id != TYPE_NETCDFMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	//if (version != NetCDFMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	if (version > NetCDFMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the NetCDF mover (should always be rectgrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in NetCDFMover::Read()."); return -1;
	}
	
	if (err = fGrid -> Read (bfpb)) goto done;
	
	if (err = ReadMacValue(bfpb, &fNumRows)) goto done;	
	if (err = ReadMacValue(bfpb, &fNumCols)) goto done;	
	if (err = ReadMacValue(bfpb, fVar.pathName, kMaxNameLen)) goto done;
	ResolvePath(fVar.pathName); // JLM 6/3/10
	if (!FileExists(0,0,fVar.pathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],/*fileName[64],*/*p;
		strcpy(fileName,"");
		strcpy(newPath,fVar.pathName);
		p = strrchr(newPath,DIRDELIMITER);
		if (p) 
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{/*err=-1;*/ /*sprintf(msg,"The file path %s is no longer valid.",fVar.pathName); printNote(msg); strcpy(fileName,"");*/ bPathIsValid = false;/*goto done;*/}
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
		sprintf(msg,"This save file references a netCDF file which cannot be found.  Please find the file \"%s\".",fVar.pathName);printNote(msg);
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		//if (!reply.good) return USERCANCEL;	// just keep going...
		if (reply.good)
		{
			strcpy(newPath, reply.fullPath);
			strcpy (s, newPath);
			SplitPathFile (s, newFileName);
			strcpy (fVar.pathName, newPath);
			strcpy (fVar.userName, newFileName);
		}
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
			SplitPathFile (s, newFileName);
			strcpy (fVar.pathName, newPath);
			strcpy (fVar.userName, newFileName);
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
	if (version>3)
	{
		if (err = ReadMacValue(bfpb, &fVar.maxNumDepths)) return err;
		if (err = ReadMacValue(bfpb, &fVar.gridType)) return err;
	}
	//
	if (err = ReadMacValue(bfpb, &fVar.bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bShowArrows)) return err;
	if (err = ReadMacValue(bfpb, &fVar.bUncertaintyPointOpen)) return err;
	if (err = ReadMacValue(bfpb, &fVar.arrowScale)) return err;
	if (version>3) {if (err = ReadMacValue(bfpb, &fVar.arrowDepth)) return err;}
	//
	if (err = ReadMacValue(bfpb, &fFillValue)) return err;
	if (err = ReadMacValue(bfpb, &fIsNavy)) return err;
	//
	if (version>3)
	{
		if (err = ReadMacValue(bfpb, &numDepths)) goto done;	
		if (numDepths>0)
		{
			fDepthsH = (FLOATH)_NewHandleClear(sizeof(float)*numDepths);
			if (!fDepthsH)
			{ TechError("NetCDFMover::Read()", "_NewHandleClear()", 0); goto done; }
			
			for (i = 0 ; i < numDepths ; i++) {
				if (err = ReadMacValue(bfpb, &val)) goto done;
				INDEXH(fDepthsH, i) = val;
			}
		}
	}
	if (err = ReadMacValue(bfpb, &numTimes)) goto done;	
	fTimeHdl = (Seconds**)_NewHandleClear(sizeof(Seconds)*numTimes);
	if(!fTimeHdl)
	{TechError("NetCDFMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numTimes ; i++) {
		if (err = ReadMacValue(bfpb, &time)) goto done;
		INDEXH(fTimeHdl, i) = time;
	}
	if (version>3)
	{
		if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
		fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(DepthDataInfo)*numPoints);
		if(!fDepthDataInfo)
		{TechError("NetCDFMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i = 0 ; i < numPoints ; i++) {
			if (err = ReadMacValue(bfpb, &depthData.totalDepth)) goto done;
			if (err = ReadMacValue(bfpb, &depthData.indexToDepthData)) goto done;
			if (err = ReadMacValue(bfpb, &depthData.numDepths)) goto done;
			INDEXH(fDepthDataInfo, i) = depthData;
		}
	}
	if (version > 1) {if (err = ReadMacValue(bfpb, &fTimeShift)) goto done;}
	if (version > 2) {if (err = ReadMacValue(bfpb, &fAllowExtrapolationOfCurrentsInTime)) goto done;}
	if (version > 3 )
	{
		if (err = ReadMacValue(bfpb, &fAllowVerticalExtrapolationOfCurrents)) goto done;
		if (err = ReadMacValue(bfpb, &fMaxDepthForExtrapolation)) goto done;
	}
	//fNumDepthLevels, fDepthLevelsHdl
	if (version>3)
	{
		if (err = ReadMacValue(bfpb, &numPoints)) goto done;
		if (numPoints>0)
		{
			fNumDepthLevels = numPoints;
			fDepthLevelsHdl = (FLOATH)_NewHandleClear(numPoints * sizeof(float));
			if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
			for (i = 0 ; i < numPoints ; i++) 
			{
				if (err = ReadMacValue(bfpb, &depthLevel)) goto done;
				INDEXH(fDepthLevelsHdl, i) = depthLevel;
			}
		}
	}
	
	if (version>3)
	{
		if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
		if (numFiles > 0)
		{
			fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
			if(!fInputFilesHdl)
			{TechError("NetCDFMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
			for (i = 0 ; i < numFiles ; i++) {
				if (err = ReadMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
				ResolvePath(fileInfo.pathName); // JLM 6/3/10
				// code goes here, check the path (or get an error returned...) and ask user to find it, but not every time...
				if (!fileInfo.pathName[0] || !FileExists(0,0,fileInfo.pathName)) 
					bPathIsValid = false;	// if any one can not be found try to re-load the file list
				else bPathIsValid = true;
				if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
				if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
				INDEXH(fInputFilesHdl,i) = fileInfo;
			}
			if (err = ReadMacValue(bfpb, &fOverLap)) return err;
			if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
		//}
		// otherwise ask the user, trusting that user actually chooses the same data file (should insist name is the same?)
		if(!bPathIsValid)
		{
			if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
			Point where;
			OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
			MySFReply reply;
			where = CenteredDialogUpLeft(M38c);
			char newPath[kMaxNameLen], s[kMaxNameLen];
			//sprintf(msg,"This save file references a wind file list which cannot be found.  Please find the file \"%s\".",fPathName);printNote(msg);
			sprintf(msg,"This save file references a current file list which cannot be found.  Please find the file.");printNote(msg);
#if TARGET_API_MAC_CARBON
			mysfpgetfile(&where, "", -1, typeList,
						 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
			if (!reply.good) return USERCANCEL;
			strcpy(newPath, reply.fullPath);
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
			}
#endif
			err = ReadInputFileNames(newPath);
		}
		}
	}
	
	if (version>4)
	{	// need to increase version number
		if (err = ReadMacValue(bfpb, &numPoints)) goto done;
		if (numPoints>0)
		{
			//fNumDepthLevels = numPoints;	// this should be the same as above
			fDepthLevelsHdl2 = (FLOATH)_NewHandleClear(numPoints * sizeof(float));
			if (!fDepthLevelsHdl2) {err = memFullErr; goto done;}
			for (i = 0 ; i < numPoints ; i++) 
			{
				if (err = ReadMacValue(bfpb, &depthLevel)) goto done;
				INDEXH(fDepthLevelsHdl2, i) = depthLevel;
			}
		}
		if (err = ReadMacValue(bfpb, &hc)) goto done;
	}
	if (version>5)
		if (err = ReadMacValue(bfpb, &fFileScaleFactor)) goto done;
		
done:
	if(err)
	{
		TechError("NetCDFMover::Read(char* path)", " ", 0); 
		if(fDepthsH) {DisposeHandle((Handle)fDepthsH); fDepthsH=0;}
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		if(fDepthDataInfo) {DisposeHandle((Handle)fDepthDataInfo); fDepthDataInfo=0;}
		if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFMover::CheckAndPassOnMessage(TModelMessage *message)
{
	OSErr err = 0;
	//return TCurrentMover::CheckAndPassOnMessage(message); 
	/*fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);*/
	// code goes here, either check for messages about alongCur and crossCur here or reset in case fUp,fDown... were changed
	err = TCurrentMover::CheckAndPassOnMessage(message); 
	fVar.alongCurUncertainty = fUpCurUncertainty; 	
	fVar.crossCurUncertainty = fRightCurUncertainty;  
	fVar.durationInHrs = (double)fDuration/3600.; //24 hrs as seconds 
	fVar.startTimeInHrs = (double)fUncertainStartTime/3600.;
	
	return err;
}

/////////////////////////////////////////////////
long NetCDFMover::GetListLength()
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
}

ListItem NetCDFMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64], dateStr[64];
	long numTimesInFile = GetNumTimesInFile();
	ListItem item = { dynamic_cast<NetCDFMover *>(this), 0, indent, 0 };
	
	if (n == 0) {
		item.index = I_NETCDFNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Currents: \"%s\"", fVar.userName);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	if (bOpen) {
		
		
		if (--n == 0) {
			item.index = I_NETCDFACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_NETCDFGRID;
			item.bullet = fVar.bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			sprintf(text, "Show Grid");
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_NETCDFARROWS;
			item.bullet = fVar.bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,fVar.arrowScale,6);
			//sprintf(text, "Show Velocities (@ 1 in = %s m/s) ", valStr);
			if (fVar.gridType==TWO_D)
				sprintf(text, "Show Velocities (@ 1 in = %s m/s)", valStr);
			else
			{
				if (fVar.arrowDepth>=0)
					sprintf(text, "Show Velocities (@ 1 in = %s m/s) at %g m", valStr, fVar.arrowDepth);
				else
					sprintf(text, "Show Velocities (@ 1 in = %s m/s) at bottom", valStr);
			}
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_NETCDFSCALE;
			StringWithoutTrailingZeros(valStr,fVar.curScale,6);
			sprintf(text, "Multiplicative Scalar: %s", valStr);
			//item.indent++;
			return item;
		}
		
		// release time
		if (GetNumFiles()>1)
		{
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fInputFilesHdl)[0].startTime + fTimeShift;
				Secs2DateString2 (time, dateStr);
				/*if(numTimesInFile>0)*/ sprintf (text, "Start Time: %s", dateStr);
				//else sprintf (text, "Time: %s", dateStr);
				return item;
			}
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fInputFilesHdl)[GetNumFiles()-1].endTime + fTimeShift;				
				Secs2DateString2 (time, dateStr);
				sprintf (text, "End Time: %s", dateStr);
				return item;
			}
		}
		else
		{
			if (numTimesInFile>0)
			{
				if (--n == 0) {
					//item.indent++;
					Seconds time = (*fTimeHdl)[0] + fTimeShift;				
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
					Seconds time = (*fTimeHdl)[numTimesInFile-1] + fTimeShift;				
					Secs2DateString2 (time, dateStr);
					sprintf (text, "End Time: %s", dateStr);
					return item;
				}
			}
		}
		
		
		if(model->IsUncertain())
		{
			if (--n == 0) {
				item.index = I_NETCDFUNCERTAINTY;
				item.bullet = fVar.bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				item.indent++;
				return item;
			}
			
			if (fVar.bUncertaintyPointOpen) {
				
				if (--n == 0) {
					item.index = I_NETCDFALONGCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.alongCurUncertainty*100,6);
					sprintf(text, "Along Current: %s %%",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_NETCDFCROSSCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.crossCurUncertainty*100,6);
					sprintf(text, "Cross Current: %s %%",valStr);
					return item;
				}
				
				/*if (--n == 0) {
				 item.index = I_NETCDFMINCURRENT;
				 item.indent++;
				 StringWithoutTrailingZeros(valStr,fVar.uncertMinimumInMPS,6);
				 sprintf(text, "Current Minimum: %s m/s",valStr);
				 return item;
				 }*/
				
				if (--n == 0) {
					item.index = I_NETCDFSTARTTIME;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fVar.startTimeInHrs,6);
					sprintf(text, "Start Time: %s hours",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_NETCDFDURATION;
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

Boolean NetCDFMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_NETCDFNAME: bOpen = !bOpen; return TRUE;
			case I_NETCDFGRID: fVar.bShowGrid = !fVar.bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_NETCDFARROWS: fVar.bShowArrows = !fVar.bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_NETCDFUNCERTAINTY: fVar.bUncertaintyPointOpen = !fVar.bUncertaintyPointOpen; return TRUE;
			case I_NETCDFACTIVE:
				bActive = !bActive;
				model->NewDirtNotification(); 
				return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_NETCDFNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
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

Boolean NetCDFMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_NETCDFNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (bIAmPartOfACompoundMover)
						return TCurrentMover::FunctionEnabled(item, buttonID);
					
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

OSErr NetCDFMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

/*OSErr NetCDFMover::AddItem(ListItem item)
 {
 if (item.index == I_NETCDFNAME)
 return TMover::AddItem(item);
 
 return 0;
 }*/

OSErr NetCDFMover::DeleteItem(ListItem item)
{
	if (item.index == I_NETCDFNAME)
		return moverMap -> DropMover(dynamic_cast<NetCDFMover *>(this));
	
	return 0;
}

Boolean NetCDFMover::DrawingDependsOnTime(void)
{
	Boolean depends = fVar.bShowArrows;
	// if this is a constant current, we can say "no"
	//if(this->GetNumTimesInFile()==1) depends = false;
	if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	return depends;
}

void NetCDFMover::DrawContourScale(Rect r, WorldRect view)
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
	
	// code goes here, need separate cases for each grid type - have depth data on points, not triangles...
	long timeDataInterval;
	Boolean loaded;
	
	return;	// will need to select grid cells or points rather than triangles here
	
	err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
	
	if(err) return;
	
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
	
	// code goes here, probably need different code for each grid type - how to select a grid box?, allow to select triangles on curvilinear grid? different for regular grid	
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
			float inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
			float inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
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

void NetCDFMover::Draw(Rect r, WorldRect view) 
{	// Use this for regular grid or regridded data
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newCATSgridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of NetCDFMover
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	Boolean showSubsurfaceVel = false;
	OSErr err = 0;
	long depthIndex1,depthIndex2;	// default to -1?
	Rect currentMapDrawingRect = MapDrawingRect();
	WorldRect cmdr;
	long startRow,endRow,startCol,endCol,dx,dy;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	
	
	if (!fVar.bShowArrows && !fVar.bShowGrid) return;
	
	bounds = rectGrid->GetBounds();
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / fNumCols) / 2;
	dLat = (WRectHeight(bounds) / fNumRows) / 2;
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&fColor);
	
	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);
	
	if ((fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= fVar.arrowDepth) || (!fAllowVerticalExtrapolationOfCurrents && fVar.arrowDepth > 0)) showSubsurfaceVel = true;
	
	if (fVar.bShowArrows)
	{
		err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
		
		if(err && !fVar.bShowGrid) return;	// want to show grid even if there's no current data
		
		loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
		
		if(!loaded && !fVar.bShowGrid) return;
		
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime + fTimeShift;
			else
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
			//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
			if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
			{
				timeAlpha = 1;
			}
			else
			{	//return false;
				endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
		}
	}	
	
	GetDepthIndices(0,fVar.arrowDepth,&depthIndex1,&depthIndex2);
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return;	// no value for this point at chosen depth
		//continue;	// no value for this point at chosen depth

	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}
	// only draw the vectors and grid points that are in the current view
	cmdr = ScreenToWorldRect(currentMapDrawingRect, MapDrawingRect(), settings.currentView);	// have a look at this to see how to recognize out of view points
	dx = (boundsRect.hiLong - boundsRect.loLong) / (fNumCols - 1);
	dy = (boundsRect.hiLat - boundsRect.loLat) / (fNumRows - 1);
	if (boundsRect.loLong < cmdr.loLong) startCol = (cmdr.loLong - boundsRect.loLong) / dx; else startCol = 0;
	if (boundsRect.hiLong > cmdr.hiLong) endCol = fNumCols - (boundsRect.hiLong - cmdr.hiLong) / dx; else endCol = fNumCols;
	if (boundsRect.loLat < cmdr.loLat) endRow =  fNumRows - (cmdr.loLat - boundsRect.loLat) / dy; else endRow = fNumRows;
	if (boundsRect.hiLat > cmdr.hiLat) startRow = (boundsRect.hiLat - cmdr.hiLat) / dy; else startRow = 0;
	
	for (row = startRow ; row < endRow ; row++)
	{
		for (col = startCol ; col < endCol ; col++) {
			
			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newCATSgridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = dynamic_cast<NetCDFMover *>(this)->GetVelocityIndex(wp);	// need alternative for curvilinear
				
				if (fVar.bShowArrows && index >= 0)
				{
					// Check for constant current 
					if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
					{
						if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
						{
							velocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
							velocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
						}
						else
						{
							velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
							velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
						}
					}
					else // time varying current
					{
						if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
						{
							velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
							velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
						}
						else	// below surface velocity
						{
							velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
							velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
							velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
							velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
						}
						
					}
				}
			}
			
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (fVar.bShowGrid && fVar.bShowArrows && (velocity.u != 0 || velocity.v != 0)) 
				PaintRect(&c);	// should check fill_value
			if (fVar.bShowGrid && !fVar.bShowArrows) 
				PaintRect(&c);	// should check fill_value
			
			if (fVar.bShowArrows && (velocity.u != 0 || velocity.v != 0) && (fVar.arrowDepth==0 || showSubsurfaceVel))
			{
				inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
				inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
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

/////////////////////////////////////////////////////////////////

OSErr MonthStrToNum(char *monthStr, short *monthNum)
{	// not used
	OSErr err = 0;
	if (!strcmpnocase(monthStr,"JAN")) *monthNum = 1;
	else if (!strcmpnocase(monthStr,"FEB")) *monthNum = 2;
	else if (!strcmpnocase(monthStr,"MAR")) *monthNum = 3;
	else if (!strcmpnocase(monthStr,"APR")) *monthNum = 4;
	else if (!strcmpnocase(monthStr,"MAY")) *monthNum = 5;
	else if (!strcmpnocase(monthStr,"JUN")) *monthNum = 6;
	else if (!strcmpnocase(monthStr,"JUL")) *monthNum = 7;
	else if (!strcmpnocase(monthStr,"AUG")) *monthNum = 8;
	else if (!strcmpnocase(monthStr,"SEP")) *monthNum = 9;
	else if (!strcmpnocase(monthStr,"OCT")) *monthNum = 10;
	else if (!strcmpnocase(monthStr,"NOV")) *monthNum = 11;
	else if (!strcmpnocase(monthStr,"DEC")) *monthNum = 12;
	else err = -1;
	
	return err;
}


OSErr NetCDFMover::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// this code is for regular grids
	// Will need to check for Navy and split code - redefined variables so Navy and non-Navy match
	// For regridded data files don't have the real latitude/longitude values
	// Also may want to get fill_Value and scale_factor here, rather than every time velocities are read
	OSErr err = 0;
	long i,j, numScanned;
	int status, ncid, latid, lonid, depthid, recid, timeid, numdims;
	int latvarid, lonvarid, depthvarid;
	size_t latLength, lonLength, depthLength, recs, t_len, t_len2;
	double startLat,startLon,endLat,endLon,dLat,dLon,timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	WorldRect bounds;
	double *lat_vals=0,*lon_vals=0,*depthLevels=0;
	TRectGridVel *rectGrid = nil;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0;
	static size_t pt_count[3];
	Seconds startTime, startTime2;
	double timeConversion = 1., scale_factor = 1.;
	char errmsg[256] = "";
	char fileName[64],s[256],*modelTypeStr=0,outPath[256];
	Boolean bStartTimeYearZero = false;
	
	if (!path || !path[0]) return 0;
	strcpy(fVar.pathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fVar.userName, fileName);	// maybe use a name from the file
	
	status = nc_open(path, NC_NOWRITE, &ncid);

	if (status != NC_NOERR) 
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	
	status = nc_inq_dimid(ncid, "time", &recid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR || recid==-1) {err = -1; goto done;}
	}
	
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {status = nc_inq_varid(ncid, "TIME", &timeid);if (status != NC_NOERR) {err = -1; goto done;} } 	// for Ferret files, everything is in CAPS
	/////////////////////////////////////////////////

	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		err = -1; goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];
		
		timeUnits = new char[t_len+1];
		status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
						  unitStr, junk, &time.year, &time.month, &time.day,
						  &time.hour, &time.minute, &time.second) ;
		if (numScanned==5)	
			//if (numScanned<8)	
		{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned==7)	time.second = 0;
		else if (numScanned<8)	
		//else if (numScanned!=8)	
		{ err = -1; TechError("NetCDFMover::TextRead()", "sscanf() == 8", 0); goto done; }
		if (/*time.year==0 ||*/ time.year==1) {time.year+=2000; bStartTimeYearZero = true;}
		DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
		if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
			timeConversion = 3600.;
		else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
			timeConversion = 60.;
		else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
			timeConversion = 1.;
		else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
			timeConversion = 24*3600.;
	} 
	
	// probably don't need this field anymore
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; /*goto done;*/}}	// will need to split for regridded or non-Navy cases
	else 
	{
		fIsNavy = true;
		// may only need to see keyword is there, since already checked grid type
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; goto done;}}	// will need to split for regridded or non-Navy cases 
		modelTypeStr[t_len2] = '\0';
		
		strcpy(fVar.userName, modelTypeStr); // maybe use a name from the file
		//if (!strncmp (modelTypeStr, "SWAFS", 5) || !strncmp (modelTypeStr, "NCOM", 4))
		if (!strncmp (modelTypeStr, "SWAFS", 5) || strstr (modelTypeStr, "NCOM"))
			fIsNavy = true;
		else
			fIsNavy = false;
	}
	
	// changed standard format to match Navy's for regular grid
	status = nc_inq_dimid(ncid, "lat", &latid); //Navy
	if (status != NC_NOERR) 
	{	// add new check if error for LON, LAT with extensions based on subset from LAS 1/29/09
		status = nc_inq_dimid(ncid, "LAT_UV", &latid);	if (status != NC_NOERR) {err = -1; goto LAS;}	// this is for SSH files which have 2 sets of lat,lon (LAT,LON is for SSH)
	}
	status = nc_inq_varid(ncid, "lat", &latvarid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "LAT_UV", &latvarid);	if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_dimlen(ncid, latid, &latLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimid(ncid, "lon", &lonid);	//Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "LON_UV", &lonid);	
		if (status != NC_NOERR) {err = -1; goto done;}	// this is for SSH files which have 2 sets of lat,lon (LAT,LON is for SSH)
	}
	status = nc_inq_varid(ncid, "lon", &lonvarid);	//Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "LON_UV", &lonvarid);	
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_dimlen(ncid, lonid, &lonLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_dimid(ncid, "depth", &depthid);	//3D
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "levels", &depthid);	//3D
		if (status != NC_NOERR) 
		{depthLength=1;/*err = -1; goto done;*/}	// surface data only
		else
		{
			status = nc_inq_varid(ncid, "depth_levels", &depthvarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, depthid, &depthLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (depthLength>1) fVar.gridType = MULTILAYER;
		}
	}
	else
	{
		status = nc_inq_varid(ncid, "depth", &depthvarid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, depthid, &depthLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		if (depthLength>1) fVar.gridType = MULTILAYER;
	}
	
LAS:
	// check number of dimensions - 2D or 3D
	// allow more flexibility with dimension names
	if (err)
	{
		Boolean bLASStyleNames = false, bHaveDepth = false;
		char latname[NC_MAX_NAME],levname[NC_MAX_NAME],lonname[NC_MAX_NAME],dimname[NC_MAX_NAME];
		err = 0;
		status = nc_inq_ndims(ncid, &numdims);
		if (status != NC_NOERR) {err = -1; goto done;}
		for (i=0;i<numdims;i++)
		{
			if (i == recid) continue;
			status = nc_inq_dimname(ncid,i,dimname);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (strstrnocase(dimname,"LON"))
			{
				lonid = i; bLASStyleNames = true;
				strcpy(lonname,dimname);
			}
			if (strstrnocase(dimname,"LAT"))
			{
				latid = i; bLASStyleNames = true;
				strcpy(latname,dimname);
			}
			if (strstrnocase(dimname,"LEV"))
			{
				depthid = i; bHaveDepth = true;
				strcpy(levname,dimname);
			}
		}
		if (bLASStyleNames)
		{
			status = nc_inq_varid(ncid, latname, &latvarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, latid, &latLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, lonname, &lonvarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, lonid, &lonLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (bHaveDepth)
			{
				status = nc_inq_varid(ncid, levname, &depthvarid);
				if (status != NC_NOERR) {err = -1; goto done;}
				status = nc_inq_dimlen(ncid, depthid, &depthLength);
				if (status != NC_NOERR) {err = -1; goto done;}
				if (depthLength>1) fVar.gridType = MULTILAYER;
			}
			else
			{depthLength=1;/*err = -1; goto done;*/}	// surface data only
		}
		else
		{err = -1; goto done;}
		
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	pt_count[2] = depthLength;
	
	lat_vals = new double[latLength]; 
	lon_vals = new double[lonLength]; 
	if (depthLength>1) {depthLevels = new double[depthLength]; if (!depthLevels) {err = memFullErr; goto done;}}
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_double(ncid, latvarid, &ptIndex, &pt_count[0], lat_vals);
	if (status != NC_NOERR) {err=-1; goto done;}
	status = nc_get_vara_double(ncid, lonvarid, &ptIndex, &pt_count[1], lon_vals);
	if (status != NC_NOERR) {err=-1; goto done;}
	if (depthLength>1)
	{
		status = nc_get_vara_double(ncid, depthvarid, &ptIndex, &pt_count[2], depthLevels);
		if (status != NC_NOERR) {err=-1; goto done;}
		status = nc_get_att_double(ncid, depthvarid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
		
		fDepthLevelsHdl = (FLOATH)_NewHandleClear(depthLength * sizeof(float));
		if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
		for (i=0;i<depthLength;i++)
		{
			INDEXH(fDepthLevelsHdl,i) = (float)depthLevels[i] * scale_factor;
		}
	}
	
	latIndex = 0;
	lonIndex = 0;
	
	status = nc_get_var1_double(ncid, latvarid, &latIndex, &startLat);
	if (status != NC_NOERR) {err=-1; goto done;}
	status = nc_get_var1_double(ncid, lonvarid, &lonIndex, &startLon);
	if (status != NC_NOERR) {err=-1; goto done;}
	latIndex = latLength-1;
	lonIndex = lonLength-1;
	status = nc_get_var1_double(ncid, latvarid, &latIndex, &endLat);
	if (status != NC_NOERR) {err=-1; goto done;}
	status = nc_get_var1_double(ncid, lonvarid, &lonIndex, &endLon);
	if (status != NC_NOERR) {err=-1; goto done;}
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		if (bStartTimeYearZero) newTime = RoundDateSeconds(round(startTime2+(timeVal-/*730500*/730487)*timeConversion));	// this is assuming time in days since year 1
		INDEXH(fTimeHdl,i) = newTime;	// which start time where?
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion + fTimeShift;	// time zone conversion
		if (i==0) startTime = newTime;
	}
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
			if(!gCommandFileRun)	// also may want to skip for location files...
				buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
			switch(buttonSelected){
				case 1: // reset model start time
					model->SetModelTime(startTime);
					model->SetStartTime(startTime);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
					break;  
				case 3: // don't reset model start time
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
	}
	dLat = (endLat - startLat) / (latLength - 1);
	dLon = (endLon - startLon) / (lonLength - 1);
	
	bounds.loLat = ((startLat-dLat/2.))*1e6;
	bounds.hiLat = ((endLat+dLat/2.))*1e6;
	
	if (startLon>180.)	// need to decide how to handle hawaii...
	{
		bounds.loLong = (((startLon-dLon/2.)-360.))*1e6;
		bounds.hiLong = (((endLon+dLon/2.)-360.))*1e6;
	}
	else
	{	// if endLon>180 ask user if he wants to shift
		if (endLon>180.)	// if endLon>180 ask user if he wants to shift (e.g. a Hawaii ncom subset might be 170 to 220, but bna is around -180)
		{
			short buttonSelected;
			buttonSelected  = MULTICHOICEALERT(1688,"Do you want to shift the latitudes by 360?",FALSE);
			switch(buttonSelected){
				case 1: // reset model start time
					bounds.loLong = (((startLon-dLon/2.)-360.))*1e6;
					bounds.hiLong = (((endLon+dLon/2.)-360.))*1e6;
					break;  
				case 3: // don't reset model start time
					bounds.loLong = ((startLon-dLon/2.))*1e6;
					bounds.hiLong = ((endLon+dLon/2.))*1e6;
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
		else
		{
			bounds.loLong = ((startLon-dLon/2.))*1e6;
			bounds.hiLong = ((endLon+dLon/2.))*1e6;
		}
	}
	rectGrid = new TRectGridVel;
	if (!rectGrid)
	{		
		err = true;
		TechError("Error in NetCDFMover::TextRead()","new TRectGridVel" ,err);
		goto done;
	}
	
	fNumRows = latLength;
	fNumCols = lonLength;
	fNumDepthLevels = depthLength;	//  here also do we want all depths?
	fGrid = (TGridVel*)rectGrid;
	
	rectGrid -> SetBounds(bounds); 
	
	// code goes here, look for map and bathymetry information - or do in ReadTimeData?
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	
done:
	if (err)
	{
		if (!errmsg[0]) 
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);

		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		if (fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
	}

	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depthLevels) delete [] depthLevels;
	if (modelTypeStr) delete [] modelTypeStr;
	if (timeUnits) delete [] timeUnits;
	return err;
}


OSErr NetCDFMover::ReadInputFileNames(char *fileNamesPath)
{
	// for netcdf files, header file just has the paths, the start and end times will be read from the files
	long i,numScanned,line=0, numFiles, numLinesInText;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], path[256], outPath[256], classicPath[kMaxNameLen];
	CHARH fileBufH = 0;
	PtCurFileInfoH inputFilesHdl = 0;
	int status, ncid, recid, timeid;
	size_t recs, t_len, t_len2;
	double timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	static size_t timeIndex;
	Seconds startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	
	if (err = ReadFileContents(TERMINATED,0, 0, fileNamesPath, 0, 0, &fileBufH)) goto done;
	
	numLinesInText = NumLinesInText(*fileBufH);
	numFiles = numLinesInText - 1;	// subtract off the header
	inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("NetCDFMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
	NthLineInTextNonOptimized(*fileBufH, (line)++, s, 1024); 	// header line
	for (i=0;i<numFiles;i++)	// should count files as go along
	{
		NthLineInTextNonOptimized(*fileBufH, (line)++, s, 1024); 	// check it is a [FILE] line
		//strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE]\t"));
		RemoveLeadingAndTrailingWhiteSpace(s);
		strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace((*inputFilesHdl)[i].pathName);
		ResolvePathFromInputFile(fileNamesPath,(*inputFilesHdl)[i].pathName); // JLM 6/8/10
		strcpy(path,(*inputFilesHdl)[i].pathName);
		if((*inputFilesHdl)[i].pathName[0] && FileExists(0,0,(*inputFilesHdl)[i].pathName))
		{
			status = nc_open(path, NC_NOWRITE, &ncid);
			if (status != NC_NOERR) /*{err = -1; goto done;}*/
			{
#if TARGET_API_MAC_CARBON
				err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
				status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			
			status = nc_inq_dimid(ncid, "time", &recid); 
			if (status != NC_NOERR) 
			{
				status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			
			status = nc_inq_varid(ncid, "time", &timeid); 
			if (status != NC_NOERR) {err = -2; goto done;} 
			
			/////////////////////////////////////////////////
			status = nc_inq_attlen(ncid, timeid, "units", &t_len);
			if (status != NC_NOERR) 
			{
				err = -2; goto done;
			}
			else
			{
				DateTimeRec time;
				char unitStr[24], junk[10];
				
				timeUnits = new char[t_len+1];
				status = nc_get_att_text(ncid, timeid, "units", timeUnits);
				if (status != NC_NOERR) {err = -2; goto done;} 
				timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
				StringSubstitute(timeUnits, ':', ' ');
				StringSubstitute(timeUnits, '-', ' ');
				
				numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
								  unitStr, junk, &time.year, &time.month, &time.day,
								  &time.hour, &time.minute, &time.second) ;
				if (numScanned==5)	
				{time.hour = 0; time.minute = 0; time.second = 0; }
				else if (numScanned==7)	time.second = 0;
				else if (numScanned<8)	
					//if (numScanned!=8)	
				{ err = -1; TechError("NetCDFMover::ReadInputFileNames()", "sscanf() == 8", 0); goto done; }
				DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
				if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
					timeConversion = 3600.;
				else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
					timeConversion = 60.;
				else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
					timeConversion = 1.;
				else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
					timeConversion = 24*3600.;
			} 
			
			status = nc_inq_dim(ncid, recid, recname, &recs);
			if (status != NC_NOERR) {err = -2; goto done;}
			{
				Seconds newTime;
				// possible units are, HOURS, MINUTES, SECONDS,...
				timeIndex = 0;	// first time
				status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
				if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); printError(errmsg); err = -1; goto done;}
				newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
				(*inputFilesHdl)[i].startTime = newTime;
				timeIndex = recs-1;	// last time
				status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
				if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); printError(errmsg); err = -1; goto done;}
				newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
				(*inputFilesHdl)[i].endTime = newTime;
			}
			status = nc_close(ncid);
			if (status != NC_NOERR) {err = -2; goto done;}
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to NetCDF data File does not exist.%s%s",NEWLINESTRING,(*inputFilesHdl)[i].pathName);
			printError(msg);
			err = true;
			goto done;
		}
		
		
	}
	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}	// so could replace list
	fInputFilesHdl = inputFilesHdl;
	
done:
	if(fileBufH) { DisposeHandle((Handle)fileBufH); fileBufH = 0;}
	if (err)
	{
		if (err==-2) {printError("Error reading netCDF file");}
		if(inputFilesHdl) {DisposeHandle((Handle)inputFilesHdl); inputFilesHdl=0;}
	}
	return err;
}
