#include "Cross.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "NetCDFMover.h"
#include "netcdf.h"
#include "DagTreeIO.h"
//#include "v5d.h"
//#include "binio.h"


#ifdef MAC
#ifdef MPW
#pragma SEGMENT NETCDFMOVER
#endif
#endif

enum {
		I_NETCDFNAME = 0 ,
		I_NETCDFACTIVE, 
		I_NETCDFGRID, 
		I_NETCDFARROWS,
	   I_NETCDFSCALE,
		I_NETCDFUNCERTAINTY,
		I_NETCDFSTARTTIME,
		I_NETCDFDURATION, 
		I_NETCDFALONGCUR,
		I_NETCDFCROSSCUR,
		//I_NETCDFMINCURRENT
		};


/////////////////////////////////////////////////

static PopInfoRec NetCDFMoverPopTable[] = {
		{ M33, nil, M33TIMEZONEPOPUP, 0, pTIMEZONES, 0, 1, FALSE, nil }
	};

static NetCDFMover *sNetCDFDialogMover;
static Boolean sDialogUncertaintyChanged;

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
	//ShowHideDialogItem(dialog, M33ARROWDEPTHAT, (sNetCDFDialogMover->fVar.gridType!=TWO_D || extrapolateVertically) && okToExtrapolate); 
	//ShowHideDialogItem(dialog, M33ARROWDEPTH, (sNetCDFDialogMover->fVar.gridType!=TWO_D || extrapolateVertically) && okToExtrapolate); 
	//ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, (sNetCDFDialogMover->fVar.gridType!=TWO_D || extrapolateVertically) && okToExtrapolate); 
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
			if (map && map->IAm(TYPE_PTCURMAP))
			{
				maxDepth = (dynamic_cast<PtCurMap *>(map)) -> GetMaxDepth2();	// 2D vs 3D ?
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
		err = ScanFileForTimes((*fInputFilesHdl)[0].pathName,&fTimeHdl,false);
		err = this->SetInterval(errmsg);
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
	
	sNetCDFDialogMover = this; // should pass in what is needed only
	sDialogUncertaintyChanged = false;
	item = MyModalDialog(M33, mapWindow, 0, NetCDFMoverSettingsInit, NetCDFMoverSettingsClick);
	sNetCDFDialogMover = 0;

	if(M33OK == item)	
	{
		if (sDialogUncertaintyChanged) this->UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
		model->NewDirtNotification();// tell model about dirt
	}
	return M33OK == item ? 0 : -1;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////

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
	//fVar.uncertMinimumInMPS = .05;
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	//fVar.durationInHrs = 24.0;
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
	
	/*fOffset_u = 0.;
	fOffset_v = 0.;
	fCurScale_u = 1.;
	fCurScale_v = 1.;*/

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

OSErr NetCDFMover::InitMover()
{	
	OSErr	err = noErr;
	err = TCurrentMover::InitMover ();
	return err;
}

OSErr NetCDFMover::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty

	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(velocity->u*velocity->u + velocity->v * velocity->v);
		
	
		u = velocity->u;
		v = velocity->v;

		if(lengthS < fVar.uncertMinimumInMPS)
		{
			// use a diffusion  ??
			printError("nonzero UNCERTMIN is unimplemented");
			//err = -1;
		}
		else
		{	// normal case, just use cross and down stuff
			alpha = unrec.downStream;
			beta = unrec.crossStream;
		
			velocity->u = u*(1+alpha)+v*beta;
			velocity->v = v*(1+alpha)-u*beta;	
		}
	}
	else 
	{
		TechError("NetCDFMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}

OSErr NetCDFMover::PrepareForModelStep()
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;

	if (model->GetModelTime() == model->GetStartTime())	// first step
	{
		if (this->IAm(TYPE_NETCDFMOVERCURV) || this->IAm(TYPE_NETCDFMOVERTRI))
		{
			//PtCurMap* ptCurMap = (PtCurMap*)moverMap;
			//PtCurMap* ptCurMap = GetPtCurMap();
			//if (ptCurMap)
			if (moverMap->IAm(TYPE_PTCURMAP))
			{
				(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
				(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
				if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)
					((TTriGridVel3D*)fGrid)->ClearOutputHandles();
			}
		}
	}
	if (!bActive) return noErr;

	err = this -> SetInterval(errmsg); // SetInterval checks to see that the time interval is loaded
	if (err) goto done;

	fIsOptimizedForStep = true;
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	return err;
}

void NetCDFMover::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}

OSErr NetCDFMover::CheckAndScanFile(char *errmsg)
{
	Seconds time = model->GetModelTime(), startTime, endTime, lastEndTime, testTime, firstStartTime;
	long i,numFiles = GetNumFiles();
	OSErr err = 0;

	errmsg[0]=0;
	if (fEndData.timeIndex!=UNASSIGNEDINDEX)
		testTime = (*fTimeHdl)[fEndData.timeIndex];	// currently loaded end time

	firstStartTime = (*fInputFilesHdl)[0].startTime + fTimeShift;
	for (i=0;i<numFiles;i++)
	{
		startTime = (*fInputFilesHdl)[i].startTime + fTimeShift;
		endTime = (*fInputFilesHdl)[i].endTime + fTimeShift;
		if (startTime<=time&&time<=endTime && !(startTime==endTime))
		{
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);
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
				if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeHdl,false);
				DisposeLoadedData(&fEndData);
				strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeHdl,false);
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
				if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[i-1].pathName,&fTimeHdl,false);
				DisposeLoadedData(&fEndData);
				strcpy(fVar.pathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		lastEndTime = endTime;
	}
	if (fAllowExtrapolationOfCurrentsInTime && time > lastEndTime)
	{
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		err = ScanFileForTimes((*fInputFilesHdl)[numFiles-1].pathName,&fTimeHdl,false);
		// code goes here, check that start/end times match
		strcpy(fVar.pathName,(*fInputFilesHdl)[numFiles-1].pathName);
		fOverLap = false;
		return err;
	}
	if (fAllowExtrapolationOfCurrentsInTime && time < firstStartTime)
	{
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		err = ScanFileForTimes((*fInputFilesHdl)[0].pathName,&fTimeHdl,false);
		// code goes here, check that start/end times match
		strcpy(fVar.pathName,(*fInputFilesHdl)[0].pathName);
		fOverLap = false;
		return err;
	}
	strcpy(errmsg,"Time outside of interval being modeled");
	return -1;	
	//return err;
}

Boolean NetCDFMover::CheckInterval(long &timeDataInterval)
{
	Seconds time =  model->GetModelTime(), startTime, endTime;
	long i,numTimes,numFiles = GetNumFiles();

	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists

	// check for constant current
	if (numTimes==1 && !(GetNumFiles()>1)) 
	//if (numTimes==1) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
				return true;
		else
			return false;
	}
	
	if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
	{
		if (time>=((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && time<=((*fTimeHdl)[fEndData.timeIndex] + fTimeShift))
				{	// we already have the right interval loaded
					timeDataInterval = fEndData.timeIndex;
					return true;
				}
	}

	if (GetNumFiles()>1 && fOverLap)
	{	
		if (time>=fOverLapStartTime + fTimeShift && time<=(*fTimeHdl)[fEndData.timeIndex] + fTimeShift)
			return true;	// we already have the right interval loaded, time is in between two files
		else fOverLap = false;
	}
	
	//for (i=0;i<numTimes;i++) 
	for (i=0;i<numTimes-1;i++) 
	{	// find the time interval
		if (time>=((*fTimeHdl)[i] + fTimeShift) && time<=((*fTimeHdl)[i+1] + fTimeShift))
		{
			timeDataInterval = i+1; // first interval is between 0 and 1, and so on
			return false;
		}
	}	
	// don't allow time before first or after last
	if (time<((*fTimeHdl)[0] + fTimeShift)) 
	{
		// if number of files > 1 check that first is the one loaded
		timeDataInterval = 0;
		if (numFiles > 0)
		{
			//startTime = (*fInputFilesHdl)[0].startTime + fTimeShift;
			startTime = (*fInputFilesHdl)[0].startTime;
			if ((*fTimeHdl)[0] != startTime)
				return false;
		}
		if (fAllowExtrapolationOfCurrentsInTime && fEndData.timeIndex == UNASSIGNEDINDEX && !(fStartData.timeIndex == UNASSIGNEDINDEX))	// way to recognize last interval is set
		{
			//check if time > last model time in all files
			//timeDataInterval = 1;
			return true;
		}
	}
	if (time>((*fTimeHdl)[numTimes-1] + fTimeShift) )
	// code goes here, check if this is last time in all files and user has set flag to continue
	// then if last time is loaded as start time and nothing as end time this is right interval
	{
		// if number of files > 1 check that last is the one loaded
		timeDataInterval = numTimes;
		if (numFiles > 0)
		{
			//endTime = (*fInputFilesHdl)[numFiles-1].endTime + fTimeShift;
			endTime = (*fInputFilesHdl)[numFiles-1].endTime;
			if ((*fTimeHdl)[numTimes-1] != endTime)
				return false;
		}
		if (fAllowExtrapolationOfCurrentsInTime && fEndData.timeIndex == UNASSIGNEDINDEX && !(fStartData.timeIndex == UNASSIGNEDINDEX))	// way to recognize last interval is set
		{
			//check if time > last model time in all files
			return true;
		}
	}
	return false;
		
}

void NetCDFMover::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void NetCDFMover::DisposeAllLoadedData()
{
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);
}

void NetCDFMover::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long NetCDFMover::GetNumDepthLevelsInFile()
{
	long numDepthLevels = 0;

	if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	return numDepthLevels;     
}

long NetCDFMover::GetNumDepthLevels()
{
	long numDepthLevels = 0;

	if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	else
	{
		long numDepthLevels = 0;
		OSErr err = 0;
		char path[256], outPath[256];
		int status, ncid, sigmaid, sigmavarid;
		size_t sigmaLength=0;
		//if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
		//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
		strcpy(path,fVar.pathName);
		if (!path || !path[0]) return -1;
		
		status = nc_open(path, NC_NOWRITE, &ncid);
		if (status != NC_NOERR) /*{err = -1; goto done;}*/
		{
	#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
				status = nc_open(outPath, NC_NOWRITE, &ncid);
	#endif
			if (status != NC_NOERR) {err = -1; return -1;}
		}
		//if (status != NC_NOERR) {/*err = -1; goto done;*/return -1;}
		status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
		if (status != NC_NOERR) 
		{
			numDepthLevels = 1;	// check for zgrid option here
		}	
		else
		{
			status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
			if (status != NC_NOERR) {numDepthLevels = 1;}	// require variable to match the dimension
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {numDepthLevels = 1;}	// error in file
			//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
			//fVar.maxNumDepths = sigmaLength;
			numDepthLevels = sigmaLength;
			//status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
			//if (status != NC_NOERR) {err = -1; goto done;}
			// once depth is read in 
		}
	}
	return numDepthLevels;     
}

long NetCDFMover::GetNumTimesInFile()
{
	long numTimes = 0;

	if (fTimeHdl) numTimes = _GetHandleSize((Handle)fTimeHdl)/sizeof(**fTimeHdl);
	return numTimes;     
}

long NetCDFMover::GetNumFiles()
{
	long numFiles = 0;

	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

long NetCDFMover::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

OSErr NetCDFMover::SetInterval(char *errmsg)
{
	long timeDataInterval = 0;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
		
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
		
	// check for constant current 
	//if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	if(numTimesInFile==1 && !(GetNumFiles()>1))	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0 && fAllowExtrapolationOfCurrentsInTime)
	{
		indexOfStart = 0;
		indexOfEnd = -1;
	}
	/*if(timeDataInterval == 0)
	{	// before the first step in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}
	else if(timeDataInterval == numTimesInFile) 
	{	// past the last information in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}*/
	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile /*|| (timeDataInterval==1 && fAllowExtrapolationOfCurrentsInTime)*/)
	{	// before the first step in the file

		if (GetNumFiles()>1)
		{
			if ((err = CheckAndScanFile(errmsg)) || fOverLap) goto done;	// overlap is special case
			intervalLoaded = this -> CheckInterval(timeDataInterval);
			indexOfStart = timeDataInterval-1;
			indexOfEnd = timeDataInterval;
			numTimesInFile = this -> GetNumTimesInFile();
			if (fAllowExtrapolationOfCurrentsInTime && (timeDataInterval==numTimesInFile || timeDataInterval == 0))
			{
				if(intervalLoaded) 
					return 0;
				indexOfEnd = -1;
				if (timeDataInterval == 0) indexOfStart = 0;	// if we allow extrapolation we need to load the first time
			}
		}
		else
		{
			if (fAllowExtrapolationOfCurrentsInTime && timeDataInterval == numTimesInFile) 
			{
				fStartData.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else if (fAllowExtrapolationOfCurrentsInTime && timeDataInterval == 0) 
			{
				fStartData.timeIndex = 0;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else
			{
				err = -1;
				strcpy(errmsg,"Time outside of interval being modeled");
				goto done;
			}
		}
		// code goes here, if time > last time in files allow user to continue
		// leave last two times loaded? move last time to start and nothing for end?
		// redefine as constant or just check if time > last time and some flag set
		// careful with timeAlpha, really just want to use the last time but needs to be loaded
		// want to check so that don't reload at every step, should recognize last time is ok
	}
	//else // load the two intervals
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
		if(!errmsg[0])strcpy(errmsg,"Error in NetCDFMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;

}

WorldPoint3D NetCDFMover::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index; 
	long depthIndex1,depthIndex2;	// default to -1?
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	Boolean useVelSubsurface = false;	// don't seem to need this, just return 
	char errmsg[256];
	
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
							
	if ((*theLE).z>0 && fVar.gridType==TWO_D)
	{		
		if (fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= (*theLE).z) useVelSubsurface = true;
		else
		{	// may allow 3D currents later
			deltaPoint.p.pLong = 0.;
			deltaPoint.p.pLat = 0.;
			deltaPoint.z = 0;
			return deltaPoint; 
		}
	}
							
	GetDepthIndices(0,(*theLE).z,&depthIndex1,&depthIndex2);
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}

	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	//if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
	//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				//scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
				//scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
				scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else
			{
				scaledPatVelocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
				scaledPatVelocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
			}
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		 
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				//scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
				//scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
				scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else	// below surface velocity
			{
				scaledPatVelocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
				scaledPatVelocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
				scaledPatVelocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
				scaledPatVelocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
			}
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}

scale:

	scaledPatVelocity.u *= fVar.curScale; 
	scaledPatVelocity.v *= fVar.curScale; 
	//scaledPatVelocity.u *= fCurScale_u; 
	//scaledPatVelocity.v *= fCurScale_v; 
	
	//if (scaledPatVelocity.u != 0) scaledPatVelocity.u += fOffset_u; 
	//if (scaledPatVelocity.v != 0) scaledPatVelocity.v += fOffset_v; 
	
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
		
VelocityRec NetCDFMover::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("NetCDFMover::GetScaledPatValue is unimplemented");
	return v;
}

Seconds NetCDFMover::GetTimeValue(long index)
{
	if (index<0) printError("Access violation in NetCDFMover::GetTimeValue()");
	Seconds time = (*fTimeHdl)[index] + fTimeShift;
	return time;
}

VelocityRec NetCDFMover::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("NetCDFMover::GetPatValue is unimplemented");
	return v;
}

long NetCDFMover::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	double dRowNum, dColNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;

	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFMover

	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	//SetLRect (&gridLRect, 0, fNumRows-1, fNumCols-1, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	//colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	//rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	//dColNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	//dRowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;

	//dColNum = round((p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5);
	//dRowNum = round((p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5);
	dColNum = (p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5;
	dRowNum = (p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5;
	//if (dColNum<0) dColNum = -1; if (dRowNum<0) dRowNum = -1;
	//colNum = dColNum;
	//rowNum = dRowNum;
	colNum = round(dColNum);
	rowNum = round(dRowNum);

	//if (colNum < 0 || colNum >= fNumCols-1 || rowNum < 0 || rowNum >= fNumRows-1)
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)

		{ return -1; }
		
	return rowNum * fNumCols + colNum;
	//return rowNum * (fNumCols-1) + colNum;
}

LongPoint NetCDFMover::GetVelocityIndices(WorldPoint p) 
{
	long rowNum, colNum;
	double dRowNum, dColNum;
	LongPoint indices = {-1,-1};
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;

	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFMover

	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	//colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	//rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;

	dColNum = round((p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5);
	dRowNum = round((p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5);
	//if (dColNum<0) dColNum = -1; if (dRowNum<0) dRowNum = -1;
	colNum = dColNum;
	rowNum = dRowNum;

	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)

		{ return indices; }
		
	//return rowNum * fNumCols + colNum;
	indices.h = colNum;
	indices.v = rowNum;
	return indices;
}


/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize netcdf currents
double NetCDFMover::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (index>=0)
	{
		if (fStartData.dataHdl) u = INDEXH(fStartData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFMover::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		if (fEndData.dataHdl) u = INDEXH(fEndData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFMover::GetStartVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fStartData.dataHdl) v = INDEXH(fStartData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

double NetCDFMover::GetEndVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fEndData.dataHdl) v = INDEXH(fEndData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

OSErr NetCDFMover::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX)
		*startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

OSErr NetCDFMover::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX)
		*endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

double NetCDFMover::GetDepthAtIndex(long depthIndex, double totalDepth)
{	// really can combine and use GetDepthAtIndex - could move to base class
	double depth = 0;
	float sc_r, Cs_r;
	if (fVar.gridType == SIGMA_ROMS)
	{
		sc_r = INDEXH(fDepthLevelsHdl,depthIndex);
		Cs_r = INDEXH(fDepthLevelsHdl2,depthIndex);
		//depth = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
		depth = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
	}
	else
		depth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth; // times totalDepth

	return depth;
}

Boolean NetCDFMover::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	Boolean useVelSubsurface = false;
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index;
	LongPoint indices;
	long depthIndex1,depthIndex2;	// default to -1?

	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!fVar.bShowArrows && !fVar.bShowGrid) return 0;
	err = this -> SetInterval(errmsg);
	if(err) return false;

	if (fVar.arrowDepth>0 && fVar.gridType==TWO_D)
	{		
		if (fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= fVar.arrowDepth) useVelSubsurface = true;
		else
		{
			velocity.u = 0.;
			velocity.v = 0.;
			goto CalcStr;
		}
	}

	GetDepthIndices(0,fVar.arrowDepth,&depthIndex1,&depthIndex2);
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}

	if(this->GetNumTimesInFile()>1)
	//&& loaded && !err)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) /*return false;*/
		{
			if ((time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
			{
				timeAlpha = 1;
			}
			else
				return false;
		}
		else
			timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}
	//if (loaded && !err)
	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular

		indices = this->GetVelocityIndices(wp.p);

		if (index >= 0)
		{
			// Check for constant current 
			if(this->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
					//velocity.u = this->GetStartUVelocity(index);
					//velocity.v = this->GetStartVVelocity(index);
					velocity.u = this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols);
					velocity.v = this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols);
				}
				else
				{
					velocity.u = depthAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*this->GetStartUVelocity(index+depthIndex2*fNumRows*fNumCols);
					velocity.v = depthAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*this->GetStartVVelocity(index+depthIndex2*fNumRows*fNumCols);
				}
			}
			else // time varying current
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
					//velocity.u = timeAlpha*this->GetStartUVelocity(index) + (1-timeAlpha)*this->GetEndUVelocity(index);
					//velocity.v = timeAlpha*this->GetStartVVelocity(index) + (1-timeAlpha)*this->GetEndVVelocity(index);
					velocity.u = timeAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex1*fNumRows*fNumCols);
					velocity.v = timeAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex1*fNumRows*fNumCols);
				}
				else	// below surface velocity
				{
					velocity.u = depthAlpha*(timeAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex1*fNumRows*fNumCols));
					velocity.u += (1-depthAlpha)*(timeAlpha*this->GetStartUVelocity(index+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex2*fNumRows*fNumCols));
					velocity.v = depthAlpha*(timeAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex1*fNumRows*fNumCols));
					velocity.v += (1-depthAlpha)*(timeAlpha*this->GetStartVVelocity(index+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex2*fNumRows*fNumCols));
				}
			}
		}
	}

CalcStr:

	/*if (this->fOffset_u != 0 && velocity.u!=0&& velocity.v!=0) 
	{
		velocity.u = this->fCurScale_u * velocity.u + this->fOffset_u; 
		velocity.v = this->fCurScale_v * velocity.v + this->fOffset_v;
		lengthU = lengthS = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	}
	else
	{*/
		lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
		lengthS = this->fVar.curScale * lengthU;
	//}
	//if (this->fVar.offset != 0 && lengthS!=0) lengthS += this->fVar.offset;

	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
								this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
								this->className, uStr, sStr);
	}

	return true;
}

/////////////////////////////////////////////////
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


//#define NetCDFMoverREADWRITEVERSION 1 //JLM
//#define NetCDFMoverREADWRITEVERSION 2 //JLM
//#define NetCDFMoverREADWRITEVERSION 3 //JLM
//#define NetCDFMoverREADWRITEVERSION 4 //JLM
#define NetCDFMoverREADWRITEVERSION 5 //JLM

OSErr NetCDFMover::Write (BFPB *bfpb)
{
	long i, version = NetCDFMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numTimes = GetNumTimesInFile(), numPoints = 0, numPts = 0, numFiles = 0;
	long 	numDepths = GetNumDepths();
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
				if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
				if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
				INDEXH(fInputFilesHdl,i) = fileInfo;
			}
			if (err = ReadMacValue(bfpb, &fOverLap)) return err;
			if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
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
	return TCurrentMover::CheckAndPassOnMessage(message); 
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
	ListItem item = { this, 0, indent, 0 };
	
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
		return moverMap -> DropMover(this);
	
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
	err = this -> SetInterval(errmsg);
	if(err) return;
	
	loaded = this -> CheckInterval(timeDataInterval);
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
		err = this -> SetInterval(errmsg);
		if(err && !fVar.bShowGrid) return;	// want to show grid even if there's no current data
		
		loaded = this -> CheckInterval(timeDataInterval);
		if(!loaded && !fVar.bShowGrid) return;
	
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		//if(GetNumTimesInFile()>1 && loaded && !err)
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
	//if (boundsRect.loLat < cmdr.loLat) startRow = (cmdr.loLat - boundsRect.loLat) / dy; else startRow = 0;
	//if (boundsRect.hiLat > cmdr.hiLat) endRow = fNumRows - (boundsRect.hiLat - cmdr.hiLat) / dy; else endRow = fNumRows;
	if (boundsRect.loLat < cmdr.loLat) endRow =  fNumRows - (cmdr.loLat - boundsRect.loLat) / dy; else endRow = fNumRows;
	if (boundsRect.hiLat > cmdr.hiLat) startRow = (boundsRect.hiLat - cmdr.hiLat) / dy; else startRow = 0;
	//newCATSgridRect = {startRow, startCol, endRow-1, endCol-1};
	//MySetRect(&newCATSgridRect, startCol, startRow, endCol-1, endRow-1);
	//MySetRect(&newCATSgridRect, startCol, endRow-1, endCol-1, startRow);

	//for (row = 0 ; row < fNumRows ; row++)
	for (row = startRow ; row < endRow ; row++)
	{
		//for (col = 0 ; col < fNumCols ; col++) {
		for (col = startCol ; col < endCol ; col++) {

			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newCATSgridRect, boundsRect);
			//wp = ScreenToWorldPoint(p, newCATSgridRect, cmdr);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = GetVelocityIndex(wp);	// need alternative for curvilinear
	
				if (fVar.bShowArrows && index >= 0)
				{
					// Check for constant current 
					if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
					//if(GetNumTimesInFile()==1)
					{
						if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
						{
							//velocity.u = INDEXH(fStartData.dataHdl,index).u;
							//velocity.v = INDEXH(fStartData.dataHdl,index).v;
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
							//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
							//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
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
				inchesX = (velocity.u * fVar.curScale) / fVar.arrowScale;
				inchesY = (velocity.v * fVar.curScale) / fVar.arrowScale;
				//inchesX = (velocity.u * fCurScale_u + fOffset_u) / fVar.arrowScale;
				//inchesY = (velocity.v * fCurScale_v + fOffset_v) / fVar.arrowScale;
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

float NetCDFMover::GetMaxDepth()
{
	float maxDepth = 0;
	if (fDepthsH)
	{
		float depth=0;
		long i,numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
		for (i=0;i<numDepths;i++)
		{
			depth = INDEXH(fDepthsH,i);
			if (depth > maxDepth) 
				maxDepth = depth;
		}
		return maxDepth;
	}
	else
	{
		long numDepthLevels = GetNumDepthLevelsInFile();
		if (numDepthLevels<=0) return maxDepth;
		if (fDepthLevelsHdl) maxDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	}
	return maxDepth;
}

float NetCDFMover::GetTotalDepth(WorldPoint wp, long triNum)
{	// z grid only 
#pragma unused(wp)
#pragma unused(triNum)
	long numDepthLevels = GetNumDepthLevelsInFile();
	float totalDepth = 0;
	if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	return totalDepth;
}

void NetCDFMover::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData = 0;
	long numDepthLevels = GetNumDepthLevelsInFile();
	float totalDepth = 0;
	

	if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	else
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}
/*	switch(fVar.gridType) 
	{
		case TWO_D:	// no depth data
			*depthIndex1 = indexToDepthData;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
		case BAROTROPIC:	// values same throughout column, but limit on total depth
			if (depthAtPoint <= totalDepth)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			else
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		case MULTILAYER: //
			//break;
		case SIGMA: // */
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{
				long j;
				for(j=0;j<numDepthLevels-1;j++)
				{
					if(INDEXH(fDepthLevelsHdl,indexToDepthData+j)<depthAtPoint &&
						depthAtPoint<=INDEXH(fDepthLevelsHdl,indexToDepthData+j+1))
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
					}
					else if(INDEXH(fDepthLevelsHdl,indexToDepthData+j)==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
					}
				}
				if(INDEXH(fDepthLevelsHdl,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			//break;
		/*default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}*/
}

/////////////////////////////////////////////////

Seconds RoundDateSeconds(Seconds timeInSeconds)
{
	double	DaysInMonth[13] = {0.0,31.0,28.0,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,30.0,31.0};
	DateTimeRec date;
	Seconds roundedTimeInSeconds;
	// get rid of the seconds since they get garbled in the dialogs
	SecondsToDate(timeInSeconds,&date);
	if (date.second == 0) return timeInSeconds;
	if (date.second > 30) 
	{
		if (date.minute<59) date.minute++;
		else
		{
			date.minute = 0;
			if (date.hour < 23) date.hour++;
			else
			{
				if( (date.year % 4 == 0 && date.year % 100 != 0) || date.year % 400 == 0) DaysInMonth[2]=29.0;
				date.hour = 0;
				if (date.day < DaysInMonth[date.month]) date.day++;
				else
				{
					date.day = 1;
					if (date.month < 12) date.month++;
					else
					{
						date.month = 1;
						if (date.year>2019) {printError("Time outside of model range"); /*err=-1; goto done;*/}
						else date.year++;
						date.year++;
					}
				}
			}
		}
	}
	date.second = 0;
	DateToSeconds(&date,&roundedTimeInSeconds);
	return roundedTimeInSeconds;
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


#define TIMES 3
#define LATS 5
#define LONS 10
// for testing - may use in CATS
OSErr NetCDFMover::SaveAsNetCDF(char *path1,double *lats, double *lons)
{	// not used
	OSErr err = 0;	
	int status, ncid, lat_dim, lon_dim, time_dim, rh_id, rh_dimids[3], old_fill_mode;
	int i, j, k, lat_id, lon_id, time_id, reftime_id, timelen_dim, dimid[1];
	//double rh_vals[TIMES*LATS*LONS],fillVal;
	double *rh_vals=0,fillVal;
	long timeInHrs;
	double rh_range[] = {0.0, 100.0};
	char path[] = "MacintoshHD:Users:coconnor:Projects:VeryImportant:GNOME:codeFiles:foo.nc";
	//char path[] = "MacintoshHD:VeryImportant:GNOME:codeFiles:foo.nc";
	char title[] = "example netCDF dataset";
	static size_t time_index[1];
	float lat[] = {-90, -87.5, -85, -82.5, -80};
	float lon[] = {-180, -179, -178, -177, -176, -175, -174, -173, -172, -171};
	static char reftime[] = {"1992 03 04 12:00"};
	// should check if file already exists, and ask about overwriting
	// for some reason the nc_create command gives an error, have to use nccreate with no error checking
	//status = nc_create(path, NC_NOCLOBBER, &ncid);
	ncid = nccreate(path,NC_CLOBBER);
	//if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_open(path, NC_WRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	// need to open, put into define mode? - create automatically sets up for 
	//status = nc_redef(ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	rh_vals = new double[TIMES*fNumRows*fNumCols]; 
	if(!rh_vals) {TechError("NetCDFMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}

	//status = nc_def_dim(ncid, "lat", 5L, &lat_dim);
	status = nc_def_dim(ncid, "lat", fNumRows, &lat_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_def_dim(ncid, "lon", 10L, &lon_dim);
	status = nc_def_dim(ncid, "lon", fNumCols, &lon_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
   status = nc_def_dim(ncid, "timelen", 20L, &timelen_dim);
	if (status != NC_NOERR) {err = -1; goto done;}

	rh_dimids[0] = time_dim;
	rh_dimids[1] = lat_dim;
	rh_dimids[2] = lon_dim;
	
	status = nc_def_var(ncid, "rh", NC_DOUBLE, 3, rh_dimids, &rh_id);
	if (status != NC_NOERR) {err = -1; goto done;}

	//status = nc_set_fill(ncid, NC_NOFILL, &old_fill_mode);
	//if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_put_att_double (ncid, rh_id, "valid_range", NC_DOUBLE, 2, rh_range);
	if (status != NC_NOERR) {err = -1; goto done;}
   fillVal = 1e-34;
	status = nc_put_att_double (ncid, rh_id, "_FillValue", NC_DOUBLE, 1, &fillVal);
	if (status != NC_NOERR) {err = -1; goto done;}
	dimid[0] = lat_dim;
   status = nc_def_var (ncid, "lat", NC_DOUBLE, 1, dimid, &lat_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, lat_id, "long_name",/* NC_CHAR,*/ strlen("latitude"), "latitude");
	status = nc_put_att_text (ncid, lat_id, "units",/* NC_CHAR,*/ strlen("degrees_north"), "degrees_north");
	dimid[0] = lon_dim;
   status = nc_def_var (ncid, "lon", NC_DOUBLE, 1, dimid, &lon_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, lon_id, "long_name",/* NC_CHAR,*/ strlen("longitude"), "longitude");
	status = nc_put_att_text (ncid, lon_id, "units",/* NC_CHAR,*/ strlen("degrees_east"), "degrees_east");
   dimid[0] = time_dim;
   status = nc_def_var (ncid, "time", NC_LONG, 1, dimid, &time_id);
 	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, time_id, "long_name",/* NC_CHAR,*/ strlen("time"), "time");
	status = nc_put_att_text (ncid, time_id, "units",/* NC_CHAR,*/ strlen("hours"), "hours");
   dimid[0] = timelen_dim;
   status = nc_def_var (ncid, "reftime", NC_CHAR, 1, dimid, &reftime_id);
 	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, reftime_id, "long_name",/* NC_CHAR,*/ strlen("reference time"), "reference time");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, reftime_id, "units",/* NC_CHAR,*/ strlen("text_time"), "text_time");
	if (status != NC_NOERR) {err = -1; goto done;}


	status = nc_put_att_text (ncid, NC_GLOBAL, "title",/* NC_CHAR,*/ strlen(title), title);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (k=0;k<TIMES;k++)	// only have 1 or 2 times loaded, could get them from the other file
	{
		for (i=0; i<fNumRows; i++)
		{
			for (j=0; j<fNumCols; j++)
			{
				if (k==0) 
				{
					if (INDEXH(fStartData.dataHdl,(fNumRows-i-1)*fNumCols+j).u==0)	// shouldn't be changing fillVals to zero in the first place...
						rh_vals[i*fNumCols+j] = fillVal;
					else		
						rh_vals[i*fNumCols+j] = INDEXH(fStartData.dataHdl,(fNumRows-i-1)*fNumCols+j).u;
				}
				else rh_vals[i*fNumCols+j+k*fNumRows*fNumCols] = .5*(k+1);
			}
		}
	}
	status = nc_enddef(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	// store lat 
	status = nc_put_var_double(ncid, lat_id, lats);
	if (status != NC_NOERR) {err = -1; goto done;}

	// store lon 
	status = nc_put_var_double(ncid, lon_id, lons);
	if (status != NC_NOERR) {err = -1; goto done;}

	// store time 
	for (j=0;j<TIMES;j++)
	{
		time_index[0] = j;
		timeInHrs = 6*(j+2);
		nc_put_var1_long(ncid,time_id,time_index,&timeInHrs);
	}
	// store reftime 
	status = nc_put_var_text(ncid, reftime_id, reftime);
	if (status != NC_NOERR) {err = -1; goto done;}

	// store rh_vals
	status = nc_put_var_double(ncid, rh_id, rh_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

done:
	if (err)
	{
		printError("Error writing out netcdf file");
	}
	if (rh_vals) delete [] rh_vals;
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
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}

	/*status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "time", &recid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
	}*/

	status = nc_inq_dimid(ncid, "time", &recid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR || recid==-1) {err = -1; goto done;}
	}

	status = nc_inq_varid(ncid, "time", &timeid); 
	//if (status != NC_NOERR) {err = -1; goto done;} 
	if (status != NC_NOERR) {status = nc_inq_varid(ncid, "TIME", &timeid);if (status != NC_NOERR) {err = -1; goto done;} /*timeid = recid;*/} 	// for Ferret files, everything is in CAPS
/////////////////////////////////////////////////
	//status = nc_inq_attlen(ncid, recid, "units", &t_len);	// recid is the dimension id not the variable id
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		/*err = -1; goto done;*/
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];
		
		timeUnits = new char[t_len+1];
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);	// recid is the dimension id not the variable id
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
		else if (numScanned!=8)	
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
		status = nc_inq_dimid(ncid, "LON_UV", &lonid);	if (status != NC_NOERR) {err = -1; goto done;}	// this is for SSH files which have 2 sets of lat,lon (LAT,LON is for SSH)
	}
	status = nc_inq_varid(ncid, "lon", &lonvarid);	//Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "LON_UV", &lonvarid);	if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_dimlen(ncid, lonid, &lonLength);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_dimid(ncid, "depth", &depthid);	//3D
	if (status != NC_NOERR) 
	{depthLength=1;/*err = -1; goto done;*/}	// surface data only
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
	//status = nc_get_vara_double(ncid, latid, &ptIndex, &pt_count[0], lat_vals);
	status = nc_get_vara_double(ncid, latvarid, &ptIndex, &pt_count[0], lat_vals);
	if (status != NC_NOERR) {err=-1; goto done;}
	//status = nc_get_vara_double(ncid, lonid, &ptIndex, &pt_count[1], lon_vals);
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
	//status = nc_get_var1_double(ncid, latid, &latIndex, &startLat);
	status = nc_get_var1_double(ncid, latvarid, &latIndex, &startLat);
	if (status != NC_NOERR) {err=-1; goto done;}
	//status = nc_get_var1_double(ncid, lonid, &lonIndex, &startLon);
	status = nc_get_var1_double(ncid, lonvarid, &lonIndex, &startLon);
	if (status != NC_NOERR) {err=-1; goto done;}
	latIndex = latLength-1;
	lonIndex = lonLength-1;
	//status = nc_get_var1_double(ncid, latid, &latIndex, &endLat);
	status = nc_get_var1_double(ncid, latvarid, &latIndex, &endLat);
	if (status != NC_NOERR) {err=-1; goto done;}
	//status = nc_get_var1_double(ncid, lonid, &lonIndex, &endLon);
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
		//status = nc_get_var1_double(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		if (bStartTimeYearZero) newTime = RoundDateSeconds(round(startTime2+(timeVal-/*730500*/730487)*timeConversion));	// this is assuming time in days since year 1
		INDEXH(fTimeHdl,i) = newTime;	// which start time where?
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion + fTimeShift;	// time zone conversion
		if (i==0) startTime = newTime;
	}
	//{
		//Seconds modStTime = model->GetStartTime(), modTime = model->GetModelTime();
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
			//buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			//if(!gCommandFileErrorLogPath[0])
			if(!gCommandFileRun)	// also may want to skip for location files...
				buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
			switch(buttonSelected){
				case 1: // reset model start time
					//bTopFile = true;
					model->SetModelTime(startTime);
					model->SetStartTime(startTime);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
					break;  
				case 3: // don't reset model start time
					//bTopFile = false;
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
		//model->SetModelTime(startTime);
		//model->SetStartTime(startTime);
		//model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
	}
	//}
	dLat = (endLat - startLat) / (latLength - 1);
	dLon = (endLon - startLon) / (lonLength - 1);

	bounds.loLat = ((startLat-dLat/2.))*1e6;
	bounds.hiLat = ((endLat+dLat/2.))*1e6;
	
	/*if (bounds.loLat > bounds.hiLat)
	{
		bounds.loLat = ((endLat+dLat/2.))*1e6;
		bounds.hiLat = ((startLat-dLat/2.))*1e6;
	}*/
	//bounds.loLat = ((startLat))*1e6;
	//bounds.hiLat = ((endLat))*1e6;
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
			//bounds.loLong = ((startLon))*1e6;
			//bounds.hiLong = ((endLon))*1e6;
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

	//err = this -> SetInterval(errmsg);	// if going to allow user to not reset start time to file start time should ignore error here?
	//if(err) goto done;

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
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		if (fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
	}
	//SaveAsNetCDF("junk.dat",lat_vals,lon_vals);	// just for testing
	//SaveAsVis5d("junk.dat",lat_vals,lon_vals);	// just for testing - moved to separate MyVis5D.cpp
	//SaveAsVis5d(endLat,startLon,dLat,dLon);	// just for testing
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depthLevels) delete [] depthLevels;
	if (modelTypeStr) delete [] modelTypeStr;
	if (timeUnits) delete [] timeUnits;
	return err;
}
	 	 
/////////////////////////////////////////////////

OSErr NetCDFMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	// note: no need to split based on fIsNavy (since Navy variables are used for regular format) 
	// only an issue in curvilinear case where we have server/PMEL variables and Navy variables
	// removed the fIsNavy stuff 9/22/03 (Navy stores u,v as shorts and scales later)
	OSErr err = 0;
	long i,j,k;
	char path[256], outPath[256]; 
	char *velUnits=0; 
	int status, ncid, numdims, uv_ndims, numvars;
	int curr_ucmp_id, curr_vcmp_id, depthid;
	static size_t curr_index[] = {0,0,0,0};
	static size_t curr_count[4];
	size_t velunit_len;
	//float *curr_uvals=0,*curr_vvals=0, fill_value;
	double *curr_uvals=0,*curr_vvals=0, fill_value, velConversion=1.;
	long totalNumberOfVels = fNumRows * fNumCols;
	VelocityFH velH = 0;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	//long depthlength = fNumDepthLevels;	// code goes here, do we want all depths? maybe if movermap is a ptcur map??
	long depthlength = 1;	// code goes here, do we want all depths?
	//float scale_factor = 1.;
	double scale_factor = 1./*, scale_factor_v = 1.*/;
	//double add_offset = 0., add_offset_v = 0.;
	Boolean bDepthIncluded = false;
	
	errmsg[0]=0;

	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;

	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}

	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}

	if (numdims>=4)
	{	// code goes here, do we really want to use all the depths - for big files Gnome can't handle it
		status = nc_inq_dimid(ncid, "depth", &depthid);	//3D
		if (status != NC_NOERR) 
		{
			//bDepthIncluded = false;
			status = nc_inq_dimid(ncid, "sigma", &depthid);	//3D - need to check sigma values in TextRead...
			if (status != NC_NOERR) bDepthIncluded = false;
			else bDepthIncluded = true;
		}
		else bDepthIncluded = true;
		// code goes here, might want to check other dimensions (lev), or just how many dimensions uv depend on
		//status = nc_inq_dimid(ncid, "sigma", &depthid);	//3D
		//if (status != NC_NOERR) bDepthIncluded = false;
		//else bDepthIncluded = true;
	}

	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	//if (numdims>=4)	// should check what the dimensions are
	if (bDepthIncluded)
	{
		if (moverMap->IAm(TYPE_PTCURMAP)) depthlength = fNumDepthLevels;
		//curr_count[1] = 1;	// depth
		curr_count[1] = depthlength;	// depth
		curr_count[2] = latlength;
		curr_count[3] = lonlength;
	}
	else
	{
		curr_count[1] = latlength;	
		curr_count[2] = lonlength;
	}

	//curr_uvals = new double[latlength*lonlength]; 
	curr_uvals = new double[latlength*lonlength*depthlength]; 
	//curr_uvals = new float[latlength*lonlength]; 
	if(!curr_uvals) {TechError("NetCDFMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	//curr_vvals = new float[latlength*lonlength]; 
	//curr_vvals = new double[latlength*lonlength]; 
	curr_vvals = new double[latlength*lonlength*depthlength]; 
	if(!curr_vvals) {TechError("NetCDFMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}

	status = nc_inq_varid(ncid, "water_u", &curr_ucmp_id);
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "curr_ucmp", &curr_ucmp_id); 
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id); // allow u,v since so many people get confused
			if (status != NC_NOERR) {status = nc_inq_varid(ncid, "U", &curr_ucmp_id); if (status != NC_NOERR)	// ferret uses CAPS
			{err = -1; goto LAS;}}	// broader check for variable names coming out of LAS
		}	
	}
	status = nc_inq_varid(ncid, "water_v", &curr_vcmp_id);	// what if only input one at a time (u,v separate movers)?
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "curr_vcmp", &curr_vcmp_id); 
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "v", &curr_vcmp_id); // allow u,v since so many people get confused
			if (status != NC_NOERR) {status = nc_inq_varid(ncid, "V", &curr_vcmp_id); if (status != NC_NOERR)	// ferret uses CAPS
				{err = -1; goto done;}}
		}	
	}
	
LAS:
	if (err)
	{
		Boolean bLASStyleNames = false;
		char uname[NC_MAX_NAME],vname[NC_MAX_NAME],levname[NC_MAX_NAME],varname[NC_MAX_NAME];
		err = 0;
		status = nc_inq_nvars(ncid, &numvars);
		if (status != NC_NOERR) {err = -1; goto done;}
		for (i=0;i<numvars;i++)
		{
			//if (i == recid) continue;
			status = nc_inq_varname(ncid,i,varname);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (varname[0]=='U' || varname[0]=='u' || strstrnocase(varname,"EVEL"))	// careful here, could end up with wrong u variable (like u_wind for example)
			{
				curr_ucmp_id = i; bLASStyleNames = true;
				strcpy(uname,varname);
			}
			if (varname[0]=='V' || varname[0]=='v' || strstrnocase(varname,"NVEL"))
			{
				curr_vcmp_id = i; bLASStyleNames = true;
				strcpy(vname,varname);
			}
			if (strstrnocase(varname,"LEV"))
			{
				depthid = i; bDepthIncluded = true;
				strcpy(levname,varname);
				curr_count[1] = depthlength;	// depth (set to 1)
				curr_count[2] = latlength;
				curr_count[3] = lonlength;
			}
		}
		if (!bLASStyleNames){err = -1; goto done;}
	}


	status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
	if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {curr_count[1] = latlength; curr_count[2] = lonlength;}}	// could have more dimensions than are used in u,v
	if (uv_ndims==4) {curr_count[1] = depthlength;curr_count[2] = latlength;curr_count[3] = lonlength;}
	//status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	status = nc_get_vara_double(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	status = nc_get_vara_double(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}


	status = nc_inq_attlen(ncid, curr_ucmp_id, "units", &velunit_len);
	if (status == NC_NOERR)
	{
		velUnits = new char[velunit_len+1];
		status = nc_get_att_text(ncid, curr_ucmp_id, "units", velUnits);
		if (status == NC_NOERR)
		{
			velUnits[velunit_len] = '\0'; 
			if (!strcmpnocase(velUnits,"cm/s") ||!strcmpnocase(velUnits,"Centimeters per second") )
				velConversion = .01;
			else if (!strcmpnocase(velUnits,"m/s"))
				velConversion = 1.0;
		}
	}


	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);	// should get this in text_read and store, but will have to go short to float and back
	status = nc_get_att_double(ncid, curr_ucmp_id, "_FillValue", &fill_value);	// should get this in text_read and store, but will have to go short to float and back
	if (status != NC_NOERR) 
	{status = nc_get_att_double(ncid, curr_ucmp_id, "FillValue", &fill_value); 
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "missing_value", &fill_value); /*if (status != NC_NOERR) {err = -1; goto done;}*/ }}	// require fill value (took this out 12.12.08)

#ifdef MAC
	//if (fill_value==NAN)	// Miami SSH server uses NaN for fill value ?? Windows doesn't like it
	if (isnan(fill_value))	// Miami SSH server uses NaN for fill value ?? Windows doesn't like it
		fill_value=-99999.;
#else
	if (_isnan(fill_value))
		fill_value=-99999;
#endif

	//status = nc_get_att_float(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
	status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
	if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor

	//status = nc_get_att_double(ncid, curr_vcmp_id, "scale_factor", &scale_factor_v);
	//if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor

	//status = nc_get_att_double(ncid, curr_ucmp_id, "add_offset", &add_offset);
	//if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require offset

	//status = nc_get_att_double(ncid, curr_vcmp_id, "add_offset", &add_offset_v);
	//if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require offset

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	//velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec) * depthlength);
	if (!velH) {err = memFullErr; goto done;}
	for (k=0;k<depthlength;k++)
	{
	for (i=0;i<latlength;i++)
	{
		for (j=0;j<lonlength;j++)
		{
			//if (curr_uvals[(latlength-i-1)*lonlength+j]==fill_value)	// should store in current array and check before drawing or moving
				//curr_uvals[(latlength-i-1)*lonlength+j]=0.;
			//if (curr_vvals[(latlength-i-1)*lonlength+j]==fill_value)
				//curr_vvals[(latlength-i-1)*lonlength+j]=0.;
			//INDEXH(velH,i*lonlength+j).u = (float)curr_uvals[(latlength-i-1)*lonlength+j];
			//INDEXH(velH,i*lonlength+j).v = (float)curr_vvals[(latlength-i-1)*lonlength+j];
			if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)	// should store in current array and check before drawing or moving
				curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
			if (curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
				curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
#ifdef MAC
			if (isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
			//if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==NAN)	// should store in current array and check before drawing or moving
				curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
			if (isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
			//if (curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==NAN)
				curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
#else
			if (_isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))	// should store in current array and check before drawing or moving
				curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
			if (_isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
				curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
#endif
			INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = (float)curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
			INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = (float)curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
		}
	}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	if (scale_factor!=1.) fVar.curScale = scale_factor;
	//if (scale_factor!=1.) {fVar.curScale = scale_factor; fCurScale_u = scale_factor;  fCurScale_v = scale_factor_v;}
	//else fCurScale_v = fCurScale_u = fVar.curScale;
	//if (add_offset!=0.) {fOffset_u = add_offset; fOffset_v = add_offset_v;}
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading current data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		//printNote("Error opening NetCDF file");
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) delete [] curr_uvals;
	if (curr_vvals) delete [] curr_vvals;
	if (velUnits) {delete [] velUnits;}
	return err;
}



//OSErr NetCDFMover::ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH)
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
			//
			status = nc_open(path, NC_NOWRITE, &ncid);
			if (status != NC_NOERR) /*{err = -1; goto done;}*/
			{
		#if TARGET_API_MAC_CARBON
				err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
					status = nc_open(outPath, NC_NOWRITE, &ncid);
		#endif
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			//if (status != NC_NOERR) {err = -2; goto done;}
		
			//status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being dimension name
			status = nc_inq_dimid(ncid, "time", &recid); 
			if (status != NC_NOERR) 
			{
				//status = nc_inq_dimid(ncid, "time", &recid); 
				status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
				if (status != NC_NOERR) {err = -2; goto done;}
			}
		
			status = nc_inq_varid(ncid, "time", &timeid); 
			if (status != NC_NOERR) {err = -2; goto done;} 
		
			/////////////////////////////////////////////////
			status = nc_inq_attlen(ncid, timeid, "units", &t_len);
			if (status != NC_NOERR) 
			{
				timeUnits = 0;	// files should always have this info
				timeConversion = 3600.;		// default is hours
				startTime2 = model->GetStartTime();	// default to model start time
				/*err = -2; goto done;*/
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
				else if (numScanned!=8)	
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

OSErr NetCDFMover::ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime)
{
	OSErr err = 0;
	long i,numScanned,line=0;
	DateTimeRec time;
	Seconds timeSeconds;
	char s[1024], outPath[256];
	CHARH fileBufH = 0;
	int status, ncid, recid, timeid;
	size_t recs, t_len, t_len2;
	double timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	static size_t timeIndex;
	Seconds startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	Seconds **timeHdl = 0;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	//if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_dimid(ncid, "time", &recid); 
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
		if (status != NC_NOERR) {err = -1; goto done;}
	}

	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {err = -1; goto done;} 

	/////////////////////////////////////////////////
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		/*err = -1; goto done;*/
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
		else if (numScanned!=8)	
		//if (numScanned!=8)	
			{ err = -1; TechError("NetCDFMover::ScanFileForTimes()", "sscanf() == 8", 0); goto done; }
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
	timeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!timeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -2; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		INDEXH(timeHdl,i) = newTime;	// which start time where?
	}
	*timeH = timeHdl;
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -2; goto done;}


done:
	if (err)
	{
		if (err==-2) {printError("Error reading times from NetCDF file");}
		if (timeHdl) {DisposeHandle((Handle)timeHdl); timeHdl=0;}
	}
	return err;
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
// Curvilinear grid code - separate mover
// read in grid values for first time and set up transformation (dagtree?)
// need to read in lat/lon since won't be evenly spaced

NetCDFMoverCurv::NetCDFMoverCurv (TMap *owner, char *name) : NetCDFMover(owner, name)
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
}

Boolean NetCDFMoverCurv::IAmA3DMover()
{
	if (fVar.gridType != TWO_D) return true;
	return false;
}
LongPointHdl NetCDFMoverCurv::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

long NetCDFMoverCurv::GetVelocityIndex(WorldPoint wp)
{
	long index = -1;
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	return index;
}

LongPoint NetCDFMoverCurv::GetVelocityIndices(WorldPoint wp)
{
	LongPoint indices={-1,-1};
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	return indices;
}

Boolean NetCDFMoverCurv::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{	// code goes here, this is triangle code, not curvilinear
	char uStr[32],vStr[32],sStr[32],depthStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;

	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth, totalDepth = 0.;
	long index;
	LongPoint indices;

	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	long depthIndex1,depthIndex2;	// default to -1?

	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!fVar.bShowArrows && !fVar.bShowGrid) return 0;
	err = this -> SetInterval(errmsg);
	if(err) return false;

	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (index < 0) return 0;
		indices = this->GetVelocityIndices(wp.p);
	}
	totalDepth = GetTotalDepth(wp.p,index);
	GetDepthIndices(index,fVar.arrowDepth,totalDepth,&depthIndex1,&depthIndex2);
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return false;	// no value for this point at chosen depth - should show as 0,0 or nothing?
		
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		/*if (fDepthsH)
		{
			totalDepth = INDEXH(fDepthsH,index);
		}
		else 
		{
			totalDepth = 0;
		}*/
		//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
		//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
		topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
		bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth); // times totalDepth
		//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
		//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
		if (totalDepth == 0) depthAlpha = 1;
		else
			depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}

	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//velocity.u = INDEXH(fStartData.dataHdl,index).u;
			//velocity.v = INDEXH(fStartData.dataHdl,index).v;
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
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		 
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else
			{
				velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
				velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
				velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
				velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
			}
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}

	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fWindScale * lengthU;
	lengthS = this->fVar.curScale * lengthU;

	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		//sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
							//this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
		StringWithoutTrailingZeros(uStr,velocity.u,4);
		StringWithoutTrailingZeros(vStr,velocity.v,4);
		StringWithoutTrailingZeros(depthStr,depthIndex1,4);
		sprintf(diagnosticStr, " [grid: %s, u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld]",
							this->className, uStr, vStr, fNumRows-indices.v-1, indices.h);
		//if (depthIndex1>0 || !(depthIndex2==UNASSIGNEDINDEX))
		if (fVar.gridType!=TWO_D)
		sprintf(diagnosticStr, " [grid: %s, u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld, %ld]",
							this->className, uStr, vStr, fNumRows-indices.v-1, indices.h, depthIndex1);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
							this->className, uStr, sStr);
	}

	return true;
}

WorldPoint3D NetCDFMoverCurv::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depthAlpha, depth = (*theLE).z;
	float topDepth, bottomDepth;
	long index = -1, depthIndex1, depthIndex2; 
	float totalDepth; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	// might want to check for fFillValue and set velocity to zero - shouldn't be an issue unless we interpolate
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	
	totalDepth = GetTotalDepth(refPoint,index);
	if (index>=0)
		GetDepthIndices(index,depth,totalDepth,&depthIndex1,&depthIndex2);	// if not ?? point is off grid but not beached (map mismatch)
	else 
		return deltaPoint;
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return deltaPoint;	// no value for this point at chosen depth - should this be an error? question of show currents below surface vs an actual LE moving
		
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		/*if (fDepthsH)
		{
			totalDepth = INDEXH(fDepthsH,index);
		}
		else 
		{
			totalDepth = 0;
		}*/
		//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
		//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
		topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
		bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth);
		//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
		//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
		if (totalDepth == 0) depthAlpha = 1;
		else
			depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
	}

	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			//scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else
			{
				scaledPatVelocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
				scaledPatVelocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
			}
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		 
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else	// below surface velocity
			{
				scaledPatVelocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
				scaledPatVelocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
				scaledPatVelocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
				scaledPatVelocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
			}
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}

scale:

	scaledPatVelocity.u *= fVar.curScale; // may want to allow some sort of scale factor, though should be in file
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
		
void NetCDFMoverCurv::Dispose ()
{
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}

	NetCDFMover::Dispose ();
}


#define NetCDFMoverCurvREADWRITEVERSION 1 //JLM

OSErr NetCDFMoverCurv::Write (BFPB *bfpb)
{
	long i, version = NetCDFMoverCurvREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints = 0, numPts = 0, index;
	WorldPointF vertex;
	OSErr err = 0;

	if (err = NetCDFMover::Write (bfpb)) return err;

	StartReadWriteSequence("NetCDFMoverCurv::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////

	if (fVerdatToNetCDFH) numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}

	if (fVertexPtsH) numPts = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPts)) goto done;
	for (i=0;i<numPts;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}

done:
	if(err)
		TechError("NetCDFMoverCurv::Write(char* path)", " ", 0); 

	return err;
}

OSErr NetCDFMoverCurv::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMoverCurv::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFMoverCurv::Read()", "id != TYPE_NETCDFMOVERCURV", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != NetCDFMoverCurvREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
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
	
done:
	if(err)
	{
		TechError("NetCDFMoverCurv::Read(char* path)", " ", 0); 
		if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
		if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFMoverCurv::CheckAndPassOnMessage(TModelMessage *message)
{
	return NetCDFMover::CheckAndPassOnMessage(message); 
}

float NetCDFMoverCurv::GetTotalDepthFromTriIndex(long triNum)
{
	long index1, index2, index3, index4, numDepths;
	OSErr err = 0;
	float totalDepth = 0;
	Boolean useTriNum = true;
	WorldPoint refPoint = {0.,0.};

	if (fVar.gridType == SIGMA_ROMS)	// should always be true
	{
		//if (triNum < 0) useTriNum = false;
		err = ((TTriGridVel*)fGrid)->GetRectCornersFromTriIndexOrPoint(&index1, &index2, &index3, &index4, refPoint, triNum, useTriNum, fVerdatToNetCDFH, fNumCols+1);

		if (err) return 0;
		if (fDepthsH)
		{	// issue with extended grid not having depths - probably need to rework that idea
			long numCorners = 4;
			numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
			if (index1<numDepths && index1>=0) totalDepth += INDEXH(fDepthsH,index1); else numCorners--;
			if (index2<numDepths && index2>=0) totalDepth += INDEXH(fDepthsH,index2); else numCorners--;
			if (index3<numDepths && index3>=0) totalDepth += INDEXH(fDepthsH,index3); else numCorners--;
			if (index4<numDepths && index4>=0) totalDepth += INDEXH(fDepthsH,index4); else numCorners--;
			if (numCorners>0)
				totalDepth = totalDepth/(float)numCorners;
		}
	}
	//else totalDepth = INDEXH(fDepthsH,ptIndex);
	return totalDepth;

}
float NetCDFMoverCurv::GetTotalDepth(WorldPoint refPoint,long ptIndex)
{
	long index1, index2, index3, index4, numDepths;
	OSErr err = 0;
	float totalDepth = 0;
	Boolean useTriNum = false;
	long triNum = 0;

	if (fVar.gridType == SIGMA_ROMS)
	{
		//if (triNum < 0) useTriNum = false;
		err = ((TTriGridVel*)fGrid)->GetRectCornersFromTriIndexOrPoint(&index1, &index2, &index3, &index4, refPoint, triNum, useTriNum, fVerdatToNetCDFH, fNumCols+1);

		if (err) return 0;
		if (fDepthsH)
		{	// issue with extended grid not having depths - probably need to rework that idea
			long numCorners = 4;
			numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
			if (index1<numDepths && index1>=0) totalDepth += INDEXH(fDepthsH,index1); else numCorners--;
			if (index2<numDepths && index2>=0) totalDepth += INDEXH(fDepthsH,index2); else numCorners--;
			if (index3<numDepths && index3>=0) totalDepth += INDEXH(fDepthsH,index3); else numCorners--;
			if (index4<numDepths && index4>=0) totalDepth += INDEXH(fDepthsH,index4); else numCorners--;
			if (numCorners>0)
				totalDepth = totalDepth/(float)numCorners;
		}
	}
	else 
	{
		if (fDepthsH) totalDepth = INDEXH(fDepthsH,ptIndex);
	}
	return totalDepth;

}
void NetCDFMoverCurv::GetDepthIndices(long ptIndex, float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2)
{
// probably eventually switch to NetCDFMover only
	long indexToDepthData = 0;
	long numDepthLevels = GetNumDepthLevelsInFile();
	//float totalDepth = 0;
	//FLOATH depthsH = ((TTriGridVel3D*)fGrid)->GetDepths();
	
	/*if (fDepthsH)
	{
		totalDepth = INDEXH(fDepthsH,ptIndex);
	}
	else*/
	if (totalDepth==0)
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}

	if (fDepthLevelsHdl && numDepthLevels>0) 
	{
		//totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	}
	else
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}
	switch(fVar.gridType) 
	{	// function should not be called for TWO_D, haven't used BAROTROPIC yet
		/*case TWO_D:	// no depth data
			*depthIndex1 = indexToDepthData;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
		case BAROTROPIC:	// values same throughout column, but limit on total depth
			if (depthAtPoint <= totalDepth)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			else
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;*/
		//case MULTILAYER: //
			//break;
		case SIGMA: // 
		// code goes here, add SIGMA_ROMS, using z[k,:,:] = hc * (sc_r-Cs_r) + Cs_r * depth
			if (depthAtPoint<0)
			{	// keep in mind for grids with values at the bottom (rather than mid-cell) they may all be zero
				*depthIndex1 = indexToDepthData+numDepthLevels-1;
				*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				return;
			}
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// is sigma always 0-1 ?
				long j;
				float sigma, sigmaNext, depthAtLevel, depthAtNextLevel;
				for(j=0;j<numDepthLevels-1;j++)
				{
					sigma = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					sigmaNext = INDEXH(fDepthLevelsHdl,indexToDepthData+j+1);
					depthAtLevel = sigma * totalDepth;
					depthAtNextLevel = sigmaNext * totalDepth;
					if(depthAtLevel<depthAtPoint &&
						depthAtPoint<=depthAtNextLevel)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
						return;
					}
					else if(depthAtLevel==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
						return;
					}
				}
				if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
					return;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)*totalDepth<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
					return;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
					return;
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
				return;
			}
			//break;
		case SIGMA_ROMS: // 
		// code goes here, add SIGMA_ROMS, using z[k,:,:] = hc * (sc_r-Cs_r) + Cs_r * depth
		//WorldPoint wp; 
		//long triIndex;
		//totalDepth = GetTotalDepth(wp,triIndex);
			if (depthAtPoint<0)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				return;
			}
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// is sigma always 0-1 ?
				long j;
				float sc_r, sc_r2, Cs_r, Cs_r2, depthAtLevel, depthAtNextLevel;
				//for(j=0;j<numDepthLevels-1;j++)
				for(j=numDepthLevels-1;j>0;j--)
				{
					// sc and Cs are negative so need abs value
					/*float sc_r = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					float sc_r2 = INDEXH(fDepthLevelsHdl,indexToDepthData+j+1);
					float Cs_r = INDEXH(fDepthLevelsHdl2,indexToDepthData+j);
					float Cs_r2 = INDEXH(fDepthLevelsHdl2,indexToDepthData+j+1);*/
					sc_r = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					sc_r2 = INDEXH(fDepthLevelsHdl,indexToDepthData+j-1);
					Cs_r = INDEXH(fDepthLevelsHdl2,indexToDepthData+j);
					Cs_r2 = INDEXH(fDepthLevelsHdl2,indexToDepthData+j-1);
					//depthAtLevel = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
					//depthAtNextLevel = abs(hc * (sc_r2-Cs_r2) + Cs_r2 * totalDepth);
					depthAtLevel = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
					depthAtNextLevel = abs(totalDepth*(hc*sc_r2+totalDepth*Cs_r2))/(totalDepth+hc);
					if(depthAtLevel<depthAtPoint &&
						depthAtPoint<=depthAtNextLevel)
					{
						*depthIndex1 = indexToDepthData+j;
						//*depthIndex2 = indexToDepthData+j+1;
						*depthIndex2 = indexToDepthData+j-1;
						return;
					}
					else if(depthAtLevel==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
						return;
					}
				}
				//if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth==depthAtPoint)	// handles single depth case
				if(GetDepthAtIndex(indexToDepthData+numDepthLevels-1,totalDepth)==depthAtPoint)	// handles single depth case
				//if(GetTopDepth(indexToDepthData+numDepthLevels-1,totalDepth)==depthAtPoint)	// handles single depth case
				{
					//*depthIndex1 = indexToDepthData;
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX;
					return;
				}
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)*totalDepth<depthAtPoint)
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth<depthAtPoint)	// 0 is bottom
				else if(GetDepthAtIndex(indexToDepthData,totalDepth)<depthAtPoint)	// 0 is bottom
				//else if(GetBottomDepth(indexToDepthData,totalDepth)<depthAtPoint)	// 0 is bottom
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
					return;
				}
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth>depthAtPoint)
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)*totalDepth>depthAtPoint)
				else if(GetDepthAtIndex(indexToDepthData+numDepthLevels-1,totalDepth)>depthAtPoint)
				//else if(GetTopDepth(indexToDepthData+numDepthLevels-1,totalDepth)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
					return;
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
				return;
			}
			break;
		default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}
}
/*double NetCDFMoverCurv::GetTopDepth(long depthIndex, double totalDepth)
{	// really can combine and use GetDepthAtIndex - could move to base class
	double topDepth = 0;
	float sc_r, Cs_r;
	if (fVar.gridType == SIGMA_ROMS)
	{
		sc_r = INDEXH(fDepthLevelsHdl,depthIndex);
		Cs_r = INDEXH(fDepthLevelsHdl2,depthIndex);
		//topDepth = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
		topDepth = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
	}
	else
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth; // times totalDepth

	return topDepth;
}
double NetCDFMoverCurv::GetBottomDepth(long depthIndex, double totalDepth)
{
	double bottomDepth = 0;
	float sc_r, Cs_r;
	if (fVar.gridType == SIGMA_ROMS)
	{
		sc_r = INDEXH(fDepthLevelsHdl,depthIndex);
		Cs_r = INDEXH(fDepthLevelsHdl2,depthIndex);
		//bottomDepth = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
		bottomDepth = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
	}
	else
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth;
		
	return bottomDepth;
}*/
OSErr NetCDFMoverCurv::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// this code is for curvilinear grids
	OSErr err = 0;
	long i,j,k, numScanned, indexOfStart = 0;
	int status, ncid, latIndexid, lonIndexid, latid, lonid, recid, timeid, sigmaid, sigmavarid, sigmavarid2, hcvarid, depthid, mask_id, numdims;
	size_t latLength, lonLength, recs, t_len, t_len2, sigmaLength=0;
	float startLat,startLon,endLat,endLon,hc_param=0.;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	char dimname[NC_MAX_NAME], s[256], topPath[256], outPath[256];
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0, sigmaLevelsH=0;
	float yearShift=0.;
	//float *lat_vals=0,*lon_vals=0,timeVal;
	double *lat_vals=0,*lon_vals=0,timeVal;
	float *depth_vals=0,*sigma_vals=0,*sigma_vals2=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex[2]={0,0},sigmaIndex=0;
	static size_t pt_count[2], sigma_count;
	Seconds startTime, startTime2;
	double timeConversion = 1., scale_factor = 1.;
	char errmsg[256] = "";
	char fileName[64],*modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, isLandMask = true;
	VelocityFH velocityH = 0;
	//long numTimesInFile = 0;

	if (!path || !path[0]) return 0;
	strcpy(fVar.pathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fVar.userName, fileName); // maybe use a name from the file
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	// check number of dimensions - 2D or 3D
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; /*goto done;*/}}	// will need to split for Navy vs LAS
	else 
	{
		fIsNavy = true;
		// may only need to see keyword is there, since already checked grid type
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; goto done;}}	// will need to split for regridded or non-Navy cases 
		modelTypeStr[t_len2] = '\0';
		
		strcpy(fVar.userName, modelTypeStr); // maybe use a name from the file
		/*
		if (!strncmp (modelTypeStr, "SWAFS", 5))
		 fIsNavy = true;
		else if (!strncmp (modelTypeStr, "fictitious test data", strlen("fictitious test data")))
		 fIsNavy = true;
		else
		 fIsNavy = false;*/
	}
	
	/*if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "time", &recid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	else
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR) {err = -1; goto done;}
	}*/

	status = nc_inq_dimid(ncid, "time", &recid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR || recid==-1) {err = -1; goto done;}
	}

	//if (fIsNavy)
		status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {status = nc_inq_varid(ncid, "TIME", &timeid);if (status != NC_NOERR) {err = -1; goto done;} /*timeid = recid;*/} 	// for Ferret files, everything is in CAPS
	//if (status != NC_NOERR) {/*err = -1; goto done;*/ timeid = recid;} 	// for LAS files, variable names unstable

	//if (!fIsNavy)
		//status = nc_inq_attlen(ncid, recid, "units", &t_len);	// recid is the dimension id not the variable id
	//else	// LAS has them in order, and time is unlimited, but variable/dimension names keep changing so leave this way for now
		status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		//err = -1; goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];
		
		timeUnits = new char[t_len+1];
		//if (!fIsNavy)
			//status = nc_get_att_text(ncid, recid, "units", timeUnits);	// recid is the dimension id not the variable id
		//else
			status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
					  unitStr, junk, &time.year, &time.month, &time.day,
					  &time.hour, &time.minute, &time.second) ;
		if (numScanned==5 || numScanned==4)	
			{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned!=8)	
		//if (numScanned!=8)	
		{ 
			timeUnits = 0;	// files should always have this info
			timeConversion = 3600.;		// default is hours
			startTime2 = model->GetStartTime();	// default to model start time
			/*err = -1; TechError("NetCDFMoverCurv::TextRead()", "sscanf() == 8", 0); goto done;*/
		}
		else
		{
			// code goes here, trouble with the DAYS since 1900 format, since converts to seconds since 1904
			if (time.year ==1900) {time.year += 40; time.day += 1; /*for the 1900 non-leap yr issue*/ yearShift = 40.;}
		DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
		if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
			timeConversion = 3600.;
		else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
			timeConversion = 60.;
		else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
			timeConversion = 1.;
		else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
			timeConversion = 24.*3600.;
		}
	} 

	if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "gridy", &latIndexid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimid(ncid, "gridx", &lonIndexid);	//Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		// option to use index values?
		status = nc_inq_varid(ncid, "grid_lat", &latid);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "grid_lon", &lonid);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	else
	{
		for (i=0;i<numdims;i++)
		{
			if (i == recid) continue;
			status = nc_inq_dimname(ncid,i,dimname);
			if (status != NC_NOERR) {err = -1; goto done;}
			//if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3))
			if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3) || !strncmpnocase(dimname,"NX",2))
			{
				lonIndexid = i;
			}
			//if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3))
			if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3) || !strncmpnocase(dimname,"NY",2))
			{
				latIndexid = i;
			}
		}
		
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}

		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
	
		status = nc_inq_varid(ncid, "LATITUDE", &latid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lat", &latid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_varid(ncid, "LONGITUDE", &lonid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lon", &lonid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(latLength*lonLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	//lat_vals = new float[latLength*lonLength]; 
	lat_vals = new double[latLength*lonLength]; 
	//lon_vals = new float[latLength*lonLength]; 
	lon_vals = new double[latLength*lonLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	//status = nc_get_vara_float(ncid, latid, ptIndex, pt_count, lat_vals);
	status = nc_get_vara_double(ncid, latid, ptIndex, pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_vara_float(ncid, lonid, ptIndex, pt_count, lon_vals);
	status = nc_get_vara_double(ncid, lonid, ptIndex, pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
				//lat_vals[(latLength-i-1)*lonLength+j]=0.;
			//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
				//lon_vals[(latLength-i-1)*lonLength+j]=0.;
			// grid ordering does matter for creating ptcurmap, assume increases fastest in x/lon, then in y/lat
			INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(latLength-i-1)*lonLength+j];	
			INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(latLength-i-1)*lonLength+j];
			//INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(i)*lonLength+j];	
			//INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(i)*lonLength+j];
		}
	}
	fVertexPtsH	 = vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	//latIndex = 0;
	//lonIndex = 0;
	//status = nc_get_var1_float(ncid, latIndexid, &latIndex, &startLat);	// this won't work for curvilinear case
	//status = nc_get_var1_float(ncid, lonIndexid, &lonIndex, &startLon);
	//latIndex = latLength-1;
	//lonIndex = lonLength-1;
	//status = nc_get_var1_float(ncid, latIndexid, &latIndex, &endLat);	// this won't work for curvilinear case
	//status = nc_get_var1_float(ncid, lonIndexid, &lonIndex, &endLon);


	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR || fIsNavy) {fVar.gridType = TWO_D; /*err = -1; goto done;*/}	// check for zgrid option here
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "sc_r", &sigmavarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, "Cs_r", &sigmavarid2);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			fVar.gridType = SIGMA_ROMS;
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_vals2 = new float[sigmaLength];
			if (!sigma_vals2) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_vara_float(ncid, sigmavarid2, &sigmaIndex, &sigma_count, sigma_vals2);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, "hc", &hcvarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_var1_float(ncid, hcvarid, &sigmaIndex, &hc_param);
			if (status != NC_NOERR) {err = -1; goto done;}
			//{err = -1; goto done;}
		}
		else
		{
			// code goes here, for SIGMA_ROMS the variable isn't sigma but sc_r and Cs_r, with parameter hc
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			// check if sigmaLength > 1
			fVar.gridType = SIGMA;
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			//sigmaLevelsH = (FLOATH)_NewHandleClear(sigmaLength*sizeof(sigmaLevelsH));
			//if (!sigmaLevelsH) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		// once depth is read in 
	}

	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR || fIsNavy) {fVar.gridType = TWO_D;/*err = -1; goto done;*/}
	else
	{	
		totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
		if (!totalDepthsH) {err = memFullErr; goto done;}
		depth_vals = new float[latLength*lonLength];
		if (!depth_vals) {err = memFullErr; goto done;}
		status = nc_get_vara_float(ncid, depthid, ptIndex,pt_count, depth_vals);
		if (status != NC_NOERR) {err = -1; goto done;}

		status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor

	}

	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}

	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		//if (!fIsNavy)
			//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		//else
			//status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
			status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		// get rid of the seconds since they get garbled in the dialogs
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		INDEXH(fTimeHdl,i) = newTime-yearShift*3600.*24.*365.25;	// which start time where?
		if (i==0) startTime = newTime-yearShift*3600.*24.*365.25;
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion -yearShift*3600.*24.*365.25;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion -yearShift*3600.*24.*365.25 + fTimeShift;
	}
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
			//buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first\n time in the file?",FALSE);
			//if(!gCommandFileErrorLogPath[0])
			if(!gCommandFileRun)	// also may want to skip for location files...
				buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
			switch(buttonSelected){
				case 1: // reset model start time
					//bTopFile = true;
					model->SetModelTime(startTime);
					model->SetStartTime(startTime);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
					break;  
				case 3: // don't reset model start time
					//bTopFile = false;
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
		//model->SetModelTime(startTime);
		//model->SetStartTime(startTime);
		//model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
	}

	fNumRows = latLength;
	fNumCols = lonLength;

	status = nc_inq_varid(ncid, "mask", &mask_id);
	if (status != NC_NOERR)	{/*err=-1; goto done;*/ isLandMask = false;}

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	//err = this -> SetInterval(errmsg);
	//if(err) goto done;

	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	//{if (topFilePath[0]) {strcpy(fTopFilePath,topFilePath); err = ReadTopology(fTopFilePath,newMap); goto done;}}
	{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto depths;}}
	//if (isLandMask/*fIsNavy*//*true*/)	// allow for the LAS files too ?
	if (true)	// allow for the LAS files too ?
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
		//if (!reply.good) return USERCANCEL;
		if (!reply.good) /*return 0;*/
		{
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}
		else
			strcpy(topPath, reply.fullPath);

		/*{
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}*/
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
			//numTimesInFile = this -> GetNumTimesInFile();	// use recs?
			//if (numTimesInFile>0)
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
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
		goto depths;
		//SplitPathFile (s, fileName);
	}

	//numTimesInFile = this -> GetNumTimesInFile();
	//if (numTimesInFile>0)
	if (recs>0)
		err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
	else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
	if(err) goto done;
	if (isLandMask) err = ReorderPoints(velocityH,newMap,errmsg);
	else err = ReorderPointsNoMask(velocityH,newMap,errmsg);
	//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?

depths:
	if (err) goto done;
	// also translate to fDepthDataInfo and fDepthsH here, using sigma or zgrid info
	
	if (totalDepthsH)
	{
		fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols);
		if(!fDepthsH){TechError("NetCDFMoverCurv::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i=0;i<latLength;i++)
		{
			for (j=0;j<lonLength;j++)
			{
				//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
					//lat_vals[(latLength-i-1)*lonLength+j]=0.;
				//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
					//lon_vals[(latLength-i-1)*lonLength+j]=0.;
				INDEXH(totalDepthsH,i*lonLength+j) = depth_vals[(latLength-i-1)*lonLength+j] * scale_factor;	
				INDEXH(fDepthsH,i*lonLength+j) = depth_vals[(latLength-i-1)*lonLength+j] * scale_factor;	
			}
		}
		//((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}

	fNumDepthLevels = sigmaLength;
	if (sigmaLength>1)
	{
		//status = nc_get_vara_double(ncid, depthvarid, &ptIndex, &pt_count[2], depthLevels);
		//if (status != NC_NOERR) {err=-1; goto done;}
		float sigma = 0;
		fDepthLevelsHdl = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
		if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
		for (i=0;i<sigmaLength;i++)
		{	// decide what to do here, may be upside down for ROMS
			sigma = sigma_vals[i];
			if (sigma_vals[0]==1) 
				INDEXH(fDepthLevelsHdl,i) = (1-sigma);	// in this case velocities will be upside down too...
			else
			{
				if (fVar.gridType == SIGMA_ROMS)
					INDEXH(fDepthLevelsHdl,i) = sigma;
				else
					INDEXH(fDepthLevelsHdl,i) = abs(sigma);
			}
			
		}
		if (fVar.gridType == SIGMA_ROMS)
		{
			fDepthLevelsHdl2 = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
			if (!fDepthLevelsHdl2) {err = memFullErr; goto done;}
			for (i=0;i<sigmaLength;i++)
			{
				sigma = sigma_vals2[i];
				//if (sigma_vals[0]==1) 
					//INDEXH(fDepthLevelsHdl,i) = (1-sigma);	// in this case velocities will be upside down too...
				//else
					INDEXH(fDepthLevelsHdl2,i) = sigma;
			}
			hc = hc_param;
		}
	}

	/*if (totalDepthsH) 
	{	// use fDepths only if 
		fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols);
		if(!fDepthsH){TechError("NetCDFMoverCurv::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		fDepthsH = totalDepthsH;
	}*/	// may be null, call it barotropic if depths exist??
	// CalculateVerticalGrid(sigmaLength,sigmaLevelsH,totalDepthsH);	// maybe multigrid
	/*{
		long j,index = 0;
		fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**fDepthDataInfo)*fNumRows*fNumCols);
		if(!fDepthDataInfo){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		if (fVar.gridType==TWO_D) 
			{if (totalDepthsH) fDepthsH = totalDepthsH;}	// may be null, call it barotropic if depths exist?? - should copy here
		// assign arrays
		else
		{	//TWO_D grid won't need fDepthsH
			fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols*fVar.maxNumDepths);
			if(!fDepthsH){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
		}
		for (i=0;i<fNumRows;i++)
		{
			for (j=0;j<fNumCols;j++)
			{
				// might want to order all surface depths, all sigma1, etc., but then indexToDepthData wouldn't work
				// have 2D case, zgrid case as well
				if (fVar.gridType==TWO_D)
				{
					if (totalDepthsH) (*fDepthDataInfo)[i*fNumCols+j].totalDepth = (*totalDepthsH)[i*fNumCols+j];
					else (*fDepthDataInfo)[i*fNumCols+j].totalDepth = -1;	// no depth data
					(*fDepthDataInfo)[i*fNumCols+j].indexToDepthData = i*fNumCols+j;
					(*fDepthDataInfo)[i*fNumCols+j].numDepths = 1;
				}
				else
				{
					(*fDepthDataInfo)[i*fNumCols+j].totalDepth = (*totalDepthsH)[i*fNumCols+j];
					(*fDepthDataInfo)[i*fNumCols+j].indexToDepthData = index;
					(*fDepthDataInfo)[i*fNumCols+j].numDepths = sigmaLength;
					for (k=0;k<sigmaLength;k++)
					{
						//(*fDepthsH)[index+j] = (*totalDepthsH)[i]*(1-(*sigmaLevelsH)[j]);
						// any other option than 1:0 or 0:1 ?
						if (sigma_vals[0]==1) (*fDepthsH)[index+j] = (*totalDepthsH)[i*fNumCols+j]*(1-sigma_vals[k]);
						else (*fDepthsH)[index+j] = (*totalDepthsH)[i*fNumCols+j]*sigma_vals[k];	// really need a check on all values
						//(*fDepthsH)[j*fNumNodes+i] = totalDepthsH[i]*(1-sigmaLevelsH[j]);
					}
					index+=sigmaLength;
				}
			}
		}
	}*/
	if (totalDepthsH)
	{	// may need to extend the depth grid along with lat/lon grid - not sure what to use for the values though...
		// not sure what map will expect in terms of depths order
		long n,ptIndex,iIndex,jIndex;
		long numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
		//_SetHandleSize((Handle)totalDepthsH,(fNumRows+1)*(fNumCols+1)*sizeof(float));
		_SetHandleSize((Handle)totalDepthsH,numPoints*sizeof(float));
		//for (i=0; i<fNumRows*fNumCols; i++)
		//for (i=0; i<(fNumRows+1)*(fNumCols+1); i++)

			/*if (iIndex==0)
			{
				if (jIndex<fNumCols)
				{
					dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
					fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
					fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
				}
				else
				{
					dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
				}
			}
			else 
			{
				if (jIndex<fNumCols)
				{
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
					u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
					v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
				}
				else
				{
					dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
				}
			}*/

		for (i=0; i<numPoints; i++)
		{	// works okay for simple grid except for far right column (need to extend depths similar to lat/lon)
			// if land use zero, if water use point next to it?
			ptIndex = INDEXH(fVerdatToNetCDFH,i);
			iIndex = ptIndex/(fNumCols+1);
			jIndex = ptIndex%(fNumCols+1);
			if (iIndex>0 && jIndex<fNumCols)
				ptIndex = (iIndex-1)*(fNumCols)+jIndex;
			else
				ptIndex = -1;

			//n = INDEXH(fVerdatToNetCDFH,i);
			//if (n<0 || n>= fNumRows*fNumCols) {printError("indices messed up"); err=-1; goto done;}
			//INDEXH(totalDepthsH,i) = depth_vals[n];
			if (ptIndex<0 || ptIndex>= fNumRows*fNumCols) 
			{
				//printError("indices messed up"); 
				//err=-1; goto done;
				INDEXH(totalDepthsH,i) = 0;	// need to figure out what to do here...
				continue;
			}
			//INDEXH(totalDepthsH,i) = depth_vals[ptIndex];
			INDEXH(totalDepthsH,i) = INDEXH(fDepthsH,ptIndex);
		}
		((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}

done:
	// code goes here, set bathymetry
	if (err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
		if(sigmaLevelsH) {DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH = 0;}
		if (fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
		if (fDepthLevelsHdl2) {DisposeHandle((Handle)fDepthLevelsHdl2); fDepthLevelsHdl2=0;}
	}

	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (sigma_vals) delete [] sigma_vals;
	if (modelTypeStr) delete [] modelTypeStr;
	if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
	return err;
}
	 

OSErr NetCDFMoverCurv::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	OSErr err = 0;
	long i,j,k;
	char path[256], outPath[256];
	char *velUnits=0; 
	int status, ncid, numdims;
	int curr_ucmp_id, curr_vcmp_id, curr_wcmp_id, angle_id, mask_id, uv_ndims;
	static size_t curr_index[] = {0,0,0,0}, angle_index[] = {0,0};
	static size_t curr_count[4], angle_count[2];
	size_t velunit_len;
	//float *curr_uvals = 0,*curr_vvals = 0, fill_value=-1e-72;
	//float *landmask = 0;
	double *curr_uvals = 0,*curr_vvals = 0, *curr_wvals = 0, fill_value=-1e+34, test_value=8e+10;
	double *landmask = 0, velConversion=1.;
	//short *curr_uvals_Navy = 0,*curr_vvals_Navy = 0, fill_value_Navy;
	//float *angle_vals = 0,debug_mask;
	double *angle_vals = 0,debug_mask;
	//long totalNumberOfVels = fNumRows * fNumCols;
	long totalNumberOfVels = fNumRows * fNumCols * fVar.maxNumDepths;
	VelocityFH velH = 0;
	FLOATH wvelH = 0;
	long latlength = fNumRows, numtri = 0;
	long lonlength = fNumCols;
	//float scale_factor = 1.,angle = 0.,u_grid,v_grid;
	double scale_factor = 1.,angle = 0.,u_grid,v_grid;
	long numDepths = fVar.maxNumDepths;	// assume will always have full set of depths at each point for now
	Boolean bRotated = true, isLandMask = true, bIsWVel = false;
	
	errmsg[0]=0;

	// write out verdat file for debugging
	/*FILE *outfile = 0;
	char name[32], verdatpath[256],m[300];
	SFReply reply;
	Point where = CenteredDialogUpLeft(M55);
	Boolean changeExtension = false;	// for now
	char previousPath[256]="",defaultExtension[3]="";
	char ibmBackwardsTypeStr[32] = "";
	strcpy(name,"NewVerdat.dat");
	//errmsg[0]=0;
	
#if TARGET_API_MAC_CARBON
		err = AskUserForSaveFilename("verdat.dat",verdatpath,".dat",true);
		if (err) return USERCANCEL;
#else
 #ifdef MAC
		sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
 #else
		sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	           M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
 #endif
	if (!reply.good) {err = -1; goto done;}

	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath (reply.vRefNum, 0, (char *) "", verdatpath);
	strcat (verdatpath, ":");
	strcat (verdatpath, (char *) reply.fName);
#else
	strcpy(verdatpath, reply.fName);
#endif
#endif
	//strcpy(sExportSelectedTriPath, verdatpath); // remember the path for the user
	SetWatchCursor();
	sprintf(m, "Exporting VERDAT to %s...",verdatpath);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(m);*/
	/////////////////////////////////////////////////


	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;

	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}

	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	if (numdims>=4)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		curr_count[1] = numDepths;	// depth
		curr_count[2] = latlength;
		curr_count[3] = lonlength;
	}
	else
	{
		curr_count[1] = latlength;	
		curr_count[2] = lonlength;
	}
	angle_count[0] = latlength;
	angle_count[1] = lonlength;
	
	//outfile=fopen(verdatpath,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");

	if (fIsNavy)
	{
		numDepths = 1;
		// need to check if type is float or short, if float no scale factor?
		//curr_uvals = new float[latlength*lonlength]; 
		curr_uvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_uvals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		//curr_vvals = new float[latlength*lonlength]; 
		curr_vvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_vvals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		//curr_uvals_Navy = new short[latlength*lonlength]; 
		//if(!curr_uvals_Navy) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		//curr_vvals_Navy = new short[latlength*lonlength]; 
		//if(!curr_vvals_Navy) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		//angle_vals = new float[latlength*lonlength]; 
		angle_vals = new double[latlength*lonlength]; 
		if(!angle_vals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		status = nc_inq_varid(ncid, "water_gridu", &curr_ucmp_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "water_gridv", &curr_vcmp_id);	// what if only input one at a time (u,v separate movers)?
		if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_short(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals_Navy);
		//if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_short(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals_Navy);
		//if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_att_short(ncid, curr_ucmp_id, "_FillValue", &fill_value_Navy);
		//if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		status = nc_get_vara_double(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		status = nc_get_vara_double(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		status = nc_get_att_double(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		//if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_att_float(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
		status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
		//if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "grid_orient", &angle_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_float(ncid, angle_id, angle_index, angle_count, angle_vals);
		status = nc_get_vara_double(ncid, angle_id, angle_index, angle_count, angle_vals);
		if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
	}
	else
	{
		status = nc_inq_varid(ncid, "mask", &mask_id);
		if (status != NC_NOERR)	{/*err=-1; goto done;*/ isLandMask = false;}
		//curr_uvals = new float[latlength*lonlength]; 
		status = nc_inq_varid(ncid, "ang", &angle_id);
		if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
		//status = nc_get_vara_float(ncid, angle_id, angle_index, angle_count, angle_vals);
		else
		{
			angle_vals = new double[latlength*lonlength]; 
			if(!angle_vals) {TechError("GridVel::ReadNetCDFFile()", "new[ ]", 0); err = memFullErr; goto done;}
			status = nc_get_vara_double(ncid, angle_id, angle_index, angle_count, angle_vals);
			if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
		}
		curr_uvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_uvals) 
		{
			TechError("GridVel::ReadNetCDFFile()", "new[]", 0); 
			err = memFullErr; 
			goto done;
		}
		//curr_vvals = new float[latlength*lonlength]; 
		curr_vvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_vvals) 
		{
			TechError("GridVel::ReadNetCDFFile()", "new[]", 0); 
			err = memFullErr; 
			goto done;
		}
		curr_wvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_wvals) 
		{
			TechError("GridVel::ReadNetCDFFile()", "new[]", 0); 
			err = memFullErr; 
			goto done;
		}
		if (isLandMask)
		{
			//landmask = new float[latlength*lonlength]; 
			landmask = new double[latlength*lonlength]; 
			if(!landmask) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		}
		status = nc_inq_varid(ncid, "U", &curr_ucmp_id);
		if (status != NC_NOERR)
		{
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
			if (status != NC_NOERR)
			{
				status = nc_inq_varid(ncid, "water_u", &curr_ucmp_id);
				if (status != NC_NOERR)
				{err = -1; goto done;}
			}
			//{err = -1; goto done;}
		}
		status = nc_inq_varid(ncid, "V", &curr_vcmp_id);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
			if (status != NC_NOERR) 
			{
				status = nc_inq_varid(ncid, "water_v", &curr_vcmp_id);
				if (status != NC_NOERR)
				{err = -1; goto done;}
			}
			//{err = -1; goto done;}
		}
		status = nc_inq_varid(ncid, "W", &curr_wcmp_id);
		if (status != NC_NOERR)
		{
			status = nc_inq_varid(ncid, "w", &curr_wcmp_id);
			if (status != NC_NOERR)
			{
				status = nc_inq_varid(ncid, "water_w", &curr_wcmp_id);
				if (status != NC_NOERR)
				//{err = -1; goto done;}
					bIsWVel = false;
				else
					bIsWVel = true;
			}
			//{err = -1; goto done;}
		}
		if (isLandMask)
		{
			//status = nc_get_vara_float(ncid, mask_id, angle_index, angle_count, landmask);
			status = nc_get_vara_double(ncid, mask_id, angle_index, angle_count, landmask);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
		if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {curr_count[1] = latlength; curr_count[2] = lonlength;}}	// could have more dimensions than are used in u,v
		//status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		status = nc_get_vara_double(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		status = nc_get_vara_double(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		if (bIsWVel)
		{	
			status = nc_get_vara_double(ncid, curr_wcmp_id, curr_index, curr_count, curr_wvals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_attlen(ncid, curr_ucmp_id, "units", &velunit_len);
		if (status == NC_NOERR)
		{
			velUnits = new char[velunit_len+1];
			status = nc_get_att_text(ncid, curr_ucmp_id, "units", velUnits);
			if (status == NC_NOERR)
			{
				velUnits[velunit_len] = '\0';
				if (!strcmpnocase(velUnits,"cm/s"))
					velConversion = .01;
				else if (!strcmpnocase(velUnits,"m/s"))
					velConversion = 1.0;
			}
		}


		//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		status = nc_get_att_double(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		//if (status != NC_NOERR) {status = nc_get_att_float(ncid, curr_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){err = -1; goto done;}*/}	// don't require
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "FillValue", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "missing_value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
		//if (status != NC_NOERR) {err = -1; goto done;}	// don't require
		status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
	}	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) 
	{
		err = memFullErr; 
		goto done;
	}
	//for (i=0;i<totalNumberOfVels;i++)
	for (k=0;k<numDepths;k++)
	{
	for (i=0;i<latlength;i++)
	{
		for (j=0;j<lonlength;j++)
		{
			if (fIsNavy)
			{
				//if (curr_uvals_Navy[(latlength-i-1)*lonlength+j]==fill_value_Navy)
					//curr_uvals_Navy[(latlength-i-1)*lonlength+j]=0.;
				//if (curr_vvals_Navy[(latlength-i-1)*lonlength+j]==fill_value_Navy)
					//curr_vvals_Navy[(latlength-i-1)*lonlength+j]=0.;
				//u_grid = (float)curr_uvals_Navy[(latlength-i-1)*lonlength+j];
				//v_grid = (float)curr_vvals_Navy[(latlength-i-1)*lonlength+j];
				if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
					curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
				if (curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
					curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
				//u_grid = (float)curr_uvals[(latlength-i-1)*lonlength+j];
				//v_grid = (float)curr_vvals[(latlength-i-1)*lonlength+j];
				u_grid = (double)curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols];
				v_grid = (double)curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols];
				if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
				INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = u_grid*cos(angle*PI/180.)-v_grid*sin(angle*PI/180.);
				INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = u_grid*sin(angle*PI/180.)+v_grid*cos(angle*PI/180.);
			}
			else
			{
				// Look for a land mask, but do this if don't find one - float mask(lat,lon) - 1,0 which is which?
				// Until the files have land masks the work around for NYNJ is to make sure zero is treated as a velocity
				// while for Galveston (and Tampa Bay) zero is a land value, not sure for Lake Erie 
				//if (curr_uvals[(latlength-i-1)*lonlength+j]==0. && curr_vvals[(latlength-i-1)*lonlength+j]==0.)
					//curr_uvals[(latlength-i-1)*lonlength+j] = curr_vvals[(latlength-i-1)*lonlength+j] = 1e-06;

				// just leave fillValue as velocity for new algorithm - comment following lines out
				// should eliminate the above problem, assuming fill_value is a land mask
				/*if (curr_uvals[(latlength-i-1)*lonlength+j]==fill_value)
					curr_uvals[(latlength-i-1)*lonlength+j]=0.;
				if (curr_vvals[(latlength-i-1)*lonlength+j]==fill_value)
					curr_vvals[(latlength-i-1)*lonlength+j]=0.;*/

				if (isLandMask)
				{
					if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value || curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
					if (abs(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols])>test_value || abs(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols])>test_value)
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#ifdef MAC
					//if (__isnan(curr_uvals[(latlength-i-1)*lonlength+j]) || __isnan(curr_vvals[(latlength-i-1)*lonlength+j]))
					//if ((curr_uvals[(latlength-i-1)*lonlength+j])==NAN || (curr_vvals[(latlength-i-1)*lonlength+j])==NAN)
					if (isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#else
					if (_isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || _isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#endif
						debug_mask = landmask[(latlength-i-1)*lonlength+j];
						//if (debug_mask == 1.1) numtri++;
						if (debug_mask > 0) 
						{
							numtri++;
						}
					//if (landmask[(latlength-i-1)*lonlength+j]<1)	// land
					if (landmask[(latlength-i-1)*lonlength+j]<1 || landmask[(latlength-i-1)*lonlength+j]>8)	// land
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=fill_value;
					/*else
					{
						float dLat = INDEXH(fVertexPtsH,i*fNumCols+j).pLat;
						float dLon = INDEXH(fVertexPtsH,i*fNumCols+j).pLong;
						long index = i*fNumCols+j+1;
						float dZ = 1.;
						fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, dLon, dLat, dZ);	
					}*/

					//if (landmask[(latlength-i-1)*lonlength+j]<1)
					if (landmask[(latlength-i-1)*lonlength+j]<1 || landmask[(latlength-i-1)*lonlength+j]>8)
						curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=fill_value;
					/*if (landmask[(i)*lonlength+(lonlength-j-1)]<1 || landmask[(i)*lonlength+(lonlength-j-1)]>8)	// land
						curr_uvals[(i)*lonlength+(lonlength-j-1)]=fill_value;
					//if (landmask[(latlength-i-1)*lonlength+j]<1)
					if (landmask[(i)*lonlength+(lonlength-j-1)]<1 || landmask[(i)*lonlength+(lonlength-j-1)]>8)
						curr_vvals[(i)*lonlength+(lonlength-j-1)]=fill_value;*/
					/*if (landmask[(i)*lonlength+j]<1 || landmask[(i)*lonlength+j]>8)	// land
						curr_uvals[(i)*lonlength+j]=fill_value;
					//if (landmask[(latlength-i-1)*lonlength+j]<1)
					if (landmask[(i)*lonlength+j]<1 || landmask[(i)*lonlength+j]>8)
						curr_vvals[(i)*lonlength+j+k*fNumRows*fNumCols]=fill_value;*/
				}
				else
				{
					if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value || curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#ifdef MAC
					//if (__isnan(curr_uvals[(latlength-i-1)*lonlength+j]) || __isnan(curr_vvals[(latlength-i-1)*lonlength+j]))
					//if ((curr_uvals[(latlength-i-1)*lonlength+j])==NAN || (curr_vvals[(latlength-i-1)*lonlength+j])==NAN)
					if (isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#else
					if (_isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || _isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#endif					// if use fill_value need to be sure to check for it in GetMove and VelocityStrAtPoint
					//if (curr_uvals[(latlength-i-1)*lonlength+j]==0 && curr_vvals[(latlength-i-1)*lonlength+j]==0)
						//curr_uvals[(latlength-i-1)*lonlength+j] = curr_vvals[(latlength-i-1)*lonlength+j] = fill_value;
				}
/////////////////////////////////////////////////
				if (bRotated)
				{
					u_grid = (double)curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
					v_grid = (double)curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
					if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
					//INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = u_grid*cos(angle*PI/180.)-v_grid*sin(angle*PI/180.);
					//INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = u_grid*sin(angle*PI/180.)+v_grid*cos(angle*PI/180.);
					INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = u_grid*cos(angle)-v_grid*sin(angle);	//in radians
					INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = u_grid*sin(angle)+v_grid*cos(angle);
				}
				else
				{
					INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;	// need units
					INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
				}
				//INDEXH(velH,i*lonlength+j).u = curr_uvals[(i)*lonlength+j] * velConversion;	// need units
				//INDEXH(velH,i*lonlength+j).v = curr_vvals[(i)*lonlength+j] * velConversion;
				//INDEXH(velH,i*lonlength+j).u = curr_uvals[(latlength-i-1)*lonlength+j];	// need units
				//INDEXH(velH,i*lonlength+j).v = curr_vvals[(latlength-i-1)*lonlength+j];
			}
		}
	}
	}
	*velocityH = velH;
	//fclose(outfile);
	//if (fIsNavy)
		//fFillValue = fill_value_Navy;
	//else 
		//fFillValue = fill_value;
		fFillValue = fill_value * velConversion;
	
	if (scale_factor!=1.) fVar.curScale = scale_factor;	// hmm, this forces a reset of scale factor each time, overriding any set by hand
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading current data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		//printNote("Error opening NetCDF file");
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) 
	{
		delete [] curr_uvals; 
		curr_uvals = 0;
	}
	if (curr_vvals) 
	{
		delete [] curr_vvals; 
		curr_vvals = 0;
	}
	if (curr_wvals) 
	{
		delete [] curr_wvals; 
		curr_wvals = 0;
	}
	//if (curr_uvals_Navy) delete [] curr_uvals_Navy;
	//if (curr_vvals_Navy) delete [] curr_vvals_Navy;
	if (landmask) {delete [] landmask; landmask = 0;}
	if (angle_vals) {delete [] angle_vals; angle_vals = 0;}
	if (velUnits) {delete [] velUnits;}
	return err;
}

long NetCDFMoverCurv::CheckSurroundingPoints(LONGH maskH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long neighbor;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows - 1) ? row + 1 : fNumRows - 1;
	jEnd = (col < fNumCols - 1) ? col + 1 : fNumCols - 1;
	// don't allow diagonals for now,they could be separate small islands 
	/*for (i = iStart; i< iEnd+1; i++)
	{
		for (j = jStart; j< jEnd+1; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols + j);
			if (neighbor >= 3 && neighbor < lowestLandIndex)
				lowestLandIndex = neighbor;
		}
	}*/
	for (i = iStart; i< iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols + col);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols + j);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	return lowestLandIndex;
}

Boolean NetCDFMoverCurv::ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long neighbor;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows - 1) ? row + 1 : fNumRows - 1;
	jEnd = (col < fNumCols - 1) ? col + 1 : fNumCols - 1;
	/*for (i = iStart; i < iEnd+1; i++)
	{
		for (j = jStart; j < jEnd+1; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols + j);
			// eventually should use a land mask or fill value to identify land
			if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==0. && INDEXH(velocityH,i*fNumCols+j).v==0.)) return true;
		}
	}*/
	for (i = iStart; i < iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols + col);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+col).u==fFillValue && INDEXH(velocityH,i*fNumCols+col).v==fFillValue)) return true;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols + j);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || (INDEXH(velocityH,row*fNumCols+j).u==fFillValue && INDEXH(velocityH,row*fNumCols+j).v==fFillValue)) return true;
	}
	return false;
}

Boolean NetCDFMoverCurv::InteriorLandPoint(LONGH maskH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd;
	long neighbor;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows_ext - 1) ? row + 1 : fNumRows_ext - 1;
	jEnd = (col < fNumCols_ext - 1) ? col + 1 : fNumCols_ext - 1;
	/*for (i = iStart; i < iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols_ext + col);
		if (neighbor < 3)	// water point
			return false;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols_ext + j);
		if (neighbor < 3)	// water point
			return false;
	}*/
	//for (i = iStart; i < iEnd+1; i++)
	// point is in lower left corner of grid box (land), so only check 3 other quadrants of surrounding 'square'
	for (i = row; i < iEnd+1; i++)
	{
		//for (j = jStart; j< jEnd+1; j++)
		for (j = jStart; j< jEnd; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + j);
			if (neighbor < 3 /*&& neighbor != -1*/)	// water point
				return false;
			//if (row==1 && INDEXH(maskH,j)==1) return false;
		}
	}
	return true;
}

Boolean NetCDFMoverCurv::ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) 
{
	long iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long i, j, neighbor, landPolyNum;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows_ext - 1) ? row + 1 : fNumRows_ext - 1;
	jEnd = (col < fNumCols_ext - 1) ? col + 1 : fNumCols_ext - 1;
	
	landPolyNum = INDEXH(maskH, row*fNumCols_ext + col);
	for (i = iStart; i< iEnd+1; i++)
	{
			if (i==row) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + col);
			if (neighbor >= 3 && neighbor < landPolyNum) 
			{
				*lowerPolyNum = neighbor;
				return true;
			}
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols_ext + j);
		if (neighbor >= 3 && neighbor < landPolyNum) 
		{
			*lowerPolyNum = neighbor;
			return true;
		}
	}
	// don't allow diagonals for now, they could be separate small islands
	/*for (i = iStart; i< iEnd+1; i++)
	{
		for (j = jStart; j< jEnd+1; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + j);
			if (neighbor >= 3 && neighbor < landPolyNum) 
			{
				*lowerPolyNum = neighbor;
				return true;
			}
		}
	}*/
	return false;
}

void NetCDFMoverCurv::ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin)
{	// merges adjoining land blocks and then renumbers any higher numbered land blocks
	long i,j,val;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{	
			val = INDEXH(maskH,i*fNumCols_ext+j);
			if (val==landBlockToMerge) INDEXH(maskH,i*fNumCols_ext+j) = landBlockToJoin;
			if (val>landBlockToMerge) INDEXH(maskH,i*fNumCols_ext+j) -= 1;
		}
	}
}

OSErr NetCDFMoverCurv::NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo, long *numIslands) 
{
	OSErr err = 0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long i, j, n, landPolyNum = 1, lowestSurroundingNum = 0;
	long islandNum, maxIslandNum=3;
	LONGH maskH = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	*islandNumberH = 0;
	
	if (!maskH || !maskH2) {err = memFullErr; goto done;}
	// use surface velocity values at time zero
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j) == -1)// 1 water, -1 land
			{
				if (i==0 || i==fNumRows-1 || j==0 || j==fNumCols-1)
				{
					INDEXH(maskH,i*fNumCols+j) = 3;	// set outer boundary to 3
				}
				else
				{
					if (landPolyNum==1)
					{	// Land point
						INDEXH(maskH,i*fNumCols+j) = landPolyNum+3;
						landPolyNum+=3;
					}
					else
					{
						// check for nearest land poly number
						if (lowestSurroundingNum = CheckSurroundingPoints(maskH,i,j)>=3)
						{
							INDEXH(maskH,i*fNumCols+j) = lowestSurroundingNum;
						}
						else
						{
							INDEXH(maskH,i*fNumCols+j) = landPolyNum;
							landPolyNum += 1;
						}
					}
				}
			}
			else
			{
				if (i==0 || i==fNumRows-1 || j==0 || j==fNumCols-1)
					INDEXH(maskH,i*fNumCols+j) = 1;	// Open water boundary
				else if (ThereIsAdjacentLand2(maskH,velocityH,i,j))
					INDEXH(maskH,i*fNumCols+j) = 2;	// Water boundary, not open water
				else
					INDEXH(maskH,i*fNumCols+j) = 0;	// Interior water point
			}
		}
	}
	// extend grid by one row/col up/right since velocities correspond to lower left corner of a grid box
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (i==0) 
			{
				if (j!=fNumCols)
					INDEXH(maskH2,j) = INDEXH(maskH,j);	// flag for extra boundary point
				else
					INDEXH(maskH2,j) = INDEXH(maskH,j-1);	
					
			}
			else if (i!=0 && j==fNumCols) 
				INDEXH(maskH2,i*fNumCols_ext+fNumCols) = INDEXH(maskH,(i-1)*fNumCols+fNumCols-1);
			else 
			{	
				INDEXH(maskH2,i*fNumCols_ext+j) = INDEXH(maskH,(i-1)*fNumCols+j);
			}
		}
	}

	// set original top/right boundaries to interior water points 
	// probably don't need to do this since we aren't paying attention to water types anymore
	for (j=1;j<fNumCols_ext-1;j++)	 
	{
		if (INDEXH(maskH2,fNumCols_ext+j)==1) INDEXH(maskH2,fNumCols_ext+j) = 2;
	}
	for (i=1;i<fNumRows_ext-1;i++)
	{
		if (INDEXH(maskH2,i*fNumCols_ext+fNumCols-1)==1) INDEXH(maskH2,i*fNumCols_ext+fNumCols-1) = 2;
	}
	// now merge any contiguous land blocks (max of landPolyNum)
	// as soon as find one, all others of that number change, and every higher landpoint changes
	// repeat until nothing changes
startLoop:
	{
		long lowerPolyNum = 0;
		for (i=0;i<fNumRows_ext;i++)
		{
			for (j=0;j<fNumCols_ext;j++)
			{
				if (INDEXH(maskH2,i*fNumCols_ext+j) < 3) continue;	// water point
				if (ThereIsALowerLandNeighbor(maskH2,&lowerPolyNum,i,j))
				{
					ResetMaskValues(maskH2,INDEXH(maskH2,i*fNumCols_ext+j),lowerPolyNum);
					goto startLoop;
				}
				if ((i==0 || i==fNumRows_ext-1 || j==0 || j==fNumCols_ext-1) && INDEXH(maskH2,i*fNumCols_ext+j)>3)
				{	// shouldn't get here
					ResetMaskValues(maskH2,INDEXH(maskH2,i*fNumCols_ext+j),3);
					goto startLoop;
				}
			}
		}
	}
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{	// note, the numbers start at 3
			islandNum = INDEXH(maskH2,i*fNumCols_ext+j);
			if (islandNum < 3) continue;	// water point
			if (islandNum > maxIslandNum) maxIslandNum = islandNum;
		}
	}
	*islandNumberH = maskH2;
	*numIslands = maxIslandNum;
done:
	if (err) 
	{
		printError("Error numbering islands for map boundaries");
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	if (maskH) {DisposeHandle((Handle)maskH); maskH = 0;}
	return err;
}

OSErr NetCDFMoverCurv::ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long iIndex,jIndex,index,currentIndex,startIndex; 
	long triIndex1,triIndex2,waterCellNum=0;
	long ptIndex = 0,cellNum = 0,diag = 1;
	Boolean foundPt = false, isOdd;
	OSErr err = 0;

	LONGH landWaterInfo = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));

	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));

	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;

	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;

	//TTriGridVel *triGrid = nil;
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	

	// write out verdat file for debugging
	//FILE *outfile = 0;
	//char name[32], path[256],m[300];
	//strcpy(name,"NewVerdat.dat");
	//errmsg[0]=0;
	
			//err = AskUserForSaveFilename(name,path,".dat",true);
			//if(err) return USERCANCEL; 
	//strcpy(sExportSelectedTriPath, path); // remember the path for the user
	//SetWatchCursor();
	//sprintf(m, "Exporting VERDAT to %s...",path);
	//DisplayMessage("NEXTMESSAGETEMP");
	//DisplayMessage(m);
	/////////////////////////////////////////////////


	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}

	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			// eventually will need to have a land mask
			//if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
			if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				//float dLat = INDEXH(fVertexPtsH,i*fNumCols+j).pLat;
				//float dLon = INDEXH(fVertexPtsH,i*fNumCols+j).pLong;
				//long index = i*fNumCols+j+1;
				//float dZ = 1.;
				//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, dLon, dLat, dZ);	
				INDEXH(landWaterInfo,i*fNumCols+j) = 1;
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*fNumCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1) = -2;
			}
		}
	}

	//fclose(outfile);
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*fNumCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
				if (INDEXH(landWaterInfo,i*fNumCols+j)>0)
				{
					INDEXH(gridCellInfo,i*fNumCols+j).cellNum = cellNum;
					cellNum++;
					INDEXH(gridCellInfo,i*fNumCols+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols_ext+j);
					INDEXH(gridCellInfo,i*fNumCols+j).topRight = INDEXH(ptIndexHdl,i*fNumCols_ext+j+1);
					INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j);
					INDEXH(gridCellInfo,i*fNumCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1);
				}
				else INDEXH(gridCellInfo,i*fNumCols+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv_ext;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			//INDEXH(verdatPtsH,INDEXH(ptIndexHdl,i)) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/fNumCols_ext;
			jIndex = n%fNumCols_ext;
			if (iIndex==0)
			{
				if (jIndex<fNumCols)
				{
					dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
					fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
					fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
				}
				else
				{
					dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
				}
			}
			else 
			{
				if (jIndex<fNumCols)
				{
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
					u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
					v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
				}
				else
				{
					dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f,%.2f,%.2f\n", index, fLong, fLat, fDepth, u, v);	
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		/////////////////////////////////////////////////

	}
	//fclose(outfile);
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
	
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");

/////////////////////////////////////////////////
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
			}
		}
	}
	// wondering if for regular grids using the curvilinear format it would be better to have 
	// the diagonals not all be the same - it seems to cause the dagtree to not work
	// would also need to change the island numbering stuff to make sure boundary gets done correctly
	/*for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (diag>0)
			{
				if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
					{diag = diag*-1; continue;}
				waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
				triIndex1 = 2*waterCellNum;
				triIndex2 = 2*waterCellNum+1;
				// top/left tri in rect
				(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
				(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
				(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
				if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
					(*topo)[triIndex1].adjTri1 = -1;
				else
				{
					(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
				}
				(*topo)[triIndex1].adjTri2 = triIndex2;
				if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
					(*topo)[triIndex1].adjTri3 = -1;
				else
				{
					(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
				}
				// bottom/right tri in rect
				(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
				(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
				(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
				if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
					(*topo)[triIndex2].adjTri1 = -1;
				else
				{
					(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
				}
				(*topo)[triIndex2].adjTri2 = triIndex1;
				if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
					(*topo)[triIndex2].adjTri3 = -1;
				else
				{
					(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
				}
			}
			else
			{
				if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
					{diag = diag*-1; continue;}
				waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
				triIndex1 = 2*waterCellNum;
				triIndex2 = 2*waterCellNum+1;
				// bot/left tri in rect
				(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
				(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
				(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
				if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
					(*topo)[triIndex1].adjTri1 = -1;
				else
				{
					(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
				}
				(*topo)[triIndex1].adjTri2 = triIndex2;
				if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
					(*topo)[triIndex1].adjTri3 = -1;
				else
				{
					(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
				}
				// top/right tri in rect
				(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
				(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
				(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
				if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
					(*topo)[triIndex2].adjTri1 = -1;
				else
				{
					(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
				}
				(*topo)[triIndex2].adjTri2 = triIndex1;
				if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
					(*topo)[triIndex2].adjTri3 = -1;
				else
				{
					(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
				}
			}
			diag = -1.*diag;
		}
	}*/


	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	/*MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
		{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));*/
/////////////////////////////////////////////////
	if (this -> moverMap != model -> uMap) goto setFields;	// don't try to create a map
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints

	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**waterBoundaryPtsH));
	flagH = (LONGH)_NewHandleClear(nv_ext * sizeof(**flagH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Numbering Islands");
	MySpinCursor(); // JLM 8/4/99
	err = NumberIslands(&maskH2, velocityH, landWaterInfo, &numIslands);	// numbers start at 3 (outer boundary)
	MySpinCursor(); // JLM 8/4/99
	if (err) goto done;
	for (i=0;i<ntri;i++)
	{
		if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
		// the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
		// odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
		if ((*topo)[i].adjTri1 == -1)
		{
			// add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
			(*segList)[segNum].pt1 = (*topo)[i].vertex2;
			(*segList)[segNum].pt2 = (*topo)[i].vertex3;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check left rectangle for L/W border 
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (jIndex>0 && INDEXH(maskH2,iIndex*fNumCols_ext + jIndex-1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols_ext + jIndex-1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			else 
			{	
				// check right rectangle for L/W border convert back to row/col
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (jIndex<fNumCols && INDEXH(maskH2,iIndex*fNumCols_ext + jIndex+1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols_ext + jIndex+1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			segNum++;
		}
		
		if ((*topo)[i].adjTri3 == -1)
		{
			// add segment pt 1 - pt 2 to list
			// odd top, even bottom
			(*segList)[segNum].pt1 = (*topo)[i].vertex1;
			(*segList)[segNum].pt2 = (*topo)[i].vertex2;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check top rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*fNumCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*fNumCols_ext + jIndex);
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;
				}
			}
			else 
			{
				// check bottom rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (iIndex<fNumRows && INDEXH(maskH2,(iIndex+1)*fNumCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols_ext + jIndex);		// this should be the neighbor's value
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;		
				}
			}
			segNum++;
		}
	}
	nSegs = segNum;
	_SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	_SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	// go through list of segments, and make list of boundary segments
	// as segment is taken mark so only use each once
		// get a starting point, add the first and second to the list
	islandNum = 3;
findnewstartpoint:
	if (islandNum > numIslands) 
	{
		_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	}
	foundPt = false;
	for (i=0;i<nSegs;i++)
	{
		if ((*segUsed)[i]) continue;
		waterStartPoint = nBoundaryPts;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		(*flagH)[(*segList)[i].pt1] = 1;
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		(*flagH)[(*segList)[i].pt2] = 1;
		currentIndex = (*segList)[i].pt2;
		startIndex = (*segList)[i].pt1;
		currentIsland = (*segList)[i].islandNumber;	
		foundPt = true;
		(*segUsed)[i] = true;
		break;
	}
	if (!foundPt)
	{
		printNote("Lost trying to set boundaries");
		// clean up handles and set grid without a map
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		goto setFields;
	}

findnextpoint:
	for (i=0;i<nSegs;i++)
	{
		// look for second point of the previous selected segment, add the second to point list
		if ((*segUsed)[i]) continue;
		if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		index = (*segList)[i].pt1;
		if (index == currentIndex)	// found next point
		{
			currentIndex = (*segList)[i].pt2;
			(*segUsed)[i] = true;
			if (currentIndex == startIndex) // completed a segment
			{
				islandNum++;
				(*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				(*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				goto findnewstartpoint;
			}
			else
			{
				(*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				(*flagH)[(*segList)[i].pt2] = 1;
				(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				nBoundaryPts++;
				goto findnextpoint;
			}
		}
	}
	// shouldn't get here unless there's a problem...
	_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));

setFields:	

	fVerdatToNetCDFH = verdatPtsH;


	{	//for debugging
		/////////////////////////////////////////////////
		// write out the file
		/////////////////////////////////////////////////
		/*outfile=fopen(path,"w");
		if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
		fprintf(outfile,"DOGS\tMETERS\n");
	
		float fLong, fLat, fDepth = 1.0;
		long index, index1, startver, endver, count = 0;
		LongPoint vertex;

		for(i = 0; i < nEndPts; i++)
		{
			// boundary points may be used in more than one segment, this will mess up verdat 
			startver = i == 0? 0: (*boundaryEndPtsH)[i-1] + 1;
			endver = (*boundaryEndPtsH)[i]+1;
			index1 = (*boundaryPtsH)[startver];
			vertex = (*pts)[index1];
			fLong = ((float)vertex.h) / 1e6;
			fLat = ((float)vertex.v) / 1e6;
			count++;
			fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", count, fLong, fLat, fDepth);	
			for(j = startver + 1; j < endver; j++)
			{
				index = (*boundaryPtsH)[j];
				vertex = (*pts)[index];
				fLong = ((float)vertex.h) / 1e6;
				fLat = ((float)vertex.v) / 1e6;
				count++;
				fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", count, fLong, fLat, fDepth);	
			}
		}
		for (i = 0; i < numVerdatPts; i++)
		{
			if ((*flagH)[i] == 1) continue;
			count++;
			vertex = (*pts)[i];
			fLong = ((float)vertex.h) / 1e6;
			fLat = ((float)vertex.v) / 1e6;
			fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", count, fLong, fLat, fDepth);	
		}
		fprintf(outfile, "0,0.,0.,0.\n");	

		// write out the number of boundary segments
		fprintf(outfile,"%ld\n",nEndPts);
		
		// now write out out the break points
		for(i = 0; i < nEndPts; i++ )
		{
			fprintf(outfile,"%ld\n",INDEXH(boundaryEndPtsH,i)+1);
		}
		/////////////////////////////////////////////////
	
		fclose(outfile);*/
	}
/////////////////////////////////////////////////
	
	//triGrid = new TTriGridVel;
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverCurv::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}

	//fGrid = (TTriGridVel*)triGrid;
	fGrid = (TTriGridVel3D*)triGrid;

	triGrid -> SetBounds(triBounds); 

			MySpinCursor(); // JLM 8/4/99
			tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
			MySpinCursor(); // JLM 8/4/99
			if (errmsg[0])	
				{err = -1; goto done;} 
			// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
			_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it

	if (waterBoundaryPtsH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundaryEndPtsH);	
		map->SetWaterBoundaries(waterBoundaryPtsH);
		map->SetBoundaryPoints(boundaryPtsH);

		*newMap = map;
	}
	else
	{
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH=0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH=0;}
	}

	/////////////////////////////////////////////////
done:
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
		if (segList) {DisposeHandle((Handle)segList); segList = 0;}
		if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverCurv::ReorderPoints");
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
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}

		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}
	return err;
}

// simplify for codar data - no map needed, no mask 
OSErr NetCDFMoverCurv::ReorderPointsNoMask(VelocityFH velocityH, TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
	OSErr err = 0;

	LONGH landWaterInfo = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));

	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));

	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;

	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	// write out verdat file for debugging
	/*FILE *outfile = 0;
	char name[32], path[256],m[300];
	strcpy(name,"NewVerdat.dat");
	errmsg[0]=0;

			err = AskUserForSaveFilename(name,path,".dat",true);
			if(err) return USERCANCEL; 

	SetWatchCursor();
	sprintf(m, "Exporting VERDAT to %s...",path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(m);*/
	/////////////////////////////////////////////////


	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}

	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			// eventually will need to have a land mask, for now assume fillValue represents land
			if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
			// if use fill_value need to be sure to check for it in GetMove and VelocityStrAtPoint
			//if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = 1;
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*fNumCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1) = -2;
			}
		}
	}

	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*fNumCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
				if (INDEXH(landWaterInfo,i*fNumCols+j)>0)
				{
					INDEXH(gridCellInfo,i*fNumCols+j).cellNum = cellNum;
					cellNum++;
					INDEXH(gridCellInfo,i*fNumCols+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols_ext+j);
					INDEXH(gridCellInfo,i*fNumCols+j).topRight = INDEXH(ptIndexHdl,i*fNumCols_ext+j+1);
					INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j);
					INDEXH(gridCellInfo,i*fNumCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1);
				}
				else INDEXH(gridCellInfo,i*fNumCols+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv_ext;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			//INDEXH(verdatPtsH,INDEXH(ptIndexHdl,i)) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	index = 0;
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			//index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/fNumCols_ext;
			jIndex = n%fNumCols_ext;
			if (iIndex==0)
			{
				if (jIndex<fNumCols)
				{
					dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
					fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
					fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
				}
				else
				{
					dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
				}
			}
			else 
			{
				if (jIndex<fNumCols)
				{
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
					u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
					v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
				}
				else
				{
					dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth, u, v);	
		//if (u!=0. && v!=0.) {index=index+1; fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);}	
		/////////////////////////////////////////////////

	}
	//fclose(outfile);
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
	
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");

/////////////////////////////////////////////////
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
			}
		}
	}

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
		{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
/////////////////////////////////////////////////

	fVerdatToNetCDFH = verdatPtsH;

/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverCurv::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;

	triGrid -> SetBounds(triBounds); 
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it

	/////////////////////////////////////////////////
done:
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverCurv::ReorderPoints");
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
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	return err;
}

OSErr NetCDFMoverCurv::GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp)
{
	float dLat, dLon, dLat1, dLon1, dLat2, dLon2, fLat, fLong;
	
	if (iIndex<0 || jIndex>fNumCols) return -1;
	if (iIndex==0)	// along the outer top or right edge need to add on dlat/dlon
	{					// velocities at a gridpoint correspond to lower left hand corner of a grid box, draw in grid center
		if (jIndex<fNumCols)
		{
			dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
			fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
			dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
			fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
		}
		else
		{
			dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
			dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
			fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
			dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
			dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
			fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
		}
	}
	else 
	{
		if (jIndex<fNumCols)
		{
			fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
			fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
		}
		else
		{
			dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
			fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
			dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
			fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
		}
	}
	(*wp).pLat = (long)(fLat*1e6);
	(*wp).pLong = (long)(fLong*1e6);

	return noErr;
}

long NetCDFMoverCurv::GetNumDepthLevels()
{
	// should have only one version of this for all grid types, but will have to redo the regular grid stuff with depth levels
	// and check both sigma grid and multilayer grid (and maybe others)
	long numDepthLevels = 0;
	OSErr err = 0;
	char path[256], outPath[256];
	int status, ncid, sigmaid, sigmavarid;
	size_t sigmaLength=0;
	//if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; return -1;}
	}

	//if (status != NC_NOERR) {/*err = -1; goto done;*/return -1;}
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR) 
	{
		numDepthLevels = 1;	// check for zgrid option here
	}	
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "sc_r", &sigmavarid);
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {numDepthLevels = 1;}	// require variable to match the dimension
			else numDepthLevels = sigmaLength;
		}
		else
		{
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {numDepthLevels = 1;}	// error in file
		//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
		//fVar.maxNumDepths = sigmaLength;
			else numDepthLevels = sigmaLength;
		//status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
		//if (status != NC_NOERR) {err = -1; goto done;}
		// once depth is read in 
		}
	}

//done:
	return numDepthLevels;     
}


OSErr NetCDFMoverCurv::GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH)
{	// may want to input a time index to do time varying profiles for CDOG
	// what if file has u,v but not temp, sal?
	// should have only one version of this for all grid types, but will have to redo the regular grid stuff with depth levels
	// and check both sigma grid and multilayer grid (and maybe others)
	DepthValuesSetH depthLevelsH=0;
	float depth_val, lat_val, lon_val, *sigma_vals=0, *hydrodynamicField_vals=0, debugVal = 0;
	long i, j, index, numDepthLevels = 0, jIndex, iIndex;
	static size_t sigma_count;
	static size_t curr_index[] = {0,0,0,0}, depth_index[] = {0,0};
	static size_t curr_count[4];
	int curr_ucmp_id, curr_vcmp_id, temp_id, sal_id;
	char path[256], outPath[256]; 
	OSErr err = 0;
	int status, ncid, sigmaid, sigmavarid, depthid, numdims, latid, lonid;
	size_t sigmaLength=0,ptIndex=0,pt_count;
	LongPoint indices;
	//if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	else return nil;
	iIndex = indices.v;
	iIndex = fNumRows-iIndex-1;
	jIndex = indices.h;
	// really want to get the i, j values to index the file - need to be sure the lat is reordered						
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}

	// code goes here, support the s-coordinate grid (generalized sigma) used in ROMS
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR) 
	{
		err = -1;
		goto done;
	}	
	status = nc_inq_varid(ncid, "sigma", &sigmavarid); 
	if (status != NC_NOERR) {err = -1; goto done;}	// require variable to match the dimension
	status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
	if (status != NC_NOERR) {err = -1; goto done;}	// error in file
	//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
	//fVar.maxNumDepths = sigmaLength;
	sigma_vals = new float[sigmaLength];
	if (!sigma_vals) {err = memFullErr; goto done;}
	depthLevelsH = (DepthValuesSetH)_NewHandleClear(sigmaLength*sizeof(**depthLevelsH));
	if (!depthLevelsH) {err = memFullErr; goto done;}
	sigma_count = sigmaLength;
	status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	//curr_index[0] = index;	// time 
	curr_index[0] = timeIndex;	// time - for now just use first (0), eventually want a whole set
	curr_count[0] = 1;	// take one at a time
	if (numdims>=4)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		//curr_count[2] = latlength;
		//curr_count[3] = lonlength;
		curr_count[1] = sigmaLength;	// depth
		curr_count[2] = 1;
		curr_count[3] = 1;
		curr_index[2] = iIndex;	// point closest to spill lat - I think this needs to be reversed
		curr_index[3] = jIndex;	// point closest to spill lon
		depth_index[0] = iIndex;	// point closest to spill lat - I think this needs to be reversed
		depth_index[1] = jIndex;	// point closest to spill lon
	}
	else
	{
		err = -1;
		goto done;
	}

	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {err = -1; goto done;}

	//  get depth at specific point
	status = nc_get_var1_float(ncid, depthid, depth_index, &depth_val);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_varid(ncid, "lat", &latid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_varid(ncid, "lon", &lonid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {err = -1; goto done;}

	//  for testing get lat at specific point
	status = nc_get_var1_float(ncid, latid, depth_index, &lat_val);
	if (status != NC_NOERR) {err = -1; goto done;}
	//  for testing get lon at specific point
	status = nc_get_var1_float(ncid, lonid, depth_index, &lon_val);
	if (status != NC_NOERR) {err = -1; goto done;}

	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].depth = depth_val*sigma_vals[i];
	}


	// reuse the same array for u,v,temp,sal - check for w or set it to zero
	hydrodynamicField_vals = new float[sigmaLength];
	if (!hydrodynamicField_vals) {err = memFullErr; goto done;}

	status = nc_inq_varid(ncid, "U", &curr_ucmp_id);
	if (status != NC_NOERR)
	{
		status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
		if (status != NC_NOERR)
		{
			status = nc_inq_varid(ncid, "water_u", &curr_ucmp_id);
			if (status != NC_NOERR)
			{err = -1; goto done;}
		}
		//{err = -1; goto done;}
	}
	status = nc_inq_varid(ncid, "V", &curr_vcmp_id);
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "water_v", &curr_vcmp_id);
			if (status != NC_NOERR)
			{err = -1; goto done;}
		}
		//{err = -1; goto done;}
	}
	status = nc_inq_varid(ncid, "temp", &temp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "salt", &sal_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	else {if (sal_id==0) sal_id = temp_id+1;}	// for some reason returns zero rather than ten for GOM
	status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].value.u = hydrodynamicField_vals[i];
	}
	status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].value.v = hydrodynamicField_vals[i];
		(*depthLevelsH)[i].w = 0.;
	}
	status = nc_get_vara_float(ncid, temp_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].temp = hydrodynamicField_vals[i];
	}
	status = nc_get_vara_float(ncid, sal_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].sal = hydrodynamicField_vals[i];
	}
	// should check units and scale factors here (GOM uses m/s and 1.0)
	// don't think we need to check fill or missing values since model should have data everywhere there isn't a mask
	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);
	//if (status != NC_NOERR) {status = nc_get_att_float(ncid, curr_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){err = -1; goto done;}*/}	// don't require

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

done:
	if  (err) 
	{
		printError("Problem exporting hydrodynamic profile at spill site");
		if (depthLevelsH)  {DisposeHandle((Handle)depthLevelsH); depthLevelsH=0;}
	}
	if (sigma_vals) delete [] sigma_vals;
	if (hydrodynamicField_vals) delete [] hydrodynamicField_vals;

	*profilesH =  depthLevelsH;    
	return err;
}

void NetCDFMoverCurv::DrawContourScale(Rect r, WorldRect view)
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
	long indexToDepthData = 0, index;
	long numDepthLevels = GetNumDepthLevelsInFile();
	long j;
	float sc_r, sc_r2, Cs_r, Cs_r2, depthAtLevel, depthAtNextLevel;
	
	// code goes here, need separate cases for each grid type - have depth data on points, not triangles...
	long timeDataInterval;
	Boolean loaded;

	//if (fVar.gridType != SIGMA_ROMS) return;
	err = this -> SetInterval(errmsg);
	if(err) return;
	
	loaded = this -> CheckInterval(timeDataInterval);
	if(!loaded) return;	

	//if (!fDepthDataInfo) return;
	//numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);	// depth from input file (?) at triangle center
	numTris = triGrid->GetNumTriangles();

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
		return;
		//triNum = GetRandom(0,numTris-1);	// show or not show anything ?
		
	// code goes here, probably need different code for each grid type - how to select a grid box?, allow to select triangles on curvilinear grid? different for regular grid	
	//numDepths = INDEXH(fDepthDataInfo,triNum).numDepths;
	//totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	if (fGrid) 
	// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
	{
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex2(triNum,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	else return;
	//if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	//else return;
	if (fVar.gridType==SIGMA_ROMS)	// maybe always do it this way...
		totalDepth = GetTotalDepthFromTriIndex(triNum);
	else
	{
		if (fDepthsH)
		{
			totalDepth = INDEXH(fDepthsH, index);
		}
	}

	if (totalDepth==0 || numDepthLevels==0) return;
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
	//for (i=0;i<numDepthLevels;i+=istep)
	//for (i=0;i<numDepthLevels;i++)
	//if (fVar.gridType==SIGMA_ROMS)
	{
	for(j=numDepthLevels-1;j>=0;j--)	// also want j==0?
	{
		WorldPoint wp;
		Point p,p2;
		VelocityRec velocity = {0.,0.};
		Boolean offQuickDrawPlane = false;
		long depthIndex1/*, depthIndex2*/;
		Seconds time, startTime, endTime;
		double timeAlpha;
	
		//long velDepthIndex1 = (*fDepthDataInfo)[triNum].indexToDepthData+i;
		//sc_r = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					//sc_r2 = INDEXH(fDepthLevelsHdl,indexToDepthData+j-1);
		//Cs_r = INDEXH(fDepthLevelsHdl2,indexToDepthData+j);
					//Cs_r2 = INDEXH(fDepthLevelsHdl2,indexToDepthData+j-1);
		//depthAtLevel = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);	// may want this eventually
		//depthAtLevel = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
			
		if (fVar.gridType==SIGMA_ROMS)
			depthIndex1 = indexToDepthData+j;
		else
			depthIndex1 = indexToDepthData+numDepthLevels-j-1;
		//depthIndex2 = UNASSIGNEDINDEX;
					
		if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
		{
			if (index >= 0 && depthIndex1 >= 0) 
			{
				velocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				velocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
		}
		else
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime + fTimeShift;
			else
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
			//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
			time = model->GetModelTime();
			endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
			 
			if (index >= 0 && depthIndex1 >= 0) 
			{
					velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
					velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
		}
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
/*void NetCDFMoverCurv::DrawContourScale(Rect r, WorldRect view)
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

	err = this -> SetInterval(errmsg);
	if(err) return;
	
	loaded = this -> CheckInterval(timeDataInterval);
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
}*/

void NetCDFMoverCurv::Draw(Rect r, WorldRect view) 
{	// use for curvilinear
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	long depthIndex1,depthIndex2;	// default to -1?
	long amtOfDepthData = 0;
	Rect currentMapDrawingRect = MapDrawingRect();
	WorldRect cmdr;
	

	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&fColor);

	if (fDepthLevelsHdl) amtOfDepthData = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);

	if(fVar.bShowArrows || fVar.bShowGrid)
	{
		Boolean overrideDrawArrows = FALSE;
		/*if (fVar.bShowGrid) 	// make sure to draw grid even if don't draw arrows
		{
			((TTriGridVel*)fGrid)->DrawCurvGridPts(r,view);
			//return;
		}*/	// I think this is redundant with the draw triangle (maybe just a diagnostic)
		if (fVar.bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;

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
			if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
			//if(GetNumTimesInFile()>1 && loaded && !err)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
				//if (fEndData.timeIndex == UNASSIGNEDINDEX && time != startTime && fAllowExtrapolationOfCurrentsInTime)
				{
					timeAlpha = 1;
				}
				else
				{	//return false;
					endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
					timeAlpha = (endTime - time)/(double)(endTime - startTime);
				}
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			 
			cmdr = ScreenToWorldRect(currentMapDrawingRect, MapDrawingRect(), settings.currentView);	// have a look at this to see how to recognize out of view points
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex=-1,iIndex,jIndex;
				//long ptIndex2=-1,iIndex2,jIndex2;
				WorldPoint wp,wp2;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;				
				float totalDepth=0.;

				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				ptIndex = INDEXH(fVerdatToNetCDFH,i);
				
				iIndex = ptIndex/(fNumCols+1);
				jIndex = ptIndex%(fNumCols+1);
				if (iIndex>0 && jIndex<fNumCols)
					ptIndex = (iIndex-1)*(fNumCols)+jIndex;
				else
					{ptIndex = -1; continue;}

				totalDepth = GetTotalDepth(wp,ptIndex);
	 			if (amtOfDepthData>0 && ptIndex>=0)
				{
					GetDepthIndices(ptIndex,fVar.arrowDepth,totalDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					//depthIndex1 = ptIndex;
					depthIndex1 = 0;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth

				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					/*if (fDepthsH)
					{
						totalDepth = INDEXH(fDepthsH,ptIndex);
						//totalDepth = INDEXH(depthsH,i);
					}
					else 
					{
						totalDepth = 0;	// error
					}*/
					//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
					//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
					topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
					bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth);
					//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
					//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
					if (totalDepth == 0) depthAlpha = 1;
					else
						depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}
				// for now draw arrow at midpoint of diagonal of gridbox
				// this will result in drawing some arrows more than once
				if (GetLatLonFromIndex(iIndex-1,jIndex+1,&wp2)!=-1)	// may want to get all four points and interpolate
				{
					wp.pLat = (wp.pLat + wp2.pLat)/2.;
					wp.pLong = (wp.pLong + wp2.pLong)/2.;
				}
				
				if (wp.pLong < cmdr.loLong || wp.pLong > cmdr.hiLong || wp.pLat < cmdr.loLat || wp.pLat > cmdr.hiLat) 
					continue;
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	// should put velocities in center of grid box
				
				// Should check vs fFillValue
				// Check for constant current 
				if(((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha == 1) && ptIndex!=-1)
				//if(GetNumTimesInFile()==1 && ptIndex!=-1)
				{
					//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
					//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						velocity.v = GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*GetStartUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols);
						velocity.v = depthAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*GetStartVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols);
						//velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u;
						//velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v;
					}
				}
				else if (ptIndex!=-1)// time varying current
				{
					// need to rescale velocities for Navy case, store angle
					// should check for fillValue, don't want to try to interpolate in that case
					//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
					//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = timeAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						velocity.v = timeAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols));
						velocity.u += (1-depthAlpha)*(timeAlpha*GetStartUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols));
						velocity.v = depthAlpha*(timeAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols));
						velocity.v += (1-depthAlpha)*(timeAlpha*GetStartVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols));
						//velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u);
						//velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u);
						//velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v);
						//velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v);
					}
					//velocity.u = timeAlpha*GetStartUVelocity(ptIndex) + (1-timeAlpha)*GetEndUVelocity(ptIndex);
					//velocity.v = timeAlpha*GetStartVVelocity(ptIndex) + (1-timeAlpha)*GetEndVVelocity(ptIndex);
				}
				if ((velocity.u != 0 || velocity.v != 0) && (velocity.u != fFillValue && velocity.v != fFillValue)) // should already have handled fill value issue
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
	if (fVar.bShowGrid) fGrid->Draw(r,view,wayOffMapPt,fVar.curScale,fVar.arrowScale,false,true);
	if (bShowDepthContours && fVar.gridType!=TWO_D) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);// careful with 3D grid
		
	RGBForeColor(&colors[BLACK]);
}

Boolean IsTransposeArrayHeaderLine(char *s, long* numPts)
{		
	char* strToMatch = "TransposeArray";
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
/////////////////////////////////////////////////////////////////
//OSErr NetCDFMoverCurv::ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg)
OSErr ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{ // May want to combine this with read vertices if it becomes a mandatory component of PtCur files
	OSErr err=0;
	char s[64];
	long i,numScanned,index;
	LONGH verdatToNetCDFH = 0;
	
	strcpy(errmsg,""); // clear it

	verdatToNetCDFH = (LONGH)_NewHandle(sizeof(long)*numPts);
	if(!verdatToNetCDFH){TechError("NetCDFMover::ReadTransposeArray()", "_NewHandle()", 0); err = memFullErr; goto done;}

	for(i=0;i<numPts;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 64); 
		numScanned=sscanf(s,"%ld",&index) ;
		if (numScanned!= 1)
			{ err = -1; TechError("NetCDFMover::ReadTransposeArray()", "sscanf() == 1", 0); goto done; }
		(*verdatToNetCDFH)[i] = index;
	}
	*transposeArray = verdatToNetCDFH;

done:
	
	if(err) 
	{
		if(verdatToNetCDFH) {DisposeHandle((Handle)verdatToNetCDFH); verdatToNetCDFH=0;}
	}
	return err;		

}

OSErr NetCDFMoverCurv::ReadTopology(char* path, TMap **newMap)
{
	// import NetCDF curvilinear info so don't have to regenerate
	char s[1024], errmsg[256]/*, s[256], topPath[256]*/;
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
	LONGH boundarySegs=0, waterBoundaries=0, boundaryPts=0;

	//Point where;
	//OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	//MySFReply reply;

	errmsg[0]=0;
		

	/*where = CenteredDialogUpLeft(M38c);
	sfpgetfile(&where, "",
				(FileFilterUPP)0,
				-1, typeList,
				(DlgHookUPP)0,
				&reply, M38c,
				(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	if (!reply.good) 
	{
		//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
		//goto done;
		return -1;
	}
	
	my_p2cstr(reply.fName);
	
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
#else
	strcpy(topPath, reply.fName);
#endif
	
	strcpy (s, topPath);
	//err = ReadTopology(topPath,newMap);	// newMap here
	goto done;
	//SplitPathFile (s, fileName);*/


	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("NetCDFMover::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	{
		if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	
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
		if (numBoundaryPts>0)
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
	
	if(IsBoundaryPointsHeaderLine(s,&numBoundaryPts)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			err = ReadBoundaryPts(f,&line,&boundaryPts,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not always needed ? probably always needed for curvilinear
	}
	MySpinCursor(); // JLM 8/4/99

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
	
	if (waterBoundaries && waterBoundaries && boundaryPts && (this -> moverMap == model -> uMap))
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetBoundaryPoints(boundaryPts);	
		map->SetWaterBoundaries(waterBoundaries);

		*newMap = map;
	}

	//if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	else	// maybe assume rectangle grids will have map?
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts=0;}
	}

	/////////////////////////////////////////////////


	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMover::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}

	fGrid = (TTriGridVel3D*)triGrid;

	triGrid -> SetBounds(bounds); 
	//triGrid -> SetDepths(depths);

	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);

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
			strcpy(errmsg,"An error occurred in NetCDFMover::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
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
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	return err;
}

OSErr NetCDFMoverCurv::ExportTopology(char* path)
{
	// export NetCDF curvilinear info so don't have to regenerate each time
	// move to NetCDFMover so Tri can use it too
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
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
		boundaryTypeH = (dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		boundarySegmentsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		boundaryPointsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundaryPoints();
		if (!boundaryTypeH || !boundarySegmentsH || !boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
	}
	else
	{
		// any issue with trying to write out non-existent fields?
	}

	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else {printError("There is no transpose array"); err = -1; goto done;}
	sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<n;i++)
	{	
		sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

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
	//boundary points - an optional handle, only for curvilinear case

	if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]);
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	nBoundarySegs = 0;
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

	nBoundaryPts = 0;
	if (boundaryPointsH) 
	{
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	// should be same size as previous handle
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
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

/////////////////////////////////////////////////
// Triangular grid code - separate mover, could be derived from NetCDFMoverCurv or vice versa
// read in grid values for first time and set up transformation (dagtree?)
// need to read in lat/lon since won't be evenly spaced

//NetCDFMoverTri::NetCDFMoverTri (TMap *owner, char *name) : NetCDFMover(owner, name)
NetCDFMoverTri::NetCDFMoverTri (TMap *owner, char *name) : NetCDFMoverCurv(owner, name)
{
	//fVerdatToNetCDFH = 0;	
	//fVertexPtsH = 0;
	fNumNodes = 0;
	fNumEles = 0;
	bVelocitiesOnTriangles = false;
}

LongPointHdl NetCDFMoverTri::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

Boolean NetCDFMoverTri::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;

	Seconds startTime, endTime, time = model->GetModelTime();
	double topDepth, bottomDepth, depthAlpha, timeAlpha;
	VelocityRec pt1interp = {0.,0.}, pt2interp = {0.,0.}, pt3interp = {0.,0.};
	long index, amtOfDepthData = 0, triIndex;

	//long ptIndex1,ptIndex2,ptIndex3; 
	long ptIndex1=-1,ptIndex2=-1,ptIndex3=-1; 
	long pt1depthIndex1, pt1depthIndex2, pt2depthIndex1, pt2depthIndex2, pt3depthIndex1, pt3depthIndex2;
	InterpolationVal interpolationVal;

	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!fVar.bShowArrows && !fVar.bShowGrid) return 0;
	err = this -> SetInterval(errmsg);
	if(err) return false;

	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	if (!bVelocitiesOnTriangles)
	{
		interpolationVal = fGrid -> GetInterpolationValues(wp.p);
		if (interpolationVal.ptIndex1<0) return false;
	}
	else
	{
		LongPoint lp;
		//long triIndex;
		TDagTree *dagTree = 0;
		//TTriGridVel3D* triGrid = GetGrid3D(false);	
		dagTree = ((TTriGridVel3D*)fGrid) -> GetDagTree();
		if(!dagTree) return false;
		lp.h = wp.p.pLong;
		lp.v = wp.p.pLat;
		triIndex = dagTree -> WhatTriAmIIn(lp);
		interpolationVal.ptIndex1 = -1;
	}


	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
		//index1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		//index2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		//index3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
		//ptIndex1 =  (*fDepthDataInfo)[index1].indexToDepthData;
		//ptIndex2 =  (*fDepthDataInfo)[index2].indexToDepthData;
		//ptIndex3 =  (*fDepthDataInfo)[index3].indexToDepthData;
		// probably want to extend to show the velocity at level that is being shown
		if (fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 	if (amtOfDepthData>0)
	 	{
			GetDepthIndices(ptIndex1,fVar.arrowDepth,&pt1depthIndex1,&pt1depthIndex2);	
			GetDepthIndices(ptIndex2,fVar.arrowDepth,&pt2depthIndex1,&pt2depthIndex2);	
			GetDepthIndices(ptIndex3,fVar.arrowDepth,&pt3depthIndex1,&pt3depthIndex2);	
		}
		else
		{	// old version that didn't use fDepthDataInfo, must be 2D
	 		pt1depthIndex1 = ptIndex1;	pt1depthIndex2 = -1;
	 		pt2depthIndex1 = ptIndex2;	pt2depthIndex2 = -1;
	 		pt3depthIndex1 = ptIndex3;	pt3depthIndex2 = -1;
		} 
	}
	else
	{
		if (!bVelocitiesOnTriangles)
			return false;
	}

	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	{
		if (bVelocitiesOnTriangles)
		{
			pt1depthIndex1 = -1;
			pt2depthIndex1 = -1;
			pt3depthIndex1 = -1;
			if (triIndex > 0)
			{
				pt1interp.u = INDEXH(fStartData.dataHdl,triIndex).u; 
				pt1interp.v = INDEXH(fStartData.dataHdl,triIndex).v; 
			}
		}
		// Calculate the interpolated velocity at the point
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
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
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
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
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
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


		// Calculate the interpolated velocity at the point
		//if (interpolationVal.ptIndex1 >= 0) 
		/*{
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
		}*/
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		 
		if (bVelocitiesOnTriangles)
		{
			pt1depthIndex1 = -1;
			pt2depthIndex1 = -1;
			pt3depthIndex1 = -1;
			if (triIndex > 0)
			{
				pt1interp.u = INDEXH(fStartData.dataHdl,triIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).u; 
				pt1interp.v = INDEXH(fStartData.dataHdl,triIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).v; 
			}
		}

		// Calculate the interpolated velocity at the point
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
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
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
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
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
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



		// Calculate the interpolated velocity at the point
		/*if (interpolationVal.ptIndex1 >= 0) 
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
		}*/
	}
	//velocity.u *= fVar.curScale; 
	//velocity.v *= fVar.curScale; 

	velocity.u = pt1interp.u + pt2interp.u + pt3interp.u; 
	velocity.v = pt1interp.v + pt2interp.v + pt3interp.v; 

	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fVar.curScale * lengthU;

	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);

	if (interpolationVal.ptIndex1 >= 0 && ptIndex1>=0 && ptIndex2>=0 && ptIndex3>=0)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices to triangle vertices : [%ld, %ld, %ld]",
								this->className, uStr, sStr, ptIndex1, ptIndex2, ptIndex3);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
								this->className, uStr, sStr);
	}
	return true;
}

WorldPoint3D NetCDFMoverTri::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	// see PtCurMover::GetMove - will depend on what is in netcdf files and how it's stored
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depth = (*theLE).z;
	long ptIndex1,ptIndex2,ptIndex3,triIndex; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
							
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	if (!bVelocitiesOnTriangles)
		interpolationVal = fGrid -> GetInterpolationValues(refPoint);
	else
	{
		LongPoint lp;
		//long triIndex;
		TDagTree *dagTree = 0;
		//TTriGridVel3D* triGrid = GetGrid3D(false);	
		dagTree = ((TTriGridVel3D*)fGrid) -> GetDagTree();
		if(!dagTree) return deltaPoint;
		lp.h = refPoint.pLong;
		lp.v = refPoint.pLat;
		triIndex = dagTree -> WhatTriAmIIn(lp);
		interpolationVal.ptIndex1 = -1;
	}

	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
	}
	else
	{
		if (!bVelocitiesOnTriangles)
			return deltaPoint;	// set to zero, avoid any accidental access violation
	}

	// code goes here, need interpolation in z if LE is below surface
	// what kind of weird things can triangles do below the surface ??
	if (/*depth>0 &&*/ interpolationVal.ptIndex1 >= 0) 
	{
		scaledPatVelocity = GetMove3D(interpolationVal,depth);
		goto scale;
	}						
							
	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	//if(GetNumTimesInFile()==1)
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
			if (bVelocitiesOnTriangles && triIndex > 0)
			{
				scaledPatVelocity.u = INDEXH(fStartData.dataHdl,triIndex).u;
				scaledPatVelocity.v = INDEXH(fStartData.dataHdl,triIndex).v;
			}
			else
			{
				scaledPatVelocity.u = 0.;
				scaledPatVelocity.v = 0.;
			}
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
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
			if (bVelocitiesOnTriangles && triIndex > 0)
			{
				scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,triIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).u;
				scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,triIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).v;
			}
			else
			{
				scaledPatVelocity.u = 0.;
				scaledPatVelocity.v = 0.;
			}
		}
	}

scale:

	scaledPatVelocity.u *= fVar.curScale; // may want to allow some sort of scale factor, though should be in file
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
		
VelocityRec NetCDFMoverTri::GetMove3D(InterpolationVal interpolationVal,float depth)
{
	// figure out which depth values the LE falls between
	// will have to interpolate in lat/long for both levels first
	// and some sort of check on the returned indices, what to do if one is below bottom?
	// for sigma model might have different depth values at each point
	// for multilayer they should be the same, so only one interpolation would be needed
	// others don't have different velocities at different depths so no interpolation is needed
	// in theory the surface case should be a subset of this case, may eventually combine

	long pt1depthIndex1, pt1depthIndex2, pt2depthIndex1, pt2depthIndex2, pt3depthIndex1, pt3depthIndex2;
	long ptIndex1, ptIndex2, ptIndex3, amtOfDepthData = 0;
	double topDepth, bottomDepth, depthAlpha, timeAlpha;
	VelocityRec pt1interp = {0.,0.}, pt2interp = {0.,0.}, pt3interp = {0.,0.};
	VelocityRec scaledPatVelocity = {0.,0.};
	Seconds startTime, endTime, time = model->GetModelTime();

	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
	}
	else
		return scaledPatVelocity;
	
	if (fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
 	if (amtOfDepthData>0)
 	{
		GetDepthIndices(ptIndex1,depth,&pt1depthIndex1,&pt1depthIndex2);	
		GetDepthIndices(ptIndex2,depth,&pt2depthIndex1,&pt2depthIndex2);	
		GetDepthIndices(ptIndex3,depth,&pt3depthIndex1,&pt3depthIndex2);	
	}
 	else
 	{	// old version that didn't use fDepthDataInfo, must be 2D
 		pt1depthIndex1 = ptIndex1;	pt1depthIndex2 = -1;
 		pt2depthIndex1 = ptIndex2;	pt2depthIndex2 = -1;
 		pt3depthIndex1 = ptIndex3;	pt3depthIndex2 = -1;
 	}
	//GetDepthIndices(interpolationVal.ptIndex1,depth,&pt1depthIndex1,&pt1depthIndex2);	
	//GetDepthIndices(interpolationVal.ptIndex2,depth,&pt2depthIndex1,&pt2depthIndex2);	
	//GetDepthIndices(interpolationVal.ptIndex3,depth,&pt3depthIndex1,&pt3depthIndex2);	
 
 	// the contributions from each point will default to zero if the depth indicies
	// come back negative (ie the LE depth is out of bounds at the grid point)
	if(GetNumTimesInFile()==1 && !(GetNumFiles()>1) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
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
			startTime = (*fTimeHdl)[fStartData.timeIndex];
		endTime = (*fTimeHdl)[fEndData.timeIndex];
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
		
void NetCDFMoverTri::Dispose ()
{
	//if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	//if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}

	//NetCDFMover::Dispose ();
	NetCDFMoverCurv::Dispose ();
}


//#define NetCDFMoverTriREADWRITEVERSION 1 //JLM
#define NetCDFMoverTriREADWRITEVERSION 2 //JLM

OSErr NetCDFMoverTri::Write (BFPB *bfpb)
{
	long i, version = NetCDFMoverTriREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints = 0, numPts = 0, index;
	WorldPointF vertex;
	OSErr err = 0;

	//if (err = NetCDFMover::Write (bfpb)) return err;
	if (err = NetCDFMoverCurv::Write (bfpb)) return err;

	StartReadWriteSequence("NetCDFMoverTri::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////

	if (err = WriteMacValue(bfpb, fNumNodes)) goto done;
	if (err = WriteMacValue(bfpb, fNumEles)) goto done;
	if (err = WriteMacValue(bfpb, bVelocitiesOnTriangles)) goto done;
	
	/*if (fVerdatToNetCDFH) numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}

	if (fVertexPtsH) numPts = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPts)) goto done;
	for (i=0;i<numPts;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}*/

done:
	if(err)
		TechError("NetCDFMoverTri::Write(char* path)", " ", 0); 

	return err;
}

OSErr NetCDFMoverTri::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	//if (err = NetCDFMover::Read(bfpb)) return err;
	if (err = NetCDFMoverCurv::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMoverTri::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFMoverTri::Read()", "id != TYPE_NETCDFMOVERTRI", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != NetCDFMoverTriREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &fNumNodes)) goto done;	
	
	if (version>1)
	{
		if (err = ReadMacValue(bfpb, &fNumEles)) goto done;
		if (err = ReadMacValue(bfpb, &bVelocitiesOnTriangles)) goto done;
	}

	/*if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	
	if(!fVerdatToNetCDFH)
		{TechError("NetCDFMoverTri::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	
	if(!fVertexPtsH)
		{TechError("NetCDFMoverTri::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}*/
	
done:
	if(err)
	{
		TechError("NetCDFMoverTri::Read(char* path)", " ", 0); 
		//if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
		//if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFMoverTri::CheckAndPassOnMessage(TModelMessage *message)
{
	//return NetCDFMover::CheckAndPassOnMessage(message); 
	return NetCDFMoverCurv::CheckAndPassOnMessage(message); 
}

float NetCDFMoverTri::GetTotalDepth(WorldPoint refPoint, long triNum)
{
#pragma unused(refPoint)
	float totalDepth = 0;
	if (fDepthDataInfo) 
	{
		//indexToDepthData = (*fDepthDataInfo)[ptIndex].indexToDepthData;
		//numDepths = (*fDepthDataInfo)[ptIndex].numDepths;
		totalDepth = (*fDepthDataInfo)[triNum].totalDepth;
	}
	return totalDepth; // this should be an error
}
// probably eventually switch to NetCDFMover only
void NetCDFMoverTri::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData;
	long numDepths;
	float totalDepth;
	
	if (fDepthDataInfo) 
	{
		indexToDepthData = (*fDepthDataInfo)[ptIndex].indexToDepthData;
		numDepths = (*fDepthDataInfo)[ptIndex].numDepths;
		totalDepth = (*fDepthDataInfo)[ptIndex].totalDepth;
	}
	else
	return; // this should be an error
	
	switch(fVar.gridType) 
	{
		case TWO_D:	// no depth data
			*depthIndex1 = indexToDepthData;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
		case BAROTROPIC:	// values same throughout column, but limit on total depth
			if (depthAtPoint <= totalDepth)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			else
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		case MULTILAYER: //
		/*	long numDepthLevels = GetNumDepthLevelsInFile();
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{
				long j;
				for(j=0;j<numDepthLevels-1;j++)
				{
					if(INDEXH(fDepthLevelsHdl,indexToDepthData+j)<depthAtPoint &&
						depthAtPoint<=INDEXH(fDepthLevelsHdl,indexToDepthData+j+1))
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
					}
					else if(INDEXH(fDepthLevelsHdl,indexToDepthData+j)==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
					}
				}
				if(INDEXH(fDepthLevelsHdl,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			//if (*depthIndex1 != UNASSIGNEDINDEX) *depthIndex1 = ptIndex + (*depthIndex1)*fNumNodes;
			//if (*depthIndex2 != UNASSIGNEDINDEX) *depthIndex2 = ptIndex + (*depthIndex2)*fNumNodes;
			break;*/
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// if depths are measured from the bottom this is confusing
				long j;
				for(j=0;j<numDepths-1;j++)
				{
					if(INDEXH(fDepthsH,indexToDepthData+j)<depthAtPoint &&
						depthAtPoint<=INDEXH(fDepthsH,indexToDepthData+j+1))
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
					}
					else if(INDEXH(fDepthsH,indexToDepthData+j)==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
					}
				}
				if(INDEXH(fDepthsH,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
				}
				else if(INDEXH(fDepthsH,indexToDepthData+numDepths-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepths-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				}
				else if(INDEXH(fDepthsH,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		case SIGMA: // should rework the sigma to match Gnome_beta's simpler method
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{
				long j;
				for(j=0;j<numDepths-1;j++)
				{
					if(INDEXH(fDepthsH,indexToDepthData+j)<depthAtPoint &&
						depthAtPoint<=INDEXH(fDepthsH,indexToDepthData+j+1))
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
					}
					else if(INDEXH(fDepthsH,indexToDepthData+j)==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
					}
				}
				if(INDEXH(fDepthsH,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
				}
				else if(INDEXH(fDepthsH,indexToDepthData+numDepths-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepths-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				}
				else if(INDEXH(fDepthsH,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}
}

OSErr NetCDFMoverTri::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// needs to be updated once triangle grid format is set
	
	OSErr err = 0;
	long i, numScanned;
	int status, ncid, nodeid, nbndid, bndid, neleid, latid, lonid, recid, timeid, sigmaid, sigmavarid, depthid, nv_varid, nbe_varid;
	int curr_ucmp_id, uv_dimid[3], uv_ndims;
	size_t nodeLength, nbndLength, neleLength, recs, t_len, sigmaLength=0;
	float timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0, sigmaLevelsH=0;
	float *lat_vals=0,*lon_vals=0,*depth_vals=0, *sigma_vals=0;
	//short *bndry_indices=0, *bndry_nums=0, *bndry_type=0;
	long *bndry_indices=0, *bndry_nums=0, *bndry_type=0, *top_verts=0, *top_neighbors=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0,bndIndex[2]={0,0};
	static size_t pt_count, bnd_count[2], sigma_count,topIndex[2]={0,0}, top_count[2];
	Seconds startTime, startTime2;
	double timeConversion = 1., scale_factor = 1.;
	char errmsg[256] = "";
	char fileName[64],s[256],topPath[256], outPath[256];

	char *modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false;

	if (!path || !path[0]) return 0;
	strcpy(fVar.pathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fVar.userName, fileName); // maybe use a name from the file

	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	/*status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) 
		//{err = -1; goto done;}
	{
		status = nc_inq_dimid(ncid, "time", &recid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
	}*/
	status = nc_inq_dimid(ncid, "time", &recid); 
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
		if (status != NC_NOERR) {err = -1; goto done;}
	}

	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {err = -1; goto done;} 

	//status = nc_inq_attlen(ncid, recid, "units", &t_len);
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		//err = -1; goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];
		
		timeUnits = new char[t_len+1];
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);// recid is the dimension id not the variable id
		status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
					  unitStr, junk, &time.year, &time.month, &time.day,
					  &time.hour, &time.minute, &time.second) ;
		if (numScanned==7) // has two extra time entries ??	
			time.second = 0;
		else if (numScanned<8) // has two extra time entries ??	
		//if (numScanned<8) // has two extra time entries ??	
			{ err = -1; TechError("NetCDFMoverTri::TextRead()", "sscanf() == 8", 0); goto done; }
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
	//bndry_indices = new short[nbndLength]; 
	bndry_indices = new long[nbndLength]; 
	//bndry_nums = new short[nbndLength]; 
	//bndry_type = new short[nbndLength]; 
	bndry_nums = new long[nbndLength]; 
	bndry_type = new long[nbndLength]; 
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//bndIndex[1] = 0;
	bndIndex[1] = 1;	// take second point of boundary segments instead, so that water boundaries work out
	//status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 2;
	//status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 3;
	//status = nc_get_vara_short(ncid, bndid, bndIndex, bnd_count, bndry_type);
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_type);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//status = nc_inq_dimid(ncid, "nele", &neleid);	
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required
	//status = nc_inq_dimlen(ncid, neleid, &neleLength);
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required

	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "zloc", &sigmaid); 	
		if (status != NC_NOERR) 
		{
			fVar.gridType = TWO_D; /*err = -1; goto done;*/
		}
		else
		{	// might change names to depth rather than sigma here
			status = nc_inq_varid(ncid, "zloc", &sigmavarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			fVar.gridType = MULTILAYER;
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			//sigmaLevelsH = (FLOATH)_NewHandleClear(sigmaLength*sizeof(sigmaLevelsH));
			//if (!sigmaLevelsH) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
			fDepthLevelsHdl = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
			if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
			for (i=0;i<sigmaLength;i++)
			{
				INDEXH(fDepthLevelsHdl,i) = (float)sigma_vals[i];
			}
			fNumDepthLevels = sigmaLength;	//  here also do we want all depths?
			// once depth is read in 
		}
	}	// check for zgrid option here
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		fVar.gridType = SIGMA;
		fVar.maxNumDepths = sigmaLength;
		sigma_vals = new float[sigmaLength];
		if (!sigma_vals) {err = memFullErr; goto done;}
		//sigmaLevelsH = (FLOATH)_NewHandleClear(sigmaLength*sizeof(sigmaLevelsH));
		//if (!sigmaLevelsH) {err = memFullErr; goto done;}
		sigma_count = sigmaLength;
		status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
		if (status != NC_NOERR) {err = -1; goto done;}
		// once depth is read in 
	}

	// option to use index values?
	status = nc_inq_varid(ncid, "lat", &latid);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "lon", &lonid);
	if (status != NC_NOERR) {err = -1; goto done;}

	pt_count = nodeLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(nodeLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	lat_vals = new float[nodeLength]; 
	lon_vals = new float[nodeLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_float(ncid, latid, &ptIndex, &pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, lonid, &ptIndex, &pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {fVar.gridType = TWO_D;/*err = -1; goto done;*/}
	else
	{	
		totalDepthsH = (FLOATH)_NewHandleClear(nodeLength*sizeof(float));
		if (!totalDepthsH) {err = memFullErr; goto done;}
		depth_vals = new float[nodeLength];
		if (!depth_vals) {err = memFullErr; goto done;}
		status = nc_get_vara_float(ncid, depthid, &ptIndex, &pt_count, depth_vals);
		if (status != NC_NOERR) {err = -1; goto done;}

		status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor

	}

	for (i=0;i<nodeLength;i++)
	{
		INDEXH(vertexPtsH,i).pLat = lat_vals[i];	
		INDEXH(vertexPtsH,i).pLong = lon_vals[i];
	}
	fVertexPtsH	 = vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon

	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		//newTime = startTime2+timeVal*timeConversion;
		INDEXH(fTimeHdl,i) = newTime;	// which start time where?
		if (i==0) startTime = newTime + fTimeShift;
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion + fTimeShift;
	}
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{	// might want to move this so time doesn't get changed if user cancels or there is an error
			short buttonSelected;
			//buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			//if(!gCommandFileErrorLogPath[0])
			if(!gCommandFileRun)	// also may want to skip for location files...
				buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
			switch(buttonSelected){
				case 1: // reset model start time
					//bTopFile = true;
					model->SetModelTime(startTime);
					model->SetStartTime(startTime);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
					break;  
				case 3: // don't reset model start time
					//bTopFile = false;
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
		//model->SetModelTime(startTime);
		//model->SetStartTime(startTime);
		//model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
	}

	fNumNodes = nodeLength;

	// check if file has topology in it
	{
		status = nc_inq_varid(ncid, "nv", &nv_varid); //Navy
		if (status != NC_NOERR) {/*err = -1; goto done;*/}
		else
		{
			status = nc_inq_varid(ncid, "nbe", &nbe_varid); //Navy
			if (status != NC_NOERR) {/*err = -1; goto done;*/}
			else bTopFile = true;
		}
		if (bTopFile)
		{
			status = nc_inq_dimid(ncid, "nele", &neleid);	
			if (status != NC_NOERR) {err = -1; goto done;}	
			status = nc_inq_dimlen(ncid, neleid, &neleLength);
			if (status != NC_NOERR) {err = -1; goto done;}	
			fNumEles = neleLength;
			top_verts = new long[neleLength*3]; 
			if (!top_verts ) {err = memFullErr; goto done;}
			top_neighbors = new long[neleLength*3]; 
			if (!top_neighbors ) {err = memFullErr; goto done;}
			top_count[0] = 3;
			top_count[1] = neleLength;
			status = nc_get_vara_long(ncid, nv_varid, topIndex, top_count, top_verts);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_vara_long(ncid, nbe_varid, topIndex, top_count, top_neighbors);
			if (status != NC_NOERR) {err = -1; goto done;}

			//determine if velocities are on triangles
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
			if (status != NC_NOERR) {err = -1; goto done;}

			status = nc_inq_vardimid (ncid, curr_ucmp_id, uv_dimid);	// see if dimid(1) or (2) == nele or node, depends on uv_ndims
			if (status==NC_NOERR) 
			{
				if (uv_ndims == 3 && uv_dimid[2] == neleid)
					{bVelocitiesOnTriangles = true;}
				else if (uv_ndims == 2 && uv_dimid[1] == neleid)
					{bVelocitiesOnTriangles = true;}
			}

			// fill topology handle with data, makedagtree, boundaries
			//err = ReorderPoints2(/*newMap,*/top_verts,top_neighbors,neleLength/*bndry_indices,bndry_nums,bndry_type,nbndLength*/);	 
			err = ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);	 
			//err = ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			if (err) goto done;
			goto depths;
		}
	}

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	//err = this -> SetInterval(errmsg);
	//if(err) goto done;

	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	//ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 


	//{if (topFilePath[0]) {strcpy(fTopFilePath,topFilePath); err = ReadTopology(fTopFilePath,newMap); goto done;}}
	//{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto done;}}
	{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto depths;}}
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	//if (/*fIsNavy*/true)
	if (!bTopFile)
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
		//if (!reply.good) return USERCANCEL;
		if (!reply.good) /*return 0;*/
		{
			err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto depths;
		}
		else
			strcpy(topPath, reply.fullPath);

		/*{
			err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto depths;
		}*/
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
			err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
			//goto done;	
			goto depths;	
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
		goto depths;
		//goto done;
		//SplitPathFile (s, fileName);
	}

	err = ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
		
depths:
	if (err) goto done;
	// also translate to fDepthDataInfo and fDepthsH here, using sigma or zgrid info
	
	if (totalDepthsH)
	{
		for (i=0; i<fNumNodes; i++)
		{
			long n;
			
			//n = INDEXH(fVerdatToNetCDFH,i);
			n = i;
			if (n<0 || n>= fNumNodes) {printError("indices messed up"); err=-1; goto done;}
			INDEXH(totalDepthsH,i) = depth_vals[n] * scale_factor;
		}
		//((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}

	// CalculateVerticalGrid(sigmaLength,sigmaLevelsH,totalDepthsH);	// maybe multigrid
	{
		long j,index = 0;
		fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**fDepthDataInfo)*fNumNodes);
		if(!fDepthDataInfo){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		//if (fVar.gridType==TWO_D || fVar.gridType==MULTILAYER) 
		if (fVar.gridType==TWO_D) 
			{if (totalDepthsH) fDepthsH = totalDepthsH;}	// may be null, call it barotropic if depths exist??
		// assign arrays
		else
		{	//TWO_D grid won't need fDepthsH
			fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumNodes*fVar.maxNumDepths);
			if(!fDepthsH){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
		}
		// code goes here, if velocities on triangles need to interpolate total depth I think, or use this differently
		for (i=0;i<fNumNodes;i++)
		{
			// might want to order all surface depths, all sigma1, etc., but then indexToDepthData wouldn't work
			// have 2D case, zgrid case as well
			if (fVar.gridType==TWO_D)
			{
				if (totalDepthsH) (*fDepthDataInfo)[i].totalDepth = (*totalDepthsH)[i];
				else (*fDepthDataInfo)[i].totalDepth = -1;	// no depth data
				(*fDepthDataInfo)[i].indexToDepthData = i;
				(*fDepthDataInfo)[i].numDepths = 1;
			}
			/*else if (fVar.gridType==MULTILAYER)
			{
				if (totalDepthsH) (*fDepthDataInfo)[i].totalDepth = (*totalDepthsH)[i];
				else (*fDepthDataInfo)[i].totalDepth = -1;	// no depth data, this should be an error I think
				(*fDepthDataInfo)[i].indexToDepthData = 0;
				(*fDepthDataInfo)[i].numDepths = sigmaLength;
			}*/
			else
			{
				(*fDepthDataInfo)[i].totalDepth = (*totalDepthsH)[i];
				(*fDepthDataInfo)[i].indexToDepthData = index;
				(*fDepthDataInfo)[i].numDepths = sigmaLength;
				for (j=0;j<sigmaLength;j++)
				{
					//(*fDepthsH)[index+j] = (*totalDepthsH)[i]*(1-(*sigmaLevelsH)[j]);
					//if (fVar.gridType==MULTILAYER) (*fDepthsH)[index+j] = (*totalDepthsH)[i]*(j);	// check this
					if (fVar.gridType==MULTILAYER) /*(*fDepthsH)[index+j] = (sigma_vals[j]);*/	// check this, measured from the bottom
					// since depth is measured from bottom should recalculate the depths for each point
					{
						if (( (*totalDepthsH)[i] - sigma_vals[sigmaLength - j - 1]) >= 0) 
							(*fDepthsH)[index+j] = (*totalDepthsH)[i] - sigma_vals[sigmaLength - j - 1] ; 
						else (*fDepthsH)[index+j] = (*totalDepthsH)[i]+1;
					}
					else (*fDepthsH)[index+j] = (*totalDepthsH)[i]*(1-sigma_vals[j]);
					//(*fDepthsH)[j*fNumNodes+i] = totalDepthsH[i]*(1-sigmaLevelsH[j]);
				}
				index+=sigmaLength;
			}
		}
	}
	if (totalDepthsH)	// why is this here twice?
	{
		for (i=0; i<fNumNodes; i++)
		{
			long n;
			
			n = INDEXH(fVerdatToNetCDFH,i);
			if (n<0 || n>= fNumNodes) {printError("indices messed up"); err=-1; goto done;}
			INDEXH(totalDepthsH,i) = depth_vals[n] * scale_factor;
		}
		((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}

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
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
		if(sigmaLevelsH) {DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH = 0;}
	}
	//printNote("NetCDF triangular grid model current mover is not yet implemented");

	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (sigma_vals) delete [] sigma_vals;
	if (bndry_indices) delete [] bndry_indices;
	if (bndry_nums) delete [] bndry_nums;
	if (bndry_type) delete [] bndry_type;

	return err;
}
	 

OSErr NetCDFMoverTri::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{	// - needs to be updated once triangle grid format is set
	OSErr err = 0;
	long i,j;
	char path[256], outPath[256]; 
	int status, ncid, numdims, uv_ndims;
	int curr_ucmp_id, curr_vcmp_id, uv_dimid[3], nele_id;
	//static size_t curr_index[] = {0,0,0};
	//static size_t curr_count[3];
	static size_t curr_index[] = {0,0,0,0};
	static size_t curr_count[4];
	float *curr_uvals,*curr_vvals, fill_value, dry_value = 0;
	long totalNumberOfVels = fNumNodes * fVar.maxNumDepths, numVelsAtDepthLevel=0;
	VelocityFH velH = 0;
	long numNodes = fNumNodes;
	long numTris = fNumEles;
	long numDepths = fVar.maxNumDepths;	// assume will always have full set of depths at each point for now
	
	errmsg[0]=0;

	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;

	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_ndims(ncid, &numdims);	// in general it's not the total number of dimensions but the number the variable depends on
	if (status != NC_NOERR) {err = -1; goto done;}

	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	//curr_count[1] = 1;	// depth
	//curr_count[2] = numNodes;

	// check for sigma or zgrid dimension
	if (numdims>=6)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		curr_count[1] = numDepths;	// depth
		//curr_count[1] = depthlength;	// depth
		curr_count[2] = numNodes;
	}
	else
	{
		curr_count[1] = numNodes;	
	}
	status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
	if (status==NC_NOERR){if (numdims < 6 && uv_ndims==3) {curr_count[1] = numDepths; curr_count[2] = numNodes;}}	// could have more dimensions than are used in u,v

	status = nc_inq_vardimid (ncid, curr_ucmp_id, uv_dimid);	// see if dimid(1) or (2) == nele or node, depends on uv_ndims
	if (status==NC_NOERR) 
	{
		status = nc_inq_dimid (ncid, "nele", &nele_id);
		if (status==NC_NOERR)
		{
			if (uv_ndims == 3 && uv_dimid[2] == nele_id)
				{bVelocitiesOnTriangles = true; curr_count[2] = numTris;}
			else if (uv_ndims == 2 && uv_dimid[1] == nele_id)
				{bVelocitiesOnTriangles = true; curr_count[1] = numTris;}
		}
	}
	if (bVelocitiesOnTriangles) 
	{
		totalNumberOfVels = numTris * fVar.maxNumDepths;
		numVelsAtDepthLevel = numTris;
	}
	else
		numVelsAtDepthLevel = numNodes;
	//curr_uvals = new float[numNodes]; 
	curr_uvals = new float[totalNumberOfVels]; 
	if(!curr_uvals) {TechError("NetCDFMoverTri::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	//curr_vvals = new float[numNodes]; 
	curr_vvals = new float[totalNumberOfVels]; 
	if(!curr_vvals) {TechError("NetCDFMoverTri::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}

	status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);// missing_value vs _FillValue
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_att_float(ncid, curr_ucmp_id, "missing_value", &fill_value);// missing_value vs _FillValue
	if (status != NC_NOERR) {/*err = -1; goto done;*/fill_value=-9999.;}
	status = nc_get_att_float(ncid, curr_ucmp_id, "dry_value", &dry_value);// missing_value vs _FillValue
	if (status != NC_NOERR) {/*err = -1; goto done;*/}  
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) {err = memFullErr; goto done;}
	for (j=0;j<numDepths;j++)
	{
	//for (i=0;i<totalNumberOfVels;i++)
	for (i=0;i<numVelsAtDepthLevel;i++)
	//for (i=0;i<numNodes;i++)
	{
		// really need to store the fill_value data and check for it when moving or drawing
		/*if (curr_uvals[i]==0.||curr_vvals[i]==0.)
			curr_uvals[i] = curr_vvals[i] = 1e-06;
		if (curr_uvals[i]==fill_value)
			curr_uvals[i]=0.;
		if (curr_vvals[i]==fill_value)
			curr_vvals[i]=0.;
		// for now until we decide what to do with the dry value flag
		if (curr_uvals[i]==dry_value)
			curr_uvals[i]=0.;
		if (curr_vvals[i]==dry_value)
			curr_vvals[i]=0.;
		INDEXH(velH,i).u = curr_uvals[i];	// need units
		INDEXH(velH,i).v = curr_vvals[i];*/
		/*if (curr_uvals[j*fNumNodes+i]==0.||curr_vvals[j*fNumNodes+i]==0.)
			curr_uvals[j*fNumNodes+i] = curr_vvals[j*fNumNodes+i] = 1e-06;
		if (curr_uvals[j*fNumNodes+i]==fill_value)
			curr_uvals[j*fNumNodes+i]=0.;
		if (curr_vvals[j*fNumNodes+i]==fill_value)
			curr_vvals[j*fNumNodes+i]=0.;*/
		if (curr_uvals[j*numVelsAtDepthLevel+i]==0.||curr_vvals[j*numVelsAtDepthLevel+i]==0.)
			curr_uvals[j*numVelsAtDepthLevel+i] = curr_vvals[j*numVelsAtDepthLevel+i] = 1e-06;
		if (curr_uvals[j*numVelsAtDepthLevel+i]==fill_value)
			curr_uvals[j*numVelsAtDepthLevel+i]=0.;
		if (curr_vvals[j*numVelsAtDepthLevel+i]==fill_value)
			curr_vvals[j*numVelsAtDepthLevel+i]=0.;
		//if (fVar.gridType==MULTILAYER /*sigmaReversed*/)
		/*{
			INDEXH(velH,(numDepths-j-1)*fNumNodes+i).u = curr_uvals[j*fNumNodes+i];	// need units
			INDEXH(velH,(numDepths-j-1)*fNumNodes+i).v = curr_vvals[j*fNumNodes+i];	// also need to reverse top to bottom (if sigma is reversed...)
		}
		else*/
		{
			//INDEXH(velH,i*numDepths+(numDepths-j-1)).u = curr_uvals[j*fNumNodes+i];	// need units
			//INDEXH(velH,i*numDepths+(numDepths-j-1)).v = curr_vvals[j*fNumNodes+i];	// also need to reverse top to bottom
			INDEXH(velH,i*numDepths+(numDepths-j-1)).u = curr_uvals[j*numVelsAtDepthLevel+i];	// need units
			INDEXH(velH,i*numDepths+(numDepths-j-1)).v = curr_vvals[j*numVelsAtDepthLevel+i];	// also need to reverse top to bottom
		}
	}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading current data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) delete [] curr_uvals;
	if (curr_vvals) delete [] curr_vvals;
	return err;
}

//OSErr NetCDFMoverTri::ReorderPoints2(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts) 
OSErr NetCDFMoverTri::ReorderPoints2(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = fNumNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;

	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));

	TopologyHdl topo=0;
	DAGTreeStruct tree;

	tree.treeHdl = 0;
	TDagTree *dagTree = 0;

	LongPointHdl pts=0;
	VelocityFH velH = 0;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	LONGH boundaryPtsH = 0;

	TTriGridVel3D *triGrid = nil;
	
	Boolean addOne = false;	// for debugging

	// write out verdat file for debugging
	/*FILE *outfile = 0;
	char name[32], path[256],m[300];
	SFReply reply;
	Point where = CenteredDialogUpLeft(M55);
	char ibmBackwardsTypeStr[32] = "";
	strcpy(name,"NewVerdat.dat");
	errmsg[0]=0;
	
 #ifdef MAC
		sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
 #else
		sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	           M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
 #endif
	if (!reply.good) {err = -1; goto done;}

	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath (reply.vRefNum, 0, (char *) "", path);
	strcat (path, ":");
	strcat (path, (char *) reply.fName);
#else
	strcpy(path, reply.fName);
#endif
	//strcpy(sExportSelectedTriPath, path); // remember the path for the user
	SetWatchCursor();
	sprintf(m, "Exporting VERDAT to %s...",path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(m);*/
	/////////////////////////////////////////////////


	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list

	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		//short islandNum, index;
		long islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
		//INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;

	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// shrink handle
		_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}

	numVerdatPts = nv;	//for now, may reorder later
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	
	//numVerdatPts = nv;	//for now, may reorder later
	for (i=0; i<=numVerdatPts; i++)
	{
		//long index;
		float fLong, fLat/*, fDepth*/;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			//index = i+1;
			//n = INDEXH(verdatPtsH,i);
			n = i;	// for now, not sure if need to reorder
			fLat = INDEXH(fVertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(fVertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			//fDepth = 1.;	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
		}
		else { // the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
	
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// write out the number of chains
	//fprintf(outfile,"%ld\n",numVerdatBreakPts);
	
	// now write out out the break points
	/*for(i = 0; i < numVerdatBreakPts; i++ )
	{
		fprintf(outfile,"%ld\n",INDEXH(verdatBreakPtsH,i));
	}*/
	/////////////////////////////////////////////////

	//fclose(outfile);
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	//if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology))))goto done;	

	// point and triangle indices should start with zero
	for(i = 0; i < 3*ntri; i ++)
	{
		/*if (tri_neighbors[i]==0)
			tri_neighbors[i]=-1;
		else */
			tri_neighbors[i] = tri_neighbors[i] - 1;
		tri_verts[i] = tri_verts[i] - 1;
	}
	for(i = 0; i < ntri; i ++)
	{	// topology data needs to be CCW
		long debugTest = tri_verts[i];
		(*topo)[i].vertex1 = tri_verts[i];
		debugTest = tri_verts[i+ntri];
		//(*topo)[i].vertex2 = tri_verts[i+ntri];
		(*topo)[i].vertex3 = tri_verts[i+ntri];
		debugTest = tri_verts[i+2*ntri];
		//(*topo)[i].vertex3 = tri_verts[i+2*ntri];
		(*topo)[i].vertex2 = tri_verts[i+2*ntri];
		debugTest = tri_neighbors[i];
		(*topo)[i].adjTri1 = tri_neighbors[i];
		debugTest = tri_neighbors[i+ntri];
		//(*topo)[i].adjTri2 = tri_neighbors[i+ntri];
		(*topo)[i].adjTri3 = tri_neighbors[i+ntri];
		debugTest = tri_neighbors[i+2*ntri];
		//(*topo)[i].adjTri3 = tri_neighbors[i+2*ntri];
		(*topo)[i].adjTri2 = tri_neighbors[i+2*ntri];
	}

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
		{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}

	//fGrid = (TTriGridVel*)triGrid;
	fGrid = (TTriGridVel3D*)triGrid;

	triGrid -> SetBounds(triBounds); 

	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	//if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);	// should be set in TextRead

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	boundaryPtsH = (LONGH)_NewHandleClear(numBoundaryPts * sizeof(**boundaryPtsH));
	if (!boundaryPtsH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
		INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}

	if (waterBoundariesH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(verdatBreakPtsH);	
		map->SetWaterBoundaries(waterBoundariesH);
		map->SetBoundaryPoints(boundaryPtsH);

		*newMap = map;
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}

	/////////////////////////////////////////////////
	//fVerdatToNetCDFH = verdatPtsH;	// this should be resized

done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReorderPoints");
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
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	return err;
}

//OSErr NetCDFMoverTri::ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts) 
OSErr NetCDFMoverTri::ReorderPoints(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = fNumNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;

	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));

	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;

	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	Boolean addOne = false;	// for debugging

	// write out verdat file for debugging
	/*FILE *outfile = 0;
	char name[32], path[256],m[300];
	SFReply reply;
	Point where = CenteredDialogUpLeft(M55);
	char ibmBackwardsTypeStr[32] = "";
	strcpy(name,"NewVerdat.dat");
	errmsg[0]=0;
	
 #ifdef MAC
		sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
 #else
		sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	           M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
 #endif
	if (!reply.good) {err = -1; goto done;}

	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath (reply.vRefNum, 0, (char *) "", path);
	strcat (path, ":");
	strcat (path, (char *) reply.fName);
#else
	strcpy(path, reply.fName);
#endif
	//strcpy(sExportSelectedTriPath, path); // remember the path for the user
	SetWatchCursor();
	sprintf(m, "Exporting VERDAT to %s...",path);
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage(m);*/
	/////////////////////////////////////////////////


	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list

	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		//short islandNum, index;
		long islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;

	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// shrink handle
		_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	
	for (i=0; i<=numVerdatPts; i++)
	{
		long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			fLat = INDEXH(fVertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(fVertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
		}
		else { // the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
	
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// write out the number of chains
	//fprintf(outfile,"%ld\n",numVerdatBreakPts);
	
	// now write out out the break points
	/*for(i = 0; i < numVerdatBreakPts; i++ )
	{
		fprintf(outfile,"%ld\n",INDEXH(verdatBreakPtsH,i));
	}*/
	/////////////////////////////////////////////////

	//fclose(outfile);
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
	//if (err = maketriangles2(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts,verdatPtsH,fNumCols_ext))
		goto done;

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
		{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}

	//fGrid = (TTriGridVel*)triGrid;
	fGrid = (TTriGridVel3D*)triGrid;

	triGrid -> SetBounds(triBounds); 

	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
	}

	if (waterBoundariesH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(verdatBreakPtsH);	
		map->SetWaterBoundaries(waterBoundariesH);

		*newMap = map;
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
	}

	/////////////////////////////////////////////////
	fVerdatToNetCDFH = verdatPtsH;	// this should be resized

done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReorderPoints");
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
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	}
	return err;
}

long NetCDFMoverTri::GetNumDepthLevels()
{
	// should have only one version of this for all grid types, but will have to redo the regular grid stuff with depth levels
	// and check both sigma grid and multilayer grid (and maybe others)
	long numDepthLevels = 0;
	OSErr err = 0;
	char path[256], outPath[256];
	int status, ncid, sigmaid, sigmavarid;
	size_t sigmaLength=0;
	//if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {/*err = -1; goto done;*/return -1;}
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; return -1;}
	}
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR) 
	{
		numDepthLevels = 1;	// check for zgrid option here
	}	
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) {numDepthLevels = 1;}	// require variable to match the dimension
		status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
		if (status != NC_NOERR) {numDepthLevels = 1;}	// error in file
		//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
		//fVar.maxNumDepths = sigmaLength;
		numDepthLevels = sigmaLength;
		//status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
		//if (status != NC_NOERR) {err = -1; goto done;}
		// once depth is read in 
	}

//done:
	return numDepthLevels;     
}

void NetCDFMoverTri::Draw(Rect r, WorldRect view) 
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
	long amtOfDepthData = 0;

	RGBForeColor(&fColor);

	if(fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	
	if(fGrid && (fVar.bShowArrows || fVar.bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,fVar.curScale,fVar.arrowScale,overrideDrawArrows,fVar.bShowGrid);
		if(fVar.bShowArrows && bVelocitiesOnTriangles == false)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't need 3D stuff to draw here

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
			//if(GetNumTimesInFile()>1)
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
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			 
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				//long ptIndex = INDEXH(fVerdatToNetCDFH,i);
				long index = INDEXH(fVerdatToNetCDFH,i);
				//long ptIndex = (*fDepthDataInfo)[index].indexToDepthData;	// not used ?
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case

				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				//amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 			if (amtOfDepthData>0)
				{
					GetDepthIndices(index,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					depthIndex1 = index;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth

				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}

				/*if (fVar.gridType == MULTILAYER)
				{
					if (fDepthLevelsHdl) 
					{
						topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
						bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
						depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
					}
					//else //this should be an error
					depthIndex1 = index + depthIndex1*fNumNodes;
					if (depthIndex2!=UNASSIGNEDINDEX) depthIndex2 = index + depthIndex2*fNumNodes;
				}*/

				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
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
		else if (fVar.bShowArrows && bVelocitiesOnTriangles)
		{ // we have to draw the arrows
			short row, col, pixX, pixY;
			float inchesX, inchesY;
			Point p, p2;
			Rect c;
			WorldPoint wp;
			VelocityRec velocity;
			LongPoint wp1,wp2,wp3;
			Boolean offQuickDrawPlane = false;
			long numVertices,i,numTri;
			LongPointHdl ptsHdl = 0;
			TopologyHdl topH = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't need 3D stuff to draw here

			err = this -> SetInterval(errmsg);
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval);
			if(!loaded) return;

			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			topH = triGrid -> GetTopologyHdl();
			if (topH)
				numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
			else 
				numTri = 0;

			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
			//if(GetNumTimesInFile()>1)
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
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			 
			//for(i = 0; i < numVertices; i++)
			for(i = 0; i < numTri; i++)
			{
			 	// get the value at each vertex and draw an arrow
				//LongPoint pt = INDEXH(ptsHdl,i);
				//long index = INDEXH(fVerdatToNetCDFH,i);
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case

				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				//amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 			if (amtOfDepthData>0 && !bVelocitiesOnTriangles)	// for now, will have to figure out how depth data is handled
				{
					//GetDepthIndices(index,fVar.arrowDepth,&depthIndex1,&depthIndex2);
					GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					//depthIndex1 = index;
					depthIndex1 = i;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth

				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}

				wp1 = (*ptsHdl)[(*topH)[i].vertex1];
				wp2 = (*ptsHdl)[(*topH)[i].vertex2];
				wp3 = (*ptsHdl)[(*topH)[i].vertex3];
		
				wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
				wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
				//velocity = GetPatValue(wp);

				/*if (fVar.gridType == MULTILAYER)
				{
					if (fDepthLevelsHdl) 
					{
						topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
						bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
						depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
					}
					//else //this should be an error
					depthIndex1 = index + depthIndex1*fNumNodes;
					if (depthIndex2!=UNASSIGNEDINDEX) depthIndex2 = index + depthIndex2*fNumNodes;
				}*/

				//wp.pLat = pt.v;
				//wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = INDEXH(fStartData.dataHdl,i).u;
						velocity.v = INDEXH(fStartData.dataHdl,i).v;
						//velocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u;
						//velocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v;
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
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,i).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,i).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,i).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,i).v;
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
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
					inchesX = (velocity.u * fVar.curScale) / fVar.arrowScale;
					inchesY = (velocity.v * fVar.curScale) / fVar.arrowScale;
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
	}
	if (bShowDepthContours && fVar.gridType!=TWO_D) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
		
	RGBForeColor(&colors[BLACK]);
}

OSErr NetCDFMoverTri::ReadTopology(char* path, TMap **newMap)
{
	// import NetCDF triangle info so don't have to regenerate
	// this is same as curvilinear mover so may want to combine later
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
		TechError("NetCDFMoverTri::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	{
		if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}

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
	
	if (waterBoundaries && boundarySegs && (this -> moverMap == model -> uMap))
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
		TechError("Error in NetCDFMoverTri::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}

	fGrid = (TTriGridVel3D*)triGrid;

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

/////////////////////////////////////////////////
PtCurMap* GetPtCurMap(void)
{
	long i,n;
	TMap *map;
	PtCurMap *ptCurMap = 0;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_COMPOUNDMAP))
		{
			return dynamic_cast<PtCurMap *>(map);
		} 
		if (map->IAm(TYPE_PTCURMAP)) 
		{
			ptCurMap = dynamic_cast<PtCurMap *>(map);
			return ptCurMap;
			//return (PtCurMap*)map;
		}
	}
	return nil;
}

OSErr NetCDFMoverTri::ExportTopology(char* path)
{
	// export NetCDF triangle info so don't have to regenerate each time
	// same as curvilinear so may want to combine at some point
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
	PtCurMap *map = GetPtCurMap();

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
	
	//if (moverMap->IAm(TYPE_PTCURMAP))
	if (map)
	{
		//boundaryTypeH = ((PtCurMap*)moverMap)->GetWaterBoundaries();
		//boundarySegmentsH = ((PtCurMap*)moverMap)->GetBoundarySegs();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
		{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
		{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }


	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else {printError("There is no transpose array"); err = -1; goto done;}
	sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<n;i++)
	{	
		sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}

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
}
