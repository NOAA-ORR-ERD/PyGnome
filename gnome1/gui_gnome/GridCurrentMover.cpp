#include "Earl.h"
#include "TypeDefs.h"
#include "Cross.h"
#include "Uncertainty.h"
#include "GridVel.h"
#include "GridCurrentMover.h"
#include "DagTreeIO.h"
#include "TimeGridVel.h"


static PopInfoRec GridCurrentMoverPopTable[] = {
	{ M33, nil, M33TIMEZONEPOPUP, 0, pTIMEZONES, 0, 1, FALSE, nil }
};

static GridCurrentMover *sGridCurrentDialogMover;
static Boolean sDialogUncertaintyChanged2;

GridCurrentMover::GridCurrentMover (TMap *owner, char *name) : TCurrentMover(owner, name)
{
	//memset(&fUncertainParameters,0,sizeof(fUncertainParams));
	fArrowScale = 1.;
	fArrowDepth = 0;
	bShowGrid = false;
	bShowArrows = false;
	bUncertaintyPointOpen=false;
	/*if (gNoaaVersion)
	{
		fUncertainParams.alongCurUncertainty = .5;
		fUncertainParams.crossCurUncertainty = .25;
		fUncertainParams.durationInHrs = 24.0;
	}
	else
	{
		fUncertainParams.alongCurUncertainty = 0.;
		fUncertainParams.crossCurUncertainty = 0.;
		fUncertainParams.durationInHrs = 0.;
	}
	fUncertainParams.uncertMinimumInMPS = 0.0;
	fCurScale = 1.0;
	fUncertainParams.startTimeInHrs = 0.0;
	//fVar.gridType = TWO_D; // 2D default
	//fVar.maxNumDepths = 1;	// 2D default - may not need this
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fUncertainParams.alongCurUncertainty; 
	fUpCurUncertainty = fUncertainParams.alongCurUncertainty; 	
	fRightCurUncertainty = fUncertainParams.crossCurUncertainty;  
	fLeftCurUncertainty = -fUncertainParams.crossCurUncertainty; 
	fDuration=fUncertainParams.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fUncertainParams.startTimeInHrs*3600.);
	*///
	//bShowDepthContours = false;
	//bShowDepthContourLabels = false;
	
	fIsOptimizedForStep = false;

	SetClassName (name); // short file name
	
	//fAllowExtrapolationOfCurrentsInTime = false;
	//fAllowVerticalExtrapolationOfCurrents = false;
	//fMaxDepthForExtrapolation = 0.;	// assume 2D is just surface
	
	//memset(&fLegendRect,0,sizeof(fLegendRect)); 
}

/*void GridCurrentMover::Dispose ()
{
	if (timeGrid)
	{
		timeGrid -> Dispose();
		//delete timeGrid;	// this causes a crash...
		timeGrid = nil;
	}
	
	TCurrentMover::Dispose ();
}*/


void ShowGridCurrentMoverDialogItems(DialogPtr dialog)
{
	Boolean bShowGMTItems = true;
	short timeZone = GetPopSelection(dialog, M33TIMEZONEPOPUP);
	if (timeZone == 1) bShowGMTItems = false;
	
	ShowHideDialogItem(dialog, M33TIMESHIFTLABEL, bShowGMTItems); 
	ShowHideDialogItem(dialog, M33TIMESHIFT, bShowGMTItems); 
	ShowHideDialogItem(dialog, M33GMTOFFSETS, bShowGMTItems); 
}

void ShowHideVerticalExtrapolationDialogItems2(DialogPtr dialog)
{
	Boolean extrapolateVertically, okToExtrapolate = false, showVelAtBottom = GetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX);
	TMap *map = sGridCurrentDialogMover -> GetMoverMap();
	
	if (map && map->IAm(TYPE_PTCURMAP))
	{
		if ((dynamic_cast<PtCurMap *>(map))->GetMaxDepth2() > 0 || ((TimeGridVelRect*) (sGridCurrentDialogMover -> timeGrid))->GetMaxDepth() > 0) okToExtrapolate = true;
	} 
	
	if (sGridCurrentDialogMover->timeGrid->fVar.gridType!=TWO_D || !okToExtrapolate)	// if model has depth data assume that is what user wants to use
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
	ShowHideDialogItem(dialog, M33ARROWDEPTHAT, (sGridCurrentDialogMover->timeGrid->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate))); 
	ShowHideDialogItem(dialog, M33ARROWDEPTH, (sGridCurrentDialogMover->timeGrid->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate)) && !showVelAtBottom); 
	ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, (sGridCurrentDialogMover->timeGrid->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate)) && !showVelAtBottom); 
	ShowHideDialogItem(dialog, M33VELOCITYATBOTTOMCHECKBOX, (sGridCurrentDialogMover->timeGrid->fVar.gridType!=TWO_D || (extrapolateVertically && okToExtrapolate))); 
}

short GridCurrentMoverSettingsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	long menuID_menuItem;
	switch (itemNum) {
		case M33OK:
		{
			char errmsg[256];
			short timeZone = GetPopSelection(dialog, M33TIMEZONEPOPUP);
			Seconds timeShift = sGridCurrentDialogMover->timeGrid->fTimeShift;
			float arrowDepth = EditText2Float(dialog, M33ARROWDEPTH), maxDepth=0;
			float maxDepthForExtrapolation = EditText2Float(dialog, M33EXTRAPOLATETOVALUE);
			double tempAlong, tempCross, tempDuration, tempStart;
			long timeShiftInHrs;
			Boolean extrapolateVertically = GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX);
			//Boolean extrapolateVertically = false;
			Boolean showBottomVel = GetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX);
			TMap *map = sGridCurrentDialogMover -> GetMoverMap();
			
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
				maxDepth = ((TimeGridVelRect*) (sGridCurrentDialogMover -> timeGrid)) -> GetMaxDepth();
				//maxDepth = 1000;
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
			mygetitext(dialog, M33NAME, sGridCurrentDialogMover->timeGrid->fVar.userName, kPtCurUserNameLen-1);
			sGridCurrentDialogMover->bActive = GetButton(dialog, M33ACTIVE);
			sGridCurrentDialogMover->bShowArrows = GetButton(dialog, M33SHOWARROWS);
			sGridCurrentDialogMover->fArrowScale = EditText2Float(dialog, M33ARROWSCALE);
			sGridCurrentDialogMover->fArrowDepth = arrowDepth;
			//sGridCurrentDialogMover->timeGrid->fVar.fileScaleFactor = EditText2Float(dialog, M33SCALE);
			sGridCurrentDialogMover->fCurScale = EditText2Float(dialog, M33SCALE);
			
			if (sGridCurrentDialogMover->fUncertainParams.alongCurUncertainty != tempAlong || sGridCurrentDialogMover->fUncertainParams.crossCurUncertainty != tempCross
				|| sGridCurrentDialogMover->fUncertainParams.startTimeInHrs != tempStart || sGridCurrentDialogMover->fUncertainParams.durationInHrs != tempDuration)
				sDialogUncertaintyChanged2 = true;
			sGridCurrentDialogMover->fUncertainParams.alongCurUncertainty = EditText2Float(dialog, M33ALONG)/100;
			sGridCurrentDialogMover->fUncertainParams.crossCurUncertainty = EditText2Float(dialog, M33CROSS)/100;
			//sNetCDFDialogMover->fVar.uncertMinimumInMPS = EditText2Float(dialog, M33MINCURRENT);
			sGridCurrentDialogMover->fUncertainParams.startTimeInHrs = EditText2Float(dialog, M33STARTTIME);
			sGridCurrentDialogMover->fUncertainParams.durationInHrs = EditText2Float(dialog, M33DURATION);
			
			sGridCurrentDialogMover->fDownCurUncertainty = -sGridCurrentDialogMover->fUncertainParams.alongCurUncertainty; 
			sGridCurrentDialogMover->fUpCurUncertainty = sGridCurrentDialogMover->fUncertainParams.alongCurUncertainty; 	
			sGridCurrentDialogMover->fRightCurUncertainty = sGridCurrentDialogMover->fUncertainParams.crossCurUncertainty;  
			sGridCurrentDialogMover->fLeftCurUncertainty = -sGridCurrentDialogMover->fUncertainParams.crossCurUncertainty; 
			sGridCurrentDialogMover->fDuration = sGridCurrentDialogMover->fUncertainParams.durationInHrs * 3600.;  
			sGridCurrentDialogMover->fUncertainStartTime = (long) (sGridCurrentDialogMover->fUncertainParams.startTimeInHrs * 3600.); 
			//if (timeZone>1) sNetCDFDialogMover->fTimeShift = EditText2Long(dialog, M33TIMESHIFT)*3600*-1;
			if (timeZone>1) sGridCurrentDialogMover->timeGrid->fTimeShift =(long)( EditText2Float(dialog, M33TIMESHIFT)*3600);
			else sGridCurrentDialogMover->timeGrid->fTimeShift = 0;	// file is in local time
			
			//if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
			// code goes here, should also check if start time has been shifted correctly already (if there is another current)
			//if (timeShift != sNetCDFDialogMover->fTimeShift || sNetCDFDialogMover->GetTimeValue(0) != model->GetStartTime())
			// GetTImeValue adds in the time shift
			// What if time zone hasn't changed but start time has from loading in a different file??
			if ((timeShift != sGridCurrentDialogMover->timeGrid->fTimeShift && sGridCurrentDialogMover->timeGrid->GetTimeValue(0) != model->GetStartTime()))
			{
				//model->SetModelTime(model->GetModelTime() - (sNetCDFDialogMover->fTimeShift-timeShift));
				if (model->GetStartTime() + sGridCurrentDialogMover->timeGrid->fTimeShift - timeShift == sGridCurrentDialogMover->timeGrid->GetTimeValue(0))
					model->SetStartTime(model->GetStartTime() + (sGridCurrentDialogMover->timeGrid->fTimeShift-timeShift));
				//model->SetStartTime(sNetCDFDialogMover->GetTimeValue(0));	// just in case a different file has been loaded in the meantime, but what if user doesn't want start time to change???
				//sNetCDFDialogMover->SetInterval(errmsg);
				model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
			}
			sGridCurrentDialogMover->timeGrid->fAllowExtrapolationInTime = GetButton(dialog, M33EXTRAPOLATECHECKBOX);
			//((TimeGridVelRect*) (sGridCurrentDialogMover->timeGrid))->fAllowVerticalExtrapolationOfCurrents = GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX);
			sGridCurrentDialogMover->timeGrid->fAllowVerticalExtrapolationOfCurrents = GetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX);
			// code goes here, add box to dialog to input desired depth to extrapolate to 
			if (((TimeGridVelRect*) (sGridCurrentDialogMover->timeGrid))-> fAllowVerticalExtrapolationOfCurrents) ((TimeGridVelRect*) (sGridCurrentDialogMover->timeGrid))->fMaxDepthForExtrapolation = EditText2Float(dialog, M33EXTRAPOLATETOVALUE); 
			else ((TimeGridVelRect*) (sGridCurrentDialogMover->timeGrid))->fMaxDepthForExtrapolation = 0.;
			
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
			ShowHideVerticalExtrapolationDialogItems2(dialog);
			if (!GetButton(dialog,M33VELOCITYATBOTTOMCHECKBOX))
			Float2EditText(dialog, M33ARROWDEPTH, 0, 6);
			break;
			
		case M33EXTRAPOLATEVERTCHECKBOX:
			ToggleButton(dialog, itemNum);
			ShowHideVerticalExtrapolationDialogItems2(dialog);
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
			ShowGridCurrentMoverDialogItems(dialog);
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
			//sGridCurrentDialogMover->ReplaceMover();
			//mysetitext(dialog, M33NAME, sGridCurrentDialogMover->fVar.userName); // use short file name for now		
		}
	}
	
	return 0;
}
OSErr GridCurrentMoverSettingsInit(DialogPtr dialog, VOIDPTR data)
{
	char curStr[64], blankStr[32];
	strcpy(blankStr,"");
	
	RegisterPopTable (GridCurrentMoverPopTable, sizeof (GridCurrentMoverPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog(M33, dialog);
	
	SetDialogItemHandle(dialog, M33HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M33UNCERTAINTYBOX, (Handle)FrameEmbossed);
	
	mysetitext(dialog, M33NAME, sGridCurrentDialogMover->timeGrid->fVar.userName); // use short file name for now
	SetButton(dialog, M33ACTIVE, sGridCurrentDialogMover->bActive);
	
	if (sGridCurrentDialogMover->timeGrid->fTimeShift == 0) SetPopSelection (dialog, M33TIMEZONEPOPUP, 1);
	else SetPopSelection (dialog, M33TIMEZONEPOPUP, 2);
	//Long2EditText(dialog, M33TIMESHIFT, (long) (-1.*sGridCurrentDialogMover->fTimeShift/3600.));
	Float2EditText(dialog, M33TIMESHIFT, (float)(sGridCurrentDialogMover->timeGrid->fTimeShift)/3600.,1);
	
	SetButton(dialog, M33SHOWARROWS, sGridCurrentDialogMover->bShowArrows);
	Float2EditText(dialog, M33ARROWSCALE, sGridCurrentDialogMover->fArrowScale, 6);
	Float2EditText(dialog, M33ARROWDEPTH, sGridCurrentDialogMover->fArrowDepth, 6);
	
	ShowHideDialogItem(dialog, M33MINCURRENTLABEL, false); 
	ShowHideDialogItem(dialog, M33MINCURRENT, false); 
	ShowHideDialogItem(dialog, M33MINCURRENTUNITS, false); 
	
	//ShowHideDialogItem(dialog, M33SCALE, false); 
	//ShowHideDialogItem(dialog, M33SCALELABEL, false); 
	
	//ShowHideDialogItem(dialog, M33ARROWDEPTHAT, sGridCurrentDialogMover->fNumDepthLevels > 1); 
	//ShowHideDialogItem(dialog, M33ARROWDEPTH, sGridCurrentDialogMover->fNumDepthLevels > 1 && sGridCurrentDialogMover->fVar.arrowDepth >= 0); 
	//ShowHideDialogItem(dialog, M33ARROWDEPTHUNITS, sGridCurrentDialogMover->fNumDepthLevels > 1 && sGridCurrentDialogMover->fVar.arrowDepth >= 0); 
	
	if (!gNoaaVersion)
		ShowHideDialogItem(dialog, M33REPLACEMOVER, false);	// code goes here, eventually allow this for outside
	
	SetButton(dialog, M33EXTRAPOLATECHECKBOX, sGridCurrentDialogMover->timeGrid->fAllowExtrapolationInTime);
	//SetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX, ((TimeGridVelRect*) (sGridCurrentDialogMover->timeGrid))->fAllowVerticalExtrapolationOfCurrents);
	SetButton(dialog, M33EXTRAPOLATEVERTCHECKBOX, sGridCurrentDialogMover->timeGrid->fAllowVerticalExtrapolationOfCurrents);
	SetButton(dialog, M33VELOCITYATBOTTOMCHECKBOX, sGridCurrentDialogMover->fArrowDepth < 0);
	
	//ShowHideDialogItem(dialog, M33VELOCITYATBOTTOMCHECKBOX, sGridCurrentDialogMover->fNumDepthLevels > 1); 
	
	//Float2EditText(dialog, M33SCALE, sGridCurrentDialogMover->timeGrid->fVar.fileScaleFactor, 6);
	Float2EditText(dialog, M33SCALE, sGridCurrentDialogMover->fCurScale, 6);
	if(sGridCurrentDialogMover->fUncertainParams.alongCurUncertainty == 0) 
	{	// assume if one is not set, none are
		mysetitext(dialog, M33ALONG, blankStr); // force user to set uncertainty
		mysetitext(dialog, M33CROSS, blankStr); // force user to set uncertainty
		mysetitext(dialog, M33STARTTIME, blankStr); // force user to set
		mysetitext(dialog, M33DURATION, blankStr); // force user to set 
	}
	else
	{
		Float2EditText(dialog, M33ALONG, sGridCurrentDialogMover->fUncertainParams.alongCurUncertainty*100, 6);
		Float2EditText(dialog, M33CROSS, sGridCurrentDialogMover->fUncertainParams.crossCurUncertainty*100, 6);
		Float2EditText(dialog, M33STARTTIME, sGridCurrentDialogMover->fUncertainParams.startTimeInHrs, 6);
		Float2EditText(dialog, M33DURATION, sGridCurrentDialogMover->fUncertainParams.durationInHrs, 6);
	}
	
	Float2EditText(dialog, M33EXTRAPOLATETOVALUE, ((TimeGridVelRect*) (sGridCurrentDialogMover->timeGrid))->fMaxDepthForExtrapolation, 6);
	
	ShowHideVerticalExtrapolationDialogItems2(dialog);
	ShowGridCurrentMoverDialogItems(dialog);
	if (sGridCurrentDialogMover->timeGrid->fTimeShift == 0) MySelectDialogItemText(dialog, M33ALONG, 0, 100);
	else MySelectDialogItemText(dialog, M33TIMESHIFT, 0, 100);
	
	//ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, sNetCDFDialogMover->fVar.gridType==TWO_D); 	// if current already has 3D info don't confuse with extrapolation option
	if (!gNoaaVersion) ShowHideDialogItem(dialog, M33EXTRAPOLATEVERTCHECKBOX, false); 	// for now hide until decide how to handle 3D currents...
	
	return 0;
}



OSErr GridCurrentMover::SettingsDialog()
{
	short item;
	
	sGridCurrentDialogMover = this; // should pass in what is needed only
	sDialogUncertaintyChanged2 = false;
	item = MyModalDialog(M33, mapWindow, 0, GridCurrentMoverSettingsInit, GridCurrentMoverSettingsClick);
	sGridCurrentDialogMover = 0;
	
	if(M33OK == item)	
	{
		if (sDialogUncertaintyChanged2) this -> UpdateUncertaintyValues(model->GetModelTime()-model->GetStartTime());
		model->NewDirtNotification();// tell model about dirt
	}
	return M33OK == item ? 0 : -1;
}

OSErr GridCurrentMover::InitMover(TimeGridVel *grid)
{	
	OSErr	err = noErr;
	timeGrid = grid;
	err = TCurrentMover::InitMover ();
	return err;
}


#define GridCurrentMoverREADWRITEVERSION 1 //don't bother with this for now

OSErr GridCurrentMover::Write (BFPB *bfpb)
{
	long version = GridCurrentMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	if (err = TCurrentMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("GridCurrentMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	id = timeGrid -> GetClassID ();
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = timeGrid -> Write (bfpb)) goto done;
	
	// code goes here, may want to add a name/path

done:
	if(err)
		TechError("GridCurrentMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr GridCurrentMover::Read(BFPB *bfpb)
{
	long version;
	ClassID id;
	OSErr err = 0;
	
	if (err = TCurrentMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("GridCurrentMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("GridCurrentMover::Read()", "id != TYPE_GRIDCURRENTMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > GridCurrentMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the GridCurrent mover
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_TIMEGRIDVELRECT: timeGrid = new TimeGridVelRect; break;
		//case TYPE_TIMEGRIDVEL: timeGrid = new TimeGridVel; break;
		case TYPE_TIMEGRIDVELCURV: timeGrid = new TimeGridVelCurv; break;
		case TYPE_TIMEGRIDVELTRI: timeGrid = new TimeGridVelTri; break;
		case TYPE_TIMEGRIDCURRECT: timeGrid = new TimeGridCurRect; break;
		case TYPE_TIMEGRIDCURTRI: timeGrid = new TimeGridCurTri; break;
		default: printError("Unrecognized Grid type in GridCurrentMover::Read()."); return -1;
	}

	// code goes here, should also have a name/path
done:
	if(err)
	{
		TechError("GridCurrentMover::Read(char* path)", " ", 0); 
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr GridCurrentMover::CheckAndPassOnMessage(TModelMessage *message)
{
	return TCurrentMover::CheckAndPassOnMessage(message); 
}

/////////////////////////////////////////////////
long GridCurrentMover::GetListLength()
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

ListItem GridCurrentMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64], dateStr[64];
	long numTimesInFile = timeGrid->GetNumTimesInFile();
	//long numTimesInFile = 0;
	ListItem item = { this, 0, indent, 0 };
	
	if (n == 0) {
		item.index = I_GRIDCURRENTNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Currents: \"%s\"", timeGrid->fVar.userName);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	if (bOpen) {
		
		
		if (--n == 0) {
			item.index = I_GRIDCURRENTACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_GRIDCURRENTGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			sprintf(text, "Show Grid");
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_GRIDCURRENTARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,fArrowScale,6);
			//sprintf(text, "Show Velocities (@ 1 in = %s m/s) ", valStr);
			if (timeGrid->fVar.gridType==TWO_D)
				sprintf(text, "Show Velocities (@ 1 in = %s m/s)", valStr);
			else
			{
				if (fArrowDepth>=0)
					sprintf(text, "Show Velocities (@ 1 in = %s m/s) at %g m", valStr, fArrowDepth);
				else
					sprintf(text, "Show Velocities (@ 1 in = %s m/s) at bottom", valStr);
			}
			item.indent++;
			return item;
		}
		
		if (--n == 0) {
			item.index = I_GRIDCURRENTSCALE;
			StringWithoutTrailingZeros(valStr,fCurScale,6);
			sprintf(text, "Multiplicative Scalar: %s", valStr);
			//item.indent++;
			return item;
		}
		
		// release time
		if (timeGrid->GetNumFiles()>1)
		{
			if (--n == 0) {
				//item.indent++;
				//Seconds time = (*fInputFilesHdl)[0].startTime + fTimeShift;
				Seconds time = timeGrid->GetStartTimeValue(0);				
				Secs2DateString2 (time, dateStr);
				/*if(numTimesInFile>0)*/ sprintf (text, "Start Time: %s", dateStr);
				//else sprintf (text, "Time: %s", dateStr);
				return item;
			}
			if (--n == 0) {
				//item.indent++;
				//Seconds time = (*fInputFilesHdl)[GetNumFiles()-1].endTime + fTimeShift;				
				Seconds time = timeGrid->GetStartTimeValue(timeGrid->GetNumFiles()-1);				
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
					//Seconds time = (*fTimeHdl)[0] + fTimeShift;				
					Seconds time = timeGrid->GetTimeValue(0);					// includes the time shift
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
					//Seconds time = (*fTimeHdl)[numTimesInFile-1] + fTimeShift;				
					Seconds time = timeGrid->GetTimeValue(numTimesInFile-1) + timeGrid->fTimeShift;				
					Secs2DateString2 (time, dateStr);
					sprintf (text, "End Time: %s", dateStr);
					return item;
				}
			}
		}
		
		
		if(model->IsUncertain())
		{
			if (--n == 0) {
				item.index = I_GRIDCURRENTUNCERTAINTY;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				item.indent++;
				return item;
			}
			
			if (bUncertaintyPointOpen) {
				
				if (--n == 0) {
					item.index = I_GRIDCURRENTALONGCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fUncertainParams.alongCurUncertainty*100,6);
					sprintf(text, "Along Current: %s %%",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_GRIDCURRENTCROSSCUR;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fUncertainParams.crossCurUncertainty*100,6);
					sprintf(text, "Cross Current: %s %%",valStr);
					return item;
				}
				
				/*if (--n == 0) {
				 item.index = I_GRIDCURRENTMINCURRENT;
				 item.indent++;
				 StringWithoutTrailingZeros(valStr,fUncertainParams.uncertMinimumInMPS,6);
				 sprintf(text, "Current Minimum: %s m/s",valStr);
				 return item;
				 }*/
				
				if (--n == 0) {
					item.index = I_GRIDCURRENTSTARTTIME;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fUncertainParams.startTimeInHrs,6);
					sprintf(text, "Start Time: %s hours",valStr);
					return item;
				}
				
				if (--n == 0) {
					item.index = I_GRIDCURRENTDURATION;
					//item.bullet = BULLET_DASH;
					item.indent++;
					StringWithoutTrailingZeros(valStr,fUncertainParams.durationInHrs,6);
					sprintf(text, "Duration: %s hours",valStr);
					return item;
				}
				
				
			}
			
		}  // uncertainty is on
		
	} // bOpen
	
	item.owner = 0;
	
	return item;
}

Boolean GridCurrentMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_GRIDCURRENTNAME: bOpen = !bOpen; return TRUE;
			case I_GRIDCURRENTGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_GRIDCURRENTARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_GRIDCURRENTUNCERTAINTY: bUncertaintyPointOpen = !bUncertaintyPointOpen; return TRUE;
			case I_GRIDCURRENTACTIVE:
				bActive = !bActive;
				model->NewDirtNotification(); 
				return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_GRIDCURRENTNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
	}
	
	if (doubleClick && !inBullet)
	{
		Boolean userCanceledOrErr ;
		(void) this -> SettingsDialog();	// will need to add a dialog...
		return TRUE;
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean GridCurrentMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_GRIDCURRENTNAME:
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

OSErr GridCurrentMover::SettingsItem(ListItem item)
{
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = this -> ListClick(item,inBullet,doubleClick);
	return 0;
}

/*OSErr GridCurrentMover::AddItem(ListItem item)
 {
 if (item.index == I_GRIDCURRENTNAME)
 return TMover::AddItem(item);
 
 return 0;
 }*/

OSErr GridCurrentMover::DeleteItem(ListItem item)
{
	if (item.index == I_GRIDCURRENTNAME)
		return moverMap -> DropMover(this);
	
	return 0;
}

Boolean GridCurrentMover::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
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
	if (!bShowArrows && !bShowGrid) return 0;
	err = timeGrid -> SetInterval(errmsg, model->GetModelTime()); 
	
	if(err) return false;
	
	if (err = timeGrid->VelocityStrAtPoint(wp, diagnosticStr, fArrowDepth)) return err;
		
/*	if (fVar.arrowDepth>0 && fVar.gridType==TWO_D)
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
	
	if(dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()>1)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) 
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

	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular
		
		indices = this->GetVelocityIndices(wp.p);
		
		if (index >= 0)
		{
			// Check for constant current 
			if(dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
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
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v) * this->fFileScaleFactor;
	lengthS = this->fVar.curScale * lengthU;
	
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
	*/
	return true;
}

Boolean GridCurrentMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant current, we can say "no"
	//if(this->GetNumTimesInFile()==1) depends = false;
	if(timeGrid->GetNumTimesInFile()==1 && !(timeGrid->GetNumFiles()>1)) depends = false;
	return depends;
}

void GridCurrentMover::Draw(Rect r, WorldRect view) 
{	// Use this for regular grid or regridded data
	timeGrid->Draw(r,view,fCurScale,fArrowScale,fArrowDepth,bShowArrows,bShowGrid,fColor);
}

/////////////////////////////////////////////////////////////////


