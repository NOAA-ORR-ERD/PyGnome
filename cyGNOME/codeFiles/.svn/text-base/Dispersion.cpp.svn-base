
#include "Cross.h"
#include "TimUtils.h"

static DispersionRec sDInfo;
static Seconds sStartRelTime;
static PopInfoRec DispPopTable[] = {
		{ DISPERSION_DLGID, nil, DISPTOPLATDIR, 0, pNORTHSOUTH1, 0, 1, FALSE, nil },
		{ DISPERSION_DLGID, nil, DISPLEFTLONGDIR, 0, pEASTWEST1, 0, 1, FALSE, nil },
		{ DISPERSION_DLGID, nil, DISPBOTTOMLATDIR, 0, pNORTHSOUTH2, 0, 1, FALSE, nil },
		{ DISPERSION_DLGID, nil, DISPRIGHTLONGDIR, 0, pEASTWEST2, 0, 1, FALSE, nil },
		{ DISPERSION_DLGID, nil, DISP_STARTMONTH, 0, pMONTHS, 0, 1, FALSE, nil },
		{ DISPERSION_DLGID, nil, DISP_STARTYEAR, 0, pYEARS, 0, 1, FALSE, nil },
		{ DISPERSION_DLGID, nil, DISP_STARTTIMEPOPUP, 0, pDISPTIMETYPE, 0, 1, FALSE, nil }
		//{ DISPERSION_DLGID, nil, DISP_APILABEL, 0, pAPIVISCOSITY, 0, 1, FALSE, nil }
	};

void ShowHideDisperseRegion(DialogPtr dialog)
{
	Boolean show  = GetButton (dialog, DISP_REGION); 

	SwitchLLFormatHelper(dialog, DISPTOPLATDEGREES, DISPDEGREES, show);
	SwitchLLFormatHelper(dialog, DISPBOTTOMLATDEGREES, DISPDEGREES, show); 
	
	ShowHideDialogItem(dialog, DISPDEGREES, show); 
	ShowHideDialogItem(dialog, DISPDEGMIN, show); 
	ShowHideDialogItem(dialog, DISPDMS, show); 

	ShowHideDialogItem(dialog, DISPTOPLATLABEL, show); 
	ShowHideDialogItem(dialog, DISPLEFTLONGLABEL, show); 
	ShowHideDialogItem(dialog, DISPBOTTOMLATLABEL, show); 
	ShowHideDialogItem(dialog, DISPRIGHTLONGLABEL, show); 
	
	if (!show)
	{	// there must be a better way ... this is to get rid of shadow on drop downs that show up when item is hidden
		ShowHideDialogItem(dialog, DISP_FROST1, show);
	}
	ShowHideDialogItem(dialog, DISP_FROST1, true);
}

void ShowHideTimeItems(DialogPtr dialog)
{
	Boolean showRealTime, showHrsAfterSpill;
	short timeType = GetPopSelection(dialog, DISP_STARTTIMEPOPUP);

	switch (timeType)
	{
		//case Real Time:
		case 1:
			showRealTime=TRUE;
			showHrsAfterSpill=FALSE;
			break;
		//case Hours After Spill:
		case 2:
			showRealTime=FALSE;
			showHrsAfterSpill=TRUE;
			break;
	}
	// Real time items
	ShowHideDialogItem(dialog, DISP_STARTMONTH, showRealTime); 
	ShowHideDialogItem(dialog, DISP_STARTYEAR, showRealTime); 
	ShowHideDialogItem(dialog, DISP_STARTDAY, showRealTime); 
	ShowHideDialogItem(dialog, DISP_STARTHOURS, showRealTime); 
	ShowHideDialogItem(dialog, DISP_STARTMINUTES, showRealTime); 
	ShowHideDialogItem(dialog, DISP_COLON, showRealTime); 
	ShowHideDialogItem(dialog, DISP_TIMELABEL, showRealTime); 
	ShowHideDialogItem(dialog, DISP_FROST2, showRealTime); 

	ShowHideDialogItem(dialog, DISP_STARTTIME, showHrsAfterSpill); 
	ShowHideDialogItem(dialog, DISP_STARTTIMEUNITS, showHrsAfterSpill); 
}

short DispersionClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	WorldPoint p, p2;
	Boolean changed, tempUseBounds;
	WorldRect origBounds = emptyWorldRect;
	Seconds dispTime;
	long menuID_menuItem;
	OSErr err = 0;

	StandardLLClick(dialog, itemNum, DISPTOPLATDEGREES, DISPDEGREES, &p, &changed);
	StandardLLClick(dialog, itemNum, DISPBOTTOMLATDEGREES, DISPDEGREES, &p2, &changed);

	{	
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map)
		{
			origBounds = map -> GetMapBounds();
		}
	}
	
	switch(itemNum)
	{
		case DISP_OK:
			
			tempUseBounds = GetButton (dialog, DISP_REGION);
			short timeType, depthInputType, seaType;
			
			if(tempUseBounds)
			{
				long oneSecond = (1000000/3600);
				// retrieve the extendedBounds
				if (err = EditTexts2LL(dialog, DISPTOPLATDEGREES, &p, TRUE)) break;
				if (err = EditTexts2LL(dialog, DISPBOTTOMLATDEGREES, &p2, TRUE)) break;

				// check extended bounds (oneSecond handles accuracy issue in reading from dialog)			
				if (p.pLat > origBounds.hiLat + oneSecond || p2.pLat < origBounds.loLat - oneSecond
					|| p.pLong < origBounds.loLong - oneSecond || p2.pLong > origBounds.hiLong + oneSecond)
				{
					printError("The dispersion area cannot be greater than the map bounds."); 
					return 0; 
				}
				
				if (p.pLat < p2.pLat || p.pLong > p2.pLong)
				{
					printError("The dispersion area bounds are not consistent (top < bot or left > right)."); 
					return 0; 
				}
				
				// just in case of round off
				p.pLat = _min(p.pLat,origBounds.hiLat);
				p.pLong = _max(p.pLong,origBounds.loLong);
				p2.pLat = _max(p2.pLat,origBounds.loLat);
				p2.pLong = _min(p2.pLong,origBounds.hiLong);
			}
	
			if (sDInfo.lassoSelectedLEsToDisperse==true)
			{
				// point out there is already a file selected
				short buttonSelected  = MULTICHOICEALERT(1688,"Altering this dialog will deselect lassoed LEs. Do you want to continue?",TRUE);
				switch(buttonSelected){
					case 1:// continue
						sDInfo.lassoSelectedLEsToDisperse=false;
						break;
					case 3: // cancel
					{
						return DISP_CANCEL;
					}
				}
			}

			timeType = GetPopSelection(dialog, DISP_STARTTIMEPOPUP);
			
			if (timeType == 1)
			{
				// get dispersant time
				dispTime = RetrievePopTime(dialog, DISP_STARTMONTH, &err);
				if(err) break;
				//sDInfo.timeToDisperse = dispTime - model->GetStartTime(); // should use spill start time...
				sDInfo.timeToDisperse = dispTime - sStartRelTime; // should use spill start time...
				//sDInfo.timeToDisperse = dispTime; 
			}
			else
				sDInfo.timeToDisperse = round(EditText2Float(dialog, DISP_STARTTIME)*3600);

			sDInfo.duration = round(EditText2Float(dialog, DISP_DURATION)*3600);
			sDInfo.amountToDisperse = EditText2Float(dialog, DISP_AMOUNT)/100;
			sDInfo.api = EditText2Float(dialog, DISP_API);

			// restore to original bounds if uncheck box
			if (tempUseBounds)
			{
				sDInfo.areaToDisperse.loLat = p2.pLat;
				sDInfo.areaToDisperse.hiLat = p.pLat;
				sDInfo.areaToDisperse.loLong = p.pLong;
				sDInfo.areaToDisperse.hiLong = p2.pLong;
			}
			else
				sDInfo.areaToDisperse = origBounds;

			return DISP_OK;
			
		case DISP_CANCEL:
			return DISP_CANCEL;
			break;
			
		case DISP_AMOUNT:		
			CheckNumberTextItem(dialog, itemNum, FALSE); //  don't allow decimals
			break;

		case DISP_API:
		case DISP_DURATION:
		case DISP_STARTTIME:		
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;

		case DISP_REGION:
			ToggleButton(dialog, itemNum);
			ShowHideDisperseRegion(dialog);
			break;

		case DISPDEGREES:
		case DISPDEGMIN:
		case DISPDMS:
				if (err = EditTexts2LL(dialog, DISPTOPLATDEGREES, &p, TRUE)) break;
				if (err = EditTexts2LL(dialog, DISPBOTTOMLATDEGREES, &p2, TRUE)) break;
				if (itemNum == DISPDEGREES) settings.latLongFormat = DEGREES;
				if (itemNum == DISPDEGMIN) settings.latLongFormat = DEGMIN;
				if (itemNum == DISPDMS) settings.latLongFormat = DMS;
				//ShowHideDisperseRegion(dialog);
				SwitchLLFormatHelper(dialog, DISPTOPLATDEGREES, DISPDEGREES, true);
				SwitchLLFormatHelper(dialog, DISPBOTTOMLATDEGREES, DISPDEGREES, true); 
				LL2EditTexts(dialog, DISPBOTTOMLATDEGREES, &p2);
				LL2EditTexts(dialog, DISPTOPLATDEGREES, &p);
			break;

		case DISP_STARTDAY:
		case DISP_STARTHOURS:
		case DISP_STARTMINUTES:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case DISP_STARTMONTH:
		case DISP_STARTYEAR:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
			
		//case DISP_APILABEL:	// may have option to input viscosity at ref temp instead
			//PopClick(dialog, itemNum, &menuID_menuItem);

		case DISP_STARTTIMEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideTimeItems(dialog);
			if (GetPopSelection(dialog, DISP_STARTTIMEPOPUP) == 2)
				MySelectDialogItemText(dialog, DISP_STARTTIME, 0, 255);
			else
				MySelectDialogItemText(dialog, DISP_STARTDAY, 0, 255);
			break;
	}
	return 0;
}

OSErr DispersionInit(DialogPtr dialog, VOIDPTR data)
{
	#pragma unused (data)
	WorldPoint wp;
	DateTimeRec	time;

	SetDialogItemHandle(dialog, DISP_HILITE, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, DISP_FROST1, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, DISP_FROST2, (Handle)FrameEmbossed);

	//RegisterPopTable (DispPopTable, sizeof (DispPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (DISPERSION_DLGID, dialog);
	
	SetPopSelection (dialog, DISP_STARTTIMEPOPUP, 1);	// may want to store this info...

	OriginalFloat2EditText(dialog, DISP_AMOUNT, sDInfo.amountToDisperse*100, 0);
	Float2EditText(dialog, DISP_API, sDInfo.api, 2);
	Float2EditText(dialog, DISP_STARTTIME, sDInfo.timeToDisperse / 3600.0, 2);
	Float2EditText(dialog, DISP_DURATION, sDInfo.duration / 3600.0, 2);

	SetButton (dialog, DISP_REGION, true); // or should default be whole map...
	wp.pLat = sDInfo.areaToDisperse.hiLat;
	wp.pLong = sDInfo.areaToDisperse.loLong;
	LL2EditTexts (dialog, DISPTOPLATDEGREES, &wp);
	
	wp.pLat = sDInfo.areaToDisperse.loLat;
	wp.pLong = sDInfo.areaToDisperse.hiLong;
	LL2EditTexts (dialog, DISPBOTTOMLATDEGREES, &wp);

	//SecondsToDate (sDInfo.timeToDisperse + model->GetStartTime(), &time);
	SecondsToDate (sDInfo.timeToDisperse + sStartRelTime, &time);
	SetPopSelection (dialog, DISP_STARTMONTH, time.month);
	SetPopSelection (dialog, DISP_STARTYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, DISP_STARTDAY, time.day);
	Long2EditText (dialog, DISP_STARTHOURS, time.hour);
	Long2EditText (dialog, DISP_STARTMINUTES, time.minute);
	//ShowHideDisperseRegion(dialog);
	SwitchLLFormatHelper(dialog, DISPTOPLATDEGREES, DISPDEGREES, true);
	SwitchLLFormatHelper(dialog, DISPBOTTOMLATDEGREES, DISPDEGREES, true); 
	ShowHideTimeItems(dialog);
	//ShowHideDialogItem(dialog, DISP_API, false); 
	//ShowHideDialogItem(dialog, DISP_APILABEL, false); 
	ShowHideDialogItem(dialog, DISP_APIUNITS, false); 
	//MySelectDialogItemText(dialog, DISP_STARTTIME, 0, 255);
			if (GetPopSelection(dialog, DISP_STARTTIMEPOPUP) == 2)
				MySelectDialogItemText(dialog, DISP_STARTTIME, 0, 255);
			else
				MySelectDialogItemText(dialog, DISP_STARTDAY, 0, 255);

	return 0;
}

OSErr DispersionDialog(DispersionRec *info, Seconds startRelTime, WindowPtr parentWindow)
{
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[20];

	if(parentWindow == nil) parentWindow = mapWindow; // we need the parent on the IBM
	if(info == nil) return -1;
	sDInfo = *info;
	sStartRelTime = startRelTime;

	if(UseExtendedYears()) {
		DispPopTable[5].menuID = pYEARS_EXTENDED;
	}
	else {
		DispPopTable[5].menuID = pYEARS;
	}

	// code to allow a dialog on top of another with pops
	// will want to skip this if don't call from spill dialog
	for(j = 0; j < sizeof(DispPopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = DispPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);

	item = MyModalDialog(DISPERSION_DLGID, parentWindow, 0, DispersionInit, DispersionClick);
	RestorePopTableInfo(saveTable);
	if (item == DISP_OK) {
		*info = sDInfo;
		if(parentWindow == mapWindow) {
			model->NewDirtNotification(); // when a dialog is the parent, we rely on that dialog to notify about Dirt 
			// that way we don't get the map redrawing behind the parent dialog on the IBM
		}
	}
	if (item == DISP_CANCEL) {return USERCANCEL;}
	return item == DISP_OK? 0 : -1;
}
