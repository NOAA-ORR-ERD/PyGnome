
#include "CROSS.h"
#include "TimUtils.h"

TSprayLEList* gSprayLEListInDialog = 0;				
static SprayDialogInfo	sharedSprayDialogInfo;

#ifdef MAC
	#pragma segment SPRAYLELISTS
#endif

enum { I_SP_LEFIRSTLINE = 1, I_SP_ACTIVE, I_SP_WINDAGE, I_SP_DISPERSE, /*I_SP_DRAWRECT,*/ I_SP_NATURAL_DISPERSE, I_SP_LESHOWHIDE, I_SP_UNSPECIFIEDLINE, I_SP_LERELEASE_MASSBALANCE };


Boolean IsEditTool(long tool)
{
	switch(tool)
	{
		case SPRAYCANTOOL:
		case ERASETOOL:
			return TRUE;
	}
	return false;
}


short SprayCanRadiusInPixels(void)
{	// returns the current spray can radius in pixels
	short size = 5;
	switch(settings.sprayCanSize) {
		case SPRAY_SIZE_SMALL:
			size = 3; break;
		default:
		case SPRAY_SIZE_MEDIUM:
			size = 5; break;
		case SPRAY_SIZE_LARGE:
			size = 7; break;
	}
	return size;
}

short EraserRadiusInPixels(void)
{	// returns the current spray can radius in pixels
	return 7;
}


Boolean CanSpray(Boolean *overWaterPtr)
{
	Point mousePt;
	Boolean canSpray = false;
	WorldPoint	wp;
	Point pixel;
	short radius = SprayCanRadiusInPixels();
	short i,j, radiusSquared = radius*radius;
	GrafPtr oldPort;

	GetPortGrafPtr(&oldPort);
	SetPortWindowPort(mapWindow);
	GetMouse(&mousePt);
	SetPortGrafPort(oldPort);
	
	if(overWaterPtr)
		*overWaterPtr = false;
	
	for(i = -radius; i <= radius && !canSpray; i++) 
	{
		for(j = -radius; j <= radius && !canSpray; j++) 
		{
			Boolean insideCircle = (i*i+j*j <= radiusSquared);
			if(insideCircle) {
				SetPt(&pixel,mousePt.h+i,mousePt.v+j);
				wp = ScreenToWorldPoint(pixel, MapDrawingRect(), settings.currentView);
				if(model->IsWaterPoint(wp) && overWaterPtr)
					*overWaterPtr = true;
				if(model->IsAllowableSpillPoint(wp) || (gNoaaVersion && model->bHindcast)) 
					return true;
			}
		}
	}
	return false;
}



/////////////////////////////////////////////////

short SprayCanClick (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	Boolean changed;
	long menuID_menuItem;
	WorldPoint p, p2;
	DateTimeRec time;
	OSErr err = 0;
	char errStr[256] = "";
	long amountMin = 0, amountMax;
	long spraySize;

	switch (itemNum) {
		case M31OK:
			
			spraySize = SPRAY_SIZE_MEDIUM;
			if (GetButton(dialog, M31RADIUSSMALL))  spraySize = SPRAY_SIZE_SMALL;
			if (GetButton(dialog, M31RADIUSMEDIUM)) spraySize = SPRAY_SIZE_MEDIUM;
			if (GetButton(dialog, M31RADIUSLARGE))  spraySize = SPRAY_SIZE_LARGE;
			settings.sprayCanSize = spraySize;
			
			return M31OK;


		case M31RADIUSSMALL:
		case M31RADIUSMEDIUM:
		case M31RADIUSLARGE:
			ToggleButton(dialog, itemNum);
			SetButton(dialog, M31RADIUSSMALL,  itemNum == M31RADIUSSMALL);
			SetButton(dialog, M31RADIUSMEDIUM, itemNum == M31RADIUSMEDIUM);
			SetButton(dialog, M31RADIUSLARGE,  itemNum == M31RADIUSLARGE);
			break;

	}
	
	return 0;
}


OSErr SprayCanInit (DialogPtr dialog, VOIDPTR data)
{
#pragma unused (data)
	long spraySize = settings.sprayCanSize;

	SetDialogItemHandle(dialog, M31FROST, (Handle) FrameEmbossed);

	SetButton(dialog, M31RADIUSSMALL,  spraySize == SPRAY_SIZE_SMALL);
	SetButton(dialog, M31RADIUSMEDIUM, spraySize == SPRAY_SIZE_MEDIUM);
	SetButton(dialog, M31RADIUSLARGE,  spraySize == SPRAY_SIZE_LARGE);

	return 0;
}

void SprayCanSettingsDialog(void)
{	
	short 		dialogItem;
	dialogItem = MyModalDialog(M31, mapWindow, nil, SprayCanInit, SprayCanClick);
}


///////////////////////////////////////////////////////////////////////////

static PopInfoRec dsprayLEPopTable[] = {
		{ M15, nil, M15POLLUTANT, 0, pPOLLUTANTS, 0, 1, FALSE, nil },
		{ M15, nil, M15MASSUNITS, 0, pLEMASSUNITS, 0, 1, FALSE, nil },
		{ M15, nil, M15STARTMONTH, 0, pMONTHS, 0, 1, FALSE, nil },
		{ M15, nil, M15STARTYEAR, 0, pYEARS, 0, 1, FALSE, nil },
		{ M15, nil, M15AMOUNTTIMEPOPUP, 0, pTIMEAMOUNTTYPES, 0, 1, FALSE, nil },
		{ M15, nil, M15SPILLMONTH, 0, pMONTHS2, 0, 1, FALSE, nil },
		{ M15, nil, M15SPILLYEAR, 0, pYEARS2, 0, 1, FALSE, nil },
		{ M15, nil, M15DISPERSANT, 0, pDISPERSANTMODE, 0, 1, FALSE, nil }
	};

static AdiosInfoRecH gAdiosDataH = 0;
static char gOilTypeName[256];
static DispersionRec sDispInfo;
static WindageRec sWindageInfo;
static AdiosInfoRecH sAdiosInfoH = 0;
static char sOilTypeName[256];

void DisposeSprayDialogDataH(void)
{
	if (gAdiosDataH) {
		if(gAdiosDataH != sAdiosInfoH)
		{	// only dispose of this if it is different
			DisposeHandle((Handle)gAdiosDataH); 
		}
		gAdiosDataH = nil;
	}
}

void ShowHideSprayLEDialogItems(DialogPtr dialog)
{
	Boolean showReleaseTimeItems;
	Boolean showSpillTimeItems;
	Boolean showWhenAmountIsSpecified;
	
	short whenAmountIsSpecified = GetPopSelection(dialog, M15AMOUNTTIMEPOPUP);
	short oiltype = GetPopSelection(dialog, M15POLLUTANT);

	if (oiltype==OIL_CONSERVATIVE) // hide everything related to pollutant age
	{
		showReleaseTimeItems = FALSE;
		showSpillTimeItems = FALSE;
		showWhenAmountIsSpecified = FALSE; 
	}
	else
	{
		showWhenAmountIsSpecified = TRUE; 
		switch (whenAmountIsSpecified)
		{
			default:
			case ATTIMEOFSPILL:
				showReleaseTimeItems=TRUE;
				showSpillTimeItems=FALSE;
				break;
			case ATOVERFLIGHTTIME:
				showSpillTimeItems=TRUE;
				showReleaseTimeItems=FALSE;
				break;
		}
	}
	
	ShowHideDialogItem(dialog, M15AMOUNTTIMEPOPUP, showWhenAmountIsSpecified); 	// maybe a poperase() ? something bigger than the itembox to get the shadow on os x
	ShowHideDialogItem(dialog, M15AMOUNTTIMEPOPUPAT, showWhenAmountIsSpecified); 
	
	ShowHideDialogItem(dialog, M15SPILLMONTH, showReleaseTimeItems); 
	ShowHideDialogItem(dialog, M15SPILLDAY, showReleaseTimeItems); 
	ShowHideDialogItem(dialog, M15SPILLYEAR, showReleaseTimeItems); 
	ShowHideDialogItem(dialog, M15SPILLHOURS, showReleaseTimeItems); 
	ShowHideDialogItem(dialog, M15SPILLMINUTES, showReleaseTimeItems); 
	ShowHideDialogItem(dialog, M15FROST2, showReleaseTimeItems); 
	ShowHideDialogItem(dialog, M15FROST2LABEL, showReleaseTimeItems);
	ShowHideDialogItem(dialog, M15SPILLTIME24HRLABEL, showReleaseTimeItems);
	ShowHideDialogItem(dialog, M15SPILLTIMECOLON, showReleaseTimeItems);

	ShowHideDialogItem(dialog, M15RELEASEAGELABEL, showSpillTimeItems); 
	ShowHideDialogItem(dialog, M15RELEASEAGEHOURS, showSpillTimeItems); 
	ShowHideDialogItem(dialog, M15RELEASEAGEUNITSLABEL, showSpillTimeItems); 
	ShowHideDialogItem(dialog, M15RELEASEAGEUNITSEXPLANATION, showSpillTimeItems); 

}
			
short SprayLEClick (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
#pragma unused (data)
	Boolean changed;
	long menuID_menuItem;
	WorldPoint p, p2;
	DateTimeRec time;
	OSErr err = 0;
	char errStr[256] = "";
	long amountMin = 0, amountMax;
	SprayDialogInfo localSprayInfo;
	char str[256];

	switch (itemNum) {
		case M15OK:
			localSprayInfo = sharedSprayDialogInfo;

			localSprayInfo.overflightTime = RetrievePopTime(dialog, M15STARTMONTH,&err);
			if (err) break;
			
			localSprayInfo.numOfLEs = EditText2Long(dialog, M15NUMLES);
			localSprayInfo.pollutantType = GetPopSelection(dialog, M15POLLUTANT);
			localSprayInfo.totalMass = EditText2Float(dialog, M15MASS);
			localSprayInfo.massUnits = GetPopSelection(dialog, M15MASSUNITS);
			
					
			/////////////////////////////
			// enforce input limits on the amount 
			/////////////////////////////

			switch(localSprayInfo.massUnits)
			{
				case GALLONS: amountMax = 20000000;break;
				case BARRELS: amountMax = 476000;break;
				case CUBICMETERS: amountMax = 76000;break;
				///
				case METRICTONS: amountMax = 70000;break;
				case SHORTTONS:  amountMax = 77000;break;
			 	case KILOGRAMS: amountMax = 70000000;break;
				default: amountMax =-1;break; // -1 means don't enforce a limit
			}
			if(localSprayInfo.totalMass <= amountMin) strcpy(errStr,"The amount released must be greater than zero.");	
			else if(localSprayInfo.totalMass > amountMax && amountMax > 0)
			{
				char unitsStr[64];
				GetLeUnitsStr(unitsStr,localSprayInfo.massUnits);
				sprintf(errStr,"The amount released cannot exceed %ld %s.",amountMax,unitsStr);
			}
			
			if(errStr[0])
			{
				printError(errStr);
				MySelectDialogItemText(dialog, M15MASS,0,100);
				return 0;
			}

			if(localSprayInfo.numOfLEs > 10000 && !gDispersedOilVersion) // Yasuo requested no limit on number of LEs, 11/15/04
			{
				printError("The number of Splots cannot exceed 10000."); // CJ requested limit on number of LEs, 5/8/00
				break;
			}
			if(localSprayInfo.numOfLEs != sharedSprayDialogInfo.numOfLEs && sDispInfo.lassoSelectedLEsToDisperse)
			{
				short buttonSelected;
				{
					// point out that this spill is not in the time interval being modeled
					buttonSelected  = MULTICHOICEALERT(1690,"If you change the number of LEs the lasso selected LEs will be unmarked.  Are you sure you want to continue?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							sDispInfo.lassoSelectedLEsToDisperse = false;
							break;  
						case 3: // cancel
							return 0;// stay at this dialog
							break;
					}
				}
			}
			/////////////////////////////////////////////////

			if(localSprayInfo.pollutantType == OIL_CONSERVATIVE)
			{
				localSprayInfo.ageHours = -1;
				localSprayInfo.spillTime = localSprayInfo.overflightTime;
			}
			else
			{

				localSprayInfo.whenAmountIsSpecified = GetPopSelection(dialog, M15AMOUNTTIMEPOPUP);
				
				switch (localSprayInfo.whenAmountIsSpecified)
				{
					default:
					case ATTIMEOFSPILL:
						localSprayInfo.spillTime = RetrievePopTime(dialog, M15SPILLMONTH,&err);
						if (err) return 0;
						if(localSprayInfo.spillTime >= localSprayInfo.overflightTime)
						{
							printError("The Time of Spill must be before the Overflight Time.");
							return 0;
						}
						localSprayInfo.ageHours = -1;
						break;
					case ATOVERFLIGHTTIME:
						mygetitext (dialog, M15RELEASEAGEHOURS,str,255);
						if(str[0] == 0) {
							printError("Please fill in the Pollutant Age.  (The number of hours the pollutant has been in the water.)");
							MySelectDialogItemText(dialog, M15RELEASEAGEHOURS,0,100);
							return 0;
						}
						localSprayInfo.ageHours =  EditText2Long(dialog, M15RELEASEAGEHOURS);
						localSprayInfo.spillTime = localSprayInfo.overflightTime; // this way the user will have to change the value if they want to use it
						break;
				}
			}
			
			/////
			// final check
			if((!(model->bHindcast) && localSprayInfo.overflightTime != model -> GetStartTime()) || (model->bHindcast && localSprayInfo.overflightTime != model -> GetEndTime()) )
			{
				if (model->bHindcast)
				{
				
					short buttonSelected;
					if(model -> ThereIsALaterSpill(localSprayInfo.overflightTime,(TLEList*)gSprayLEListInDialog))
					{	// there is an later spill
						if (localSprayInfo.overflightTime > model -> GetEndTime())
						//if(localSprayInfo.overflightTime < model -> GetStartTime())
						{
							// they are already in trouble, so they are on their own
						}
						else if (localSprayInfo.overflightTime < model -> GetStartTime())
						{
							// point out that this spill is not in the time interval being modeled
							buttonSelected  = MULTICHOICEALERT(1690,"The Overflight Time is outside of the time interval being modeled.  Are you sure you want to continue?",TRUE);
							switch(buttonSelected){
								case 1:// continue
									break;  
								case 3: // cancel
									return 0;// stay at this dialog
									break;
							}
						}
					}
					else 
					{	// no other spill is later than this one
						char msg[256];
						strcpy(msg,"Normal usage is to set the Model End Time to the Overflight Time. Change Model End Time to Overflight Time?");
						
						buttonSelected  = MULTICHOICEALERT(1682,msg,TRUE);
						switch(buttonSelected){
							case 1:// change
								model -> SetStartTime(localSprayInfo.overflightTime - model->GetDuration());
								model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
								break;  
							case 3: // don't change
								break;
							case 4: // cancel
								return 0;// stay at this dialog
								break;
						}
					}
				}
				else
				{
					short buttonSelected;
						if(model -> ThereIsAnEarlierSpill(localSprayInfo.overflightTime,(TLEList*)gSprayLEListInDialog))
						{	// there is an earlier spill
							if(localSprayInfo.overflightTime < model -> GetStartTime())
							{
								// they are already in trouble, so they are on their own
							}
							else if (localSprayInfo.overflightTime > model -> GetEndTime())
							{
								// point out that this spill is not in the time interval being modeled
								buttonSelected  = MULTICHOICEALERT(1690,"The Overflight Time is outside of the time interval being modeled.  Are you sure you want to continue?",TRUE);
								switch(buttonSelected){
									case 1:// continue
										break;  
									case 3: // cancel
										return 0;// stay at this dialog
										break;
								}
							}
						}
						else 
						{	// no other spill is earlier than this one
							buttonSelected  = MULTICHOICEALERT(1680,"",TRUE);
							switch(buttonSelected){
								case 1:// change
									model -> SetStartTime(localSprayInfo.overflightTime);
									model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
									break;  
								case 3: // don't change
									break;
								case 4: // cancel
									return 0;// stay at this dialog
									break;
							}
						}
					}
				
			}
			/////////////////////////////
			
			// dispersed oil 10/2/01
			if (sDispInfo.bDisperseOil || gAdiosDataH)
				localSprayInfo.density = 141.5/(sDispInfo.api+131.5);
			else
				localSprayInfo.density = GetPollutantDensity(localSprayInfo.pollutantType);
			//

			mygetitext(dialog, M15SPILLNAME, localSprayInfo.spillName, kMaxNameLen - 1);		// get the mover's nameStr

			if (!err) sharedSprayDialogInfo = localSprayInfo;

			if (/*gAdiosDataH && */!err && (gAdiosDataH != sAdiosInfoH)) 
			{
				if (sAdiosInfoH)
				{
					DisposeHandle((Handle)sAdiosInfoH); 
				}
				sAdiosInfoH = gAdiosDataH;
				strcpy(sOilTypeName,gOilTypeName);
			}
			return M15OK;

		case M15CANCEL: 
		{
			if (gAdiosDataH && (gAdiosDataH != sAdiosInfoH))
			{
				DisposeHandle((Handle)gAdiosDataH); 
				gAdiosDataH = 0;
			}
			return M15CANCEL;
		}

		case M15STARTDAY:
		case M15STARTHOURS:
		case M15STARTMINUTES:
		case M15SPILLDAY:
		case M15SPILLHOURS:
		case M15SPILLMINUTES:
		case M15NUMLES:
		case M15RELEASEAGEHOURS:
			CheckNumberTextItem(dialog, itemNum, FALSE);
			break;
		
		case M15MASS:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M15MASSUNITS:
		case M15STARTMONTH:
		case M15STARTYEAR:
		case M15SPILLMONTH:
		case M15SPILLYEAR:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
			
		case M15POLLUTANT:
		case M15AMOUNTTIMEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideSprayLEDialogItems(dialog);
			break;

		case M15DISPERSANT:
		{
			short theType;
			long menuID_menuItem;
			PopClick(dialog, itemNum, &menuID_menuItem);
			theType = GetPopSelection (dialog, M15DISPERSANT);
			if (theType==1)
			{
				// selected No Dispersants, leave current values as defaults 
				DisposeSprayDialogDataH();
				sDispInfo.bDisperseOil = 0;
				sDispInfo.lassoSelectedLEsToDisperse = false;
				break;
			}
			else if (theType==2)
			{
				Boolean userCanceledOrErr;
				DispersionRec dispersantData = sDispInfo;
				userCanceledOrErr = DispersionDialog(&dispersantData,sharedSprayDialogInfo.overflightTime,GetDialogWindow(dialog));
				// refresh pop-ups
				RegisterPopTable (dsprayLEPopTable, sizeof (dsprayLEPopTable) / sizeof (PopInfoRec));
				if(userCanceledOrErr)
				{	// leave in previous state
					//SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
					if (gAdiosDataH)
						SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 3);
					else
						SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
					PopDraw(dialog, M15DISPERSANT);
				}
				else
				{
					//if (dispersantData.timeToDisperse>=sharedSprayDialogInfo.spillTime) // absolute time, rather than hours after spill
						//dispersantData.timeToDisperse -= sharedSprayDialogInfo.spillTime;
					sDispInfo = dispersantData;
					sDispInfo.bDisperseOil = 1;
					//sharedSprayDialogInfo.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
					//Float2EditText (dialog, M15NUMLES, 10000, 0);// may want to go back to default if turn off dispersion ?
				}
				DisposeSprayDialogDataH();

				break;
			}
			else if (theType == 3) // natural dispersants (info from Adios Budget Table)
			{
				char path[256];
				if (gAdiosDataH) // check that user really wants to delete old file
				{
					// point out there is already a file selected
					short buttonSelected  = MULTICHOICEALERT(1688,"There is already an active Adios Budget Table. Do you want to read in a different one?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							break;  
						case 3: // cancel
						{
							sDispInfo.bDisperseOil = 0;
							sDispInfo.lassoSelectedLEsToDisperse = false;
							goto doneAdiosFile;
							break;
						}
					}
				}
				// should allow user to keep file 
				if (err = GetAdiosFileName(path)) break;
				if (IsAdiosBudgetTable(path))
				{
					char oilType[256];
					float apiNum=0;
					DisposeSprayDialogDataH(); // will lose it if file is messed up
					err = ReadAdiosBudgetTable(path,&gAdiosDataH,oilType,&apiNum);
					// if successful should set dispersant info, else leave in previous state
					if(err)
					{	// leave in previous state
						if (gAdiosDataH)
							SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 3);
						else
							SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
						PopDraw(dialog, M15DISPERSANT);
					}
					else
					{
						sDispInfo.bDisperseOil = 0;
						sDispInfo.lassoSelectedLEsToDisperse = false;
						//sharedSprayDialogInfo.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
						//Float2EditText (dialog, M15NUMLES, 10000, 0);// may want to go back to default if turn off dispersion ?
						strcpy(gOilTypeName,oilType);
						sDispInfo.api = apiNum;
					}
				}
				else
				{
					printError("Selected file is not an adios budget table");
					break;
				}
				break;
			}
			else if (theType==4) //chemical and natural dispersants
			{
				Boolean userCanceledOrErr;
				DispersionRec dispersantData = sDispInfo;
				if (sDispInfo.lassoSelectedLEsToDisperse)
				{
					short buttonSelected  = MULTICHOICEALERT(1688,"Do you want to unmark the lasso selected LEs?",TRUE);
					switch(buttonSelected){
						case 1:// continue
							userCanceledOrErr = DispersionDialog(&dispersantData,sharedSprayDialogInfo.overflightTime,GetDialogWindow(dialog));
							dispersantData.lassoSelectedLEsToDisperse = false;
							break;  
						case 3: // cancel
						{
							userCanceledOrErr = 0;
							break;
						}
					}
				}
				else
					userCanceledOrErr = DispersionDialog(&dispersantData,sharedSprayDialogInfo.overflightTime,GetDialogWindow(dialog));
				//userCanceledOrErr = DispersionDialog(&dispersantData,sharedSprayDialogInfo.overflightTime,dialog);
				// refresh pop-ups
				//RegisterPopTable (dlePopTable, sizeof (dlePopTable) / sizeof (PopInfoRec));
				if(userCanceledOrErr)
				{	// leave in previous state
					goto doneAdiosFile;
				}
				else
				{
					char path[256];
					if (gAdiosDataH) 
					{
						// point out there is already a file selected, which will be lost
						short buttonSelected  = MULTICHOICEALERT(1688,"There is already an active Adios Budget Table. Do you want to read in a different one?",TRUE);
						switch(buttonSelected){
							case 1:// continue
								break;  
							case 3: // cancel
							{
								sDispInfo = dispersantData;
								sDispInfo.bDisperseOil = 1;
								goto doneAdiosFile;
								break;
							}
						}
					}
					if (err = GetAdiosFileName(path)) goto doneAdiosFile;
					if (IsAdiosBudgetTable(path))
					{
						char oilType[256];
						float apiNum=0;
						DisposeSprayDialogDataH(); // will lose it if file is messed up
						err = ReadAdiosBudgetTable(path,&gAdiosDataH,oilType,&apiNum);
						// if successful should set dispersant info, else leave in previous state
						if(err)
						{	// leave in previous state
							if (gAdiosDataH)
								SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 3);
							else
								SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
							PopDraw(dialog, M15DISPERSANT);
						}
						else
						{
							// set the chemical dispersion information
							sDispInfo = dispersantData;
							sDispInfo.bDisperseOil = 1;
							//sharedSprayDialogInfo.numOfLEs = 10000;	// this causes problems with lasso saving LERecH
							//Float2EditText (dialog, M15NUMLES, 10000, 0);// may want to go back to default if turn off dispersion ?
							strcpy(gOilTypeName,oilType);
							sDispInfo.api = apiNum;
						}
					}
					else
					{
						printError("Selected file is not an adios budget table");
						break;
					}

				}
				break;
			}
doneAdiosFile:
			if (gAdiosDataH)
				SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 3);
			else
				SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
			PopDraw(dialog, M15DISPERSANT);
			break;
		}
		case M15WINDAGE:
		{
			WindageRec windageData = sWindageInfo;
			OSErr err = WindageSettingsDialog(&windageData,GetDialogWindow(dialog));
			RegisterPopTable (dsprayLEPopTable, sizeof (dsprayLEPopTable) / sizeof (PopInfoRec));
			if (!err) sWindageInfo = windageData;
			break;
		}
	}
	return 0;
}

OSErr SprayLEInit (DialogPtr dialog, VOIDPTR data)
{
#pragma unused (data)
	DateTimeRec	time;
	float arrowDepth;
	char blankStr[32];
	strcpy(blankStr,"");	// latest developer studio doesn't accept passing in string itself, needs a variable 2/25/05

	gAdiosDataH = sAdiosInfoH;
	
	SetDialogItemHandle(dialog, M15HILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M15FROST1, (Handle) FrameEmbossed);
	SetDialogItemHandle(dialog, M15FROST2, (Handle) FrameEmbossed);

	if(UseExtendedYears()) {
		dsprayLEPopTable[3].menuID = pYEARS_EXTENDED;
		dsprayLEPopTable[6].menuID = pYEARS2_EXTENDED;
	}
	else {
		dsprayLEPopTable[3].menuID = pYEARS;
		dsprayLEPopTable[6].menuID = pYEARS2;
	}
	if (gDispersedOilVersion)
	{
		dsprayLEPopTable[0].menuID = pPOLLUTANTS2;
	}
	else
	{
		dsprayLEPopTable[0].menuID = pPOLLUTANTS;
	}
	RegisterPopTable (dsprayLEPopTable, sizeof (dsprayLEPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M15, dialog);

	SetPopSelection (dialog, M15AMOUNTTIMEPOPUP, sharedSprayDialogInfo.whenAmountIsSpecified);

	SecondsToDate (sharedSprayDialogInfo.overflightTime, &time);
	SetPopSelection (dialog, M15STARTMONTH, time.month);
	SetPopSelection (dialog, M15STARTYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M15STARTDAY, time.day);
	Long2EditText (dialog, M15STARTHOURS, time.hour);
	Long2EditText (dialog, M15STARTMINUTES, time.minute);

	//if(sharedSprayDialogInfo.totalMass < 0) mysetitext(dialog, M15MASS, ""); // BLANK
	if(sharedSprayDialogInfo.totalMass < 0) mysetitext(dialog, M15MASS, blankStr); // BLANK
	else Float2EditText (dialog, M15MASS, sharedSprayDialogInfo.totalMass, 0);
	SetPopSelection (dialog, M15MASSUNITS, sharedSprayDialogInfo.massUnits);

	//if(sharedSprayDialogInfo.ageHours < 0) mysetitext(dialog, M15RELEASEAGEHOURS, ""); // BLANK
	if(sharedSprayDialogInfo.ageHours < 0) mysetitext(dialog, M15RELEASEAGEHOURS, blankStr); // BLANK
	else Long2EditText (dialog, M15RELEASEAGEHOURS, sharedSprayDialogInfo.ageHours);

	Long2EditText (dialog, M15NUMLES, sharedSprayDialogInfo.numOfLEs);
	SetPopSelection (dialog, M15POLLUTANT, sharedSprayDialogInfo.pollutantType);

	mysetitext(dialog, M15SPILLNAME, sharedSprayDialogInfo.spillName);

	if (model -> GetModelMode () < ADVANCEDMODE)
	{
		ShowHideDialogItem(dialog, M15LELABEL, false);
		ShowHideDialogItem(dialog, M15NUMLES,  false);
	}
	
	SecondsToDate (sharedSprayDialogInfo.spillTime, &time);
	SetPopSelection (dialog, M15SPILLMONTH, time.month);
	SetPopSelection (dialog, M15SPILLYEAR,  time.year - (FirstYearInPopup()  - 1));
	Long2EditText (dialog, M15SPILLDAY, time.day);
	Long2EditText (dialog, M15SPILLHOURS, time.hour);
	Long2EditText (dialog, M15SPILLMINUTES, time.minute);
	
	ShowHideSprayLEDialogItems(dialog);

	if (sAdiosInfoH)
		SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 3);
	else
		SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
	//SetPopSelection(dialog, M15DISPERSANT, sDispInfo.bDisperseOil + 1);
	ShowHideDialogItem(dialog, M15DISPERSANT, model->ThereIsA3DMover(&arrowDepth)); // if there is a ptcurmap, but not a 2D grid...
	ShowHideDialogItem(dialog, M15WINDAGE, model->GetModelMode() == ADVANCEDMODE); 

	MySelectDialogItemText(dialog, M15MASS, 0, 255);

	return 0;
}

OSErr OverflightSettingsDialog (SprayDialogInfo *sprayDialogInfo, DispersionRec *dialogDispInfo, WindageRec *windageInfo, AdiosInfoRecH *adiosInfo, char *oilTypeName)
{	// the calling routine is responsible for calling dirty
	short 		dialogItem;
	TOLEList 	*uncertaintyLEList=0;
	Boolean weCreatedThisLEList = false;
	Boolean weCreatedUncertaintyLEList = false;
	OSErr err = 0;
	
	sharedSprayDialogInfo = *sprayDialogInfo;
	sDispInfo = *dialogDispInfo;
	sWindageInfo = *windageInfo;
	sAdiosInfoH = *adiosInfo;
	strcpy(sOilTypeName,oilTypeName);
	dialogItem = MyModalDialog(M15, mapWindow, nil, SprayLEInit, SprayLEClick);
	if (dialogItem == M15CANCEL)
		return M15CANCEL;
	else if (dialogItem == M15OK)
	{
		*sprayDialogInfo = sharedSprayDialogInfo;
		*dialogDispInfo = sDispInfo;
		*windageInfo = sWindageInfo;
		*adiosInfo = sAdiosInfoH;
		strcpy(oilTypeName,sOilTypeName);
		// the calling routine is responsible for calling dirty
		return 0;
	}
	
	return err;
}

OSErr  DoCJFile(char *path)
{
	long i,j;
	SprayDialogInfo	sprayDialogInfo;
	WindageRec windageDialogInfo;
	DispersionRec dispDialogInfo;
	//AdiosInfoRecH adiosDialogInfo = 0;
	WorldPoint wp;
	long displayWidth,displayHeight;
	long totalNumLEs,itemHit;
	char defaultAns[256];
	char ans[256],msg[256];
	OSErr err = 0;
	long numPtsAcrossScreen = 40;
	long maxPtsAcrossScreen = 100;
	TSprayLEList *list=nil,*uncertaintyLEList = nil;
	
	WorldRect currentView = settings.currentView;  
	Rect currentMapDrawingRect = MapDrawingRect();


	sprintf(defaultAns,"%ld",numPtsAcrossScreen);
	while (1)
	{
		// ask for numPtsAcrossScreen
		itemHit = REQUEST("Enter number of LEs across the screen",defaultAns,ans);
		//if (itemHit == cancel) return 0;	// user cancelled
		if (itemHit == 2) return 0;	// user cancelled
		short numScanned = sscanf(ans,"%ld",&numPtsAcrossScreen);
		if(! numScanned == 1) 
		{
			printError("Error reading number of LEs. Invalid entry.");
			return -1;
		}
	
		if(numPtsAcrossScreen > 0 && numPtsAcrossScreen <= maxPtsAcrossScreen) // valid entry, go on
			break;
		
		else // too many points crashes the Mac 4/11/00, go back to dialog
		{
			if (numPtsAcrossScreen > maxPtsAcrossScreen) sprintf(msg,"Maximum number of points is %ld",maxPtsAcrossScreen);
			if (numPtsAcrossScreen <= 0) sprintf(msg,"Number of points must be positive");
		
			printNote(msg); 
		}
	}


	// create the spray sets
	// ?? create uncertainty ones ?
	list = new TSprayLEList();
	if (!list) { OutOffMemoryAlert(); err= -1; goto done;}
	
	memset(&sprayDialogInfo,0,sizeof(sprayDialogInfo));

	sprayDialogInfo.numOfLEs = 0;
	sprayDialogInfo.pollutantType = OIL_CONSERVATIVE;
	sprayDialogInfo.totalMass = 0;
	sprayDialogInfo.massUnits = BARRELS;
	sprayDialogInfo.density = 1;
	sprayDialogInfo.overflightTime = model -> GetStartTime ();
	sprayDialogInfo.spillTime = model->GetStartTime(); 
	sprayDialogInfo.ageHours = 0; 
	sprayDialogInfo.whenAmountIsSpecified = ATOVERFLIGHTTIME;
	//strcpy(sprayDialogInfo.spillName,"LE Grid");
	
	err = list->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;
		
	uncertaintyLEList = new TSprayLEList ();
	if (!uncertaintyLEList) { OutOffMemoryAlert(); err= -1; goto done; }
	uncertaintyLEList->fLeType = UNCERTAINTY_LE;
	uncertaintyLEList->fOwnersUniqueID = list->GetUniqueID();
	
	err = uncertaintyLEList->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;
	
	// set default dispersion parameters
	dispDialogInfo.bDisperseOil = 0.;
	dispDialogInfo.amountToDisperse = 1.;	 
	dispDialogInfo.timeToDisperse = 0.;
	dispDialogInfo.duration = 3600.;
	dispDialogInfo.api = 10.;
	dispDialogInfo.lassoSelectedLEsToDisperse = false;
	dispDialogInfo.areaToDisperse = emptyWorldRect;	// GetMapBounds
	{	
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map)
		{
			dispDialogInfo.areaToDisperse = map -> GetMapBounds();
		}
	}
	
	list->SetDispersionInfo(dispDialogInfo);

	// windage defaults should be set automatically
	//(*(TOLEList*)list).fWindageData.windageA = .01;
	//(*(TOLEList*)list).fWindageData.windageB = .04;
	//(*(TOLEList*)list).fWindageData.persistence = .25;	// in hours
	// fill in blue dots array as well as the LE array

	// look at Movement to find land pts  - look at MapBounds first?
	displayWidth = RectWidth(currentMapDrawingRect);
	displayHeight = RectHeight(currentMapDrawingRect);
	totalNumLEs = 0;
	for (i = 0; i < numPtsAcrossScreen; i++)
	{
		Point pixel;
		pixel.h = currentMapDrawingRect.left+ round(displayWidth*(i+1)/(float)(numPtsAcrossScreen+1));
		for (j = 0; j < numPtsAcrossScreen; j++)
		{
			pixel.v = currentMapDrawingRect.top + round(displayHeight*(j+1)/(float)(numPtsAcrossScreen+1));
			wp = ScreenToWorldPoint(pixel,currentMapDrawingRect,currentView);
			if(model->IsWaterPoint(wp)) {
				err = list ->AddSprayPoint(wp);
				if(err) goto done;
				totalNumLEs++;
			}
		}
	}

	sprayDialogInfo.numOfLEs = totalNumLEs;
	sprayDialogInfo.totalMass = totalNumLEs;
	
	model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
	// since the model might be in negative time, in which case the LE's would be deleted and
	// readded from the files, which do not contain these new LE sets
	// Note: we want to do this before Initialize() is called
	// because it uses GetModelTime() -- JLM 10/18/00

	err = list->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;
	err = uncertaintyLEList->SetSprayDialogInfo(sprayDialogInfo);
	if(err) goto done;

	model->NewDirtNotification();

	err = model->AddLEList(list, 0);
	if(err) goto done;
	err = model->AddLEList(uncertaintyLEList, 0);
	if(err) goto done;
	
	{	// force user to go through map dialog if they dispersed the spill
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map)
		{
				if (dispDialogInfo.bDisperseOil)	// also force through map dialog for a bottom spill?
				if (err = TMapSettingsDialog(map)) goto done;
		}
	}


	return 0;
done:
	if(err)
	{
		if(list) {list -> Dispose(); delete list; list = 0;}
		if(uncertaintyLEList) {uncertaintyLEList -> Dispose(); delete uncertaintyLEList; uncertaintyLEList = 0;}
	}
	
	return err;
}

OSErr CreateSprayLESet(void)
{
	TSprayLEList 	*list=nil,*uncertaintyLEList = nil;
	OSErr err = 0;
	SprayDialogInfo	sprayDialogInfo;
	DispersionRec dispDialogInfo;
	WindageRec windageDialogInfo;
	AdiosInfoRecH adiosDialogInfo = 0;
	char oilTypeName[256];
	long numMaps = 0;
	
	numMaps = model -> mapList -> GetItemCount();
	
	if (numMaps <= 0) {
		printError("You cannot spray splots because there is no map.");
		return -1;
	}
	
	memset(&sprayDialogInfo,0,sizeof(sprayDialogInfo));
	//memset(&windageDialogInfo,0,sizeof(windageDialogInfo));

	sprayDialogInfo.numOfLEs =1000;
	sprayDialogInfo.pollutantType = OIL_CONSERVATIVE;
	if (model->GetModelMode() == ADVANCEDMODE) sprayDialogInfo.totalMass = 1;// give diagnostic mode users a default value 2/25/05
	else	sprayDialogInfo.totalMass = -1;// makes the user fill in the value
	sprayDialogInfo.massUnits = BARRELS;
	sprayDialogInfo.overflightTime = model -> GetStartTime ();
	sprayDialogInfo.spillTime = sprayDialogInfo.overflightTime; // this way the user will have to change the value if they want to use it
	sprayDialogInfo.ageHours = -1; // makes the user fill in the value
	sprayDialogInfo.whenAmountIsSpecified = ATOVERFLIGHTTIME;
	//strcpy(sprayDialogInfo.spillName,"Overflight");
	
	windageDialogInfo.windageA = .01;
	windageDialogInfo.windageB = .04;
	windageDialogInfo.persistence = .25;	// in hours
	
	dispDialogInfo.bDisperseOil = 0.;
	dispDialogInfo.amountToDisperse = 1.;	 
	dispDialogInfo.timeToDisperse = 0.;
	dispDialogInfo.duration = 3600.;
	dispDialogInfo.api = 10.;
	dispDialogInfo.lassoSelectedLEsToDisperse = false;
	dispDialogInfo.areaToDisperse = emptyWorldRect;	// GetMapBounds
	{	
		PtCurMap *map = GetPtCurMap();	// still could be 2D...
		if (map)
		{
			dispDialogInfo.areaToDisperse = map -> GetMapBounds();
		}
	}
	oilTypeName[0]=0;
	
	err = OverflightSettingsDialog (&sprayDialogInfo,&dispDialogInfo,&windageDialogInfo,&adiosDialogInfo,oilTypeName);
	
	if(!err)
	{	// the user said OK, so create the objects

		list = new TSprayLEList();
		if (!list) { TechError("CreateSprayLESet()", "new TSprayLEList()", 0); err= -1; goto done;}
		
		err = list->SetSprayDialogInfo(sprayDialogInfo);
		if(err) goto done;
		list->SetDispersionInfo(dispDialogInfo);
		list->SetWindageInfo(windageDialogInfo);
		list->SetAdiosInfo(adiosDialogInfo);
		strcpy(list->fOilTypeName,oilTypeName);
			
		uncertaintyLEList = new TSprayLEList ();
		if (!uncertaintyLEList) { TechError("CreateSprayLESet()", "new TSprayLEList()", 0); err = -1; goto done; }
		uncertaintyLEList->fLeType = UNCERTAINTY_LE;
		uncertaintyLEList->fOwnersUniqueID = list->GetUniqueID();
		
		err = uncertaintyLEList->SetSprayDialogInfo(sprayDialogInfo);
		if(err) goto done;

		uncertaintyLEList->SetDispersionInfo(dispDialogInfo);
		uncertaintyLEList->SetWindageInfo(windageDialogInfo);
		uncertaintyLEList->SetAdiosInfo(adiosDialogInfo);
		strcpy(uncertaintyLEList->fOilTypeName,oilTypeName);

		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		// since the model might be in negative time, in which case the LE's would be deleted and
		// readded from the files, which do not contain these new LE sets
		model->NewDirtNotification();

		err = model->AddLEList(list, 0);
		if(err) goto done;
		err = model->AddLEList(uncertaintyLEList, 0);
		if(err) goto done;
	}
	
done:
	if(err)
	{
		if(list) {list -> Dispose(); delete list; list = 0;}
		if(uncertaintyLEList) {uncertaintyLEList -> Dispose(); delete uncertaintyLEList; uncertaintyLEList = 0;}
	}
	
	return err;

}

/////////////////////////////////////////////////
/////////////////////////////////////////////////


TSprayLEList::TSprayLEList()
{
	fSprayedH = 0;
	fNumSprayedPts = 0;
	//
	fPixelBitsH = 0;
	fPixelWorldRect = emptyWorldRect;
	memset(&fPixelScreenRect,0,sizeof(fPixelScreenRect));
	fInPixelEditMode = FALSE;
	//
	fWhenAmountIsSpecified = ATOVERFLIGHTTIME;
	fSpillTime = 0;
	fAgeAtOverflightTimeInHoursEnteredByUser = -1;
	fAmountEnteredByUser = 0;
}

void TSprayLEList::Dispose ()
{
	this -> DisposeSprayedPoints();
	this -> DisposePixelBits();
	/////
	TOLEList::Dispose ();
}

OSErr TSprayLEList::Reset(Boolean newKeys)
{
	OSErr err = TOLEList::Reset(newKeys);
	TSprayLEList *sprayedHOwner = this;
	if(err) return err;
	
	
	if (this->fLeType == UNCERTAINTY_LE)
	{ // then we need to use the sprayed points kept in our certainty buddy
		sprayedHOwner =  (TSprayLEList *) model -> GetLEListOwner(this);
		if (!sprayedHOwner)
			return 0; // this will happen on the resets called before we are put in the model's list
	}
	
	// override the LE positions
	if(this -> LEHandle && sprayedHOwner && sprayedHOwner -> fSprayedH)
	{
		long i,j;
		WorldPoint wp;
		LERec leRec;
	
		for(i = 0; i < this -> numOfLEs; i++)
		{
			j = sprayedHOwner -> LEIndexToSprayedIndex(i);
			if(j < 0) 
				return -1; // there is some problem
			wp = INDEXH(sprayedHOwner -> fSprayedH ,j);
			leRec = INDEXH(this -> LEHandle,i);
			leRec.p = wp;
			INDEXH(this -> LEHandle,i) = leRec;
		}
	}

	return 0;
}


double TSprayLEList::AgeInHrsAtOverflightTime()
{
	// IMPORTANT NOTE !!
	// This code is duplicated in SetSprayDialogInfo()
	// so BE CAREFUL when changing this code

	if(fWhenAmountIsSpecified == ATOVERFLIGHTTIME)
	{
		return fAgeAtOverflightTimeInHoursEnteredByUser;
	}
	else if(fWhenAmountIsSpecified == ATTIMEOFSPILL)
	{
		return (fSetSummary.startRelTime - fSpillTime)/3600.;
	}
	else
	{
		printError("Bad whenAmountIsSpecified in TSprayLEList::AgeInHrsAtOverflightTime()");
		return -1;
	}
}


double TSprayLEList::AmountAtOverflightTime()
{
	// IMPORTANT NOTE !!
	// This code is duplicated in SetSprayDialogInfo()
	// so BE CAREFUL when changing this code
	
	if(fWhenAmountIsSpecified == ATOVERFLIGHTTIME)
	{
		return fAmountEnteredByUser;
	}
	else if(fWhenAmountIsSpecified == ATTIMEOFSPILL)
	{
		double amount = fAmountEnteredByUser;
		double ageInHrs = this->AgeInHrsAtOverflightTime();
		double fractionLeft = FractionOilLeftAtTime(fSetSummary.pollutantType,ageInHrs);
		// calculate the amount after evaporation
		amount =  amount * fractionLeft;
		return amount;
	}
	else
	{
		printError("Bad whenAmountIsSpecified in TSprayLEList::AmountAtOverflightTime()");
		return -1;
	}
}

long TSprayLEList::LEIndexToSprayedIndex(long leIndex)
{ 	// this function tells us which sprayed point
	// a given LE is associated with (i.e. is initialized to)
	// returns a negative index if there is a problem
	long sprayIndex = -1;
	
	// The important thing is that we get the same LE's each time we reset

	if(fSprayedH && fNumSprayedPts > 0) {
		double frac = 0;
		long maxIndex = this -> numOfLEs;

		if (maxIndex > 0)
			frac = leIndex/(float)(maxIndex -1);
		
		sprayIndex = round(frac*(fNumSprayedPts-1));
		
		// just paranoid, enforce limits
		if(sprayIndex < 0) 
			sprayIndex = 0;
		if(sprayIndex > fNumSprayedPts-1) 
			sprayIndex = fNumSprayedPts-1;
	}
	
	return sprayIndex;
}

OSErr TSprayLEList::Initialize(LESetSummary *summary,Boolean deleteLERecH)
{
	OSErr err = 0;
	err = TOLEList::Initialize(summary,deleteLERecH);
	if(err) return err;
	
	err = this -> Reset(TRUE);
	
	return err;
}



#define kTSprayLEListVersion 1

OSErr TSprayLEList::Write(BFPB *bfpb)
{
	long i, version = kTSprayLEListVersion;
	ClassID id = GetClassID ();
	LERec LE;
	OSErr	err = noErr;
	WorldPoint	wp;

	if(err = TOLEList::Write(bfpb)) return err;
	
	StartReadWriteSequence("TSprayLEList::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
//	if (err = WriteMacValue(bfpb, bOpen)) return err;

	if (err = WriteMacValue(bfpb, fNumSprayedPts)) return err;

	if (fNumSprayedPts && fSprayedH)
	{
		for(i = 0; i < fNumSprayedPts; i++)
		{
			wp = INDEXH(fSprayedH,i);
			if (err = WriteMacValue(bfpb, wp)) return err;
		}
	}
	
	// no need to read/write "Pixel" fields 
	// fPixelBitsH,fPixelWorldRect,fPixelScreenRect;
	// because we recreate them when needed

	if (err = WriteMacValue(bfpb, fWhenAmountIsSpecified)) return err;
	if (err = WriteMacValue(bfpb, fSpillTime)) return err;
	if (err = WriteMacValue(bfpb, fAgeAtOverflightTimeInHoursEnteredByUser)) return err;
	if (err = WriteMacValue(bfpb, fAmountEnteredByUser)) return err;

	return 0;
}

OSErr TSprayLEList::Read(BFPB *bfpb)
{
	char 	c;
	long 	i, version;
	ClassID id;
	LERec 	LE;
	OSErr	err = noErr;
	long	LERecSize;
	WorldPoint	wp;

	this -> Dispose();

	if(err = TOLEList::Read(bfpb)) return err;
	
	StartReadWriteSequence("TSprayLEList::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TSprayLEList::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version != kTSprayLEListVersion) { printSaveFileVersionError(); return -1; }

	if (err = ReadMacValue(bfpb, &fNumSprayedPts)) return err;

	if (fNumSprayedPts > 0)
	{
		fSprayedH = (WORLDPOINTH) _NewHandle(fNumSprayedPts * sizeof(**fSprayedH));
		if (!fSprayedH) 
			return memFullErr; 

		for(i = 0; i < fNumSprayedPts; i++)
		{
			if(err = ReadMacValue (bfpb, &wp)) return err;
			INDEXH(fSprayedH,i) = wp;
		}
	}
	
	// no need to read/write "Pixel" fields 
	// fPixelBitsH,fPixelWorldRect,fPixelScreenRect;
	// because we recreate them when needed
	
	if (err = ReadMacValue(bfpb, &fWhenAmountIsSpecified)) return err;
	if (err = ReadMacValue(bfpb, &fSpillTime)) return err;
	if (err = ReadMacValue(bfpb, &fAgeAtOverflightTimeInHoursEnteredByUser)) return err;
	if (err = ReadMacValue(bfpb, &fAmountEnteredByUser)) return err;

	return err;
}


long TSprayLEList::GetListLength()
{
	long summaryLines = 0;
	Boolean showAgeInfo = !(fSetSummary.pollutantType == OIL_CONSERVATIVE);
	Boolean showTimeOfSpillInfo = fWhenAmountIsSpecified == ATTIMEOFSPILL;
	
	Boolean owned = (fOwnersUniqueID.counter != 0);
	if(owned) return 0; // we will not be in the list
	
	summaryLines++; //oil type as a toggle
	
	if(bOpen)
	{
		if(model->GetModelMode() == ADVANCEDMODE) summaryLines++; // Active button

		if(model->GetModelMode() == ADVANCEDMODE) summaryLines += 1;// windage

		if (fDispersantData.bDisperseOil) summaryLines++;
		//if (fDispersantData.bDisperseOil) summaryLines++;	// draw disperse area

		if (fAdiosDataH) summaryLines++;	// natural dispersion
	
		if(showTimeOfSpillInfo) summaryLines++; // time of spill
		if(showTimeOfSpillInfo) summaryLines++; // amount at time of spill
		
		summaryLines++; // overflight time
		summaryLines++; // amount at overflight time
		if(showAgeInfo) summaryLines++; // age at overflight time
	
		summaryLines++; // Mass Balance toggle
		
		if (bMassBalanceOpen)
		{
			summaryLines++;// released 
			summaryLines++;// floating
			summaryLines++;// beached
			summaryLines++;// evaporated
			if (fDispersantData.bDisperseOil || fAdiosDataH) summaryLines++;// dispersed
			summaryLines++;// offmap
			// code goes here, removed?
		}
				
	}
	
	return summaryLines;
}

ListItem TSprayLEList::GetNthListItem(long n, short indent, short *style, char *text)
{
	char dateStr [64],infoStr [255];
	char polutantName [64];
	char unitsStr [64];
	char latString[20],longString[20];
	char valStr[20], valStr2[20], valStr3[20];
	long i,degLat,degLong;
	float minutesLat,minutesLong;
	char roundLat,roundLong;
	LERec LE;
	ListItem item = { this, 0, indent, 0 };
	Boolean showAgeInfo = !(fSetSummary.pollutantType == OIL_CONSERVATIVE);
	Boolean showTimeOfSpillInfo = fWhenAmountIsSpecified == ATTIMEOFSPILL;
	double val;
	
	if (n == 0) {
		//oil type and amount as a toggle
		item = TOLEList::GetNthListItem(n, indent, style, text);
		GetPollutantName (fSetSummary.pollutantType, polutantName);
		//sprintf(text, "%s : Overflight", polutantName);
		//sprintf(text, "%s : %s : Overflight", fSetSummary.spillName, polutantName);
		sprintf(text, "%s : %s", fSetSummary.spillName, polutantName);
		if(!bActive)*style = italic; // JLM 6/14/10
		return item;
	}
	n -= 1;
	
	
	if (bOpen)
	{
		
		if (model->GetModelMode() == ADVANCEDMODE)
		{
			if (n == 0) {
				item.indent++;
				item.index = I_SP_ACTIVE;
				item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Active");
				return item;
			}
			n -= 1;
	
			////
			if (n == 0) 
			{
				item.index = I_SP_WINDAGE;
				StringWithoutTrailingZeros(valStr,fWindageData.windageA * 100,2);
				StringWithoutTrailingZeros(valStr2,fWindageData.windageB * 100,2);
				if (fWindageData.persistence==-1) 
					sprintf(text, "Windage: %s%% to %s%%, Persistence: Infinite", valStr, valStr2);
				else
				{
					StringWithoutTrailingZeros(valStr3,fWindageData.persistence,2);
					sprintf(text, "Windage: %s%% to %s%%, Persistence: %s hrs", valStr, valStr2, valStr3);
				}
				return item;
			}
			else
				n -= 1;
		}
		
		if (fDispersantData.bDisperseOil)
		{
			// number  and type of LE's
			if (n == 0) {
				item.index = I_SP_DISPERSE;
				//item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				sprintf(text, "Disperse chemically after %g hrs",fDispersantData.timeToDisperse/3600.);
				return item;
			}
			n -= 1;
			//  check box to draw dispersion area
			/*if (n == 0) {
				item.index = I_SP_DRAWRECT;
				item.indent++;
				item.bullet = this->bShowDispersantArea ?  BULLET_FILLEDBOX : BULLET_EMPTYBOX;
				strcpy(text, "Show dispersant application area");
				return item;
			}
			n -= 1;*/
		}
		
		if (fAdiosDataH) 
		{
			// natural dispersion
			if (n == 0) {
				item.index = I_SP_NATURAL_DISPERSE;
				//item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				//sprintf(text, "Disperse after %g hrs",fDispersantData.timeToDisperse/3600.);
				strcpy(text, "Natural Dispersion from Adios");
				return item;
			}
			n -= 1;
		}
		
		if(showTimeOfSpillInfo)  // time of spill
		{
			if (n == 0) {
				item.index = I_SP_UNSPECIFIEDLINE;
				Secs2DateString2 (fSpillTime, dateStr);
				sprintf (text, "Time of Spill: %s", dateStr);
				return item;
			}
			n -= 1;
		}
		
		if(showTimeOfSpillInfo) // amount at time of spill
		{
			if (n == 0) {
				item.index = I_SP_UNSPECIFIEDLINE;
				val = fAmountEnteredByUser;
				GetLeUnitsStr(unitsStr,fSetSummary.massUnits);
				StringWithoutTrailingZeros(infoStr,val,6); 
				sprintf (text, "Amount at Time of Spill:  %s %s", infoStr,unitsStr);
				return item;
			}
			n -= 1;
		}
		
		if (n == 0) {
			item.index = I_SP_UNSPECIFIEDLINE;
			Secs2DateString2 (fSetSummary.startRelTime, dateStr);
			sprintf (text, "Overflight Time: %s", dateStr);
			return item;
		}
		n -= 1;
		
		
		if (n == 0) {
			item.index = I_SP_UNSPECIFIEDLINE;
			val = this->AmountAtOverflightTime();
			GetLeUnitsStr(unitsStr,fSetSummary.massUnits);
			if(val < 0.1)
				strcpy(infoStr,"< 0.1"); 
			else if (val < 100)
				StringWithoutTrailingZeros(infoStr,val,1); // 1 decimal place
			else 
				StringWithoutTrailingZeros(infoStr,val,0); // no decimal places
			sprintf (text, "Amount at Overflight Time:  %s %s", infoStr,unitsStr);
			return item;
		}
		n -= 1;
		
		
		if(showAgeInfo) // age at overflight time
		{
			if (n == 0) {
				item.index = I_SP_UNSPECIFIEDLINE;
				val = this->AgeInHrsAtOverflightTime();
				StringWithoutTrailingZeros(infoStr,val,2);
				sprintf(text, "Age At Overflight Time: %s hrs",infoStr);
				return item;
			}
			n -= 1;
		}
	

		/////////////////////////////////////////////////
		//// mass balance
		item.indent++; // all items are indented

		if (n == 0) {
			item.index = I_SP_LERELEASE_MASSBALANCE;
			//item.indent++;
			item.bullet = bMassBalanceOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			sprintf(text, "Splot Mass Balance (Best Estimate)");
			return item;
		}
		n -= 1;

		////
		if (bMassBalanceOpen) // released, evaporated, offmap, beached
		{
			////////////
			if (n == 0) {// released
				this->GetMassBalanceLines(TRUE,text,nil,nil,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// floating
				this->GetMassBalanceLines(TRUE,nil,text,nil,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// beached
				this->GetMassBalanceLines(TRUE,nil,nil,text,nil,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (n == 0) {// evaporated
				this->GetMassBalanceLines(TRUE,nil,nil,nil,text,nil,nil,nil);
				//item.indent++;
				return item;
			}
			n -= 1;

			if (fDispersantData.bDisperseOil || fAdiosDataH)
			{
				if (n == 0) {// dispersed
					this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,text,nil,nil);
					//item.indent++;
					return item;
				}
				n -= 1;
			}

			if (n == 0) {// offmap
				this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,nil,text,nil);
				//item.indent++;
				return item;
			}
			n -= 1;
		
			if (fAdiosDataH)
			{
				if (n == 0) {// dispersed
					this->GetMassBalanceLines(TRUE,nil,nil,nil,nil,nil,nil,text);
					//item.indent++;
					return item;
				}
				n -= 1;
			}
		}

	}
	
	item.owner = 0;
	
	return item;
}

Boolean TSprayLEList::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet && item.index == I_SP_LEFIRSTLINE)
		{ bOpen = !bOpen; return TRUE; }
		
	if (inBullet && item.index == I_SP_LERELEASE_MASSBALANCE)
		{ bMassBalanceOpen = !bMassBalanceOpen; return TRUE; }
	
	if (inBullet && item.index == I_SP_ACTIVE)
		{ 
			TOLEList 	*uncertaintyLEList = (TOLEList*)model->GetMirroredLEList(this);
			bActive = !bActive; 
			uncertaintyLEList->SetActive(bActive);
			model->NewDirtNotification(DIRTY_RUNBAR); model->NewDirtNotification(); return TRUE; 
		}

	/*if (inBullet && item.index == I_SP_DRAWRECT) {
		this->bShowDispersantArea = !this->bShowDispersantArea;
		InvalMapDrawingRect();
		return TRUE;
	}*/
				
	if(doubleClick)  return this -> SettingsItem (item); // do the settings
	
	return FALSE;
}

Boolean TSprayLEList::FunctionEnabled(ListItem item, short buttonID)
{
	switch (buttonID) {
		case UPBUTTON:
		case DOWNBUTTON:
			return FALSE;
	}
	
	if (item.index == I_SP_LEFIRSTLINE)
		switch (buttonID) {
			case ADDBUTTON: return FALSE;
			case SETTINGSBUTTON: return TRUE;
			case DELETEBUTTON: return TRUE;
		}
	
	switch (buttonID) {
		case ADDBUTTON: return FALSE;
		case SETTINGSBUTTON: return TRUE;
		case DELETEBUTTON: return FALSE;
	}
	
	return FALSE;
}

OSErr TSprayLEList::SettingsItem(ListItem item)
{
	SprayDialogInfo sprayDialogInfo,sprayDialogInfob4;
	DispersionRec dispDialogInfo;
	WindageRec windageDialogInfo;
	AdiosInfoRecH adiosDialogInfo = 0;
	char oilTypeName[256];
	OSErr err = 0;
	Boolean madeChanges = false;
	
	if (item.owner != this) 
		return -1;
	
	sprayDialogInfob4 = sprayDialogInfo = this -> GetSprayDialogInfo();
	//windageDialogInfo = windageDialogInfob4 =this -> GetWindageInfo(); 
	dispDialogInfo = this -> GetDispersionInfo();
	windageDialogInfo = this -> GetWindageInfo(); 
	adiosDialogInfo = this -> GetAdiosInfo();
	strcpy(oilTypeName,this->fOilTypeName);
	
	err = OverflightSettingsDialog (&sprayDialogInfo, &dispDialogInfo, &windageDialogInfo, &adiosDialogInfo,oilTypeName);
	
	//if(!memcmp(&sprayDialogInfob4,&sprayDialogInfo,sizeof(sprayDialogInfo)) && !memcmp(&windageDialogInfob4,&windageDialogInfo,sizeof(windageDialogInfo)))
		//return 0; // user made no changes - with all the new stuff this probably doesn't work anymore
		// does it really matter if we reset even if nothing was changed?
	
	if(!memcmp(&sprayDialogInfob4,&sprayDialogInfo,sizeof(sprayDialogInfo)))
		madeChanges = false;
	else
		madeChanges = true;
		//return 0; // user made no changes
	
	if (!err) 
	{
		this -> SetDispersionInfo(dispDialogInfo);
		this -> SetWindageInfo(windageDialogInfo); 
		this -> SetAdiosInfo(adiosDialogInfo);
		strcpy(this->fOilTypeName,oilTypeName);
		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		model->NewDirtNotification();
		if (madeChanges) err = this -> SetSprayDialogInfo(sprayDialogInfo);
	}

	if (!err) {
		TSprayLEList *uncertaintyLEList = 0;
		
		//err = this -> SetSprayDialogInfo(sprayDialogInfo) ;
		//this -> SetWindageInfo(windageDialogInfo); 
		
		uncertaintyLEList = (TSprayLEList*)model->GetMirroredLEList(this);
		if (uncertaintyLEList)
		{
			//err = uncertaintyLEList ->  SetSprayDialogInfo(sprayDialogInfo) ;
			uncertaintyLEList -> SetWindageInfo(windageDialogInfo);
			uncertaintyLEList -> SetDispersionInfo(dispDialogInfo);
			uncertaintyLEList -> SetAdiosInfo(adiosDialogInfo);
			strcpy(uncertaintyLEList->fOilTypeName,oilTypeName);
			if (madeChanges) err = uncertaintyLEList ->  SetSprayDialogInfo(sprayDialogInfo) ;
		}

		model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
		// since the model might be in negative time, in which case the LE's would be deleted and
		// readded from the files, which do not contain these changes
		model->NewDirtNotification();
	}
	
	return TRUE;
}

OSErr TSprayLEList::DeleteItem(ListItem item)
{
	if (item.index == I_SP_LEFIRSTLINE)
		return model->DropLEList(this, false);
	
	return 0;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////


void TSprayLEList::Draw(Rect r, WorldRect view)
{
	// code goes here, is knowing about the editing of another LE set
	// our problem or the model's problem ??
	
	if (this -> UserIsEditingMeInMapDrawingRect()) 
	{
		if(fSprayedH) 
		{
			// we are the object being edited, draw the blue spray points
			long i;
			//short x2,y2;
			Point pt2;
			Boolean offQuickDrawPlane = false;
			WorldPoint wp;
			RGBForeColor(&colors[BLUE]);
			for (i = 0 ; i < fNumSprayedPts ; i++) {
				wp = INDEXH(fSprayedH,i);
				if (WPointInWRect(wp.pLong, wp.pLat, &view))
				{
					//x2 = SameDifferenceX(wp.pLong);
					//y2 = (r.bottom + r.top) - SameDifferenceY(wp.pLat);
					pt2 = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
					
					//MyMoveTo(x2, y2);
					MyMoveTo(pt2.h, pt2.v);
					#ifdef MAC
						Line(0, 0);
					#else
						Line(1, 1);
					#endif
				}
			}
			RGBForeColor(&colors[BLACK]);
		}
	}
	else TOLEList::Draw(r,view);
}


/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

long TSprayLEList::PixelToIndex(Point pixel)
{ // returns a neg index when pixel is invalid
	long index = -1;
	this -> VerifyPixelBitmap();
	if (!fPixelBitsH) return index; // return a negative index if we have no pixel map
	if (MyPtInRect(pixel,&fPixelScreenRect))
	{
		long dh = pixel.h - fPixelScreenRect.left;
		long dv = pixel.v - fPixelScreenRect.top;
		long pixPerRow = fPixelScreenRect.right - fPixelScreenRect.left;
		index = dv*pixPerRow + dh;
	}
	return index;
}

Boolean TSprayLEList::CanSprayPixel(Point pixel)
{
	// returns TRUE if the pixel is not set
	long index = PixelToIndex(pixel);
	
	if(index >=0 && fPixelBitsH) 
	{
		Boolean bitIsSet;
		WorldPoint wp;
		_HLock((Handle)fPixelBitsH);
		bitIsSet = BitTst(DEREFH(fPixelBitsH),index);
		_HUnlock((Handle)fPixelBitsH);
		
		// bit must not be set 
		if (bitIsSet) 
			return FALSE;
		
		// the point needs to be a water point and in spillable region
		wp = ScreenToWorldPoint(pixel,fPixelScreenRect,fPixelWorldRect);
		if (!model -> IsAllowableSpillPoint(wp) && !(gNoaaVersion && model->bHindcast))
			return FALSE; 
		
		return TRUE;// it passes all the tests
			
	}
	return FALSE;
}

void TSprayLEList::SetPixel(Point pixel)
{
	long index = PixelToIndex(pixel);
	if(index >=0 && fPixelBitsH) 
	{
		_HLock((Handle)fPixelBitsH);
		BitSet(DEREFH(fPixelBitsH),index);
		_HUnlock((Handle)fPixelBitsH);
	}
}

void TSprayLEList::ClearPixel(Point pixel)
{
	long index = PixelToIndex(pixel);
	if(index >=0 && fPixelBitsH) 
	{
		_HLock((Handle)fPixelBitsH);
		BitClr(DEREFH(fPixelBitsH),index);
		_HUnlock((Handle)fPixelBitsH);
	}
}

Boolean TSprayLEList::OilAtPixel(Point pixel)
{
	long index = PixelToIndex(pixel);
	if (index >=0 && fPixelBitsH) {
		Boolean val;
		_HLock((Handle)fPixelBitsH);
		val = BitTst(DEREFH(fPixelBitsH),index);
		_HUnlock((Handle)fPixelBitsH);
		return val;
	}
	return FALSE;
}
	
	
Boolean TSprayLEList::OilAtPixel(WorldPoint wp)
{
	Point pixel = WorldToScreenPoint(wp,fPixelWorldRect,fPixelScreenRect);
	return this -> OilAtPixel(pixel);
}


Point TSprayLEList::PixelToSpray(void)
{ // returns a random pixel in the area of the spray can
	Point mousePt;
	Point pixel;
	float rand1,rand2;
	short radius = SprayCanRadiusInPixels();
	GetMouse(&mousePt);
	GetRandomVectorInUnitCircle(&rand1,&rand2);
	pixel.h = round(mousePt.h + rand1*radius);
	pixel.v = round(mousePt.v + rand2*radius);
	return pixel;
}

void TSprayLEList::SprayCanMouseDown(void)
{
	unsigned long lastTicks;
	OSErr err = 0;
	const long kTickLimit = 1; // 60 splots per seconds
	Rect saveClip;
	Boolean overWater=false;
	if (settings.inBackground) return; // just a safety check
	lastTicks = TickCount();
	saveClip = MyClipRect(MapDrawingRect());
	
	if(StillDown() && !CanSpray(&overWater)) 
	{
		// they have clicked while the cursor is a "can't spray" cursor
		if(overWater){
			char msg[256];
			Boolean haveOpenWizFile = (model->fWizard && model->fWizard->HaveOpenWizardFile());
			if(haveOpenWizFile)
				//printError("This Location File has not been set up for spills in this area.  The cursor appears as an open circle over allowable spill locations."); 
				strcpy(msg,"This Location File has not been set up for spills in this area.  The cursor appears as an open circle over allowable spill locations."); 
			else
				//printError("The map has not been set up for spills in this area.  The cursor appears as an open circle over allowable spill locations.");
				strcpy(msg,"The map has not been set up for spills in this area.  The cursor appears as an open circle over allowable spill locations.");
			if(model->GetModelMode()==ADVANCEDMODE)
			{
				Point pixel, mousePt;
				GetMouse(&mousePt);
				SetPt(&pixel,mousePt.h,mousePt.v);
				WorldPoint wp;
				wp = ScreenToWorldPoint(pixel, MapDrawingRect(), settings.currentView);
				if (model->HaveAllowableSpillLayer(wp)) strcat(msg," To access this area, make the Spillable Area inactive in the left hand list.");
			}
			printError(msg);
		}
		else // over land
			SysBeep(5);
	}
	
	while(StillDown() && !err)
	{
		unsigned long nowTicks = TickCount();
		Point pixel;
		if(labs(nowTicks - lastTicks) < kTickLimit) continue; // we use labs to take care of when the tickcount wraps
		CommonSetCursor();
		lastTicks = nowTicks;
		pixel = PixelToSpray();
		if (this -> CanSprayPixel(pixel)) {
			err = this -> AddSprayPoint(pixel);
			if (!err) {
				this -> SetPixel(pixel);
				// provide user feedback, draw a dot on the screen
				RGBForeColor(&colors[BLUE]);
				MyMoveTo(pixel.h, pixel.v);
				#ifdef MAC
					Line(0, 0);
				#else
					Line(1, 1);
				#endif
			}
		}
	}
done:
	RGBForeColor(&colors[BLACK]);
	(void) MyClipRect(saveClip);
}

void TSprayLEList::EraserMouseDown(void)
{
	unsigned long lastTicks;
	OSErr err = 0;
	Boolean canceled = FALSE;
	Boolean haveErased = FALSE;
	Rect eraserRect;
	short  radius = EraserRadiusInPixels();
	Rect saveClip;
	Point pixel;

	if (settings.inBackground) return; // just a safety check
	
	err = this -> VerifyPixelBitmap();
	
	saveClip = MyClipRect(MapDrawingRect());
	while(StillDown() && !err)
	{
		// just clear the bits in fPixelBitsH
		// while the mouse is down 
		// then fix fNumSprayedPts after the mouse up
		Point mousePt;
		
		canceled = CmdPeriod(); // allow uer to hit the escape key to cancel/ undo the erasing
		if(canceled) break;

		GetMouse(&mousePt);
		MySetRect(&eraserRect,mousePt.h-radius,mousePt.v-radius,mousePt.h+radius,mousePt.v+radius);
		
		// deal with the screen to provide user feedback
		EraseRect(&eraserRect);
		InvalRectInWindow(eraserRect,mapWindow);
		haveErased = TRUE;
		
		// clear the corresponding pixels in fSprayedH
		if(fSprayedH) {
			// we could be smarter than this but this should work OK
			for(pixel.h = eraserRect.left; pixel.h < eraserRect.right; pixel.h++)
				for(pixel.v = eraserRect.top; pixel.v < eraserRect.bottom; pixel.v++)
					this -> ClearPixel(pixel);
		}
		
	}
	(void) MyClipRect(saveClip);
	
	if(canceled && haveErased && fSprayedH) {
		// we can just call DisposePixelBits and let the redraw reinitialize the bits
		this -> DisposePixelBits();
	}
	else if(!canceled && haveErased && fSprayedH)
	{
		//////////
		// go through fSprayedH 
		// deleting the sprayed point if its pixel was cleared
		////////////
		long i,j, maxIndex = fNumSprayedPts; 
		long numStillSprayed = 0;
		WorldPoint wp;
		Point pixel;
		Boolean haveOilAtPixel;
		Boolean outsideTheBitmap;

		// JLM 6/11/2010
		// bug reported by Chris and Amy. 
		// if you are zoomed in, not all of the pixels are in the pixel bitmap,
		// and using the erase tool inappropriately erases all the pixels outside the bitmap.
		// We fix this by keeping any sprayed points that are outside of the bitmap area
		
		for(i = 0,numStillSprayed = 0; i < maxIndex; i++)
		{
			wp = INDEXH(fSprayedH,i);
			outsideTheBitmap = !WPointInWRect(wp.pLong, wp.pLat, &fPixelWorldRect);
			haveOilAtPixel = OilAtPixel(wp);
			if(haveOilAtPixel || outsideTheBitmap) {
				if (numStillSprayed < i)  { // then we need to move the value
					INDEXH(fSprayedH,numStillSprayed) = INDEXH(fSprayedH,i);
				}
				numStillSprayed++;
			}
		}
		
		// if smaller, make the handle smaller
		if (numStillSprayed < fNumSprayedPts) {
			WORLDPOINTH oldHdl = fSprayedH;
			fNumSprayedPts = numStillSprayed;
			if (fNumSprayedPts > 0) {
				_SetHandleSize((Handle)fSprayedH,fNumSprayedPts * sizeof(**fSprayedH));
			}
			else { // no more points, they were all erased
				this -> DisposeSprayedPoints();
			}
		}
	}
}

void TSprayLEList::DisposeSprayedPoints(void)
{
	if (fSprayedH) {
		DisposeHandle ((Handle)fSprayedH);
		fSprayedH = 0;
	}
	fNumSprayedPts = 0;
}
	

void TSprayLEList::DisposePixelBits(void)
{
	if(fPixelBitsH) 
		{DisposeHandle((Handle)fPixelBitsH); fPixelBitsH = 0;}
	memset(&fPixelScreenRect,0,sizeof(fPixelScreenRect));
	fPixelWorldRect = emptyWorldRect;
}

OSErr TSprayLEList::VerifyPixelBitmap(void)
{
	// makes sure the pixels correspond to the current window
	// if not allocates and fills in fPixelBitsH
	Boolean notValid = FALSE;
	Rect currentScreenRect = MapDrawingRect();
	long i;
	OSErr err = 0;
	
	if (!fPixelBitsH) notValid = TRUE;
	else if (!EqualRects(fPixelScreenRect,currentScreenRect))  notValid = TRUE;
	else if (!EqualWRects(fPixelWorldRect,settings.currentView))  notValid = TRUE;
	
	if(notValid)
	{	// we need to allocate and fill in the bitmap
		long size;
		long width = RectWidth(currentScreenRect);
		long height = RectHeight(currentScreenRect);
		long numBits = width*height;
		long numBytes = (numBits/8) + 1;
		
		this -> DisposePixelBits();
		
		/// allocate
		fPixelBitsH = _NewHandleClear(numBytes);
		if(fPixelBitsH) {
			fPixelScreenRect = currentScreenRect;
			fPixelWorldRect = settings.currentView;
			if(fSprayedH) {
				// Set the pixels which have Sprayed LEs
				for(i = 0; i < fNumSprayedPts; i++) {
					WorldPoint wp = INDEXH(fSprayedH,i);
					Point pixel = WorldToScreenPoint(wp,fPixelWorldRect,fPixelScreenRect);
					long index = PixelToIndex(pixel);
					this -> SetPixel(pixel);
				}
			}
		}
		else  goto MemoryError;
	}
	
	return err;

MemoryError:
	TechError("TSprayLEList::VerifyPixelBitmap()", "_NewHandleClear()", 0); 
	return memFullErr;
}

OSErr TSprayLEList::AddSprayPoint(Point pixel)
{
	OSErr err =0;
	WorldPoint wp = ScreenToWorldPoint(pixel,fPixelScreenRect,fPixelWorldRect);
	err = this -> AddSprayPoint(wp);
	return err;
}

OSErr TSprayLEList::AddSprayPoint(WorldPoint wp)
{
	OSErr err =0;
	
	if (!fSprayedH) {// first point
		fNumSprayedPts = 0; // we increment later
		fSprayedH = (WORLDPOINTH) _NewHandle(0);
		if (!fSprayedH) 
			goto MemoryError; 
	}
	// allocate more space
	_SetHandleSize((Handle)fSprayedH, (fNumSprayedPts+1) * sizeof(**fSprayedH));
	if(_MemError()) 
		goto MemoryError; 

	// assign the value
	INDEXH(fSprayedH,fNumSprayedPts) = wp;
	fNumSprayedPts++;

	return err;
	
MemoryError:
	TechError("TSprayLEList::AddSprayPoint()", "_NewHandleClear()", 0); 
	return memFullErr;

}
/////////////////////////////////////////////////
Boolean TSprayLEList::IAmEditableInMapDrawingRect(void)
{	
	// the uncertainty LE's are not editable
	if (this->fLeType == UNCERTAINTY_LE)
		return FALSE;
		
	return TRUE;
}

Boolean TSprayLEList::IAmCurrentlyEditableInMapDrawingRect(void)
{	
	// the uncertainty LE's are not editable
	if (this->fLeType == UNCERTAINTY_LE)
		return FALSE;

	// we are editable if they have to have the spray can or the eraser selected
	if (IsEditTool(settings.currentTool))
		return TRUE;

	// if they were editing us and they want to change the view, we stay editable
	if (fInPixelEditMode)
	{
		switch(settings.currentTool)
		{
			case ZOOMPLUSTOOL:
			case ZOOMMINUSTOOL:
				// if they will see splots, then leave edit mode
				// else stay in edit mode
				//
				// in order to see splots, we have to have sprayed pixels 
				// and the release time must be before or equal to the model time
				//
				// hmmmm... what are we doing about editing splots 
				//  when the model time is after the model start time ??
				if (fNumSprayedPts <= 0) 
					return TRUE; // stay in edit mode
				if (model -> GetStartTime() < fSetSummary.startRelTime)
					return TRUE;
				break;
			case HANDTOOL:
				return TRUE;
		}
	}
	
	return FALSE; // reject edit mode
	
}

Boolean TSprayLEList::UserIsEditingMeInMapDrawingRect(void)
{	
	return fInPixelEditMode;
}


void TSprayLEList::StartEditingInMapDrawingRect(void)
{	
	fInPixelEditMode = TRUE;
	model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); // to redraw map with or without sprayed LE's
}


OSErr TSprayLEList::StopEditingInMapDrawingRect(Boolean *deleteMe)
{	
	OSErr err = 0;

	*deleteMe = FALSE;
	
	// if there are no sprayed LE's verify they wish to delete me
	if ( fNumSprayedPts <= 0)
	{
		short buttonSelected;
		buttonSelected  = MULTICHOICEALERT(1685,"Since the sprayed set has no points, it will be deleted.",FALSE);
		switch(buttonSelected){
			case 1: // user accepts the stop editing
				*deleteMe = TRUE;
				break;  
			case 3: // user rejects the stop editing
				SelectListItemOfOwner(this);
				// we need to make sure the tool is an edit tool
				if(!this -> IAmCurrentlyEditableInMapDrawingRect())
					SetTool(SPRAYCANTOOL);
				return -1; 
				break;
		}
	}
	else
	{
		// we have stopped editing, we need to reset the LE's
		TSprayLEList *uncertaintyLEList = 0;

		this -> Reset(TRUE);
	
		uncertaintyLEList = (TSprayLEList*)model->GetMirroredLEList(this);
		if (uncertaintyLEList)
			err = uncertaintyLEList -> Reset(TRUE);
			
	}
	
	fInPixelEditMode = FALSE;
	
	model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar, even in advanced mode
	model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); // to redraw map with or without sprayed LE's

	if( !*deleteMe && model -> GetModelTime() < fSetSummary.startRelTime)
	{ 	// the blue dots are disappearing and the black dots will not be visible because they are not yet released
		// so tell the user they will be disappearing
		static Boolean sHaveToldUser =  FALSE;
		if(!sHaveToldUser) {
			printNote("These sprayed splots will not be visible until you run the model and the model time reaches the overflight time.");
			sHaveToldUser = TRUE;
		}
	}
	return 0; // we accept the stop editing
}

// code goes here, this class will have to support the .LOC wizard save files


///////////////////////////////////////////////////////////////////////////
OSErr TSprayLEList::CheckAndPassOnMessage(TModelMessage *message)
{	
	char ourName[kMaxNameLen];
	OSErr err = 0;
	long	prevToolNum,currentToolNumber;
	UNIQUEID ourUniqueID = this -> GetUniqueID();
	Boolean checkEditMode = FALSE;
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	
	/////////////////////////////////////////////////
	if(message->IsMessage(M_TOOLDOWN,ourUniqueID))
	{
		switch(settings.currentTool)
		{
			case SPRAYCANTOOL:
				this -> SprayCanMouseDown();
				break;
			case ERASETOOL:
				this -> EraserMouseDown();
				break;
		}
	}
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TOLEList::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////
SprayDialogInfo TSprayLEList::GetSprayDialogInfo ()
{
	SprayDialogInfo	info;

	memset(&info,0,sizeof(info));

	info.numOfLEs = fSetSummary.numOfLEs;
	info.pollutantType = fSetSummary.pollutantType;
	info.massUnits = fSetSummary.massUnits;
	info.density = fSetSummary.density;
	info.overflightTime = fSetSummary.startRelTime;
	info.whenAmountIsSpecified = fWhenAmountIsSpecified;	
	info.spillTime = fSpillTime;
	info.ageHours = fAgeAtOverflightTimeInHoursEnteredByUser;
	info.totalMass = fAmountEnteredByUser;
	strcpy(info.spillName, fSetSummary.spillName);
	
	return info;
}

OSErr TSprayLEList::SetSprayDialogInfo (SprayDialogInfo info)
{
	OSErr err = 0;
	LESetSummary summary;
	WorldPoint startPoint = {0,0};
	WorldPoint endPoint = {0,0};
		
	fWhenAmountIsSpecified=info.whenAmountIsSpecified;	
	fSpillTime=info.spillTime;
	fAgeAtOverflightTimeInHoursEnteredByUser=info.ageHours;
	fAmountEnteredByUser = info.totalMass;
		
	memset(&summary,0,sizeof(summary));

	summary.numOfLEs = info.numOfLEs;
	summary.pollutantType = info.pollutantType;
	summary.massUnits = info.massUnits;
	summary.startRelPos = startPoint;
	summary.endRelPos = endPoint;
	summary.startRelTime = info.overflightTime;
	summary.endRelTime = summary.startRelTime;
	summary.bWantEndRelTime = false;
	summary.bWantEndRelPosition = false;
	strcpy(summary.spillName, info.spillName);

	summary.z = 0;
	//summary.density = GetPollutantDensity(summary.pollutantType);
	summary.density = info.density;
	/////////////////////////////////////////////////
	//summary.ageInHrsWhenReleased = this->AgeInHrsAtOverflightTime(); // this uses fSetSummary values not set yet
	if(info.whenAmountIsSpecified == ATOVERFLIGHTTIME)
	{
		summary.ageInHrsWhenReleased = info.ageHours;
	}
	else if(info.whenAmountIsSpecified == ATTIMEOFSPILL)
	{
		summary.ageInHrsWhenReleased =  (summary.startRelTime - info.spillTime)/3600.;
	}
	if (summary.pollutantType == OIL_CONSERVATIVE)summary.ageInHrsWhenReleased = 0;

	/////////////////////////////////////////////////
	//summary.totalMass = this->AmountAtOverflightTime();  // this uses fSetSummary values not set yet
	if(info.whenAmountIsSpecified == ATOVERFLIGHTTIME)
	{
		summary.totalMass = info.totalMass;
	}
	else if(info.whenAmountIsSpecified == ATTIMEOFSPILL)
	{
		double amount = info.totalMass;
		double ageInHrs = summary.ageInHrsWhenReleased;
		double fractionLeft = FractionOilLeftAtTime(summary.pollutantType,ageInHrs);
		// calculate the amount after evaporation
		amount =  amount * fractionLeft;
		summary.totalMass = amount;
	}
	/////////////////////////////////////////////////


	err = this -> Initialize(&summary,true); // this makes sure we use the new info
	

	return err;
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////


