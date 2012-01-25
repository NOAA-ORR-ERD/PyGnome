#include "Cross.h"
#include "OSSM.h"
#include "EditCDOGProfilesDialog.h"


#ifdef MPW
#pragma SEGMENT EDITPROFILES
#endif
	
static VList sgObjects;
static CDischargeTimes *sgDischargeTimes=0;
static DischargeDataH sgDischargeTimesH=0;

static float sIncrementInHours = .5;

// code goes here, rework for variable discharge and stick in the edit cdog dialog file

///////////////////////////////////////////////////////////////////////////

#define INCREMENT_TIME true
#define REPLACE true

static short TIME_COL,Q_OIL_COL,Q_GAS_COL,TEMP_COL,DIAM_COL,RHO_OIL_COL/*,N_DEN_COL,OUTPUT_INT_COL*/;
static CDOGParameters sDialogCDOGParameters;
static CDOGSpillParameters sCDOGSpillParameters2;

static PopInfoRec CDOGVarDischargePopTable[] = {
		{ VAR_DISCHARGE_DLGID, nil, CDOGSETTINGSHYDPROC, 0, pHYDPROCESS, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSETTINGSEQ, 0, pEQCURVES, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSETTINGSSEP, 0, pSEPFROMPLUME, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSETTINGSDROPSIZE, 0, pDROPLET, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSPILLDRPOPUP, 0, pDISCHARGERATE, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, VARDISCHARGETYPE_POPUP, 0, pDISCHARGETYPE, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, DIAMETERUNITS_POPUP, 0, pDIAMETERUNITS, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, TEMPUNITS_POPUP, 0, pTEMPUNITS, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, DENSITYUNITS_POPUP, 0, pDENSITYUNITS, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, GORUNITS_POPUP, 0, pGOR, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSETTINGSMOLWTUNITS, 0, pMOLWT, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSPILLDRUNITSPOPUP, 0, pOILDISCHARGE, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, CDOGSPILLDRUNITSPOPUP2, 0, pGASDISCHARGE, 0, 1, FALSE, nil },
		{ VAR_DISCHARGE_DLGID, nil, GASYESNO_POPUP, 0, pYESNO, 0, 1, FALSE, nil }
	};

Boolean AddRecordRowIsSelected3(void)
{
	long curSelection;
	Boolean  isLastItem;
	VLGetSelect(&curSelection,&sgObjects);
	isLastItem = curSelection == -1 || curSelection == sgObjects.numItems-1;
	return isLastItem;
}

/*Boolean ShowAutoIncrement2(void)
{
	Boolean show = AddRecordRowIsSelected3();
	return show;
}
*/
static void ShowHideAutoIncrement(DialogPtr dialog,long curSelection)
{
	//Boolean show = ShowAutoIncrement2();
	Boolean show = AddRecordRowIsSelected3();
	ShowHideDialogItem(dialog,VARDISCHARGE_INCREMENT,show);
	ShowHideDialogItem(dialog,VARDISCHARGE_AUTOINCRTEXT,show);
	ShowHideDialogItem(dialog,VARDISCHARGE_AUTOINCRMETERS,show);
}

OSErr RetrieveIncrementDischargeTime(DialogPtr dialog)
{
	OSErr err = 0;
	float incrementTimeInHours  = EditText2Float(dialog,VARDISCHARGE_INCREMENT);
	if(incrementTimeInHours < 0.000001 || incrementTimeInHours > 500.0)
	{
		printError("The increment time must be between 0. and 500.");
		MySelectDialogItemText (dialog, VARDISCHARGE_INCREMENT, 0, 255);
		err = -1; return err;
	}
	sIncrementInHours = incrementTimeInHours;// set the static
	return noErr;
}

static void IncrementDischargeTime(DialogPtr dialog, float time)
{
	float incr;
	incr = sIncrementInHours;
	time += incr;
	// update dialog value
	char timestr[30];
	StringWithoutTrailingZeros(timestr,time,1);
	mysetitext(dialog,VARDISCHARGE_TIME,timestr);	// or something
}

/*static void MyFloat2EditText(DialogPtr dialog, short itemnum,float num,short numdec)
{
	char numstr[30];
	StringWithoutTrailingZeros(numstr,num,numdec);
	mysetitext(dialog,itemnum,numstr);
}*/

void EnableOilDischargeUnits(DialogPtr dialog, Boolean bEnable,short dischargeUnits)
{
	// will need to enable and disable the discharge rates popups depending on whether it's gas or oil
	
	Boolean show  = bEnable; //JLM

	//CDOGVarDischargePopTable[11].bStatic = !bEnable;	// this is not quite working as advertised...don't need to gray out just hide

	//CDOGVarDischargePopTable[12].bStatic = bEnable;
		
	ShowHideDialogItem(dialog, CDOGSPILLDRUNITSPOPUP, show); 

	ShowHideDialogItem(dialog, CDOGSPILLDRUNITSPOPUP2, !show); 

/*	if (bEnable) CDOGVarDischargePopTable[10].menuID = pOILDISCHARGE;
	else CDOGVarDischargePopTable[10].menuID = pGASDISCHARGE;
	PopDraw(dialog, CDOGSPILLDRUNITSPOPUP);*/
	

	if(show)
	{
		PopDraw(dialog, CDOGSPILLDRUNITSPOPUP);
		SetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP,dischargeUnits);
	}
	else
	{
		PopDraw(dialog, CDOGSPILLDRUNITSPOPUP2);
		SetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP2,dischargeUnits);
	}
	
//	EnableTextItem(dialog, CDOGSPILLENDDAY, bEnable);
//	EnableTextItem(dialog, CDOGSPILLENDHOURS, bEnable);
//	EnableTextItem(dialog, CDOGSPILLENDMINUTES, bEnable);
}

// need this
static OSErr GetTimeVals(DialogPtr dialog,DischargeData *dischargeTimes)
{
	float time = EditText2Float(dialog,VARDISCHARGE_TIME);
	// code goes here, check popup type and scale if necessary
	float q_oil = EditText2Float(dialog,CDOGSPILLDRPOPUPVAL);
	float q_gas = EditText2Float(dialog,VARDISCHARGE_QGAS);
	float diam = EditText2Float(dialog,VARDISCHARGE_DIAM);
	float temp = EditText2Float(dialog,VARDISCHARGE_TEMP);
	float rho_oil = EditText2Float(dialog,VARDISCHARGE_RHOOIL);
	OSErr  err = 0;
	char msg[256] = "";;
	 
	 // might want some checking here
	 // or wait until user hits ok?
	 
	 // how to handle more than one out of range variable

	if (GetPopSelection (dialog, CDOGSPILLDRPOPUP) == 1)
	{
		if (GetPopSelection(dialog, CDOGSPILLDRUNITSPOPUP) == 1)
		{
			if (q_oil < 0 || q_oil > .92) /*{printNote("Oil discharge rate must be less than .92 m3/s"); return -1;}*/
			{strcpy(msg,"CDOG recommends an oil discharge rate of less than .92 m3/s.  Are you sure you want to continue?");}
		}
		else
			if (q_oil < 0 || q_oil > 499782.70) /*{printNote("Oil discharge rate must be less than 499782 bbls/day"); return -1;}*/
			{strcpy(msg,"CDOG recommends an oil discharge rate of less than 499782 bbls/day.  Are you sure you want to continue?");}

	}
	else
	{
		if (GetPopSelection(dialog, CDOGSPILLDRUNITSPOPUP2) == 1)
		{
			if (q_oil < 0 || q_oil > 165) /*{printNote("Gas discharge rate must be less than 165 m3/s"); return -1;}*/
			{strcpy(msg,"CDOG recommends a gas discharge rate of less than 165 m3/s.  Are you sure you want to continue?");}
		}
		else
			if (q_oil < 0 || q_oil > 503389.8288) /*{printNote("Gas discharge rate must be less than 503389 MSCF"); return -1;}*/
			{strcpy(msg,"CDOG recommends a gas discharge rate of less than 503389 MSCF.  Are you sure you want to continue?");}
	}

	if(msg[0]) 
	{
		short buttonSelected;
		{
			// point out that some value is not within the recommended range
			buttonSelected  = MULTICHOICEALERT(1690,msg,TRUE);
			switch(buttonSelected){
				case 1:// continue
					strcpy(msg,"");
					break;  
				case 3: // cancel
					return -1;// don't update the list
					break;
			}
		}
	}

	if (GetPopSelection (dialog, GORUNITS_POPUP) == 1)
	{	
		if (q_gas < 0 || q_gas > 2000) /*{printNote("Gas oil ratio must be less than 2000 in dimensionless SI Units"); return -1;}*/
		{strcpy(msg,"CDOG recommends a gas oil ratio of less than 2000 in dimensionless SI Units.  Are you sure you want to continue?");}
	}
	else if (GetPopSelection (dialog, GORUNITS_POPUP) == 2)
	{	
		if (q_gas < 0 || q_gas > 11227.89) /*{printNote("Gas oil ratio must be less than 11228 in SCFD/BOPD"); return -1;}*/
		{strcpy(msg,"CDOG recommends a gas oil ratio of less than 11228 in SCFD/BOPD.  Are you sure you want to continue?");}
	}
	else
	{	
		if (q_gas < 0 || q_gas > 11.22789) /*{printNote("Gas oil ratio must be less than 11.227 in MSCF/BOPD"); return -1;}*/
		{strcpy(msg,"CDOG recommends a gas oil ratio of less than 11.227 in MSCF/BOPD.  Are you sure you want to continue?");}
	}

	if(msg[0]) 
	{
		short buttonSelected;
		{
			// point out that some value is not within the recommended range
			buttonSelected  = MULTICHOICEALERT(1690,msg,TRUE);
			switch(buttonSelected){
				case 1:// continue
					strcpy(msg,"");
					break;  
				case 3: // cancel
					return -1;// don't update the list
					break;
			}
		}
	}

	if (GetPopSelection (dialog, TEMPUNITS_POPUP) == 1)
	{
		if (temp < 0 || temp > 232) /*{printNote("Temperature must be less than 232 degrees C"); return -1;}*/
		{strcpy(msg,"CDOG recommends a temperature of less than 232 degrees C.  Are you sure you want to continue?");}
	}
	else
	{
		if (temp < 0 || temp > 165) /*{printNote("Temperature must be less than 449.6 degrees F"); return -1;}*/
		{strcpy(msg,"CDOG recommends a temperature of less than 449.6 degrees F.  Are you sure you want to continue?");}
	}

	if(msg[0]) 
	{
		short buttonSelected;
		{
			// point out that some value is not within the recommended range
			buttonSelected  = MULTICHOICEALERT(1690,msg,TRUE);
			switch(buttonSelected){
				case 1:// continue
					strcpy(msg,"");
					break;  
				case 3: // cancel
					return -1;// don't update the list
					break;
			}
		}
	}

	if (GetPopSelection (dialog, DIAMETERUNITS_POPUP) == 1)
	{
		if (diam < 0 || diam > 2) /*{printNote("Orifice diameter must be less than 2 meters"); return -1;}*/
		{strcpy(msg,"CDOG recommends an orifice diameter of less than 2 meters.  Are you sure you want to continue?");}
	}
	else if (GetPopSelection (dialog, DIAMETERUNITS_POPUP) == 2)
	{
		if (diam < 0 || diam > 200) /*{printNote("Orifice diameter must be less than 200 centimeters"); return -1;}*/
		{strcpy(msg,"CDOG recommends an orifice diameter of less than 200 centimeters.  Are you sure you want to continue?");}
	}
	else 
	{
		if (diam < 0 || diam > 78.74) /*{printNote("Orifice diameter must be less than 78.74 inches"); return -1;}*/
		{strcpy(msg,"CDOG recommends an orifice diameter of less than 78.74 inches.  Are you sure you want to continue?");}
	}

	if(msg[0]) 
	{
		short buttonSelected;
		{
			// point out that some value is not within the recommended range
			buttonSelected  = MULTICHOICEALERT(1690,msg,TRUE);
			switch(buttonSelected){
				case 1:// continue
					strcpy(msg,"");
					break;  
				case 3: // cancel
					return -1;// don't update the list
					break;
			}
		}
	}

	if (GetPopSelection (dialog, DENSITYUNITS_POPUP) == 1)
	{
		if (rho_oil < 700 || rho_oil > 1060) /*{printNote("Density must be between 700 and 1060 kg/m3"); return -1;}*/
		{strcpy(msg,"CDOG recommends a density between 700 and 1060 kg/m3.  Are you sure you want to continue?");}
	}
	else
	{
		if (rho_oil < 2 || rho_oil > 70.64) /*{printNote("Density must be between 2 and 70.64 API"); return -1;}*/
		{strcpy(msg,"CDOG recommends a density between 2 and 70.64 API.  Are you sure you want to continue?");}
	}

	if(msg[0]) 
	{
		short buttonSelected;
		{
			// point out that some value is not within the recommended range
			buttonSelected  = MULTICHOICEALERT(1690,msg,TRUE);
			switch(buttonSelected){
				case 1:// continue
					break;  
				case 3: // cancel
					return -1;// don't update the list
					break;
			}
		}
	}

	dischargeTimes->time = time;
	dischargeTimes->q_oil = q_oil;
	dischargeTimes->q_gas = q_gas;
	dischargeTimes->diam = diam;
	dischargeTimes->temp = temp;
	dischargeTimes->rho_oil = rho_oil;
	return err;
}

void UpdateDisplayWithDischargeValuesSet(DialogPtr dialog,DischargeData dischargeTimes)
{
	char numstr[30];
	StringWithoutTrailingZeros(numstr,dischargeTimes.time,2);
	mysetitext(dialog,VARDISCHARGE_TIME,numstr);

	//MyFloat2EditText(dialog,CDOGSPILLDRPOPUPVAL,dischargeTimes.q_oil,4);
	//MyFloat2EditText(dialog,VARDISCHARGE_QGAS,dischargeTimes.q_gas,4);
//	MyFloat2EditText(dialog,VARDISCHARGE_TEMP,dischargeTimes.temp,2);
	//MyFloat2EditText(dialog,VARDISCHARGE_DIAM,dischargeTimes.diam,2);
	//MyFloat2EditText(dialog,VARDISCHARGE_RHOOIL,dischargeTimes.rho_oil,2);
	Float2EditText(dialog,CDOGSPILLDRPOPUPVAL,dischargeTimes.q_oil,4);
	Float2EditText(dialog,VARDISCHARGE_QGAS,dischargeTimes.q_gas,4);
	Float2EditText(dialog,VARDISCHARGE_TEMP,dischargeTimes.temp,2);
	Float2EditText(dialog,VARDISCHARGE_DIAM,dischargeTimes.diam,2);
	Float2EditText(dialog,VARDISCHARGE_RHOOIL,dischargeTimes.rho_oil,2);
}

	
static void UpdateDisplayWithCurSelection(DialogPtr dialog)
{
	DischargeData dischargeTimes;
	long curSelection;
	
	if(!AddRecordRowIsSelected3())
	{	// set the item text
		{
			VLGetSelect(&curSelection,&sgObjects);
			sgDischargeTimes->GetListItem((Ptr)&dischargeTimes,curSelection);
		}
		
		UpdateDisplayWithDischargeValuesSet(dialog,dischargeTimes);
	}

	ShowHideAutoIncrement(dialog,curSelection); // JLM 9/17/98
}

static void SelectNthRow(DialogPtr dialog,long nrow)
{
	VLSetSelect(nrow, &sgObjects); 
	if(nrow > -1)
	{
		VLAutoScroll(&sgObjects);
	}
	VLUpdate(&sgObjects);
	UpdateDisplayWithCurSelection(dialog);	
}

static OSErr AddReplaceRecord(DialogPtr dialog,Boolean incrementTime,Boolean replace,DischargeData dischargeTimes)
{
	long itemnum;
	OSErr err=0;
	
	if(!err)
	{
		err=sgDischargeTimes->InsertSorted ((Ptr)&dischargeTimes,&itemnum,false);// false means don't allow duplicate times
		
		if(!err) // new record
		{
			VLAddItem(1,&sgObjects);
			VLSetSelect(itemnum, &sgObjects); 
			VLAutoScroll(&sgObjects);
			VLUpdate(&sgObjects);
			if(incrementTime)IncrementDischargeTime(dialog,dischargeTimes.time);
		}
		else if(err == -2) // found existing record. Replace if okay to replace
		{
			if(replace)
			{
				sgDischargeTimes->DeleteItem(itemnum);
				VLDeleteItem(itemnum,&sgObjects);
				err = AddReplaceRecord(dialog,!INCREMENT_TIME,REPLACE,dischargeTimes);
				VLUpdate(&sgObjects);
				if(incrementTime)IncrementDischargeTime(dialog,dischargeTimes.time);
				err=0;
			}
			else
			{
				printError("A record with the specified depth already exists."
					"If you want to edit the existing record, select it."
					"If you want to add a new record, change the specified depth.");
				VLUpdate(&sgObjects);
			}
		}
		else SysBeep(5);
	}
	return err;
}

void 	DisposeVARDISCHARGEStuff(void)
{
	if(sgDischargeTimes)
	{
		sgDischargeTimes->Dispose();// JLM 12/14/98
		delete sgDischargeTimes;
		sgDischargeTimes = 0;
	}

	//?? VLDispose(&sgObjects);// JLM 12/10/98, is this automatic on the mac ??
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
}
//static float sCellLength;
//static long sNumCells;
static Boolean sHydSulfide;
CDOGUserUnits sUserUnits;

	
void ShowHideParameters(DialogPtr dialog)
{
	Boolean show;
	short theType = GetPopSelection (dialog, CDOGSETTINGSEQ);
	if (theType == 1) show = false;
	else show = true;

	ShowHideDialogItem(dialog, CDOGSETTINGSMOLWTLABEL, show); 
	ShowHideDialogItem(dialog, CDOGSETTINGSMOLWT, show); 
	ShowHideDialogItem(dialog, CDOGSETTINGSMOLWTUNITS, show); 

	ShowHideDialogItem(dialog, CDOGSETTINGSHYDDENSLABEL, show); 
	ShowHideDialogItem(dialog, CDOGSETTINGSHYDDENS, show); 
	ShowHideDialogItem(dialog, CDOGSETTINGSHYDDENSUNITS, show); 
}

void ShowCDOGVarDischargeItems(DialogPtr dialog)
{
	Boolean bShowVarDischargeRateItems = true;
	short dischargeType = GetPopSelection(dialog, VARDISCHARGETYPE_POPUP);
	Boolean continuousRelease = GetButton(dialog, VARDISCHARGE_CONTINUOUS);
	if (dischargeType == 1) bShowVarDischargeRateItems = false;

	ShowHideDialogItem(dialog, VARDISCHARGE_INCREMENT, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_TIME_LIST_LABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_QOIL_LIST_LABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_TEMP_LIST_LABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_RHOOIL_LIST_LABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_LIST, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_AUTOINCRTEXT, bShowVarDischargeRateItems); 
	//ShowHideDialogItem(dialog, CDOGSPILLDRPOPUP, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_AUTOINCRMETERS, bShowVarDischargeRateItems); 
	//ShowHideDialogItem(dialog, VARDISCHARGE_QGAS_LABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_QGAS_LIST_LABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_DIAM_LIST_LABEL, bShowVarDischargeRateItems); 

	ShowHideDialogItem(dialog, VARDISCHARGE_DELETEROWS_BTN, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_DELETEALL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_REPLACE, bShowVarDischargeRateItems); 

	ShowHideDialogItem(dialog, VARDISCHARGE_TIMELABEL, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_TIME, bShowVarDischargeRateItems); 
	ShowHideDialogItem(dialog, VARDISCHARGE_TIMEUNITS, bShowVarDischargeRateItems); 

	ShowHideDialogItem(dialog, VARDISCHARGE_DURATIONLABEL, !bShowVarDischargeRateItems && !continuousRelease); 
	ShowHideDialogItem(dialog, VARDISCHARGE_DURATION, !bShowVarDischargeRateItems && !continuousRelease); 
	ShowHideDialogItem(dialog, VARDISCHARGE_DURATIONUNITS, !bShowVarDischargeRateItems && !continuousRelease); 
	ShowHideDialogItem(dialog, VARDISCHARGE_CONTINUOUS, !bShowVarDischargeRateItems); 

	ShowHideDialogItem(dialog, VARDISCHARGE_FRAMEINPUT2, bShowVarDischargeRateItems); 
	ShowHideVList(dialog, &sgObjects, bShowVarDischargeRateItems);
}

void UpdateListItemWithUnitsSelected(DialogPtr dialog, short itemNum, short oldUnits, short newUnits)
{
	OSErr err = 0;
	DischargeData dischargeTimes;
	long i, n = sgDischargeTimes->GetItemCount();
	
	if (n<=0) return;
	if (GetPopSelection(dialog, VARDISCHARGETYPE_POPUP) == 1) return;
	for (i=0;i<n;i++)
	{
		sgDischargeTimes->GetListItem((Ptr)&dischargeTimes,i);
		switch(itemNum)
		{
			case TEMPUNITS_POPUP: 
				if (oldUnits==kDegreesC) dischargeTimes.temp = (dischargeTimes.temp * 9) / 5. + 32.;
				else dischargeTimes.temp = (dischargeTimes.temp - 32) * 5. / 9.;
				break;
			case DIAMETERUNITS_POPUP:  
				if (oldUnits==kMeters) 
				{
					if (newUnits==kCentimeters) dischargeTimes.diam = dischargeTimes.diam * 100;
					else dischargeTimes.diam = dischargeTimes.diam * 100. / 2.54;
				}
				else if (oldUnits==kCentimeters) 
				{
					if (newUnits==kMeters) dischargeTimes.diam = dischargeTimes.diam / 100;
					else dischargeTimes.diam = dischargeTimes.diam / 2.54;
				}
				else
				{
					if (newUnits==kCentimeters) dischargeTimes.diam = dischargeTimes.diam * 2.54;
					else dischargeTimes.diam = dischargeTimes.diam * 2.54 * 100.;
				}
				break;
			case DENSITYUNITS_POPUP:  
				if (oldUnits==kKgM3) dischargeTimes.rho_oil = (141.5/dischargeTimes.rho_oil) *1000. - 131.5;
				else dischargeTimes.rho_oil = 141.5/(dischargeTimes.rho_oil + 131.5) * 1000.;
				break;
		}		
		err = AddReplaceRecord(dialog,!INCREMENT_TIME,REPLACE,dischargeTimes);
	}
	VLUpdate(&sgObjects);
	return;
}

short VarDischargeClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Point pos;
	long curSelection;
	long i,n;
	OSErr err=0;
	DischargeData dischargeTimes;
	long menuID_menuItem;
	
	if (AddRecordRowIsSelected3())
	{
		//Last row is selected
		//Disable delete button
		MyEnableControl(dialog,VARDISCHARGE_DELETEROWS_BTN,FALSE);
		// And change title in replace dialog to "Add new record"
		MySetControlTitle(dialog, VARDISCHARGE_REPLACE, "Add New Record");
	}
	else
	{
		MySetControlTitle(dialog, VARDISCHARGE_REPLACE, "Replace Selected");
		MyEnableControl(dialog,VARDISCHARGE_DELETEROWS_BTN,TRUE);
	}
	
	switch(itemNum)
	{
		case VARDISCHARGE_OK:
		{
			short	theTypeEQ, theTypeHydProc, theTypeSep, theTypeDropSize, molWtUnits;
			double bubbleRadius, /*orificeDiameter,*/ hydrateDensity, molecularWt;
			
			if(sgDischargeTimes)	// should always be true
			{
				if (GetPopSelection(dialog, VARDISCHARGETYPE_POPUP) == 2)
				{
					DischargeDataH dischargeTimesh = sgDischargeTimesH;
					n = sgDischargeTimes->GetItemCount();
					if(n == 0)
					{	// no items are entered, tell the user
						char msg[512],buttonName[64];
						GetWizButtonTitle_Cancel(buttonName);
						sprintf(msg,"You have not entered any data values.  Either enter data values and use the 'Add New Record' button, or use the '%s' button to exit the dialog.",buttonName);
						printError(msg);
						break;
					}
					
					if(n == 1)
					{	// single discharge in variable discharge dialog
						printError("You have only entered a single discharge rate. Either use the constant discharge option or add another record.");
						break;
					}
					// check that all the values are in range - if there is some range
					// or may allow the user to change units
					for(i=0;i<n;i++)
					{
						err=sgDischargeTimes->GetListItem((Ptr)&dischargeTimes,i);
						if(err) {SysBeep(5); break;}// this shouldn't ever happen
					}
					
					/////////////
					// point of no return
					//////////////
					if(dischargeTimesh == 0)
					{
						dischargeTimesh = (DischargeDataH)_NewHandle(n*sizeof(DischargeData));
						if(!dischargeTimesh)
						{
							TechError("VarDischargeClick:OKAY", "_NewHandle()", 0);
							//return VARDISCHARGE_CANCEL;
							break; // make them cancel so that code gets executed
						}
						sgDischargeTimesH = dischargeTimesh;
					}
					else
					{
						 _SetHandleSize((Handle)dischargeTimesh,n*sizeof(DischargeData));
						 if(_MemError())
						 {
							 TechError("VarDischargeClick:OKAY", "_NewHandle()", 0);
							//return VARDISCHARGE_CANCEL;
							break; // make them cancel, so that code gets executed
						 }
					}
					
					for(i=0;i<n;i++)
					{
						if(err=sgDischargeTimes->GetListItem((Ptr)&dischargeTimes,i))return VARDISCHARGE_OK;
						(*dischargeTimesh)[i]=dischargeTimes;					
					}
				}

				else
				{
					DischargeData dischargeTimes;
					Boolean continuousDischarge = false;
					DischargeDataH dischargeTimesh = sgDischargeTimesH;
					char msg[256] = "";;
					dischargeTimes.time = 0;	//constant discharge
					dischargeTimes.q_oil = EditText2Float(dialog,CDOGSPILLDRPOPUPVAL);
					dischargeTimes.q_gas = EditText2Float(dialog,VARDISCHARGE_QGAS);
					dischargeTimes.diam = EditText2Float(dialog,VARDISCHARGE_DIAM);
					dischargeTimes.temp = EditText2Float(dialog,VARDISCHARGE_TEMP);
					dischargeTimes.rho_oil = EditText2Float(dialog,VARDISCHARGE_RHOOIL);
					if (dischargeTimes.diam==0) {printNote("Parameter value must be greater than zero"); break;}
					n=1;	// will actually want 2 entries to handle constant discharge
					 
					 // might want some checking here

					if (GetPopSelection (dialog, GORUNITS_POPUP) == 1)
					{	
						if (dischargeTimes.q_gas < 0 || dischargeTimes.q_gas > 2000) /*{printNote("Gas oil ratio must be less than 2000 in dimensionless SI Units"); break;}*/
						{strcpy(msg,"CDOG recommends a gas oil ratio of less than 2000 in dimensionless SI Units.  Are you sure you want to continue?");}
					}
					else if (GetPopSelection (dialog, GORUNITS_POPUP) == 2)
					{	
						if (dischargeTimes.q_gas < 0 || dischargeTimes.q_gas > 11227.89) /*{printNote("Gas oil ratio must be less than 11228 in SCFD/BOPD"); break;}*/
						{strcpy(msg,"CDOG recommends a gas oil ratio of less than 11228 in SCFD/BOPD.  Are you sure you want to continue?");}
					}
					else
					{	
						if (dischargeTimes.q_gas < 0 || dischargeTimes.q_gas > 11.22789) /*{printNote("Gas oil ratio must be less than 11.227 in MSCF/BOPD"); break;}*/
						{strcpy(msg,"CDOG recommends a gas oil ratio of less than 11.227 in MSCF/BOPD.  Are you sure you want to continue?");}
					}
			
					if (GetPopSelection (dialog, CDOGSPILLDRPOPUP) == 1)
					{
						if (GetPopSelection(dialog, CDOGSPILLDRUNITSPOPUP) == 1)
						{
							if (dischargeTimes.q_oil < 0 || dischargeTimes.q_oil > .92)/* {printNote("Oil discharge rate must be less than .92 m3/s"); break;}*/
							{strcpy(msg,"CDOG recommends an oil discharge rate of less than .92 m3/s.  Are you sure you want to continue?");}
						}
						else
							if (dischargeTimes.q_oil < 0 || dischargeTimes.q_oil > 499782.70) /*{printNote("Oil discharge rate must be less than 499782 bbls/day"); break;}*/
							{strcpy(msg,"CDOG recommends an oil discharge rate of less than 499782 bbls/day.  Are you sure you want to continue?");}
					}
					else
					{
						if (GetPopSelection(dialog, CDOGSPILLDRUNITSPOPUP2) == 1)
						{
							if (dischargeTimes.q_oil < 0 || dischargeTimes.q_oil > 165) /*{printNote("Gas discharge rate must be less than 165 m3/s"); break;}*/
							{strcpy(msg,"CDOG recommends a gas discharge rate of less than 165 m3/s.  Are you sure you want to continue?");}
						}
						else
							if (dischargeTimes.q_oil < 0 || dischargeTimes.q_oil > 503389.8288) /*{printNote("Gas discharge rate must be less than 503389 MSCF"); break;}*/
							{strcpy(msg,"CDOG recommends a gas discharge rate of less than 503389 MSCF.  Are you sure you want to continue?");}
					}

					if (GetPopSelection (dialog, TEMPUNITS_POPUP) == 1)
					{
						if (dischargeTimes.temp < 0 || dischargeTimes.temp > 232) /*{printNote("Temperature must be less than 232 degrees C"); break;}*/
						{strcpy(msg,"CDOG recommends a temperature of less than 232 degrees C.  Are you sure you want to continue?");}
					}
					else
					{
						if (dischargeTimes.temp < 0 || dischargeTimes.temp > 165) /*{printNote("Temperature must be less than 449.6 degrees F"); break;}*/
						{strcpy(msg,"CDOG recommends a temperature of less than 449.6 degrees F.  Are you sure you want to continue?");}
					}

					if (GetPopSelection (dialog, DENSITYUNITS_POPUP) == 1)
					{
						if (dischargeTimes.rho_oil < 700 || dischargeTimes.rho_oil > 1060) /*{printNote("Density must be between 700 and 1060 kg/m3"); break;}*/
						{strcpy(msg,"CDOG recommends a density between 700 and 1060 kg/m3.  Are you sure you want to continue?");}
					}
					else
					{
						if (dischargeTimes.rho_oil < 2 || dischargeTimes.rho_oil > 70.64) /*{printNote("Density must be between 2 and 70.64 API"); break;}*/
						{strcpy(msg,"CDOG recommends a density between 2 and 70.64 API.  Are you sure you want to continue?");}
					}

					if (GetPopSelection (dialog, DIAMETERUNITS_POPUP) == 1)
					{
						if (dischargeTimes.diam < 0 || dischargeTimes.diam > 2) /*{printNote("Orifice diameter must be less than 2 meters"); break;}*/
						{strcpy(msg,"CDOG recommends an orifice diameter of less than 2 meters.  Are you sure you want to continue?");}
					}
					else if (GetPopSelection (dialog, DIAMETERUNITS_POPUP) == 2)
					{
						if (dischargeTimes.diam < 0 || dischargeTimes.diam > 200) /*{printNote("Orifice diameter must be less than 200 centimeters"); break;}*/
						{strcpy(msg,"CDOG recommends an orifice diameter of less than 200 centimeters.  Are you sure you want to continue?");}
					}
					else 
					{
						if (dischargeTimes.diam < 0 || dischargeTimes.diam > 78.74) /*{printNote("Orifice diameter must be less than 78.74 inches"); break;}*/
						{strcpy(msg,"CDOG recommends an orifice diameter of less than 78.74 inches.  Are you sure you want to continue?");}
					}

					if(msg[0]) 
					{
						short buttonSelected;
						{
							// point out that some value is not within the recommended range
							buttonSelected  = MULTICHOICEALERT(1690,msg,TRUE);
							switch(buttonSelected){
								case 1:// continue
									break;  
								case 3: // cancel
									return 0;// stay at this dialog
									break;
							}
						}
					}

					if(dischargeTimesh == 0)
					{
						dischargeTimesh = (DischargeDataH)_NewHandle(n*sizeof(DischargeData));
						if(!dischargeTimesh)
						{
							TechError("ConstDischargeClick:OKAY", "_NewHandle()", 0);
							//return VARDISCHARGE_CANCEL;
							break; // make them cancel so that code gets executed
						}
						sgDischargeTimesH = dischargeTimesh;
					}
					else
					{
						 _SetHandleSize((Handle)dischargeTimesh,n*sizeof(DischargeData));
						 if(_MemError())
						 {
							 TechError("ConstDischargeClick:OKAY", "_NewHandle()", 0);
							//return VARDISCHARGE_CANCEL;
							break; // make them cancel, so that code gets executed
						 }
					}
					(*dischargeTimesh)[0]=dischargeTimes;					
					continuousDischarge = GetButton (dialog, VARDISCHARGE_CONTINUOUS);
					sDialogCDOGParameters.duration = EditText2Float(dialog,VARDISCHARGE_DURATION) * 3600.;
					sDialogCDOGParameters.isContinuousRelease = continuousDischarge;
			}
				theTypeEQ = GetPopSelection (dialog, CDOGSETTINGSEQ);
				bubbleRadius = EditText2Float(dialog,CDOGSETTINGSRAD)/1000.;
				//orificeDiameter = EditText2Float(dialog,CDOGSETTINGSDO);
				hydrateDensity = EditText2Float(dialog,CDOGSETTINGSHYDDENS);
				molecularWt = EditText2Float(dialog,CDOGSETTINGSMOLWT);
				if (theTypeEQ==1)
				{
					//if(hydrateDensity!=900){printNote("Methane hydrate density is 900"); break;}
				}
				else if (theTypeEQ==2)
				{
					if(hydrateDensity>940 ||hydrateDensity<900){printNote("Natural gas hydrate density must be between 900 and 940"); break;}	// hard limit
				}
				if (bubbleRadius>.01) {printNote("Maximum bubble radius is 10 mm"); break;}	// switch to soft limit?
				if (bubbleRadius==0 /*|| orificeDiameter==0*/) {printNote("Parameter value must be greater than zero"); break;}	// hard limit
				if (theTypeEQ == 1)
				{
					sDialogCDOGParameters.equilibriumCurves = 1;	// methane
					sDialogCDOGParameters.hydrateDensity = 900;
					sDialogCDOGParameters.molecularWt = .016;
				}
				else if (theTypeEQ == 2)
				{
					molecularWt = EditText2Float(dialog,CDOGSETTINGSMOLWT);
					molWtUnits = GetPopSelection (dialog, CDOGSETTINGSMOLWTUNITS);
					if (molWtUnits == 1)
					{
						if (molecularWt < 15 || molecularWt > 36) {printNote("Natural gas molecular weight must be between 15 and 36 g/mol"); break;}	// hard limit
					}
					else
					{
						if (molecularWt < .015 || molecularWt > .036) {printNote("Natural gas molecular weight must be between .015 and .036 kg/mol"); break;}	// hard limit
					}
					sDialogCDOGParameters.equilibriumCurves = 2; 	// natural gas
					sDialogCDOGParameters.hydrateDensity = EditText2Float(dialog,CDOGSETTINGSHYDDENS);	// 900-940
					sDialogCDOGParameters.molecularWt = EditText2Float(dialog,CDOGSETTINGSMOLWT);	//~.019
					sUserUnits.molWtUnits = GetPopSelection (dialog, CDOGSETTINGSMOLWTUNITS);
				}
				theTypeHydProc = GetPopSelection (dialog, CDOGSETTINGSHYDPROC);
				if (theTypeHydProc == 1)
					sDialogCDOGParameters.hydrateProcess = 1;	// do not suppress
				else if (theTypeHydProc == 2)
					sDialogCDOGParameters.hydrateProcess = 0; 	// suppress
				theTypeSep = GetPopSelection (dialog, CDOGSETTINGSSEP);
				if (theTypeSep == 1)
					sDialogCDOGParameters.separationFlag = 0;	// no separation
				else if (theTypeSep == 2)
					sDialogCDOGParameters.separationFlag = 1; 	// separation possible
				theTypeDropSize = GetPopSelection (dialog, CDOGSETTINGSDROPSIZE);
				if (theTypeDropSize == 1)
					sDialogCDOGParameters.dropSize = 0;	// use CDOG default
				else if (theTypeDropSize == 2)
					sDialogCDOGParameters.dropSize = 1; 	// user supplied drop sizes
				sDialogCDOGParameters.bubbleRadius = EditText2Float(dialog,CDOGSETTINGSRAD)/1000.;
				//sDialogCDOGParameters.molecularWt = EditText2Float(dialog,CDOGSETTINGSMOLWT);
				//sDialogCDOGParameters.hydrateDensity = EditText2Float(dialog,CDOGSETTINGSHYDDENS);
				sDialogCDOGParameters.dropSize = GetPopSelection (dialog, CDOGSETTINGSDROPSIZE)-1;
				sDialogCDOGParameters.dropSize = GetPopSelection (dialog, CDOGSETTINGSDROPSIZE)-1;
				
				sUserUnits.temperatureUnits = GetPopSelection (dialog, TEMPUNITS_POPUP);
				sUserUnits.densityUnits = GetPopSelection (dialog, DENSITYUNITS_POPUP);
				sUserUnits.diameterUnits = GetPopSelection (dialog, DIAMETERUNITS_POPUP);
				sUserUnits.dischargeType = GetPopSelection (dialog, CDOGSPILLDRPOPUP);
				sUserUnits.gorUnits = GetPopSelection (dialog, GORUNITS_POPUP);
				if (sUserUnits.dischargeType==1)
					sUserUnits.dischargeUnits = GetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP);
				else			
					sUserUnits.dischargeUnits = GetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP2);
				if (GetPopSelection (dialog, GASYESNO_POPUP) ==1)
					sHydSulfide = true;  
				else
					sHydSulfide = false;
				
				/////////////////////////////
				DisposeVARDISCHARGEStuff();
				
				return VARDISCHARGE_OK;
			}
		}
			
		case VARDISCHARGE_CANCEL:
			//SetEPDialogNonPtrFields(sgWindMover,&sharedEPDialogNonPtrFields);
			DisposeVARDISCHARGEStuff();
			return VARDISCHARGE_CANCEL;
			break;
			
		case VARDISCHARGE_INCREMENT:
		case VARDISCHARGE_TIME:
		case CDOGSPILLDRPOPUPVAL:
		case VARDISCHARGE_QGAS:
		case VARDISCHARGE_DIAM:
		case VARDISCHARGE_TEMP:	// code goes here, allow negative? (if degree C only)
		case VARDISCHARGE_RHOOIL:
		case VARDISCHARGE_DURATION:
		//case VARDISCHARGE_NDEN:
		//case VARDISCHARGE_OUTPUTINT:
		//case VARDISCHARGE_DXDY:
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;
			
		//case VARDISCHARGE_NUMCELLS:
			//CheckNumberTextItem(dialog, itemNum, FALSE); // don't allow decimals
			//break;
						
		case VARDISCHARGE_DELETEALL:
			sgDischargeTimes->ClearList();
			VLReset(&sgObjects,1);
			UpdateDisplayWithCurSelection(dialog);
			break;
		case VARDISCHARGE_DELETEROWS_BTN:
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				sgDischargeTimes->DeleteItem(curSelection);
				VLDeleteItem(curSelection,&sgObjects);
				if(sgObjects.numItems == 0)
				{
					VLAddItem(1,&sgObjects);
					VLSetSelect(0,&sgObjects);
				}
				--curSelection;
				if(curSelection >-1)
				{
					VLSetSelect(curSelection,&sgObjects);
				}
				VLUpdate(&sgObjects);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
		case CDOGSPILLDRPOPUP:	// may put in oil/gas discharge popup here
			{
				short dischargeUnits, dischargeType = GetPopSelection (dialog, CDOGSPILLDRPOPUP);
				if (dischargeType==1)
					dischargeUnits = GetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP);
				else			
					dischargeUnits = GetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP2);

				
				PopClick(dialog, itemNum, &menuID_menuItem);
				// code goes here, enable appropriate units popup
				if (GetPopSelection(dialog, CDOGSPILLDRPOPUP) == 1) 
					{mysetitext(dialog,VARDISCHARGE_QOIL_LIST_LABEL,"Q oil");EnableOilDischargeUnits(dialog,true,dischargeUnits);}
				else 
					{mysetitext(dialog,VARDISCHARGE_QOIL_LIST_LABEL,"Q gas");EnableOilDischargeUnits(dialog,false,dischargeUnits);}
			}
			break;
		case VARDISCHARGE_REPLACE:
			err = RetrieveIncrementDischargeTime(dialog);
			if(err) break;
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				err=GetTimeVals(dialog,&dischargeTimes);
				if(err) break;
	
				if(curSelection==sgDischargeTimes->GetItemCount())
				{
					// replacing blank record
					err = AddReplaceRecord(dialog,INCREMENT_TIME,!REPLACE,dischargeTimes);
					SelectNthRow(dialog, curSelection+1 ); 
				}
				else // replacing existing record
				{
					VLGetSelect(&curSelection,&sgObjects);
					sgDischargeTimes->DeleteItem(curSelection);
					VLDeleteItem(curSelection,&sgObjects);		
					err = AddReplaceRecord(dialog,!INCREMENT_TIME,REPLACE,dischargeTimes);
				}
			}
			break;

		case VARDISCHARGE_LIST:
			// retrieve every time they click on the list
			// because clicking can cause the increment to be hidden
			// and we need to verify it before it gets hidden
			err = RetrieveIncrementDischargeTime(dialog);
			if(err) break;
			///////////
			pos=GetMouseLocal(GetDialogWindow(dialog));
			VLClick(pos, &sgObjects);
			VLUpdate(&sgObjects);
			VLGetSelect(&curSelection,&sgObjects);
			if(curSelection == -1 )
			{
				curSelection = sgObjects.numItems-1;
				VLSetSelect(curSelection,&sgObjects);
				VLUpdate(&sgObjects);
			}
			
			//ShowHideAutoIncrement(dialog,curSelection);
			// moved into UpdateDisplayWithCurSelection()
		
			if (AddRecordRowIsSelected3())
			{
				DischargeData dischargeTimes;
				sgDischargeTimes->GetListItem((Ptr)&dischargeTimes,sgDischargeTimes->GetItemCount()-1);
				err = RetrieveIncrementDischargeTime(dialog);
				if(err) break;
				IncrementDischargeTime(dialog,dischargeTimes.time);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
			
		case VARDISCHARGE_CONTINUOUS:
		{
			Boolean continuousRelease; 
			ToggleButton(dialog, itemNum);
			continuousRelease = GetButton (dialog, itemNum);
			ShowHideDialogItem(dialog, VARDISCHARGE_DURATIONLABEL, !continuousRelease); 
			ShowHideDialogItem(dialog, VARDISCHARGE_DURATION, !continuousRelease); 
			ShowHideDialogItem(dialog, VARDISCHARGE_DURATIONUNITS, !continuousRelease); 
			break;
		}
		case VARDISCHARGETYPE_POPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowCDOGVarDischargeItems(dialog);
			if (GetPopSelection(dialog, VARDISCHARGETYPE_POPUP) == 2) MySelectDialogItemText(dialog, VARDISCHARGE_TIME, 0, 100);
			break;
		
		case CDOGSETTINGSMOLWTUNITS:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
			
		case GASYESNO_POPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			if (GetPopSelection(dialog, GASYESNO_POPUP) == 1)
			{
				printNote("Hydrogen sulfide can be dangerous or deadly to workers at the surface");
			}
			break;
			
		case CDOGSETTINGSHYDPROC: 
		case CDOGSETTINGSSEP: 
		case CDOGSETTINGSDROPSIZE:  
	//	case TEMPUNITS_POPUP:  
	//	case DIAMETERUNITS_POPUP:  
	//	case DENSITYUNITS_POPUP:  
		case CDOGSPILLDRUNITSPOPUP:
		case CDOGSPILLDRUNITSPOPUP2:
		case GORUNITS_POPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
		case CDOGSETTINGSEQ:  
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideParameters(dialog);
			break;

		case TEMPUNITS_POPUP:  
		case DIAMETERUNITS_POPUP:  
		case DENSITYUNITS_POPUP:  
		{
			//short unitsAfterClick,unitsB4Click = GetPopSelection (dialog, itemNum);
			PopClick(dialog, itemNum, &menuID_menuItem);
			//unitsAfterClick = GetPopSelection (dialog, itemNum);
			//if (unitsB4Click!=unitsAfterClick) UpdateListItemWithUnitsSelected(dialog,itemNum,unitsB4Click,unitsAfterClick);
			break;
		}
		case CDOGSETTINGSRAD:  
		case CDOGSETTINGSMOLWT:  
		case CDOGSETTINGSHYDDENS: 
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
	}
	 
	return 0;
}

		
void DrawDischargeItemsList(DialogPtr w, RECTPTR r, long n)
{
	char s[256];
	DischargeData dischargeTimes;
	
	if(n == sgObjects.numItems-1)
	{
		strcpy(s,"****");
	 	MyMoveTo(TIME_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(Q_OIL_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(Q_GAS_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(TEMP_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(DIAM_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	//MyMoveTo(DIR_COL-20,r->bottom); //JLM
	 	MyMoveTo(RHO_OIL_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	//MyMoveTo(N_DEN_COL-stringwidth(s)/2,r->bottom);
	//	drawstring(s);
	 //	MyMoveTo(OUTPUT_INT_COL-stringwidth(s)/2,r->bottom);
	//	drawstring(s);
	 	return; 
	}
		
	sgDischargeTimes->GetListItem((Ptr)&dischargeTimes,n);
	
	StringWithoutTrailingZeros(s,dischargeTimes.time,1);
	MyMoveTo(TIME_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	StringWithoutTrailingZeros(s,dischargeTimes.q_oil,3);
	MyMoveTo(Q_OIL_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	StringWithoutTrailingZeros(s,dischargeTimes.q_gas,1);
	MyMoveTo(Q_GAS_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);

	StringWithoutTrailingZeros(s,dischargeTimes.temp,1);
	MyMoveTo(TEMP_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	StringWithoutTrailingZeros(s,dischargeTimes.diam,1);
	MyMoveTo(DIAM_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	//MyMoveTo(DIR_COL-20,r->bottom);//JLM
	StringWithoutTrailingZeros(s,dischargeTimes.rho_oil,1);
	MyMoveTo(RHO_OIL_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	//StringWithoutTrailingZeros(s,dischargeTimes.n_den,1);
	//MyMoveTo(N_DEN_COL-stringwidth(s)/2,r->bottom);
	//drawstring(s);
	
	//StringWithoutTrailingZeros(s,dischargeTimes.output_int,1);
	//MyMoveTo(OUTPUT_INT_COL-stringwidth(s)/2,r->bottom);
	//drawstring(s);
	
	return;
}


pascal_ifMac void DischargeItemsListUpdate(DialogPtr dialog, short itemNum)
{
	Rect r = GetDialogItemBox(dialog,VARDISCHARGE_LIST);
	
	VLUpdate(&sgObjects);
}

OSErr VarDischargeInit(DialogPtr dialog, VOIDPTR data)
{
	Rect r = GetDialogItemBox(dialog, VARDISCHARGE_LIST);
	DischargeData dischargeTimes;
	long i,n;
	OSErr err = 0;
	short IBMoffset;
	
	//RegisterPopTable(CDOGVarDischargePopTable, 3);
	RegisterPopUpDialog(VAR_DISCHARGE_DLGID, dialog);
	
	SetDialogItemHandle(dialog, VARDISCHARGE_HILITEDDEFAULT, (Handle)FrameDefault);

	if (sDialogCDOGParameters.equilibriumCurves == 1)
	{
		mysetitext(dialog, CDOGSETTINGSMOLWT, ".016");	// .016 kg/mol methane
		//Float2EditText(dialog,CDOGSETTINGSHYDDENS,sDialogCDOGParameters.hydrateDensity,2);
	}
	else if (sDialogCDOGParameters.equilibriumCurves == 2)
	{
		//mysetitext(dialog, CDOGSETTINGSMOLWT, ".0191");	// .0191 kg/mol natural gas
		Float2EditText(dialog,CDOGSETTINGSMOLWT,sDialogCDOGParameters.molecularWt,2);
		//Float2EditText(dialog,CDOGSETTINGSHYDDENS,sDialogCDOGParameters.hydrateDensity,2);
	}
	SetPopSelection (dialog, CDOGSETTINGSHYDPROC, sDialogCDOGParameters.hydrateProcess == 1 ? 1 : 2);
	SetPopSelection (dialog, CDOGSETTINGSEQ, sDialogCDOGParameters.equilibriumCurves == 1 ? 1 : 2);
	SetPopSelection (dialog, CDOGSETTINGSSEP, sDialogCDOGParameters.separationFlag == 0 ? 1 : 2);
	SetPopSelection (dialog, CDOGSETTINGSDROPSIZE, sDialogCDOGParameters.dropSize == 0 ? 1 : 2);

	//code goes here, will need to store and remember
	SetPopSelection (dialog, TEMPUNITS_POPUP, sUserUnits.temperatureUnits);
	SetPopSelection (dialog, DIAMETERUNITS_POPUP, sUserUnits.diameterUnits);
	SetPopSelection (dialog, DENSITYUNITS_POPUP, sUserUnits.densityUnits);
	SetPopSelection (dialog, GORUNITS_POPUP, sUserUnits.gorUnits);
	if (sUserUnits.dischargeType==1)
		SetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP, sUserUnits.dischargeUnits);
	else
		SetPopSelection (dialog, CDOGSPILLDRUNITSPOPUP2, sUserUnits.dischargeUnits);
	
	//SetPopSelection (dialog, CDOGSPILLDRPOPUP, sCDOGSpillParameters.dischargeRateType);
	//if (sCDOGSpillParameters.dischargeRateType == 1)
	SetPopSelection (dialog, CDOGSPILLDRPOPUP, sUserUnits.dischargeType);
	if (sUserUnits.dischargeType == 1)
	{
		Float2EditText (dialog, CDOGSPILLDRPOPUPVAL, sCDOGSpillParameters2.oilDischargeRate, 2);
		EnableOilDischargeUnits(dialog,true,sUserUnits.dischargeUnits);
	}
	else
	{
		Float2EditText (dialog, CDOGSPILLDRPOPUPVAL, sCDOGSpillParameters2.gasDischargeRate, 2);
		EnableOilDischargeUnits(dialog,false,sUserUnits.dischargeUnits);
	}
	Float2EditText(dialog,VARDISCHARGE_DURATION, sDialogCDOGParameters.duration / 3600. , 2);
	SetButton (dialog, VARDISCHARGE_CONTINUOUS, sDialogCDOGParameters.isContinuousRelease);

	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
		
	{
		DischargeDataH dischargeTimesh = sgDischargeTimesH;

		sgDischargeTimes = new CDischargeTimes(sizeof(DischargeData));
		if(!sgDischargeTimes)return -1;
		if(sgDischargeTimes->IList())return -1;
		if(dischargeTimesh)
		{
			// copy list to temp list
			n = _GetHandleSize((Handle)dischargeTimesh)/sizeof(DischargeData);
			for(i=0;i<n;i++)
			{
				dischargeTimes=(*dischargeTimesh)[i];
				err=sgDischargeTimes->AppendItem((Ptr)&dischargeTimes);
				if(err)return err;
			}
		}
		else  n=0;
		
		n++; // Always have blank row at bottom
			
		err = VLNew(dialog, VARDISCHARGE_LIST, &r,n, DrawDischargeItemsList, &sgObjects);
		if(err) return err;
	}
	
	if (n<=2) SetPopSelection (dialog, VARDISCHARGETYPE_POPUP, 1);
	else 
		SetPopSelection (dialog, VARDISCHARGETYPE_POPUP, 2);


	SetPopSelection (dialog, CDOGSETTINGSMOLWTUNITS, sUserUnits.molWtUnits);	// for now set to g/mol, should store and remember
	SetPopSelection (dialog, GASYESNO_POPUP, (sHydSulfide ? 1 : 2));	
	//ShowCDOGVarDischargeItems(dialog);
	ShowHideDialogItem(dialog,VARDISCHARGE_FRAMEINPUT,false);// hide this frame
	//SetDialogItemHandle(dialog,VARDISCHARGE_FRAMEINPUT,(Handle)FrameEmbossed);
	//SetDialogItemHandle(dialog,VARDISCHARGE_BUTTONFRAME,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,VARDISCHARGE_FRAMEINPUT2,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,VARDISCHARGE_LIST,(Handle)DischargeItemsListUpdate);
	
	r = GetDialogItemBox(dialog,VARDISCHARGE_LIST);
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(dialog, VARDISCHARGE_TIME_LIST_LABEL);TIME_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, VARDISCHARGE_QOIL_LIST_LABEL);Q_OIL_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, VARDISCHARGE_QGAS_LIST_LABEL);Q_GAS_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, VARDISCHARGE_TEMP_LIST_LABEL);TEMP_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, VARDISCHARGE_DIAM_LIST_LABEL);DIAM_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, VARDISCHARGE_RHOOIL_LIST_LABEL);RHO_OIL_COL=(r.left+r.right)/2-IBMoffset;
	//r = GetDialogItemBox(dialog, VARDISCHARGE_SAL_LIST_LABEL);N_DEN_COL=(r.left+r.right)/2-IBMoffset;
	//r = GetDialogItemBox(dialog, VARDISCHARGE_SAL_LIST_LABEL);OUTPUT_INT_COL=(r.left+r.right)/2-IBMoffset;

	// might want to use values of first set if there are any
	Float2EditText(dialog, VARDISCHARGE_INCREMENT,sIncrementInHours, 0);
	Float2EditText(dialog, VARDISCHARGE_TIME,0.0, 0);
	Float2EditText(dialog, CDOGSPILLDRPOPUPVAL,0.01829, 0);	// turn into a popup oil/gas
	Float2EditText(dialog, VARDISCHARGE_QGAS,3.659, 0);	// turn into GOR ??
	//Float2EditText(dialog, VARDISCHARGE_QGAS,200, 0);	// turn into GOR ??
	Float2EditText(dialog, VARDISCHARGE_DIAM,0.1, 0);
	Float2EditText(dialog, VARDISCHARGE_TEMP,80.0, 0);
	Float2EditText(dialog, VARDISCHARGE_RHOOIL,842.5, 0);
	//Float2EditText(dialog, VARDISCHARGE_NDEN,0.0, 0);
	//Float2EditText(dialog, VARDISCHARGE_OUTPUTINT,0.0, 0);

	Float2EditText(dialog,CDOGSETTINGSRAD,sDialogCDOGParameters.bubbleRadius*1000.,2);
	//Float2EditText(dialog,CDOGSETTINGSMOLWT,sDialogCDOGParameters.molecularWt,2);
	Float2EditText(dialog,CDOGSETTINGSHYDDENS,sDialogCDOGParameters.hydrateDensity,2);
	ShowHideParameters(dialog);

	//SetDialogItemHandle(dialog, VARDISCHARGE_HILITEDDEFAULT, (Handle)FrameDefault);
	
	//ShowHideDialogItem(dialog,VARDISCHARGE_HILITEDDEFAULT,false);//JLM, hide this item, this dialog has no default

	UpdateDisplayWithCurSelection(dialog);
	
	ShowCDOGVarDischargeItems(dialog);
	//MySelectDialogItemText(dialog, VARDISCHARGE_TIME, 0, 100);//JLM
	SetDialogItemHandle(dialog, VARDISCHARGE_HILITEDDEFAULT, (Handle)FrameDefault);
	return 0;
}

									   
OSErr CDOGVarDischargeDialog(DischargeDataH *dischargeTimes, Boolean *hydrogenSulfide,CDOGUserUnits *userUnits, CDOGParameters *spillParameters, CDOGSpillParameters *spillParameters2,WindowPtr parentWindow)
{
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j, numItems = 0;
	PopInfoRec combinedDialogsPopTable[21];

	sgDischargeTimesH = *dischargeTimes;
	sDialogCDOGParameters = *spillParameters;
	sCDOGSpillParameters2 = *spillParameters2;
	
	//sCellLength = *cellLength;
	//sNumCells = *numCells;
	sHydSulfide = *hydrogenSulfide;
	sUserUnits = *userUnits;

	for(j = 0; j < sizeof(CDOGVarDischargePopTable) / sizeof(PopInfoRec);j++)
		combinedDialogsPopTable[numItems++] = CDOGVarDischargePopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedDialogsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedDialogsPopTable,numItems);

	item = MyModalDialog(VAR_DISCHARGE_DLGID, mapWindow, 0, VarDischargeInit, VarDischargeClick);
	RestorePopTableInfo(saveTable);
	SetWatchCursor();
	if(item == VARDISCHARGE_OK)
	{
		*dischargeTimes = sgDischargeTimesH;
		//*cellLength = sCellLength;
		//*numCells = sNumCells;
		*hydrogenSulfide = sHydSulfide;
		*userUnits = sUserUnits;
		*spillParameters = sDialogCDOGParameters;
		model->NewDirtNotification();// JLM
		return 0;
	}
	else if(item == VARDISCHARGE_CANCEL) return USERCANCEL;
	else return -1;
}
