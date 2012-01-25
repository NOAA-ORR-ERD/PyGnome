#include "Cross.h"
#include "OSSM.h"
#include "EditCDOGProfilesDialog.h"


#ifdef MPW
#pragma SEGMENT EDITPROFILES
#endif
	
static VList sgObjects;
static CProfilesList *sgDepthVals=0;
static DepthValuesSetH sgDepthValuesH=0;

static float sIncrementInMeters = 250.0;


///////////////////////////////////////////////////////////////////////////

#define INCREMENT_DEPTH true
#define REPLACE true

static short DEPTH_COL,U_COL,V_COL,W_COL,TEMP_COL,SAL_COL;

/*Boolean AddRecordRowIsSelected2(void)
{
	long curSelection;
	Boolean  isLastItem;
	VLGetSelect(&curSelection,&sgObjects);
	isLastItem = curSelection == -1 || curSelection == sgObjects.numItems-1;
	return isLastItem;
}*/

/*Boolean ShowAutoIncrement2(void)
{
	//Boolean show = AddRecordRowIsSelected2();
	Boolean show = VLAddRecordRowIsSelected(&sgObjects);
	return show;
}
*/
static void ShowHideAutoIncrement(DialogPtr dialog,long curSelection)
{
	//Boolean show = ShowAutoIncrement2();
	//Boolean show = AddRecordRowIsSelected2();
	Boolean show = VLAddRecordRowIsSelected(&sgObjects);
	ShowHideDialogItem(dialog,EPINCREMENT,show);
	ShowHideDialogItem(dialog,EPAUTOINCRTEXT,show);
	ShowHideDialogItem(dialog,EPAUTOINCRMETERS,show);
}

OSErr RetrieveIncrementDepth(DialogPtr dialog)
{
	OSErr err = 0;
	float incrementDepthInMeters  = EditText2Float(dialog,EPINCREMENT);
	if(incrementDepthInMeters < 20. || incrementDepthInMeters > 500.0)
	{
		printError("The increment depth must be between 20. and 500.");
		MySelectDialogItemText (dialog, EPINCREMENT, 0, 255);
		return -1;
	}
	sIncrementInMeters = incrementDepthInMeters;// set the static
	return noErr;
}

static void IncrementDepth(DialogPtr dialog, float depth)
{
	float incr;
	incr = sIncrementInMeters;
	depth += incr;
	// update dialog value
	char depthstr[30];
	StringWithoutTrailingZeros(depthstr,depth,1);
	mysetitext(dialog,EPDEPTH,depthstr);	// or something
}

static void MyFloat2EditText(DialogPtr dialog, short itemnum,float num,short numdec)
{
	char numstr[30];
	StringWithoutTrailingZeros(numstr,num,numdec);
	mysetitext(dialog,itemnum,numstr);
}

// need this
static OSErr GetDepthVals(DialogPtr dialog,DepthValuesSet *dvals)
{
	float depth = EditText2Float(dialog,EPDEPTH);
	float vel1 = EditText2Float(dialog,EPU);
	float vel2 = EditText2Float(dialog,EPV);
	//float vel3 = EditText2Float(dialog,EPW);
	float vel3 =0.;
	float temp = EditText2Float(dialog,EPTEMP);
	float sal = EditText2Float(dialog,EPSAL);
	OSErr  err = 0;
	char errStr[256] = "", msg[256] = "";
	 
	 // might want some checking here
	 // should check multiple combinations. Keep copying

	if (depth < 0 || depth > 12000) /*{printNote("Depth must be less than 12000 meters"); return -1;}*/
	{strcat(msg,"CDOG recommends a depth of less than 12000 meters.\n");}
	if (vel1 < -50 || vel1 > 50 || vel2 < -50 || vel2 > 50) /*{printNote("Current speed must be less than 50 m/s"); return -1;}*/
	{strcat(msg,"CDOG recommends a current speed of less than 50 m/s.\n");}
	if (temp < -1 || temp > 35) /*{printNote("Temperature must be less than 35 degrees C"); return -1;}*/
	{strcat(msg,"CDOG recommends a temperature of less than 35 deg C.\n");}
	if (sal < 0 || sal > 50) /*{printNote("Salinity must be less than 50 psu"); return -1;}*/
	{strcat(msg,"CDOG recommends a salinity of less than 35 psu.\n");}

	if(msg[0]) 
	{
		strcat(msg, " Are you sure you want to continue?");
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

	dvals->depth = depth;
	dvals->value.u = vel1;
	dvals->value.v = vel2;
	dvals->w = vel3;
	dvals->temp = temp;
	dvals->sal = sal;
	return err;
}

void UpdateDisplayWithDepthValuesSet(DialogPtr dialog,DepthValuesSet dvals)
{
	char numstr[30];
	StringWithoutTrailingZeros(numstr,dvals.depth,2);
	mysetitext(dialog,EPDEPTH,numstr);

	//MyFloat2EditText(dialog,EPU,dvals.value.u,4);
	//MyFloat2EditText(dialog,EPV,dvals.value.v,4);
	//MyFloat2EditText(dialog,EPTEMP,dvals.temp,2);
	//MyFloat2EditText(dialog,EPSAL,dvals.sal,2);
	Float2EditText(dialog,EPU,dvals.value.u,6);
	Float2EditText(dialog,EPV,dvals.value.v,6);
	Float2EditText(dialog,EPTEMP,dvals.temp,2);
	Float2EditText(dialog,EPSAL,dvals.sal,2);
}

	
static void UpdateDisplayWithCurSelection(DialogPtr dialog)
{
	DepthValuesSet dvals;
	Point pos,mp;
	long curSelection;
	
	//if(!AddRecordRowIsSelected2())
	if(!VLAddRecordRowIsSelected(&sgObjects))
	{	// set the item text
		{
			VLGetSelect(&curSelection,&sgObjects);
			sgDepthVals->GetListItem((Ptr)&dvals,curSelection);
		}
		
		UpdateDisplayWithDepthValuesSet(dialog,dvals);
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

static OSErr AddReplaceRecord(DialogPtr dialog,Boolean incrementDepth,Boolean replace,DepthValuesSet dvals)
{
	long n,itemnum,curSelection;
	OSErr err=0;
	
	if(!err)
	{
		// will need to define InsertSorted for the CDOG profiles data type, sort by depth
		err=sgDepthVals->InsertSorted ((Ptr)&dvals,&itemnum,false);// false means don't allow duplicate times
		
		if(!err) // new record
		{
			VLAddItem(1,&sgObjects);
			VLSetSelect(itemnum, &sgObjects); 
			VLAutoScroll(&sgObjects);
			VLUpdate(&sgObjects);
			if(incrementDepth)IncrementDepth(dialog,dvals.depth);
		}
		else if(err == -2) // found existing record. Replace if okay to replace
		{
			if(replace)
			{
				sgDepthVals->DeleteItem(itemnum);
				VLDeleteItem(itemnum,&sgObjects);
				err = AddReplaceRecord(dialog,!INCREMENT_DEPTH,REPLACE,dvals);
				VLUpdate(&sgObjects);
				if(incrementDepth)IncrementDepth(dialog,dvals.depth);
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

void 	DisposeEPStuff(void)
{
	if(sgDepthVals)
	{
		sgDepthVals->Dispose();// JLM 12/14/98
		delete sgDepthVals;
		sgDepthVals = 0;
	}

	//?? VLDispose(&sgObjects);// JLM 12/10/98, is this automatic on the mac ??
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
}
//static float sCellLength;
//static long sNumCells;

	
short EditProfilesClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	Point pos,mp,clippedPos;
	Rect r;
	double speed, direction;
	long curSelection;
	long dir,i,n;
	unsigned long incr;
	char s[30];
	OSErr err=0,settingsErr = 0;
	CProfilesList *tlist;
	DepthValuesSet dvals;
	
	//if (AddRecordRowIsSelected2())
	if (VLAddRecordRowIsSelected(&sgObjects))
	{
		//Last row is selected
		//Disable delete button
		MyEnableControl(dialog,EPDELETEROWS_BTN,FALSE);
		// And change title in replace dialog to "Add new record"
		MySetControlTitle(dialog, EPREPLACE, "Add New Record");
	}
	else
	{
		MySetControlTitle(dialog, EPREPLACE, "Replace Selected");
		MyEnableControl(dialog,EPDELETEROWS_BTN,TRUE);
	}
	
	switch(itemNum)
	{
		case EPOK:
		{
			// don't retrieve increment here
			// Just use the value from the last time they incremented.
			// Why bother them if we are not going to use the value.
			//if(ShowAutoIncrement2()) err = RetrieveIncrementDepth(dialog);
			//if(err) break;
			
			//sgSpeedUnits = GetPopSelection(dialog, EPSPEEDPOPUP);			

			if(sgDepthVals)
			{
				DepthValuesSetH dvalsh = sgDepthValuesH;
				n = sgDepthVals->GetItemCount();
				if(n == 0)
				{	// no items are entered, tell the user
					char msg[512],buttonName[64];
					GetWizButtonTitle_Cancel(buttonName);
					sprintf(msg,"You have not entered any data values.  Either enter data values and use the 'Add New Record' button, or use the '%s' button to exit the dialog.",buttonName);
					printError(msg);
					break;
				}
				
				// check that all the values are in range - if there is some range
				// or may allow the user to change units
				for(i=0;i<n;i++)
				{
					char errStr[256] = "";
					err=sgDepthVals->GetListItem((Ptr)&dvals,i);
					if(err) {SysBeep(5); break;}// this shouldn't ever happen
					/*UV2RTheta(dvals.value.u,dvals.value.v,&r,&theta);
					err = CheckWindSpeedLimit(r,sgSpeedUnits,errStr);
					if(err)
					{
						strcat(errStr,"  Check your units and each of the records you entered.");
						printError(errStr);
						return 0; // stay in the dialog
					}*/
				}
				//sCellLength = EditText2Float(dialog,EPDXDY);	// use map size instead and calculate km from that 
				//sNumCells = EditText2Float(dialog,EPNUMCELLS);
				// will want to check that spill is inside of the grid, and grid is not super small
				
				/////////////
				// point of no return
				//////////////
				if(dvalsh == 0)
				{
					dvalsh = (DepthValuesSetH)_NewHandle(n*sizeof(DepthValuesSet));
					if(!dvalsh)
					{
						TechError("EditProfilesClick:OKAY", "_NewHandle()", 0);
						//return EPCANCEL;
						break; // make them cancel so that code gets executed
					}
					sgDepthValuesH = dvalsh;
				}
				else
				{
					 _SetHandleSize((Handle)dvalsh,n*sizeof(DepthValuesSet));
					 if(_MemError())
					 {
						 TechError("EditProfilesClick:OKAY", "_NewHandle()", 0);
						//return EPCANCEL;
						break; // make them cancel, so that code gets executed
					 }
				}
				
				for(i=0;i<n;i++)
				{
					if(err=sgDepthVals->GetListItem((Ptr)&dvals,i))return EPOK;
					(*dvalsh)[i]=dvals;					
				}
			}

			/////////////////////////////
			DisposeEPStuff();
			return EPOK;
		}
			
		case EPCANCEL:
			//SetEPDialogNonPtrFields(sgWindMover,&sharedEPDialogNonPtrFields);
			DisposeEPStuff();
			return EPCANCEL;
			break;
				
		case EPINCREMENT:
		case EPDEPTH:
		case EPTEMP:
		case EPSAL:
		//case EPDXDY:
			CheckNumberTextItem(dialog, itemNum, TRUE); //  allow decimals
			break;
			
		case EPU:
		case EPV:
		//case EPDXDY:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE); //  allow decimals
			break;
		//case EPNUMCELLS:
			//CheckNumberTextItem(dialog, itemNum, FALSE); // don't allow decimals
			//break;
						
		case EPDELETEALL:
			sgDepthVals->ClearList();
			VLReset(&sgObjects,1);
			UpdateDisplayWithCurSelection(dialog);
			break;
		case EPDELETEROWS_BTN:
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				sgDepthVals->DeleteItem(curSelection);
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
		//case EPSPEEDPOPUP:
			//{
				//PopClick(dialog, itemNum, &sgSpeedUnits);
			//}
			//break;
		case EPREPLACE:
			err = RetrieveIncrementDepth(dialog);
			if(err) break;
			if (VLGetSelect(&curSelection, &sgObjects))
			{
				err=GetDepthVals(dialog,&dvals);
				if(err) break;
	
				if(curSelection==sgDepthVals->GetItemCount())
				{
					// replacing blank record
					err = AddReplaceRecord(dialog,INCREMENT_DEPTH,!REPLACE,dvals);
					SelectNthRow(dialog, curSelection+1 ); 
				}
				else // replacing existing record
				{
					VLGetSelect(&curSelection,&sgObjects);
					sgDepthVals->DeleteItem(curSelection);
					VLDeleteItem(curSelection,&sgObjects);		
					err = AddReplaceRecord(dialog,!INCREMENT_DEPTH,REPLACE,dvals);
				}
			}
			break;

		case EPLIST:
			// retrieve every time they click on the list
			// because clicking can cause the increment to be hidden
			// and we need to verify it before it gets hidden
			err = RetrieveIncrementDepth(dialog);
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
		
			//if (AddRecordRowIsSelected2())
			if (VLAddRecordRowIsSelected(&sgObjects))
			{
				DepthValuesSet dvals;
				sgDepthVals->GetListItem((Ptr)&dvals,sgDepthVals->GetItemCount()-1);
				err = RetrieveIncrementDepth(dialog);
				if(err) break;
				IncrementDepth(dialog,dvals.depth);
			}
			UpdateDisplayWithCurSelection(dialog);
			break;
		
	}
	 
	return 0;
}

		
void DrawProfilesList(DialogPtr w, RECTPTR r, long n)
{
	char s[256];
	DepthValuesSet dvals;
	
	if(n == sgObjects.numItems-1)
	{
		strcpy(s,"****");
	 	MyMoveTo(DEPTH_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(U_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(V_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(W_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	MyMoveTo(TEMP_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	//MyMoveTo(DIR_COL-20,r->bottom); //JLM
	 	MyMoveTo(SAL_COL-stringwidth(s)/2,r->bottom);
		drawstring(s);
	 	return; 
	}
		
	sgDepthVals->GetListItem((Ptr)&dvals,n);
	
	StringWithoutTrailingZeros(s,dvals.depth,1);
	MyMoveTo(DEPTH_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	StringWithoutTrailingZeros(s,dvals.value.u,4);
	MyMoveTo(U_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	StringWithoutTrailingZeros(s,dvals.value.v,4);
	MyMoveTo(V_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);

	StringWithoutTrailingZeros(s,dvals.w,1);
	MyMoveTo(W_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	StringWithoutTrailingZeros(s,dvals.temp,1);
	MyMoveTo(TEMP_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	//MyMoveTo(DIR_COL-20,r->bottom);//JLM
	StringWithoutTrailingZeros(s,dvals.sal,1);
	MyMoveTo(SAL_COL-stringwidth(s)/2,r->bottom);
	drawstring(s);
	
	return;
}


pascal_ifMac void ProfilesListUpdate(DialogPtr dialog, short itemNum)
{
	Rect r = GetDialogItemBox(dialog,EPLIST);
	
	VLUpdate(&sgObjects);
}

OSErr EditProfilesInit(DialogPtr dialog, VOIDPTR data)
{
	Rect r = GetDialogItemBox(dialog, EPLIST);
	float startdepth;
	CProfilesList *tlist;
	DepthValuesSet dvals;
	long i,n;
	OSErr err = 0;
	short IBMoffset;
	
	//RegisterPopTable(prefPopTable, 3);
	//RegisterPopUpDialog(EDIT_PROFILES_DLGID, dialog);
	
	//sharedEPDialogNonPtrFields = GetEPDialogNonPtrFields(sgWindMover);// JLM 11/25/98
	memset(&sgObjects,0,sizeof(sgObjects));// JLM 12/10/98
		
	{
		DepthValuesSetH dvalsh = sgDepthValuesH;

		sgDepthVals = new CProfilesList(sizeof(DepthValuesSet));
		if(!sgDepthVals)return -1;
		if(sgDepthVals->IList())return -1;
		if(dvalsh)
		{
			// copy list to temp list
			n = _GetHandleSize((Handle)dvalsh)/sizeof(DepthValuesSet);
			for(i=0;i<n;i++)
			{
				dvals=(*dvalsh)[i];
				err=sgDepthVals->AppendItem((Ptr)&dvals);
				if(err)return err;
			}
		}
		else  n=0;
		
		n++; // Always have blank row at bottom
			
		err = VLNew(dialog, EPLIST, &r,n, DrawProfilesList, &sgObjects);
		if(err) return err;
	}
	

	SetDialogItemHandle(dialog,EPFRAMEINPUT,(Handle)FrameEmbossed);
	//SetDialogItemHandle(dialog,EPBUTTONFRAME,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,EPLIST,(Handle)ProfilesListUpdate);

	//ShowHideDialogItem(dialog,EPBUTTONFRAME,false);//JLM, hide this frame, we have a different button arrangement
	
	r = GetDialogItemBox(dialog,EPLIST);
#ifdef IBM
	IBMoffset = r.left;
#else 
	IBMoffset = 0;
#endif
	r = GetDialogItemBox(dialog, EPDEPTH_LIST_LABEL);DEPTH_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EPU_LIST_LABEL);U_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EPV_LIST_LABEL);V_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EPW_LIST_LABEL);W_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EPTEMP_LIST_LABEL);TEMP_COL=(r.left+r.right)/2-IBMoffset;
	r = GetDialogItemBox(dialog, EPSAL_LIST_LABEL);SAL_COL=(r.left+r.right)/2-IBMoffset;

	// might want to use values of first set if there are any
	Float2EditText(dialog, EPINCREMENT,sIncrementInMeters, 0);
	Float2EditText(dialog, EPDEPTH,0.0, 0);
	Float2EditText(dialog, EPU,0.0, 0);
	Float2EditText(dialog, EPV,0.0, 0);
	//Float2EditText(dialog, EPW,0.0, 0);
	Float2EditText(dialog, EPTEMP,0.0, 0);
	Float2EditText(dialog, EPSAL,0.0, 0);

	//Float2EditText(dialog, EPDXDY,sCellLength, 0);
	//Float2EditText(dialog, EPNUMCELLS,sNumCells, 0);
	//SetPopSelection (dialog, EPSPEEDPOPUP, sgSpeedUnits);
	//MyDisplayTime(dialog,EPMONTHSPOPUP,sgStartTime);

	//////////

	SetDialogItemHandle(dialog, EPHILITEDDEFAULT, (Handle)FrameDefault);
	
	ShowHideDialogItem(dialog,EPHILITEDDEFAULT,false);//JLM, hide this item, this dialog has no default

	UpdateDisplayWithCurSelection(dialog);
	
	MySelectDialogItemText(dialog, EPDEPTH, 0, 100);//JLM
	return 0;
}

									   
OSErr EditCDOGProfilesDialog(DepthValuesSetH *depthvals, /*float *cellLength, long *numCells,*/ WindowPtr parentWindow)
{
	short item;
	sgDepthValuesH = *depthvals;
	//sCellLength = *cellLength;
	//sNumCells = *numCells;

	item = MyModalDialog(EDIT_PROFILES_DLGID, mapWindow, 0, EditProfilesInit, EditProfilesClick);
	SetWatchCursor();
	if(item == EPOK)
	{
		*depthvals = sgDepthValuesH;
		//*cellLength = sCellLength;
		//*numCells = sNumCells;
		model->NewDirtNotification();// JLM
		return 0;
	}
	else if(item == EPCANCEL) return USERCANCEL;
	else return -1;
}
