// This is a cross between TWindMover and NetCDFMover, possibly reorganize to build off one or the other
// The uncertainty is from the wind, the reading, storing, accessing, displaying data is from NetCDFMover


#include "Cross.h"
#include "NetCDFMover.h"
#include "TWindMover.h"
#include "GridCurMover.h"
#include "GridWindMover.h"
#include "Outils.h"
#include "DagTreeIO.h"
#include "PtCurMover.h"


GridWindMover::GridWindMover(TMap *owner,char* name) : TWindMover(owner, name)
{
	if(!name || !name[0]) this->SetClassName("Gridded Wind");
	else 	SetClassName (name); // short file name
	
	// use wind defaults for uncertainty
	bShowGrid = false;
	bShowArrows = false;
	
	timeGrid = 0;

	fIsOptimizedForStep = false;
	
	fUserUnits = kMetersPerSec;	
	fWindScale = 1.;
	fArrowScale = 10.;
	
}


/*void GridWindMover::Dispose()
{
	if (timeGrid)
	{
		timeGrid -> Dispose();
		//delete fGrid;
		timeGrid = nil;
	}

	TWindMover::Dispose ();
}*/
	 	 
long GridWindMover::GetListLength()
{
	long count = 1; // wind name
	long mode = model->GetModelMode();
	long numTimesInFile = timeGrid->GetNumTimesInFile();
	
	if (bOpen) {
		if(mode == ADVANCEDMODE) count += 1; // active
		if(mode == ADVANCEDMODE) count += 1; // showgrid
		if(mode == ADVANCEDMODE) count += 1; // showarrows
		if(mode == ADVANCEDMODE && model->IsUncertain())count++;
		if(mode == ADVANCEDMODE && model->IsUncertain() && bUncertaintyPointOpen)count+=4;
		if (numTimesInFile>0 || timeGrid->GetNumFiles()>1) count +=2;	// start and end times, otherwise it's steady state
	}
	
	return count;
}

ListItem GridWindMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64], dateStr[64];
	long numTimesInFile = timeGrid->GetNumTimesInFile();
	ListItem item = { this, n, indent, 0 };
	long mode = model->GetModelMode();
	
	if (n == 0) {
		item.index = I_GRIDWINDNAME;
		if (mode == ADVANCEDMODE) item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text,"Wind File: ");
		strcat(text,fFileName);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	if (bOpen) {
		
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_GRIDWINDACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_GRIDWINDSHOWGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Grid");
			
			return item;
		}
		
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_GRIDWINDSHOWARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,fArrowScale,6);
			//strcpy(text, "Show Velocity Vectors");
			sprintf(text, "Show Velocities (@ 1 in = %s m/s) ",valStr);
			
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
		
		if(mode == ADVANCEDMODE && model->IsUncertain())
		{
			if (--n == 0) 
			{
				item.indent++;
				item.index = I_NETCDFWINDUNCERTAIN;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				
				return item;
			}
			
			if(bUncertaintyPointOpen)
			{
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDSTARTTIME;
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainStartTime)/3600.);
					return item;
				}
				
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDDURATION;
					sprintf(text, "Duration: %.2f hr", (float)(fDuration / 3600.0));
					return item;
				}
				
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDSPEEDSCALE;
					sprintf(text, "Speed Scale: %.2f ", fSpeedScale);
					return item;
				}
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDANGLESCALE;
					sprintf(text, "Angle Scale: %.2f ", fAngleScale);
					return item;
				}
			}
		}
	}
	
	item.owner = 0;
	
	return item;
}

Boolean GridWindMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_NETCDFWINDNAME: bOpen = !bOpen; return TRUE;
			case I_NETCDFWINDACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
			case I_NETCDFWINDSHOWGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_NETCDFWINDSHOWARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_NETCDFWINDUNCERTAIN:bUncertaintyPointOpen = !bUncertaintyPointOpen;return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_NETCDFWINDNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
	}
	
	if (doubleClick)
	{	
		switch(item.index)
		{
			case I_NETCDFWINDACTIVE:
			case I_NETCDFWINDUNCERTAIN:
			case I_NETCDFWINDSPEEDSCALE:
			case I_NETCDFWINDANGLESCALE:
			case I_NETCDFWINDSTARTTIME:
			case I_NETCDFWINDDURATION:
			case I_NETCDFWINDNAME:
				GridWindSettingsDialog(this, this -> moverMap,false,mapWindow);
				//WindSettingsDialog(this, this -> moverMap,false,mapWindow,false);
				break;
			default:	// why not call this for everything?
				GridWindSettingsDialog(this, this -> moverMap,false,mapWindow);
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean GridWindMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_NETCDFWINDNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON:
					if(model->GetModelMode() < ADVANCEDMODE) return FALSE; // shouldn't happen
					else return TRUE;
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
	
	return TWindMover::FunctionEnabled(item, buttonID);
}

OSErr GridWindMover::SettingsItem(ListItem item)
{
	//return NetCDFWindSettingsDialog(this, this -> moverMap,false,mapWindow);
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = ListClick(item,inBullet,doubleClick);
	return 0;
}

OSErr GridWindMover::DeleteItem(ListItem item)
{
	if (item.index == I_NETCDFWINDNAME)
		return moverMap -> DropMover(this);
	
	return 0;
}

OSErr GridWindMover::CheckAndPassOnMessage(TModelMessage *message)
{	
	
	/*char ourName[kMaxNameLen];
	 
	 // see if the message is of concern to us
	 
	 this->GetClassName(ourName);
	 if(message->IsMessage(M_SETFIELD,ourName))
	 {
	 double val;
	 char str[256];
	 OSErr err = 0;
	 ////////////////
	 err = message->GetParameterAsDouble("uncertaintySpeedScale",&val);
	 if(!err) this->fSpeedScale = val; 
	 ////////////////
	 err = message->GetParameterAsDouble("uncertaintyAngleScale",&val);
	 if(!err) this->fAngleScale = val; 
	 ////////////////
	 model->NewDirtNotification();// tell model about dirt
	 }*/
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TWindMover::CheckAndPassOnMessage(message);
}


OSErr GridWindMover::InitMover(TimeGridVel *grid)
{	
	OSErr	err = noErr;
	timeGrid = grid;
	//err = TWindMover::InitMover ();
	return err;
}

#define GridWindMoverREADWRITEVERSION 1 //JLM	5/3/10

OSErr GridWindMover::Write(BFPB *bfpb)
{
	long i, version = GridWindMoverREADWRITEVERSION;
	ClassID id = GetClassID ();
	long numTimes = timeGrid->GetNumTimesInFile();
	Seconds time;
	OSErr err = 0;
	
	if (err = TWindMover::Write(bfpb)) return err;
	
	StartReadWriteSequence("GridWindMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	////
	id = timeGrid -> GetClassID ();
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = timeGrid -> Write (bfpb)) goto done;
	
	//if (err = WriteMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	//if (err = WriteMacValue(bfpb, fFileName, kPtCurUserNameLen)) return err;
	
	if (err = WriteMacValue(bfpb, bShowGrid)) return err;
	if (err = WriteMacValue(bfpb, bShowArrows)) return err;
	if (err = WriteMacValue(bfpb, fUserUnits)) return err;
	if (err = WriteMacValue(bfpb, fArrowScale)) return err;
	if (err = WriteMacValue(bfpb, fWindScale)) return err;
	////////////////
	
	
done:
	if(err)
		TechError("GridWindMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr GridWindMover::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[256], newFileName[64];
	long i, version, numTimes, numPoints;
	ClassID id;
	float val;
	Seconds time;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TWindMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("GridWindMover::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("GridWindMover::Read()", "id != TYPE_GRIDWINDMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > GridWindMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	
	// read the type of grid used for the GridWind mover
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
		case TYPE_TIMEGRIDWINDRECT: timeGrid = new TimeGridVelRect; break;
			//case TYPE_TIMEGRIDVEL: timeGrid = new TimeGridVel; break;
		case TYPE_TIMEGRIDWINDCURV: timeGrid = new TimeGridVelCurv; break;
		default: printError("Unrecognized Grid type in GridWindMover::Read()."); return -1;
	}
	
	//
	if (err = ReadMacValue(bfpb, &bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &bShowArrows)) return err;
	if (err = ReadMacValue(bfpb, &fUserUnits)) return err;
	if (err = ReadMacValue(bfpb, &fArrowScale)) return err;

	if (err = ReadMacValue(bfpb, &fWindScale)) return err;
	
	/////////////////
	
done:
	if(err)
	{
		TechError("GridWindMover::Read(char* path)", " ", 0); 
	}
	return err;
}


Boolean GridWindMover::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, arrowDepth = 0.;
	long index;
	LongPoint indices;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!bShowArrows && !bShowGrid) return 0;
	err = timeGrid -> SetInterval(errmsg, model->GetModelTime()); 
	
	if(err) return false;
	
	if (err = timeGrid->VelocityStrAtPoint(wp, diagnosticStr, arrowDepth)) return err;

	/*if(dynamic_cast<GridWindMover *>(this)->GetNumTimesInFile()>1)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) 
		{
			if ((time > startTime || time < startTime) && fAllowExtrapolationOfWinds)
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
			if(dynamic_cast<GridWindMover *>(this)->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				velocity.u = this->GetStartUVelocity(index);
				velocity.v = this->GetStartVVelocity(index);
			}
			else // time varying current
			{
				velocity.u = timeAlpha*this->GetStartUVelocity(index) + (1-timeAlpha)*this->GetEndUVelocity(index);
				velocity.v = timeAlpha*this->GetStartVVelocity(index) + (1-timeAlpha)*this->GetEndVVelocity(index);
			}
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fWindScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
			this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
	*/
	return true;
}

Boolean GridWindMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant wind, we can say "no"
	if(timeGrid->GetNumTimesInFile()==1 && !(timeGrid->GetNumFiles()>1)) depends = false;
	return depends;
}

void GridWindMover::Draw(Rect r, WorldRect view) 
{	// Use this for regular grid
	timeGrid->Draw(r,view,fWindScale,fArrowScale,0,bShowArrows,bShowGrid,fColor);
}




static PopInfoRec GridWindMoverPopTable[] = {
	{ M18b, nil, M18bTIMEZONEPOPUP, 0, pTIMEZONES, 0, 1, FALSE, nil },
	{ M18b, nil, M18ANGLEUNITSPOPUP, 0, pANGLEUNITS, 0, 1, FALSE, nil }
};

static GridWindMover *sharedWMover;

void ShowGridWindMoverDialogItems(DialogPtr dialog)
{
	Boolean bShowGMTItems = true;
	short timeZone = GetPopSelection(dialog, M18bTIMEZONEPOPUP);
	if (timeZone == 1) bShowGMTItems = false;
	
	ShowHideDialogItem(dialog, M18bTIMESHIFTLABEL, bShowGMTItems); 
	ShowHideDialogItem(dialog, M18bTIMESHIFT, bShowGMTItems); 
	ShowHideDialogItem(dialog, M18bGMTOFFSETS, bShowGMTItems); 
}

short GridWindClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{	
	long menuID_menuItem;
	switch (itemNum) {
		case M18OK:
		{	
			short timeZone = GetPopSelection(dialog, M18bTIMEZONEPOPUP);
			Seconds timeShift = sharedWMover->timeGrid->fTimeShift;
			long timeShiftInHrs;
			
			short angleUnits = GetPopSelection(dialog, M18ANGLEUNITSPOPUP);
			
			timeShiftInHrs = EditText2Long(dialog, M18bTIMESHIFT);
			if (timeShiftInHrs < -12 || timeShiftInHrs > 14)	// what should limits be?
			{
				printError("Time offsets must be in the range -12 : 14");
				MySelectDialogItemText(dialog, M18bTIMESHIFT,0,100);
				break;
			}
			
			//mygetitext(dialog, M18bFILENAME, sharedWMover->fFileName, kPtCurUserNameLen-1);
			mygetitext(dialog, M18bFILENAME, sharedWMover->timeGrid->fVar.userName, kPtCurUserNameLen-1);
			sharedWMover -> bActive = GetButton(dialog, M18ACTIVE);
			sharedWMover->bShowArrows = GetButton(dialog, M18bSHOWARROWS);
			sharedWMover->fArrowScale = EditText2Float(dialog, M18bARROWSCALE);
			
			sharedWMover -> fAngleScale = EditText2Float(dialog,M18ANGLESCALE);
			if (angleUnits==2) sharedWMover->fAngleScale *= PI/180.;		
			sharedWMover -> fSpeedScale = EditText2Float(dialog,M18SPEEDSCALE);
			sharedWMover -> fUncertainStartTime = (long) round(EditText2Float(dialog,M18UNCERTAINSTARTTIME)*3600);
			
			sharedWMover -> fDuration = EditText2Float(dialog, M18DURATION) * 3600;
			
			if (timeZone>1) sharedWMover->timeGrid->fTimeShift =(long)( EditText2Float(dialog, M18bTIMESHIFT)*3600);
			else sharedWMover->timeGrid->fTimeShift = 0;	// file is in local time
			
			//if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
			//if (timeShift != sharedWMover->fTimeShift || sharedWMover->GetTimeValue(0) != model->GetStartTime())
			// code goes here, if decide to use this check GetTimeValue is using new fTimeShift...
			//{
			//Seconds timeValZero = sharedWMover->GetTimeValue(0), startTime = model->GetStartTime();
			if (timeShift != sharedWMover->timeGrid->fTimeShift && sharedWMover->timeGrid->GetTimeValue(0) != model->GetStartTime())
			{
				model->SetStartTime(model->GetStartTime() + (sharedWMover->timeGrid->fTimeShift-timeShift));
				//model->SetStartTime(sharedWMover->GetTimeValue(0));
				//sNetCDFDialogMover->SetInterval(errmsg);
				model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
			}
			//}
			sharedWMover->timeGrid->fAllowExtrapolationInTime = GetButton(dialog, M18bEXTRAPOLATECHECKBOX);
			// code goes here, check interval?
			return M18OK;
		}
		case M18CANCEL: return M18CANCEL;
			
		case M18ACTIVE:
		case M18bSHOWARROWS:
		case M18bEXTRAPOLATECHECKBOX:
			ToggleButton(dialog, itemNum);
			break;
			
		case M18bARROWSCALE:
		case M18DURATION:
		case M18UNCERTAINSTARTTIME:
		case M18ANGLESCALE:
		case M18SPEEDSCALE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
			
		case M18ANGLEUNITSPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
			
		case M18bTIMESHIFT:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);	// decide whether to allow half hours
			break;
			
		case M18bTIMEZONEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowGridWindMoverDialogItems(dialog);
			if (GetPopSelection(dialog, M18bTIMEZONEPOPUP) == 2) MySelectDialogItemText(dialog, M18bTIMESHIFT, 0, 100);
			break;
			
		case M18bGMTOFFSETS:
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
	}
	
	return noErr;
}

OSErr GridWindInit(DialogPtr dialog, VOIDPTR data)
{
	RegisterPopTable (GridWindMoverPopTable, sizeof (GridWindMoverPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog(M18b, dialog);
	
	SetDialogItemHandle(dialog, M18HILITEDEFAULT, (Handle)FrameDefault);
	
	//mysetitext(dialog, M18bFILENAME, sharedWMover->fFileName); // use short file name for now
	mysetitext(dialog, M18bFILENAME, sharedWMover->timeGrid->fVar.userName); // use short file name for now
	SetButton(dialog, M18ACTIVE, sharedWMover->bActive);
	
	SetPopSelection (dialog, M18ANGLEUNITSPOPUP, 1);
	
	if (sharedWMover->timeGrid->fTimeShift == 0) SetPopSelection (dialog, M18bTIMEZONEPOPUP, 1);
	else SetPopSelection (dialog, M18bTIMEZONEPOPUP, 2);
	//Long2EditText(dialog, M33TIMESHIFT, (long) (-1.*sNetCDFDialogMover->fTimeShift/3600.));
	Float2EditText(dialog, M18bTIMESHIFT, (float)(sharedWMover->timeGrid->fTimeShift)/3600.,1);
	
	SetButton(dialog, M18bSHOWARROWS, sharedWMover->bShowArrows);
	Float2EditText(dialog, M18bARROWSCALE, sharedWMover->fArrowScale, 6);
	
	Float2EditText(dialog, M18SPEEDSCALE, sharedWMover->fSpeedScale, 4);
	Float2EditText(dialog, M18ANGLESCALE, sharedWMover->fAngleScale, 4);
	
	Float2EditText(dialog, M18DURATION, sharedWMover->fDuration / 3600.0, 2);
	Float2EditText(dialog, M18UNCERTAINSTARTTIME, sharedWMover->fUncertainStartTime / 3600.0, 2);
	
	MySelectDialogItemText(dialog, M18SPEEDSCALE, 0, 255);
	//ShowHideDialogItem(dialog,M18ANGLEUNITSPOPUP,false);	// for now don't allow the units option here
	
	//SetButton(dialog, M18bEXTRAPOLATECHECKBOX, sharedWMover->fAllowExtrapolationInTime);
	SetButton(dialog, M18bEXTRAPOLATECHECKBOX, sharedWMover->timeGrid->fAllowExtrapolationInTime);
	
	ShowGridWindMoverDialogItems(dialog);
	//if (sharedWMover->fTimeShift == 0) MySelectDialogItemText(dialog, M33ALONG, 0, 100);
	//else MySelectDialogItemText(dialog, M18bTIMESHIFT, 0, 100);
	
	if (sharedWMover->timeGrid->fTimeShift != 0) MySelectDialogItemText(dialog, M18bTIMESHIFT, 0, 100);
	
	//SetDialogItemHandle(dialog,M18SETTINGSFRAME,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,M18UNCERTAINFRAME,(Handle)FrameEmbossed);
	
	return 0;
}

OSErr GridWindSettingsDialog(GridWindMover *mover, TMap *owner,Boolean bAddMover,WindowPtr parentWindow)
{ // Note: returns USERCANCEL when user cancels
	OSErr err = noErr;
	short item;
	
	if(!owner && bAddMover) {printError("Programmer error"); return -1;}
	
	sharedWMover = mover;			// existing mover is being edited
	
	if(parentWindow == 0) parentWindow = mapWindow; // JLM 6/2/99
	item = MyModalDialog(1825, parentWindow, 0, GridWindInit, GridWindClick);
	if (item == M18OK)
	{
		if (bAddMover)
		{
			err = owner -> AddMover (sharedWMover, 0);
		}
		model->NewDirtNotification();
	}
	if(item == M18CANCEL)
	{
		err = USERCANCEL;
	}
	
	return err;
}
