
#include "CROSS.h"
#include "OUtils.h"
#include "TShioTimeValue.h"

#ifdef MAC
#pragma segment SHIO
#endif


/////////////////////////////////////////////////
 
Boolean IsShioFile(char* path)
{
	// the first line of the file needs to be "[StationInfo]"
	Boolean	bIsOurFile = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [256];
	char	firstPartOfFile [256];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(256,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 256);
		RemoveTrailingWhiteSpace(strLine);
		if (!strcmpnocase(strLine,"[StationInfo]"))
			bIsOurFile = true;
	}
	return bIsOurFile;
}

///////////////////////////////////////////////////////////////////////////
void TShioTimeValue::InitInstanceVariables(void)
{
	this->fUserUnits = kKnots; // ??? JLM code goes here

	// initialize any fields we have
	this->fStationName[0] = 0;
	this->fStationType = 0;
	this->fLatitude = 0;
	this->fLongitude = 0;
	//HEIGHTOFFSET fHeightOffset;
	memset(&this->fConstituent,0,sizeof(this->fConstituent));
	memset(&this->fHeightOffset,0,sizeof(this->fHeightOffset));
	memset(&this->fCurrentOffset,0,sizeof(this->fCurrentOffset));
	this->fHighLowValuesOpen = false;
	this->fEbbFloodValuesOpen = false;
	this->fEbbFloodDataHdl=0;
	this->fHighLowDataHdl=0;

}

TShioTimeValue::TShioTimeValue(TMover *theOwner,TimeValuePairH tvals) : TOSSMTimeValue(theOwner) 
{ 	// having this this function is inherited but meaningless
	this->ProgrammerError("TShioTimeValue constructor");
	this->InitInstanceVariables();
}

TShioTimeValue::TShioTimeValue(TMover *theOwner) : TOSSMTimeValue(theOwner) 
{ 
	this->InitInstanceVariables();
}

OSErr TShioTimeValue::MakeClone(TClassID **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	ClassID *clone;
	OSErr err = 0;
	Boolean weCreatedIt = false;
	if(!clonePtrPtr) return -1; 
	if(*clonePtrPtr == nil)
	{	// create and return a cloned object.
		*clonePtrPtr = new TShioTimeValue(this->owner);
		weCreatedIt = true;
		if(!*clonePtrPtr) { TechError("MakeClone()", "new TShioTimeValue()", 0); return memFullErr;}	
	}
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			TShioTimeValue * cloneP = dynamic_cast<TShioTimeValue *>(*clonePtrPtr);// typecast 
			err =  TOSSMTimeValue::MakeClone(clonePtrPtr);//  pass clone to base class
			if(!err) 
			{
				strcpy(cloneP->fStationName,this->fStationName);
				cloneP->fStationType = this->fStationType;
				cloneP->fLatitude = this->fLatitude;
				cloneP->fLongitude = this->fLongitude;
				cloneP->fHighLowValuesOpen = this->fHighLowValuesOpen;
				cloneP->fEbbFloodValuesOpen = this->fEbbFloodValuesOpen;
			
				if(this->fEbbFloodDataHdl)
				{
					cloneP->fEbbFloodDataHdl = this->fEbbFloodDataHdl;
					err = _HandToHand((Handle *)&cloneP->fEbbFloodDataHdl);
					if(err) 
					{
						cloneP->fEbbFloodDataHdl = nil;
						goto done;
					}
				}
				if(this->fHighLowDataHdl)
				{
					cloneP->fHighLowDataHdl = this->fHighLowDataHdl;
					err = _HandToHand((Handle *)&cloneP->fHighLowDataHdl);
					if(err) 
					{
						cloneP->fHighLowDataHdl = nil;
						goto done;
					}
				}
				
				cloneP->fConstituent.DatumControls = this->fConstituent.DatumControls;

				if(this->fConstituent.H)
				{
					cloneP->fConstituent.H = this->fConstituent.H;
					err = _HandToHand((Handle *)&cloneP->fConstituent.H);
					if(err) 
					{
						cloneP->fConstituent.H = nil;
						goto done;
					}
				}
				
				if(this->fConstituent.kPrime)
				{
					cloneP->fConstituent.kPrime = this->fConstituent.kPrime;
					err = _HandToHand((Handle *)&cloneP->fConstituent.kPrime);
					if(err) 
					{
						cloneP->fConstituent.kPrime = nil;
						goto done;
					}
				}

				cloneP->fHeightOffset = this->fHeightOffset;
				cloneP->fCurrentOffset = this->fCurrentOffset;
			}
		}
	}
done:
	if(err && *clonePtrPtr) 
	{
		(*clonePtrPtr)->Dispose();
		if(weCreatedIt)
		{
			delete *clonePtrPtr;
			*clonePtrPtr = nil;
		}
	}
	return err;
}


OSErr TShioTimeValue::BecomeClone(TClassID *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			TShioTimeValue * cloneP = dynamic_cast<TShioTimeValue *>(clone);// typecast
			
			this->Dispose(); // get rid of any memory we currently are using
			////////////////////
			// do the memory stuff first, in case it fails
			////////
			if(cloneP->fEbbFloodDataHdl)
			{
				this->fEbbFloodDataHdl = cloneP->fEbbFloodDataHdl;
				err = _HandToHand((Handle *)&this->fEbbFloodDataHdl);
				if(err) 
				{
					this->fEbbFloodDataHdl = nil;
					goto done;
				}
			}
			
			if(cloneP->fHighLowDataHdl)
			{
				this->fHighLowDataHdl = cloneP->fHighLowDataHdl;
				err = _HandToHand((Handle *)&this->fHighLowDataHdl);
				if(err) 
				{
					this->fHighLowDataHdl = nil;
					goto done;
				}
			}
			
			this->fConstituent.DatumControls = cloneP->fConstituent.DatumControls;

			if(cloneP->fConstituent.H)
			{
				this->fConstituent.H = cloneP->fConstituent.H;
				err = _HandToHand((Handle *)&this->fConstituent.H);
				if(err) 
				{
					this->fConstituent.H = nil;
					goto done;
				}
			}
			
			if(cloneP->fConstituent.kPrime)
			{
				this->fConstituent.kPrime = cloneP->fConstituent.kPrime;
				err = _HandToHand((Handle *)&this->fConstituent.kPrime);
				if(err) 
				{
					this->fConstituent.kPrime = nil;
					goto done;
				}
			}
			
			err =  TOSSMTimeValue::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;

			strcpy(this->fStationName,cloneP->fStationName);
			this->fStationType = cloneP->fStationType;
			this->fLatitude = cloneP->fLatitude;
			this->fLongitude = cloneP->fLongitude;
			this->fHighLowValuesOpen = cloneP->fHighLowValuesOpen;
			this->fEbbFloodValuesOpen = cloneP->fEbbFloodValuesOpen;
			
			this->fHeightOffset = cloneP->fHeightOffset;
			this->fCurrentOffset = cloneP->fCurrentOffset;

		}
	}
done:
	if(err) this->Dispose(); // don't leave ourselves in a weird state
	return err;
}

void TShioTimeValue::Dispose()
{
	if(fEbbFloodDataHdl)DisposeHandle((Handle)fEbbFloodDataHdl);
	if(fHighLowDataHdl)DisposeHandle((Handle)fHighLowDataHdl);
	if(fConstituent.H)DisposeHandle((Handle)fConstituent.H);
	if(fConstituent.kPrime)DisposeHandle((Handle)fConstituent.kPrime);
	TOSSMTimeValue::Dispose();
	this->InitInstanceVariables();
}


OSErr TShioTimeValue::InitTimeFunc ()
{
 	OSErr err = TOSSMTimeValue::InitTimeFunc();
 	if(!err)
	{ // init the stuff for this class
		this->InitInstanceVariables();
	}
	return err;
}


/////////////////////////////////////////////////
Boolean sDialogStandingWave;
float sDialogScaleFactor;
OSErr ShioHtsInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	
	SetButton (dialog, SHIOHTSSTANDINGWAVE, sDialogStandingWave);
	SetButton (dialog, SHIOHTSPROGRESSIVEWAVE, !sDialogStandingWave);
	Float2EditText (dialog, SHIOHTSSCALEFACTOR, sDialogScaleFactor, 0);

	ShowHideDialogItem(dialog, SHIOHTSSCALEFACTOR, false ); 
	ShowHideDialogItem(dialog, SHIOHTSSCALEFACTORLABEL, false); 

	return 0;
}


short ShioHtsClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	switch (itemNum) {
		case SHIOHTSCANCEL: return SHIOHTSCANCEL;

		case SHIOHTSOK:
			sDialogStandingWave = GetButton (dialog, SHIOHTSSTANDINGWAVE);
			sDialogScaleFactor = EditText2Float(dialog, SHIOHTSSCALEFACTOR);

			return itemNum;
			
		case SHIOHTSSTANDINGWAVE:
		//	ToggleButton(dialog, itemNum);
		//	ToggleButton(dialog, SHIOHTSPROGRESSIVEWAVE);
			//CheckNumberTextItem(dialog, itemNum, FALSE);
		//	break;
		case SHIOHTSPROGRESSIVEWAVE:
			SetButton(dialog, SHIOHTSSTANDINGWAVE, itemNum == SHIOHTSSTANDINGWAVE);
			SetButton(dialog, SHIOHTSPROGRESSIVEWAVE, itemNum == SHIOHTSPROGRESSIVEWAVE);
			//ToggleButton(dialog, itemNum);
		//	ToggleButton(dialog, SHIOHTSSTANDINGWAVE);
			//CheckNumberTextItem(dialog, itemNum, FALSE);
			break;

		case SHIOHTSSCALEFACTOR:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

	}

	return 0;
}

OSErr ShioHtsDialog(Boolean *standingWave,float *scaleFactor,WindowPtr parentWindow)
{
	short item;
	sDialogStandingWave = *standingWave;
	sDialogScaleFactor = *scaleFactor;
	item = MyModalDialog(SHIOHTSDLG, mapWindow, 0, ShioHtsInit, ShioHtsClick);
	if(item == SHIOHTSCANCEL) return USERCANCEL; 
	model->NewDirtNotification();	// is this necessary ?
	if(item == SHIOHTSOK) 
	{
		*standingWave = sDialogStandingWave;
		*scaleFactor = sDialogScaleFactor;
		return 0; 
	}
	else return -1;
}

/////////////////////////////////////////////////

/////////////////////////////////////////////////

long TShioTimeValue::I_SHIOHIGHLOWS(void)
{
	return 1; // always item #2
}

long  TShioTimeValue::I_SHIOEBBFLOODS(void)
{
	if (this->fStationType == 'H')
	{
		long i = 2; // basically it's item 3
		if (fHighLowValuesOpen) {
			i += this->GetNumHighLowValues();
		}
		return i;
	}
	else
		return 99999;
	
}

/////////////////////////////////////////////////
long TShioTimeValue::GetListLength() 
{//JLM
	long numInList = 0;
	numInList++; // always show the station name
	//numInList+= this->GetNumValues(); //  show all time values
	if (this->fStationType == 'C')
		numInList+= this->GetNumEbbFloodValues(); // show only max/mins
	else if (this->fStationType == 'H')
	{
		numInList+=2;	// high lows and ebb floods
		if (fHighLowValuesOpen)
			numInList+= this->GetNumHighLowValues();	
		if (fEbbFloodValuesOpen)
			numInList+= 2*(this->GetNumHighLowValues())-1;
	}
	else if (this->fStationType == 'P')
	{
		numInList+=1;	// high lows and ebb floods
		if (fHighLowValuesOpen)
			numInList+= this->GetNumHighLowValues();	
		//if (fEbbFloodValuesOpen)
			//numInList+= 2*(this->GetNumHighLowValues())-1;
	}
	return numInList;
}



ListItem TShioTimeValue::GetNthListItem(long n, short indent, short *style, char *text)
{//JLM
	ListItem item = { this, n, indent, 0 };
	text[0] = 0; 
	*style = normal;
	
	/////////////
	if(n == 0)
	{ 	// line 1 station name
		item.indent--;
		sprintf(text,"Station Name: %s",fStationName); 
		return item; 
	}
	n--;
	/////////
	// code goes here, possible check if num max/mins below some threshold and then show all values...
	/*if( 0 <= n && n < this->GetNumValues())
	{
		return TOSSMTimeValue::GetNthListItem(n, indent, style, text); // to show all time values
	}*/
	/////////
	if (this->fStationType == 'C')
	{
		if( 0 <= n && n < this->GetNumEbbFloodValues()) // show only max/mins
		{
			EbbFloodData ebbFloodData;
			DateTimeRec time;
			char *p,timeStr[32],valStr[32],typeStr[32],unitsStr[32]=" kts ";
	
			ebbFloodData = INDEXH(fEbbFloodDataHdl, n);
			switch(ebbFloodData.type)
			{
				case	MinBeforeFlood:
					strcpy(typeStr,"MinBFld ");
					break;
				case	MaxFlood:
					strcpy(typeStr,"MaxFld ");
					break;
				case 	MinBeforeEbb:
					strcpy(typeStr,"MinBEbb ");
					break;
				case	MaxEbb:
					strcpy(typeStr,"MaxEbb ");
					break;
			}
			StringWithoutTrailingZeros(valStr,ebbFloodData.speedInKnots,1);
			SecondsToDate (ebbFloodData.time, &time);
			//time.year = time.year %100;// year 2000 fix , JLM 1/25/99  (two digit year)
			sprintf (timeStr, "%2.2d:%2.2d %02hd/%02hd/%02hd", time.hour, time.minute, time.month, time.day, time.year);
			//Date2String(&time, timeStr);
			//if (p = strrchr(timeStr, ':')) p[0] = 0; // remove seconds
			sprintf(text, "%s%s%s%s",typeStr,valStr,unitsStr,timeStr);
			return item;
		}
	}
	/////////
	else if (this->fStationType == 'H')
	{
		if (n == 0) {
			item.bullet = fHighLowValuesOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Show High Lows");
			
			return item;
		}
		n--;
		if( 0 <= n && n < this->GetNumHighLowValues() && fHighLowValuesOpen) // show only high/lows
		{
			HighLowData highLowData;
			DateTimeRec time;
			char timeStr[32],valStr[32],typeStr[32],unitsStr[32]=" ft ";
	
			highLowData = INDEXH(fHighLowDataHdl, n);
			switch(highLowData.type)
			{
				case	LowTide:
					strcpy(typeStr,"Low Tide ");
					break;
				case	HighTide:
					strcpy(typeStr,"High Tide ");
					break;
				default:
					strcpy(typeStr,"Unknown ");
					break;
			}
			StringWithoutTrailingZeros(valStr,highLowData.height * fScaleFactor,1);
			SecondsToDate (highLowData.time, &time);
			sprintf (timeStr, "%2.2d:%2.2d %02hd/%02hd/%02hd", time.hour, time.minute, time.month, time.day, time.year);
			sprintf(text, "%s%s%s%s",typeStr,valStr,unitsStr,timeStr);
			return item;
		}

		if (n>=this->GetNumHighLowValues() && fHighLowValuesOpen) n-=this->GetNumHighLowValues();
		
		if (n == 0) {
			item.bullet = fEbbFloodValuesOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Show Ebb Floods");
			
			return item;
		}
		n--;

		if( (0 <= n && n < 2*(this->GetNumHighLowValues()) - 1 && fEbbFloodValuesOpen)) // show only max/mins, converted from high/lows
		{
			HighLowData startHighLowData,endHighLowData;
			DateTimeRec time;
			double maxMinDeriv;
			Seconds midTime,derivTime;
			long index = floor(n/2.);
			char timeStr[32],valStr[32],typeStr[32],/*unitsStr[32]=" ft/hr ",*/unitsStr[32]=" kts ";
	
			startHighLowData = INDEXH(fHighLowDataHdl, index);
			endHighLowData = INDEXH(fHighLowDataHdl, index+1);
	
			midTime = (endHighLowData.time - startHighLowData.time)/2 + startHighLowData.time;
	
			switch(startHighLowData.type)
			{
				case	LowTide:
					if (fmod(n,2.) == 0)	
					{
						strcpy(typeStr,"MinBFld ");
						derivTime = startHighLowData.time;
					}
					else	
					{
						strcpy(typeStr,"MaxFld ");
						derivTime = midTime;
					}
					break;
				case	HighTide:
					if (fmod(n,2.) == 0)	
					{
						strcpy(typeStr,"MinBEbb ");
						derivTime = startHighLowData.time;
					}
					else 
					{
						strcpy(typeStr,"MaxEbb ");
						derivTime = midTime;
					}
					break;
			}
			maxMinDeriv = GetDeriv(startHighLowData.time, startHighLowData.height,
				endHighLowData.time, endHighLowData.height, derivTime) * fScaleFactor / KNOTSTOMETERSPERSEC;

			StringWithoutTrailingZeros(valStr,maxMinDeriv,1);
			SecondsToDate(derivTime, &time);
			sprintf (timeStr, "%2.2d:%2.2d %02hd/%02hd/%02hd", time.hour, time.minute, time.month, time.day, time.year);
			sprintf(text, "%s%s%s%s",typeStr,valStr,unitsStr,timeStr);
			return item;
		}
	}

	else if (this->fStationType == 'P')
	{
		if (n == 0) {
			item.bullet = fHighLowValuesOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Show High Lows");
			
			return item;
		}
		n--;
		if( 0 <= n && n < this->GetNumHighLowValues() && fHighLowValuesOpen) // show only high/lows
		{
			HighLowData highLowData;
			DateTimeRec time;
			char timeStr[32],valStr[32],typeStr[32],unitsStr[32]=" ft ";
	
			highLowData = INDEXH(fHighLowDataHdl, n);
			switch(highLowData.type)
			{
				case	LowTide:
					strcpy(typeStr,"Low Tide ");
					break;
				case	HighTide:
					strcpy(typeStr,"High Tide ");
					break;
				default:
					strcpy(typeStr,"Unknown ");
					break;
			}
			StringWithoutTrailingZeros(valStr,highLowData.height * fScaleFactor,1);
			SecondsToDate (highLowData.time, &time);
			sprintf (timeStr, "%2.2d:%2.2d %02hd/%02hd/%02hd", time.hour, time.minute, time.month, time.day, time.year);
			sprintf(text, "%s%s%s%s",typeStr,valStr,unitsStr,timeStr);
			return item;
		}

	}

	item.owner = 0; // not our item
	return item;
}

Boolean TShioTimeValue::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	
	if (inBullet) {
		if (item.index == I_SHIOHIGHLOWS()) 
			{fHighLowValuesOpen = !fHighLowValuesOpen; return TRUE;}
		if (item.index == I_SHIOEBBFLOODS())
			{fEbbFloodValuesOpen = !fEbbFloodValuesOpen; return TRUE;}
		return TRUE;
	}

	if (doubleClick && !inBullet)
	{
		TCATSMover *theOwner = (TCATSMover*)this->owner;
		Boolean timeFileChanged = false;
		if(theOwner)
		{
			CATSSettingsDialog (theOwner, theOwner -> moverMap, &timeFileChanged);
		}
		return TRUE;
	}

	// do other click operations...
	
	return FALSE;
}

Boolean TShioTimeValue::FunctionEnabled(ListItem item, short buttonID)
{
	if (buttonID == SETTINGSBUTTON) return TRUE;
	return FALSE;
}

OSErr TShioTimeValue::CheckAndPassOnMessage(TModelMessage *message)
{	
	//char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	long messageCode = message->GetMessageCode();
	switch(messageCode)
	{
		case M_UPDATEVALUES:
			VelocityRec dummyValue;
			// new data, make sure we are displaying data for the current model range
			this->GetTimeValue(model->GetStartTime(),&dummyValue);
			this->GetTimeValue(model->GetEndTime(),&dummyValue);	// in case model duration was increased
			break;
	}
	
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TOSSMTimeValue::CheckAndPassOnMessage(message);
}


Boolean DaylightSavingTimeInEffect(DateTimeRec *dateStdTime)
{
	// Assume that the change from standard to daylight
	// savings time is on the first Sunday in April at 0200 and 
	// the switch back to Standard time is on the
	// last Sunday in October at 0200.             
	
	//return false;	// code goes here, outside US don't use daylight savings
	if (settings.daylightSavingsTimeFlag == DAYLIGHTSAVINGSOFF) return false;
	
	switch(dateStdTime->month)
	{
		case 1:
		case 2:
		case 3:
		case 11:
		case 12:
			return false;
		
		case 5:
		case 6:
		case 7:
		case 8:
		case 9:
			return true;

		case 4: // april
			if(dateStdTime->day > 7) return true; // past the first week
			if(dateStdTime->dayOfWeek == 1) 
			{	// first sunday
				if(dateStdTime->hour >= 2) return true;  // after 2AM
				else return false; // before 2AM
			}
			else
			{	// not Sunday
				short prevSundayDay = dateStdTime->day - dateStdTime->dayOfWeek + 1;
				if(prevSundayDay >= 1) return true; // previous Sunday was this month, so we are after the magic Sunday
				else return false;// previous Sunday was previous month, so we are before the magic Sunday
			}
		
		case 10://Oct
			if(dateStdTime->day < 25) return true; // before the last week
			if(dateStdTime->dayOfWeek == 1) 
			{	// last sunday
				if(dateStdTime->hour >= 2) return false;  // after 2AM
				else return true; // before 2AM
			}
			else
			{	// not Sunday
				short nextSundayDay = dateStdTime->day - dateStdTime->dayOfWeek + 8;
				if(nextSundayDay > 31) return false; // next Sunday is next month, so we are after the magic Sunday
				else return true;// next Sunday is this month, so we are before the magic Sunday
			}

	}
	return false;// shouldn't get here
}

/////////////////////////////////////////////////

#define TShioMoverREADWRITEVERSION 1 //JLM

OSErr TShioTimeValue::Write(BFPB *bfpb)
{
	long i, n = 0, version = TShioMoverREADWRITEVERSION;
	ClassID id = GetClassID ();
	TimeValuePair pair;
	OSErr err = 0;
	
	if (err = TOSSMTimeValue::Write(bfpb)) return err;
	
	StartReadWriteSequence("TShioTimeValue::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	/////////////////////////////////
	
	if (err = WriteMacValue(bfpb, fStationName, MAXSTATIONNAMELEN)) return err; // don't swap !! 
	if (err = WriteMacValue(bfpb,fStationType)) return err;  
	if (err = WriteMacValue(bfpb,fLatitude)) return err;
	if (err = WriteMacValue(bfpb,fLongitude)) return err;

	if (err = WriteMacValue(bfpb,fConstituent.DatumControls.datum)) return err;
	if (err = WriteMacValue(bfpb,fConstituent.DatumControls.FDir)) return err;
	if (err = WriteMacValue(bfpb,fConstituent.DatumControls.EDir)) return err;
	if (err = WriteMacValue(bfpb,fConstituent.DatumControls.L2Flag)) return err;
	if (err = WriteMacValue(bfpb,fConstituent.DatumControls.HFlag)) return err;
	if (err = WriteMacValue(bfpb,fConstituent.DatumControls.RotFlag)) return err;

	// write the number of elements in the handle, then the handle values
	if(fConstituent.H) n = _GetHandleSize((Handle)fConstituent.H)/sizeof(**fConstituent.H);
	else n = 0;
	if (err = WriteMacValue(bfpb,n)) return err;
	for(i = 0; i<n;i++)	{
		if (err = WriteMacValue(bfpb,INDEXH(fConstituent.H,i))) return err;
	}
	
	// write the number of elements in the handle, then the handle values
	if(fConstituent.kPrime) n = _GetHandleSize((Handle)fConstituent.kPrime)/sizeof(**fConstituent.kPrime);
	else n = 0;
	if (err = WriteMacValue(bfpb,n)) return err;
	for(i = 0; i<n;i++)	{
		if (err = WriteMacValue(bfpb,INDEXH(fConstituent.kPrime,i))) return err;
	}

	if (err = WriteMacValue(bfpb,fHeightOffset.HighTime.val)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.HighTime.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.LowTime.val)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.LowTime.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.HighHeight_Mult.val)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.HighHeight_Mult.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.HighHeight_Add.val)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.HighHeight_Add.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.LowHeight_Mult.val)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.LowHeight_Mult.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.LowHeight_Add.val)) return err;
	if (err = WriteMacValue(bfpb,fHeightOffset.LowHeight_Add.dataAvailFlag)) return err;

	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBefFloodTime.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBefFloodTime.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.FloodTime.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.FloodTime.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBefEbbTime.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBefEbbTime.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.EbbTime.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.EbbTime.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.FloodSpdRatio.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.FloodSpdRatio.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.EbbSpdRatio.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.EbbSpdRatio.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBFloodSpd.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBFloodSpd.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBFloodDir.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBFloodDir.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxFloodSpd.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxFloodSpd.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxFloodDir.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxFloodDir.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBEbbSpd.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBEbbSpd.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBEbbDir.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MinBEbbDir.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxEbbSpd.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxEbbSpd.dataAvailFlag)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxEbbDir.val)) return err;
	if (err = WriteMacValue(bfpb,fCurrentOffset.MaxEbbDir.dataAvailFlag)) return err;

/////////////////////////////////////////////////

	if (fStationType == 'C')	// values to show on list for tidal currents
	{
		long i,numEbbFloods = GetNumEbbFloodValues();
		if (err = WriteMacValue(bfpb,numEbbFloods)) return err;
		for (i=0;i<numEbbFloods;i++)
		{
			if (err = WriteMacValue(bfpb,INDEXH(fEbbFloodDataHdl,i).time)) return err;
			if (err = WriteMacValue(bfpb,INDEXH(fEbbFloodDataHdl,i).speedInKnots)) return err;
			if (err = WriteMacValue(bfpb,INDEXH(fEbbFloodDataHdl,i).type)) return err;
		}		
	}
	if (fStationType == 'H')	// values to show on list for tidal heights
	{
		long i,numHighLows = GetNumHighLowValues();
		if (err = WriteMacValue(bfpb,numHighLows)) return err;
		for (i=0;i<numHighLows;i++)
		{
			if (err = WriteMacValue(bfpb,INDEXH(fHighLowDataHdl,i).time)) return err;
			if (err = WriteMacValue(bfpb,INDEXH(fHighLowDataHdl,i).height)) return err;
			if (err = WriteMacValue(bfpb,INDEXH(fHighLowDataHdl,i).type)) return err;
		}		
		if (err = WriteMacValue(bfpb,fHighLowValuesOpen)) return err;
		if (err = WriteMacValue(bfpb,fEbbFloodValuesOpen)) return err;
	}
	if (fStationType == 'P')	// values to show on list for tidal heights
	{
		long i,numHighLows = GetNumHighLowValues();
		if (err = WriteMacValue(bfpb,numHighLows)) return err;
		for (i=0;i<numHighLows;i++)
		{
			if (err = WriteMacValue(bfpb,INDEXH(fHighLowDataHdl,i).time)) return err;
			if (err = WriteMacValue(bfpb,INDEXH(fHighLowDataHdl,i).height)) return err;
			if (err = WriteMacValue(bfpb,INDEXH(fHighLowDataHdl,i).type)) return err;
		}		
		if (err = WriteMacValue(bfpb,fHighLowValuesOpen)) return err;
		//if (err = WriteMacValue(bfpb,fEbbFloodValuesOpen)) return err;
	}
	
	
	return 0;
}

OSErr TShioTimeValue::Read(BFPB *bfpb)
{
	long i, n, version;
	ClassID id;
	TimeValuePair pair;
	float f;
	OSErr err = 0;
	
	if (err = TOSSMTimeValue::Read(bfpb)) return err;
	
	StartReadWriteSequence("TShioTimeValue::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("TShioTimeValue::Read()", "id != TYPE_SHIOMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != TShioMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	
	/////////////////////////////////
	
	if (err = ReadMacValue(bfpb, fStationName, MAXSTATIONNAMELEN)) return err; // don't swap !! 
	if (err = ReadMacValue(bfpb,&fStationType)) return err; 
	if (err = ReadMacValue(bfpb,&fLatitude)) return err;
	if (err = ReadMacValue(bfpb,&fLongitude)) return err;

	if (err = ReadMacValue(bfpb,&fConstituent.DatumControls.datum)) return err;
	if (err = ReadMacValue(bfpb,&fConstituent.DatumControls.FDir)) return err;
	if (err = ReadMacValue(bfpb,&fConstituent.DatumControls.EDir)) return err;
	if (err = ReadMacValue(bfpb,&fConstituent.DatumControls.L2Flag)) return err;
	if (err = ReadMacValue(bfpb,&fConstituent.DatumControls.HFlag)) return err;
	if (err = ReadMacValue(bfpb,&fConstituent.DatumControls.RotFlag)) return err;

	// read the number of elements in the handle, then the handle values
	if (err = ReadMacValue(bfpb,&n)) return err;
	if(n > 0) {
		if(sizeof(f) != sizeof(**fConstituent.H))  { printError("fConstituent.H size mismatch"); return -1; }
		fConstituent.H = (float**)_NewHandle(n*sizeof(**fConstituent.H));
		if(!fConstituent.H) {TechError("TLEList::Read()", "_NewHandle()", 0); return -1; }
		for(i = 0; i<n;i++)	{
			if (err = ReadMacValue(bfpb,&f)) return err;
			INDEXH(fConstituent.H,i) = f;
		}
	}
	
	// read the number of elements in the handle, then the handle values
	if (err = ReadMacValue(bfpb,&n)) return err;
	if(n > 0) {
		if(sizeof(f) != sizeof(**fConstituent.kPrime))  { printError("fConstituent.kPrime size mismatch"); return -1; }
		fConstituent.kPrime = (float**)_NewHandle(n*sizeof(**fConstituent.kPrime));
		if(!fConstituent.kPrime) {TechError("TLEList::Read()", "_NewHandle()", 0); return -1; }
		for(i = 0; i<n;i++)	{
			if (err = ReadMacValue(bfpb,&f)) return err;
			INDEXH(fConstituent.kPrime,i) = f;
		}
	}

	if (err = ReadMacValue(bfpb,&fHeightOffset.HighTime.val)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.HighTime.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.LowTime.val)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.LowTime.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.HighHeight_Mult.val)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.HighHeight_Mult.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.HighHeight_Add.val)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.HighHeight_Add.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.LowHeight_Mult.val)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.LowHeight_Mult.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.LowHeight_Add.val)) return err;
	if (err = ReadMacValue(bfpb,&fHeightOffset.LowHeight_Add.dataAvailFlag)) return err;

	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBefFloodTime.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBefFloodTime.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.FloodTime.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.FloodTime.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBefEbbTime.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBefEbbTime.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.EbbTime.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.EbbTime.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.FloodSpdRatio.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.FloodSpdRatio.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.EbbSpdRatio.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.EbbSpdRatio.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBFloodSpd.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBFloodSpd.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBFloodDir.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBFloodDir.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxFloodSpd.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxFloodSpd.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxFloodDir.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxFloodDir.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBEbbSpd.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBEbbSpd.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBEbbDir.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MinBEbbDir.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxEbbSpd.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxEbbSpd.dataAvailFlag)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxEbbDir.val)) return err;
	if (err = ReadMacValue(bfpb,&fCurrentOffset.MaxEbbDir.dataAvailFlag)) return err;

		//
/////////////////////////////////////////////////

	if (fStationType == 'C')	// values to show on list for tidal currents
	{
		long i,numEbbFloods;
		EbbFloodData ebbFloodData;	
		if (err = ReadMacValue(bfpb,&numEbbFloods)) return err;
		fEbbFloodDataHdl = (EbbFloodDataH)_NewHandleClear(sizeof(EbbFloodData)*numEbbFloods);
		if(!fEbbFloodDataHdl) {TechError("TShioTimeValue::Read()", "_NewHandleClear()", 0); err = memFullErr; return err;}
		for (i=0;i<numEbbFloods;i++)
		{
			if (err = ReadMacValue(bfpb,&(ebbFloodData.time))) return err;
			if (err = ReadMacValue(bfpb,&(ebbFloodData.speedInKnots))) return err;
			if (err = ReadMacValue(bfpb,&(ebbFloodData.type))) return err;
			INDEXH(fEbbFloodDataHdl,i) = ebbFloodData;
		}		
	}
	if (fStationType == 'H')	// values to show on list for tidal heights
	{
		long i,numHighLows;
		HighLowData highLowData;		
		if (err = ReadMacValue(bfpb,&numHighLows)) return err;
		fHighLowDataHdl = (HighLowDataH)_NewHandleClear(sizeof(HighLowData)*numHighLows);
		if(!fHighLowDataHdl) {TechError("TShioTimeValue::Read()", "_NewHandleClear()", 0); err = memFullErr; return err;}
		for (i=0;i<numHighLows;i++)
		{
			if (err = ReadMacValue(bfpb,&(highLowData.time))) return err;
			if (err = ReadMacValue(bfpb,&(highLowData.height))) return err;
			if (err = ReadMacValue(bfpb,&(highLowData.type))) return err;
			INDEXH(fHighLowDataHdl,i) = highLowData;
		}		
		if (err = ReadMacValue(bfpb,&fHighLowValuesOpen)) return err;
		if (err = ReadMacValue(bfpb,&fEbbFloodValuesOpen)) return err;
	}
	if (fStationType == 'P')	// values to show on list for tidal heights
	{
		long i,numHighLows;
		HighLowData highLowData;		
		if (err = ReadMacValue(bfpb,&numHighLows)) return err;
		fHighLowDataHdl = (HighLowDataH)_NewHandleClear(sizeof(HighLowData)*numHighLows);
		if(!fHighLowDataHdl) {TechError("TShioTimeValue::Read()", "_NewHandleClear()", 0); err = memFullErr; return err;}
		for (i=0;i<numHighLows;i++)
		{
			if (err = ReadMacValue(bfpb,&(highLowData.time))) return err;
			if (err = ReadMacValue(bfpb,&(highLowData.height))) return err;
			if (err = ReadMacValue(bfpb,&(highLowData.type))) return err;
			INDEXH(fHighLowDataHdl,i) = highLowData;
		}		
		if (err = ReadMacValue(bfpb,&fHighLowValuesOpen)) return err;
		//if (err = ReadMacValue(bfpb,&fEbbFloodValuesOpen)) return err;
	}
	
	return err;
}




/////////////////////////////////////////////////

#define kMAXNUMSAVEDYEARS 30
YEARDATAHDL gYearDataHdl1980Plus[kMAXNUMSAVEDYEARS];

// will need to read from text file instead
YEARDATAHDL GetYearData(short year)
{
	// IMPORTANT: The calling function should NOT dispose the handle it gets
	YEARDATAHDL		yrHdl=nil;
	short yearMinus1980 = year-1980;
	long i,n,resSize=0;
	 
	if(0<= yearMinus1980 && yearMinus1980 <kMAXNUMSAVEDYEARS)
	{
		if(gYearDataHdl1980Plus[yearMinus1980]) return gYearDataHdl1980Plus[yearMinus1980];
	}

#ifdef MAC
	Handle r = nil;
	r=GetResource('YEAR',(long)year);
#ifdef SWAP_BINARY	
	resSize = GetMaxResourceSize(r);
	if(resSize > 0 && r) 
	{
		yrHdl = (YEARDATAHDL)_NewHandle(resSize);
		if(yrHdl)
		{
			_HLock(r); // so it can't be purged !!!
			YEARDATAHDL rHdl = (YEARDATAHDL)_NewHandle(resSize);
			DetachResource(r);
			rHdl = (YEARDATAHDL) r;
			// copy and swap the bytes
			n = resSize/sizeof(YEARDATA);
			for(i = 0; i< n; i++)
			{
				YEARDATA yrd  = (YEARDATA)INDEXH(rHdl,i);
				SwapFloat(&yrd.XODE);
				SwapFloat(&yrd.VPU);
				INDEXH(yrHdl,i) = yrd;
			}
			// I don't think we free something gotten from a resource
		}
		ReleaseResource(r);// don't dispose of a resource handle !!!
		r = 0;
	}
#else
	if(r) 
	{
		DetachResource(r);
		yrHdl = (YEARDATAHDL) r;
	}
#endif
#else
	char numStr[32];
	HRSRC hResInfo =0;
	HGLOBAL r = 0;
	sprintf(numStr,"#%ld",year);
	hResInfo = FindResource(hInst,numStr,"YEAR");
	if(hResInfo) 
	{
		// copy the handle so we can be
		// just like the mac
		//
		//also we need to swap the bytes
		//
		// be careful r is a HGLOBAL, not one of our special fake handles
		resSize = SizeofResource(hInst,hResInfo);
		if(resSize > 0) r = LoadResource(hInst,hResInfo);
		if(resSize > 0 && r) 
		{
			yrHdl = (YEARDATAHDL)_NewHandle(resSize);
			if(yrHdl)
			{
				YEARDATAPTR rPtr = (YEARDATAPTR) LockResource(r);
				// copy and swap the bytes
				n = resSize/sizeof(YEARDATA);
				for(i = 0; i< n; i++)
				{
					YEARDATA yrd  = rPtr[i];
					SwapFloat(&yrd.XODE);
					SwapFloat(&yrd.VPU);
					INDEXH(yrHdl,i) = yrd;
				}
				// WIN32 applications do not have to unlock resources locked by LockResource
				// I don't think we free something gotten from a resource
			}
		}
	}
#endif

	if(yrHdl && 0<= yearMinus1980 && yearMinus1980 <kMAXNUMSAVEDYEARS)
	{
		gYearDataHdl1980Plus[yearMinus1980] = yrHdl;
	}
	
	return(yrHdl);
}

