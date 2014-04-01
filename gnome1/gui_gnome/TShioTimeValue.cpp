
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

bool IsShioFile(vector<string> &linesInFile)
{
	long line = 0;
	string value;
	
	// the first line of the file needs to be "[StationInfo]"
	if (ParseKeyedLine(linesInFile[line++], "[StationInfo]", value))
		return true;
	else
		return false;
}

/*Boolean IsShioFile(char* path)
{
	vector<string> linesInFile;
	
	if (ReadLinesInFile(path, linesInFile))
		return IsShioFile(linesInFile);
	else
		return false;
}*/

TShioTimeValue::TShioTimeValue(TMover *theOwner,TimeValuePairH tvals) : TOSSMTimeValue(theOwner)
{ 	// having this this function is inherited but meaningless
	this->ProgrammerError("TShioTimeValue constructor");
	this->InitInstanceVariables();
}

TShioTimeValue::TShioTimeValue(TMover *theOwner) : TOSSMTimeValue(theOwner)
{ 
	this->InitInstanceVariables();
}

OSErr TShioTimeValue::MakeClone(TShioTimeValue **clonePtrPtr)
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
			TOSSMTimeValue *tObj = dynamic_cast<TOSSMTimeValue *>(*clonePtrPtr);
			err =  TOSSMTimeValue::MakeClone(&tObj);//  pass clone to base class
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


OSErr TShioTimeValue::BecomeClone(TShioTimeValue *clone)
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

/*void TShioTimeValue::Dispose()
{
	if(fEbbFloodDataHdl)DisposeHandle((Handle)fEbbFloodDataHdl);
	if(fHighLowDataHdl)DisposeHandle((Handle)fHighLowDataHdl);
	if(fConstituent.H)DisposeHandle((Handle)fConstituent.H);
	if(fConstituent.kPrime)DisposeHandle((Handle)fConstituent.kPrime);
	TOSSMTimeValue::Dispose();
	this->InitInstanceVariables();
}*/



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
		TCATSMover *theOwner = dynamic_cast<TCATSMover*>(this->owner);
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
			this->GetTimeValue(model->GetModelTime(),&dummyValue);	// changed shio to compute only a few days, this caused a flash once run got past 3 days
			//this->GetTimeValue(model->GetStartTime(),&dummyValue);											// minus AH 07/10/2012
			//this->GetTimeValue(model->GetEndTime(),&dummyValue);	// in case model duration was increased		// minus AH 07/10/2012
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


OSErr TShioTimeValue::ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance)
{
	// code goes here, use unitsIfKnownInAdvance to tell if we're coming from a location file, 
	// if not and it's a heights file ask if progressive or standing wave (add new field or track as 'P')
	//#pragma unused(unitsIfKnownInAdvance)	
	char strLine[kMaxKeyedLineLength];
	long i,numValues;
	double value1, value2, magnitude, degrees;
	CHARH f = 0;
	DateTimeRec time;
	TimeValuePair pair;
	OSErr	err = noErr,scanErr;
	long lineNum = 0;
	char *p;
	long numScanned;
	double value;
	CONTROLVAR  DatumControls;
	
	if (err = OSSMTimeValue_c::InitTimeFunc()) return err;
	
	timeValues = 0;
	fileName[0] = 0;
	
	if (!path) return -1;
	
	strcpy(strLine, path);
	SplitPathFile(strLine, this->fileName);
	
	err = ReadFileContents(TERMINATED, 0, 0, path, 0, 0, &f);
	if(err)	{ TechError("TShioTimeValue::ReadTimeValues()", "ReadFileContents()", 0); return -1; }
	
	lineNum = 0;
	// first line
	if(!(p = GetKeyedLine(f, "[StationInfo]",lineNum++,strLine)))  goto readError;
	// 2nd line
	if(!(p = GetKeyedLine(f, "Type=",lineNum++,strLine)))  goto readError;
	switch(p[0])
	{
			//case 'c': case 'C': this->fStationType = 'C'; break;
			//case 'h': case 'H': this->fStationType = 'H'; break;
			//case 'p': case 'P': this->fStationType = 'P';	// for now assume progressive waves selected in file, maybe change to user input
		case 'c': case 'C': 
			this->fStationType = 'C'; break;
		case 'h': case 'H': 
			this->fStationType = 'H'; 
		{
			// if not a location file, Ask user if this is a progressive or standing wave
			// also ask for scale factor here
			/*if (unitsIfKnownInAdvance==kFudgeFlag)
			 {
			 short buttonSelected;
			 buttonSelected  = MULTICHOICEALERT(1690,"The file you selected is a shio heights file. Are you sure you want to treat it as a progressive wave?",TRUE);
			 switch(buttonSelected){
			 case 1:// continue, treat as progressive wave
			 this->fStationType = 'P';
			 break;  
			 case 3: // cancel
			 //return 0;// leave as standing wave?
			 return -1;
			 break;
			 }
			 //printNote("The shio heights file will be treated as a progressive wave file");
			 //this->fStationType = 'P';
			 }*/
			if (unitsIfKnownInAdvance!=-2)	// not a location file
			{
				Boolean bStandingWave = true;
				float scaleFactor = fScaleFactor;
				err = ShioHtsDialog(&bStandingWave,&scaleFactor,mapWindow);
				if (!err)
				{
					if (!bStandingWave) this->fStationType = 'P';
					//this->fScaleFactor = scaleFactor;
				}
			}
		}
			break;
		case 'p': case 'P': 
			this->fStationType = 'P';	// for now assume progressive waves selected in file, maybe change to user input
			
			//printError("You have selected a SHIO heights file.  Only SHIO current files can be used in GNOME.");
			//return -1;
			break;	// Allow heights files to be read in 9/18/00
		default:	goto readError; 	
	}
	// 3nd line
	if(!(p = GetKeyedLine(f, "Name=",lineNum++,strLine)))  goto readError;
	strncpy(this->fStationName,p,MAXSTATIONNAMELEN);
	this->fStationName[MAXSTATIONNAMELEN-1] = 0;
	// 
	if(err = this->GetKeyedValue(f,"Latitude=",lineNum++,strLine,&this->fLatitude))  goto readError;
	if(err = this->GetKeyedValue(f,"Longitude=",lineNum++,strLine,&this->fLongitude))  goto readError;
	//
	if(!(p = GetKeyedLine(f, "[Constituents]",lineNum++,strLine)))  goto readError;
	// code goes here in version 1.2.7 these lines won't be required for height files, but should still allow old format
	//if(err = this->GetKeyedValue(f,"DatumControls.datum=",lineNum++,strLine,&this->fConstituent.DatumControls.datum))  goto readError;
	if(err = this->GetKeyedValue(f,"DatumControls.datum=",lineNum++,strLine,&this->fConstituent.DatumControls.datum))  
	{
		if(this->fStationType=='h' || this->fStationType=='H')
		{
			lineNum--;	// possibly new Shio output which eliminated the unused datumcontrols for height files
			goto skipDatumControls;
		}
		else
		{
			goto readError;
		}
	}
	if(err = this->GetKeyedValue(f,"DatumControls.FDir=",lineNum++,strLine,&this->fConstituent.DatumControls.FDir))  goto readError;
	if(err = this->GetKeyedValue(f,"DatumControls.EDir=",lineNum++,strLine,&this->fConstituent.DatumControls.EDir))  goto readError;
	if(err = this->GetKeyedValue(f,"DatumControls.L2Flag=",lineNum++,strLine,&this->fConstituent.DatumControls.L2Flag))  goto readError;
	if(err = this->GetKeyedValue(f,"DatumControls.HFlag=",lineNum++,strLine,&this->fConstituent.DatumControls.HFlag))  goto readError;
	if(err = this->GetKeyedValue(f,"DatumControls.RotFlag=",lineNum++,strLine,&this->fConstituent.DatumControls.RotFlag))  goto readError;
	
skipDatumControls:
	if(err = this->GetKeyedValue(f,"H=",lineNum++,strLine,&this->fConstituent.H))  goto readError;
	if(err = this->GetKeyedValue(f,"kPrime=",lineNum++,strLine,&this->fConstituent.kPrime))  goto readError;
	
	if(!(p = GetKeyedLine(f, "[Offset]",lineNum++,strLine)))  goto readError;
	
	switch(this->fStationType)
	{
		case 'c': case 'C': 
			if(err = this->GetKeyedValue(f,"MinBefFloodTime=",lineNum++,strLine,&this->fCurrentOffset.MinBefFloodTime))  goto readError;
			if(err = this->GetKeyedValue(f,"FloodTime=",lineNum++,strLine,&this->fCurrentOffset.FloodTime))  goto readError;
			if(err = this->GetKeyedValue(f,"MinBefEbbTime=",lineNum++,strLine,&this->fCurrentOffset.MinBefEbbTime))  goto readError;
			if(err = this->GetKeyedValue(f,"EbbTime=",lineNum++,strLine,&this->fCurrentOffset.EbbTime))  goto readError;
			if(err = this->GetKeyedValue(f,"FloodSpdRatio=",lineNum++,strLine,&this->fCurrentOffset.FloodSpdRatio))  goto readError;
			if(err = this->GetKeyedValue(f,"EbbSpdRatio=",lineNum++,strLine,&this->fCurrentOffset.EbbSpdRatio))  goto readError;
			if(err = this->GetKeyedValue(f,"MinBFloodSpd=",lineNum++,strLine,&this->fCurrentOffset.MinBFloodSpd))  goto readError;
			if(err = this->GetKeyedValue(f,"MinBFloodDir=",lineNum++,strLine,&this->fCurrentOffset.MinBFloodDir))  goto readError;
			if(err = this->GetKeyedValue(f,"MaxFloodSpd=",lineNum++,strLine,&this->fCurrentOffset.MaxFloodSpd))  goto readError;
			if(err = this->GetKeyedValue(f,"MaxFloodDir=",lineNum++,strLine,&this->fCurrentOffset.MaxFloodDir))  goto readError;
			if(err = this->GetKeyedValue(f,"MinBEbbSpd=",lineNum++,strLine,&this->fCurrentOffset.MinBEbbSpd))  goto readError;
			if(err = this->GetKeyedValue(f,"MinBEbbDir=",lineNum++,strLine,&this->fCurrentOffset.MinBEbbDir))  goto readError;
			if(err = this->GetKeyedValue(f,"MaxEbbSpd=",lineNum++,strLine,&this->fCurrentOffset.MaxEbbSpd))  goto readError;
			if(err = this->GetKeyedValue(f,"MaxEbbDir=",lineNum++,strLine,&this->fCurrentOffset.MaxEbbDir))  goto readError;
			SetFileType(SHIOCURRENTSFILE);
			break;
		case 'h': case 'H': 
			if(err = this->GetKeyedValue(f,"HighTime=",lineNum++,strLine,&this->fHeightOffset.HighTime))  goto readError;
			if(err = this->GetKeyedValue(f,"LowTime=",lineNum++,strLine,&this->fHeightOffset.LowTime))  goto readError;
			if(err = this->GetKeyedValue(f,"HighHeight_Mult=",lineNum++,strLine,&this->fHeightOffset.HighHeight_Mult))  goto readError;
			if(err = this->GetKeyedValue(f,"HighHeight_Add=",lineNum++,strLine,&this->fHeightOffset.HighHeight_Add))  goto readError;
			if(err = this->GetKeyedValue(f,"LowHeight_Mult=",lineNum++,strLine,&this->fHeightOffset.LowHeight_Mult))  goto readError;
			if(err = this->GetKeyedValue(f,"LowHeight_Add=",lineNum++,strLine,&this->fHeightOffset.LowHeight_Add))  goto readError;
			SetFileType(SHIOHEIGHTSFILE);
			break;
		case 'p': case 'P': 
			if(err = this->GetKeyedValue(f,"HighTime=",lineNum++,strLine,&this->fHeightOffset.HighTime))  goto readError;
			if(err = this->GetKeyedValue(f,"LowTime=",lineNum++,strLine,&this->fHeightOffset.LowTime))  goto readError;
			if(err = this->GetKeyedValue(f,"HighHeight_Mult=",lineNum++,strLine,&this->fHeightOffset.HighHeight_Mult))  goto readError;
			if(err = this->GetKeyedValue(f,"HighHeight_Add=",lineNum++,strLine,&this->fHeightOffset.HighHeight_Add))  goto readError;
			if(err = this->GetKeyedValue(f,"LowHeight_Mult=",lineNum++,strLine,&this->fHeightOffset.LowHeight_Mult))  goto readError;
			if(err = this->GetKeyedValue(f,"LowHeight_Add=",lineNum++,strLine,&this->fHeightOffset.LowHeight_Add))  goto readError;
			SetFileType(PROGRESSIVETIDEFILE);
			break;
	}
	
	
	if(f) DisposeHandle((Handle)f); f = nil;
	return 0;
	
readError:
	if(f) DisposeHandle((Handle)f); f = nil;
	sprintf(strLine,"Error reading SHIO time file %s on line %ld",this->fileName,lineNum);
	printError(strLine);
	this->Dispose();
	return -1;
	
Error:
	if(f) DisposeHandle((Handle)f); f = nil;
	return -1;
	
}

/////////////////////////////////////////////////


//#define kMAXNUMSAVEDYEARS 30
YEARDATAHDL gYearDataHdl1990Plus[kMAXNUMSAVEDYEARS];

// will need to read from text file instead
YEARDATAHDL GetYearData(short year)
{
	// IMPORTANT: The calling function should NOT dispose the handle it gets
	YEARDATAHDL		yrHdl=nil;
	short yearMinus1990 = year-1990;
	long i,n,resSize=0;
	
	if(0<= yearMinus1990 && yearMinus1990 <kMAXNUMSAVEDYEARS)
	{
		if(gYearDataHdl1990Plus[yearMinus1990]) return gYearDataHdl1990Plus[yearMinus1990];
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
	
	if(yrHdl && 0<= yearMinus1990 && yearMinus1990 <kMAXNUMSAVEDYEARS)
	{
		gYearDataHdl1990Plus[yearMinus1990] = yrHdl;
	}
	
	return(yrHdl);
}
/////////////////////////////////////////////////
/*YEARDATA2* gYearDataHdl1990Plus2[kMAXNUMSAVEDYEARS];

YEARDATA2* ReadYearData(short year, const char *path, char *errStr)

{
	// Get year data which are amplitude corrections XODE and epoc correcton VPU

	// NOTE: as per Andy & Mikes code, this func provides only a single year's
	// data: it does not handle requests spanning years. Both Andy and Mike would
	// just ask for the year at the start of the data request.

	// Each year has its own file of data, named "#2002" for year 2002, for example

	// NOTE: you must pass in the file path because it is platform specific:
	// the files live in the sub-directory "yeardata"
	// - on Mac the path is ":yeardata:#2002" and works off the app dir as current directory
	// - on Mac running in python in terminal, the path is "yeardata/#2002"

	// if errStr not empty then don't bother, something has already gone wrong

	YEARDATA2	*result = 0;
	FILE		*stream = 0;
	double		*xode = 0;
	double		*vpu = 0;
	double		data1, data2;
	char		filePathName[256];
	short		cnt, numPoints = 0, err = 0;

	short yearMinus1990 = year-1990;
	
	if(0<= yearMinus1990 && yearMinus1990 <kMAXNUMSAVEDYEARS)
	{
		if(gYearDataHdl1990Plus2[yearMinus1990]) return gYearDataHdl1990Plus2[yearMinus1990];
	}
	//if (errStr[0] != 0) return 0;
	errStr[0] = 0;

	// create the filename of the proper year data file
	sprintf(filePathName, "%s#%d", path, year);
		
	// It appears that there are 128 data values for XODE and VPU in each file
	// NOTE: we only collect the first 128 data points (BUG if any more ever get added)

	try
	{
		result = new YEARDATA2;
		xode = new double[128];
		vpu = new double[128];
	}
	catch (...)
	{
		err = -1;
		strcpy(errStr, "Memory error in ReadYearData");
	}

	if (!err) stream = fopen(filePathName, "r");

	if (stream)
	{
		for (cnt = 0; cnt < 128; cnt++)
		{
			if (fscanf(stream, "%lf %lf", &data1, &data2) == 2)	// assigned 2 values
			{
				numPoints++;
				xode[cnt] = data1;
				vpu[cnt] = data2;
			}
			else	// we are not getting data points, init to zero
			{
				xode[cnt] = 0.0;
				vpu[cnt] = 0.0;
			}
		}
		fclose(stream);
	}
	else
	{
		err = -1;
		sprintf(errStr, "Could not open file '%s'", filePathName);
	}

	if (err)

	{
		if (vpu) delete [] vpu;
		if (xode) delete [] xode;
		if (result) delete result;
		return 0;
	}	

	result->numElements = numPoints;
	result->XODE = xode;
	result->VPU = vpu;

	if(result && 0<= yearMinus1990 && yearMinus1990 <kMAXNUMSAVEDYEARS)
	{
		gYearDataHdl1990Plus2[yearMinus1990] = result;
	}
	
	return result;

}*/

