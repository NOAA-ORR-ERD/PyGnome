
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

char* GetKeyedLine(CHARH f, char*key, long lineNum, char *strLine)
{	// copies the next line into strLine
	// and returns ptr to first char after the key
	// returns NIL if key does not match
	char* p = 0;
	long keyLen = strlen(key);
	NthLineInTextOptimized (*f,lineNum, strLine, kMaxKeyedLineLength); 
	RemoveTrailingWhiteSpace(strLine);
	if (!strncmpnocase(strLine,key,keyLen)) 
		p = strLine+keyLen;
	return p;
}

OSErr TShioTimeValue::GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float *** val)
{
	#define kMaxNumVals 100
	#define kMaxStrChar 32
	float** h = (float**)_NewHandleClear(kMaxNumVals*sizeof(***val));
	char *p;
	OSErr scanErr = 0;
	double value;
	long i,numVals = 0;
	char str[kMaxStrChar];
	OSErr err = 0;
	
	if(!h) return -1;
	
	*val = nil;
	if(!(p = GetKeyedLine(f,key,lineNum,strLine)))  {err = -2; goto done;}
	for(;;) //forever
	{
		str[0] = 0;
		for(;*p == ' ' && *p == '\t';p++){} // move past leading white space
		if(*p == 0) goto done;
		for(i = 0;i < kMaxStrChar && *p != ' ' && *p != '\t' && *p ;str[i++] = (*p++)){} // copy to next white space or end of string
		if(i == kMaxStrChar) {err = -3; goto done;}
		str[i] = 0;
		p++;
		scanErr =  StringToDouble(str,&value);
		if(scanErr) return scanErr;
		(*h)[numVals++] = value;
		if(numVals >= kMaxNumVals) {err = -4; goto done;}
	}
	
done:
	if(numVals < 10) err = -5;// probably a bad line
	if(err && h) {DisposeHandle((Handle)h); h = 0;}
	else _SetHandleSize((Handle)h,numVals*sizeof(***val));
	*val = h;
	return err;
}


OSErr TShioTimeValue::GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,DATA * val)
{
	char *p,*p2;
	OSErr scanErr = 0;
	double value;
	if(!(p = GetKeyedLine(f,key,lineNum,strLine)))  return -1;
	// find the second part of the string
	for(p2 = p; TRUE ; p2++)
	{//advance to the first space
		if(*p2 == 0) return -1; //error, only one part to the string
		if(*p2 == ' ') 
		{
			*p2 = 0; // null terminate the first part
			p2++;
			break;
		}
	}
	scanErr =  StringToDouble(p,&value);
	if(scanErr) return scanErr;
	val->val = value;
	scanErr =  StringToDouble(p2,&value);
	if(scanErr) return scanErr;
	val->dataAvailFlag = round(value);
	return 0;
}

OSErr TShioTimeValue::GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,short * val)
{
	char *p;
	OSErr scanErr = 0;
	double value;
	if(!(p = GetKeyedLine(f,key,lineNum,strLine)))  return -1;
	scanErr =  StringToDouble(p,&value);
	if(scanErr) return scanErr;
	*val = round(value);
	return 0;
}


OSErr TShioTimeValue::GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float * val)
{
	char *p;
	OSErr scanErr = 0;
	double value;
	if(!(p = GetKeyedLine(f,key,lineNum,strLine)))  return -1;
	scanErr =  StringToDouble(p,&value);
	if(scanErr) return scanErr;
	*val = value;
	return 0;
}

OSErr TShioTimeValue::GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,double * val)
{
	char *p;
	OSErr scanErr = 0;
	double value;
	if(!(p = GetKeyedLine(f,key,lineNum,strLine)))  return -1;
	scanErr =  StringToDouble(p,&value);
	if(scanErr) return scanErr;
	*val = value;
	return 0;
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
	
	if (err = TOSSMTimeValue::InitTimeFunc()) return err;
	
	timeValues = 0;
	fileName[0] = 0;
	
	if (!path) return -1;
	
	strcpy(strLine, path);
	SplitPathFile(strLine, this->fileName);
	
	err = ReadFileContents(TERMINATED, 0, 0, path, 0, 0, &f);
	if(err)	{ TechError("TShioTimeValue::ReadTimeValues()", "ReadFileContents()", 0); return -1; }
	
	lineNum = 0;
	// first line
	if(!(p = GetKeyedLine(f,"[StationInfo]",lineNum++,strLine)))  goto readError;
	// 2nd line
	if(!(p = GetKeyedLine(f,"Type=",lineNum++,strLine)))  goto readError;
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
	if(!(p = GetKeyedLine(f,"Name=",lineNum++,strLine)))  goto readError;
	strncpy(this->fStationName,p,MAXSTATIONNAMELEN);
	this->fStationName[MAXSTATIONNAMELEN-1] = 0;
	// 
	if(err = this->GetKeyedValue(f,"Latitude=",lineNum++,strLine,&this->fLatitude))  goto readError;
	if(err = this->GetKeyedValue(f,"Longitude=",lineNum++,strLine,&this->fLongitude))  goto readError;
	//
	if(!(p = GetKeyedLine(f,"[Constituents]",lineNum++,strLine)))  goto readError;
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

	if(!(p = GetKeyedLine(f,"[Offset]",lineNum++,strLine)))  goto readError;

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

long	TShioTimeValue::GetNumEbbFloodValues()
{
 	long numEbbFloodValues = _GetHandleSize((Handle)fEbbFloodDataHdl)/sizeof(**fEbbFloodDataHdl);
	return numEbbFloodValues;
}

long	TShioTimeValue::GetNumHighLowValues()
{
 	long numHighLowValues = _GetHandleSize((Handle)fHighLowDataHdl)/sizeof(**fHighLowDataHdl);
	return numHighLowValues;
}

OSErr TShioTimeValue::GetTimeValue(Seconds forTime, VelocityRec *value)
{
	OSErr err = 0;
	Boolean needToCompute = true;
	Seconds modelStartTime = model->GetStartTime();
	Seconds modelEndTime = model->GetEndTime();
	DateTimeRec beginDate, endDate;
	Seconds beginSeconds;
	Boolean daylightSavings;
	YEARDATAHDL		YHdl = 0; 
	double *XODE=0, *VPU=0;
    //YEARDATA		*yearData = (YEARDATA *) NULL;
	long numConstituents;
	CONSTITUENT		*conArray = 0;
	
	long numValues = this->GetNumValues(), amtYearData, i;
	
	// check that to see if the value is in our already computed range
	if(numValues > 0)
	{
		if(INDEXH(this->timeValues, 0).time <= forTime 
			&& forTime <= INDEXH(this->timeValues, numValues-1).time)
		{ // the value is in the computed range
			if (this->fStationType == 'C')	// allow scale factor for 'P' case
				return TOSSMTimeValue::GetTimeValue(forTime,value);
			else if (this->fStationType == 'P')
				return GetProgressiveWaveValue(forTime,value);
			else if (this->fStationType == 'H')
				return GetConvertedHeightValue(forTime,value);
		}
		//this->fScaleFactor = 0;	//if we want to ask for a scale factor for each computed range...
	}

	// else we need to re-compute the values
	this->SetTimeValueHandle(nil);

	// calculate the values every hour for the interval containing the model run time
	//SecondsToDate(modelStartTime,&beginDate);
	SecondsToDate(modelStartTime-6*3600,&beginDate);	// for now to handle the new tidal current mover 1/27/04
	beginDate.hour = 0; // Shio expects this to be the start of the day
	beginDate.minute = 0;
	beginDate.second = 0;
	DateToSeconds(&beginDate, &beginSeconds);
	
	SecondsToDate(modelEndTime+24*3600,&endDate);// add one day so that we can truncate to start of the day
	endDate.hour = 0; // Shio expects this to be the start of the day
	endDate.minute = 0;
	endDate.second = 0;
	
	daylightSavings = DaylightSavingTimeInEffect(&beginDate);// code goes here, set the daylight flag
	YHdl = GetYearData(beginDate.year); 
	if(!YHdl)  { TechError("TShioTimeValue::GetTimeValue()", "GetYearData()", 0); return -1; }

	amtYearData = GetHandleSize((Handle)YHdl)/sizeof(**YHdl);
	try
	{
		XODE = new double[amtYearData];
		VPU = new double[amtYearData];
	}
	catch (...)
	{
		TechError("TShioTimeValue::GetTimeValue()", "new double()", 0); return -1;
	}
	for (i = 0; i<amtYearData; i++)
	{
		XODE[i] = (double) INDEXH(YHdl,i).XODE;	
		VPU[i] = (double) INDEXH(YHdl,i).VPU;
	}

	numConstituents = GetHandleSize((Handle)fConstituent.H)/sizeof(**fConstituent.H);
	conArray = new CONSTITUENT[numConstituents];
	for (i = 0; i<numConstituents; i++)
	{
			conArray[i].H = INDEXH(fConstituent.H,i);
			conArray[i].kPrime = INDEXH(fConstituent.kPrime,i);
			
			// NOTE: all these other fields in CONTROLVAR are undefined for height stations
			conArray[i].DatumControls.datum = fConstituent.DatumControls.datum;
			conArray[i].DatumControls.FDir = fConstituent.DatumControls.FDir;
			conArray[i].DatumControls.EDir = fConstituent.DatumControls.EDir;
			conArray[i].DatumControls.L2Flag = fConstituent.DatumControls.L2Flag;
			conArray[i].DatumControls.HFlag = fConstituent.DatumControls.HFlag;
			conArray[i].DatumControls.RotFlag = fConstituent.DatumControls.RotFlag;
	}

	if(this->fStationType == 'C')
	{
		// get the currents
		COMPCURRENTS *answers ;
		//memset(&answers,0,sizeof(answers));
		answers = new COMPCURRENTS;		
		
		answers->nPts = 0;
		answers->time = 0;
		answers->speed = 0;
		answers->u = 0;
		answers->v = 0;
		answers->uMinor = 0;
		answers->vMajor = 0;
		answers->speedKey = 0;
		answers->numEbbFloods = 0;
		answers->EbbFloodSpeeds = 0;
		answers->EbbFloodTimes = 0;
		answers->EbbFlood = 0;
		
		err = GetTideCurrent(&beginDate,&endDate,
						numConstituents,
						//&fConstituent,	
						conArray,
						&fCurrentOffset,		
						answers,		// Current-time struc with answers
						//YHdl,
						XODE,VPU,
						daylightSavings,
						fStationName);
		//model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
		if(!err)
		{
			model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
			long i,num10MinValues = answers->nPts,numCopied = 0;
			if(num10MinValues > 0 && answers->time && answers->speed)
			{
				// copy these values into a handle
				TimeValuePairH tvals = (TimeValuePairH)_NewHandle(num10MinValues* sizeof(TimeValuePair));
				if(!tvals) 
					{TechError("TShioTimeValue::GetTimeValue()", "GetYearData()", 0); err = memFullErr; goto done_currents;}
				else
				{
					TimeValuePair tvPair;
					for(i = 0; i < num10MinValues; i++)
					{
						if(answers->time[i].flag == outOfSequenceFlag) continue;// skip this value, code goes here
						if(answers->time[i].flag == 1) continue;// skip this value, 1 is don't plot flag - junk at beginning or end of data
						// note:timeHdl values are in hrs from start
						tvPair.time = beginSeconds + (long) (answers->time[i].val*3600); 
						tvPair.value.u = KNOTSTOMETERSPERSEC * answers->speed[i];// convert from knots to m/s
						tvPair.value.v = 0; // not used
						//INDEXH(tvals,i) = tvPair;	
						INDEXH(tvals,numCopied) = tvPair;	// if skip outOfSequence values don't want holes in the handle
						numCopied++;
					}
					_SetHandleSize((Handle)tvals, numCopied* sizeof(TimeValuePair));
					this->SetTimeValueHandle(tvals);
				}
			////// JLM  , try saving the highs and lows for displaying on the left hand side
				{
				
					short	numEbbFloods = answers->numEbbFloods;			// Number of ebb and flood occurrences.								
					double *EbbFloodSpeedsPtr = answers->EbbFloodSpeeds;	// double knots
					EXTFLAG *EbbFloodTimesPtr = answers->EbbFloodTimes;		// double hours, flag=0 means plot
					short			*EbbFloodPtr = answers->EbbFlood;			// 0 -> Min Before Flood.
															// 1 -> Max Flood.
															// 2 -> Min Before Ebb.
															// 3 -> Max Ebb.
					short numToShowUser;
					short i;
					double EbbFloodSpeed;
					EXTFLAG EbbFloodTime;
					short EbbFlood;
					
					
					/*short numEbbFloodSpeeds = 0;
					short numEbbFloodTimes = 0;
					short numEbbFlood = 0;
					double dBugEbbFloodSpeedArray[40];
					EXTFLAG dBugEbbFloodTimesArray[40];
					short dBugEbbFloodArray[40];
					
					// just double check the size of the handles is what we expect
					
					if(EbbFloodSpeedsHdl)
						numEbbFloodSpeeds = _GetHandleSize((Handle)EbbFloodSpeedsHdl)/sizeof(**EbbFloodSpeedsHdl);
					
					if(EbbFloodTimesHdl)
						numEbbFloodTimes = _GetHandleSize((Handle)EbbFloodTimesHdl)/sizeof(**EbbFloodTimesHdl);
					
					if(EbbFloodHdl)
						numEbbFlood = _GetHandleSize((Handle)EbbFloodHdl)/sizeof(**EbbFloodHdl);
					
					if(numEbbFlood == numEbbFloodSpeeds 
						&& numEbbFlood == numEbbFloodTimes
						)
					{
						for(i = 0; i < numEbbFlood && i < 40; i++)
						{
							dBugEbbFloodSpeedArray[i] = INDEXH(answers.EbbFloodSpeedsHdl,i);	// double knots
							dBugEbbFloodTimesArray[i]  = INDEXH(answers.EbbFloodTimesHdl,i);	// double hours, flag=0 means plot
							dBugEbbFloodArray[i] = INDEXH(answers.EbbFloodHdl,i);			// 0 -> Min Before Flood.
																	// 1 -> Max Flood.
																	// 2 -> Min Before Ebb.
																	// 3 -> Max Ebb.
						
						}
						dBugEbbFloodArray[39] = dBugEbbFloodArray[39]; // just a break point
					
					}*/
					
					
					/////////////////////////////////////////////////

					// count the number of values we wish to show to the user
					// (we show the user if the plot flag is set)
					numToShowUser = 0;
					if(EbbFloodSpeedsPtr && EbbFloodTimesPtr && EbbFloodPtr)
					{
						for(i = 0; i < numEbbFloods; i++)
						{
							EbbFloodTime = EbbFloodTimesPtr[i];
							if(EbbFloodTime.flag == 0)
								numToShowUser++;
						}
					}
					
					// now allocate a handle of this size to hold the values for the user
					
					if(fEbbFloodDataHdl) 
					{
						DisposeHandle((Handle)fEbbFloodDataHdl); 
						fEbbFloodDataHdl = 0;
					}
					if(numToShowUser > 0)
					{
						short j;
						fEbbFloodDataHdl = (EbbFloodDataH)_NewHandleClear(sizeof(EbbFloodData)*numToShowUser);
						if(!fEbbFloodDataHdl) {TechError("TShioTimeValue::GetTimeValue()", "_NewHandleClear()", 0); err = memFullErr; if(tvals)DisposeHandle((Handle)tvals); goto done_currents;}
						for(i = 0, j=0; i < numEbbFloods; i++)
						{
							EbbFloodTime = EbbFloodTimesPtr[i];
							EbbFloodSpeed = EbbFloodSpeedsPtr[i];
							EbbFlood = EbbFloodPtr[i];
							
							if(EbbFloodTime.flag == 0)
							{
								EbbFloodData ebbFloodData;
								ebbFloodData.time = beginSeconds + (long) (EbbFloodTime.val*3600); // value in seconds
								ebbFloodData.speedInKnots = EbbFloodSpeed; // value in knots
								ebbFloodData.type = EbbFlood; // 0 -> Min Before Flood.
																		// 1 -> Max Flood.
																		// 2 -> Min Before Ebb.
																		// 3 -> Max Ebb.
								INDEXH(fEbbFloodDataHdl,j++) = ebbFloodData;
							}
						}
					}
								
				}
				/////////////////////////////////////////////////
				
			}
		}
		
done_currents:
		
		// dispose of GetTideCurrent allocated handles
		//CleanUpCompCurrents(&answers);
		if (answers)
		{
			if (answers->time) {delete [] answers->time; answers->time = 0;}
			if (answers->speed) {delete [] answers->speed; answers->speed = 0;}
			if (answers->u) {delete [] answers->u; answers->u = 0;}
			if (answers->v) {delete [] answers->v; answers->v = 0;}
			if (answers->uMinor) {delete [] answers->uMinor; answers->uMinor = 0;}
			if (answers->vMajor) {delete [] answers->vMajor; answers->vMajor = 0;}
			if (answers->EbbFloodSpeeds) {delete [] answers->EbbFloodSpeeds; answers->EbbFloodSpeeds = 0;}
			if (answers->EbbFloodTimes) {delete [] answers->EbbFloodTimes; answers->EbbFloodTimes = 0;}
			if (answers->EbbFlood) {delete [] answers->EbbFlood; answers->EbbFlood = 0;}
			
			delete answers;
			
			answers = 0;
		}
		if (XODE)  {delete [] XODE; XODE = 0;}
		if (VPU)  {delete [] VPU; VPU = 0;}
		if (conArray) {delete [] conArray; conArray = 0;}
		if(err) return err;
	}

	else if (this->fStationType == 'H')
	{	
		// get the heights
		COMPHEIGHTS *answers;
		//memset(&answers,0,sizeof(answers));
		answers = new COMPHEIGHTS;		
		
		answers->nPts = 0;
		answers->time = 0;
		answers->height = 0;
		answers->numHighLows = 0;
		answers->xtra = 0;
		answers->HighLowHeights = 0;
		answers->HighLowTimes = 0;
		answers->HighLow = 0;
		
		err = GetTideHeight(&beginDate,&endDate,
						XODE,VPU,
						numConstituents,
						conArray,	
						//YHdl,
						&fHeightOffset,
						answers,
						//&fConstituent,	
						//**minmaxvalhdl, // not used
						//**minmaxtimehdl, // not used
						//nminmax, // not used
						//*cntrlvars, // not used
						daylightSavings);

		//model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid

		if (!err)
		{
			model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
			long i,num10MinValues = answers->nPts,numCopied = 0;
			if(num10MinValues > 0 && answers->time && answers->height)
			{
				// convert the heights into speeds
				TimeValuePairH tvals = (TimeValuePairH)_NewHandle(num10MinValues*sizeof(TimeValuePair));
				if(!tvals) 
					{TechError("TShioTimeValue::GetTimeValue()", "_NewHandle()", 0); err = memFullErr; goto done_heights;}
				else
				{
					// first copy non-flagged values, then do the derivative in a second loop
					// note this data is no longer used
					double **heightHdl = (DOUBLEH)_NewHandle(num10MinValues*sizeof(double));
					TimeValuePair tvPair;
					double deriv;
					for(i = 0; i < num10MinValues; i++)
					{
						if(answers->time[i].flag == outOfSequenceFlag) continue;// skip this value, code goes here
						if(answers->time[i].flag == 1) continue;// skip this value, 1 is don't plot flag - junk at beginning or end of data
						// note:timeHdl values are in hrs from start
						tvPair.time = beginSeconds + (long) (answers->time[i].val*3600); 
						tvPair.value.u = 0; // fill in later
						tvPair.value.v = 0; // not used
						INDEXH(tvals,numCopied) = tvPair;
						INDEXH(heightHdl,numCopied) = answers->height[i];
						numCopied++;
					}
					_SetHandleSize((Handle)tvals, numCopied*sizeof(TimeValuePair));
					_SetHandleSize((Handle)heightHdl, numCopied*sizeof(double));

					long imax = numCopied-1;
					for(i = 0; i < numCopied; i++)
					{					
						if (i>0 && i<imax) 
						{	// temp fudge, need to approx derivative to convert to speed (m/s), 
							// from ft/min (time step is 10 minutes), use centered difference
							// for now use extra 10000 factor to get reasonable values
							// will also need the scale factor, check about out of sequence flag
							deriv = 3048*(INDEXH(heightHdl,i+1)-INDEXH(heightHdl,i-1))/1200.; 
						}
						else if (i==0) 
						{	
							deriv = 3048*(INDEXH(heightHdl,i+1)-INDEXH(heightHdl,i))/600.; 
						}
						else if (i==imax) 
						{	
							deriv = 3048*(INDEXH(heightHdl,i)-INDEXH(heightHdl,i-1))/600.; 
						}
						INDEXH(tvals,i).value.u = deriv;	// option to have standing (deriv) vs progressive wave (no deriv)
					}
					this->SetTimeValueHandle(tvals);
					DisposeHandle((Handle)heightHdl);heightHdl=0;
				}
			////// JLM  , save the highs and lows for displaying on the left hand side
				{
					short	numHighLows = answers->numHighLows;			// Number of high and low tide occurrences.								
					double *HighLowHeightsPtr = answers->HighLowHeights;	// double feet
					EXTFLAG *HighLowTimesPtr = answers->HighLowTimes;		// double hours, flag=0 means plot
					short			*HighLowPtr = answers->HighLow;			// 0 -> Low Tide.
																							// 1 -> High Tide.
					short numToShowUser;
					short i;
					double HighLowHeight;
					EXTFLAG HighLowTime;
					short HighLow;
					
					/////////////////////////////////////////////////

					// count the number of values we wish to show to the user
					// (we show the user if the plot flag is set)
					numToShowUser = 0;
					if(HighLowHeightsPtr && HighLowTimesPtr && HighLowPtr)
					{
						for(i = 0; i < numHighLows; i++)
						{
							HighLowTime = HighLowTimesPtr[i];
							if(HighLowTime.flag == 0)
								numToShowUser++;
						}
					}
					
					// now allocate a handle of this size to hold the values for the user
					
					if(fHighLowDataHdl) 
					{
						DisposeHandle((Handle)fHighLowDataHdl); 
						fHighLowDataHdl = 0;
					}
					if(numToShowUser > 0)
					{
						short j;
						fHighLowDataHdl = (HighLowDataH)_NewHandleClear(sizeof(HighLowData)*numToShowUser);
						if(!fHighLowDataHdl) {TechError("TShioTimeValue::GetTimeValue()", "_NewHandleClear()", 0); err = memFullErr; if(tvals)DisposeHandle((Handle)tvals); goto done_heights;}
						for(i = 0, j=0; i < numHighLows; i++)
						{
							HighLowTime = HighLowTimesPtr[i];
							HighLowHeight = HighLowHeightsPtr[i];
							HighLow = HighLowPtr[i];
							
							if(HighLowTime.flag == 0)
							{
								HighLowData highLowData;
								highLowData.time = beginSeconds + (long) (HighLowTime.val*3600); // value in seconds
								highLowData.height = HighLowHeight; // value in feet
								highLowData.type = HighLow; // 0 -> Low Tide.
																	 // 1 -> High Tide.
								INDEXH(fHighLowDataHdl,j++) = highLowData;
							}
						}
					}
								
				}
				/////////////////////////////////////////////////
			}
		}

		/////////////////////////////////////////////////
		// Find derivative
		if(!err)
		{
			long i;
			Boolean valueFound = false;
			Seconds midTime;
			double forHeight, maxMinDeriv, largestDeriv = 0.;
			HighLowData startHighLowData,endHighLowData;
			double scaleFactor;
			char msg[256];
			for( i=0 ; i<this->GetNumHighLowValues()-1; i++) 
			{
				startHighLowData = INDEXH(fHighLowDataHdl, i);
				endHighLowData = INDEXH(fHighLowDataHdl, i+1);
				if (forTime == startHighLowData.time || forTime == this->GetNumHighLowValues()-1)
				{
					(*value).u = 0.;	// derivative is zero at the highs and lows
					(*value).v = 0.;
					valueFound = true;
				}
				if (forTime > startHighLowData.time && forTime < endHighLowData.time && !valueFound)
				{
					(*value).u = GetDeriv(startHighLowData.time, startHighLowData.height, 
						endHighLowData.time, endHighLowData.height, forTime);
					(*value).v = 0.;
					valueFound = true;
				}
				// find the maxMins for this region...
				midTime = (endHighLowData.time - startHighLowData.time)/2 + startHighLowData.time;
				maxMinDeriv = GetDeriv(startHighLowData.time, startHighLowData.height,
					endHighLowData.time, endHighLowData.height, midTime);
					// track largest and save all for left hand list, but only do this first time...
				if (abs(maxMinDeriv)>largestDeriv) largestDeriv = abs(maxMinDeriv);
			}		
			/////////////////////////////////////////////////
			// ask for a scale factor if not known from wizard
			sprintf(msg,lfFix("The largest calculated derivative was %.4lf"),largestDeriv);
			strcat(msg, ".  Enter scale factor for heights coefficients file : ");
			if (fScaleFactor==0)
			{
				err = GetScaleFactorFromUser(msg,&scaleFactor);
				if (err) goto done_heights;
				fScaleFactor = scaleFactor;
			}
			(*value).u = (*value).u * fScaleFactor;
		}
	
		
done_heights:
		
		// dispose of GetTideHeight allocated handles
		//CleanUpCompHeights(&answers);
		if (answers)
		{
			if (answers->time) {delete [] answers->time; answers->time = 0;}
			if (answers->height) {delete [] answers->height; answers->height = 0;}
			if (answers->HighLowHeights) {delete [] answers->HighLowHeights; answers->HighLowHeights = 0;}
			if (answers->HighLowTimes) {delete [] answers->HighLowTimes; answers->HighLowTimes = 0;}
			if (answers->HighLow) {delete [] answers->HighLow; answers->HighLow = 0;}
			
			delete answers;
			
			answers = 0;
		}
		if (XODE)  {delete [] XODE; XODE = 0;}
		if (VPU)  {delete [] VPU; VPU = 0;}
		if (conArray) {delete [] conArray; conArray = 0;}
		return err;
	}
	
	else if (this->fStationType == 'P')
	{	
		// get the heights
		//COMPHEIGHTS answers;
		COMPHEIGHTS *answers;
		//memset(&answers,0,sizeof(answers));
		answers = new COMPHEIGHTS;		
		
		answers->nPts = 0;
		answers->time = 0;
		answers->height = 0;
		answers->numHighLows = 0;
		answers->xtra = 0;
		answers->HighLowHeights = 0;
		answers->HighLowTimes = 0;
		answers->HighLow = 0;
		
		err = GetTideHeight(&beginDate,&endDate,
						XODE,VPU,
						numConstituents,
						conArray,	
						//YHdl,
						&fHeightOffset,
						answers,
						//**minmaxvalhdl, // not used
						//**minmaxtimehdl, // not used
						//nminmax, // not used
						//*cntrlvars, // not used
						daylightSavings);

		//model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid

		if (!err)
		{
			model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
			long i,num10MinValues = answers->nPts,numCopied = 0;
			if(num10MinValues > 0 && answers->time && answers->height)
			{
				// code goes here, convert the heights into speeds
				TimeValuePairH tvals = (TimeValuePairH)_NewHandle(num10MinValues*sizeof(TimeValuePair));
				if(!tvals) 
					{TechError("TShioTimeValue::GetTimeValue()", "_NewHandle()", 0); err = memFullErr; goto done_heights2;}
				else
				{
					// first copy non-flagged values, then do the derivative in a second loop
					// note this data is no longer used
					//double **heightHdl = (DOUBLEH)_NewHandle(num10MinValues*sizeof(double));
					TimeValuePair tvPair;
					//double deriv;
					for(i = 0; i < num10MinValues; i++)
					{
						if(answers->time[i].flag == outOfSequenceFlag) continue;// skip this value, code goes here
						if(answers->time[i].flag == 1) continue;// skip this value, 1 is don't plot flag - junk at beginning or end of data
						// note:timeHdl values are in hrs from start
						tvPair.time = beginSeconds + (long)(answers->time[i].val*3600); 
						tvPair.value.u = 3048*answers->height[i]/1200.; // convert to m/s
						tvPair.value.v = 0; // not used
						INDEXH(tvals,numCopied) = tvPair;
						//INDEXH(heightHdl,numCopied) = answers->height[i];
						numCopied++;
					}
					_SetHandleSize((Handle)tvals, numCopied*sizeof(TimeValuePair));
					this->SetTimeValueHandle(tvals);
	
				}
			////// JLM  , save the highs and lows for displaying on the left hand side
				{
					short	numHighLows = answers->numHighLows;			// Number of high and low tide occurrences.								
					double *HighLowHeightsPtr = answers->HighLowHeights;	// double feet
					EXTFLAG *HighLowTimesPtr = answers->HighLowTimes;		// double hours, flag=0 means plot
					short			*HighLowPtr = answers->HighLow;			// 0 -> Low Tide.
																							// 1 -> High Tide.
					short numToShowUser;
					short i;
					double HighLowHeight;
					EXTFLAG HighLowTime;
					short HighLow;
					
					/////////////////////////////////////////////////

					// count the number of values we wish to show to the user
					// (we show the user if the plot flag is set)
					numToShowUser = 0;
					if(HighLowHeightsPtr && HighLowTimesPtr && HighLowPtr)
					{
						for(i = 0; i < numHighLows; i++)
						{
							HighLowTime = HighLowTimesPtr[i];
							if(HighLowTime.flag == 0)
								numToShowUser++;
						}
					}
					
					// now allocate a handle of this size to hold the values for the user
					
					if(fHighLowDataHdl) 
					{
						DisposeHandle((Handle)fHighLowDataHdl); 
						fHighLowDataHdl = 0;
					}
					if(numToShowUser > 0)
					{
						short j;
						fHighLowDataHdl = (HighLowDataH)_NewHandleClear(sizeof(HighLowData)*numToShowUser);
						if(!fHighLowDataHdl) {TechError("TShioTimeValue::GetTimeValue()", "_NewHandleClear()", 0); err = memFullErr; if(tvals)DisposeHandle((Handle)tvals); goto done_heights2;}
						for(i = 0, j=0; i < numHighLows; i++)
						{
							HighLowTime = HighLowTimesPtr[i];
							HighLowHeight = HighLowHeightsPtr[i];
							HighLow = HighLowPtr[i];
							
							if(HighLowTime.flag == 0)
							{
								HighLowData highLowData;
								highLowData.time = beginSeconds + (long) (HighLowTime.val*3600); // value in seconds
								highLowData.height = HighLowHeight; // value in feet
								highLowData.type = HighLow; // 0 -> Low Tide.
																	 // 1 -> High Tide.
								INDEXH(fHighLowDataHdl,j++) = highLowData;
							}
						}
					}
								
				}
				/////////////////////////////////////////////////
			}
		}

		/////////////////////////////////////////////////
		// Find derivative
		if(!err)
		{
			/////////////////////////////////////////////////
			// ask for a scale factor if not known from wizard
				/*fScaleFactor = 1;	// will want a scale factor, but not related to derivative
			(*value).u = (*value).u * fScaleFactor;*/
			/////////////////////////////////////////////////
			// ask for a scale factor if not known from wizard
			//sprintf(msg,lfFix("The largest calculated derivative was %.4lf"),largestDeriv);
			//strcat(msg, ".  Enter scale factor for heights coefficients file : ");
			char msg[256];
			double scaleFactor;
			strcpy(msg, "Enter scale factor for progressive wave coefficients file : ");
			if (fScaleFactor==0)
			{
				err = GetScaleFactorFromUser(msg,&scaleFactor);
				if (err) goto done_heights2;
				fScaleFactor = scaleFactor;
			}
			(*value).u = (*value).u * fScaleFactor;
		}
	
		
done_heights2:
		
		// dispose of GetTideHeight allocated handles
		//CleanUpCompHeights(&answers);
		if (answers)
		{
			if (answers->time) {delete [] answers->time; answers->time = 0;}
			if (answers->height) {delete [] answers->height; answers->height = 0;}
			if (answers->HighLowHeights) {delete [] answers->HighLowHeights; answers->HighLowHeights = 0;}
			if (answers->HighLowTimes) {delete [] answers->HighLowTimes; answers->HighLowTimes = 0;}
			if (answers->HighLow) {delete [] answers->HighLow; answers->HighLow = 0;}
			
			delete answers;
			
			answers = 0;
		}
		if (XODE)  {delete [] XODE; XODE = 0;}
		if (VPU)  {delete [] VPU; VPU = 0;}
		if (conArray) {delete [] conArray; conArray = 0;}
		return err;
	}
	return TOSSMTimeValue::GetTimeValue(forTime,value);
}

double TShioTimeValue::GetDeriv (Seconds t1, double val1, Seconds t2, double val2, Seconds theTime)
{
	double dt = float (t2 - t1) / 3600.;
	if( dt<0.000000001){
		return val2;
	}
	double x = (theTime - t1) / (3600. * dt);
	double deriv = 6. * x * (val1 - val2) * (x - 1.) / dt;
	return (3048./3600.) * deriv;	// convert from ft/hr to m/s, added 10^4 fudge factor so scale is O(1)
}

OSErr TShioTimeValue::GetConvertedHeightValue(Seconds forTime, VelocityRec *value)
{
	long i;
	OSErr err = 0;
	HighLowData startHighLowData,endHighLowData;
	for( i=0 ; i<this->GetNumHighLowValues()-1; i++) 
	{
		startHighLowData = INDEXH(fHighLowDataHdl, i);
		endHighLowData = INDEXH(fHighLowDataHdl, i+1);
		if (forTime == startHighLowData.time || forTime == this->GetNumHighLowValues()-1)
		{
			(*value).u = 0.;	// derivative is zero at the highs and lows
			(*value).v = 0.;
			return noErr;
		}
		if (forTime>startHighLowData.time && forTime<endHighLowData.time)
		{
			(*value).u = GetDeriv(startHighLowData.time, startHighLowData.height, 
				endHighLowData.time, endHighLowData.height, forTime) * fScaleFactor;
			(*value).v = 0.;
			return noErr;
		}
	}
	return -1; // point not found
}

OSErr TShioTimeValue::GetProgressiveWaveValue(Seconds forTime, VelocityRec *value)
{
	OSErr err = 0;
	(*value).u = 0;
	(*value).v = 0;
	if (err = TOSSMTimeValue::GetTimeValue(forTime,value)) return err;

	(*value).u = (*value).u * fScaleFactor;	// derivative is zero at the highs and lows
	(*value).v = (*value).v * fScaleFactor;

	return err;
}

WorldPoint TShioTimeValue::GetRefWorldPoint (void)
{
	WorldPoint wp;
	wp.pLat = fLatitude * 1000000;
	wp.pLong = fLongitude * 1000000;
	return wp;
}

/////////////////////////////////////////////////
OSErr TShioTimeValue::GetLocationInTideCycle(short *ebbFloodType, float *fraction)
{

	Seconds time = model->GetModelTime(), ebbFloodTime;	
	EbbFloodData ebbFloodData1, ebbFloodData2;
	long i, numValues;
	short type;
	float factor;
	OSErr err = 0;
	
	*ebbFloodType = -1;
	*fraction = 0;
	//////////////////////
	if (this->fStationType == 'C')
	{	
		numValues = this->GetNumEbbFloodValues();
		for (i=0; i<numValues-1; i++)
		{
			ebbFloodData1 = INDEXH(fEbbFloodDataHdl, i);
			ebbFloodData2 = INDEXH(fEbbFloodDataHdl, i+1);
			if (ebbFloodData1.time <= time && ebbFloodData2.time > time)
			{
				*ebbFloodType = ebbFloodData1.type;
				*fraction = (float)(time - ebbFloodData1.time) / (float)(ebbFloodData2.time - ebbFloodData1.time);
				return 0;
			}
			if (i==0 && ebbFloodData1.time > time)
			{
				if (ebbFloodData1.type>0) *ebbFloodType = ebbFloodData1.type - 1;
				else *ebbFloodType = 3;
				*fraction = (float)(ebbFloodData1.time - time)/ (float)(ebbFloodData2.time - ebbFloodData1.time);	// a fudge for now
				if (*fraction>1) *fraction=1;
				return 0;
			}
		}
		if (time==ebbFloodData2.time)
		{
			*ebbFloodType = ebbFloodData2.type;
			*fraction = 0;
			return 0;
		}
		printError("Ebb Flood data could not be determined");
		return -1;
	}
	/*else 
	{
		printNote("Shio height files aren't implemented for tidal cycle current mover");
		return -1;
	}*/
	/////////
	else if (this->fStationType == 'H')
	{
		// this needs work
		Seconds derivTime,derivTime2;
		short type1,type2;
		numValues = 2*(this->GetNumHighLowValues())-1;
		for (i=0; i<numValues; i++) // show only max/mins, converted from high/lows
		{
			HighLowData startHighLowData,endHighLowData,nextHighLowData;
			double maxMinDeriv;
			Seconds midTime,midTime2;
			long index = floor(i/2.);
	
			startHighLowData = INDEXH(fHighLowDataHdl, index);
			endHighLowData = INDEXH(fHighLowDataHdl, index+1);
	
			midTime = (endHighLowData.time - startHighLowData.time)/2 + startHighLowData.time;
			midTime2 = (endHighLowData.time - startHighLowData.time)/2 + endHighLowData.time;
	
			switch(startHighLowData.type)
			{
				case	LowTide:
					if (fmod(i,2.) == 0)	
					{
						derivTime = startHighLowData.time;
						type1 = MinBeforeFlood;
					}
					else	
					{
						derivTime = midTime;
						type1 = MaxFlood;
					}
					break;
				case	HighTide:
					if (fmod(i,2.) == 0)	
					{
						derivTime = startHighLowData.time;
						type1 = MinBeforeEbb;
					}
					else 
					{
						derivTime = midTime;
						type1 = MaxEbb;
					}
					break;
			}
			switch(endHighLowData.type)
			{
				case	LowTide:
					if (fmod(i,2.) == 0)	
					{
						derivTime2 = endHighLowData.time;
						type2 = MinBeforeFlood;
					}
					else	
					{
						derivTime2 = midTime2;
						type2 = MaxFlood;
					}
					break;
				case	HighTide:
					if (fmod(i,2.) == 0)	
					{
						derivTime2 = endHighLowData.time;
						type2 = MinBeforeEbb;
					}
					else 
					{
						derivTime2 = midTime2;
						type2 = MaxEbb;
					}
					break;
			}
			//maxMinDeriv = GetDeriv(startHighLowData.time, startHighLowData.height,
				//endHighLowData.time, endHighLowData.height, derivTime) * fScaleFactor / KNOTSTOMETERSPERSEC;

			//StringWithoutTrailingZeros(valStr,maxMinDeriv,1);
			//SecondsToDate(derivTime, &time);
			if (derivTime <= time && derivTime2 > time)
			{
				*ebbFloodType = type1;
				*fraction = (float)(time - derivTime) / (float)(derivTime2 - derivTime);
				return 0;
			}
			if (i==0 && derivTime > time)
			{
				if (type1>0) *ebbFloodType = type1 - 1;
				else *ebbFloodType = 3;
				*fraction = (float)(derivTime - time)/ (float)(derivTime2 - derivTime);	// a fudge for now
				if (*fraction>1) *fraction=1;
				return 0;
			}
		}
		if (time==derivTime2)
		{
			*ebbFloodType = type2;
			*fraction = 0;
			return 0;
		}
		printError("Ebb Flood data could not be determined");
		return -1;
	}

	return 0;
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



void TShioTimeValue::ProgrammerError(char* routine)
{
	char str[256];
	if(routine)  sprintf(str,"Programmer error: can't call %s() for TShioTimeValue objects",routine);
	else sprintf(str,"Programmer error: TShioTimeValue object");
	printError(str);
}



OSErr TShioTimeValue::GetTimeChange(long a, long b, Seconds *dt)
{	// having this this function is inherited but meaningless
	this->ProgrammerError("GetTimeChange");
	*dt = 0;
	return -1;
}

OSErr TShioTimeValue::GetInterpolatedComponent(Seconds forTime, double *value, short index)
{	// having this function is inherited but meaningless
	this->ProgrammerError("GetInterpolatedComponent");
	*value = 0;
	return -1;
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

