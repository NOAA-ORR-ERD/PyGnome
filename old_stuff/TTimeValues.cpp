
#include "CROSS.h"
#include "OUtils.h"
#include "MapUtils.h"
#include "OSSM.h"

#include "TShioTimeValue.h"
#include "EditWindsDialog.h"

#ifdef MAC
	#pragma segment TTimeValues
#endif


/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
// OSSM supports what Bushy calls a "Long Wind File"
// A Long Wind file is a wind file with a 4 line 
// header. Here is the info from Bushy
/////////////////////////////////////////////////
// 
//INCHON    
//37,30,126,38
//knots
//LTIME
//0,0,0,0,0,0,0,0
//

//Station name
//Station latitude degree, latitude minute, longitude degree, longitude minute
//units
//time units
//bounding box
//
//
//Notes:
//The latitudes and longitudes are always positive.  OSSM figures out hemisphere from the maps.
//
//Allowable speed units ... not case sensitive
//knots
//meters per second
//centimeters per second
//miles per hour
//
//Time zone label is used to correct time offset if the data is not in local time 
//and the user is running in local time.  Often times wind data comes in GMT or local standard time.  
//In TAP we convert it all to local standard time and never bother with daylight savings time.
//
//Bounding box is an interpolation option for dealing with multiple wind files.  
//There are 4 wind interpolation options in OSSM and bounding boxes that can overlap is one.  
//If you chose another scheme, the bounding box data is ignored, that's why you see all the zeros.
//Upper left latitude, Lower right latitude, Upper left longitude, Lower Right Longitude
//

double UorV(VelocityRec vector, short index);
double UorV(VelocityRec3D vector, short index);

Boolean IsLongWindFile(char* path,short *selectedUnitsP,Boolean *dataInGMTP)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined;
	Boolean dataInGMT = FALSE;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	// code goes here, if lines are long may run out of space in array
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line is either LTIME or GMTTIME
		// that the 6th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		selectedUnits = StrToSpeedUnits(strLine);// note we are not supporting cm/sec in gnome
		if(selectedUnits == kUndefined)
			bIsValid = false; 
		
		/////////////////////////////////////////////////
		
		// fourth line, local or GMT time  
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if(!strcmpnocase(strLine,"LTIME"))
			dataInGMT = FALSE;
		if(!strncmpnocase(strLine,"GMT",strlen("GMT"))) 
			dataInGMT = TRUE;
		else
		{
			dataInGMT = FALSE; // Bushy says the flags can be things like PST, but they all boil down to local time
			// check if this is a valid data line, then it is probably a valid tide file
			// tide files with header have same first 3 lines as long wind files, followed by data
			StringSubstitute(strLine, ',', ' ');
			numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute, value1S, value2S);
			if (numScanned == 7)	
				bIsValid = false;
		}	
		/////////////////////////////////////////////////
		
		// fifth line, grid
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore the grid
		/////////////////////////////////////////////////

		// sixth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
		*dataInGMTP = dataInGMT;
	}
	return bIsValid;
}

Boolean IsHydrologyFile(char* path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units, only cubic feet per second for now
		// added cubic meters per second, and the k versions which should cover all cases 5/18/01
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if (!strcmpnocase(strLine,"CFS") || !strcmpnocase(strLine,"KCFS") 
			|| !strcmpnocase(strLine,"CMS") || !strcmpnocase(strLine,"KCMS")) 
			bIsValid = true;
		else 
		{
			bIsValid = false;
			return bIsValid;
		}
		
		/////////////////////////////////////////////////
				
		// fourth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	
	return bIsValid;
}

Boolean IsOSSMTideFile(char* path,short *selectedUnitsP)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		selectedUnits = StrToSpeedUnits(strLine);// note we are not supporting cm/sec in gnome
		if(selectedUnits == kUndefined)
			bIsValid = false; 
		
		/////////////////////////////////////////////////
				
		// fourth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
	}
	return bIsValid;
}

Boolean IsOSSMHeightFile(char* path,short *selectedUnitsP)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		// we check 
		// that the 3rd line contains the units 
		// that the 4th line can scan 7 values
		line = 0;
		bIsValid = true;
		/////////////////////////////////////////////////
		
		// first line , station name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		// for now we ignore station name
		/////////////////////////////////////////////////
		
		// second line, station position
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		// for now we ignore station position
		/////////////////////////////////////////////////
		
		// third line, units
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);	// code goes here, decide what units to support - m, ft,...
		selectedUnits = StrToSpeedUnits(strLine);// note we are not supporting cm/sec in gnome
		if(selectedUnits == kUndefined)
			bIsValid = false; 
		
		/////////////////////////////////////////////////
				
		// fourth line, first line of data
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned != 7)	
			bIsValid = false;
		/////////////////////////////////////////////////
	}
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
	}
	return bIsValid;
}

OSErr GetScaleFactorFromUser(char *msg, double *scaleFactor)
{
	long itemHit;
	char defaultAns[256];
	char ans[256];
	OSErr err = 0;
	sprintf(defaultAns,lfFix("%.1lf"),1.0);
	while(1)
	{
		// ask for a scale factor
		itemHit = REQUEST(msg,defaultAns,ans);
		if (itemHit == 2)  // user cancelled
		{
			err = -1;
			break;
		}
		short numScanned = sscanf(ans,lfFix("%lf"),scaleFactor);
		if(! numScanned == 1) 
		{
			printError("Error reading scale factor. Invalid entry.");
			err = -1;
		}
	
		if(*scaleFactor > 0 /*&& *scaleFactor <= maxScaleFactor*/) // valid entry, go on
			break;
		
		else // go back to dialog
		{
			//if (*scaleFactor > maxScaleFactor) sprintf(msg,lfFix("Maximum scale factor is %.1lf"),maxScaleFactor);
			if (*scaleFactor <= 0) sprintf(msg,"Scale factor must be positive");
		
			printNote(msg); 
		}
	}
	return err;
}



Boolean IsTimeFile(char* path)
{
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long	line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		DateTimeRec time;
		char value1S[256], value2S[256];
		long numScanned;
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		StringSubstitute(strLine, ',', ' ');
		numScanned = sscanf(strLine, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S);
		if (numScanned == 7)	
			bIsValid = true;
	}
	return bIsValid;
}

TOSSMTimeValue* CreateTOSSMTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance)
{
	char tempStr[256];
	OSErr err = 0;
	
	if(IsShioFile(path))
	{
		TShioTimeValue *timeValObj = new TShioTimeValue(theOwner);
		if (!timeValObj)
			{ TechError("LoadTOSSMTimeValue()", "new TShioTimeValue()", 0); return nil; }

		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return nil;}  
		err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
		if(err) { delete timeValObj; timeValObj = nil; return nil;}
		return timeValObj;
	}
	else if (IsTimeFile(path) || IsHydrologyFile(path) || IsOSSMTideFile(path, &unitsIfKnownInAdvance))
	{
		TOSSMTimeValue *timeValObj = new TOSSMTimeValue(theOwner);
		
		if (!timeValObj)
			{ TechError("LoadTOSSMTimeValue()", "new TOSSMTimeValue()", 0); return nil; }

		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return nil;}  
	
		err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
		if(err) { delete timeValObj; timeValObj = nil; return nil;}
		return timeValObj;
	}	
	// code goes here, add code for OSSMHeightFiles, need scale factor to calculate derivative
	else
	{
		sprintf(tempStr,"File %s is not a recognizable time file.",shortFileName);
		printError(tempStr);
	}
		
	return nil;
}

TOSSMTimeValue* LoadTOSSMTimeValue(TMover *theOwner, short unitsIfKnownInAdvance)
{
	char path[256],shortFileName[256];
	char tempStr[256];
	Point where = CenteredDialogUpLeft(M38d);
	WorldPoint p;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	OSErr err = 0;
	
#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
			   (MyDlgHookUPP)0, &reply, M38d, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return 0;
	strcpy(path, reply.fullPath);
	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);
#else
	sfpgetfile(&where, "",
				(FileFilterUPP)0,
				-1, typeList,
				(DlgHookUPP)0,
				&reply, M38d,
				(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));

	if (!reply.good) return nil; // user canceled
	
	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	strcpy(shortFileName,(char*) reply.fName);
#else
	strcpy(path, reply.fName);
	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);
#endif
#endif	
	
//	return  CreateTOSSMTimeValue(theOwner,path,shortFileName,kUndefined);	// ask user for units 
	return  CreateTOSSMTimeValue(theOwner,path,shortFileName,unitsIfKnownInAdvance);	// ask user for units 
}


/////////////////////////////////////////////////


static short gSUUnits; // don't use data, I think it might be buggy, JLM
OSErr SUInit(DialogPtr dialog, VOIDPTR data)
{
#pragma unused(data)
	short selection = gSUUnits;
	SetDialogItemHandle(dialog, SU_HILITEDEFAULT, (Handle) FrameDefault);
	RegisterPopUpDialog(SELECTUNITS, dialog);
	SetPopSelection (dialog, SU_UNITS, selection); 

	return 0;
}

short SUClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{
	long	menuID_menuItem;
	short selectedUnits;

	switch (itemNum) 
	{
		case SU_CANCEL: return SU_CANCEL;
		
		case SU_OK:
			selectedUnits = GetPopSelection (dialog, SU_UNITS); 
			gSUUnits = selectedUnits;
			return SU_OK;

		case SU_UNITS:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;
	}

	return 0;
}

static PopInfoRec selectUnitsPopTable[] = {
		{ SELECTUNITS, nil, SU_UNITS, 0, pSPEEDUNITS, 0, 1, FALSE, nil }
		};

OSErr AskUserForUnits(short* selectedUnits,Boolean *userCancel)
{
	OSErr err = 0;
	short item;
	PopTableInfo saveTable = SavePopTable();
	short j,numItems= 0;
	PopInfoRec combinedUnitsPopTable[20];
	
	// code to allow a dialog on top of another with pops
	for(j = 0; j < sizeof(selectUnitsPopTable) / sizeof(PopInfoRec);j++)
		combinedUnitsPopTable[numItems++] = selectUnitsPopTable[j];
	for(j= 0; j < saveTable.numPopUps ; j++)
		combinedUnitsPopTable[numItems++] = saveTable.popTable[j];
	
	RegisterPopTable(combinedUnitsPopTable,numItems);

	*userCancel = false;
	gSUUnits = *selectedUnits;
	item = MyModalDialog(SELECTUNITS, mapWindow,nil, SUInit, SUClick);
	RestorePopTableInfo(saveTable);
	if(item == SU_CANCEL) 
	{
		 *userCancel = true;
		 return USERCANCEL; 
	}

	if(item == SU_OK) 
	{
		*selectedUnits = gSUUnits;
		return 0; // JLM 7/8/98
	}
	
	return -1;
}

///////////////////////////////////////////////////////////////////////////

ADCPTimeValue::ADCPTimeValue(TMover *theOwner,TimeValuePairH3D tvals,short userUnits) : TTimeValue(theOwner) 
{ 
	fileName[0]=0;
	timeValues = tvals;
	fUserUnits = userUnits;
	fFileType = ADCPTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	fStationDepth = 0.;
	fBinSize = 0.;
	fNumBins = 1;
	fGMTOffset = 0;
	fSensorOrientation = 0;	// 1:up, 2:down, 0:unknown
	//bOSSMStyle = true;	// may want something to identify different ADCP type?
	bOpen = false;
	bStationPositionOpen = false;
	bStationDataOpen = false;
	fBinDepthsH = 0;
}


ADCPTimeValue::ADCPTimeValue(TMover *theOwner) : TTimeValue(theOwner) 
{ 
	fileName[0]=0;
	timeValues = 0;
	fUserUnits = kUndefined; 
	fFileType = ADCPTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	fStationDepth = 0.;
	fBinSize = 0.;
	fNumBins = 1;
	fGMTOffset = 0;
	fSensorOrientation = 0;	// 1:up, 2:down, 0:unknown
	//bOSSMStyle = true;
	bOpen = false;
	bStationPositionOpen = false;
	bStationDataOpen = false;
	fBinDepthsH = 0;
}


OSErr ADCPTimeValue::MakeClone(ADCPTimeValue **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	ClassID *clone;
	OSErr err = 0;
	Boolean weCreatedIt = false;
	if(!clonePtrPtr) return -1; 
	if(*clonePtrPtr == nil)
	{	// create and return a cloned object.
		*clonePtrPtr = new ADCPTimeValue(this->owner);
		weCreatedIt = true;
		if(!*clonePtrPtr) { TechError("MakeClone()", "new TConstantMover()", 0); return memFullErr;}	
	}
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			ADCPTimeValue * cloneP = dynamic_cast<ADCPTimeValue *>(*clonePtrPtr);// typecast
			TTimeValue *tObj = dynamic_cast<TTimeValue *>(*clonePtrPtr);
			err =  TTimeValue::MakeClone(&tObj);//  pass clone to base class
			if(!err) 
			{
				if(this->timeValues)
				{
					cloneP->timeValues = this->timeValues;
					err = _HandToHand((Handle *)&cloneP->timeValues);
					if(err) 
					{
						cloneP->timeValues = nil;
						goto done;
					}
				}
				
				strcpy(cloneP->fileName,this->fileName);
				cloneP->fUserUnits = this->fUserUnits;
				cloneP->fFileType = this->fFileType;
				cloneP->fScaleFactor = this->fScaleFactor;
				strcpy(cloneP->fStationName,this->fStationName);
				cloneP->fStationPosition = this->fStationPosition;
				//cloneP->bOSSMStyle = this->bOSSMStyle;
			
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


OSErr ADCPTimeValue::BecomeClone(ADCPTimeValue *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			ADCPTimeValue * cloneP = dynamic_cast<ADCPTimeValue *>(clone);// typecast
			
			this->Dispose(); // get rid of any memory we currently are using
			////////////////////
			// do the memory stuff first, in case it fails
			////////
			if(cloneP->timeValues)
			{
				this->timeValues = cloneP->timeValues;
				err = _HandToHand((Handle *)&this->timeValues);
				if(err) 
				{
					this->timeValues = nil;
					goto done;
				}
			}
			
			err =  TTimeValue::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;

			strcpy(this->fileName,cloneP->fileName);
			this->fUserUnits = cloneP->fUserUnits;
			this->fFileType = cloneP->fFileType;
			this->fScaleFactor = cloneP->fScaleFactor;
			strcpy(this->fStationName,cloneP->fStationName);
			this->fStationPosition = cloneP->fStationPosition;
			//this->bOSSMStyle = cloneP->bOSSMStyle;
			
		}
	}
done:
	if(err) this->Dispose(); // don't leave ourselves in a weird state
	return err;
}

long ADCPTimeValue::GetNumValues()
{
	return timeValues == 0 ? 0 : _GetHandleSize((Handle)timeValues)/sizeof(TimeValuePair3D);
}

OSErr ADCPTimeValue::InitTimeFunc ()
{

	return  TTimeValue::InitTimeFunc();

}

double ADCPTimeValue::GetMaxValue()
{
	long i,numValues = GetNumValues();
	TimeValuePair3D tv;
	double maxval = -1,val;
	for(i=0;i<numValues;i++)
	{
		tv=(*timeValues)[i];
		val = sqrt(tv.value.v * tv.value.v + tv.value.u * tv.value.u);
		if(val > maxval)maxval = val;
	}
	return maxval; // JLM
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
// Metadata.dat file
/*
#
# Raw data have not been subjected to the National Ocean Service's 
# quality control or quality assurance procedures and do not meet 
# the criteria and standards of official National Ocean Service data. 
# They are released for limited public use as preliminary data to 
# be used only with appropriate caution. 
#

Metadata used from user-selected 1. deployment.

Station ID                      : HAI1006
Station Name                    : Ordnance Reef, NW Corner
Project Name                    : Ordinance Reef Transport Study
Project Type                    : Tidal Current Survey                                                            
Requested Data Start            : 2010/01/03 22:25
Requested Data End              : 2010/02/17 22:25

Deployment Depth (m)            : 94.3
Deployment Latitude (deg)       : 21.44138
Deployment Longitude (deg)      : -158.21038
GMT Offset (hrs)                : 10

Sensor Type                     : Workhorse ADCP                                                                  
Sensor Orientation              : up
Number of Beams                 : 4
Number of Bins                  : 10
Bin Size (m)                    : 8.0
Blanking Distance (m)           : 1.76
Center to Bin 1 Distance (m)    : 10.2
Platform Height From Bottom (m) : 0.6
*/
Boolean IsCMistMetaDataFile(char* path)
{
	OSErr	err = noErr;
	long	line, i, numHeaderLines = 10;
	char	strLine [512];	//maybe not enough data
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	//short selectedUnits = kUndefined;
	
	// decide what to do about various format options
	// for now just assume default format
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	// code goes here, if lines are long may run out of space in array
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		// we check 
		// that the 2nd line starts '# Raw data' 
		// that the 9th line starts 'Metadata' 
		// that the 11th line starts 'Station ID'
		// maybe only care about first line or 2
		// later check that there is Dispersed Oil, maybe # items in first line of data
		line = 0;
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		/////////////////////////////////////////////////
		
		// first line, oil name
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if(strncmpnocase(strLine,"# Raw data",strlen("# Raw data")))
			return false;
		/////////////////////////////////////////////////
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); 

		// second line, api - will want this eventually
		NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);
		RemoveLeadingAndTrailingWhiteSpace(strLine);
		if(strncmpnocase(strLine,"Station ID",strlen("Station ID")))
			return false;
		/////////////////////////////////////////////////
		// may want to scan further to see what format was used		
	}
	
	return true;
}

OSErr ADCPTimeValue::ReadMetaDataFile (char *path)
{
	CHARH f;
	long numDataLines, numLines, numScanned;
	long i, numHeaderLines = 10, numValues, numBins, gmtOffset;
	OSErr err = 0;
	char s[512], str1[32], str2[32], str3[32], valStr[32], unitStr[32], latDir = 'N', lonDir = 'E';
	double depth,lat,lon, binSize,centerToBinDist,platformHt = 0, sensorDepth = 0.;
	short sensorOrientation;
	DateTimeRec time;
	Seconds startTime, endTime;
	char *p;

	if (!IsCMistMetaDataFile(path)) return -1;
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
		{ TechError("ADCPTimeValue::ReadMetaDataFile()", "ReadFileContents()", 0); goto done; }
	
	numLines = NumLinesInText(*f);
	
	numDataLines = numLines - numHeaderLines;
			
	//time.second = 0;
	
	numValues = 0;
	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, i, s, 512); 
		if(i < numHeaderLines)
			continue; // skip any header lines
		//if(i%200 == 0) MySpinCursor(); 
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line

		if (i==28)	/*StringSubstitute(s, 'to Bin 1', ' ');*/ 
			{if(p = strrchr(s, ':')) {strcpy(s,"Center Distance (m) "); strcat(s,p);} else {err = -1; goto done;}}
		if (i==29)	/*StringSubstitute(s, 'Height From', ' ');*/
		{
			if (strstr(s,"Platform Height"))
			{
				if(p = strrchr(s, ':')) {strcpy(s,"Platform Height (m) "); strcat(s,p);} else {err = -1; goto done;}
			}
		}
		StringSubstitute(s, ':', ' ');
		if (i==10 || i==23)
		{
			numScanned=sscanf(s, "%s %s %s",
						  str1, str2, valStr) ;
			if (numScanned<3)	
				{ err = -1; TechError("ADCPTimeValue::ReadMetaDataFile()", "sscanf() < 3", 0); goto done; }
		}
		else if (i==14 || i==15)
		{
			time.second = 0;
			StringSubstitute(s, '/', ' ');
			numScanned=sscanf(s, "%s %s %s %hd %hd %hd %hd %hd",
						  str1, str2, str3, &time.year, &time.month, &time.day, &time.hour, &time.minute) ;
			if (numScanned<8)	
				{ err = -1; TechError("ADCPTimeValue::ReadMetaDataFile()", "sscanf() < 8", 0); goto done; }
		}
		else
		{
			numScanned=sscanf(s, "%s %s %s %s",
						  str1, str2, unitStr, valStr) ;
			if (numScanned<4)	
				{ err = -1; TechError("ADCPTimeValue::ReadMetaDataFile()", "sscanf() < 4", 0); goto done; }
		}	
		if (i==10) this->SetStationName(valStr);
		if (i==14) DateToSeconds (&time, &startTime);
		if (i==15) DateToSeconds (&time, &endTime);
		if (i==17) {err = StringToDouble(valStr,&depth); if (err) goto done; fStationDepth = depth;}
		if (i==18) {err = StringToDouble(valStr,&lat); if (err) goto done;}
		if (i==19) {err = StringToDouble(valStr,&lon); if (err) goto done; DoublesToWorldPoint(lat,lon,latDir,lonDir,&fStationPosition);}
		if (i==20) {numScanned=sscanf(valStr, "%ld", &gmtOffset); if (numScanned<1) {err = -1; goto done;} fGMTOffset = gmtOffset;}
		if (i==23) 
		{
			if(!strcmpnocase(valStr,"up")) fSensorOrientation = 1; 
			else if (!strcmpnocase(valStr,"down")) fSensorOrientation = 2;
			else {err = -1; goto done;}
		}	
		if (i==25) {numScanned=sscanf(valStr, "%ld", &numBins); if (numScanned<1) {err = -1; goto done;} fNumBins = numBins;}
		if (i==26) {err = StringToDouble(valStr,&binSize); if (err) goto done; fBinSize = binSize;}
		if (i==28) {err = StringToDouble(valStr,&centerToBinDist); if (err) goto done; /*fBinSize = binSize;*/}
		if (i==29) {err = StringToDouble(valStr,&platformHt); if (err) {platformHt = 0; err = 0;} sensorDepth = platformHt;/*goto done;*/ /*fBinSize = binSize;*/}

		// check date is valid - do we care about the start and end times here?
		/*if (time.day<1 || time.day>31 || time.month<1 || time.month>12)
		{
			err = -1;
			printError("Invalid data in time file");
			goto done;
		}
		else if (time.year < 1900)					// two digit date, so fix it
		{
			if (time.year >= 40 && time.year <= 99)	// JLM
				time.year += 1900;
			else
				time.year += 2000;					// correct for year 2000 (00 to 40)
		}*/
	}
	if (!err)
	{
		if (fNumBins>0)
		{
			
			fBinDepthsH = (DOUBLEH)_NewHandleClear(fNumBins * sizeof(double));
			if(!fBinDepthsH){TechError("ADCPTimeValue::ReadMetaDataFile()", "_NewHandleClear()", 0); err = memFullErr; return -1;}
			for (i=0;i<fNumBins; i++)
			{	// order based on sensor orientation
				if (fSensorOrientation == 1) INDEXH(fBinDepthsH,i) = fStationDepth - (platformHt + centerToBinDist + i*fBinSize);
				else INDEXH(fBinDepthsH,i) = sensorDepth + centerToBinDist + i*fBinSize;
				// code goes here - check if binDepth is below the stationDepth and don't use this bin
			} 
		}
	}
done:
	if (err)
	{
		if (fBinDepthsH)
		{
			DisposeHandle((Handle)fBinDepthsH);
			fBinDepthsH = 0;
		}
	}
	return err;

}

OSErr ADCPTimeValue::ReadTimeValues2 (char *path, short format, short unitsIfKnownInAdvance)
{
	long i, j, numDataLines;
	OSErr err = 0;
	CHARH f;
	long numLines, numScanned;
	long numHeaderLines = 11, numValues, numLinesInFirstFile, totalNumValues = 0;
	char s[512], binPath[256], fileName[64], dateStr[32], timeStr[32], stationName[32], metaDataFilePath[256], adcpPath[256];
	char fileNum[64];
	DateTimeRec time;
	TimeValuePair3D pair;
	double u,v,w,julianTime,speed,dir;
	double conversionFactor = .01;	// units are cm/s
	Seconds startTime;
	//TimeValuePairH3D localTimeValues = 0;

	// code goes here, want to store data from surface to bottom? upward looking vs downward looking adcp have different file ordering...

	strcpy(adcpPath,path);
	SplitPathFile(adcpPath,fileName);

	if (!strcmp(fileName,"metadata.dat")) 
		{if (err = ReadMetaDataFile(path)) return err;}
	else
	{
		strcpy(metaDataFilePath,adcpPath);
		strcat(metaDataFilePath,"metadata.dat");
		if (err = ReadMetaDataFile(metaDataFilePath)) return err;
	}

	GetStationName(stationName);
	
	//for (j=0; j<1; j++)	// may need to track size of each bin if they can vary, though I think times should match...
	for (j=0; j<fNumBins; j++)	// may need to track size of each bin if they can vary, though I think times should match...
	{
		strcpy(binPath,adcpPath);
		//strcat(binPath,stationName);	
		sprintf(fileNum,"%s_bin%02ld.dat",stationName,j+1);
		strcat(binPath,fileNum);
		//strcat(binPath,"_bin01.dat");	// will need to put this together and loop through all bins
		//if (!IsBinDataFile(binPath)) return -1;
		if (err = ReadFileContents(TERMINATED,0, 0,binPath, 0, 0, &f))
			{ TechError("ADCPTimeValue::ReadTimeValues2()", "ReadFileContents()", 0); goto done; }
		
		numLines = NumLinesInText(*f);
		
		numDataLines = numLines - numHeaderLines;
			
		// each bin has a file, named stationid_bin#.dat (HA11006_bin01.dat)
		// the metadata.dat file has info on how many bins, etc.
		// not sure if the .hdr file is necessary
		// 11 header lines followed by data, line 9 is the data info
	//# DATE_TIME              JULIAN_TIME   SPEED   DIR VEL_NORTH  VEL_EAST  VEL_VERT
	//#
	//2010-01-03 22:33:00        3.93958333   20.6 164.0     -19.8       5.7      -2.1

		//localTimeValues = (TimeValuePairH3D)_NewHandle(numDataLines * sizeof(TimeValuePair3D));
		if (j==0)
		{
			numLinesInFirstFile = numLines;
			//timeValues = (TimeValuePairH3D)_NewHandle(numDataLines * sizeof(TimeValuePair3D));
			timeValues = (TimeValuePairH3D)_NewHandleClear(fNumBins * numDataLines * sizeof(TimeValuePair3D));
			if (!timeValues)
				{ err = -1; TechError("ADCPTimeValue::ReadTimeValues2()", "_NewHandle()", 0); return err; }
		}
		else
		{
			if (numLines != numLinesInFirstFile) {err = -1; goto done;}	// may want to handle different amount of data in depth bins
			/*long newSize = (i+2)*numDataLines*sizeof(**timeValues); 
			_SetHandleSize((Handle)timeValues,newSize);
			err = _MemError();*/
		}
		
		time.second = 0;
		
		numValues = 0;
		for (i = 0 ; i < numLines ; i++) {
			NthLineInTextOptimized(*f, i, s, 512); // day, month, year, hour, min, value1, value2
			if(i < numHeaderLines)
				continue; // skip any header lines
			if(i%200 == 0) MySpinCursor(); 
			RemoveLeadingAndTrailingWhiteSpace(s);
			if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
			//StringSubstitute(s, ',', ' ');
			
			numScanned=sscanf(s, lfFix("%s %s %lf %lf %lf %lf %lf %lf"),
						  dateStr, timeStr, &julianTime,
						  &speed, &dir, &v, &u, &w) ;
			if (numScanned!=8)	
				{ err = -1; TechError("ADCPTimeValue::ReadTimeValues()", "sscanf() == 8", 0); goto done; }

			StringSubstitute(timeStr, ':', ' ');
			numScanned=sscanf(timeStr, "%hd %hd %hd", &time.hour, &time.minute, &time.second);
			if (numScanned!=3)	
				{ err = -1; TechError("ADCPTimeValue::ReadTimeValues()", "sscanf() == 3", 0); goto done; }

			StringSubstitute(dateStr, '-', ' ');
			numScanned=sscanf(dateStr, "%hd %hd %hd", &time.year, &time.month, &time.day);
			if (numScanned!=3)	
				{ err = -1; TechError("ADCPTimeValue::ReadTimeValues()", "sscanf() == 3", 0); goto done; }
			// check if last line all zeros (an OSSM requirement) if so ignore the line
			//if (i==numLines-1 && time.day==0 && time.month==0 && time.year==0 && time.hour==0 && time.minute==0)
				//continue;
			// check date is valid
			if (time.day<1 || time.day>31 || time.month<1 || time.month>12)
			{
				err = -1;
				printError("Invalid data in time file");
				goto done;
			}
			else if (time.year < 1900)					// two digit date, so fix it
			{
				if (time.year >= 40 && time.year <= 99)	// JLM
					time.year += 1900;
				else
					time.year += 2000;					// correct for year 2000 (00 to 40)
			}

			memset(&pair,0,sizeof(pair));
			DateToSeconds (&time, &pair.time);	// subtract GMT offset here?? convert from hours to seconds
			pair.time = pair.time - fGMTOffset*3600.;
			if (abs(u)>500.) {u=0.;v=0.;w=0.;}	// they are using -3276.8 as a fill value
			pair.value.u = u*conversionFactor;
			pair.value.v = v*conversionFactor;
			pair.value.w = w*conversionFactor;

			if (numValues>0)
			{
				//Seconds timeVal = INDEXH(timeValues, numValues-1).time;
				Seconds timeVal = INDEXH(timeValues, totalNumValues-1).time;
				if (pair.time < timeVal) 
				{
					err=-1;
					printError("Time values are out of order");
					goto done;
				}
			}
			
			//INDEXH(localTimeValues, numValues++) = pair;
			INDEXH(timeValues, totalNumValues++) = pair;
			numValues++;
		}
		/*for (i = 0 ; i < numDataLines ; i++) 
		{
			memset(&pair,0,sizeof(pair));
			//DateToSeconds (&time, &pair.time);
			pair.time = model->GetModelTime() + 3600*i;
			pair.value.u = 1.;
			pair.value.v = 1.;
			pair.value.w = 0.;

			
			//INDEXH(timeValues, numValues++) = pair;
			INDEXH(timeValues, i) = pair;
		}*/
		if(numValues > 0)
		{
			/*long actualSize = numValues*sizeof(**timeValues); 
			_SetHandleSize((Handle)timeValues,actualSize);
			err = _MemError();*/
		}
		else {
			printError("No lines were found");
			err = true;
			goto done;
		}
		//if (localTimeValues) {DisposeHandle((Handle)localTimeValues); localTimeValues = 0;}
	}
	// ask about setting time to first time in file
	startTime = INDEXH(timeValues,0).time;
	// deal with timezone

	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
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
done:
	if(f) {DisposeHandle((Handle)f); f = 0;}
	if(err && timeValues)  {DisposeHandle((Handle)timeValues); timeValues = 0;}
	//if (localTimeValues) {DisposeHandle((Handle)localTimeValues); localTimeValues = 0;}
	
	return err;
	
}
OSErr ADCPTimeValue::ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance)
{
	char s[512], value1S[256], value2S[256];
	long i,numValues,numLines,numScanned;
	double value1, value2, magnitude, degrees;
	CHARH f;
	DateTimeRec time;
	TimeValuePair3D pair;
	OSErr scanErr;
	double conversionFactor = 1.0;
	OSErr err = noErr;
	Boolean askForUnits = TRUE; 
	Boolean isLongWindFile = FALSE, isHydrologyFile = FALSE;
	short selectedUnits = unitsIfKnownInAdvance;
	long numDataLines;
	long numHeaderLines = 0;
	Boolean dataInGMT = FALSE;
	
	if (err = TTimeValue::InitTimeFunc()) return err;
	
	timeValues = 0;
	this->fileName[0] = 0;
	
	if (!path) return 0;
	
	strcpy(s, path);
	SplitPathFile(s, this->fileName);
	
	paramtext(fileName, "", "", "");
	
	// here might need to parse through all files in adcp folder
	isLongWindFile = IsLongWindFile(path,&selectedUnits,&dataInGMT);
	if(isLongWindFile) {
		if(format != M19MAGNITUDEDIRECTION)
		{ // JLM thinks this is a user error, someone selecting a long wind file when creating a non-wind object
			printError("isLongWindFile but format != M19MAGNITUDEDIRECTION");
			{ err = -1; goto done;}
		}
		askForUnits = false;
		numHeaderLines = 5;
	}

	else if(IsOSSMTideFile(path,&selectedUnits))
		numHeaderLines = 3;
		
	else if(isHydrologyFile = IsHydrologyFile(path))	// ask for scale factor, but not units
	{
		SetFileType(HYDROLOGYFILE);
		numHeaderLines = 3;
		selectedUnits = kMetersPerSec;	// so conversion factor is 1
	}

	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f))
		{ TechError("ADCPTimeValue::ReadTimeValues()", "ReadFileContents()", 0); goto done; }
	
	//code goes here, see if we can get the units from the file somehow
	
	if(selectedUnits == kUndefined )
		askForUnits = TRUE;
	else
		askForUnits = FALSE;
	
	if(askForUnits)
	{	
		// we have to ask the user for units...
		Boolean userCancel=false;
		selectedUnits = kKnots; // knots will be default
		err = AskUserForUnits(&selectedUnits,&userCancel);
		if(err || userCancel) { err = -1; goto done;}
	}
	
	switch(selectedUnits)
	{
		case kKnots: conversionFactor = KNOTSTOMETERSPERSEC; break;
		case kMilesPerHour: conversionFactor = MILESTOMETERSPERSEC; break;
		case kMetersPerSec: conversionFactor = 1.0; break;
		default: err = -1; goto done;
	}
	this->SetUserUnits(selectedUnits);
	
	if(dataInGMT)
	{
		printError("GMT data is not yet implemented.");
		err = -2; goto done;
	}
	

/////////////////////////////////////////////////
	
	numLines = NumLinesInText(*f);
	
	numDataLines = numLines - numHeaderLines;
			
	timeValues = (TimeValuePairH3D)_NewHandle(numDataLines * sizeof(TimeValuePair3D));
	if (!timeValues)
		{ err = -1; TechError("ADCPTimeValue::ReadTimeValues()", "_NewHandle()", 0); goto done; }
	
	time.second = 0;
	
	numValues = 0;
	for (i = 0 ; i < numLines ; i++) {
		NthLineInTextOptimized(*f, i, s, 512); // day, month, year, hour, min, value1, value2
		if(i < numHeaderLines)
			continue; // skip any header lines
		if(i%200 == 0) MySpinCursor(); 
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		StringSubstitute(s, ',', ' ');
		
		numScanned=sscanf(s, "%hd %hd %hd %hd %hd %s %s",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute, value1S, value2S) ;
		if (numScanned!=7)	
		// scan will allow comment at end of line, for now just ignore 
			{ err = -1; TechError("ADCPTimeValue::ReadTimeValues()", "sscanf() == 7", 0); goto done; }
		// check if last line all zeros (an OSSM requirement) if so ignore the line
		if (i==numLines-1 && time.day==0 && time.month==0 && time.year==0 && time.hour==0 && time.minute==0)
			continue;
		// check date is valid
		if (time.day<1 || time.day>31 || time.month<1 || time.month>12)
		{
			err = -1;
			printError("Invalid data in time file");
			goto done;
		}
		else if (time.year < 1900)					// two digit date, so fix it
		{
			if (time.year >= 40 && time.year <= 99)	// JLM
				time.year += 1900;
			else
				time.year += 2000;					// correct for year 2000 (00 to 40)
		}

		switch (format) {
			case M19REALREAL:
				scanErr =  StringToDouble(value1S,&value1);
				scanErr =  StringToDouble(value2S,&value2);
				value1*= conversionFactor;//JLM
				value2*= conversionFactor;//JLM
				break;
			case M19MAGNITUDEDEGREES:
				scanErr =  StringToDouble(value1S,&magnitude);
				scanErr =  StringToDouble(value2S,&degrees);
				magnitude*= conversionFactor;//JLM
				ConvertToUV(magnitude, degrees, &value1, &value2);
				break;
			case M19DEGREESMAGNITUDE:
				scanErr =  StringToDouble(value1S,&degrees);
				scanErr =  StringToDouble(value2S,&magnitude);
				magnitude*= conversionFactor;//JLM
				ConvertToUV(magnitude, degrees, &value1, &value2);
				break;
			case M19MAGNITUDEDIRECTION:
				scanErr =  StringToDouble(value1S,&magnitude);
				magnitude*= conversionFactor;//JLM
				ConvertToUV(magnitude, ConvertToDegrees(value2S), &value1, &value2);
				break;
			case M19DIRECTIONMAGNITUDE:
				scanErr =  StringToDouble(value2S,&magnitude);
				magnitude*= conversionFactor;//JLM
				ConvertToUV(magnitude, ConvertToDegrees(value1S), &value1, &value2);
		}

		memset(&pair,0,sizeof(pair));
		DateToSeconds (&time, &pair.time);
		pair.value.u = value1;
		pair.value.v = value2;

		if (numValues>0)
		{
			Seconds timeVal = INDEXH(timeValues, numValues-1).time;
			if (pair.time < timeVal) 
			{
				err=-1;
				printError("Time values are out of order");
				goto done;
			}
		}
		
		INDEXH(timeValues, numValues++) = pair;
	}
	
	if(numValues > 0)
	{
		long actualSize = numValues*sizeof(**timeValues); 
		_SetHandleSize((Handle)timeValues,actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
	}

done:
	if(f) {DisposeHandle((Handle)f); f = 0;}
	if(err &&timeValues)  {DisposeHandle((Handle)timeValues); timeValues = 0;}
	
	return err;
	
}

double ADCPTimeValue::GetBinDepth(long depthIndex)
{
	double binDepth = 0.;
	if (depthIndex < 0 || depthIndex > fNumBins - 1) return 0.;
	if (fBinDepthsH)
		binDepth = INDEXH(fBinDepthsH,depthIndex);	
	return binDepth;
}

OSErr ADCPTimeValue::GetDepthIndices(float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2)
{
	long i;
	OSErr err = 0;
	
	if (!fBinDepthsH) 
	{
		*depthIndex1 = UNASSIGNEDINDEX;
		*depthIndex2 = UNASSIGNEDINDEX;
		return -1;
	}
	// should also look at stationDepth -  don't try to use bins that are below stationDepth
	if (fSensorOrientation == 2)	// downward looking, bins go from top to bottom
	{
		if (depthAtPoint == 0 || totalDepth == 0 || depthAtPoint <= INDEXH(fBinDepthsH,0)) 
		{	// use top value
			*depthIndex1 = 0;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
		for (i=0;i<fNumBins-1;i++)
		{
			if (depthAtPoint > INDEXH(fBinDepthsH,i) && depthAtPoint < INDEXH(fBinDepthsH,i+1))
			{
				*depthIndex1 = i;
				*depthIndex2 = i+1;
				return err;
			}
		}
		if (depthAtPoint>=INDEXH(fBinDepthsH,fNumBins-1))
		{	// use bottom value
			*depthIndex1 = fNumBins-1;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
	}
	else if (fSensorOrientation == 1)	// upward looking, bins go from bottom to top
	{
		if (depthAtPoint == 0 || totalDepth == 0 || depthAtPoint <= INDEXH(fBinDepthsH,fNumBins-1)) 
		{	// use top value
			*depthIndex1 = fNumBins-1;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
		for (i=fNumBins-1;i>0;i--)
		{
			if (depthAtPoint > INDEXH(fBinDepthsH,i) && depthAtPoint < INDEXH(fBinDepthsH,i-1))
			{
				*depthIndex1 = i;
				*depthIndex2 = i-1;
				return err;
			}
		}
		if (depthAtPoint>=INDEXH(fBinDepthsH,0))
		{	// use bottom value
			*depthIndex1 = 0;
			*depthIndex2 = UNASSIGNEDINDEX;
			return err;
		}
	}	
	else 
		return -1;
	return 0;
	
}
void ADCPTimeValue::Dispose()
{
	if (timeValues)
	{
		DisposeHandle((Handle)timeValues);
		timeValues = 0;
	}
	if (fBinDepthsH)
	{
		DisposeHandle((Handle)fBinDepthsH);
		fBinDepthsH = 0;
	}
	TTimeValue::Dispose();
}


OSErr ADCPTimeValue::GetTimeChange(long a, long b, Seconds *dt)
{
	// NOTE: Must be called with a < b, else bogus value may be returned.
	
	(*dt) = INDEXH(timeValues, b).time - INDEXH(timeValues, a).time;
	
	if (*dt == 0)
	{	// better error message, JLM 4/11/01 
		// printError("Duplicate times in time/value table."); return -1; 
		char msg[256];
		char timeS[128];
		DateTimeRec time;
		char* p;
		SecondsToDate (INDEXH(timeValues, a).time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		sprintf(msg,"Duplicate times in time/value table.%s%s%s",NEWLINESTRING,timeS,NEWLINESTRING);
		SecondsToDate (INDEXH(timeValues, b).time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		strcat(msg,timeS);
		printError(msg); return -1; 
	}
	
	return 0;
}



OSErr ADCPTimeValue::GetInterpolatedComponent(Seconds forTime, double *value, short index)
{
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();	// divide by numBins..., also may store start/end time
	double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex;
	OSErr err = 0;
	
	// interpolate value from timeValues array
	n = n / fNumBins;
	// only one element => values are constant
	if (n == 1) { *value = UorV(INDEXH(timeValues, 0).value, index); return 0; }
	
	// only two elements => use linear interopolation
	if (n == 2) { a = 0; b = 1; linear = TRUE; }
	
	if (forTime < INDEXH(timeValues, 0).time) 
	{	// before first element
		if(useExtrapolationCode)
		{ 	// old method
			a = 0; b = 1; linear = TRUE;  //  => use slope to extrapolate 
		}
		else
		{
			// new method  => use first value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, 0).value, index); return 0; 
		}
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) 
	{	// after last element
		if(useExtrapolationCode)
		{ 	// old method
			 a = n - 2; b = n - 1; linear = TRUE; //  => use slope to extrapolate 
		}
		else
		{	// new method => use last value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, n-1).value, index); return 0; 
		}
	}
	
	if (linear) {
		if (err = GetTimeChange(a, b, &dt)) return err;
		
		dv = UorV(INDEXH(timeValues, b).value, index) - UorV(INDEXH(timeValues, a).value, index);
		slope = dv / dt;
		intercept = UorV(INDEXH(timeValues, a).value, index) - slope * INDEXH(timeValues, a).time;
		(*value) = slope * forTime + intercept;
		
		return 0;
	}
	
	// find before and after elements

	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0;
	endIndex = n-1;
	while(endIndex - startIndex > 3)
	{
		midIndex = (startIndex+endIndex)/2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}
	/////////////////////////////////////////
	
	
	for (i = startIndex; i < n; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE)
				{ (*value) = UorV(INDEXH(timeValues, i).value, index); return 0; } // found match
			
			a = i - 1;
			b = i;
			break;
		}
	}
	
	dv = UorV(INDEXH(timeValues, b).value, index) - UorV(INDEXH(timeValues, a).value, index);
	if (fabs(dv) < TIMEVALUE_TOLERANCE) // check for constant value
		{ (*value) = UorV(INDEXH(timeValues, b).value, index); return 0; }
	
	if (err = GetTimeChange(a, b, &dt)) return err;
	
	// interpolated value is between positions a and b
	
	// compute slopes before using Hermite()
	
	if (b == 1) { // special case: between first two elements
		slope1 = dv / dt;
		dv = UorV(INDEXH(timeValues, 2).value, index) - UorV(INDEXH(timeValues, 1).value, index);
		if (err = GetTimeChange(1, 2, &dt)) return err;
		slope2 = dv / dt;
		slope2 = 0.5 * (slope1 + slope2);
	}
	
	else if (b ==  n - 1) { // special case: between last two elements
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, n - 2).value, index) - UorV(INDEXH(timeValues, n - 3).value, index);
		if (err = GetTimeChange(n - 3, n - 2, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope2);
	}
	
	else { // general case
		slope = dv / dt;
		dv = UorV(INDEXH(timeValues, b + 1).value, index) - UorV(INDEXH(timeValues, b).value, index);
		if (err = GetTimeChange(b, b + 1, &dt)) return err;
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, a).value, index) - UorV(INDEXH(timeValues, a - 1).value, index);
		if (err = GetTimeChange(a, a - 1, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope);
		slope2 = 0.5 * (slope2 + slope);
	}
	
	// if (v1 == v2) newValue = v1;
	
	(*value) = Hermite(UorV(INDEXH(timeValues, a).value, index), slope1, INDEXH(timeValues, a).time,
					   UorV(INDEXH(timeValues, b).value, index), slope2, INDEXH(timeValues, b).time, forTime);
	
	return 0;
}

OSErr ADCPTimeValue::GetInterpolatedComponentAtDepth(long depthIndex, Seconds forTime, double *value, short index)
{
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();	// divide by numBins..., also may store start/end time
	double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex, valuesToSkip = 0;
	OSErr err = 0;
	
	// interpolate value from timeValues array
	n = n / fNumBins;
	valuesToSkip = depthIndex*n;
	// only one element => values are constant
	if (n == 1) { *value = UorV(INDEXH(timeValues, valuesToSkip).value, index); return 0; }
	
	// only two elements => use linear interopolation
	if (n == 2) { a = 0; b = 1; linear = TRUE; }
	
	if (forTime < INDEXH(timeValues, valuesToSkip).time) 
	{	// before first element
		if(useExtrapolationCode)
		{ 	// old method
			a = 0; b = 1; linear = TRUE;  //  => use slope to extrapolate 
		}
		else
		{
			// new method  => use first value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, valuesToSkip).value, index); return 0; 
		}
	}
	
	if (forTime > INDEXH(timeValues, valuesToSkip + n - 1).time) 
	{	// after last element
		if(useExtrapolationCode)
		{ 	// old method
			 a = valuesToSkip+ n - 2; b = valuesToSkip + n - 1; linear = TRUE; //  => use slope to extrapolate 
		}
		else
		{	// new method => use last value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, valuesToSkip + n-1).value, index); return 0; 
		}
	}
	
	if (linear) {
		if (err = GetTimeChange(valuesToSkip+a, valuesToSkip+b, &dt)) return err;
		
		dv = UorV(INDEXH(timeValues, valuesToSkip+b).value, index) - UorV(INDEXH(timeValues, valuesToSkip+a).value, index);
		slope = dv / dt;
		intercept = UorV(INDEXH(timeValues,valuesToSkip+ a).value, index) - slope * INDEXH(timeValues, valuesToSkip+a).time;
		(*value) = slope * forTime + intercept;
		
		return 0;
	}
	
	// find before and after elements

	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0+valuesToSkip;
	endIndex = n-1+valuesToSkip;
	while(endIndex - startIndex > 3)
	{
		midIndex = (startIndex+endIndex)/2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}
	/////////////////////////////////////////
	
	
	for (i = startIndex; i < n+valuesToSkip; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE)
				{ (*value) = UorV(INDEXH(timeValues, i).value, index); return 0; } // found match
			
			a = i - 1;
			b = i;
			break;
		}
	}
	
	dv = UorV(INDEXH(timeValues, b).value, index) - UorV(INDEXH(timeValues, a).value, index);
	if (fabs(dv) < TIMEVALUE_TOLERANCE) // check for constant value
		{ (*value) = UorV(INDEXH(timeValues, b).value, index); return 0; }
	
	if (err = GetTimeChange(a, b, &dt)) return err;
	
	// interpolated value is between positions a and b
	
	// compute slopes before using Hermite()
	
	if (b == (valuesToSkip+1)) { // special case: between first two elements
		slope1 = dv / dt;
		dv = UorV(INDEXH(timeValues, valuesToSkip+2).value, index) - UorV(INDEXH(timeValues, valuesToSkip+1).value, index);
		if (err = GetTimeChange(valuesToSkip+1, valuesToSkip+2, &dt)) return err;
		slope2 = dv / dt;
		slope2 = 0.5 * (slope1 + slope2);
	}
	
	else if (b ==  valuesToSkip + n - 1) { // special case: between last two elements
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, valuesToSkip + n - 2).value, index) - UorV(INDEXH(timeValues, valuesToSkip + n - 3).value, index);
		if (err = GetTimeChange(valuesToSkip+ n - 3, valuesToSkip + n - 2, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope2);
	}
	
	else { // general case
		slope = dv / dt;
		dv = UorV(INDEXH(timeValues, b + 1).value, index) - UorV(INDEXH(timeValues, b).value, index);
		if (err = GetTimeChange(b, b + 1, &dt)) return err;
		slope2 = dv / dt;
		dv = UorV(INDEXH(timeValues, a).value, index) - UorV(INDEXH(timeValues, a - 1).value, index);
		if (err = GetTimeChange(a, a - 1, &dt)) return err;
		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope);
		slope2 = 0.5 * (slope2 + slope);
	}
	
	// if (v1 == v2) newValue = v1;
	
	(*value) = Hermite(UorV(INDEXH(timeValues, a).value, index), slope1, INDEXH(timeValues, a).time,
					   UorV(INDEXH(timeValues, b).value, index), slope2, INDEXH(timeValues, b).time, forTime);
	
	return 0;
}

void ADCPTimeValue::SetTimeValueHandle(TimeValuePairH3D t)
{
	if(timeValues && t != timeValues)DisposeHandle((Handle)timeValues);
	timeValues=t;
}

///////////////////////////////////////////////////////////////////////////
OSErr ADCPTimeValue::CheckAndPassOnMessage(TModelMessage *message)
{	
	//char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	//this->GetClassName(ourName);
	
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TTimeValue::CheckAndPassOnMessage(message);
}

OSErr ADCPTimeValue::CheckStartTime(Seconds forTime)
{
	OSErr err = 0;
	long a, b, i, n = GetNumValues();
	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0)
	{
		//TechError("ADCPTimeValue::GetTimeValue()", "timeValues", 0); 
		// no value to return
	//	value->u = 0;
	//	value->v = 0;
		return -1; 
	}

	// only one element => values are constant
	if (n == 1) return -2;/*{ *value = UorV(INDEXH(timeValues, 0).value, index); return 0; }*/
	
	// only two elements => use linear interpolation
//	if (n == 2) { a = 0; b = 1; linear = TRUE; }	// may want warning here
	
	if (forTime < INDEXH(timeValues, 0).time) 
	{	// before first element
			// new method  => use first value,  JLM 9/16/98
		//	*value = UorV(INDEXH(timeValues, 0).value, index); return 0; 
		return -1;
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) 
	{	// after last element
		//	*value = UorV(INDEXH(timeValues, n-1).value, index); return 0;
		return -1;
	}
	
//	if (err = GetInterpolatedComponent(forTime, &value -> u, kUCode)) return err;
//	if (err = GetInterpolatedComponent(forTime, &value -> v, kVCode)) return err;
	
	return 0;
}

OSErr ADCPTimeValue::GetTimeValue(Seconds forTime, VelocityRec *value)
{	// need to have depth indices too
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();
	//double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	//Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex;
	OSErr err = 0;

	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0)
	{
		//TechError("ADCPTimeValue::GetTimeValue()", "timeValues", 0); 
		// no value to return
		value->u = 0;
		value->v = 0;
		return -1; 
	}

	if (err = GetInterpolatedComponent(forTime, &value -> u, kUCode)) return err;
	if (err = GetInterpolatedComponent(forTime, &value -> v, kVCode)) return err;
	/*if (forTime < INDEXH(timeValues, 0).time) 
	{	// before first element
		(*value).u = INDEXH(timeValues, 0).value.u;
		(*value).v = INDEXH(timeValues, 0).value.v; 
		
		return 0; 
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) 
	{	// after last element
		(*value).u = INDEXH(timeValues, n-1).value.u; 
		(*value).v = INDEXH(timeValues, n-1).value.v; 
		return 0;
	}
	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0;
	endIndex = n-1;
	while(endIndex - startIndex > 3)
	{
		midIndex = (startIndex+endIndex)/2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}
	/////////////////////////////////////////
	
	
	for (i = startIndex; i < n; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE)
				{ 
				//(*value) = UorV(INDEXH(timeValues, i).value, index); return 0; 
				(*value).u = INDEXH(timeValues, i).value.u; 
				(*value).v = INDEXH(timeValues, i).value.v; 
				return 0; 
				} // found match
			
			a = i - 1;
			b = i;
			break;
		}
	}*/
	
	
	return 0;
}

OSErr ADCPTimeValue::GetTimeValueAtDepth(long depthIndex, Seconds forTime, VelocityRec *value)
{	// need to have depth indices too
	Boolean linear = FALSE;
	long a, b, i, n = GetNumValues();
	//double dv, slope, slope1, slope2, intercept;
	Seconds dt;
	//Boolean useExtrapolationCode = false;
	long startIndex,midIndex,endIndex;
	OSErr err = 0;

	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0)
	{
		//TechError("ADCPTimeValue::GetTimeValue()", "timeValues", 0); 
		// no value to return
		value->u = 0;
		value->v = 0;
		return -1; 
	}

	if (err = GetInterpolatedComponentAtDepth(depthIndex, forTime, &value -> u, kUCode)) return err;
	if (err = GetInterpolatedComponentAtDepth(depthIndex, forTime, &value -> v, kVCode)) return err;
	/*if (forTime < INDEXH(timeValues, 0).time) 
	{	// before first element
		(*value).u = INDEXH(timeValues, 0).value.u;
		(*value).v = INDEXH(timeValues, 0).value.v; 
		
		return 0; 
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) 
	{	// after last element
		(*value).u = INDEXH(timeValues, n-1).value.u; 
		(*value).v = INDEXH(timeValues, n-1).value.v; 
		return 0;
	}
	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0;
	endIndex = n-1;
	while(endIndex - startIndex > 3)
	{
		midIndex = (startIndex+endIndex)/2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}
	/////////////////////////////////////////
	
	
	for (i = startIndex; i < n; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE)
				{ 
				//(*value) = UorV(INDEXH(timeValues, i).value, index); return 0; 
				(*value).u = INDEXH(timeValues, i).value.u; 
				(*value).v = INDEXH(timeValues, i).value.v; 
				return 0; 
				} // found match
			
			a = i - 1;
			b = i;
			break;
		}
	}*/
	
	
	return 0;
}

void ADCPTimeValue::RescaleTimeValues (double oldScaleFactor, double newScaleFactor)
{
	long i,numValues = GetNumValues();
	TimeValuePair3D tv;

	for (i=0;i<numValues;i++)
	{
		 tv = INDEXH(timeValues,i);
		 tv.value.u /= oldScaleFactor;	// get rid of old scale factor
		 tv.value.v /= oldScaleFactor;	// get rid of old scale factor
		 tv.value.u *= newScaleFactor;
		 tv.value.v *= newScaleFactor;
		INDEXH(timeValues,i) = tv;
	}
	return;
}

/////////////////////////////////////////////////
#define ADCPMAXNUMDATALINESINLIST 201
long ADCPTimeValue::GetListLength() 
{//JLM
	long listLength = 0;
	if (bOpen)
	{
		listLength += 1;	// data header
		if (bStationDataOpen)
		{
			listLength =  this->GetNumValues();
			if(listLength > ADCPMAXNUMDATALINESINLIST)
				listLength = ADCPMAXNUMDATALINESINLIST; // don't show the user too many lines in the case of a huge data record
		}
		listLength++;	// active
		listLength++;	// position
		if (bStationPositionOpen) listLength+=2;	//reference point
	}
	listLength++;	//station name
	return listLength;
}

ListItem ADCPTimeValue::GetNthListItem(long n, short indent, short *style, char *text)
{//JLM
	ListItem item = { this, 0, indent, 0 };
	text[0] = 0; 
	char latS[20], longS[20];

	/////////////
	if(n == 0)
	{ 	// line 1 station name
		item.index = I_ADCPSTATIONNAME;	// may want new set here
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		item.indent--;
		sprintf(text,"Station Name: %s",fStationName); 
		item.owner = this;
		*style = bActive ? italic : normal;
		return item; 
	}
	n--;

	if (bOpen)
	{

		if (n == 0) {
			item.index = I_ADCPSTATIONACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		n--;
		
		
		if (n == 0) {
			item.index = I_ADCPSTATIONREFERENCE;
			item.bullet = bStationPositionOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Station Location");
			
			return item;
		}
		n--;

		if (bStationPositionOpen) {
			if (n < 2) {
				item.indent++;
				item.index = (n == 0) ? I_ADCPSTATIONLAT : I_ADCPSTATIONLONG;
				//item.bullet = BULLET_DASH;
				WorldPointToStrings(fStationPosition, latS, longS);
				strcpy(text, (n == 0) ? latS : longS);
				
				return item;
			}
			
			n--;
		}
		if (bStationPositionOpen) n--;
		if (n == 0) {
			item.index = I_ADCPSTATIONDATA;
			item.bullet = bStationDataOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
			strcpy(text, "Station Data");
			
			return item;
		}
		n--;


	if(bStationDataOpen && 0 <= n && n< ADCPTimeValue::GetListLength())
	{
		DateTimeRec time;
		TimeValuePair3D pair;
		double valueInUserUnits, conversionFactor = 100.;	// convert to cm/s
		char *p,timeS[30];
		char unitsStr[32],valStr[32],valStr2[32];

		if(n >=(ADCPMAXNUMDATALINESINLIST-1))
		{	// JLM 7/21/00 ,this is the last line we will show, indicate that there are more lines but that we aren't going to show them 
			strcpy(text,"...  (there are too many lines to show here)");
			*style = normal;
			item.owner = this;
			return item;
		}
		
		pair = INDEXH(this -> timeValues, n);
		SecondsToDate (pair.time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		{
			/*switch(this->GetUserUnits())
			{
				case kKnots: conversionFactor = KNOTSTOMETERSPERSEC; break;
				case kMilesPerHour: conversionFactor = MILESTOMETERSPERSEC; break;
				case kMetersPerSec: conversionFactor = 1.0; break;
				//default: err = -1; goto done;
			}
			valueInUserUnits = pair.value.u/conversionFactor; //JLM
			ConvertToUnits (this->GetUserUnits(), unitsStr);*/
		}
		
		//StringWithoutTrailingZeros(valStr,valueInUserUnits,6); //JLM
		//valueInUserUnits = pair.value.u * conversionFactor;
		StringWithoutTrailingZeros(valStr,pair.value.u,6); //JLM
		//valueInUserUnits = pair.value.v * conversionFactor;
		StringWithoutTrailingZeros(valStr2,pair.value.v,6); //JLM
		//sprintf(text, "%s -> %s %s", timeS, valStr, unitsStr);///JLM
		//sprintf(text, "%s -> u:%s v:%s %s", timeS, valStr, valStr2, unitsStr);///JLM
		sprintf(text, "%s -> u:%s v:%s", timeS, valStr, valStr2);///JLM
		*style = normal;
		item.owner = this;
		//item.bullet = BULLET_DASH;
	}
	return item;
	}
	item.owner = 0;
	return item;
}

Boolean ADCPTimeValue::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	
	if (inBullet) {
		switch (item.index) {
			case I_ADCPSTATIONNAME: bOpen = !bOpen; return TRUE;
			case I_ADCPSTATIONREFERENCE: bStationPositionOpen = !bStationPositionOpen; return TRUE;
			case I_ADCPSTATIONDATA: bStationDataOpen = !bStationDataOpen; return TRUE;
			//case I_ADCPTIMEFILE: bTimeFileOpen = !bTimeFileOpen; return TRUE;
			//case I_ADCPTIMEFILEACTIVE: bTimeFileActive = !bTimeFileActive; 
					//model->NewDirtNotification(); return TRUE;
			case I_ADCPSTATIONACTIVE:
				bActive = !bActive;
				model->NewDirtNotification(); 
		return TRUE;
		}
	}

	if (doubleClick && !inBullet)
	{
		ADCPMover *theOwner = (ADCPMover*)this->owner;
		Boolean timeFileChanged = false;
		if(theOwner)
			ADCPSettingsDialog (theOwner, theOwner -> moverMap, &timeFileChanged);
		return TRUE;
	}

	// do other click operations...
	
	return FALSE;
}

Boolean ADCPTimeValue::FunctionEnabled(ListItem item, short buttonID)
{
	if (buttonID == SETTINGSBUTTON) return TRUE;
	return FALSE;
}

/////////////////////////////////////////////////

ADCPTimeValue* CreateADCPTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance)
{
	char tempStr[256];
	OSErr err = 0;
	
	/*if(IsShioFile(path))
	{
		TShioTimeValue *timeValObj = new TShioTimeValue(theOwner);
		if (!timeValObj)
			{ TechError("LoadADCPTimeValue()", "new TShioTimeValue()", 0); return nil; }

		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return nil;}  
		err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
		if(err) { delete timeValObj; timeValObj = nil; return nil;}
		return timeValObj;
	}*/
	//else if (IsTimeFile(path) || IsHydrologyFile(path) || IsOSSMTideFile(path, &unitsIfKnownInAdvance))
	if (IsADCPFile(path))
	{
		ADCPTimeValue *timeValObj = new ADCPTimeValue(theOwner);
		
		if (!timeValObj)
			{ TechError("LoadADCPTimeValue()", "new ADCPTimeValue()", 0); return nil; }

		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return nil;}  
	
		err = timeValObj->ReadTimeValues (path, M19REALREAL, unitsIfKnownInAdvance);
		if(err) { delete timeValObj; timeValObj = nil; return nil;}
		return timeValObj;
	}	
	// code goes here, add code for OSSMHeightFiles, need scale factor to calculate derivative
	else
	{
		sprintf(tempStr,"File %s is not a recognizable time file.",shortFileName);
		printError(tempStr);
	}
		
	return nil;
}

ADCPTimeValue* LoadADCPTimeValue(TMover *theOwner, short unitsIfKnownInAdvance)
{
	char path[256],shortFileName[256];
	char tempStr[256];
	Point where = CenteredDialogUpLeft(M38d);
	WorldPoint p;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	OSErr err = 0;
	
#if TARGET_API_MAC_CARBON
	mysfpgetfile(&where, "", -1, typeList,
			   (MyDlgHookUPP)0, &reply, M38d, MakeModalFilterUPP(STDFilter));
	if (!reply.good) return 0;
	strcpy(path, reply.fullPath);
	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);
#else
	sfpgetfile(&where, "",
				(FileFilterUPP)0,
				-1, typeList,
				(DlgHookUPP)0,
				&reply, M38d,
				(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));

	if (!reply.good) return nil; // user canceled
	
	my_p2cstr(reply.fName);
#ifdef MAC
	GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
	strcpy(shortFileName,(char*) reply.fName);
#else
	strcpy(path, reply.fName);
	strcpy(tempStr,path);
	SplitPathFile(tempStr,shortFileName);
#endif
#endif	
	
//	return  CreateADCPTimeValue(theOwner,path,shortFileName,kUndefined);	// ask user for units 
	return  CreateADCPTimeValue(theOwner,path,shortFileName,unitsIfKnownInAdvance);	// ask user for units 
}

/////////////////////////////////////////////////

OSErr ADCPTimeValue::Write(BFPB *bfpb)
{
	long i, n = 0, version = 1, numBins=0;	
	ClassID id = GetClassID ();
	TimeValuePair3D pair;
	double binDepth;
	OSErr err = 0;
	
	if (err = TTimeValue::Write(bfpb)) return err;
	
	StartReadWriteSequence("ADCPTimeValue::Write()");
	
	if (err = WriteMacValue(bfpb, fUserUnits)) return err;
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	if (err = WriteMacValue(bfpb, fileName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fFileType)) return err;
	if (err = WriteMacValue(bfpb, fScaleFactor)) return err;
	if (err = WriteMacValue(bfpb, fStationName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fStationPosition.pLat)) return err;
	if (err = WriteMacValue(bfpb, fStationPosition.pLong)) return err;
	if (err = WriteMacValue(bfpb, fStationDepth)) return err;
	if (err = WriteMacValue(bfpb, fNumBins)) return err;
	if (err = WriteMacValue(bfpb, fBinSize)) return err;
	if (err = WriteMacValue(bfpb, fGMTOffset)) return err;
	if (err = WriteMacValue(bfpb, fSensorOrientation)) return err;
	if (err = WriteMacValue(bfpb, bStationPositionOpen)) return err;
	if (err = WriteMacValue(bfpb, bStationDataOpen)) return err;

	//if (err = WriteMacValue(bfpb, bOSSMStyle)) return err;
	if (timeValues) n = GetNumValues();
	if (err = WriteMacValue(bfpb, n)) return err;
	
	if (timeValues)
		for (i = 0 ; i < n ; i++) {
			pair = INDEXH(timeValues, i);
			if (err = WriteMacValue(bfpb, pair.time)) return err;
			if (err = WriteMacValue(bfpb, pair.value.u)) return err;
			if (err = WriteMacValue(bfpb, pair.value.v)) return err;
			if (err = WriteMacValue(bfpb, pair.value.w)) return err;
		}
	
	numBins = GetNumBins();
	if (err = WriteMacValue(bfpb, numBins)) return err;
	if (fBinDepthsH)
		for (i = 0 ; i < numBins ; i++) {
			binDepth = INDEXH(fBinDepthsH, i);
			if (err = WriteMacValue(bfpb, binDepth)) return err;
		}
	
	return 0;
}

OSErr ADCPTimeValue::Read(BFPB *bfpb)
{
	long i, n, version, numBins;
	ClassID id;
	TimeValuePair3D pair;
	double binDepth;
	OSErr err = 0;
	
	if (err = TTimeValue::Read(bfpb)) return err;
	
	StartReadWriteSequence("ADCPTimeValue::Read()");

	if (err = ReadMacValue(bfpb, &fUserUnits)) return err;
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("ADCPTimeValue::Read()", "id != TYPE_ADCPTIMEVALUES", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > 1) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb, fileName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, &fFileType)) return err;
	if (err = ReadMacValue(bfpb, &fScaleFactor)) return err;
	if (err = ReadMacValue(bfpb, fStationName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, &fStationPosition.pLat)) return err;
	if (err = ReadMacValue(bfpb, &fStationPosition.pLong)) return err;
	if (err = ReadMacValue(bfpb, &fStationDepth)) return err;
	if (err = ReadMacValue(bfpb, &fNumBins)) return err;
	if (err = ReadMacValue(bfpb, &fBinSize)) return err;
	if (err = ReadMacValue(bfpb, &fGMTOffset)) return err;
	if (err = ReadMacValue(bfpb, &fSensorOrientation)) return err;
	if (err = ReadMacValue(bfpb, &bStationPositionOpen)) return err;
	if (err = ReadMacValue(bfpb, &bStationDataOpen)) return err;

	//if (err = ReadMacValue(bfpb, &bOSSMStyle)) return err;
	if (err = ReadMacValue(bfpb, &n)) return err;
	
	if(n>0)
	{	// JLM: note: n = 0 means timeValues was originally nil
		// so only allocate if n> 0
		timeValues = (TimeValuePairH3D)_NewHandle(n * sizeof(TimeValuePair3D));
		if (!timeValues)
			{ TechError("ADCPTimeValue::Read()", "_NewHandle()", 0); return -1; }

		if (timeValues)
			for (i = 0 ; i < n ; i++) {
				if (err = ReadMacValue(bfpb, &pair.time)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.u)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.v)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.w)) return err;
				INDEXH(timeValues, i) = pair;
			}
	}
	if (err = ReadMacValue(bfpb, &numBins)) return err;	// already read this in above...
	if (numBins>0)
	{
		fBinDepthsH = (DOUBLEH)_NewHandleClear(fNumBins * sizeof(double));
		if(!fBinDepthsH){TechError("ADCPTimeValue::ReadFile()", "_NewHandleClear()", 0); err = memFullErr; return -1;}
		for (i=0;i<fNumBins; i++)
		{
			if (err = ReadMacValue(bfpb, &binDepth)) return err;
			INDEXH(fBinDepthsH,i) = binDepth;
		}
	}
	
	return err;
}


/////////////////////////////////////////////////

