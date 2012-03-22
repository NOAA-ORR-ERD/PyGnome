
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

