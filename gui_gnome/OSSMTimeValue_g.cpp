/*
 *  OSSMTimeValue_g.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "OSSMTimeValue_g.h"
#include "CROSS.H"

Boolean IsLongWindFile(char* path,short *selectedUnitsP,Boolean *dataInGMTP);
Boolean IsHydrologyFile(char* path);
Boolean IsOSSMTideFile(char* path,short *selectedUnitsP);

OSErr OSSMTimeValue_g::InitTimeFunc ()
{
	
	return  TimeValue_g::InitTimeFunc();
	
}

#define TOSSMMAXNUMDATALINESINLIST 201

long OSSMTimeValue_g::GetListLength() 
{//JLM
	long listLength;
	listLength =  dynamic_cast<TOSSMTimeValue *>(this)->GetNumValues();
	if(listLength > TOSSMMAXNUMDATALINESINLIST)
		listLength = TOSSMMAXNUMDATALINESINLIST; // JLM 7/21/00 , don't show the user too many lines in the case of a huge wind record
	return listLength;
}

ListItem OSSMTimeValue_g::GetNthListItem(long n, short indent, short *style, char *text)
{//JLM
	ListItem item = { 0, 0, indent, 0 };
	text[0] = 0; 
	if( 0 <= n && n< OSSMTimeValue_g::GetListLength())
	{
		DateTimeRec time;
		TimeValuePair pair;
		double valueInUserUnits, conversionFactor = 1.0;
		char *p,timeS[30];
		char unitsStr[32],valStr[32];
		
		if(n >=(TOSSMMAXNUMDATALINESINLIST-1))
		{	// JLM 7/21/00 ,this is the last line we will show, indicate that there are more lines but that we aren't going to show them 
			strcpy(text,"...  (there are too many lines to show here)");
			*style = normal;
			item.owner = dynamic_cast<TOSSMTimeValue *>(this);
			return item;
		}
		
		pair = INDEXH(this -> timeValues, n);
		SecondsToDate (pair.time, &time);
		Date2String(&time, timeS);
		if (p = strrchr(timeS, ':')) p[0] = 0; // remove seconds
		if (fFileType == HYDROLOGYFILE)
		{
			//ConvertToUnits (2, unitsStr);	// no units input, everything done in m/s
			valueInUserUnits = pair.value.u;
			/*switch(this->GetUserUnits())
			 {
			 case 1: strcpy(unitsStr,"CMS"); break;
			 case 2: strcpy(unitsStr,"KCMS"); break;
			 case 3: strcpy(unitsStr,"CFS"); break;
			 case 4: strcpy(unitsStr,"KCFS"); break;
			 }*/
			ConvertToTransportUnits(fUserUnits,unitsStr);
		}
		else
		{
			switch(dynamic_cast<TOSSMTimeValue *>(this)->GetUserUnits())
			{
				case kKnots: conversionFactor = KNOTSTOMETERSPERSEC; break;
				case kMilesPerHour: conversionFactor = MILESTOMETERSPERSEC; break;
				case kMetersPerSec: conversionFactor = 1.0; break;
					//default: err = -1; goto done;
			}
			valueInUserUnits = pair.value.u/conversionFactor; //JLM
			ConvertToUnits (dynamic_cast<TOSSMTimeValue *>(this)->GetUserUnits(), unitsStr);
		}
		
		StringWithoutTrailingZeros(valStr,valueInUserUnits,6); //JLM
		sprintf(text, "%s -> %s %s", timeS, valStr, unitsStr);///JLM
		*style = normal;
		item.owner = dynamic_cast<TOSSMTimeValue *>(this);
		//item.bullet = BULLET_DASH;
	}
	return item;
}

Boolean OSSMTimeValue_g::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	
	/*if (inBullet) {
	 if (item.index == I_SHIOHIGHLOWS()) 
	 {fHighLowValuesOpen = !fHighLowValuesOpen; return TRUE;}
	 if (item.index == I_SHIOEBBFLOODS())
	 {fEbbFloodValuesOpen = !fEbbFloodValuesOpen; return TRUE;}
	 return TRUE;
	 }*/
	
	if (doubleClick && !inBullet)
	{
		TCATSMover *theOwner = (TCATSMover*)this->owner;
		Boolean timeFileChanged = false;
		if(theOwner)
			CATSSettingsDialog (theOwner, theOwner -> moverMap, &timeFileChanged);
		return TRUE;
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean OSSMTimeValue_g::FunctionEnabled(ListItem item, short buttonID)
{
	if (buttonID == SETTINGSBUTTON) return TRUE;
	return FALSE;
}


OSErr OSSMTimeValue_g::Write(BFPB *bfpb)
{
	long i, n = 0, version = /*1*/2;	// changed hydrology dialog 2/22/02
	ClassID id = GetClassID ();
	TimeValuePair pair;
	OSErr err = 0;
	
	if (err = TimeValue_g::Write(bfpb)) return err;
	
	StartReadWriteSequence("TOSSMTimeValue::Write()");
	
	if (err = WriteMacValue(bfpb, fUserUnits)) return err;
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	if (err = WriteMacValue(bfpb, fileName, kMaxNameLen)) return err;
	if (err = WriteMacValue(bfpb, fFileType)) return err;
	if (err = WriteMacValue(bfpb, fScaleFactor)) return err;
	//if (err = WriteMacValue(bfpb, fStationName, kMaxNameLen)) return err;
	//if (err = WriteMacValue(bfpb, fStationPosition.pLat)) return err;
	//if (err = WriteMacValue(bfpb, fStationPosition.pLong)) return err;
	if (err = WriteMacValue(bfpb, bOSSMStyle)) return err;
	if (err = WriteMacValue(bfpb, fTransport)) return err;
	if (err = WriteMacValue(bfpb, fVelAtRefPt)) return err;
	if (timeValues) n = dynamic_cast<TOSSMTimeValue *>(this)->GetNumValues();
	if (err = WriteMacValue(bfpb, n)) return err;
	
	if (timeValues)
		for (i = 0 ; i < n ; i++) {
			pair = INDEXH(timeValues, i);
			if (err = WriteMacValue(bfpb, pair.time)) return err;
			if (err = WriteMacValue(bfpb, pair.value.u)) return err;
			if (err = WriteMacValue(bfpb, pair.value.v)) return err;
		}
	
	return 0;
}

OSErr OSSMTimeValue_g::Read(BFPB *bfpb)
{
	long i, n, version;
	ClassID id;
	TimeValuePair pair;
	OSErr err = 0;
	
	if (err = TimeValue_g::Read(bfpb)) return err;
	
	StartReadWriteSequence("TOSSMTimeValue::Read()");
	
	if (err = ReadMacValue(bfpb, &fUserUnits)) return err;
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TOSSMTimeValue::Read()", "id != TYPE_OSSMTIMEVALUES", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	//if (version != 1) { printSaveFileVersionError(); return -1; }
	if (version > 2) { printSaveFileVersionError(); return -1; }
	if (err = ReadMacValue(bfpb, fileName, kMaxNameLen)) return err;
	if (err = ReadMacValue(bfpb, &fFileType)) return err;
	if (err = ReadMacValue(bfpb, &fScaleFactor)) return err;
	if (version>1)
	{
		//if (err = ReadMacValue(bfpb, fStationName, kMaxNameLen)) return err;
		//if (err = ReadMacValue(bfpb, &fStationPosition.pLat)) return err;	// could get this from CATSMover refP
		//if (err = ReadMacValue(bfpb, &fStationPosition.pLong)) return err;
		if (err = ReadMacValue(bfpb, &bOSSMStyle)) return err;
		if (err = ReadMacValue(bfpb, &fTransport)) return err;	
		if (err = ReadMacValue(bfpb, &fVelAtRefPt)) return err;
	}
	if (err = ReadMacValue(bfpb, &n)) return err;
	
	if(n>0)
	{	// JLM: note: n = 0 means timeValues was originally nil
		// so only allocate if n> 0
		timeValues = (TimeValuePairH)_NewHandle(n * sizeof(TimeValuePair));
		if (!timeValues)
		{ TechError("TOSSMTimeValue::Read()", "_NewHandle()", 0); return -1; }
		
		if (timeValues)
			for (i = 0 ; i < n ; i++) {
				if (err = ReadMacValue(bfpb, &pair.time)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.u)) return err;
				if (err = ReadMacValue(bfpb, &pair.value.v)) return err;
				INDEXH(timeValues, i) = pair;
			}
	}
	
	return err;
}



OSErr OSSMTimeValue_g::ReadHydrologyHeader (char *path)
{
	OSErr	err = noErr;
	long	line = 0;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength,numScanned;
	//long latdeg, latmin, longdeg, longmin/*, z = 0*/;
	float latdeg, latmin, longdeg, longmin/*, z = 0*/;
	WorldPoint wp;
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return err;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	if (err) return err;
	
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString		
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);    // station name
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	strcpy(fStationName, strLine);
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);   // station position - lat deg, lat min, long deg, long min
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	StringSubstitute(strLine, ',', ' ');
	
	//numScanned=sscanf(strLine, "%ld %ld %ld %ld", &latdeg, &latmin, &longdeg, &longmin);
	numScanned=sscanf(strLine, "%f %f %f %f", &latdeg, &latmin, &longdeg, &longmin);
	
	//if (numScanned!=4)	
	//{ err = -1; TechError("TOSSMTimeValue::ReadHydrologyHeader()", "sscanf() == 4", 0); goto done; }
	//wp.pLat = (latdeg + latmin/60.) * 1000000;
	//wp.pLong = -(longdeg + longmin/60.) * 1000000;	// need to have header include direction...
	if (numScanned==4)
	{	// support old OSSM style
		wp.pLat = (latdeg + latmin/60.) * 1000000;
		wp.pLong = -(longdeg + longmin/60.) * 1000000;	// need to have header include direction...
		//wp.pLong = (longdeg + longmin/60.) * 1000000;	// need to have header include direction...
		bOSSMStyle = true;
	}
	else if (numScanned==2)
	{
		wp.pLat = latdeg * 1000000;
		//wp.pLong = -latmin * 1000000;
		wp.pLong = latmin * 1000000;
		bOSSMStyle = false;
	}
	else
	{ err = -1; TechError("TOSSMTimeValue::ReadHydrologyHeader()", "sscanf() == 2", 0); goto done; }
	
	//((TCATSMover*)owner)->SetRefPosition(wp,z);
	fStationPosition = wp;
	
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);   // units
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	if (!strcmpnocase(strLine,"CMS")) fUserUnits = kCMS;
	else if (!strcmpnocase(strLine,"KCMS")) fUserUnits = kKCMS;
	else if (!strcmpnocase(strLine,"CFS")) fUserUnits = kCFS;
	else if (!strcmpnocase(strLine,"KCFS")) fUserUnits = kKCFS;
	else err = -1;
	
done:
	return err;
}

OSErr OSSMTimeValue_g::CheckAndPassOnMessage(TModelMessage *message)
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
	return TimeValue_g::CheckAndPassOnMessage(message);
}

OSErr OSSMTimeValue_g::ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance)
{
	char s[512], value1S[256], value2S[256];
	long i,numValues,numLines,numScanned;
	double value1, value2, magnitude, degrees;
	CHARH f;
	DateTimeRec time;
	TimeValuePair pair;
	OSErr scanErr;
	double conversionFactor = 1.0;
	OSErr err = noErr;
	Boolean askForUnits = TRUE; 
	Boolean isLongWindFile = FALSE, isHydrologyFile = FALSE;
	short selectedUnits = unitsIfKnownInAdvance;
	long numDataLines;
	long numHeaderLines = 0;
	Boolean dataInGMT = FALSE;
	
	if (err = TimeValue_g::InitTimeFunc()) return err;
	
	timeValues = 0;
	this->fileName[0] = 0;
	
	if (!path) return 0;
	
	strcpy(s, path);
	SplitPathFile(s, this->fileName);
	
	paramtext(fileName, "", "", "");
	
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
	{ TechError("TOSSMTimeValue::ReadTimeValues()", "ReadFileContents()", 0); goto done; }
	
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
	dynamic_cast<TOSSMTimeValue *>(this)->SetUserUnits(selectedUnits);
	
	if(dataInGMT)
	{
		printError("GMT data is not yet implemented.");
		err = -2; goto done;
	}
	
	
	/////////////////////////////////////////////////
	// ask for a scale factor
	if (isHydrologyFile && unitsIfKnownInAdvance != -2) 
	{
		// if not known from wizard message
		//if (this->fScaleFactor==0)
		{
			//err = GetScaleFactorFromUser("Enter scale factor for hydrology file : ", &conversionFactor);
			//if (err)	goto done;	// user cancelled or error
			if (err = ReadHydrologyHeader(path)) goto done;
			this->fScaleFactor = conversionFactor;
		}
	}
	
	numLines = NumLinesInText(*f);
	
	numDataLines = numLines - numHeaderLines;
	
	timeValues = (TimeValuePairH)_NewHandle(numDataLines * sizeof(TimeValuePair));
	if (!timeValues)
	{ err = -1; TechError("TOSSMTimeValue::ReadTimeValues()", "_NewHandle()", 0); goto done; }
	
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
		{ err = -1; TechError("TOSSMTimeValue::ReadTimeValues()", "sscanf() == 7", 0); goto done; }
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
OSErr OSSMTimeValue_g::MakeClone(TOSSMTimeValue **clonePtrPtr)
{
	// clone should be the address of a  ClassID ptr
	ClassID *clone;
	OSErr err = 0;
	Boolean weCreatedIt = false;
	if(!clonePtrPtr) return -1; 
	if(*clonePtrPtr == nil)
	{	// create and return a cloned object.
		*clonePtrPtr = new TOSSMTimeValue(this->owner);
		weCreatedIt = true;
		if(!*clonePtrPtr) { TechError("MakeClone()", "new TConstantMover()", 0); return memFullErr;}	
	}
	if(*clonePtrPtr)
	{	// copy the fields
		if((*clonePtrPtr)->GetClassID() == this->GetClassID()) // safety check
		{
			TOSSMTimeValue * cloneP = dynamic_cast<TOSSMTimeValue *>(*clonePtrPtr);// typecast
			TTimeValue *tObj = dynamic_cast<TTimeValue *>(*clonePtrPtr);
			err =  TimeValue_g::MakeClone(&tObj);//  pass clone to base class
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
				cloneP->bOSSMStyle = this->bOSSMStyle;
				cloneP->fTransport = this->fTransport;
				cloneP->fVelAtRefPt = this->fVelAtRefPt;
				
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


OSErr OSSMTimeValue_g::BecomeClone(TOSSMTimeValue *clone)
{
	OSErr err = 0;
	
	if(clone)
	{
		if(clone->GetClassID() == this->GetClassID()) // safety check
		{
			TOSSMTimeValue * cloneP = dynamic_cast<TOSSMTimeValue *>(clone);// typecast
			
			dynamic_cast<TOSSMTimeValue *>(this)->Dispose(); // get rid of any memory we currently are using
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
			
			err =  TimeValue_g::BecomeClone(clone);//  pass clone to base class
			if(err) goto done;
			
			strcpy(this->fileName,cloneP->fileName);
			this->fUserUnits = cloneP->fUserUnits;
			this->fFileType = cloneP->fFileType;
			this->fScaleFactor = cloneP->fScaleFactor;
			strcpy(this->fStationName,cloneP->fStationName);
			this->fStationPosition = cloneP->fStationPosition;
			this->bOSSMStyle = cloneP->bOSSMStyle;
			this->fTransport = cloneP->fTransport;
			this->fVelAtRefPt = cloneP->fVelAtRefPt;
			
		}
	}
done:
	if(err) dynamic_cast<TOSSMTimeValue *>(this)->Dispose(); // don't leave ourselves in a weird state
	return err;
}