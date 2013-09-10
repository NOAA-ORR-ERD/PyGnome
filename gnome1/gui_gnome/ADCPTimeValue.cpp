/*
 *  ADCPTimeValue.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADCPTimeValue.h"
#include "CROSS.H"
#include "TShioTimeValue.h"
#include "EditWindsDialog.h"

//Boolean IsLongWindFile(char* path,short *selectedUnitsP,Boolean *dataInGMTP);
//Boolean IsOSSMTimeFile(char* path,short *selectedUnitsP);
//Boolean IsHydrologyFile(char* path);


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
			
			dynamic_cast<ADCPTimeValue *>(this)->ADCPTimeValue::Dispose(); // get rid of any memory we currently are using
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
	if(err) dynamic_cast<ADCPTimeValue *>(this)->ADCPTimeValue::Dispose(); // don't leave ourselves in a weird state
	return err;
}

OSErr ADCPTimeValue::InitTimeFunc ()
{
	
	return  TTimeValue::InitTimeFunc();
	
}


///////////////////////////////////////////////////////////////////////////



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

OSErr ADCPTimeValue::ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance)
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
		{ TechError("ADCPTimeValue::ReadTimeValues()", "ReadFileContents()", 0); goto done; }
		
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
			{ err = -1; TechError("ADCPTimeValue::ReadTimeValues()", "_NewHandle()", 0); return err; }
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
			if (fabs(u)>500.) {u=0.;v=0.;w=0.;}	// they are using -3276.8 as a fill value
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
/*OSErr ADCPTimeValue::ReadTimeValues_old (char *path, short format, short unitsIfKnownInAdvance)
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
	
	else if(IsOSSMTimeFile(path,&selectedUnits))
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
	
}*/	

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
			listLength =  dynamic_cast<ADCPTimeValue *>(this)->GetNumValues();
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
	ListItem item = { dynamic_cast<ADCPTimeValue *>(this), 0, indent, 0 };
	text[0] = 0; 
	char latS[20], longS[20];
	
	/////////////
	if(n == 0)
	{ 	// line 1 station name
		item.index = I_ADCPSTATIONNAME;	// may want new set here
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		item.indent--;
		sprintf(text,"Station Name: %s",fStationName); 
		item.owner = dynamic_cast<ADCPTimeValue *>(this);
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
		
		
		if(bStationDataOpen && 0 <= n && n< GetListLength())
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
				item.owner = dynamic_cast<ADCPTimeValue *>(this);
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
			item.owner = dynamic_cast<ADCPTimeValue *>(this);
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
		ADCPMover *theOwner = dynamic_cast<ADCPMover*>(this->owner);
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

/*ADCPTimeValue* CreateADCPTimeValue(TMover *theOwner,char* path, char* shortFileName, short unitsIfKnownInAdvance)
{
	char tempStr[256];
	OSErr err = 0;
	
	if (IsADCPFile(path))
	{
		ADCPTimeValue *timeValObj = new ADCPTimeValue(theOwner);
		
		if (!timeValObj)
		{ TechError("LoadADCPTimeValue()", "new ADCPTimeValue()", 0); return nil; }
		
		err = timeValObj->InitTimeFunc();
		if(err) {delete timeValObj; timeValObj = nil; return nil;}  
		
		err = timeValObj->ReadTimeValues_old (path, M19REALREAL, unitsIfKnownInAdvance);
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
}*/

/*ADCPTimeValue* LoadADCPTimeValue(TMover *theOwner, short unitsIfKnownInAdvance)
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
	
	return  CreateADCPTimeValue(theOwner,path,shortFileName,unitsIfKnownInAdvance);	// ask user for units 
}*/

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
	if (timeValues) n = dynamic_cast<ADCPTimeValue *>(this)->GetNumValues();
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

