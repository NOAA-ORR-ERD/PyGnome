/*
 *  ShioTimeValue_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "Basics.h"
#include "Shio.h"
#include "OUTILS.H"
#include "ShioTimeValue_c.h"
#include "StringFunctions.h"
#include "MemUtils.h"
#include <iostream>

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

using std::cout;
using std::fstream;
using std::ios;

void ShioTimeValue_c::InitInstanceVariables(void)
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

ShioTimeValue_c::ShioTimeValue_c() : OSSMTimeValue_c()
{
	daylight_savings_off = false;	// daylight savings should be true by default (so off=false) to match SHIO output
	//daylight_savings_off = true;	// JS: What is this for? - this just means we pay attention to dst - shio still figures out whether it is the right time of year for it
	this->InitInstanceVariables();
	this->fYearDataPath[0] = 0;
}

#ifndef pyGNOME
ShioTimeValue_c::ShioTimeValue_c(TMover *theOwner,TimeValuePairH tvals) : OSSMTimeValue_c(theOwner)
{ 	// having this this function is inherited but meaningless
	this->ProgrammerError("TShioTimeValue constructor");
	this->InitInstanceVariables();
}
ShioTimeValue_c::ShioTimeValue_c(TMover *theOwner) : OSSMTimeValue_c(theOwner)
{ 
	this->InitInstanceVariables();
}
#endif

void ShioTimeValue_c::Dispose()
{
	if(fEbbFloodDataHdl)DisposeHandle((Handle)fEbbFloodDataHdl);
	if(fHighLowDataHdl)DisposeHandle((Handle)fHighLowDataHdl);
	if(fConstituent.H)DisposeHandle((Handle)fConstituent.H);
	if(fConstituent.kPrime)DisposeHandle((Handle)fConstituent.kPrime);
	OSSMTimeValue_c::Dispose();
	this->InitInstanceVariables();
}

char* GetKeyedLine(CHARH f, const char *key, long lineNum, char *strLine)
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

OSErr ShioTimeValue_c::GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, float ***val)
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
	if(err && h) {_DisposeHandle((Handle)h); h = 0;}
	else _SetHandleSize((Handle)h,numVals*sizeof(***val));
	*val = h;
	return err;
}


OSErr ShioTimeValue_c::GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, DATA *val)
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

OSErr ShioTimeValue_c::GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, short *val)
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


OSErr ShioTimeValue_c::GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, float *val)
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

OSErr ShioTimeValue_c::GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, double *val)
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


OSErr ShioTimeValue_c::InitTimeFunc ()
{
 	OSErr err = OSSMTimeValue_c::InitTimeFunc();
 	if(!err)
	{ // init the stuff for this class
		this->InitInstanceVariables();
	}
	return err;
}



/////////////////////////////////////////////////

long ShioTimeValue_c::I_SHIOHIGHLOWS(void)
{
	return 1; // always item #2
}

long  ShioTimeValue_c::I_SHIOEBBFLOODS(void)
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

Boolean ShioTimeValue_c::DaylightSavingTimeInEffect(DateTimeRec *dateStdTime)	// AH 07/09/2012

{
	// Assume that the change from standard to daylight
	// savings time is on the first Sunday in April at 0200 and 
	// the switch back to Standard time is on the
	// last Sunday in October at 0200.             
	
	//return false;	// code goes here, outside US don't use daylight savings
// 	if (settings.daylightSavingsTimeFlag == DAYLIGHTSAVINGSOFF) return false; 
	
	if (this->daylight_savings_off == DAYLIGHTSAVINGSOFF) return false;	
	
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

long	ShioTimeValue_c::GetNumEbbFloodValues()
{
 	long numEbbFloodValues = _GetHandleSize((Handle)fEbbFloodDataHdl)/sizeof(**fEbbFloodDataHdl);
	return numEbbFloodValues;
}

long	ShioTimeValue_c::GetNumHighLowValues()
{
 	long numHighLowValues = _GetHandleSize((Handle)fHighLowDataHdl)/sizeof(**fHighLowDataHdl);
	return numHighLowValues;
}
//#ifndef pyGNOME
void ShioTimeValue_c::GetYearDataDirectory(char* directoryPath)
{
	char applicationFolderPath[256];
	//char errmsg[256];
#ifdef pyGNOME
	GetYearDataPath(applicationFolderPath);
	//sprintf(errmsg,"The year data path was set to %s\n",applicationFolderPath);
	//printNote(errmsg);
	if (applicationFolderPath[0]==0)
		//strcpy(directoryPath,"SampleData/Data/yeardata/");
		printNote("The year data directory has not been set\n");
	else 
		strcpy(directoryPath,applicationFolderPath);
	//sprintf(errmsg,"The year data path being used is %s\n",directoryPath);
	//printNote(errmsg);
#else
#ifdef MAC
//#include <sys/syslimits.h>
	CFURLRef appURL = CFBundleCopyBundleURL(CFBundleGetMainBundle());
	//CFURLGetFileSystemRepresentation(appURL, TRUE, (UInt8 *)directoryPath, PATH_MAX);
	CFURLGetFileSystemRepresentation(appURL, TRUE, (UInt8 *)directoryPath, kMaxNameLen);
    strcat(directoryPath, "/Contents/Resources/Data/yeardata/");
#else
    //dataDirectory = wxGetCwd();
	PathNameFromDirID(TATdirID,TATvRefNum,applicationFolderPath);
	my_p2cstr((StringPtr)applicationFolderPath);
	strcpy(directoryPath,applicationFolderPath);
	strcat(directoryPath,"Data\\yeardata\\");
#endif
#endif
	return;
}
//#endif

/////////////////////////////////////////////////
OSErr ShioTimeValue_c::SetYearDataPath(char *pathName)
{
	strcpy(this->fYearDataPath,pathName);
	return noErr;
}
/////////////////////////////////////////////////
void ShioTimeValue_c::GetYearDataPath(char *pathName)
{
	strcpy(pathName,this->fYearDataPath);
}
/////////////////////////////////////////////////
OSErr ShioTimeValue_c::GetTimeValue(const Seconds& current_time, VelocityRec *value)
{
	OSErr err = 0;
	Boolean needToCompute = true;
	// Seconds modelStartTime = model->GetStartTime();	// minus AH 07/10/2012
	// Seconds modelEndTime = model->GetEndTime();		// minus AH 07/10/2012

	DateTimeRec beginDate, endDate;
	Seconds beginSeconds;
	Boolean daylightSavings;
	YEARDATAHDL		YHdl = 0; 
	double *XODE=0, *VPU=0;
    //YEARDATA		*yearData = (YEARDATA *) NULL;
	long numConstituents;
	CONSTITUENT		*conArray = 0;
	char	directoryPath[256], errStr[256];
	YEARDATA2* yearData = 0;
	
	long numValues = this->GetNumValues(), amtYearData = 0, i;
	// check that to see if the value is in our already computed range
	if(numValues > 0)
	{	
		
		if(INDEXH(this->timeValues, 0).time <= current_time
		   && current_time <= INDEXH(this->timeValues, numValues-1).time)	// AH 07/10/2012
		{ // the value is in the computed range
			if (this->fStationType == 'C')	// allow scale factor for 'P' case
				return OSSMTimeValue_c::GetTimeValue(current_time,value);		// minus AH 07/10/2012
			else if (this->fStationType == 'P')
				return GetProgressiveWaveValue(current_time,value);				// minus AH 07/10/2012
			else if (this->fStationType == 'H')
				return GetConvertedHeightValue(current_time,value);		// AH 07/10/2012
		}
		//this->fScaleFactor = 0;	//if we want to ask for a scale factor for each computed range...
	}
	
	// else we need to re-compute the values
	this->SetTimeValueHandle(nil);
	errStr[0] = 0;
	
	// calculate the values every hour for the interval containing the model run time
	//SecondsToDate(modelStartTime,&beginDate);
	
//	SecondsToDate(modelStartTime-6*3600,&beginDate);	// for now to handle the new tidal current mover 1/27/04 // minus AH 07/10/2012
	SecondsToDate(current_time-6*3600,&beginDate);	// AH 07/10/2012
	
	beginDate.hour = 0; // Shio expects this to be the start of the day
	beginDate.minute = 0;
	beginDate.second = 0;
	DateToSeconds(&beginDate, &beginSeconds);
	
//	SecondsToDate(modelEndTime+24*3600,&endDate);// add one day so that we can truncate to start of the day	// minus AH 07/10/2012
	SecondsToDate(current_time+48*3600,&endDate); // AH 07/10/2012
	
	endDate.hour = 0; // Shio expects this to be the start of the day
	endDate.minute = 0;
	endDate.second = 0;
	 
	daylightSavings = this->DaylightSavingTimeInEffect(&beginDate);// code goes here, set the daylight flag
#ifndef pyGNOME	
#ifdef IBM	// code goes here - decide where to put the yeardata folder
	YHdl = GetYearData(beginDate.year);
	//GetYearDataDirectory(directoryPath);	// put full path together	
	//yearData = ReadYearData(beginDate.year,directoryPath,errStr);	
#else
	//YHdl = (YEARDATAHDL)_NewHandle(0);
	GetYearDataDirectory(directoryPath);	// put full path together	
	yearData = ReadYearData(beginDate.year,directoryPath,errStr);	
#endif
	if(!YHdl && !yearData)  { TechError("TShioTimeValue::GetTimeValue()", "GetYearData()", 0); return -1; }
	
	if (YHdl) amtYearData = _GetHandleSize((Handle)YHdl)/sizeof(**YHdl);
	else if (yearData) amtYearData = yearData->numElements;
	try
	{
		XODE = new double[amtYearData];
		VPU = new double[amtYearData];
	}
	catch (...)
	{
		TechError("TShioTimeValue::GetTimeValue()", "new double()", 0); return -1;
	}
	if (YHdl)
	{
		for (i = 0; i<amtYearData; i++)
		{
			XODE[i] = (double) INDEXH(YHdl,i).XODE;	
			VPU[i] = (double) INDEXH(YHdl,i).VPU;
		}
	}
	else if (yearData)
	{
		for (i = 0; i<amtYearData; i++)
		{
			XODE[i] = yearData->XODE[i];
			VPU[i] = yearData->VPU[i];
		}
	}
#else
	// code goes here, find a place to keep the year data
	/*YHdl = (YEARDATAHDL)_NewHandle(0);
	amtYearData = 0;
	try
	{
		XODE = new double[amtYearData];
		VPU = new double[amtYearData];
	}
	catch (...)
	{
		TechError("TShioTimeValue::GetTimeValue()", "new double()", 0); return -1;
	}*/
	//char msgStr[256];
	GetYearDataDirectory(directoryPath);	// put full path together
	//sprintf(msgStr,"Path for year data = %s\n",directoryPath);
	//printNote(msgStr);
	yearData = ReadYearData(beginDate.year,directoryPath,errStr);	
	if (errStr[0] != 0) printNote(errStr);
	if(!yearData)  { TechError("TShioTimeValue::GetTimeValue()", "GetYearData()", 0); return -1; }
	
	if (yearData) amtYearData = yearData->numElements;
	try
	{
		XODE = new double[amtYearData];
		VPU = new double[amtYearData];
	}
	catch (...)
	{
		TechError("TShioTimeValue::GetTimeValue()", "new double()", 0); return -1;
	}
	if (yearData)
	{
		for (i = 0; i<amtYearData; i++)
		{
			XODE[i] = yearData->XODE[i];
			VPU[i] = yearData->VPU[i];
		}
	}
	//GetYearDataDirectory(directoryPath);	// put full path together	
	//yearData = ReadYearData(beginDate.year,directoryPath,errStr);
#endif
	
	
	numConstituents = _GetHandleSize((Handle)fConstituent.H)/sizeof(**fConstituent.H);
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
							 //yearData->XODE,yearData->VPU,
							 daylightSavings,
							 fStationName);
		//model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
		if(!err)
		{
#ifndef pyGNOME	// AH 07/10/2012
			model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
#endif
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
						if(answers->time[i].flag == outOfSequenceFlag) continue; // skip this value, code goes here
						if(answers->time[i].flag == 1) continue; // skip this value, 1 is don't plot flag - junk at beginning or end of data
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
					short	*EbbFloodPtr = answers->EbbFlood;			// 0 -> Min Before Flood.
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
						_DisposeHandle((Handle)fEbbFloodDataHdl); 
						fEbbFloodDataHdl = 0;
					}
					if(numToShowUser > 0)
					{
						short j;
						fEbbFloodDataHdl = (EbbFloodDataH)_NewHandleClear(sizeof(EbbFloodData)*numToShowUser);
						if(!fEbbFloodDataHdl) {TechError("TShioTimeValue::GetTimeValue()", "_NewHandleClear()", 0); err = memFullErr; if(tvals)_DisposeHandle((Handle)tvals); goto done_currents;}
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
#ifndef pyGNOME	// AH 07/10/2012
			model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
#endif
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
					_DisposeHandle((Handle)heightHdl);heightHdl=0;
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
						_DisposeHandle((Handle)fHighLowDataHdl); 
						fHighLowDataHdl = 0;
					}
					if(numToShowUser > 0)
					{
						short j;
						fHighLowDataHdl = (HighLowDataH)_NewHandleClear(sizeof(HighLowData)*numToShowUser);
						if(!fHighLowDataHdl) {TechError("TShioTimeValue::GetTimeValue()", "_NewHandleClear()", 0); err = memFullErr; if(tvals)_DisposeHandle((Handle)tvals); goto done_heights;}
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
			double scaleFactor = 1.;
			char msg[256];
			for( i=0 ; i<this->GetNumHighLowValues()-1; i++) 
			{
				startHighLowData = INDEXH(fHighLowDataHdl, i);
				endHighLowData = INDEXH(fHighLowDataHdl, i+1);
		//		if (forTime == startHighLowData.time || forTime == this->GetNumHighLowValues()-1)	// minus AH 07/10/2012
				if (current_time == startHighLowData.time || current_time == this->GetNumHighLowValues()-1)	// AH 07/10/2012
				{
					(*value).u = 0.;	// derivative is zero at the highs and lows
					(*value).v = 0.;
					valueFound = true;
				}
		//		if (forTime > startHighLowData.time && forTime < endHighLowData.time && !valueFound)	// minus AH 07/10/2012
				if (current_time > startHighLowData.time && current_time < endHighLowData.time && !valueFound)	// AH 07/10/2012
				{
				//	(*value).u = GetDeriv(startHighLowData.time, startHighLowData.height, 
				//						  endHighLowData.time, endHighLowData.height, forTime);		// minus AH 07/10/2012
					(*value).u = GetDeriv(startHighLowData.time, startHighLowData.height,endHighLowData.time, endHighLowData.height, current_time);	// AH 07/10/2012
										  
					(*value).v = 0.;
					valueFound = true;
				}
				// find the maxMins for this region...
				midTime = (endHighLowData.time - startHighLowData.time)/2 + startHighLowData.time;
				maxMinDeriv = GetDeriv(startHighLowData.time, startHighLowData.height,
									   endHighLowData.time, endHighLowData.height, midTime);
				// track largest and save all for left hand list, but only do this first time...
				if (fabs(maxMinDeriv) > largestDeriv) largestDeriv = fabs(maxMinDeriv);
			}		
			/////////////////////////////////////////////////
			// ask for a scale factor if not known from wizard
			sprintf(msg,"The largest calculated derivative was %.4lf", largestDeriv);
			strcat(msg, ".  Enter scale factor for heights coefficients file : ");
			if (fScaleFactor==0)
			{
#ifndef pyGNOME
				err = GetScaleFactorFromUser(msg,&scaleFactor);
#else
				err = 1;
#endif
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
#ifndef pyGNOME	// AH 07/10/2012
			model->NewDirtNotification(DIRTY_LIST);// what we display in the list is invalid
#endif
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
						_DisposeHandle((Handle)fHighLowDataHdl); 
						fHighLowDataHdl = 0;
					}
					if(numToShowUser > 0)
					{
						short j;
						fHighLowDataHdl = (HighLowDataH)_NewHandleClear(sizeof(HighLowData)*numToShowUser);
						if(!fHighLowDataHdl) {TechError("TShioTimeValue::GetTimeValue()", "_NewHandleClear()", 0); err = memFullErr; if(tvals)_DisposeHandle((Handle)tvals); goto done_heights2;}
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
#ifndef pyGNOME
				err = GetScaleFactorFromUser(msg,&scaleFactor);
#else
				err = 1;
#endif
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
	return OSSMTimeValue_c::GetTimeValue(current_time,value);	// minus AH 07/10/2012
}

double ShioTimeValue_c::GetDeriv (Seconds t1, double val1, Seconds t2, double val2, Seconds theTime)
{
	double dt = float (t2 - t1) / 3600.;
	if( dt<0.000000001){
		return val2;
	}
	double x = (theTime - t1) / (3600. * dt);
	double deriv = 6. * x * (val1 - val2) * (x - 1.) / dt;
	return (3048./3600.) * deriv;	// convert from ft/hr to m/s, added 10^4 fudge factor so scale is O(1)
}

OSErr ShioTimeValue_c::GetConvertedHeightValue(Seconds forTime, VelocityRec *value)
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

OSErr ShioTimeValue_c::GetProgressiveWaveValue(const Seconds& forTime, VelocityRec *value)
//OSErr ShioTimeValue_c::GetProgressiveWaveValue(const Seconds& start_time, const Seconds& end_time, const Seconds& current_time, VelocityRec *value)
{
	OSErr err = 0;
	(*value).u = 0;
	(*value).v = 0;
	
	if ((err = OSSMTimeValue_c::GetTimeValue(forTime,value)) > 0)
		return err; // minus AH 07/10/2012

	//if (err = OSSMTimeValue_c::GetTimeValue(start_time, end_time, current_time, value)) return err;		// AH 07/10/2012
	
	(*value).u = (*value).u * fScaleFactor;	// derivative is zero at the highs and lows
	(*value).v = (*value).v * fScaleFactor;
	
	return err;
}

WorldPoint ShioTimeValue_c::GetStationLocation (void)
{
	WorldPoint wp;
	wp.pLat = fLatitude * 1000000;
	wp.pLong = fLongitude * 1000000;
	return wp;
}

/////////////////////////////////////////////////
OSErr ShioTimeValue_c::GetLocationInTideCycle(const Seconds& model_time, short *ebbFloodType, float *fraction)
{
	
	Seconds time = model_time, ebbFloodTime;	
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

void ShioTimeValue_c::ProgrammerError(const char *routine)
{
	char str[256];
	if(routine)  sprintf(str,"Programmer error: can't call %s() for TShioTimeValue objects",routine);
	else sprintf(str,"Programmer error: TShioTimeValue object");
	printError(str);
}



OSErr ShioTimeValue_c::GetTimeChange(long a, long b, Seconds *dt)
{	// having this this function is inherited but meaningless
	this->ProgrammerError("GetTimeChange");
	*dt = 0;
	return -1;
}

OSErr ShioTimeValue_c::GetInterpolatedComponent(Seconds forTime, double *value, short index)
{	// having this function is inherited but meaningless
	this->ProgrammerError("GetInterpolatedComponent");
	*value = 0;
	return -1;
}


OSErr ShioTimeValue_c::ReadTimeValues (char *path)
{
	// code goes here, use unitsIfKnownInAdvance to tell if we're coming from a location file, 
	// if not and it's a heights file ask if progressive or standing wave (add new field or track as 'P')
	//#pragma unused(unitsIfKnownInAdvance)	
	// Note : this is a subset of the TShioTimeValue::ReadTimeValues, should look at combining the two...
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
	
	if ((err = OSSMTimeValue_c::InitTimeFunc()) > 0)
		return err;

	timeValues = 0;
	fileName[0] = 0;
	
	if (!path)
		return -1;

	strncpy(this->filePath, path, kMaxNameLen);
	this->filePath[kMaxNameLen - 1] = 0;

	strncpy(strLine, path, kMaxNameLen);
	strLine[kMaxNameLen - 1] = 0;

	SplitPathFile(strLine, this->fileName);
	
//	err = ReadFileContents(TERMINATED, 0, 0, path, 0, 0, &f);
//	if(err)	{ TechError("TShioTimeValue::ReadTimeValues()", "ReadFileContents()", 0); return -1; }
	
	char c;
	try {
		int x = i = 0;
		fstream *_ifstream = new fstream(path, ios::in);
		for(; _ifstream->get(c); x++);
		if(!(x > 0))
			throw("empty file.\n");
		f = _NewHandle(x);
		delete _ifstream;
		_ifstream = new fstream(path, ios::in);
		for(; i < x && _ifstream->get(c); i++)
			DEREFH(f)[i] = c;
		delete _ifstream;
	} catch(...) {
		
		printError("We are unable to open or read from the shio tides file. \nBreaking from ShioTimeValue_c::ReadTimeValues().");
		err = true;
		goto readError;
	}
        

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
/*#ifndef pyGNOME
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
#endif*/
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

	if ((err = this->GetKeyedValue(f, "Latitude=", lineNum++, strLine, &this->fLatitude)) > 0)  goto readError;
	if ((err = this->GetKeyedValue(f, "Longitude=", lineNum++, strLine, &this->fLongitude)) > 0)  goto readError;

	if ((!(p = GetKeyedLine(f, "[Constituents]", lineNum++, strLine))) > 0)  goto readError;

	// code goes here in version 1.2.7 these lines won't be required for height files, but should still allow old format
	//if(err = this->GetKeyedValue(f,"DatumControls.datum=",lineNum++,strLine,&this->fConstituent.DatumControls.datum))  goto readError;
	if ((err = this->GetKeyedValue(f, "DatumControls.datum=", lineNum++, strLine, &this->fConstituent.DatumControls.datum)) > 0)
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
	if ((err = this->GetKeyedValue(f, "DatumControls.FDir=", lineNum++, strLine, &this->fConstituent.DatumControls.FDir)) > 0)  goto readError;
	if ((err = this->GetKeyedValue(f, "DatumControls.EDir=", lineNum++, strLine, &this->fConstituent.DatumControls.EDir)) > 0)  goto readError;
	if ((err = this->GetKeyedValue(f, "DatumControls.L2Flag=", lineNum++, strLine, &this->fConstituent.DatumControls.L2Flag)) > 0)  goto readError;
	if ((err = this->GetKeyedValue(f, "DatumControls.HFlag=", lineNum++, strLine, &this->fConstituent.DatumControls.HFlag)) > 0)  goto readError;
	if ((err = this->GetKeyedValue(f, "DatumControls.RotFlag=", lineNum++, strLine, &this->fConstituent.DatumControls.RotFlag)) > 0)  goto readError;
	
skipDatumControls:
	if ((err = this->GetKeyedValue(f, "H=", lineNum++, strLine, &this->fConstituent.H)) > 0)  goto readError;
	if ((err = this->GetKeyedValue(f, "kPrime=", lineNum++, strLine, &this->fConstituent.kPrime)) > 0)  goto readError;
	
	if (!(p = GetKeyedLine(f,"[Offset]",lineNum++,strLine)))  goto readError;
	
	switch(this->fStationType)
	{
		case 'c': case 'C':
			if ((err = this->GetKeyedValue(f, "MinBefFloodTime=", lineNum++, strLine, &this->fCurrentOffset.MinBefFloodTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "FloodTime=", lineNum++, strLine, &this->fCurrentOffset.FloodTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MinBefEbbTime=", lineNum++, strLine, &this->fCurrentOffset.MinBefEbbTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "EbbTime=", lineNum++, strLine, &this->fCurrentOffset.EbbTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "FloodSpdRatio=", lineNum++, strLine, &this->fCurrentOffset.FloodSpdRatio)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "EbbSpdRatio=", lineNum++, strLine, &this->fCurrentOffset.EbbSpdRatio)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MinBFloodSpd=", lineNum++, strLine, &this->fCurrentOffset.MinBFloodSpd)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MinBFloodDir=", lineNum++, strLine, &this->fCurrentOffset.MinBFloodDir)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MaxFloodSpd=", lineNum++, strLine, &this->fCurrentOffset.MaxFloodSpd)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MaxFloodDir=", lineNum++, strLine, &this->fCurrentOffset.MaxFloodDir)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MinBEbbSpd=", lineNum++, strLine, &this->fCurrentOffset.MinBEbbSpd)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MinBEbbDir=", lineNum++, strLine, &this->fCurrentOffset.MinBEbbDir)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MaxEbbSpd=", lineNum++, strLine, &this->fCurrentOffset.MaxEbbSpd)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "MaxEbbDir=", lineNum++, strLine, &this->fCurrentOffset.MaxEbbDir)) > 0)  goto readError;
			SetFileType(SHIOCURRENTSFILE);
			break;
		case 'h': case 'H': 
			if ((err = this->GetKeyedValue(f, "HighTime=", lineNum++, strLine, &this->fHeightOffset.HighTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "LowTime=", lineNum++, strLine, &this->fHeightOffset.LowTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "HighHeight_Mult=", lineNum++, strLine, &this->fHeightOffset.HighHeight_Mult)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "HighHeight_Add=", lineNum++, strLine, &this->fHeightOffset.HighHeight_Add)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "LowHeight_Mult=", lineNum++, strLine, &this->fHeightOffset.LowHeight_Mult)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "LowHeight_Add=", lineNum++, strLine, &this->fHeightOffset.LowHeight_Add)) > 0)  goto readError;
			SetFileType(SHIOHEIGHTSFILE);
			break;
		case 'p': case 'P': 
			if ((err = this->GetKeyedValue(f, "HighTime=", lineNum++, strLine, &this->fHeightOffset.HighTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "LowTime=", lineNum++, strLine, &this->fHeightOffset.LowTime)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "HighHeight_Mult=", lineNum++, strLine, &this->fHeightOffset.HighHeight_Mult)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "HighHeight_Add=", lineNum++, strLine, &this->fHeightOffset.HighHeight_Add)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "LowHeight_Mult=", lineNum++, strLine, &this->fHeightOffset.LowHeight_Mult)) > 0)  goto readError;
			if ((err = this->GetKeyedValue(f, "LowHeight_Add=", lineNum++, strLine, &this->fHeightOffset.LowHeight_Add)) > 0)  goto readError;
			SetFileType(PROGRESSIVETIDEFILE);
			break;
	}
	
	
	if(f) _DisposeHandle((Handle)f); f = nil;
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
YEARDATA2* gYearDataHdl1990Plus2[kMAXNUMSAVEDYEARS];

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
	//errStr[0] = 0;
	
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
	
}

