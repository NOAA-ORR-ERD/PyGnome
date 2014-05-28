/*
 *  OSSMTimeValue_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>

#include "OSSMTimeValue_c.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include "OUTILS.H"
#include "TimeValuesIO.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

using namespace std;

OSSMTimeValue_c::OSSMTimeValue_c() : TimeValue_c()
{ 
	fileName[0]=0;
	filePath[0]=0;
	timeValues = 0;
	fUserUnits = kUndefined;
	fFileType = OSSMTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	bOSSMStyle = true;
	fTransport = 0;
	fVelAtRefPt = 0;
#ifdef pyGNOME
	fInterpolationType = LINEAR;
#else
	fInterpolationType = HERMITE;
#endif
}

#ifndef pyGNOME
OSSMTimeValue_c::OSSMTimeValue_c(TMover *theOwner) : TimeValue_c(theOwner) 
{ 
	fileName[0]=0;
	filePath[0]=0;
	timeValues = 0;
	fUserUnits = kUndefined; 
	fFileType = OSSMTIMEFILE;
	fScaleFactor = 0.;
	fStationName[0] = 0;
	fStationPosition.pLat = 0;
	fStationPosition.pLong = 0;
	bOSSMStyle = true;
	fTransport = 0;
	fVelAtRefPt = 0;
	fInterpolationType = HERMITE;	// pyGNOME doesn't use this constructor
}
#endif


OSSMTimeValue_c::~OSSMTimeValue_c ()
{
	Dispose ();
}


OSErr OSSMTimeValue_c::GetTimeChange(long a, long b, Seconds *dt)
{
	// NOTE: Must be called with a < b, else bogus value may be returned.
	
	if (a < b)
		(*dt) = INDEXH(timeValues, b).time - INDEXH(timeValues, a).time;
	else //if (b < a)
		(*dt) = INDEXH(timeValues, a).time - INDEXH(timeValues, b).time;

	if (*dt == 0) {
		// better error message, JLM 4/11/01
		// printError("Duplicate times in time/value table."); return -1; 
		char msg[256];
		char timeS[128];
		DateTimeRec time;
		char* p;

		memset(msg, 0, 256);
		memset(timeS, 0, 128);

		SecondsToDate(INDEXH(timeValues, a).time, &time);
		Date2String(&time, timeS);

		if ((p = strrchr(timeS, ':')) != NULL)
			p[0] = 0; // remove seconds

		sprintf(msg, "Duplicate times in time/value table.%s%s%s", NEWLINESTRING, timeS, NEWLINESTRING);
		SecondsToDate(INDEXH(timeValues, b).time, &time);
		Date2String(&time, timeS);

		if ((p = strrchr(timeS, ':')) != NULL)
			p[0] = 0; // remove seconds

		strcat(msg, timeS);
		printError(msg);

		return -1;
	}
	
	return 0;
}


OSErr OSSMTimeValue_c::GetInterpolatedComponent(Seconds forTime, double *value, short index)
{
	OSErr err = 0;

	long startIndex, midIndex, endIndex;

	long a, b, n = GetNumValues();
	double dv, slope, slope1, slope2, intercept;
	Seconds dt;

	bool useExtrapolationCode = false;
	bool linear = false;
	
	// interpolate value from timeValues array
	
	// only one element => values are constant
	if (n == 1) {
		VelocityRec vRec = INDEXH(timeValues, 0).value;
		*value = UorV(vRec, index); 
		return 0; 
	}
	
	// only two elements => use linear interpolation
	if (n == 2) {
		a = 0;
		b = 1;
		linear = true;
	}
	
	if (forTime < INDEXH(timeValues, 0).time) {
		// before first element
		if (useExtrapolationCode) {
			// old method
			a = 0;
			b = 1;
			linear = true;  //  => use slope to extrapolate
		}
		else {
			// new method  => use first value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, 0).value, index);
			return 0;
		}
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) {
		// after last element
		if (useExtrapolationCode) {
			// old method
			a = n - 2;
			b = n - 1;
			linear = true; //  => use slope to extrapolate
		}
		else {
			// new method => use last value,  JLM 9/16/98
			*value = UorV(INDEXH(timeValues, n - 1).value, index);
			return 0;
		}
	}
	
	if (linear) {
		if ((err = GetTimeChange(a, b, &dt)) != 0)
			return err;
		
		dv = UorV(INDEXH(timeValues, b).value, index)
		   - UorV(INDEXH(timeValues, a).value, index);

		slope = dv / dt;

		intercept = UorV(INDEXH(timeValues, a).value, index)
				  - slope * INDEXH(timeValues, a).time;
		(*value) = slope * forTime + intercept;
		
		return 0;
	}
	
	// find before and after elements
	
	/////////////////////////////////////////////////
	// JLM 7/21/00, we need to speed this up for when we have a lot of values
	// code goes here, (should we use a static to remember a guess of where to start) before we do the binary search ?
	// use a binary method 
	startIndex = 0;
	endIndex = n - 1;
	while(endIndex - startIndex > 3) {
		midIndex = (startIndex + endIndex) / 2;
		if (forTime <= INDEXH(timeValues, midIndex).time)
			endIndex = midIndex;
		else
			startIndex = midIndex;
	}

	for (long i = startIndex; i < n; i++) {
		if (forTime <= INDEXH(timeValues, i).time) {
			dt = INDEXH(timeValues, i).time - forTime;
			if (dt <= TIMEVALUE_TOLERANCE) {
				// found match
				(*value) = UorV(INDEXH(timeValues, i).value, index);
				return 0;
			}
			
			a = i - 1;
			b = i;
			break;
		}
	}

	dv = UorV(INDEXH(timeValues, b).value, index)
	   - UorV(INDEXH(timeValues, a).value, index);

	if (fabs(dv) < TIMEVALUE_TOLERANCE) {
		// check for constant value
		(*value) = UorV(INDEXH(timeValues, b).value, index);
		return 0;
	}
	
	if ((err = GetTimeChange(a, b, &dt)) != 0)
		return err;

	// use linear interpolation for pyGNOME, default is HERMITE
	if (fInterpolationType == LINEAR) {
		slope = dv / dt;

		intercept = UorV(INDEXH(timeValues, a).value, index)
				  - slope * INDEXH(timeValues, a).time;

		(*value) = slope * forTime + intercept;
		
		return 0;
	}
	
	
	// interpolated value is between positions a and b
	
	// compute slopes before using Hermite()
	
	if (b == 1) {
		// special case: between first two elements
		slope1 = dv / dt;

		dv = UorV(INDEXH(timeValues, 2).value, index)
		   - UorV(INDEXH(timeValues, 1).value, index);

		if ((err = GetTimeChange(1, 2, &dt)) != 0)
			return err;

		slope2 = dv / dt;
		slope2 = 0.5 * (slope1 + slope2);
	}
	else if (b ==  n - 1) {
		// special case: between last two elements
		slope2 = dv / dt;

		dv = UorV(INDEXH(timeValues, n - 2).value, index)
		   - UorV(INDEXH(timeValues, n - 3).value, index);

		if ((err = GetTimeChange(n - 3, n - 2, &dt)) != 0)
			return err;

		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope2);
	}
	else {
		// general case
		slope = dv / dt;

		dv = UorV(INDEXH(timeValues, b + 1).value, index)
		   - UorV(INDEXH(timeValues, b).value, index);

		if ((err = GetTimeChange(b, b + 1, &dt)) != 0)
			return err;

		slope2 = dv / dt;

		dv = UorV(INDEXH(timeValues, a).value, index)
		   - UorV(INDEXH(timeValues, a - 1).value, index);

		if ((err = GetTimeChange(a-1, a, &dt)) != 0)
			return err;	// code requires time1 < time2

		slope1 = dv / dt;
		slope1 = 0.5 * (slope1 + slope);
		slope2 = 0.5 * (slope2 + slope);
	}

	(*value) = Hermite(UorV(INDEXH(timeValues, a).value, index),
					   slope1, INDEXH(timeValues, a).time,
					   UorV(INDEXH(timeValues, b).value, index),
					   slope2, INDEXH(timeValues, b).time, forTime);

	return 0;
}

void OSSMTimeValue_c::SetTimeValueHandle(TimeValuePairH t)
{
	if (timeValues && t != timeValues)
		DisposeHandle((Handle)timeValues);

	timeValues = t;
}


void OSSMTimeValue_c::Dispose()
{
	if (timeValues) {
		DisposeHandle((Handle)timeValues);
		timeValues = 0;
	}
	
	TimeValue_c::Dispose();
}


OSErr OSSMTimeValue_c::CheckStartTime(Seconds forTime)
{
	long n = GetNumValues();

	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0) {
		return -1; 
	}

	// only one element => values are constant
	if (n == 1)
		return -2;

	if (forTime < INDEXH(timeValues, 0).time) {
		// before first element
		return -1;
	}
	
	if (forTime > INDEXH(timeValues, n - 1).time) {
		// after last element
		return -1;
	}

	return 0;
}


OSErr OSSMTimeValue_c::GetTimeValue(const Seconds& forTime, VelocityRec *value)
{
	OSErr err = 0;

	if (!timeValues || _GetHandleSize((Handle)timeValues) == 0) {
		// no value to return
		value->u = 0;
		value->v = 0;
		return -1; 
	}

	if ((err = GetInterpolatedComponent(forTime, &value->u, kUCode)) != 0)
		return err;

	if ((err = GetInterpolatedComponent(forTime, &value->v, kVCode)) != 0)
		return err;

	return 0;
}


void OSSMTimeValue_c::RescaleTimeValues (double oldScaleFactor, double newScaleFactor)
{
	long numValues = GetNumValues();
	TimeValuePair tv;

	for (long i = 0; i < numValues; i++) {
		tv = INDEXH(timeValues, i);

		tv.value.u /= oldScaleFactor;	// get rid of old scale factor
		tv.value.v /= oldScaleFactor;	// get rid of old scale factor

		tv.value.u *= newScaleFactor;
		tv.value.v *= newScaleFactor;

		INDEXH(timeValues, i) = tv;
	}

	return;
}


long OSSMTimeValue_c::GetNumValues()
{
	return timeValues == 0 ? 0 : _GetHandleSize((Handle)timeValues) / sizeof(TimeValuePair);
}


double OSSMTimeValue_c::GetMaxValue()
{
	long numValues = GetNumValues();
	TimeValuePair tv;
	double val, maxval = -1;

	for (long i = 0; i < numValues; i++) {
		tv=(*timeValues)[i];
		val = sqrt(tv.value.v * tv.value.v + tv.value.u * tv.value.u);

		if (val > maxval)
			maxval = val;
	}

	return maxval; // JLM
}


OSErr OSSMTimeValue_c::InitTimeFunc ()
{
	return  TimeValue_c::InitTimeFunc();
}


OSErr OSSMTimeValue_c::ReadNCDCWind(char *path)
{
	char s[512];
	char value1S[256], value2S[256];
	char timeStr[256], stationStr[256], hdrStr[256];
	OSErr err = noErr;
	OSErr scanErr;

	long numDataLines;
	long numHeaderLines = 1;
	long numValues, numLines, numScanned;

	double value1, value2, magnitude, degrees;

	double conversionFactor = MILESTOMETERSPERSEC;	// file speeds are in miles per hour
	short format = M19DEGREESMAGNITUDE;

	CHARH f;
	DateTimeRec time;
	TimeValuePair pair;

	memset(s, 0, 512);
	memset(value1S, 0, 256);
	memset(value2S, 0, 256);
	memset(timeStr, 0, 256);
	memset(stationStr, 0, 256);
	memset(hdrStr, 0, 256);

	if ((err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) != 0) {
		TechError("TOSSMTimeValue::ReadNCDCWind()", "ReadFileContents()", 0);
		goto done;
	}

	numLines = NumLinesInText(*f);
	numDataLines = numLines - numHeaderLines;

    this->SetUserUnits(kMilesPerHour);	//check this
	
	timeValues = (TimeValuePairH)_NewHandle(numDataLines * sizeof(TimeValuePair));
	if (!timeValues) {
		err = -1;
		TechError("TOSSMTimeValue::ReadNCDCWind()", "_NewHandle()", 0);
		goto done;
	}

	time.second = 0;

	numValues = 0;
	for (long i = 0; i < numLines; i++)
	{
		NthLineInTextOptimized(*f, i, s, 512); // day, month, year, hour, min, value1, value2

		if (i < numHeaderLines)
			continue; // skip any header lines

		if (i % 200 == 0)
			MySpinCursor();

		RemoveLeadingAndTrailingWhiteSpace(s);
		if (s[0] == 0)
			continue; // it's a blank line, allow this and skip the line

		numScanned = sscanf(s, "%s %s %s %s %s",
						  stationStr, hdrStr, timeStr,
						  value1S, value2S);
		if (numScanned < 5)	{
			// scan will allow comment at end of line, for now just ignore 
			err = -1;
			TechError("TOSSMTimeValue::ReadNDBCWind()", "sscanf() < 6", 0);
			goto done;
		}

		// scan date
		if (!strncmp (value1S,"***",strlen("***")) ||
			!strncmp (value2S,"***",strlen("***")))
		{
			continue;
		}

		numScanned = sscanf(timeStr, "%4hd %2hd %2hd %2hd %2hd",
						  &time.year, &time.month, &time.day,
						  &time.hour, &time.minute);
		if (numScanned < 5) {
			// scan will allow comment at end of line, for now just ignore 
			err = -1;
			TechError("TOSSMTimeValue::ReadNDBCWind()", "sscanf() < 6", 0);
			goto done;
		}

		time.minute = time.second = 0;
		if (time.day < 1 || time.day > 31 || time.month < 1 || time.month > 12) {
			err = -1;
			printError("Invalid data in time file");
			goto done;
		}
		else if (time.year < 1900) {
			// two digit date, so fix it
			if (time.year >= 40 && time.year <= 99)	// JLM
				time.year += 1900;
			else
				time.year += 2000; // correct for year 2000 (00 to 40)
		}

		switch (format) {
			case M19REALREAL:
				scanErr = StringToDouble(value1S, &value1);
				scanErr = StringToDouble(value2S, &value2);

				value1 *= conversionFactor; //JLM
				value2 *= conversionFactor; //JLM
				break;
			case M19MAGNITUDEDEGREES:
				scanErr = StringToDouble(value1S, &magnitude);
				scanErr = StringToDouble(value2S, &degrees);

				magnitude *= conversionFactor; //JLM
				ConvertToUV(magnitude, degrees, &value1, &value2);
				break;
			case M19DEGREESMAGNITUDE:
				scanErr = StringToDouble(value1S, &degrees);
				scanErr = StringToDouble(value2S, &magnitude);

				if (magnitude == 99.0)
					continue;

				if (degrees > 360)
					continue;

				magnitude *= conversionFactor; //JLM
				ConvertToUV(magnitude, degrees, &value1, &value2);
				break;
			case M19MAGNITUDEDIRECTION:
				scanErr = StringToDouble(value1S, &magnitude);

				magnitude *= conversionFactor; //JLM

				ConvertToUV(magnitude, ConvertToDegrees(value2S), &value1, &value2);
				break;
			case M19DIRECTIONMAGNITUDE:
				scanErr = StringToDouble(value2S, &magnitude);

				magnitude *= conversionFactor; //JLM

				ConvertToUV(magnitude, ConvertToDegrees(value1S), &value1, &value2);
		}
		
		memset(&pair, 0, sizeof(pair));
		DateToSeconds(&time, &pair.time);

		pair.value.u = value1;
		pair.value.v = value2;

		if (numValues > 0) {
			Seconds timeVal = INDEXH(timeValues, numValues - 1).time;
			if (pair.time < timeVal) {
				err = -1;
				printError("Time values are out of order");
				goto done;
			}
		}

		INDEXH(timeValues, numValues++) = pair;
	}

	if (numValues > 0) {
		// JS: 9/17/12 - Following does not work for cython.
		// Leave it commented so we can repro and try to do debugging
		//long actualSize = numValues*(long)sizeof(**timeValues);
		long sz = (long)sizeof(**timeValues);
		long actualSize = numValues * sz;

		_SetHandleSize((Handle)timeValues, actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
	}

done:

	if (f) {
		DisposeHandle((Handle)f);
		f = 0;
	}

	if (err && timeValues) {
		DisposeHandle((Handle)timeValues);
		timeValues = 0;
	}

	return err;
}

OSErr OSSMTimeValue_c::ReadNDBCWind(vector<string> &linesInFile, long numHeaderLines)
{
	OSErr err = noErr;

	DateTimeRec time;

	long numValues = 0;
	long numLines = linesInFile.size();
	long numDataLines = numLines - numHeaderLines;

	time.second = 0;
	this->SetUserUnits(kMetersPerSec);	//check this

	timeValues = (TimeValuePairH)_NewHandle(numDataLines * sizeof(TimeValuePair));
	if (!timeValues) {
		err = -1;
		TechError("OSSMTimeValue_c::ReadNDBCWind()", "_NewHandle()", 0);
		goto done;
	}


	for (long i = 0; i < numLines; i++)
	{
		if (i % 200 == 0)
			MySpinCursor();

		if (i < numHeaderLines)
			continue; // skip any header lines

		double conversionFactor = 1.0;	// wind speeds are in mps
		double u, v;
		string value1S, value2S;
		TimeValuePair pair;
		string 	currentLine = trim(linesInFile[i]);


		if (currentLine.size() == 0)
			continue; // it's a blank line, allow this and skip the line

		istringstream lineStream(currentLine); // day, month, year, hour, min, value1, value2

		// first we read the date/time values
		if (numHeaderLines == 1) {
			lineStream >> time.year >> time.month >> time.day
			>> time.hour;
			time.minute = time.second = 0;
		}
		else {
			lineStream >> time.year >> time.month >> time.day
			>> time.hour >> time.minute;
			time.second = 0;
		}

		if (lineStream.fail()) {
			err = -1;
			string errMsg = "Invalid date in data row: '";
			errMsg += linesInFile[i] + "'";
			TechError("OSSMTimeValue_c::ReadNDBCWind()",(char*)errMsg.c_str(), 0);
			goto done;
		}

		// check date is valid
		if (!DateIsValid(time)){
			err = -1;
			string errMsg = "Invalid date in data row: '";
			errMsg += linesInFile[i] + "'";
			TechError( "OSSMTimeValue_c::ReadNDBCWind()",(char*)errMsg.c_str(), 0);
			goto done;
		}

		CorrectTwoDigitYear(time);

		lineStream >> value1S >> value2S;
		if (lineStream.fail()) {
			// scan will allow comment at end of line, for now just ignore
			err = -1;
			TechError("OSSMTimeValue_c::ReadTimeValues()", "scan data values", 0);
			goto done;
		}

		ConvertRowValuesToUV(value1S, value2S,
							 M19DEGREESMAGNITUDE,
							 conversionFactor, u, v);

		memset(&pair, 0, sizeof(pair));
		DateToSeconds(&time, &pair.time);

		pair.value.u = u;
		pair.value.v = v;

		if (numValues > 0) {
			Seconds timeVal = INDEXH(timeValues, numValues-1).time;
			if (pair.time < timeVal) {
				err = -1;
				printError("Time values are out of order");
				goto done;
			}
		}

		INDEXH(timeValues, numValues++) = pair;
	}

	if (numValues > 0) {
		// JS: 9/17/12 - Following does not work for cython.
		// Leave it commented so we can repro and try to do debugging
		//long actualSize = numValues*(long)sizeof(**timeValues);
		long sz = (long)sizeof(**timeValues);
		long actualSize = numValues * sz;

		_SetHandleSize((Handle)timeValues, actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
	}

done:

	if (err && timeValues) {
		DisposeHandle((Handle)timeValues);
		timeValues = 0;
	}

	return err;
}


OSErr OSSMTimeValue_c::ReadNDBCWind(char *path, long numHeaderLines)
{
	vector<string> linesInFile;

	if (ReadLinesInFile(path, linesInFile)) {
		return ReadNDBCWind(linesInFile, numHeaderLines);
	}
	else {
		return false;
	}
}


OSErr OSSMTimeValue_c::ReadTimeValues(char *path, short format, short unitsIfKnownInAdvance)
{
	char s[512];
	OSErr err = noErr;

	long numValues = 0;
	long numLines;
	long numDataLines;
	long numHeaderLines = 0;

	double conversionFactor = 1.0;
	short selectedUnits = unitsIfKnownInAdvance;

	DateTimeRec time;
	TimeValuePair pair;

	bool askForUnits = true;
	bool isHydrologyFile = false, isLongWindFile = false;
	bool dataInGMT = false;
	
	if ((err = TimeValue_c::InitTimeFunc()) != noErr)
		return err;

	time.second = 0;

	timeValues = 0;
	this->fileName[0] = 0;
	this->filePath[0] = 0;

	if (!path)
		return 0;

	//cerr << "path = " << path << endl;

	strncpy(s, path, 512);
	s[511] = 0;
	strncpy(this->filePath, path, kMaxNameLen);
	this->filePath[kMaxNameLen - 1] = 0;

//#ifndef pyGNOME
	//SplitPathFile(s, this->fileName);
//#else
	// this gives filename but tests expect full path
	SplitPathFileName (s, this->fileName);
	//strcpy(this->fileName, path); // for now use full path
//#endif

	vector<string> linesInFile;
	if (ReadLinesInFile(path, linesInFile)) {
		linesInFile = rtrim_empty_lines(linesInFile);
	}
	else
		return -1; // we failed to read in the file.

	numLines = linesInFile.size();

	if (IsNDBCWindFile(linesInFile, &numHeaderLines)) {
		err = ReadNDBCWind(linesInFile, numHeaderLines); //scan is different
		return err;
		// or
		// selectedUnits = kMetersPerSec;
		// isNDBC = true // scan is different
		// numHeaderLines = 1; (or 2 - format includes minutes)
		// units/format always the same
	}
	
	if (IsNCDCWindFile(linesInFile)) {
		err = ReadNCDCWind(path); 
		return err;
		// or
		// selectedUnits = kMetersPerSec;
		// isNDBC = true // scan is different
		// numHeaderLines = 1;
		// units/format always the same
	}
	
	if( numLines >= 5)
	{
		if (IsLongWindFile(linesInFile, &selectedUnits, &dataInGMT)) {
			askForUnits = false;
			numHeaderLines = 5;
			isLongWindFile = true;
		}
	}
	if(numLines >= 3 && !isLongWindFile)
	{
		if (IsOSSMTimeFile(linesInFile, &selectedUnits)) {
			numHeaderLines = 3;
			ReadOSSMTimeHeader(path);
		}
		else if ((isHydrologyFile = IsHydrologyFile(linesInFile)) == true) {
			// ask for scale factor, but not units
			SetFileType(HYDROLOGYFILE);
			numHeaderLines = 3;
			selectedUnits = kMetersPerSec;	// so conversion factor is 1
		}
	}

	if (selectedUnits == kUndefined )
		askForUnits = TRUE;
	else
		askForUnits = FALSE;
	
#ifdef pyGNOME

	// askForUnits must be FALSE if using pyGNOME
	if (askForUnits) {
		err = 1;	// JS: standard error codes dont exist in C++ gnome
		goto done;
	}

#else
	if (askForUnits) {
		// we have to ask the user for units...
		Boolean userCancel = false;
		selectedUnits = kKnots; // knots will be default
		err = AskUserForUnits(&selectedUnits, &userCancel);
		if (err || userCancel) {
			err = -1;
			goto done;
		}
	}
#endif		

	switch (selectedUnits) {
		case kKnots:
			conversionFactor = KNOTSTOMETERSPERSEC;
			break;
		case kMilesPerHour:
			conversionFactor = MILESTOMETERSPERSEC;
			break;
		case kMetersPerSec:
			conversionFactor = 1.0;
			break;
		default:
			err = -1;
			goto done;
	}

	this->SetUserUnits(selectedUnits);
	
	if (dataInGMT) {
		printError("GMT data is not yet implemented.");
		err = -2;
		goto done;
	}

	// ask for a scale factor
	if (isHydrologyFile) {
		if ((err = ReadHydrologyHeader(path)) != 0)
			goto done;

		if (unitsIfKnownInAdvance != -2) {
			// if not known from wizard message
			this->fScaleFactor = conversionFactor;
		}
	}
	
	numDataLines = numLines - numHeaderLines;
	timeValues = (TimeValuePairH)_NewHandle(numDataLines * sizeof(TimeValuePair));
	if (!timeValues) {
		err = -1;
		TechError("TOSSMTimeValue::ReadTimeValues()", "_NewHandle()", 0);
		goto done;
	}
	
	for (long i = 0; i < numLines; i++)
	{
		if (i % 200 == 0)
			MySpinCursor();

		if (i < numHeaderLines)
			continue; // skip any header lines

		double u, v;
		string value1S, value2S;
		string 	currentLine = trim(linesInFile[i]);

		if (currentLine.size() == 0)
			continue; // it's a blank line, allow this and skip the line

		std::replace(currentLine.begin(), currentLine.end(), ',', ' ');
		
		istringstream lineStream(currentLine);
		lineStream >> time.day >> time.month >> time.year
					   >> time.hour >> time.minute;

		if (lineStream.fail()) {
			// scan will allow comment at end of line, for now just ignore 
			err = -1;
			TechError("TOSSMTimeValue::ReadTimeValues()", "scan date/time", 0);
			goto done;
		}

		// check if last line all zeros (an OSSM requirement) if so ignore the line
		if (i == (numLines - 1) && DateValuesAreZero(time))
			continue;

		if (!DateIsValid(time)) {
			err = -1;
			printError("Invalid data in time file");
			goto done;
		}

		CorrectTwoDigitYear(time);

		lineStream >> value1S >> value2S;
		if (lineStream.fail()) {
			// scan will allow comment at end of line, for now just ignore
			err = -1;
			TechError("TOSSMTimeValue::ReadTimeValues()", "scan data values", 0);
			goto done;
		}

		ConvertRowValuesToUV(value1S, value2S, format, conversionFactor, u, v);

		memset(&pair, 0, sizeof(pair));
		DateToSeconds(&time, &pair.time);

		pair.value.u = u;
		pair.value.v = v;

		if (numValues > 0) {
			Seconds timeVal = INDEXH(timeValues, numValues - 1).time;
			if (pair.time < timeVal) {
				err = -1;
				printError("Time values are out of order");
				goto done;
			}
		}
		
		INDEXH(timeValues, numValues++) = pair;
	}
	
	if (numValues > 0) {
		// JS: 9/17/12 - Following does not work for cython.
		// Leave it commented so we can repro and try to do debugging
		//long actualSize = numValues*(long)sizeof(**timeValues);
		long sz = (long)sizeof(**timeValues);
		long actualSize = numValues * sz;

		_SetHandleSize((Handle)timeValues, actualSize);
		err = _MemError();
	}
	else {
		printError("No lines were found");
		err = true;
	}

done:

	if (err && timeValues) {
		DisposeHandle((Handle)timeValues);
		timeValues = 0;
	}

	return err;
}


/*OSErr OSSMTimeValue_c::ReadHydrologyHeader(char *path)
{
	char	strLine [512];
	char	firstPartOfFile [512];
	OSErr	err = noErr;

	long	line = 0;
	long lenToRead, fileLength, numScanned;
	float latdeg, latmin, longdeg, longmin;

	WorldPoint wp;

	memset(strLine, 0, 512);
	memset(firstPartOfFile, 0, 512);

	err = MyGetFileSize(0, 0, path, &fileLength);
	if (err)
		return err;
	
	lenToRead = _min(512, fileLength);
	
	err = ReadSectionOfFile(0, 0, path, 0, lenToRead, firstPartOfFile, 0);
	if (err)
		return err;
	
	firstPartOfFile[lenToRead - 1] = 0; // make sure it is a cString
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); // station name
	RemoveLeadingAndTrailingWhiteSpace(strLine);

	strncpy(fStationName, strLine, kMaxNameLen);
	fStationName[kMaxNameLen - 1] = 0;
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); // station position - lat deg, lat min, long deg, long min
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	StringSubstitute(strLine, ',', ' ');
	
	numScanned = sscanf(strLine, "%f %f %f %f",
						&latdeg, &latmin, &longdeg, &longmin);
	if (numScanned == 4) {
		// support old OSSM style
		wp.pLat = (latdeg + latmin / 60.) * 1000000;
		// need to have header include direction...
		wp.pLong = -(longdeg + longmin / 60.) * 1000000;
		bOSSMStyle = true;
	}
	else if (numScanned == 2) {
		wp.pLat = latdeg * 1000000;
		wp.pLong = latmin * 1000000;
		bOSSMStyle = false;
	}
	else {
		err = -1;
		TechError("TOSSMTimeValue::ReadHydrologyHeader()", "sscanf() == 2", 0);
		goto done;
	}

	fStationPosition = wp;
	
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); // units
	RemoveLeadingAndTrailingWhiteSpace(strLine);

	if (!strcmpnocase(strLine,"CMS"))
		fUserUnits = kCMS;
	else if (!strcmpnocase(strLine,"KCMS"))
		fUserUnits = kKCMS;
	else if (!strcmpnocase(strLine,"CFS"))
		fUserUnits = kCFS;
	else if (!strcmpnocase(strLine,"KCFS"))
		fUserUnits = kKCFS;
	else
		err = -1;

done:
	return err;
}
*/

OSErr OSSMTimeValue_c::ReadHydrologyHeader(vector<string> &linesInFile)
{
	char	strLine [512];
	char	firstPartOfFile [512], errmsg[256];
	OSErr	err = noErr;
	
	long	line = 0;
	long lenToRead, fileLength, numScanned;
	float latdeg, latmin, longdeg, longmin;
	
	string currentLine;
	char* stationName;

	WorldPoint wp;
	
	memset(strLine, 0, 512);
	memset(firstPartOfFile, 0, 512);
	
	//err = MyGetFileSize(0, 0, path, &fileLength);
	//if (err)
		//return err;
	
	//lenToRead = _min(512, fileLength);
	
	//err = ReadSectionOfFile(0, 0, path, 0, lenToRead, firstPartOfFile, 0);
	//if (err)
		//return err;
	
	currentLine = trim(linesInFile[line++]);
	//firstPartOfFile[lenToRead - 1] = 0; // make sure it is a cString
	//NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); // station name
	//RemoveLeadingAndTrailingWhiteSpace(strLine);
	
	stationName = strdup(currentLine.c_str());	
	strncpy(fStationName,stationName,kMaxNameLen);

	//strncpy(fStationName, strLine, kMaxNameLen);
	//fStationName[kMaxNameLen - 1] = 0;

	currentLine = trim(linesInFile[(line)++]);
	
	std::replace(currentLine.begin(), currentLine.end(), ',', ' ');
	
	istringstream lineStream(currentLine);
	lineStream >> latdeg >> latmin >> longdeg >> longmin;
	if (lineStream.fail()) {
		//sprintf(errmsg, "Unable to read data (ptNum, h, v) from line %ld:\n", *line);
		//goto done;
		istringstream lineStream(currentLine);
		if (lineStream.fail()) {
			sprintf(errmsg, "Unable to read data (lat, lon) from line %ld:\n", line);
			goto done;
		}
		else {
			wp.pLat = latdeg * 1000000;
			wp.pLong = latmin * 1000000;
			bOSSMStyle = false;
		}

	}
	else {
		// support old OSSM style
		wp.pLat = (latdeg + latmin / 60.) * 1000000;
		// need to have header include direction...
		wp.pLong = -(longdeg + longmin / 60.) * 1000000;
		bOSSMStyle = true;
	}

	//NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); // station position - lat deg, lat min, long deg, long min
	//RemoveLeadingAndTrailingWhiteSpace(strLine);
	//StringSubstitute(strLine, ',', ' ');
	
	//numScanned = sscanf(strLine, "%f %f %f %f",
						//&latdeg, &latmin, &longdeg, &longmin);
	/*if (numScanned == 4) {
		// support old OSSM style
		wp.pLat = (latdeg + latmin / 60.) * 1000000;
		// need to have header include direction...
		wp.pLong = -(longdeg + longmin / 60.) * 1000000;
		bOSSMStyle = true;
	}*/
	
	fStationPosition = wp;
	
	currentLine = trim(linesInFile[line++]);
	std::transform(currentLine.begin(),
				   currentLine.end(),
				   currentLine.begin(),
				   ::tolower);

	if (currentLine == "cfs" )
		fUserUnits = kCFS;
	else if (currentLine == "kcfs")
		fUserUnits = kKCFS;
	else if (currentLine == "cms")
		fUserUnits = kCMS;
	else if (currentLine == "kcms")
		fUserUnits = kKCMS;
	else
		err = -1;

	//NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512); // units
	//RemoveLeadingAndTrailingWhiteSpace(strLine);
		
done:
	return err;
}

OSErr OSSMTimeValue_c::ReadHydrologyHeader(char *path)
{
	vector<string> linesInFile;
	
	if (ReadLinesInFile(path, linesInFile)) {
		return ReadHydrologyHeader(linesInFile);
	}
	else {
		return false;
	}
}

OSErr OSSMTimeValue_c::ReadOSSMTimeHeader(char *path)
{
	char strLine[512];
	char firstPartOfFile[512];
	OSErr err = noErr;

	long line = 0;
	long lenToRead, fileLength, numScanned;

	float latdeg, latmin, longdeg, longmin;
	short selectedUnits;

	WorldPoint wp = {0, 0};
	

	memset(strLine, 0, 512);
	memset(firstPartOfFile, 0, 512);

	err = MyGetFileSize(0, 0, path, &fileLength);
	if (err)
		return err;
	
	lenToRead = _min(512, fileLength);
	
	err = ReadSectionOfFile(0, 0, path, 0, lenToRead, firstPartOfFile, 0);
	if (err)
		return err;
	
	firstPartOfFile[lenToRead - 1] = 0; // make sure it is a cString
	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);    // station name
	RemoveLeadingAndTrailingWhiteSpace(strLine);

	strncpy(fStationName, strLine, kMaxNameLen);
	fStationName[kMaxNameLen - 1] = 0;

	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);   // station position - lat deg, lat min, long deg, long min
	RemoveLeadingAndTrailingWhiteSpace(strLine);

	if (!strcmpnocase(strLine, "Station Location"))
		fStationPosition = wp;
		// what to use for default ?
	else {
		StringSubstitute(strLine, ',', ' ');

		numScanned = sscanf(strLine, "%f %f %f %f",
							&latdeg, &latmin, &longdeg, &longmin);
		if (numScanned == 4) {
			// support old OSSM style
			wp.pLat = (latdeg + latmin / 60.) * 1000000;
			wp.pLong = -(longdeg + longmin / 60.) * 1000000;
		}
		else if (numScanned == 2) {
			wp.pLat = latdeg * 1000000;
			wp.pLong = latmin * 1000000;
		}
		else {
			wp.pLat = 0;
			wp.pLong = 0;
		}

		fStationPosition = wp;
	}

	NthLineInTextOptimized(firstPartOfFile, line++, strLine, 512);   // units
	RemoveLeadingAndTrailingWhiteSpace(strLine);
	selectedUnits = (short)StrToSpeedUnits(strLine);
	// note: we are not supporting cm/sec in gnome

	if(selectedUnits == kUndefined)
		err = -1;
	else
		fUserUnits = selectedUnits;

done:
	return err;
}


OSErr OSSMTimeValue_c::ConvertRowValuesToUV(string &value1, string &value2,
											short format, double conversionFactor,
											double &uOut, double &vOut)
{
	double magnitude, degrees;
	istringstream v1Stream(value1);
	istringstream v2Stream(value2);

	switch (format) {
		case M19REALREAL:
			// no UV conversion necessary, just load the values and return
			v1Stream >> uOut;
			v2Stream >> vOut;

			uOut *= conversionFactor;
			vOut *= conversionFactor;
			return noErr;
			break;
		case M19MAGNITUDEDEGREES:
			v1Stream >> magnitude;
			v2Stream >> degrees;

			magnitude *= conversionFactor;
			break;
		case M19DEGREESMAGNITUDE:
			v1Stream >> degrees;
			v2Stream >> magnitude;

			magnitude *= conversionFactor;
			break;
		case M19MAGNITUDEDIRECTION:
			v1Stream >> magnitude;
			degrees = ConvertToDegrees((char *)value2.c_str());

			magnitude *= conversionFactor;
			break;
		case M19DIRECTIONMAGNITUDE:
			v2Stream >> magnitude;
			degrees = ConvertToDegrees((char *)value1.c_str());

			magnitude*= conversionFactor;
			break;
		default:
			return -1;
	}

	ConvertToUV(magnitude, degrees, &uOut, &vOut);

	return noErr;
}


bool OSSMTimeValue_c::DateValuesAreZero(DateTimeRec &dateTime)
{
	return 	(dateTime.day == 0 && dateTime.month == 0 && dateTime.year == 0 &&
			dateTime.hour == 0 && dateTime.minute == 0);

}


bool OSSMTimeValue_c::DateIsValid(DateTimeRec &dateTime)
{
	return (dateTime.day >= 1 && dateTime.day <= 31 &&
			dateTime.month >= 1 && dateTime.month <= 12);
}


void OSSMTimeValue_c::CorrectTwoDigitYear(DateTimeRec &dateTime)
{
	if (dateTime.year < 1900) {
		// two digit date, so fix it
		if (dateTime.year >= 40 && dateTime.year <= 99)
			dateTime.year += 1900;
		else
			dateTime.year += 2000; // correct for year 2000 (00 to 40)
	}
}


