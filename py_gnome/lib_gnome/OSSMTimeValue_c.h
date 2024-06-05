/*
 *  OSSMTimeValue_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __OSSMTimeValue_c__
#define __OSSMTimeValue_c__

#include <vector>
#include <sstream>
#include <string>

#include "Basics.h"
#include "TypeDefs.h"
#include "TimeValue_c.h"
#include "ExportSymbols.h"

using namespace std;

class DLL_API OSSMTimeValue_c : virtual public TimeValue_c {

public:
	char fileName [kMaxNameLen];
	char filePath [kMaxNameLen];
	short fFileType; //JLM

	TimeValuePairH timeValues;
	short fUserUnits; //JLM
	double fScaleFactor; // user input for scaling height derivatives or hydrology files

	char fStationName [kMaxNameLen];
	WorldPoint3D fStationPosition;

	Boolean bOSSMStyle;
	Boolean extrapolationIsAllowed;

	double fTransport;
	double fVelAtRefPt;
	short fInterpolationType;

	virtual void GetTimeFileName(char *theName) { strcpy (theName, fileName); }
	virtual short GetFileType(){ if (fFileType == PROGRESSIVETIDEFILE) return SHIOHEIGHTSFILE; else return fFileType; }
	virtual void SetFileType(short fileType) { fFileType = fileType; }

#ifndef pyGNOME
	OSSMTimeValue_c (TMover *theOwner);
	OSSMTimeValue_c (TMover *theOwner, TimeValuePairH tvals, short userUnits);
#endif

	OSSMTimeValue_c();
	virtual ~OSSMTimeValue_c();

	virtual OSErr InitTimeFunc();
	virtual void Dispose();

	virtual OSErr GetTimeValue(const Seconds& current_time, VelocityRec *value);
	Seconds ClampToTimeRange(const Seconds time);

	virtual TimeValuePairH GetTimeValueHandle() { return timeValues; }
	virtual void SetTimeValueHandle(TimeValuePairH t);

	virtual void RescaleTimeValues(double oldScaleFactor, double newScaleFactor);

	OSErr GetDataStartTime(Seconds *startTime);	
	OSErr GetDataEndTime(Seconds *endTime);	
	virtual OSErr CheckStartTime(Seconds time);

	virtual long GetNumValues();
	virtual double GetMaxValue();

	virtual short GetUserUnits() {return fUserUnits;}
	virtual void SetUserUnits(short userUnits) {fUserUnits = userUnits;}

	virtual WorldPoint3D	GetStationLocation(void) {return fStationPosition;}

	virtual OSErr ReadNCDCWind(char *path);
	virtual OSErr ReadNDBCWind(vector<string> &linesInFile, long numHeaderLines);
	virtual OSErr ReadNDBCWind(char *path, long numHeaderLines);
	virtual OSErr ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);

	OSErr ReadOSSMTimeHeader(char *path);
	OSErr ReadHydrologyHeader(vector<string> &linesInFile);
	OSErr ReadHydrologyHeader(char *path);

	virtual OSErr GetLocationInTideCycle(const Seconds& model_time, short *ebbFloodType, float *fraction) {*ebbFloodType = 0; *fraction = 0; return 0;}
	TimeValuePairH CalculateRunningAverage(long pastHoursToAverage, Seconds model_time);
	
protected:
	OSErr GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr GetTimeChange (long a, long b, Seconds *dt);

	OSErr ConvertRowValuesToUV(string &value1, string &value2,
							  short format, double conversionFactor,
							  double &uOut, double &vOut);

	void CorrectTwoDigitYear(DateTimeRec &dateTime);

	bool DateValuesAreZero(DateTimeRec &dateTime);
	bool DateIsValid(DateTimeRec &dateTime);
};

#endif
