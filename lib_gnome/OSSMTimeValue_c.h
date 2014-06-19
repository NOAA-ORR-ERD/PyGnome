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
	
	TimeValuePairH			timeValues;
	char 					fileName [kMaxNameLen];
	char 					filePath [kMaxNameLen];
	short					fFileType; //JLM
	short					fUserUnits; //JLM
	double					fScaleFactor; // user input for scaling height derivatives or hydrology files
	char 					fStationName [kMaxNameLen];
	WorldPoint				fStationPosition;
	Boolean					bOSSMStyle;
	double					fTransport;
	double					fVelAtRefPt;
	short					fInterpolationType;
	
	virtual void 			GetTimeFileName (char *theName) { strcpy (theName, fileName); }
	virtual short			GetFileType	() { if (fFileType == PROGRESSIVETIDEFILE) return SHIOHEIGHTSFILE; else return fFileType; }
	virtual void			SetFileType	(short fileType) { fFileType = fileType; }
	
#ifndef pyGNOME
	OSSMTimeValue_c (TMover *theOwner);
	OSSMTimeValue_c (TMover *theOwner,TimeValuePairH tvals,short userUnits);
#endif
	OSSMTimeValue_c ();
	virtual				   ~OSSMTimeValue_c ();

	//virtual ClassID 		GetClassID () { return TYPE_OSSMTIMEVALUES; }
	//virtual Boolean			IAm(ClassID id) { if(id==TYPE_OSSMTIMEVALUES) return TRUE; return TimeValue_c::IAm(id); }
	
	virtual void			Dispose ();
	virtual OSErr			GetTimeValue(const Seconds& current_time, VelocityRec *value);
	virtual OSErr			CheckStartTime (Seconds time);
	virtual void			RescaleTimeValues (double oldScaleFactor, double newScaleFactor);
	virtual long			GetNumValues ();
	virtual TimeValuePairH	GetTimeValueHandle () { return timeValues; }
	virtual void			SetTimeValueHandle (TimeValuePairH t) ;
	
	virtual WorldPoint		GetStationLocation (void) {return fStationPosition;}
	virtual short			GetUserUnits(){return fUserUnits;}
	virtual void			SetUserUnits(short userUnits){fUserUnits=userUnits;}
	virtual double			GetMaxValue();
	virtual OSErr			InitTimeFunc ();

	virtual OSErr ReadNDBCWind(vector<string> &linesInFile, long numHeaderLines);
	virtual OSErr ReadNDBCWind(char *path, long numHeaderLines);

	virtual OSErr			ReadNCDCWind (char *path);
	virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);
	OSErr					ReadOSSMTimeHeader (char *path);
	OSErr					ReadHydrologyHeader (vector<string> &linesInFile);
	OSErr					ReadHydrologyHeader (char *path);

	virtual OSErr 			GetLocationInTideCycle(const Seconds& model_time, short *ebbFloodType, float *fraction) {*ebbFloodType=0; *fraction=0; return 0;}
	
protected:
	OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr					GetTimeChange (long a, long b, Seconds *dt);

	OSErr ConvertRowValuesToUV(string &value1, string &value2,
							   short format, double conversionFactor,
							   double &uOut, double &vOut);
	bool DateValuesAreZero(DateTimeRec &dateTime);
	bool DateIsValid(DateTimeRec &dateTime);
	void CorrectTwoDigitYear(DateTimeRec &dateTime);


};

#endif
