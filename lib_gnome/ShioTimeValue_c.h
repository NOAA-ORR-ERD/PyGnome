/*
 *  ShioTimeValue_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ShioTimeValue_c__
#define __ShioTimeValue_c__

#include "Shio.h"
#include "OSSMTimeValue_c.h"
#include "ExportSymbols.h"

#define MAXNUMSHIOYEARS  20
#define MAXSTATIONNAMELEN  128
#define kMaxKeyedLineLength	1024

#ifndef pyGNOME
#include "TMover.h"
#endif

#define kMAXNUMSAVEDYEARS 30
typedef struct
{
	Seconds time;
	double speedInKnots;
	short type;	// 0 -> MinBeforeFlood, 1 -> MaxFlood, 2 -> MinBeforeEbb, 3 -> MaxEbb
} EbbFloodData,*EbbFloodDataP,**EbbFloodDataH;

typedef struct
{
	Seconds time;
	double height;
	short type;	// 0 -> Low Tide, 1 -> High Tide
} HighLowData,*HighLowDataP,**HighLowDataH;

YEARDATAHDL GetYearData(short year);
YEARDATA2* ReadYearData(short year, const char *path, char *errStr);

class DLL_API ShioTimeValue_c : virtual public OSSMTimeValue_c {

protected:

	// instance variables
	double fLatitude;
	double fLongitude;
	CONSTITUENT2 fConstituent;
	HEIGHTOFFSET fHeightOffset;
	CURRENTOFFSET fCurrentOffset;
	//
	Boolean fHighLowValuesOpen; // for the list
	Boolean fEbbFloodValuesOpen; // for the list
	
	OSErr		GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, float ***val);
	OSErr 		GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, DATA *val);
	OSErr 		GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, short *val);
	OSErr 		GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, float *val);
	OSErr 		GetKeyedValue(CHARH f, const char *key, long lineNum, char *strLine, double *val);
	OSErr		GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr		GetTimeChange (long a, long b, Seconds *dt);
	
	void 		ProgrammerError(const char *routine);
	void 		InitInstanceVariables(void);
	
	long 		I_SHIOHIGHLOWS(void);
	long 		I_SHIOEBBFLOODS(void);
	
public:						
	char					fStationName[MAXSTATIONNAMELEN];
	char					fStationType;
	bool					daylight_savings_off;	// AH 07/09/2012
	EbbFloodDataH			fEbbFloodDataHdl;	// values to show on list for tidal currents
	HighLowDataH			fHighLowDataHdl;	// values to show on list for tidal heights
	char					fYearDataPath[kMaxNameLen];

							ShioTimeValue_c ();
#ifndef pyGNOME
							ShioTimeValue_c (TMover *theOwner);
							ShioTimeValue_c (TMover *theOwner,TimeValuePairH tvals);
#endif
	virtual					 ~ShioTimeValue_c () { this->Dispose (); }
	virtual void			Dispose ();
	//virtual ClassID 		GetClassID () { return TYPE_SHIOTIMEVALUES; }
	//virtual Boolean			IAm(ClassID id) { if(id==TYPE_SHIOTIMEVALUES) return TRUE; return OSSMTimeValue_c::IAm(id); }
	virtual OSErr			ReadTimeValues (char *path);
	virtual long			GetNumEbbFloodValues ();	
	virtual long			GetNumHighLowValues ();
	virtual OSErr			GetTimeValue(const Seconds& current_time, VelocityRec *value);
	virtual WorldPoint		GetStationLocation (void);
	
	virtual	double			GetDeriv (Seconds t1, double val1, Seconds t2, double val2, Seconds theTime);
	virtual	OSErr			GetConvertedHeightValue(Seconds forTime, VelocityRec *value);
	virtual	OSErr			GetProgressiveWaveValue(const Seconds& current_time, VelocityRec *value);
	virtual OSErr 			GetLocationInTideCycle(const Seconds& model_time, short *ebbFloodType, float *fraction);

	virtual OSErr			InitTimeFunc ();
			Boolean			DaylightSavingTimeInEffect(DateTimeRec *dateStdTime);	// AH 07/09/2012
	
	OSErr					SetYearDataPath(char *pathName);
	void					GetYearDataPath(char *pathName);
	void					GetYearDataDirectory(char* directoryPath);
	
};


//#undef TMover
#endif
