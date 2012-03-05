/*
 *  ShioTimeValue_c.h
 *  c_gnome
 *
 *  Created by Alex Hadjilambris on 2/24/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ShioTimeValue_c__
#define __ShioTimeValue_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "OSSMTimeValue/OSSMTimeValue_c.h"
#include "Shio.h"

typedef struct
{
	Seconds time;
	double height;
	short type;	// 0 -> Low Tide, 1 -> High Tide
} HighLowData,*HighLowDataP,**HighLowDataH;

typedef struct
{
	Seconds time;
	double speedInKnots;
	short type;	// 0 -> MinBeforeFlood, 1 -> MaxFlood, 2 -> MinBeforeEbb, 3 -> MaxEbb
} EbbFloodData,*EbbFloodDataP,**EbbFloodDataH;

#define MAXSTATIONNAMELEN  128
#define kMaxKeyedLineLength	1024

class ShioTimeValue_c : virtual public OSSMTimeValue_c {

public:
	
	// instance variables
	char fStationName[MAXSTATIONNAMELEN];
	char fStationType;
	double fLatitude;
	double fLongitude;
	CONSTITUENT fConstituent;
	HEIGHTOFFSET fHeightOffset;
	CURRENTOFFSET fCurrentOffset;
	//
	Boolean fHighLowValuesOpen; // for the list
	Boolean fEbbFloodValuesOpen; // for the list
	EbbFloodDataH fEbbFloodDataHdl;	// values to show on list for tidal currents
	HighLowDataH fHighLowDataHdl;	// values to show on list for tidal heights

	ShioTimeValue_c(Seconds start_time, Seconds stop_time) { 
		fEbbFloodDataHdl = 0;
		fHighLowDataHdl = 0;
		this->start_time = start_time;
		this->stop_time = stop_time;
	}
	
	virtual ClassID 		GetClassID () { return TYPE_SHIOTIMEVALUES; }
	virtual Boolean			IAm(ClassID id) { if(id==TYPE_SHIOTIMEVALUES) return TRUE; return OSSMTimeValue_c::IAm(id); }
	virtual long			GetNumEbbFloodValues ();	
	virtual long			GetNumHighLowValues ();
	virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
	virtual WorldPoint	GetRefWorldPoint (void);
	
	virtual	double 		GetDeriv (Seconds t1, double val1, Seconds t2, double val2, Seconds theTime);
	virtual	OSErr 		GetConvertedHeightValue(Seconds forTime, VelocityRec *value);
	virtual	OSErr 		GetProgressiveWaveValue(Seconds forTime, VelocityRec *value);
	OSErr 					GetLocationInTideCycle(short *ebbFloodType, float *fraction);

	OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float *** val);
	OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,DATA * val);
	OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,short * val);
	OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,float * val);
	OSErr 				GetKeyedValue(CHARH f, char*key, long lineNum, char* strLine,double * val);
	OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr					GetTimeChange (long a, long b, Seconds *dt);
	void 					ProgrammerError(char* routine);
	virtual OSErr			ReadTimeValues (char *path, short format, short unitsIfKnownInAdvance);

};


#endif