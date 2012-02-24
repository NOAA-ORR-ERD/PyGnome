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

#include "Earl.h"
#include "TypeDefs.h"
#include "TimeValue/TimeValue_c.h"

#ifdef pyGNOME
#define TMover Mover_c
#endif

class OSSMTimeValue_c : virtual public TimeValue_c {
	
public:
	
	TimeValuePairH			timeValues;
	char 					fileName [kMaxNameLen];
	short					fFileType; //JLM
	short					fUserUnits; //JLM
	double					fScaleFactor; // user input for scaling height derivatives or hydrology files
	char 					fStationName [kMaxNameLen];
	WorldPoint				fStationPosition;
	Boolean					bOSSMStyle;
	double					fTransport;
	double					fVelAtRefPt;
		
	virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
	virtual OSErr			CheckStartTime (Seconds time);
	virtual void			RescaleTimeValues (double oldScaleFactor, double newScaleFactor);
	virtual long			GetNumValues ();
	virtual TimeValuePairH	GetTimeValueHandle () { return timeValues; }
	virtual void			SetTimeValueHandle (TimeValuePairH t) ;
	
	virtual short			GetUserUnits(){return fUserUnits;}
	virtual void			SetUserUnits(short userUnits){fUserUnits=userUnits;}
	virtual double			GetMaxValue();

public:
	OSSMTimeValue_c () {}
	OSSMTimeValue_c (TMover *theOwner);
	OSSMTimeValue_c (TMover *theOwner,TimeValuePairH tvals,short userUnits);
	virtual ClassID 		GetClassID () { return TYPE_OSSMTIMEVALUES; }
	virtual Boolean			IAm(ClassID id) { if(id==TYPE_OSSMTIMEVALUES) return TRUE; return TimeValue_c::IAm(id); }
	
	OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr					GetTimeChange (long a, long b, Seconds *dt);
	virtual void			Dispose();

	virtual OSErr			InitTimeFunc ();
	virtual short			GetFileType	() { if (fFileType == PROGRESSIVETIDEFILE) return SHIOHEIGHTSFILE; else return fFileType; }
	virtual void			SetFileType	(short fileType) { fFileType = fileType; }

};

#undef TMover
#endif