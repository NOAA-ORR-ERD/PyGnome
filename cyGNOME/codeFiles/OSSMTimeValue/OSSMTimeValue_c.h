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
#include "OSSMTimeValue_b.h"
#include "TimeValue/TimeValue_c.h"

class OSSMTimeValue_c : virtual public OSSMTimeValue_b, virtual public TimeValue_c {

public:
	
	virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
	virtual OSErr			CheckStartTime (Seconds time);
	virtual void			RescaleTimeValues (double oldScaleFactor, double newScaleFactor);
	virtual long			GetNumValues ();
	virtual TimeValuePairH	GetTimeValueHandle () { return timeValues; }
	virtual void			SetTimeValueHandle (TimeValuePairH t) ;
	
	virtual short			GetUserUnits(){return fUserUnits;}
	virtual void			SetUserUnits(short userUnits){fUserUnits=userUnits;}
	virtual double			GetMaxValue();
	
protected:
	OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr					GetTimeChange (long a, long b, Seconds *dt);

};


#endif