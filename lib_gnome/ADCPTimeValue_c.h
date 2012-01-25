/*
 *  ADCPTimeValue_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPTimeValue_c__
#define __ADCPTimeValue_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "ADCPTimeValue_b.h"
#include "TimeValue/TimeValue_c.h"

class ADCPTimeValue_c : virtual public ADCPTimeValue_b, virtual public TimeValue_c {

public:
	virtual OSErr			GetTimeValue (Seconds time, VelocityRec *value);
	virtual OSErr			GetTimeValueAtDepth (long depthIndex, Seconds time, VelocityRec *value);
	double					GetBinDepth(long depthIndex);
	virtual OSErr			CheckStartTime (Seconds time);
	virtual void			RescaleTimeValues (double oldScaleFactor, double newScaleFactor);
	OSErr					GetDepthIndices(float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2);
	virtual long			GetNumValues ();
	virtual TimeValuePairH3D	GetTimeValueHandle () { return timeValues; }
	virtual void			SetTimeValueHandle (TimeValuePairH3D t) ;	
	virtual double			GetMaxValue();

protected:
	OSErr					GetInterpolatedComponent (Seconds forTime, double *value, short index);
	OSErr					GetInterpolatedComponentAtDepth (long depthIndex, Seconds forTime, double *value, short index);
	OSErr					GetTimeChange (long a, long b, Seconds *dt);
	

};


#endif
