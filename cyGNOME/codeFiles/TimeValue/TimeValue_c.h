/*
 *  TimeValue_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TimeValue_c__
#define __TimeValue_c__

#include "Earl.h"
#include "TypeDefs.h"
#include "TimeValue_b.h"
#include "ClassID/ClassID_c.h"

class TimeValue_c : virtual public TimeValue_b, virtual public ClassID_c {
	
public:
	virtual OSErr	GetTimeValue (Seconds time, VelocityRec *value);
	virtual OSErr	CheckStartTime (Seconds time);
	
};

#endif