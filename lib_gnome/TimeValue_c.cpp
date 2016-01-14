/*
 *  TimeValue_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TimeValue_c.h"

OSErr TimeValue_c::GetTimeValue(const Seconds& forTime, VelocityRec *value)
{
	value -> u = 1.0;
	value -> v = 1.0;
	
	return 0;
}

OSErr TimeValue_c::CheckStartTime(Seconds forTime)
{	
	return 0;
}

OSErr TimeValue_c::InitTimeFunc()
{
	return 0;
}