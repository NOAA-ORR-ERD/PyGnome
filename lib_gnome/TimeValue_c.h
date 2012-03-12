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

#include "Basics.h"
#include "TypeDefs.h"
#include "TimeValue_b.h"
#include "ClassID_c.h"

#ifdef pyGNOME
#define TMover Mover_c
#endif

class TimeValue_c : virtual public TimeValue_b, virtual public ClassID_c {
	
public:
	TimeValue_c (TMover *theOwner) { owner = theOwner; }
	TimeValue_c () {}
	virtual OSErr	GetTimeValue (Seconds time, VelocityRec *value);
	virtual OSErr	CheckStartTime (Seconds time);
	virtual void	Dispose () {}

};

#undef TMover
#endif
