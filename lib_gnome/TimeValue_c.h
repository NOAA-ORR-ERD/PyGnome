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
#include "ClassID_c.h"
#include "ExportSymbols.h"

#ifndef pyGNOME
class TMover;
#endif

class DLL_API TimeValue_c : virtual public ClassID_c {
	
public:
#ifndef pyGNOME
	TMover *owner;
	TimeValue_c (TMover *theOwner) { owner = theOwner; }
#endif
	TimeValue_c () {}
	virtual				   ~TimeValue_c () { Dispose (); }
	//virtual ClassID GetClassID () { return TYPE_TIMEVALUES; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEVALUES) return TRUE; return ClassID_c::IAm(id); }
	virtual OSErr   GetTimeValue(const Seconds& current_time, VelocityRec *value);
	virtual OSErr	CheckStartTime (Seconds time);
	virtual void	Dispose () {}
	virtual OSErr	InitTimeFunc ();

};

#endif
