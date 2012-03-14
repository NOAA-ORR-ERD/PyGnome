/*
 *  TTimeValue.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TTimeValue__
#define __TTimeValue__

#include "TimeValue_c.h"
#include "TClassID.h"

class TMover;
class TTimeValue : virtual public TimeValue_c,  public TClassID
{
	
public:
	TTimeValue (TMover *theOwner) { owner = theOwner; }
	virtual		   ~TTimeValue () { Dispose (); }

	virtual OSErr 	MakeClone(TTimeValue **clonePtrPtr);
	virtual OSErr 	BecomeClone(TTimeValue *clone);
	
	// I/O methods
	virtual OSErr 	Read  (BFPB *bfpb); // read from current position
	virtual OSErr 	Write (BFPB *bfpb); // write to  current position
	
};

#endif
