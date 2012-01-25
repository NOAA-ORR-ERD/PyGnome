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
#include "TImeValue_g.h"
#include "TClassID.h"

class TMover;
class TTimeValue : virtual public TimeValue_c, virtual public TimeValue_g, virtual public TClassID
{
	
public:
	TTimeValue() {}
	TTimeValue (TMover *theOwner) { owner = theOwner; }
	virtual		   ~TTimeValue () { Dispose (); }
	virtual void	Dispose () {}
	
};

#endif