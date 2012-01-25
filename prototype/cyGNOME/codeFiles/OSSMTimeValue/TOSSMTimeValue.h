/*
 *  TOSSMTimeValue.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TOSSMTimeValue__
#define __TOSSMTimeValue__

#include "Earl.h"
#include "TypeDefs.h"
#include "OSSMTimeValue_c.h"
#include "OSSMTimeValue_g.h"
#include "TimeValue/TTimeValue.h"
#include "OUTILS.H"

class TOSSMTimeValue : virtual public OSSMTimeValue_c, virtual public OSSMTimeValue_g, virtual public TTimeValue
{

public:
	TOSSMTimeValue (TMover *theOwner);
	TOSSMTimeValue (TMover *theOwner,TimeValuePairH tvals,short userUnits);
	virtual				   ~TOSSMTimeValue () { Dispose (); }
	virtual void			Dispose ();

};

#endif
