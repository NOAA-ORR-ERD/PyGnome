/*
 *  TClassID.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TClassID__
#define __TClassID__

#include "Earl.h"
#include "TypeDefs.h"
#include "ClassID_c.h"
#include "ClassID_g.h"

class TClassID : virtual public ClassID_c, virtual public ClassID_g 
{
	
public:
	TClassID ();
	virtual			   ~TClassID () { Dispose (); }
	virtual void		Dispose 	() { return; }
	
	
};

#endif