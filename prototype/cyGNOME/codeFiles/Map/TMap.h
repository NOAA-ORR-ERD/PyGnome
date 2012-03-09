/*
 *  TMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TMap__
#define __TMap__

#include "Earl.h"
#include "TypeDefs.h"
#include "Map_c.h"
#include "Map_g.h"
#include "ClassID/TClassID.h"

class TMap : virtual public Map_c, virtual public Map_g, virtual public TClassID
{
	
public:
	// map methods
	TMap() {}
	TMap (char *name, WorldRect bounds);
	virtual			   ~TMap () { Dispose (); }
	virtual void		Dispose ();		
	virtual OSErr		InitMap ();

		
};


#endif
