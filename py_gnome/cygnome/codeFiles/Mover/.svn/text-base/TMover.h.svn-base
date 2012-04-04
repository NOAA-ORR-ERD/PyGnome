/*
 *  TMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TMover__
#define __TMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "Mover_g.h"
#include "TClassID.h"

class TMap;

class TMover : virtual public Mover_c, virtual public Mover_g, virtual public TClassID
{
	
public:
	TMover() {}
	TMover (TMap *owner, char *name);
	virtual			   ~TMover () { Dispose (); }
	virtual void		Dispose ();
		
};


#endif