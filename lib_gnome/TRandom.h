/*
 *  TRandom.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TRandom__
#define __TRandom__

#include "Random_c.h"
#include "Random_g.h"
#include "TMover.h"

class TRandom : virtual public Random_c, virtual public Random_g, virtual public TMover
{
	
public:
	TRandom (TMap *owner, char *name);
	
};

#endif