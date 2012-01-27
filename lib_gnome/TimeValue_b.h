/*
 *  TimeValue_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TimeValue_b__
#define __TimeValue_b__

#include "ClassID_b.h"

#ifdef pyGNOME
#define TMover Mover_c
#endif

class TMover;

class TimeValue_b : virtual public ClassID_b {

public:
	TMover *owner;
	
};

#undef TMover
#endif
