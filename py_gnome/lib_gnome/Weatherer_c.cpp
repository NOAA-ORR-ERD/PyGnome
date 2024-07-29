/*
 *  Weatherer_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Weatherer_c.h"

Weatherer_c::Weatherer_c(char *name)
{
	SetClassName (name);
	
	bActive = TRUE;
	bOpen = TRUE;
}