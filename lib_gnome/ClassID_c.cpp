/*
 *  ClassID_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ClassID_c.h"
#include "StringFunctions.h"
#include "CompFunctions.h"

ClassID_c::ClassID_c()
{
	// initialize
	bActive = 1;	// active by default
}

void ClassID_c::SetClassName (char *newName)
{
	if (strlen (newName) > kMaxNameLen)
		newName [kMaxNameLen - 1] = 0;
	
	strnzcpy (className, newName, kMaxNameLen - 1);
}


Boolean ClassID_c::MatchesUniqueID(UNIQUEID uid)
{
	return EqualUniqueIDs(uid,this->fUniqueID);
}
