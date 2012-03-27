/*
 *  ClassID_b.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ClassID_b__
#define __ClassID_b__

#include "Basics.h"
#include "TypeDefs.h"

class ClassID_b {

public:
	Boolean				bDirty;
	Boolean				bOpen;
	Boolean				bActive;
	char				className [kMaxNameLen];
	UNIQUEID			fUniqueID;

};

#endif
