/*
 *  ClassID_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ClassID_c__
#define __ClassID_c__

#include "Earl.h"
#include "TypeDefs.h"

class ClassID_c {

public:
	Boolean				bDirty;
	Boolean				bOpen;
	Boolean				bActive;
	char				className [kMaxNameLen];
	UNIQUEID			fUniqueID;
	virtual void		Dispose 	() { return; }
	virtual ClassID 	GetClassID 	() { return TYPE_UNDENTIFIED; }
	virtual Boolean		IAm(ClassID id) { return FALSE; }	

};


#endif