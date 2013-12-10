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

#include "Basics.h"
#include "TypeDefs.h"
#include "ExportSymbols.h"


class DLL_API ClassID_c {

public:
	Boolean				bDirty;
	Boolean				bOpen;
	Boolean				bActive;
	char				className [kMaxNameLen];
	//UNIQUEID			fUniqueID;
	
						ClassID_c ();
	virtual			   ~ClassID_c () { Dispose (); }
	//virtual ClassID 	GetClassID 	() { return TYPE_UNDENTIFIED; }
	//virtual Boolean		IAm(ClassID id) { return FALSE; }
	void				GetClassName (char* theName) { strcpy (theName, className); }	// sohail
	void				SetClassName (char* name);
	//UNIQUEID			GetUniqueID () { return fUniqueID; }
	//Boolean 			MatchesUniqueID(UNIQUEID uid);	
	virtual void		Dispose 	() { return; }
	virtual Boolean		IsDirty  	() { return bDirty;  }
	virtual Boolean		IsOpen   	() { return bOpen;   }
	virtual Boolean		IsActive 	() { return bActive; }
	virtual void		SetDirty  (Boolean bNewDirty)  { bDirty  = bNewDirty; }
	virtual void		SetOpen   (Boolean bNewOpen)   { bOpen   = bNewOpen;  }
	virtual void		SetActive (Boolean bNewActive) { bActive = bNewActive;}
	
};


#endif
