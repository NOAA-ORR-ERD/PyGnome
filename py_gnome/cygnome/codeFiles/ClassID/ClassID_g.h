/*
 *  ClassID_g.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ClassID_g__
#define __ClassID_g__

#include "Earl.h"
#include "TypeDefs.h"
#include "ClassID_b.h"

class TModelMessage;

class ClassID_g : virtual public ClassID_b {

public:
	ClassID_g();
	virtual ClassID 	GetClassID 	() { return TYPE_UNDENTIFIED; }
	virtual Boolean		IAm(ClassID id) { return FALSE; }
	
	void				GetClassName (char* theName) { strcpy (theName, className); }	// sohail
	void				SetClassName (char* name);
	
	UNIQUEID			GetUniqueID () { return fUniqueID; }
	Boolean 			MatchesUniqueID(UNIQUEID uid);
	
	Boolean 			GetSelectedListItem(ListItem *item);
	Boolean 			SelectedListItemIsMine(void);
	virtual Boolean 	IAmEditableInMapDrawingRect(void);
	virtual Boolean 	IAmCurrentlyEditableInMapDrawingRect(void);
	virtual Boolean 	UserIsEditingMeInMapDrawingRect(void);
	virtual void	 	StartEditingInMapDrawingRect(void);
	virtual OSErr 		StopEditingInMapDrawingRect(Boolean *deleteMe);
	
	virtual OSErr 		MakeClone(TClassID **clonePtrPtr);
	virtual OSErr 		BecomeClone(TClassID *clone);
	
	virtual Boolean		IsDirty  	() { return bDirty;  }
	virtual Boolean		IsOpen   	() { return bOpen;   }
	virtual Boolean		IsActive 	() { return bActive; }
	virtual void		SetDirty  (Boolean bNewDirty)  { bDirty  = bNewDirty; }
	virtual void		SetOpen   (Boolean bNewOpen)   { bOpen   = bNewOpen;  }
	virtual void		SetActive (Boolean bNewActive) { bActive = bNewActive;}
	
	virtual OSErr 		Read  (BFPB *bfpb);  			
	virtual OSErr 		Write (BFPB *bfpb); 			
	
	virtual long 		GetListLength 	();				 
	virtual Boolean 	ListClick 	  	(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID) { return FALSE; }
	virtual ListItem 	GetNthListItem 	(long n, short indent, short *style, char *text);
	virtual OSErr 		UpItem 			(ListItem item) { return 0; }
	virtual OSErr 		DownItem 		(ListItem item) { return 0; }
	virtual OSErr 		AddItem 		(ListItem item) { return 0; }
	virtual OSErr 		SettingsItem 	(ListItem item) { return 0; }
	virtual OSErr 		DeleteItem 		(ListItem item) { return 0; }
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);
	
};

#endif