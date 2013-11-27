/*
 *  TClassID.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TClassID__
#define __TClassID__

#include "Earl.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "ClassID_c.h"


class TModelMessage;

class TClassID : virtual public ClassID_c
{
	
public:
	UNIQUEID			fUniqueID;

	TClassID ();
	virtual			   ~TClassID () { Dispose (); }
	
	virtual ClassID 	GetClassID 	() { return TYPE_UNDENTIFIED; }
	virtual Boolean		IAm(ClassID id) { return FALSE; }
	Boolean 			GetSelectedListItem(ListItem *item);
	Boolean 			SelectedListItemIsMine(void);
	virtual Boolean 	IAmEditableInMapDrawingRect(void);
	virtual Boolean 	IAmCurrentlyEditableInMapDrawingRect(void);
	virtual Boolean 	UserIsEditingMeInMapDrawingRect(void);
	virtual void	 	StartEditingInMapDrawingRect(void);
	virtual OSErr 		StopEditingInMapDrawingRect(Boolean *deleteMe);
	
	virtual OSErr 		MakeClone(TClassID **clonePtrPtr);
	virtual OSErr 		BecomeClone(TClassID *clone);
	
	UNIQUEID			GetUniqueID () { return fUniqueID; }
	Boolean 			MatchesUniqueID(UNIQUEID uid);	
	
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
