/*
 *  TComponentMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TComponentMover__
#define __TComponentMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "ComponentMover_c.h"
#include "TCurrentMover.h"

class TComponentMover : virtual public ComponentMover_c,  public TCurrentMover
{
	
public:
	
	TComponentMover (TMap *owner, char *name);
	virtual			   ~TComponentMover () { Dispose (); }
	//virtual void 		Dispose ();
	virtual OSErr		InitMover ();
	virtual ClassID 	GetClassID () { return TYPE_COMPONENTMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_COMPONENTMOVER) return TRUE; return TCurrentMover::IAm(id); }
	virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); 	// write to  current position
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	// list display methods
	virtual void		Draw(Rect r, WorldRect view);
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
	//		virtual OSErr 		AddItem(ListItem item);
	virtual OSErr 		SettingsItem(ListItem item);
	virtual OSErr 		DeleteItem(ListItem item);
	
};

#endif
