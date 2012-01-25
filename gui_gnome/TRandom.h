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
#include "Mover/TMover.h"

class TRandom : virtual public Random_c, virtual public TMover
{
	
public:
	TRandom (TMap *owner, char *name);
	virtual ClassID 	GetClassID () { return TYPE_RANDOMMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMMOVER) return TRUE; return TMover::IAm(id); }
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	virtual void		Draw(Rect r, WorldRect view) { }
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
	//		virtual OSErr 		AddItem(ListItem item);
	virtual OSErr 		SettingsItem(ListItem item);
	virtual OSErr 		DeleteItem(ListItem item);
	
};

#endif
