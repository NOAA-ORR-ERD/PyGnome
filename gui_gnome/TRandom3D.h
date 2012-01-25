/*
 *  TRandom3D.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __TRandom3D__
#define __TRandom3D__

#include "Earl.h"
#include "TypeDefs.h"
#include "Random3D_c.h"
#include "Random/TRandom.h"
#include "TRandom3D.h"

class TRandom3D : virtual public Random3D_c, virtual public TRandom {

public:
	TRandom3D (TMap *owner, char *name);
	
	virtual ClassID 	GetClassID () { return TYPE_RANDOMMOVER3D; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMMOVER3D) return TRUE; return TRandom::IAm(id); }
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	virtual void		Draw(Rect r, WorldRect view) { }
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
	//virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
	//		virtual OSErr 		AddItem(ListItem item);
	virtual OSErr 		SettingsItem(ListItem item);
	virtual OSErr 		DeleteItem(ListItem item);

};

#endif
