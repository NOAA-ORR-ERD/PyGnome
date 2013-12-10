/*
 *  TMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TMap__
#define __TMap__

#include "Earl.h"
#include "TypeDefs.h"
#include "Map_c.h"
#include "TClassID.h"

class TMover;

class TMap : virtual public Map_c,  public TClassID
{
	
public:
	// map methods
	TMap (char *name, WorldRect bounds);
	virtual			   ~TMap () { Dispose (); }
	virtual void		Dispose ();		

	virtual ClassID		GetClassID () { return TYPE_MAP; }
	virtual Boolean		IAm(ClassID id) {if(id==TYPE_MAP) return TRUE; return TClassID::IAm(id); }

	virtual OSErr	CheckAndPassOnMessage(TModelMessage * model);
	virtual Boolean	IsDirty ();
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean 	DrawingDependsOnTime(void);
	virtual	void 		DrawESILegend(Rect r, WorldRect view) {return;}
	virtual	void		DrawContours(Rect r, WorldRect view) {return;}
	virtual	void		TrackOutputData(void) {return;}
	virtual	void		TrackOutputDataInAllLayers(void) {return;}
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from the current position
	virtual OSErr 		Write (BFPB *bfpb); // write to the current position
	
	// list display methods
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	virtual OSErr 		UpItem (ListItem item);
	virtual OSErr 		DownItem ( ListItem item);
	virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	// mover methods
	CMyList				*GetMoverList () { return moverList; }
	virtual OSErr		DropMover (TMover *theMover);
	
	virtual OSErr 		ReplaceMap();	
	
		
};


#endif
