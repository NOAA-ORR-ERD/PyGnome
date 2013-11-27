/*
 *  TCurrentMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TCurrentMover__
#define __TCurrentMover__

#include "CurrentMover_c.h"
#include "TMover.h"

class TCurrentMover : virtual public CurrentMover_c,  public TMover
{
	
public:
	TCurrentMover (TMap *owner, char *name);
	virtual			   ~TCurrentMover () { Dispose (); }
	//virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_CURRENTMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_CURRENTMOVER) return TRUE; return TMover::IAm(id); }
	
	virtual OSErr 		UpItem (ListItem item);
	virtual OSErr 		DownItem (ListItem item);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	virtual OSErr 		ReadTopology(char* path, TMap **newMap)	{return 2;}
	virtual OSErr 		ExportTopology(char* path) {return 2;}	
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);
	
	virtual OSErr 		SettingsDialog() {return 0;}
	
};

#endif
