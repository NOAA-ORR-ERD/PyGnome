/*
 *  TMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TMover__
#define __TMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "TClassID.h"

class TMap;
class TModelMessage;

class TMover : virtual public Mover_c,  public TClassID
{
	
public:
	RGBColor			fColor;

	TMover (TMap *owner, char *name);
	virtual			   ~TMover () { Dispose (); }
	//virtual void		Dispose () {}
	
	virtual ClassID 	GetClassID () { return TYPE_MOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_MOVER) return TRUE; return TClassID::IAm(id); }
	virtual OSErr 		MakeClone(TMover **clonePtrPtr);
	virtual OSErr 		BecomeClone(TMover *clone);
	virtual	OSErr 		ReplaceMover() {return 0;}

	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	virtual void		Draw (Rect r, WorldRect view) { }
	virtual Boolean 	DrawingDependsOnTime(void) {return false;}
	
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	virtual OSErr 		UpItem (ListItem item);
	virtual OSErr 		DownItem (ListItem item);
	//virtual OSErr		AddItem(ListItem item);
	

	virtual OSErr		InitMover ();
};


#endif
