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
#include "ClassID/TClassID.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

class TMap;

class TMover : virtual public Mover_c, public TClassID
{
	
public:
	TMover() {}
	TMover (TMap *owner, char *name);
	virtual			   ~TMover () { Dispose (); }
	virtual void		Dispose ();
	virtual OSErr 		MakeClone(TClassID **clonePtrPtr);
	virtual OSErr 		BecomeClone(TClassID *clone);
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
	
	void				GetMoverName (char *name) { GetClassName (name); }
	void				SetMoverName (char *name) { SetClassName (name); }
	TMap				*GetMoverMap () { return moverMap; }
	void				SetMoverMap (TMap *owner) { moverMap = owner; }
	virtual OSErr 		PrepareForModelStep(){ return noErr; }
	virtual void 		ModelStepIsDone(){}
			
};

#undef TMap
#endif