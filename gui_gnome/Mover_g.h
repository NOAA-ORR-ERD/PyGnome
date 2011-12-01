/*
 *  Mover_g.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __Mover_g__
#define __Mover_g__

#include "Earl.h"
#include "TypeDefs.h"
#include "Mover_c.h"
#include "ClassID_g.h"

class TMap;
class TModelMessage;
class TMover;

class Mover_g : virtual public Mover_b, virtual public ClassID_g {

public:
	
	virtual OSErr 		MakeClone(TMover **clonePtrPtr);
	virtual OSErr 		BecomeClone(TMover *clone);
	virtual	OSErr 		ReplaceMover() {return 0;}
	
	virtual ClassID 	GetClassID () { return TYPE_MOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_MOVER) return TRUE; return ClassID_g::IAm(id); }
	
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



#endif