/*
 *  TCATSMover.h
 *  c_gnome
 *
 *  Created by Alex Hadjilambris on 2/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TCATSMover__
#define __TCATSMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "CATSMover_c.h"
#include "CurrentMover/TCurrentMover.h"

class TCATSMover : virtual public CATSMover_c, public TCurrentMover
{

	
public:
	TCATSMover (TMap *owner, char *name);
	virtual			   ~TCATSMover () { Dispose (); }
	virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);

	virtual void		Dispose ();
	virtual Boolean 	OkToAddToUniversalMap();
	virtual	OSErr 		ReplaceMover();
	
	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	
	virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
	virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	

	virtual OSErr 		PrepareForModelStep();
	virtual void 		ModelStepIsDone();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//		virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	virtual	OSErr 	ExportTopology(char* path);
	

};

#endif
