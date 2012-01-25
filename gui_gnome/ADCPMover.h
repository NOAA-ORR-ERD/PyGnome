/*
 *  ADCPMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __ADCPMover__
#define __ADCPMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "ADCPMover_c.h"
#include "CurrentMover/TCurrentMover.h"
#include "Uncertainty.h"

class ADCPMover : virtual public ADCPMover_c, virtual public TCurrentMover
{	// code goes here, keep a list of time files or have umbrella move that keeps list of ADCPMovers?

public:
	ADCPMover (TMap *owner, char *name);
	virtual			   ~ADCPMover () { Dispose (); }
	virtual void		Dispose ();
	virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
	virtual OSErr		InitMover ();
	virtual ClassID 	GetClassID () { return TYPE_ADCPMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_ADCPMOVER) return TRUE; return TCurrentMover::IAm(id); }
	virtual Boolean 	OkToAddToUniversalMap();
	virtual	OSErr 		ReplaceMover();
	virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
	virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	OSErr				TextRead(char *path);
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	virtual void		Draw (Rect r, WorldRect view);
	virtual	Boolean		DrawingDependsOnTime(void);
	virtual void		DrawContourScale(Rect r, WorldRect view);
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//		virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
};

#endif
