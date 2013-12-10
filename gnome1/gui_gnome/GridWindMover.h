/*
 *  GridWindMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridWindMover__
#define __GridWindMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "GridWindMover_c.h"
#include "GridWndMover_c.h"	 // for the enum - should move somewhere else...
#include "TWindMover.h"

// build off TWindMover or off TMover with a lot of WindMover carried along ??
class GridWindMover : virtual public GridWindMover_c,  public TWindMover
{
	
public:
	GridWindMover (TMap *owner, char* name);
	~GridWindMover () { Dispose (); }
	//virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_GRIDWINDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_GRIDWINDMOVER) return TRUE; return TWindMover::IAm(id); }
	
	virtual OSErr		InitMover (TimeGridVel *grid); 

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//		virtual OSErr 		AddItem (ListItem item);
	virtual OSErr		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean		DrawingDependsOnTime(void);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		
	
};

OSErr			GridWindSettingsDialog(GridWindMover *mover, TMap *owner,Boolean bAddMover,WindowPtr parentWindow);

#endif

