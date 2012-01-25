/*
 *  TCATSMover3D.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __TCATSMover3D__
#define __TCATSMover3D__

#include "Earl.h"
#include "TypeDefs.h"
#include "CATSMover3D_c.h"
#include "CATSMover/TCATSMover.h"

class TCATSMover3D : virtual public CATSMover3D_c, virtual public TCATSMover
{

public:
	TCATSMover3D (TMap *owner, char *name);
	virtual			   ~TCATSMover3D () { Dispose (); }
	virtual void		Dispose ();
	//virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
	virtual ClassID 	GetClassID () { return TYPE_CATSMOVER3D; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_CATSMOVER3D) return TRUE; return TCATSMover::IAm(id); }
	virtual Boolean		IAmA3DMover(){return true;}
	virtual Boolean 	OkToAddToUniversalMap();
	OSErr				TextRead(char *path, TMap **newMap); 
	OSErr 				ImportGrid(char *path); 
	OSErr 				CreateRefinedGrid (Boolean askForFile, char* givenPath, char* givenFileName);
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	virtual void		Draw (Rect r, WorldRect view);
	//virtual long		GetListLength ();
	//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//		virtual OSErr 		AddItem (ListItem item);
	//virtual OSErr 		SettingsItem (ListItem item);
	//virtual OSErr 		DeleteItem (ListItem item);
	virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
	virtual	OSErr 	ExportTopology(char* path);
	
};

#endif
