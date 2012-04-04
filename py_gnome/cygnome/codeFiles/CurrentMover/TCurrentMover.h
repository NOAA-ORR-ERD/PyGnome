/*
 *  TCurrentMover.h
 *  c_gnome
 *
 *  Created by Generic Programmer on 2/21/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TCurrentMover__
#define __TCurrentMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "Mover/TMover.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

#include "Map/TMap.h"

class TCurrentMover : virtual public CurrentMover_c, public TMover
{

	
public:
	TCurrentMover (TMap *owner, char *name);
	virtual			   ~TCurrentMover () { Dispose (); }
	virtual void		Dispose ();
	virtual OSErr 		PrepareForModelStep();

	virtual OSErr 		UpItem (ListItem item);
	virtual OSErr 		DownItem (ListItem item);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	
	virtual void 		UpdateUncertaintyValues(Seconds elapsedTime);
	virtual OSErr		UpdateUncertainty(void);
	virtual OSErr		AllocateUncertainty ();
	virtual void		DisposeUncertainty ();

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  // read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);
	virtual OSErr 		SettingsDialog() {return 0;}

	virtual Boolean		IAmA3DMover(){return false;}
	virtual OSErr 		ExportTopology(char* path) {return 2;}	

};

#undef TMap
#endif
