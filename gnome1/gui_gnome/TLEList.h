/*
 *  TLEList.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TLEList__
#define __TLEList__

#include "Earl.h"
#include "TypeDefs.h"
#include "LEList_c.h"

class TLEList : virtual public LEList_c, public TClassID
{

public:
	UNIQUEID		fOwnersUniqueID; // set if owned by another LE set, i.e this is a mirrored set

	TLEList ();
	virtual		   ~TLEList () { Dispose (); }
	
	virtual ClassID GetClassID () { return TYPE_LELIST; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_LELIST) return TRUE; return TClassID::IAm(id); }
	
	//virtual OSErr	Initialize (long numElements, LERec *representativeLE, short massUnits,
	//							WorldPoint startRelPos, WorldPoint endRelPos,
	//							Seconds startRelTime, Seconds endRelTime,
	//							Boolean bWantEndRelTime, Boolean bWantEndRelPosition) { return noErr; }
	//virtual OSErr	Initialize (long numElements, LERecH array, short massUnits, Seconds fileTime) { return noErr; }

	
	void			SetFileDirty (Boolean bFileDirty) { bDirty = bFileDirty; }
	Boolean			IsFileDirty () { return !bDirty; }
	
	virtual	void	Dispose ();
	
	virtual OSErr UpItem (ListItem item);	// code goes here, decide if this is okay - brought from GNOME_beta
	virtual OSErr DownItem (ListItem item);
	
	// I/O methods
	virtual OSErr 	Read  (BFPB *bfpb); // read from the current position
	virtual OSErr 	Write (BFPB *bfpb) ; // write to    the current position
	virtual	OSErr	WriteLE (BFPB *bfpb, LERec *theLE);
	virtual	OSErr	ReadLE  (BFPB *bfpb, LERec *theLE, long version);
	
	// list display methods: base class functionality
	virtual void	Draw (Rect r, WorldRect view) { return; }
	
	
};


#endif
