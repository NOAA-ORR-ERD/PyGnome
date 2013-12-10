/*
 *  TOLEList.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TOLEList__
#define __TOLEList__

#include "Basics.h"
#include "TypeDefs.h"
#include "TLEList.h"
#include "OLEList_c.h"

class TOLEList : virtual public OLEList_c, public TLEList
{

public:
	TOLEList ();
	virtual			   ~TOLEList () { Dispose (); }

	virtual ClassID 	GetClassID () { return TYPE_OSSMLELIST; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_OSSMLELIST) return TRUE; return TLEList::IAm(id); }

	//virtual OSErr		Initialize (long numElements, LERec *representativeLE, short massUnits,
	//								WorldPoint startRelPos, WorldPoint endRelPos,
	//								Seconds startRelTime, Seconds endRelTime,
	//								Boolean bWantEndRelTime, Boolean bWantEndRelPosition);
	virtual OSErr 		Initialize(LESetSummary * summary,Boolean deleteLERecH);
	virtual OSErr		Initialize (long numElements, LERecH array, short massUnits, Seconds fileTime);
	virtual OSErr		Reset (Boolean newKeys);
	
	//void				ReleaseLE (long i);
	//void				AgeLE (long i);
	//virtual void		BeachLE (long i, WorldPoint beachPosition);
	
	virtual	void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read  (BFPB *bfpb); // read from the current position
	virtual OSErr 		Write (BFPB *bfpb); // write to    the current position
	
	// list display methods: base class functionality
	virtual void		Draw (Rect r, WorldRect view);
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//		virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	OSErr 				ExportBudgetTableHdl(char* path, BFPB *bfpb);
	virtual void 		GetMassBalanceLines(Boolean includePercentage, char* line1,char*line2,char* line3,char* line4,char* line5,char* line6,char* line7);	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
};

void SetChemicalHalfLife(double halfLife);
double GetChemicalHalfLife();

#endif