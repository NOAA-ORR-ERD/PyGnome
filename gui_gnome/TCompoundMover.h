/*
 *  TCompoundMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TCompoundMover__
#define __TCompoundMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "CompoundMover_c.h"
#include "TCurrentMover.h"

class TCompoundMover : virtual public CompoundMover_c,  public TCurrentMover	// maybe just TMover
{
public:
	
	TCompoundMover (TMap *owner, char *name);
	virtual			   ~TCompoundMover () { Dispose (); }
	virtual void 		Dispose ();
	
	virtual OSErr		InitMover ();
	virtual ClassID 	GetClassID () { return TYPE_COMPOUNDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_COMPOUNDMOVER) return TRUE; return TCurrentMover::IAm(id); }
	virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	
	virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	TCurrentMover*		AddCurrent(OSErr *err,TCompoundMap **newMap);	

	
	// mover methods
	CMyList				*GetMoverList () { return moverList; }
	//virtual OSErr		AddMover (TMover *theMover, short where);
	virtual OSErr		DropMover (TMover *theMover);
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); 	// write to  current position
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	// list display methods
	virtual Boolean		DrawingDependsOnTime(void);
	virtual void		Draw(Rect r, WorldRect view);
	
	
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
	virtual OSErr 		UpItem 			(ListItem item);
	virtual OSErr 		DownItem 		(ListItem item);
	virtual OSErr 		AddItem(ListItem item);
	virtual OSErr 		SettingsItem(ListItem item);
	virtual OSErr 		DeleteItem(ListItem item);
	
	void			SetShowDepthContours();
	Boolean			ShowDepthContourChecked();
	
	
};

#endif
