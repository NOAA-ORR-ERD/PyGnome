/*
 *  TCATSMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TCATSMover__
#define __TCATSMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "CATSMover_c.h"
#include "CurrentMover/TCurrentMover.h"
#include "Uncertainty.h"

class TCATSMover : virtual public CATSMover_c, virtual public TCurrentMover
{

public:
	TCATSMover (TMap *owner, char *name);
	virtual			   ~TCATSMover () { Dispose (); }
	virtual void		Dispose ();
	virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
	virtual ClassID 	GetClassID () { return TYPE_CATSMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_CATSMOVER) return TRUE; return TCurrentMover::IAm(id); }
	virtual Boolean 	OkToAddToUniversalMap();
	virtual	OSErr 		ReplaceMover();
	virtual CurrentUncertainyInfo GetCurrentUncertaintyInfo ();
	virtual void		SetCurrentUncertaintyInfo (CurrentUncertainyInfo info);
	virtual Boolean 	CurrentUncertaintySame (CurrentUncertainyInfo info);
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	virtual void		Draw (Rect r, WorldRect view);
	
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//		virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	virtual	OSErr		ReadTopology(char* path, TMap **newMap);
	virtual	OSErr		ExportTopology(char* path);
	
};

typedef struct
{
	Seconds			fUncertainStartTime;
	double			fDuration; 				// duration time for uncertainty;
	/////
	WorldPoint 		refP; 					// location of tide station or map-join pin
	long 				refZ; 					// meters, positive up
	short 			scaleType; 				// none, constant, or file
	double 			scaleValue; 			// constant value to match at refP
	char 				scaleOtherFile[32]; 	// file to match at refP
	double 			refScale; 				// multiply current-grid value at refP by refScale to match value
	Boolean			bTimeFileActive;		// active / inactive flag
	Boolean 			bShowGrid;
	Boolean 			bShowArrows;
	double 			arrowScale;
	double			fEddyDiffusion;		
	double			fEddyV0;			
	double			fDownCurUncertainty;	
	double			fUpCurUncertainty;	
	double			fRightCurUncertainty;	
	double			fLeftCurUncertainty;	
} CATSDialogNonPtrFields;

#endif
