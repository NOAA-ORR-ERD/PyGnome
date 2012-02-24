#ifndef __TWindMover__
#define __TWindMover__

#include "WindMover_c.h"
#include "Mover/TMover.h"

class TWindMover : virtual public WindMover_c, public TMover
{

public:
	TWindMover (TMap *owner, char* name);
	virtual			   ~TWindMover () { Dispose (); }
	virtual void		Dispose ();

	virtual OSErr 		MakeClone(TClassID **clonePtrPtr);
	virtual OSErr 		BecomeClone(TClassID *clone);
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
	
	void 				DrawWindVector(Rect r, WorldRect view);
	void				GetFileName(char* fileName) ;
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	OSErr 				ExportVariableWind(char* path);

};

#endif