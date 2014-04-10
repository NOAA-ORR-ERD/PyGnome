
#ifndef __TIDECURCYCLEMOVER__
#define __TIDECURCYCLEMOVER__

#include "TideCurCycleMover_c.h"

#include "TCATSMover.h"

Boolean IsTideCurCycleFile (char *path, short *gridType);


class TideCurCycleMover : virtual public TideCurCycleMover_c,  public TCATSMover
{

	public:
							TideCurCycleMover (TMap *owner, char *name);
						   ~TideCurCycleMover () { Dispose (); }
	//virtual void		Dispose ();
	//virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
	virtual ClassID 	GetClassID () { return TYPE_TIDECURCYCLEMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_TIDECURCYCLEMOVER) return TRUE; return TCATSMover::IAm(id); }

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//virtual OSErr		TextRead(char *path, TMap **newMap);
	virtual OSErr		TextRead(char *path, TMap **newMap, char *topFilePath);
	
	virtual	OSErr		ReadTopology(vector<string> &linesInFile, TMap **newMap);
	virtual	OSErr		ReadTopology(char* path, TMap **newMap);
	virtual OSErr		ExportTopology(char* path);
	

	// list display methods
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean		DrawingDependsOnTime(void);
	
	//virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 	AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	
	//virtual OSErr 		SettingsDialog();
	

};


#endif //  __TIDECURCYCLEMOVER__
