
#ifndef __NetCDFMover__
#define __NetCDFMover__

#include "NetCDFMover_c.h"

#include "GridVel.h"
#include "PtCurMover.h"
#include "PtCurMap.h"

//Seconds RoundDateSeconds(Seconds timeInSeconds);
PtCurMap* GetPtCurMap(void);
TMap* Get3DMap(void);


class NetCDFMover : virtual public NetCDFMover_c,  public TCurrentMover
{
	public:
							NetCDFMover (TMap *owner, char *name);
						   ~NetCDFMover () { Dispose (); }
		virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVER; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVER) return TRUE; return TCurrentMover::IAm(id); }
	
	virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?

	virtual	OSErr 	ReplaceMover();


	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
	OSErr 				ReadInputFileNames(char *fileNamesPath);
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	virtual void		DrawContourScale(Rect r, WorldRect view);
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean	DrawingDependsOnTime(void);
	
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	virtual OSErr 		SettingsDialog();
	

};


#endif //  __NETCDFMOVER__
