
#ifndef __GRIDCURMOVER__
#define __GRIDCURMOVER__

#include "Earl.h"
#include "TypeDefs.h"
#include "GridCurMover_c.h"

#include "TCATSMover.h"
#include "GridVel.h"
#include "PtCurMover.h"
#include "TWindMover.h"

Boolean IsGridCurTimeFile (char *path, short *selectedUnits);

class GridCurMover : virtual public GridCurMover_c,  public TCATSMover
{
	
	public:
						GridCurMover (TMap *owner, char *name);
						~GridCurMover () { Dispose (); }
		virtual void	Dispose ();
	//virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
	virtual ClassID 	GetClassID () { return TYPE_GRIDCURMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_GRIDCURMOVER) return TRUE; return TCATSMover::IAm(id); }
	void 				DisposeLoadedData(LoadedData * dataPtr);	
	void 				ClearLoadedData(LoadedData * dataPtr);
	long 				GetNumTimesInFile();
	long 				GetNumFiles();
	virtual OSErr 		CheckAndScanFile(char *errmsg);
	virtual Boolean 	CheckInterval(long &timeDataInterval);
	virtual OSErr	 	SetInterval(char *errmsg);
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	virtual OSErr		TextRead(char *path);
	OSErr 				ReadHeaderLines(char *path, WorldRect *bounds);
	OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	OSErr 				ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataHdl,Boolean setStartTime);
	OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
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


//OSErr	GridWindSettingsDialog(GridWindMover *mover, TMap *owner,Boolean bAddMover,DialogPtr parentWindow);

#endif //  __GRIDCURMOVER__
