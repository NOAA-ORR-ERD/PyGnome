/*
 *  GridWndMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridWndMover__
#define __GridWndMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "GridWndMover_c.h"
#include "TWindMover.h"
#include <vector>
using namespace std;

Boolean IsGridWindFile (char *path, short *selectedUnits);
bool IsGridWindFile (std::vector<std::string> &linesInFile, short *selectedUnitsOut);

// build off TWindMover or off TMover with a lot of WindMover carried along ??
// for now new mover type but brought in under variable windmover
// what about constant wind case? parallel to gridcur?
class GridWndMover : virtual public GridWndMover_c,  public TWindMover
{

public:
	GridWndMover (TMap *owner, char* name);
	~GridWndMover () { Dispose (); }
	virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_GRIDWNDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_GRIDWNDMOVER) return TRUE; return TWindMover::IAm(id); }
	void 					DisposeLoadedData(LoadedData * dataPtr);	
	void 					ClearLoadedData(LoadedData * dataPtr);
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
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	long 				GetNumTimesInFile();
	long				GetNumFiles();
	//virtual OSErr		GetStartTime(Seconds *startTime);
	//virtual OSErr		GetEndTime(Seconds *endTime);
	
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& model_time);	// AH 07/17/2012
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time); // AH 07/17/2012
	
	virtual OSErr		TextRead(char *path);
	OSErr 				ReadHeaderLines(char *path, WorldRect *bounds);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	OSErr 				ScanFileForTimes(char *path, PtCurTimeDataHdl *timeDataH,Boolean setStartTime);	
	OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);
	
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean		DrawingDependsOnTime(void);
	
	
};

//Boolean IsGridWindFile(char *path,short *selectedUnits);


#endif
