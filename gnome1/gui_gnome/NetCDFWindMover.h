/*
 *  NetCDFWindMover.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMover__
#define __NetCDFWindMover__

#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFWindMover_c.h"
#include "TWindMover.h"

// build off TWindMover or off TMover with a lot of WindMover carried along ??
// for now new mover type but brought in under variable windmover
class NetCDFWindMover : virtual public NetCDFWindMover_c,  public TWindMover
{
	
public:
	NetCDFWindMover (TMap *owner, char* name);
	~NetCDFWindMover () { Dispose (); }
	virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVER; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVER) return TRUE; return TWindMover::IAm(id); }

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
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time);	// AH 07/17/2012
	
	virtual OSErr		TextRead(char *path);
	//virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean		DrawingDependsOnTime(void);
	
	OSErr		ReadInputFileNames(char *fileNamesPath);
	void		DisposeAllLoadedData();
	long		GetNumFiles();
	OSErr		CheckAndScanFile(char *errmsg, const Seconds& model_time);	// AH 07/17/2012
	OSErr		ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime);	// AH 07/17/2012
	
	
};

OSErr			NetCDFWindSettingsDialog(NetCDFWindMover *mover, TMap *owner,Boolean bAddMover,WindowPtr parentWindow);

#endif

