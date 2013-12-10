/*
 *  NetCDFWindMoverCurv.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFWindMoverCurv__
#define __NetCDFWindMoverCurv__

#include <vector>
using namespace std;

#include "Earl.h"
#include "TypeDefs.h"
#include "NetCDFWindMoverCurv_c.h"

#include "NetCDFWindMover.h"

class NetCDFWindMoverCurv : virtual public NetCDFWindMoverCurv_c,  public NetCDFWindMover
{
	
public:
	NetCDFWindMoverCurv (TMap *owner, char *name);
	~NetCDFWindMoverCurv () { Dispose (); }
	virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVERCURV; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVERCURV) return TRUE; return NetCDFWindMover::IAm(id); }
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	
	virtual	OSErr		ReadTopology(vector<string> &linesInFile, TMap **newMap);
	virtual OSErr 	 	ReadTopology(const char* path, TMap **newMap);
	virtual OSErr 		ExportTopology(char* path);
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	virtual void		Draw (Rect r, WorldRect view);
	
	//virtual long		GetListLength ();
	//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 	AddItem (ListItem item);
	//virtual OSErr 	SettingsItem (ListItem item);
	//virtual OSErr 	DeleteItem (ListItem item);
	
	//virtual OSErr 	SettingsDialog();
	
};

#endif
