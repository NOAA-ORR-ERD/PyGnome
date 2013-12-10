/*
 *  NetCDFMoverCurv.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/////////////////////////////////////////////////
// Curvilinear grid code - separate mover
// read in grid values for first time and set up transformation (dagtree?)
// need to read in lat/lon since won't be evenly spaced

#ifndef __NetCDFMoverCurv__
#define __NetCDFMoverCurv__

#include <vector>
using namespace std;

#include "NetCDFMoverCurv_c.h"
#include "NetCDFMover.h"

class NetCDFMoverCurv : virtual public NetCDFMoverCurv_c,  public NetCDFMover
{
	
public:
	NetCDFMoverCurv (TMap *owner, char *name);
	~NetCDFMoverCurv () { Dispose (); }
	virtual void		Dispose ();
	virtual Boolean		IAmA3DMover();

	virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVERCURV; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVERCURV) return TRUE; return NetCDFMover::IAm(id); }

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//virtual OSErr		TextRead(char *path,TMap **newMap);
	virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	
	virtual	OSErr		ReadTopology(vector<string> &linesInFile, TMap **newMap);
	virtual	OSErr		ReadTopology(const char* path, TMap **newMap);
	virtual	OSErr		ExportTopology(char* path);
	//OSErr 				ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg);
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);	
	virtual void		Draw (Rect r, WorldRect view);
	virtual void		DrawContourScale(Rect r, WorldRect view);
	
	//virtual long		GetListLength ();
	//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 		AddItem (ListItem item);
	//virtual OSErr 		SettingsItem (ListItem item);
	//virtual OSErr 		DeleteItem (ListItem item);
	
	//virtual OSErr 		SettingsDialog();
	
	
};


#endif
