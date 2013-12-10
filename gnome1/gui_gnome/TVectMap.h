/*
 *  TVectMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 3/11/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TVectorMap__
#define __TVectorMap__

#include "Basics.h"
#include "TypeDefs.h"
#include "TMap.h"
#include "VectMap_c.h"

class TVectorMap : virtual public VectorMap_c, public TMap
{

public:
	
#ifdef IBM
	HDIB				fLandWaterBitmap;
	HDIB				fAllowableSpillBitmap;
	HDIB				fMapBoundsBitmap;
	HDIB				fESIBitmap;
#else
	BitMap				fLandWaterBitmap; 
	BitMap				fAllowableSpillBitmap; 
	BitMap				fMapBoundsBitmap; 
	BitMap				fESIBitmap; 
	//CGrafPtr 			fESIBitmap;		// ESI segments bitmap (color)
#endif
	
	TVectorMap (char* name, WorldRect bounds);
	virtual			   ~TVectorMap () { Dispose (); }
	
	virtual ClassID 	GetClassID () { return TYPE_VECTORMAP; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_VECTORMAP) return TRUE; return TMap::IAm(id); }	

	virtual OSErr		InitMap ();
	virtual OSErr		InitMap (char *path);

	virtual void		Dispose ();
	
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean 	DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
	virtual	void 		DrawESILegend(Rect r, WorldRect view);

	
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage * model);
	
	// I/O methods
	virtual OSErr 		Read  (BFPB *bfpb); // read from the current position
	virtual OSErr 		Write (BFPB *bfpb); // write to the current position
	OSErr 				ExportAsBNAFileForGnomeAnalyst(char* path);
	OSErr 				ExportAsBNAFile(char* path);
	virtual void		GetSuggestedFileName(char* suggestedFileName,char* extension);
	OSErr 				ImportESIData (char *path);
	virtual OSErr 		ReplaceMap();
	virtual	Boolean 	InMap (WorldPoint p);
	virtual Boolean 	OnLand (WorldPoint p);
	virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	virtual Boolean 	IsAllowableSpillPoint(WorldPoint p);
	OSErr 				ChangeMapBox(WorldPoint p, WorldPoint p2);
	
private:
	OSErr 				SelectMapBox (WorldRect bounds);
	OSErr 				ImportMap (char *path); 
};


#endif