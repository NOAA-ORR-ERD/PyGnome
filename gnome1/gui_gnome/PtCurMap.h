/*
 *  PtCurMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __PtCurMap__
#define __PtCurMap__

#include "Earl.h"
#include "TypeDefs.h"
#include "PtCurMap_c.h"
#include "TMap.h"


class PtCurMap : virtual public PtCurMap_c,  public TMap
{

public:
	
#ifdef IBM
	HDIB			fWaterBitmap;
	HDIB			fLandBitmap;
#else
	BitMap			fWaterBitmap; 
	BitMap			fLandBitmap; 
#endif
	
	PtCurMap (char* name, WorldRect bounds);
	virtual		   ~PtCurMap () { Dispose (); }

	//virtual OSErr	InitMap (char *path);
	virtual void	Dispose ();
	
	virtual ClassID GetClassID () { return TYPE_PTCURMAP; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMAP) return TRUE; return TMap::IAm(id); }

	virtual OSErr 	AddMover(TMover *theMover, short where);
	OSErr		MakeBitmaps();
	virtual	OSErr 	ReplaceMap();
	virtual	OSErr		DropMover (TMover *theMover);
	void 	AddSegmentToSegHdl(long startno);
	void 	SetSelectedBeach(LONGH *segh, LONGH selh);
	void 	ClearSelectedBeach();
	void	ClearSegmentHdl();
	void 	SetBeachSegmentFlag(LONGH *beachBoundaryH, long *numBeachBoundaries);
	
	void 	FindNearestBoundary(Point where, long *verNum, long *segNo);

	virtual long 	GetLandType (WorldPoint p);
	virtual	Boolean InMap (WorldPoint p);
	virtual Boolean OnLand (WorldPoint p);
	Boolean InWater (WorldPoint p);
	virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	virtual WorldPoint3D	MovementCheck2 (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	
	virtual void	Draw (Rect r, WorldRect view);
	virtual void 	DrawBoundaries(Rect r);
	virtual void 	DrawBoundaries2(Rect r);
	virtual void 	DrawContours(Rect r, WorldRect view);	// total over all LELists
	virtual void  	DrawContourScale(Rect r, WorldRect view);
	virtual	void 	DrawDepthContourScale(Rect r, WorldRect view);
	void 			DrawSegmentLabels(Rect r);
	void 			DrawPointLabels(Rect r);
#ifdef IBM
	void 			EraseRegion(Rect r);
#endif
	virtual Boolean DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
	Boolean 		ThereAreTrianglesSelected();
	//void 			TrackOutputData(TOLEList *thisLEList);
	virtual			void TrackOutputData(void);
	virtual			void TrackOutputDataInAllLayers(void);
	OSErr 			DoAnalysisMenuItems(long menuCodedItemID);
	
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	
	Rect 			DoArrowTool(long triNum);
	void 			DoLassoTool(Point p);
	void 			MarkRect(Point p);
	
	// I/O methods
	virtual OSErr 	Read  (BFPB *bfpb); // read from the current position
	virtual OSErr 	Write (BFPB *bfpb); // write to the current position
	
	OSErr ExportOiledShorelineData(OiledShorelineDataHdl oiledShorelineHdl);

};

#endif
