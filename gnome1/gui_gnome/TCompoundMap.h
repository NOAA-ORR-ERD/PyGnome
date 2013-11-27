/*
 *  TCompoundMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TCompoundMap__
#define __TCompoundMap__

#include "Earl.h"
#include "TypeDefs.h"
#include "CompoundMap_c.h"

///////////////////////////////////////////////////////////////////////////
//class TCompoundMap : public TMap	// maybe PtCurMap ?
class TCompoundMap : virtual public CompoundMap_c,  public PtCurMap	// maybe PtCurMap ?
{
public:
	
	TCompoundMap (char* name, WorldRect bounds);
	virtual			   ~TCompoundMap () { Dispose (); }
	
	virtual void 		Dispose ();
	virtual OSErr		InitMap ();
	//virtual OSErr		InitContourLevels();
	//virtual OSErr		InitDropletSizes();
	
	virtual ClassID 	GetClassID () { return TYPE_COMPOUNDMAP; }
	virtual Boolean		IAm(ClassID id) { if(id==TYPE_COMPOUNDMAP) return TRUE; return PtCurMap::IAm(id); }
	
	//virtual OSErr		AddMover (TMover *theMover, short where);
	virtual OSErr		AddMap (TMap *theMap, short where);
	virtual OSErr		DropMap (TMap *theMap);
	virtual OSErr		DropMover (TMover *theMover);
	
	//TMap*					AddNewMap(OSErr *err);	// may not need this
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb);  	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); 	// write to  current position
	
	//virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	// list display methods
	virtual void		Draw(Rect r, WorldRect view);
	void 	FindNearestBoundary(Point where, long *verNum, long *segNo);
	
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
	virtual OSErr 		UpItem 			(ListItem item);
	virtual OSErr 		DownItem 		(ListItem item);
	//virtual OSErr 		AddItem(ListItem item);
	virtual OSErr 		SettingsItem(ListItem item);
	//virtual OSErr 		DeleteItem(ListItem item);
	
	
	virtual OSErr 	AddMover(TMover *theMover, short where);
	OSErr		MakeBitmaps()	{return 0;}
	virtual	OSErr 	ReplaceMap(){return 0;}

	//void 	SetSelectedBeach(LONGH *segh, LONGH selh);
	//void 	ClearSelectedBeach();
	//void 	SetBeachSegmentFlag(LONGH *beachBoundaryH, long *numBeachBoundaries);
	
	//virtual void 	DrawBoundaries(Rect r);
	//virtual void 	DrawBoundaries2(Rect r);
	virtual	Boolean InMap (WorldPoint p);	//  check all maps
	virtual Boolean OnLand (WorldPoint p);	// check all maps
	virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);	// have to do something here
	virtual long 	GetLandType (WorldPoint p);
	void			DrawContoursFromMapIndex(Rect r, WorldRect view, long mapIndex);
	virtual	void 	DrawContours(Rect r, WorldRect view);	// total over all LELists
	virtual void  	DrawContourScale(Rect r, WorldRect view);
	virtual	void 	DrawDepthContourScale(Rect r, WorldRect view);
	//void 			DrawSegmentLabels(Rect r);
	//void 			DrawPointLabels(Rect r);
#ifdef IBM
	void 			EraseRegion(Rect r);
#endif
	virtual Boolean DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
	Boolean 		ThereAreTrianglesSelected();
	//void 			TrackOutputData(TOLEList *thisLEList);
	virtual	void 		TrackOutputData(void);
	virtual	void		TrackOutputDataInAllLayers(void);
	void			TrackOutputDataFromMapIndex(long mapIndex);
	//OSErr 			DoAnalysisMenuItems(long menuCodedItemID);
	
	Rect 			DoArrowTool(long triNum);
	void 			DoLassoTool(Point p);
	void 			MarkRect(Point p);
	long			WhichMapIsPtIn(WorldPoint wp);
	long			WhichMapIsPtInWater(WorldPoint wp);
	virtual double	DepthAtPoint(WorldPoint wp);// check by priority
	OSErr 			GetDepthAtMaxTri(long *maxTriIndex, double *depthAtPnt);	
	WorldPoint3D	ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp);

};

#endif
