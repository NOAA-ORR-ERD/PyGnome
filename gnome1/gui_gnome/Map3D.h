/*
 *  Map3D.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __Map3D__
#define __Map3D__

#include "Earl.h"
#include "TypeDefs.h"
#include "Map3D_c.h"
#include "TMap.h"


class Map3D : virtual public Map3D_c,  public TMap
{

public:
	
	Boolean			bDrawLandBitMap;
	Boolean			bDrawWaterBitMap;
	Boolean			bShowDepthContours;
	
#ifdef IBM
	HDIB			fWaterBitmap;
	HDIB			fLandBitmap;
#else
	BitMap			fWaterBitmap; 
	BitMap			fLandBitmap; 
#endif
	
	Map3D (char* name, WorldRect bounds);
	virtual		   ~Map3D () { Dispose (); }

	//virtual OSErr	InitMap (char *path);
	virtual void	Dispose ();
	
	virtual ClassID GetClassID () { return TYPE_MAP3D; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_MAP3D) return TRUE; return TMap::IAm(id); }

	virtual OSErr 	AddMover(TMover *theMover, short where);
	OSErr		MakeBitmaps();
	virtual	OSErr 	ReplaceMap();
	virtual	OSErr		DropMover (TMover *theMover);
	
	virtual long 	GetLandType (WorldPoint p);
	virtual	Boolean InMap (WorldPoint p);
	virtual Boolean OnLand (WorldPoint p);
	Boolean InWater (WorldPoint p);
	virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	virtual WorldPoint3D	MovementCheck2 (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	
	void 	FindNearestBoundary(Point where, long *verNum, long *segNo);
	void	FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo);
	

	virtual void	Draw (Rect r, WorldRect view);
	virtual void 	DrawBoundaries(Rect r);
	virtual void 	DrawBoundaries2(Rect r);
	virtual	void 	DrawDepthContourScale(Rect r, WorldRect view);
#ifdef IBM
	void 			EraseRegion(Rect r);
#endif
	virtual Boolean DrawingDependsOnTime(void){return TMap::DrawingDependsOnTime();}
	
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	
	Rect 			DoArrowTool(long triNum);
	
	// I/O methods
	virtual OSErr 	Read  (BFPB *bfpb); // read from the current position
	virtual OSErr 	Write (BFPB *bfpb); // write to the current position
	
	OSErr 	ReadTopology(char* path);
	OSErr 	ExportTopology(char* path);

	OSErr	ReadCATSMap(char *path); 
	OSErr	GetPointsAndMask(char *path,DOUBLEH *maskH,WORLDPOINTFH *vertexPtsH, FLOATH *depthPtsH, long *numRows, long *numCols);	
	//OSErr	GetPointsAndBoundary2(char *path,long *numTri,WORLDPOINTFH *vertexPtsH, FLOATH *depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri);
	OSErr	GetPointsAndBoundary(char *path,WORLDPOINTFH *vertexPtsH, FLOATH *depthPtsH, long *numNodes, LONGPTR *bndry_indices, LONGPTR *bndry_nums, LONGPTR *bndry_type, long *numBoundaryPts, LONGPTR *tri_verts, LONGPTR *tri_neighbors, long *ntri);
};

#endif
