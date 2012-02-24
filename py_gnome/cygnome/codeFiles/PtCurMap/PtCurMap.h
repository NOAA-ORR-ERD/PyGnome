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

class PtCurMap : public TMap
{
public:
	LONGH			fBoundarySegmentsH;
	LONGH			fBoundaryTypeH;		// 1 land, 2 water
	LONGH			fBoundaryPointsH;	// for curvilinear grids
	LONGH			fSegSelectedH;
	LONGH			fSelectedBeachHdl;	//not sure if both are needed
	LONGH			fSelectedBeachFlagHdl;	//not sure if both are needed
#ifdef IBM
	HDIB			fWaterBitmap;
	HDIB			fLandBitmap;
#else
	BitMap			fWaterBitmap; 
	BitMap			fLandBitmap; 
#endif
	Boolean			bDrawLandBitMap;
	Boolean			bDrawWaterBitMap;
	Boolean			bShowSurfaceLEs;
	
public:
	float			fContourDepth1;
	float			fContourDepth2;
	float			fContourDepth1AtStartOfRun;
	float			fContourDepth2AtStartOfRun;
	float			fBottomRange;
	DOUBLEH			fContourLevelsH;
	Rect			fLegendRect;
	Boolean			bShowLegend;
	short			fDiagnosticStrType;		// 0 no diagnostic string, 1 tri area, 2 num LEs, 3 conc levels, 4 depths, 5 subsurface particles
	Boolean			bDrawContours;
	
	long			fWaterDensity;
	double			fMixedLayerDepth;
	double			fBreakingWaveHeight;
	
	double			*fTriAreaArray;
	//long			*fDepthSliceArray;	// number of LEs in each layer (1m) of depth slice
	float			*fDepthSliceArray;	//changed to ppm in each layer (1m) of depth slice 7/21/03
	
	Boolean			bUseSmoothing;
	Boolean			bUseLineCrossAlgorithm;
	float			fMinDistOffshore;	// set how far LEs reflect off shoreline, so they don't get stuck
	//Boolean			bShowElapsedTime;	// should be a model field
	short			fWaveHtInput;	// 0 from wind speed, 1 breaking wave height measure, 2 significant wave height measure
	DropletInfoRecH	fDropletSizesH;

	Boolean			bTrackAllLayers;

public:
	PtCurMap (char* name, WorldRect bounds);
	virtual		   ~PtCurMap () { Dispose (); }
	
	virtual OSErr	InitMap();
	virtual OSErr	InitContourLevels();
	virtual OSErr	InitDropletSizes();
	//virtual OSErr	InitMap (char *path);
	virtual ClassID GetClassID () { return TYPE_PTCURMAP; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMAP) return TRUE; return TMap::IAm(id); }
	virtual void	Dispose ();
	
	virtual OSErr 	AddMover(TMover *theMover, short where);
	OSErr		MakeBitmaps();
	virtual	OSErr 	ReplaceMap();
	virtual	OSErr		DropMover (TMover *theMover);
	virtual	void 	SetBoundarySegs(LONGH boundarySegs) { fBoundarySegmentsH = boundarySegs; }
	virtual	void 	SetWaterBoundaries(LONGH waterBoundaries) { fBoundaryTypeH = waterBoundaries; }
	virtual	void 	SetBoundaryPoints(LONGH boundaryPts) { fBoundaryPointsH = boundaryPts; }
	virtual	LONGH 	GetBoundarySegs() { return fBoundarySegmentsH; }
	virtual	LONGH 	GetWaterBoundaries() { return fBoundaryTypeH; }
	virtual	LONGH 	GetBoundaryPoints() { return fBoundaryPointsH; }
	void 	AddSegmentToSegHdl(long startno);
	void 	SetSelectedBeach(LONGH *segh, LONGH selh);
	void 	ClearSelectedBeach();
	void	ClearSegmentHdl();
	void 	SetBeachSegmentFlag(LONGH *beachBoundaryH, long *numBeachBoundaries);
	Boolean 	MoreSegments(LONGH segh,long *startIndex, long *endIndex,long *curIndex);
	void			SetMinDistOffshore(WorldRect wBounds);
	long 	WhichSelectedSegAmIIn(long index);
	
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
	virtual	Boolean InMap (WorldPoint p);
	virtual Boolean OnLand (WorldPoint p);
	Boolean InWater (WorldPoint p);
	virtual WorldPoint3D	MovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	virtual WorldPoint3D	MovementCheck2 (WorldPoint3D fromWPt, WorldPoint3D toWPt, Boolean isDispersed);
	virtual	Boolean	HaveMapBoundsLayer (void) { return true; }
	virtual long	PointOnWhichSeg(long p);
	virtual	long 	PointOnWhichSeg(long longVal, long latVal, long *startver, long *endver, float *dist);
	Boolean 	ContiguousPoints(long p1, long p2);
	void 	FindNearestBoundary(Point where, long *verNum, long *segNo);
	void  FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo);
	TMover* 		GetMover(ClassID desiredClassID);
	TCurrentMover* 	Get3DCurrentMover();
	double			GetSpillStartDepth();
	Boolean			ThereIsADispersedSpill();
	Boolean 		ThereAreTrianglesSelected();
	TTriGridVel* 	GetGrid(Boolean wantRefinedGrid);
	virtual TTriGridVel3D* 	GetGrid3D(Boolean wantRefinedGrid);
	LongPointHdl 	GetPointsHdl(Boolean useRefinedGrid);	
	virtual Boolean		CanReFloat (Seconds time, LERec *theLE);
	//		virtual Boolean	CanReFloat (Seconds time, LERec *theLE) { return true; }
	virtual long 	GetLandType (WorldPoint p);
	//Boolean			LEInMap(WorldPoint p);	// not used
	Boolean			InVerticalMap(WorldPoint3D wp);
	//float 			GetMaxDepth(void);
	WorldPoint3D	ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp);
	virtual	double			DepthAtPoint(WorldPoint wp);
	double 			DepthAtCentroid(long triNum);
	WorldPoint3D 	TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint);
	//WorldPoint3D 	TurnLEAlongShoreLine2(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint);	// Gnome_beta diagnostic stuff
	//WorldPoint3D	SubsurfaceMovementCheck (WorldPoint3D fromWPt, WorldPoint3D toWPt, OilStatus *status);
	//Boolean 		PointOnBoundaryLine(WorldPoint p);
	double 			GetBreakingWaveHeight(void);
	//void 			TrackOutputData(TOLEList *thisLEList);
	virtual			void TrackOutputData(void);
	virtual			void TrackOutputDataInAllLayers(void);
	OSErr 			DoAnalysisMenuItems(long menuCodedItemID);
	
	virtual long		GetListLength();
	virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	
	Rect 			DoArrowTool(long triNum);
	//OSErr 			GetDepthAtMaxTri(TOLEList *thisLEList,long *maxTriIndex,double *depthAtPnt);	
	OSErr 			GetDepthAtMaxTri(long *maxTriIndex, double *depthAtPnt);	
	//OSErr 			CreateDepthSlice(TLEList *thisLEList, long triNum)	;
	OSErr 			CreateDepthSlice(long triNum, float **depthSlice);
	//OSErr 			CreateDepthSlice(long triNum, float *depthSlice);
	void 			DoLassoTool(Point p);
	void 			MarkRect(Point p);
	
	// I/O methods
	virtual OSErr 	Read  (BFPB *bfpb); // read from the current position
	virtual OSErr 	Write (BFPB *bfpb); // write to the current position
	
	virtual	long 	GetNumBoundarySegs(void);
	virtual  long 	GetNumPointsInBoundarySeg(long segno);
	virtual	long 	GetNumBoundaryPts(void);
	long 	CountLEsOnSelectedBeach();
	OSErr ExportOiledShorelineData(OiledShorelineDataHdl oiledShorelineHdl);
	void 	FindStartEndSeg(long ptnum,long *startPt, long *endPt);
	long 	NextPointOnSeg(long segno, long point);
	long 	PrevPointOnSeg(long segno, long point);
	Boolean 	MoreBoundarySegments(long *a,long *b);
	void 	InitBoundaryIter(Boolean clockwise,long segno, long startno, long endno);
	double 	PathLength(Boolean selectionDirection,long segNo, long startno, long endno);
	Boolean 	IsBoundaryPoint(long pt);
	virtual	long 	GetNumContourLevels(void);
	virtual	float			GetMaxDepth2(void);
	DropletInfoRecH	GetDropletSizesH(void) {return fDropletSizesH;}		
};


#endif