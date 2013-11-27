/*
 *  PtCurMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __PtCurMap_c__
#define __PtCurMap_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "GuiTypeDefs.h"
#include "Map_c.h"
#include "RectUtils.h"

#ifdef pyGNOME
#include "Mover_c.h"
#include "CurrentMover_c.h"
#include "TriGridVel_c.h"
#include "TriGridVel3D_c.h"
#define TMover Mover_c
#define TCurrentMover CurrentMover_c
#define TTriGridVel TriGridVel_c
#define TTriGridVel3D TriGridVel3D_c
#endif

class PtCurMap_c : virtual public Map_c
{
	
public:
	LONGH			fBoundarySegmentsH;
	LONGH			fBoundaryTypeH;		// 1 land, 2 water
	LONGH			fBoundaryPointsH;	// for curvilinear grids
	LONGH			fSegSelectedH;
	LONGH			fSelectedBeachHdl;	//not sure if both are needed
	LONGH			fSelectedBeachFlagHdl;	//not sure if both are needed

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
	
	WorldRect		fBitMapBounds;
	Boolean			fUseBitMapBounds;
	Boolean			bDrawBitMapBounds;
	
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
	PtCurMap_c (char* name, WorldRect bounds);
	PtCurMap_c () {}

	virtual OSErr	InitMap();
	virtual OSErr	InitDropletSizes();
	//virtual ClassID GetClassID () { return TYPE_PTCURMAP; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMAP) return TRUE; return TMap::IAm(id); }
	virtual OSErr	InitContourLevels();

	virtual	void 	SetBoundarySegs(LONGH boundarySegs) { fBoundarySegmentsH = boundarySegs; }
	virtual	void 	SetWaterBoundaries(LONGH waterBoundaries) { fBoundaryTypeH = waterBoundaries; }
	virtual	void 	SetBoundaryPoints(LONGH boundaryPts) { fBoundaryPointsH = boundaryPts; }
	virtual	LONGH 	GetBoundarySegs() { return fBoundarySegmentsH; }
	virtual	LONGH 	GetWaterBoundaries() { return fBoundaryTypeH; }
	virtual	LONGH 	GetBoundaryPoints() { return fBoundaryPointsH; }
	Boolean 	MoreSegments(LONGH segh,long *startIndex, long *endIndex,long *curIndex);
	void			SetMinDistOffshore(WorldRect wBounds);
	long 	WhichSelectedSegAmIIn(long index);
	

	void		 	SetBitMapBounds (WorldRect newBounds) {fBitMapBounds = newBounds;}
	virtual WorldRect	GetMapBounds () { if (fUseBitMapBounds) return fBitMapBounds; else return fMapBounds; }
	virtual	Boolean	HaveMapBoundsLayer (void) { return true; }
	virtual long	PointOnWhichSeg(long p);
	virtual	long 	PointOnWhichSeg(long longVal, long latVal, long *startver, long *endver, float *dist);
	Boolean 	ContiguousPoints(long p1, long p2);

	TMover* 		GetMover(ClassID desiredClassID);
	TCurrentMover* 	Get3DCurrentMover();
	double			GetSpillStartDepth();
	Boolean			ThereIsADispersedSpill();
	TTriGridVel* 	GetGrid(Boolean wantRefinedGrid);
	virtual TTriGridVel3D* 	GetGrid3D(Boolean wantRefinedGrid);
	LongPointHdl 	GetPointsHdl(Boolean useRefinedGrid);	
	virtual Boolean		CanReFloat (Seconds time, LERec *theLE);
	//		virtual Boolean	CanReFloat (Seconds time, LERec *theLE) { return true; }

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
	double 			GetMixedLayerDepth(void) {return fMixedLayerDepth;}
	//OSErr 			GetDepthAtMaxTri(TOLEList *thisLEList,long *maxTriIndex,double *depthAtPnt);	
	OSErr 			GetDepthAtMaxTri(long *maxTriIndex, double *depthAtPnt);	
	//OSErr 			CreateDepthSlice(TLEList *thisLEList, long triNum)	;
	OSErr 			CreateDepthSlice(long triNum, float **depthSlice);
	//OSErr 			CreateDepthSlice(long triNum, float *depthSlice);

	virtual	long 	GetNumBoundarySegs(void);
	virtual  long 	GetNumPointsInBoundarySeg(long segno);
	virtual	long 	GetNumBoundaryPts(void);
	
	void	FindNearestBoundary(WorldPoint wp, long *verNum, long *segNo);

	long 	CountLEsOnSelectedBeach();
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

OSErr SetDefaultContours(DOUBLEH contourLevels, short contourType);
OSErr SetDefaultDropletSizes(DropletInfoRecH dropletSizes);

#undef TMover
#undef TCurrentMover
#undef TTriGridVel
#undef TTriGridVel3D
#endif
