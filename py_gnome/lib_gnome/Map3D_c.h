/*
 *  Map3D.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __Map3D_c__
#define __Map3D_c__

#include "Basics.h"
#include "TypeDefs.h"
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
#else
#include "GridVel.h"
#endif

class Map3D_c : virtual public Map_c
{
	
public:
	//TTriGridVel		*fGrid;
	TGridVel		*fGrid;
	LONGH			fBoundarySegmentsH;
	LONGH			fBoundaryTypeH;		// 1 land, 2 water
	LONGH			fBoundaryPointsH;	// for curvilinear grids

	//Boolean			bDrawLandBitMap;
	//Boolean			bDrawWaterBitMap;
	
public:
	//Boolean			bShowGrid;
	//Boolean			bShowDepthContours;
	short			fGridType;
	short			fVerticalGridType;
	//Rect			fLegendRect;
	//Boolean			bShowLegend;
	//short			fDiagnosticStrType;		// 0 no diagnostic string, 1 tri area, 2 num LEs, 3 conc levels, 4 depths, 5 subsurface particles
	//Boolean			bDrawContours;
	//DOUBLEH			fContourLevelsH;
	
	//long			fWaterDensity;
	//double			fMixedLayerDepth;
	//double			fBreakingWaveHeight;
		
	//Boolean			bUseSmoothing;
	//Boolean			bUseLineCrossAlgorithm;
	float			fMinDistOffshore;	// set how far LEs reflect off shoreline, so they don't get stuck
	//Boolean			bShowElapsedTime;	// should be a model field
	//short			fWaveHtInput;	// 0 from wind speed, 1 breaking wave height measure, 2 significant wave height measure
	//DropletInfoRecH	fDropletSizesH;
	
public:
	Map3D_c (char* name, WorldRect bounds);
	Map3D_c () {}

	virtual OSErr	InitMap();
	//virtual ClassID GetClassID () { return TYPE_MAP3D; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_MAP3D) return TRUE; return TMap::IAm(id); }

	virtual	void 	SetBoundarySegs(LONGH boundarySegs) { fBoundarySegmentsH = boundarySegs; }
	virtual	void 	SetWaterBoundaries(LONGH waterBoundaries) { fBoundaryTypeH = waterBoundaries; }
	virtual	void 	SetBoundaryPoints(LONGH boundaryPts) { fBoundaryPointsH = boundaryPts; }
	virtual	LONGH 	GetBoundarySegs() { return fBoundarySegmentsH; }
	virtual	LONGH 	GetWaterBoundaries() { return fBoundaryTypeH; }
	virtual	LONGH 	GetBoundaryPoints() { return fBoundaryPointsH; }
	
	Boolean 	MoreSegments(LONGH segh,long *startIndex, long *endIndex,long *curIndex);
	void			SetMinDistOffshore(WorldRect wBounds);
	

	virtual	Boolean	HaveMapBoundsLayer (void) { return true; }
	//virtual long	PointOnWhichSeg(long p);
	virtual	long 	PointOnWhichSeg(long longVal, long latVal, long *startver, long *endver, float *dist);
	Boolean 	ContiguousPoints(long p1, long p2);

	TMover* 		GetMover(ClassID desiredClassID);
	TCurrentMover* 	Get3DCurrentMover();
	//double			GetSpillStartDepth();
	Boolean			ThereIsADispersedSpill();
	TTriGridVel* 	GetGrid();
	void			SetGrid(TGridVel *grid) {fGrid = grid;}
	LongPointHdl 	GetPointsHdl();	
	virtual Boolean		CanReFloat (Seconds time, LERec *theLE);

	Boolean			InVerticalMap(WorldPoint3D wp);
	//float 			GetMaxDepth(void);
	WorldPoint3D	ReflectPoint(WorldPoint3D fromWPt,WorldPoint3D toWPt,WorldPoint3D wp);
	virtual	double			DepthAtPoint(WorldPoint wp);
	double 			DepthAtCentroid(long triNum);
	WorldPoint3D 	TurnLEAlongShoreLine(WorldPoint3D waterPoint, WorldPoint3D beachedPoint, WorldPoint3D toPoint);
	//double 			GetBreakingWaveHeight(void);
	//double 			GetMixedLayerDepth(void) {return fMixedLayerDepth;}
	//OSErr 			GetDepthAtMaxTri(TOLEList *thisLEList,long *maxTriIndex,double *depthAtPnt);	
	OSErr 			GetDepthAtMaxTri(long *maxTriIndex, double *depthAtPnt);	
	virtual	long 	GetNumBoundarySegs(void);
	virtual  long 	GetNumPointsInBoundarySeg(long segno);
	virtual	long 	GetNumBoundaryPts(void);
	
	OSErr			SetUpCurvilinearGrid(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg);	
	OSErr			SetUpCurvilinearGrid2(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg);	
	OSErr			SetUpTriangleGrid2(long numNodes, long numTri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors);
	OSErr			SetUpTriangleGrid(long numNodes, long numTri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts);

	Boolean 	IsBoundaryPoint(long pt);
	//virtual	long 	GetNumContourLevels(void);
	virtual	float			GetMaxDepth2(void);
	//DropletInfoRecH	GetDropletSizesH(void) {return fDropletSizesH;}	

};

//OSErr SetDefaultContours(DOUBLEH contourLevels, short contourType);
//OSErr SetDefaultDropletSizes(DropletInfoRecH dropletSizes);

#undef TMover
#undef TCurrentMover
#undef TTriGridVel
#undef TTriGridVel3D
#endif
