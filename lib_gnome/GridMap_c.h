/*
 *  GridMap.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __GridMap_c__
#define __GridMap_c__

#include <string>
#include <vector>

using namespace std;

#include "Basics.h"
#include "TypeDefs.h"
#include "RectUtils.h"
#include "ClassID_c.h"
#include "my_build_list.h"
#include "GridMapUtils.h"


#ifdef pyGNOME
#include "GridVel_c.h"
#include "TriGridVel_c.h"
#define TGridVel GridVel_c
#define TTriGridVel TriGridVel_c
#else
#include "GridVel.h"
#endif

class DLL_API GridMap_c : virtual public ClassID_c
{
	
public:
	WorldRect		fMapBounds; 	// bounding rectangle of map
	TGridVel		*fGrid;
	LONGH			fBoundarySegmentsH;
	LONGH			fBoundaryTypeH;		// 1 land, 2 water
	LONGH			fBoundaryPointsH;	// for curvilinear grids
	
	//short			fGridType;
	//short			fVerticalGridType;
	
public:
	GridMap_c ();
	//virtual ~GridMap_c (){}
	virtual ~GridMap_c () {Dispose ();}

	virtual void	Dispose ();
	virtual OSErr	InitMap();
	//virtual ClassID GetClassID () { return TYPE_GRIDMAP; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_GRIDMAP) return TRUE; return ClassID_c::IAm(id); }

	WorldRect		GetMapBounds () { return fMapBounds; }
	void			SetMapBounds (WorldRect newBounds) { fMapBounds = newBounds; }

	virtual	void 	SetBoundarySegs(LONGH boundarySegs) { fBoundarySegmentsH = boundarySegs; }
	virtual	void 	SetWaterBoundaries(LONGH waterBoundaries) { fBoundaryTypeH = waterBoundaries; }
	virtual	void 	SetBoundaryPoints(LONGH boundaryPts) { fBoundaryPointsH = boundaryPts; }
	virtual	LONGH 	GetBoundarySegs() { return fBoundarySegmentsH; }
	virtual	LONGH 	GetWaterBoundaries() { return fBoundaryTypeH; }
	virtual	LONGH 	GetBoundaryPoints() { return fBoundaryPointsH; }

	TTriGridVel* 	GetGrid();
	LongPointHdl 	GetPointsHdl();	

	Boolean			InVerticalMap(WorldPoint3D wp);
	virtual	double			DepthAtPoint(WorldPoint wp);
	double 			DepthAtCentroid(long triNum);

	virtual	long 	GetNumBoundarySegs(void);
	virtual  long 	GetNumPointsInBoundarySeg(long segno);
	virtual	long 	GetNumBoundaryPts(void);
	
	OSErr			SetUpCurvilinearGrid(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg);	
	OSErr			SetUpCurvilinearGrid2(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg);	
	OSErr			SetUpTriangleGrid2(long numNodes, long numTri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors);
	OSErr			SetUpTriangleGrid(long numNodes, long numTri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts);

	OSErr ReadTopology(std::vector<std::string> &linesInFile);
	OSErr ReadTopology(char *path);

	OSErr	TextRead(char *path);
	OSErr	ExportTopology(char* path);
	OSErr	SaveAsNetCDF(char *path);
	OSErr	ReadCATSMap(vector<string> &linesInFile); 
	OSErr	ReadCATSMap(char *path); 
	OSErr	GetPointsAndMask(char *path,DOUBLEH *maskH,WORLDPOINTFH *vertexPtsH, FLOATH *depthPtsH, long *numRows, long *numCols);	
	//OSErr	GetPointsAndBoundary2(char *path,long *numTri,WORLDPOINTFH *vertexPtsH, FLOATH *depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri);
	OSErr	GetPointsAndBoundary(char *path,WORLDPOINTFH *vertexPtsH, FLOATH *depthPtsH, long *numNodes, LONGPTR *bndry_indices, LONGPTR *bndry_nums, LONGPTR *bndry_type, long *numBoundaryPts, LONGPTR *tri_verts, LONGPTR *tri_neighbors, long *ntri);
	
	Boolean 	IsBoundaryPoint(long pt);

	virtual void 	DrawBoundaries(Rect r);	// just to have the vertex index access handy
	virtual void 	DrawBoundaries2(Rect r); // just to have the vertex index access handy
};

OSErr 	NumberIslands(LONGH *islandNumberH, DOUBLEH landmaskH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);

#undef TGridVel
#undef TTriGridVel
#endif
