/*
 *  GridMap_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <cstdio>

#include "GridMap_c.h"
#include "TimeGridVel_c.h"
#include "MemUtils.h"
#include "StringFunctions.h"
#include "CompFunctions.h"
#include "DagTreeIO.h"
#include "netcdf.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

GridMap_c::GridMap_c()
{
	fGrid = 0;
	
	fBoundarySegmentsH = 0;
	fBoundaryTypeH = 0;
	fBoundaryPointsH = 0;
	
	//fVerticalGridType = TWO_D;
	//fGridType = CURVILINEAR;
}


void GridMap_c::Dispose()
{
	if (fBoundarySegmentsH) {
		DisposeHandle((Handle)fBoundarySegmentsH);
		fBoundarySegmentsH = 0;
	}
	
	if (fBoundaryTypeH) {
		DisposeHandle((Handle)fBoundaryTypeH);
		fBoundaryTypeH = 0;
	}
	
	if (fBoundaryPointsH) {
		DisposeHandle((Handle)fBoundaryPointsH);
		fBoundaryPointsH = 0;
	}
	
	if (fGrid)
	{
		fGrid -> Dispose();
//#ifndef pyGNOME
		delete fGrid;
//#endif	
		fGrid = nil;
	}
	
}


TTriGridVel* GridMap_c::GetGrid()
{
	TTriGridVel* triGrid = 0;	
	
	triGrid = dynamic_cast<TTriGridVel*>(fGrid);	// are we sure this is a TriGrid?
	return triGrid;
}


LongPointHdl GridMap_c::GetPointsHdl()	
{
	LongPointHdl ptsHdl = 0;
	TMover *mover=0;
	
	ptsHdl = (dynamic_cast<TTriGridVel*>(fGrid)) -> GetPointsHdl();
	//ptsHdl = ((TTriGridVel*)fGrid) -> GetPointsHdl();

	
	return ptsHdl;
}

Boolean GridMap_c::InVerticalMap(WorldPoint3D wp)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	// don't use refined grid, depths aren't refined
	//TCurrentMover *mover = Get3DCurrentMover();
	
	//if (fGridType==SIGMA_ROMS)
		//depthAtPoint = (double)((NetCDFMoverCurv*)mover)->GetTotalDepth(wp.p,-1);
	//else
	{
		if (!triGrid) return false; // some error alert, no depth info to check
		interpolationVal = triGrid->GetInterpolationValues(wp.p);
		depthsHdl = triGrid->GetBathymetry();
		if (!depthsHdl) return false;	// some error alert, no depth info to check
		if (interpolationVal.ptIndex1<0)	
		{
			//printError("Couldn't find point in dagtree"); 
			return false;
		}
		
		depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
		depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
		depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
		depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;
	}
	if (wp.z >= depthAtPoint || wp.z < 0)	// allow surface but not bottom
		return false;
	else
		return true;
}

double GridMap_c::DepthAtPoint(WorldPoint wp)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	InterpolationVal interpolationVal;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	// don't use refined grid, depths aren't refined
	//TCurrentMover* mover = Get3DCurrentMover();
	
	//if (mover && mover->IAm(TYPE_NETCDFMOVERCURV) && ((NetCDFMoverCurv*)mover)->fVar.gridType==SIGMA_ROMS)
		//return (double)((NetCDFMoverCurv*)mover)->GetTotalDepth(wp,-1);
	
	if (!triGrid) return -1; // some error alert, no depth info to check
	interpolationVal = triGrid->GetInterpolationValues(wp);
	depthsHdl = triGrid->GetBathymetry();
	if (!depthsHdl) return -1;	// some error alert, no depth info to check
	if (interpolationVal.ptIndex1<0)	
	{
		//printError("Couldn't find point in dagtree"); 
		return -1;
	}
	
	depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
	depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
	depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
	depthAtPoint = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;
	
	return depthAtPoint;
}

double GridMap_c::DepthAtCentroid(long triNum)
{
	float depth1,depth2,depth3;
	double depthAtPoint;	
	long ptIndex1,ptIndex2,ptIndex3;
	FLOATH depthsHdl = 0;
	TTriGridVel* triGrid = GetGrid();	
	
	TopologyHdl topH ;
	
	if (triNum < 0) return -1;
	if (!triGrid) return -1; // some error alert, no depth info to check
	
	topH = triGrid -> GetTopologyHdl();
	if (!topH) return -1;
	
	ptIndex1 = (*topH)[triNum].vertex1;
	ptIndex2 = (*topH)[triNum].vertex2;
	ptIndex3 = (*topH)[triNum].vertex3;
	
	depthsHdl = triGrid->GetDepths();
	if (!depthsHdl) return -1;	// some error alert, no depth info to check
	
	depth1 = (*depthsHdl)[ptIndex1];
	depth2 = (*depthsHdl)[ptIndex2];
	depth3 = (*depthsHdl)[ptIndex3];
	depthAtPoint = (depth1 + depth2 + depth3) / 3.;
	
	return depthAtPoint;
}


/////////////////////////////////////////////////

long GridMap_c::GetNumBoundarySegs(void)
{
	long numInHdl = 0;
	if (fBoundarySegmentsH) numInHdl = _GetHandleSize((Handle)fBoundarySegmentsH)/sizeof(**fBoundarySegmentsH);
	
	return numInHdl;
}

long GridMap_c::GetNumPointsInBoundarySeg(long segno)
{
	if (fBoundarySegmentsH) return (*fBoundarySegmentsH)[segno] - (segno==0? 0: (*fBoundarySegmentsH)[segno-1]+1) + 1;
	else return 0;
}

long GridMap_c::GetNumBoundaryPts(void)
{
	long numInHdl = 0;
	if (fBoundaryTypeH) numInHdl = _GetHandleSize((Handle)fBoundaryTypeH)/sizeof(**fBoundaryTypeH);
	
	return numInHdl;
}

Boolean GridMap_c::IsBoundaryPoint(long pt)
{
	return pt < GetNumBoundaryPts();
}


OSErr GridMap_c::InitMap()
{
	OSErr err = 0;
	//	anything here ?
	return err;
}


/////////////////////////////////////////////////

OSErr GridMap_c::SetUpCurvilinearGrid(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg)
{
	long i, j, n, ntri, numVerdatPts=0; 
	long numRows_ext = numRows+1, numCols_ext = numCols+1;
	long nv = numRows * numCols, nv_ext = numRows_ext*numCols_ext;
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long iIndex,jIndex,index,currentIndex,startIndex; 
	long triIndex1,triIndex2,waterCellNum=0;
	long ptIndex = 0,cellNum = 0,diag = 1;
	Boolean foundPt = false, isOdd;
	OSErr err = 0;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(numRows * numCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	FLOATH depths=0;
	
	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landMaskH,i*numCols+j)==0)	// land point
			{
				INDEXH(landWaterInfo,i*numCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				INDEXH(landWaterInfo,i*numCols+j) = 1;
				INDEXH(ptIndexHdl,i*numCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*numCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*numCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*numCols_ext+j+1) = -2;
			}
		}
	}
	
	for (i=0;i<numRows_ext;i++)
	{
		for (j=0;j<numCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*numCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*numCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*numCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols+j)>0)
			{
				INDEXH(gridCellInfo,i*numCols+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*numCols+j).topLeft = INDEXH(ptIndexHdl,i*numCols_ext+j);
				INDEXH(gridCellInfo,i*numCols+j).topRight = INDEXH(ptIndexHdl,i*numCols_ext+j+1);
				INDEXH(gridCellInfo,i*numCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*numCols_ext+j);
				INDEXH(gridCellInfo,i*numCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*numCols_ext+j+1);
			}
			else INDEXH(gridCellInfo,i*numCols+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv_ext;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			//INDEXH(verdatPtsH,INDEXH(ptIndexHdl,i)) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	depths = (FLOATH)_NewHandle(sizeof(float)*(numVerdatPts));
	if(pts == nil || depths == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/numCols_ext;
			jIndex = n%numCols_ext;
			if (iIndex==0)
			{
				if (jIndex<numCols)
				{
					dLat = INDEXH(vertexPtsH,numCols+jIndex).pLat - INDEXH(vertexPtsH,jIndex).pLat;
					fLat = INDEXH(vertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(vertexPtsH,numCols+jIndex).pLong - INDEXH(vertexPtsH,jIndex).pLong;
					fLong = INDEXH(vertexPtsH,jIndex).pLong - dLon;
					fDepth = INDEXH(depthPtsH,jIndex);
				}
				else
				{
					dLat1 = (INDEXH(vertexPtsH,jIndex-1).pLat - INDEXH(vertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(vertexPtsH,numCols+jIndex-1).pLat - INDEXH(vertexPtsH,numCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(vertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(vertexPtsH,numCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(vertexPtsH,numCols+jIndex-1).pLong - INDEXH(vertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(vertexPtsH,numCols+jIndex-2).pLong - INDEXH(vertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(vertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(vertexPtsH,jIndex-2).pLong-dLon2);
					fDepth = INDEXH(depthPtsH,jIndex-1);
				}
			}
			else 
			{
				if (jIndex<numCols)
				{
					fLat = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex).pLat;
					fLong = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex).pLong;
					fDepth = INDEXH(depthPtsH,(iIndex-1)*numCols+jIndex);
				}
				else
				{
					dLat = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLat - INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-2).pLat;
					fLat = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLong - INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-2).pLong;
					fLong = INDEXH(vertexPtsH,(iIndex-1)*numCols+jIndex-1).pLong + dLon;
					fDepth = INDEXH(depthPtsH,(iIndex-1)*numCols+jIndex-1);
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			INDEXH(pts,i) = vertex;
			INDEXH(depths,i) = fDepth;
		}
		else { // for outputting a verdat the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
		
	}
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*numCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*numCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*numCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*numCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*numCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*numCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*numCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*numCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*numCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*numCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*numCols+j).topRight;
			if (j==numCols-1 || INDEXH(gridCellInfo,i*numCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*numCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==numRows-1 || INDEXH(gridCellInfo,(i+1)*numCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*numCols+j).cellNum * 2;
			}
		}
	}
	
	
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
	
	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**waterBoundaryPtsH));
	flagH = (LONGH)_NewHandleClear(nv_ext * sizeof(**flagH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Numbering Islands");
	MySpinCursor(); // JLM 8/4/99
	err = NumberIslands(&maskH2, landMaskH, landWaterInfo, numRows, numCols, &numIslands);	// numbers start at 3 (outer boundary)
	MySpinCursor(); // JLM 8/4/99
	if (err) goto done;
	for (i=0;i<ntri;i++)
	{
		if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
		// the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
		// odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
		if ((*topo)[i].adjTri1 == -1)
		{
			// add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
			(*segList)[segNum].pt1 = (*topo)[i].vertex2;
			(*segList)[segNum].pt2 = (*topo)[i].vertex3;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check left rectangle for L/W border 
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (jIndex>0 && INDEXH(maskH2,iIndex*numCols_ext + jIndex-1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols_ext + jIndex-1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			else 
			{	
				// check right rectangle for L/W border convert back to row/col
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (jIndex<numCols && INDEXH(maskH2,iIndex*numCols_ext + jIndex+1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols_ext + jIndex+1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			segNum++;
		}
		
		if ((*topo)[i].adjTri3 == -1)
		{
			// add segment pt 1 - pt 2 to list
			// odd top, even bottom
			(*segList)[segNum].pt1 = (*topo)[i].vertex1;
			(*segList)[segNum].pt2 = (*topo)[i].vertex2;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check top rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*numCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*numCols_ext + jIndex);
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;
				}
			}
			else 
			{
				// check bottom rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/numCols_ext;
				jIndex = rectIndex%numCols_ext;
				if (iIndex<numRows && INDEXH(maskH2,(iIndex+1)*numCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*numCols_ext + jIndex);		// this should be the neighbor's value
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;		
				}
			}
			segNum++;
		}
	}
	nSegs = segNum;
	_SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	_SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	// go through list of segments, and make list of boundary segments
	// as segment is taken mark so only use each once
	// get a starting point, add the first and second to the list
	islandNum = 3;
findnewstartpoint:
	if (islandNum > numIslands) 
	{
		//err = -1; goto done;
		_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	}
	foundPt = false;
	for (i=0;i<nSegs;i++)
	{
		if ((*segUsed)[i]) continue;
		waterStartPoint = nBoundaryPts;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		(*flagH)[(*segList)[i].pt1] = 1;
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		(*flagH)[(*segList)[i].pt2] = 1;
		currentIndex = (*segList)[i].pt2;
		startIndex = (*segList)[i].pt1;
		currentIsland = (*segList)[i].islandNumber;	
		foundPt = true;
		(*segUsed)[i] = true;
		break;
	}
	if (!foundPt)
	{
		printNote("Lost trying to set boundaries");
		err = -1; goto done;
		// clean up handles and set grid without a map
		/*if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		 if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		 if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		 goto setFields;*/
	}
	
findnextpoint:
	for (i=0;i<nSegs;i++)
	{
		// look for second point of the previous selected segment, add the second to point list
		if ((*segUsed)[i]) continue;
		if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		index = (*segList)[i].pt1;
		if (index == currentIndex)	// found next point
		{
			currentIndex = (*segList)[i].pt2;
			(*segUsed)[i] = true;
			if (currentIndex == startIndex) // completed a segment
			{
				islandNum++;
				(*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				(*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				goto findnewstartpoint;
			}
			else
			{
				(*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				(*flagH)[(*segList)[i].pt2] = 1;
				(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				nBoundaryPts++;
				goto findnextpoint;
			}
		}
	}
	// shouldn't get here unless there's a problem...
	_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
	
setFields:	
	
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in GridMap::SetUpCurvilinearGrid()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	triGrid -> SetDepths(depths);	// used by PtCurMap to check vertical movement
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0;
	
	if (waterBoundaryPtsH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundaryEndPtsH);	
		this->SetWaterBoundaries(waterBoundaryPtsH);
		this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);		
	}
	else
	{
		err = -1;
		goto done;
	}
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
	if (segList) {DisposeHandle((Handle)segList); segList = 0;}
	if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}
	if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridMap_c::SetUpCurvilinearGrid");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		//if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		//if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		//if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		//if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}	
		//if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}	
		
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}
	return err;
}

OSErr GridMap_c::SetUpCurvilinearGrid2(DOUBLEH landMaskH, long numRows, long numCols, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, char* errmsg)
{	// this is for the points on nodes case (old coops_mask)
	OSErr err = 0;
	long i,j,k;
	char *velUnits=0; 
	long latlength = numRows, numtri = 0;
	long lonlength = numCols;
	Boolean isLandMask = true;
	float fDepth1, fLat1, fLong1;
	long index1=0;
	
	errmsg[0]=0;
	
	long n, ntri, numVerdatPts=0;
	long numRows_minus1 = numRows-1, numCols_minus1 = numCols-1;
	long nv = numRows * numCols;
	long nCells = numRows_minus1 * numCols_minus1;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
	
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long currentIndex,startIndex; 
	long diag = 1;
	Boolean foundPt = false, isOdd;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(nCells * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nCells * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	FLOATH depths=0;
	
	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	/////////////////////////////////////////////////
	
	if (!landMaskH) return -1;
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	index1 = 0;
	for (i=0;i<numRows-1;i++)
	{
		for (j=0;j<numCols-1;j++)
		{
			if (INDEXH(landMaskH,i*numCols+j)==0)	// land point
			{
				INDEXH(landWaterInfo,i*numCols_minus1+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				if (INDEXH(landMaskH,(i+1)*numCols+j)==0 || INDEXH(landMaskH,i*numCols+j+1)==0 || INDEXH(landMaskH,(i+1)*numCols+j+1)==0)
				{
					INDEXH(landWaterInfo,i*numCols_minus1+j) = -1;	// may want to mark each separate island with a unique number
				}
				else
				{
					INDEXH(landWaterInfo,i*numCols_minus1+j) = 1;
					INDEXH(ptIndexHdl,i*numCols+j) = -2;	// water box
					INDEXH(ptIndexHdl,i*numCols+j+1) = -2;
					INDEXH(ptIndexHdl,(i+1)*numCols+j) = -2;
					INDEXH(ptIndexHdl,(i+1)*numCols+j+1) = -2;
				}
			}
		}
	}
	
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(ptIndexHdl,i*numCols+j) == -2)
			{
				INDEXH(ptIndexHdl,i*numCols+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*numCols+j) = -1;
		}
	}
	
	for (i=0;i<numRows-1;i++)
	{
		for (j=0;j<numCols-1;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols_minus1+j)>0)
			{
				INDEXH(gridCellInfo,i*numCols_minus1+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*numCols_minus1+j).topLeft = INDEXH(ptIndexHdl,i*numCols+j);
				INDEXH(gridCellInfo,i*numCols_minus1+j).topRight = INDEXH(ptIndexHdl,i*numCols+j+1);
				INDEXH(gridCellInfo,i*numCols_minus1+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*numCols+j);
				INDEXH(gridCellInfo,i*numCols_minus1+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*numCols+j+1);
			}
			else INDEXH(gridCellInfo,i*numCols_minus1+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	//index = 0;
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/numCols;
			jIndex = n%numCols;
			//fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
			//fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
			fLat = INDEXH(vertexPtsH,(iIndex)*numCols+jIndex).pLat;
			fLong = INDEXH(vertexPtsH,(iIndex)*numCols+jIndex).pLong;
			fDepth = INDEXH(depthPtsH,(iIndex)*numCols+jIndex);
			
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			//fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
		
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<numRows_minus1;i++)
	{
		for (j=0;j<numCols_minus1;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols_minus1+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*numCols_minus1+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*numCols_minus1+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*numCols_minus1+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*numCols_minus1+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*numCols_minus1+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*numCols_minus1+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*numCols_minus1+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*numCols_minus1+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*numCols_minus1+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*numCols_minus1+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*numCols_minus1+j).topRight;
			if (j==numCols-2 || INDEXH(gridCellInfo,i*numCols_minus1+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*numCols_minus1+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==numRows-2 || INDEXH(gridCellInfo,(i+1)*numCols_minus1+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*numCols_minus1+j).cellNum * 2;
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
	
	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv * sizeof(**waterBoundaryPtsH));
	flagH = (LONGH)_NewHandleClear(nv * sizeof(**flagH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Numbering Islands");
	MySpinCursor(); // JLM 8/4/99
	//err = NumberIslands(&maskH2, velocityH, landWaterInfo, fNumRows_minus1, fNumCols_minus1, &numIslands);	// numbers start at 3 (outer boundary)
	err = NumberIslands(&maskH2, landMaskH, landWaterInfo, numRows_minus1, numCols_minus1, &numIslands);	// numbers start at 3 (outer boundary)
	//numIslands++;	// this is a special case for CBOFS, right now the only coops_mask example
	MySpinCursor(); // JLM 8/4/99
	if (err) goto done;
	for (i=0;i<ntri;i++)
	{
		if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
		// the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
		// odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
		if ((*topo)[i].adjTri1 == -1)
		{
			// add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
			(*segList)[segNum].pt1 = (*topo)[i].vertex2;
			(*segList)[segNum].pt2 = (*topo)[i].vertex3;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check left rectangle for L/W border 
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/numCols;
				jIndex = rectIndex%numCols;
				if (jIndex>0 && INDEXH(maskH2,iIndex*numCols + jIndex-1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols + jIndex-1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			else 
			{	
				// check right rectangle for L/W border convert back to row/col
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/numCols;
				jIndex = rectIndex%numCols;
				//if (jIndex<fNumCols && INDEXH(maskH2,iIndex*fNumCols + jIndex+1)>=3)
				if (jIndex<numCols_minus1 && INDEXH(maskH2,iIndex*numCols + jIndex+1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					//(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols + jIndex+1);	
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*numCols + jIndex+1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			segNum++;
		}
		
		if ((*topo)[i].adjTri3 == -1)
		{
			// add segment pt 1 - pt 2 to list
			// odd top, even bottom
			(*segList)[segNum].pt1 = (*topo)[i].vertex1;
			(*segList)[segNum].pt2 = (*topo)[i].vertex2;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check top rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/numCols;
				jIndex = rectIndex%numCols;
				if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*numCols + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*numCols + jIndex);
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;
				}
			}
			else 
			{
				// check bottom rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/numCols;
				jIndex = rectIndex%numCols;
				//if (iIndex<fNumRows && INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex)>=3)
				if (iIndex<numRows_minus1 && INDEXH(maskH2,(iIndex+1)*numCols + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					//(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex);		// this should be the neighbor's value
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*numCols + jIndex);		// this should be the neighbor's value
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;		
				}
			}
			segNum++;
		}
	}
	nSegs = segNum;
	_SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	_SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	// go through list of segments, and make list of boundary segments
	// as segment is taken mark so only use each once
	// get a starting point, add the first and second to the list
	islandNum = 3;
findnewstartpoint:
	if (islandNum > numIslands) 
	{
		//err = -1; goto done
		_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	}
	foundPt = false;
	for (i=0;i<nSegs;i++)
	{
		if ((*segUsed)[i]) continue;
		waterStartPoint = nBoundaryPts;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		(*flagH)[(*segList)[i].pt1] = 1;
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		(*flagH)[(*segList)[i].pt2] = 1;
		currentIndex = (*segList)[i].pt2;
		startIndex = (*segList)[i].pt1;
		currentIsland = (*segList)[i].islandNumber;	
		foundPt = true;
		(*segUsed)[i] = true;
		break;
	}
	if (!foundPt)
	{
		printNote("Lost trying to set boundaries");
		err = -1; goto done;
		// clean up handles and set grid without a map
		//if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		//if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		//if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		//goto setFields;
	}
	
findnextpoint:
	for (i=0;i<nSegs;i++)
	{
		// look for second point of the previous selected segment, add the second to point list
		if ((*segUsed)[i]) continue;
		if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		index = (*segList)[i].pt1;
		if (index == currentIndex)	// found next point
		{
			currentIndex = (*segList)[i].pt2;
			(*segUsed)[i] = true;
			if (currentIndex == startIndex) // completed a segment
			{
				islandNum++;
				(*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				(*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				goto findnewstartpoint;
			}
			else
			{
				(*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				(*flagH)[(*segList)[i].pt2] = 1;
				(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				nBoundaryPts++;
				goto findnextpoint;
			}
		}
	}
	// shouldn't get here unless there's a problem...
	_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
	
setFields:	
	
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TimeGridVelCurv_c::ReorderPointsCOOPSMask()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	
	if (waterBoundaryPtsH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundaryEndPtsH);	
		this->SetWaterBoundaries(waterBoundaryPtsH);
		this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);		
	}
	else
	{
		err = -1;
		goto done;
	}
	
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
	if (segList) {DisposeHandle((Handle)segList); segList = 0;}
	if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}
	if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridMap_c::SetUpCurvilinearGrid2");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		//if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		//if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		//if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		//if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		//if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
		
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}
	
	return err;	
}

OSErr GridMap_c::SetUpTriangleGrid2(long numNodes, long ntri, WORLDPOINTFH vertexPtsH, FLOATH depthPtsH, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = numNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;
	
	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));
	
	TopologyHdl topo=0;
	DAGTreeStruct tree;
	
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	FLOATH depths = 0;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	LONGH boundaryPtsH = 0;
	
	TTriGridVel *triGrid = nil;
	
	Boolean addOne = false;	// for debugging
	
	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list
	
	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		//short islandNum, index;
		long islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
		//INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;
	
	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// it seems this should be an error...
		err = -1;
		goto done;
		// shrink handle
		//_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}
	
	numVerdatPts = nv;	//for now, may reorder later
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	depths = (FLOATH)_NewHandle(sizeof(float)*(numVerdatPts));
	if(pts == nil || depths == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	//numVerdatPts = nv;	//for now, may reorder later
	for (i=0; i<=numVerdatPts; i++)
	{
		//long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			//index = i+1;
			//n = INDEXH(verdatPtsH,i);
			n = i;	// for now, not sure if need to reorder
			fLat = INDEXH(vertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(vertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = INDEXH(depthPtsH,n);	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
			INDEXH(depths,i) = fDepth;
		}
		else { // the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
		}
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	//if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology))))goto done;	
	
	// point and triangle indices should start with zero
	for(i = 0; i < 3*ntri; i ++)
	{
		//if (tri_neighbors[i]==0)
		//tri_neighbors[i]=-1;
		//else 
		tri_neighbors[i] = tri_neighbors[i] - 1;
		tri_verts[i] = tri_verts[i] - 1;
	}
	for(i = 0; i < ntri; i ++)
	{	// topology data needs to be CCW
		(*topo)[i].vertex1 = tri_verts[i];
		//(*topo)[i].vertex2 = tri_verts[i+ntri];
		(*topo)[i].vertex3 = tri_verts[i+ntri];
		//(*topo)[i].vertex3 = tri_verts[i+2*ntri];
		(*topo)[i].vertex2 = tri_verts[i+2*ntri];
		(*topo)[i].adjTri1 = tri_neighbors[i];
		//(*topo)[i].adjTri2 = tri_neighbors[i+ntri];
		(*topo)[i].adjTri3 = tri_neighbors[i+ntri];
		//(*topo)[i].adjTri3 = tri_neighbors[i+2*ntri];
		(*topo)[i].adjTri2 = tri_neighbors[i+2*ntri];
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in GridMap_c::SetUpTriangleGrid2()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	triGrid -> SetDepths(depths);
	//if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);	// should be set in TextRead
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	boundaryPtsH = (LONGH)_NewHandleClear(numBoundaryPts * sizeof(**boundaryPtsH));
	if (!boundaryPtsH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
		INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}
	
	if (waterBoundariesH)	// maybe assume rectangle grids will have map?
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(verdatBreakPtsH);	
		this->SetWaterBoundaries(waterBoundariesH);
		this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	
	/////////////////////////////////////////////////
	
done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}
	if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridMap_c::SetUpTriangleGrid2");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	return err;
}

OSErr GridMap_c::SetUpTriangleGrid(long numNodes, long numTri,
								   WORLDPOINTFH vertexPtsH, FLOATH depthPtsH,
								   long *bndry_indices, long *bndry_nums, long *bndry_type,
								   long numBoundaryPts)
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = numNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;
	
	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));
	
	TopologyHdl topo = 0;
	LongPointHdl pts = 0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH = 0;
	FLOATH depths = 0;

	TTriGridVel *triGrid = 0;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	Boolean addOne = false;	// for debugging
	
	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {
		err = memFullErr;
		goto done;
	}
	
	// put boundary points into verdat list
	
	// code goes here, double check that the water boundary info is also reordered
	currentBoundary = 1;
	if (bndry_nums[0] == 0)
		addOne = true;	// for debugging

	for (i = 0; i < numBoundaryPts; i++) {
		long islandNum, index;

		index = bndry_indices[i];
		islandNum = bndry_nums[i];

		if (addOne)
			islandNum++;	// for debugging

		INDEXH(vertFlagsH, index - 1) = 1;	// note that point has been used
		INDEXH(verdatPtsH, numVerdatPts++) = index - 1;	// add to verdat list

		if (islandNum > currentBoundary) {
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH, numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;
	
	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if (INDEXH(vertFlagsH, i) == 0) {
			INDEXH(verdatPtsH, numVerdatPts++) = i;
			INDEXH(vertFlagsH, i) = 0; // mark as used
		}
	}

	if (numVerdatPts != nv) {
		printNote("Not all vertex points were used");
		// it seems this should be an error...
		err = -1;
		goto done;
		// shrink handle
		//_SetHandleSize((Handle)verdatPtsH, numVerdatPts * sizeof(long));
	}

	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	depths = (FLOATH)_NewHandle(sizeof(float)*(numVerdatPts));
	if(pts == 0 || depths == 0) {
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	for (i = 0; i <= numVerdatPts; i++) {
		long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if (i < numVerdatPts)  {
			index = i + 1;
			n = INDEXH(verdatPtsH, i);
			fLat = INDEXH(vertexPtsH, n).pLat;
			fLong = INDEXH(vertexPtsH, n).pLong;

			vertex.v = (long)(fLat * 1e6);
			vertex.h = (long)(fLong * 1e6);

			fDepth = INDEXH(depthPtsH, n);
			INDEXH(pts, i) = vertex;
			INDEXH(depths, i) = fDepth;
		}
		else {
			// the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
	}

	// figure out the bounds
	//cerr << "GridMap_c::SetUpTriangleGrid(): figure out the bounds..." << endl;
	triBounds = voidWorldRect;
	if (pts) {
		LongPoint thisLPoint;

		if (numVerdatPts > 0) {
			WorldPoint  wp;

			for(i = 0; i < numVerdatPts; i++) {
				thisLPoint = INDEXH(pts, i);

				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;

				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// shrink handle
	//cerr << "GridMap_c::SetUpTriangleGrid(): shrink verdatBreakPtsH..." << endl;
	_SetHandleSize((Handle)verdatBreakPtsH, numVerdatBreakPts * sizeof(long));
	for (i = 0; i < numVerdatBreakPts; i++ ) {
		INDEXH(verdatBreakPtsH,i)--;
	}

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column

	MySpinCursor(); // JLM 8/4/99

	//cerr << "GridMap_c::SetUpTriangleGrid(): maketriangles()..." << endl;
	err = maketriangles(&topo, pts, numVerdatPts, verdatBreakPtsH, numVerdatBreakPts);
	if (err)
		goto done;

	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");

	MySpinCursor(); // JLM 8/4/99

	//cerr << "GridMap_c::SetUpTriangleGrid(): MakeDagTree()..." << endl;
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 

	MySpinCursor(); // JLM 8/4/99

	if (errmsg[0]) {
		err = -1;
		goto done;
	}

	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));

	triGrid = new TTriGridVel;
	if (!triGrid) {
		err = true;
		TechError("Error in GridMap_c::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;
	
	//cerr << "GridMap_c::SetUpTriangleGrid(): triGrid->SetBounds()..." << endl;
	triGrid->SetBounds(triBounds);
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	triGrid -> SetDepths(depths);	// used by PtCurMap to check vertical movement
	//if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	numBoundaryPts = INDEXH(verdatBreakPtsH, numVerdatBreakPts - 1) + 1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long) * numBoundaryPts);
	if (!waterBoundariesH) {
		err = memFullErr;
		goto done;
	}

	//cerr << "GridMap_c::SetUpTriangleGrid(): set bounday types..." << endl;
	for (i = 0; i < numBoundaryPts; i++) {
		INDEXH(waterBoundariesH, i) = 1;	// default is land
		if (bndry_type[i] == 1)
			INDEXH(waterBoundariesH, i) = 2;	// water boundary, this marks start point rather than end point...
	}
	
	if (waterBoundariesH) {
		// maybe assume rectangle grids will have map?
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(verdatBreakPtsH);	
		this->SetWaterBoundaries(waterBoundariesH);
		//this->SetBoundaryPoints(boundaryPtsH);
		this->SetMapBounds(triBounds);
	}
	else {
		if (waterBoundariesH) {
			DisposeHandle((Handle)waterBoundariesH);
			waterBoundariesH = 0;
		}
		if (verdatBreakPtsH) {
			DisposeHandle((Handle)verdatBreakPtsH);
			verdatBreakPtsH = 0;
		}
	}

done:

	if (err)
		printError("Error reordering gridpoints into verdat format");

	if (vertFlagsH) {
		DisposeHandle((Handle)vertFlagsH);
		vertFlagsH = 0;
	}
	
	if (verdatPtsH) {	// this is not saved as a field
		DisposeHandle((Handle)verdatPtsH);
		verdatPtsH = 0;
	}
	
	if (err) {
		if (!errmsg[0])
			strcpy(errmsg,"An error occurred in GridMap_c::SetUpTriangleGrid");
		printError(errmsg); 

		if (pts) {
			DisposeHandle((Handle)pts);
			pts = 0;
		}
		if (topo) {
			DisposeHandle((Handle)topo);
			topo = 0;
		}
		if (velH) {
			DisposeHandle((Handle)velH);
			velH = 0;
		}
		if (tree.treeHdl) {
			DisposeHandle((Handle)tree.treeHdl);
			tree.treeHdl = 0;
		}
		if (depths) {
			DisposeHandle((Handle)depths);
			depths = 0;
		}
		
		if (fGrid) {
			fGrid->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundariesH) {
			DisposeHandle((Handle)waterBoundariesH);
			waterBoundariesH = 0;
		}
		if (verdatBreakPtsH) {
			DisposeHandle((Handle)verdatBreakPtsH);
			verdatBreakPtsH = 0;
		}
		/*if (verdatPtsH) {
			DisposeHandle((Handle)verdatPtsH);
			verdatPtsH = 0;
		}*/
	}

	return err;
}


OSErr GridMap_c::GetPointsAndMask(char *path,DOUBLEH *maskH,WORLDPOINTFH *vertexPointsH, FLOATH *depthPointsH, long *numRows, long *numCols)	
{
	// this code is for curvilinear grids
	OSErr err = 0;
	long i,j,k, indexOfStart = 0;
	int status, ncid, latIndexid, lonIndexid, latid, lonid, depthid, mask_id, numdims;
	size_t latLength, lonLength, t_len2;
	float startLat,startLon,endLat,endLon;
	char dimname[NC_MAX_NAME];
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0;
	double *lat_vals=0,*lon_vals=0;
	float *depth_vals=0;
	static size_t latIndex=0,lonIndex=0,ptIndex[2]={0,0};
	static size_t pt_count[2];
	char errmsg[256] = "";
	char *modelTypeStr=0;
	Boolean isLandMask = true, fIsNavy = false;
	static size_t mask_index[] = {0,0};
	static size_t mask_count[2];
	double *landmask = 0, scale_factor = 1; 
	DOUBLEH mylandmaskH=0;
	long fNumRows, fNumCols;
	
	if (!path || !path[0]) return 0;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{	
		if (status != NC_NOERR) {err = -1; goto done;}
	}

	// check number of dimensions - 2D or 3D
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) 
	{
			status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); 
			if (status != NC_NOERR) {fIsNavy = false; /*goto done;*/}
	}	// will need to split for Navy vs LAS
	else 
	{	
		fIsNavy = true;	// do we still need to support separate Navy format?
		// may only need to see keyword is there, since already checked grid type
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) 
		{
			status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); 
			if (status != NC_NOERR) {fIsNavy = false; goto done;}
		}	// will need to split for regridded or non-Navy cases 
		modelTypeStr[t_len2] = '\0';		
	}		
	
	if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "gridy", &latIndexid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimid(ncid, "gridx", &lonIndexid);	//Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		// option to use index values?
		status = nc_inq_varid(ncid, "grid_lat", &latid);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "grid_lon", &lonid);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	else
	{
		for (i=0;i<numdims;i++)
		{
			//if (i == recid) continue;
			status = nc_inq_dimname(ncid,i,dimname);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3) || !strncmpnocase(dimname,"NX",2))
			{
				lonIndexid = i;
			}
			if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3) || !strncmpnocase(dimname,"NY",2))
			{
				latIndexid = i;
			}
		}
		
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_inq_varid(ncid, "LATITUDE", &latid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lat", &latid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_varid(ncid, "LONGITUDE", &lonid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lon", &lonid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(latLength*lonLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	lat_vals = new double[latLength*lonLength]; 
	lon_vals = new double[latLength*lonLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_double(ncid, latid, ptIndex, pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_double(ncid, lonid, ptIndex, pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			// grid ordering does matter for creating ptcurmap, assume increases fastest in x/lon, then in y/lat
			INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(latLength-i-1)*lonLength+j];	
			INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(latLength-i-1)*lonLength+j];
		}
	}
	*vertexPointsH	= vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
		
	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	//if (status != NC_NOERR || fIsNavy) {fGridType = TWO_D;}
	if (status != NC_NOERR) 
	//{err = -1; goto done;}	// should we require bathymetry for maps or not?
	{	// will set the depths to 'infinite' below
		totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
		if (!totalDepthsH) {err = memFullErr; goto done;}
	}
	else
	{	
		totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
		if (!totalDepthsH) {err = memFullErr; goto done;}
		depth_vals = new float[latLength*lonLength];
		if (!depth_vals) {err = memFullErr; goto done;}
		status = nc_get_vara_float(ncid, depthid, ptIndex,pt_count, depth_vals);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
	}
	
	*numRows = latLength;
	*numCols = lonLength;
	fNumRows = latLength;
	fNumCols = lonLength;
	
	mask_count[0] = latLength;
	mask_count[1] = lonLength;
	
	status = nc_inq_varid(ncid, "mask", &mask_id);
	if (status != NC_NOERR)	{isLandMask = false; err=-1; goto done; }
	if (isLandMask)
	{
		landmask = new double[latLength*lonLength]; 
		if(!landmask) {TechError("GridMap_c::GetPointsAndMask()", "new[]", 0); err = memFullErr; goto done;}
		mylandmaskH = (double**)_NewHandleClear(latLength*lonLength*sizeof(double));
		if(!mylandmaskH) {TechError("GridMap_c::GetPointsAndMask()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
		status = nc_get_vara_double(ncid, mask_id, mask_index, mask_count, landmask);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			INDEXH(mylandmaskH,i*lonLength+j) = landmask[(latLength-i-1)*lonLength+j];
		}
	}
	*maskH = mylandmaskH;
	if(err) goto done;
	
depths:
	if (err) goto done;
	// also translate to fDepthDataInfo and fDepthsH here, using sigma or zgrid info
	
	if (totalDepthsH)
	{
		for (i=0;i<latLength;i++)
		{
			for (j=0;j<lonLength;j++)
			{
				// grid ordering does matter for creating ptcurmap, assume increases fastest in x/lon, then in y/lat
				//INDEXH(totalDepthsH,i*lonLength+j) = fabs(depth_vals[(latLength-i-1)*lonLength+j]);
				if (depth_vals)
					INDEXH(totalDepthsH,i*lonLength+j) = fabs(depth_vals[(latLength-i-1)*lonLength+j]) * scale_factor;
				else 
					INDEXH(totalDepthsH, i) = INFINITE_DEPTH;	// let map have infinite depth

			}
		}
		*depthPointsH = totalDepthsH;
		
	}
	
		
done:
	if (err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
		if(totalDepthsH) {DisposeHandle((Handle)totalDepthsH); totalDepthsH = 0;}
		if(mylandmaskH) {DisposeHandle((Handle)mylandmaskH); mylandmaskH = 0;}
	}
	
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (landmask) delete [] landmask;
	if (modelTypeStr) delete [] modelTypeStr;

	return err;
}


// needs to be updated once triangle grid format is set
OSErr GridMap_c::GetPointsAndBoundary(char *path,
									  WORLDPOINTFH *vertexPointsH, FLOATH *depthPtsH, long *numNodes,
									  LONGPTR *boundary_indices, LONGPTR *boundary_nums, LONGPTR *boundary_type, long *numBoundaryPts,
									  LONGPTR *triangle_verts, LONGPTR *triangle_neighbors, long *ntri)
{
	OSErr err = 0;
	char errmsg[256] = "";

	Boolean bTopInfoInFile = false, bVelocitiesOnTriangles = false;

	long i, status;
	int ncid, nodeid, nbndid, bndid, neleid;
	int nv_varid, nbe_varid;
	int latid, lonid, depthid;
	int curr_ucmp_id, uv_dimid[3], uv_ndims;

	size_t nodeLength, nbndLength, neleLength;
	static size_t latIndex = 0, lonIndex = 0, ptIndex = 0;
	static size_t pt_count;
	static size_t bndIndex[2] = {0, 0}, bnd_count[2];
	static size_t topIndex[2] = {0, 0}, top_count[2];

	float *lat_vals = 0, *lon_vals = 0, *depth_vals = 0;
	long *bndry_indices = 0, *bndry_nums = 0, *bndry_type = 0, *top_verts = 0, *top_neighbors = 0;
	double scale_factor = 1.;
	char *modelTypeStr = 0;

	FLOATH totalDepthsH = 0;
	WORLDPOINTFH vertexPtsH = 0;	
	
	if (!path || !path[0])
		return 0;

	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR){err = -1; goto done;}

	status = nc_inq_dimid(ncid, "node", &nodeid); 
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_dimlen(ncid, nodeid, &nodeLength);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_dimid(ncid, "nbnd", &nbndid);	
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_varid(ncid, "bnd", &bndid);	
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_dimlen(ncid, nbndid, &nbndLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	bnd_count[0] = nbndLength;
	bnd_count[1] = 1;
	bndry_indices = new long[nbndLength]; 
	bndry_nums = new long[nbndLength]; 
	bndry_type = new long[nbndLength]; 
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}

	bndIndex[1] = 1;	// take second point of boundary segments instead, so that water boundaries work out
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	if (status != NC_NOERR) {err = -1; goto done;}

	bndIndex[1] = 2;
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	if (status != NC_NOERR) {err = -1; goto done;}

	bndIndex[1] = 3;
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_type);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// option to use index values?
	status = nc_inq_varid(ncid, "lat", &latid);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_varid(ncid, "lon", &lonid);
	if (status != NC_NOERR) {err = -1; goto done;}

	pt_count = nodeLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(nodeLength * sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}

	lat_vals = new float[nodeLength]; 
	lon_vals = new float[nodeLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}

	status = nc_get_vara_float(ncid, latid, &ptIndex, &pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_get_vara_float(ncid, lonid, &ptIndex, &pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}

	totalDepthsH = (FLOATH)_NewHandleClear(nodeLength * sizeof(float));
	if (!totalDepthsH) {err = memFullErr; goto done;}

	status = nc_inq_varid(ncid, "depth", &depthid);
	if (status == NC_NOERR) {
		depth_vals = new float[nodeLength];
		if (!depth_vals) {err = memFullErr; goto done;}

		status = nc_get_vara_float(ncid, depthid, &ptIndex, &pt_count, depth_vals);
		if (status != NC_NOERR) {err = -1; goto done;}

		status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
	}
	for (i = 0; i < nodeLength; i++) {
		INDEXH(vertexPtsH, i).pLat = lat_vals[i];
		INDEXH(vertexPtsH, i).pLong = lon_vals[i];
		if (depth_vals)
			INDEXH(totalDepthsH, i) = depth_vals[i] * scale_factor;
		else
			INDEXH(totalDepthsH, i) = INFINITE_DEPTH;	// let map have infinite depth
	}

	*vertexPointsH	= vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	*depthPtsH = totalDepthsH;
	*numBoundaryPts = nbndLength;
	*numNodes = nodeLength;
	
	
	// check if file has topology in it
	{
		status = nc_inq_varid(ncid, "nv", &nv_varid); //Navy
		if (status != NC_NOERR) {/*err = -1; goto done;*/}
		else {
			status = nc_inq_varid(ncid, "nbe", &nbe_varid); //Navy
			if (status != NC_NOERR) {/*err = -1; goto done;*/}
			else
				bTopInfoInFile = true;
		}
		if (bTopInfoInFile)
		{
			status = nc_inq_dimid(ncid, "nele", &neleid);	
			if (status != NC_NOERR) {err = -1; goto done;}

			status = nc_inq_dimlen(ncid, neleid, &neleLength);
			if (status != NC_NOERR) {err = -1; goto done;}

			//fNumEles = neleLength;
			top_verts = new long[neleLength * 3];
			if (!top_verts) {err = memFullErr; goto done;}

			top_neighbors = new long[neleLength * 3];
			if (!top_neighbors) {err = memFullErr; goto done;}

			top_count[0] = 3;
			top_count[1] = neleLength;

			status = nc_get_vara_long(ncid, nv_varid, topIndex, top_count, top_verts);
			if (status != NC_NOERR) {err = -1; goto done;}

			status = nc_get_vara_long(ncid, nbe_varid, topIndex, top_count, top_neighbors);
			if (status != NC_NOERR) {err = -1; goto done;}

			//determine if velocities are on triangles
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
			if (status != NC_NOERR) {err = -1; goto done;}

			status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
			if (status != NC_NOERR) {err = -1; goto done;}
			
			status = nc_inq_vardimid (ncid, curr_ucmp_id, uv_dimid);	// see if dimid(1) or (2) == nele or node, depends on uv_ndims
			if (status == NC_NOERR) {
				if (uv_ndims == 3 && uv_dimid[2] == neleid)
					bVelocitiesOnTriangles = true;
				else if (uv_ndims == 2 && uv_dimid[1] == neleid)
					bVelocitiesOnTriangles = true;
			}
		}
	}

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}

	*boundary_indices = bndry_indices;
	*boundary_nums = bndry_nums;
	*boundary_type = bndry_type;
	
	if (bVelocitiesOnTriangles) {
		if (!top_verts || !top_neighbors) {err = memFullErr; goto done;}

		*ntri = neleLength;
		*triangle_verts = top_verts;
		*triangle_neighbors = top_neighbors;
	}

depths:
	if (err)
		goto done;

done:

	if (err) {
		if (!errmsg[0]) 
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
		if(totalDepthsH) {DisposeHandle((Handle)totalDepthsH); totalDepthsH = 0;}
		if (bndry_indices) delete [] bndry_indices;
		if (bndry_nums) delete [] bndry_nums;
		if (bndry_type) delete [] bndry_type;
		if (top_verts) delete [] top_verts;
		if (top_neighbors) delete [] top_neighbors;
	}

	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;

	return err;
}


OSErr GridMap_c::ReadCATSMap(vector<string> &linesInFile) 
{
	char errmsg[256];
	long i, numPoints, line = 0;
	string currentLine;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0;
	
	errmsg[0]=0;
	
	MySpinCursor(); 

	currentLine = linesInFile[(line)++];
	currentLine = linesInFile[(line)++];
	
	if(IsTVerticesHeaderLine(currentLine, numPoints))
	{
		MySpinCursor();
		err = ReadTVerticesBody(linesInFile,&line,&pts,&depths,errmsg,numPoints,true);
		if(err) goto done;
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor();

	currentLine = linesInFile[(line)++];
	
	if(IsBoundarySegmentHeaderLine(currentLine,numBoundarySegs)) // Boundary data from CATS
	{
		MySpinCursor();
		err = ReadBoundarySegs(linesInFile,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		currentLine = linesInFile[(line)++];
	}
	else
	{
		err = -1; 
		goto done;
	}
	MySpinCursor(); 
	
	if(IsWaterBoundaryHeaderLine(currentLine,numWaterBoundaries,numBoundaryPts)) // Boundary types from CATS
	{
		MySpinCursor();
		err = ReadWaterBoundaries(linesInFile,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		currentLine = linesInFile[(line)++];
	}
	else
	{
		err = -1; 
		goto done;
	}
	
	if(IsTTopologyHeaderLine(currentLine,numPoints)) // Topology from CATS
	{
		MySpinCursor();
		err = ReadTTopologyBody(linesInFile,&line,&topo,&velH,errmsg,numPoints,TRUE);
		if(err) goto done;
	}
	else
	{
		DisplayMessage("Making Triangles");

		if ((err = maketriangles(&topo, pts, numPoints, boundarySegs, numBoundarySegs)) != 0)
			err = -1; // for now we require TTopology

		DisplayMessage(0);
		velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*numPoints);
		if(!velH)
		{
			strcpy(errmsg,"Not enough memory.");
			goto done;
		}
		for (i=0;i<numPoints;i++)
		{
			INDEXH(velH,i).u = 0.;
			INDEXH(velH,i).v = 0.;
		}
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	currentLine = linesInFile[(line)++];
	
	if(IsTIndexedDagTreeHeaderLine(currentLine,numPoints))  // DagTree from CATS
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(linesInFile,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		DisplayMessage("Making Dag Tree");
		tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); // use CATSDagTree.cpp and my_build_list.h
		DisplayMessage(0);
		if (errmsg[0])	
			err = -1; // for now we require TIndexedDagTree
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	// figure out the bounds
	bounds = voidWorldRect;
	long numPts;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	
	
	/////////////////////////////////////////////////
	// create the bathymetry map 
	
	if (waterBoundaries)
	{
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundarySegs);	
		this->SetWaterBoundaries(waterBoundaries);
		this->SetMapBounds(bounds);
	}
	else
	{
		err = -1;
		goto done;
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in GridMap_c::ReadCATSMap()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(bounds); //need to set map bounds too
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Triangle Velocity file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	triGrid -> SetBathymetry(depths);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	depths = 0; // because fGrid is now responsible for it
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridMap_c::ReadCATSMap");
		printError(errmsg); 
		if(pts)DisposeHandle((Handle)pts);
		if(topo)DisposeHandle((Handle)topo);
		if(velH)DisposeHandle((Handle)velH);
		if(tree.treeHdl)DisposeHandle((Handle)tree.treeHdl);
		if(depths)DisposeHandle((Handle)depths);
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(boundarySegs)DisposeHandle((Handle)boundarySegs);
		if(waterBoundaries)DisposeHandle((Handle)waterBoundaries);
	}
	return err;
}


OSErr GridMap_c::ReadCATSMap(char *path)
{
	string strPath = path;
	if (strPath.size() == 0)
		return 0;
	
	vector<string> linesInFile;
	if (ReadLinesInFile(strPath, linesInFile)) {
		return ReadCATSMap(linesInFile);
	}
	else {
		return -1;
	}
}

// import map from a topology file so don't have to regenerate
OSErr GridMap_c::ReadTopology(vector<string> &linesInFile)
{
	OSErr err = 0;
	string currentLine;
	long line = 0;

	char errmsg[256];

	long numPoints, numTopoPoints, numPts;

	TopologyHdl topo = 0;
	LongPointHdl pts = 0;
	FLOATH depths = 0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel *triGrid = 0;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries = 0, numBoundaryPts = 0, numBoundarySegs = 0;
	LONGH boundarySegs = 0, waterBoundaries = 0, boundaryPts = 0;
	
	errmsg[0] = 0;
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99

	currentLine = linesInFile[(line)++];
	if (IsTransposeArrayHeaderLine(currentLine, numPts)) {
		LONGH verdatToNetCDFH = 0;

		err = ReadTransposeArray(linesInFile, &line, &verdatToNetCDFH, numPts, errmsg);
		if (err) {
			strcpy(errmsg, "Error in ReadTransposeArray");
			goto done;
		}

		if (verdatToNetCDFH) {
			DisposeHandle((Handle)verdatToNetCDFH);
			verdatToNetCDFH = 0;
		}
	}
	else {
		line--;
	}

	err = ReadTVertices(linesInFile, &line, &pts, &depths, errmsg);
	if (err)
		goto done;

	if (pts) {
		LongPoint thisLPoint;
		Boolean needDepths = false;

		numPts = _GetHandleSize((Handle)pts) / sizeof(LongPoint);
		if (!depths) {
			depths = (FLOATH)_NewHandle(sizeof(FLOATH) * (numPts));
			if (depths == 0) {
				strcpy(errmsg, "Not enough memory to read topology file.");
				goto done;
			}

			needDepths = true;
		}

		if (numPts > 0) {
			WorldPoint wp;

			for (long i = 0; i < numPts; i++) {
				thisLPoint = INDEXH(pts, i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;

				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);

				if (needDepths)
					INDEXH(depths, i) = INFINITE_DEPTH;
			}
		}
	}

	MySpinCursor();

	currentLine = linesInFile[(line)++];
	if (IsBoundarySegmentHeaderLine(currentLine, numBoundarySegs)) {
		// Boundary data from CATs
		MySpinCursor();

		if (numBoundarySegs > 0)
			err = ReadBoundarySegs(linesInFile, &line, &boundarySegs, numBoundarySegs, errmsg);
		if (err)
			goto done;

		currentLine = linesInFile[(line)++];
	}

	MySpinCursor(); // JLM 8/4/99
	
	if (IsWaterBoundaryHeaderLine(currentLine, numWaterBoundaries, numBoundaryPts)) {
		// Boundary types from CATs
		MySpinCursor();

		err = ReadWaterBoundaries(linesInFile, &line, &waterBoundaries, numWaterBoundaries, numBoundaryPts, errmsg);
		if (err)
			goto done;

		currentLine = linesInFile[(line)++];
	}

	MySpinCursor(); // JLM 8/4/99
	
	if (IsBoundaryPointsHeaderLine(currentLine, numBoundaryPts)) {
		// Boundary data from CATs
		MySpinCursor();

		if (numBoundaryPts > 0)
			err = ReadBoundaryPts(linesInFile, &line, &boundaryPts, numBoundaryPts, errmsg);
		if (err)
			goto done;

		currentLine = linesInFile[(line)++];
	}

	MySpinCursor(); // JLM 8/4/99
	
	if (IsTTopologyHeaderLine(currentLine, numTopoPoints)) {
		// Topology from CATs
		MySpinCursor();

		err = ReadTTopologyBody(linesInFile, &line, &topo, &velH, errmsg, numTopoPoints, FALSE);
		if (err)
			goto done;

		currentLine = linesInFile[(line)++];
	}
	else {
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}

	MySpinCursor(); // JLM 8/4/99
	
	if (IsTIndexedDagTreeHeaderLine(currentLine, numPoints)) {
		// DagTree from CATs
		MySpinCursor();

		err = ReadTIndexedDagTreeBody(linesInFile, &line, &tree, errmsg, numPoints);
		if (err)
			goto done;
	}
	else {
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg,"Error in dag tree header line");
		if(err) goto done;
	}

	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	// check if bVelocitiesOnTriangles and boundaryPts
	if (waterBoundaries && boundarySegs) {
		// maybe move up and have the map read in the boundary information
		this->SetBoundarySegs(boundarySegs);	
		this->SetWaterBoundaries(waterBoundaries);

		if (boundaryPts)
			this->SetBoundaryPoints(boundaryPts);

		this->SetMapBounds(bounds);		
	}
	else {
		// should this be an error?
		if (waterBoundaries) {
			DisposeHandle((Handle)waterBoundaries);
			waterBoundaries = 0;
		}
		if (boundarySegs) {
			DisposeHandle((Handle)boundarySegs);
			boundarySegs = 0;
		}
		if (boundaryPts) {
			DisposeHandle((Handle)boundaryPts);
			boundaryPts = 0;
		}
	}
	
	triGrid = new TTriGridVel;
	if (!triGrid) {
		err = true;
		TechError("Error in GridMap_c::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid->SetBounds(bounds);
	triGrid->SetDepths(depths);
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if (!dagTree) {
		printError("Unable to read Extended Topology file.");
		goto done;
	}

	triGrid->SetDagTree(dagTree);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it

done:

	if (err)
	{
		if (!errmsg[0])
			strcpy(errmsg,"An error occurred in GridMap_c::ReadTopology");
		printError(errmsg); 

		if (pts) {
			DisposeHandle((Handle)pts);
			pts = 0;
		}
		if (topo) {
			DisposeHandle((Handle)topo);
			topo = 0;
		}
		if (velH) {
			DisposeHandle((Handle)velH);
			velH = 0;
		}
		if (tree.treeHdl) {
			DisposeHandle((Handle)tree.treeHdl);
			tree.treeHdl = 0;
		}
		if (depths) {
			DisposeHandle((Handle)depths);
			depths = 0;
		}
		if (fGrid) {
			fGrid->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundaries) {
			DisposeHandle((Handle)waterBoundaries);
			waterBoundaries = 0;
		}
		if (boundarySegs) {
			DisposeHandle((Handle)boundarySegs);
			boundarySegs = 0;
		}
		if (boundaryPts) {
			DisposeHandle((Handle)boundaryPts);
			boundaryPts = 0;
		}
	}

	return err;
}


// import map from a topology file so don't have to regenerate
OSErr GridMap_c::ReadTopology(char *path)
{
	string strPath = path;

	if (strPath.size() == 0)
		return 0;

	vector<string> linesInFile;
	if (ReadLinesInFile(strPath, linesInFile))
		return ReadTopology(linesInFile);
	else
		return -1;
}


OSErr GridMap_c::ExportTopology(char* path)
{
	// export bounday info from map so don't have to regenerate each time
	
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y,z;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	FILE *fp = fopen(path, "w");
	
	triGrid = dynamic_cast<TTriGridVel*>(this->fGrid);
	if (!triGrid) {printError("There is no topology to export"); return -1;}
	dagTree = triGrid->GetDagTree();
	if (dagTree) 
	{
		ptsH = dagTree->GetPointsHdl();
		topH = dagTree->GetTopologyHdl();
		treeH = dagTree->GetDagTreeHdl();
	}
	else 
	{
		printError("There is no topology to export");
		return -1;
	}
	depthsH = triGrid->GetDepths();
	if(!ptsH || !topH || !treeH || !depthsH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	boundaryTypeH = GetWaterBoundaries();
	boundarySegmentsH = GetBoundarySegs();
	boundaryPointsH = GetBoundaryPoints();	// if no boundaryPointsH just a special case
	if (!boundaryTypeH || !boundarySegmentsH /*|| !boundaryPointsH*/) 
	{printError("No map info to export"); err=-1; goto done;}
	else
	{
		// any issue with trying to write out non-existent fields?
	}
	
	//(void)hdelete(0, 0, path);
	//if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	//{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	//if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	//{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	/*if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	 else {printError("There is no transpose array"); err = -1; goto done;}
	 sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	 strcpy(buffer,hdrStr);
	 if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	 for(i=0;i<n;i++)
	 {	
	 sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
	 strcpy(buffer,topoStr);
	 if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	 }*/
	
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	sprintf(hdrStr,"Vertices\t%ld\n",nver);	// total vertices
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	sprintf(hdrStr,"%ld\t%ld\n",nver,nver);	// junk line
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	for(i=0;i<nver;i++)
	{	
		x = (*ptsH)[i].h/1000000.0;
		y =(*ptsH)[i].v/1000000.0;
		z = (*depthsH)[i];
		sprintf(topoStr,"%lf\t%lf\t%lf\n",x,y,z);
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	//boundary points - an optional handle, only for curvilinear case
	
	if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
		for(i=0;i<nBoundarySegs;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);	// when reading in subtracts 1
			fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
		}
	}
	nBoundarySegs = 0;
	if (boundaryTypeH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundaryTypeH)/sizeof(long);	// should be same size as previous handle
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2) nWaterBoundaries++;
		}
		sprintf(hdrStr,"WaterBoundaries\t%ld\t%ld\n",nWaterBoundaries,nBoundarySegs);	
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2)
			{
				sprintf(topoStr,"%ld\n",i);
				fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
			}
		}
	}
	
	nBoundaryPts = 0;
	if (boundaryPointsH) 
	{
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	// should be same size as previous handle
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	// when reading in subtracts 1
			fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
		}
	}
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	sprintf(hdrStr,"Topology\t%ld\n",numTriangles);
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	for(i = 0; i< numTriangles;i++)
	{
		v1 = (*topH)[i].vertex1;
		v2 = (*topH)[i].vertex2;
		v3 = (*topH)[i].vertex3;
		n1 = (*topH)[i].adjTri1;
		n2 = (*topH)[i].adjTri2;
		n3 = (*topH)[i].adjTri3;
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
				v1, v2, v3, n1, n2, n3);
		
		/////
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	
	numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	sprintf(hdrStr,"DAGTree\t%ld\n",dagTree->fNumBranches);
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	
	for(i = 0; i<dagTree->fNumBranches; i++)
	{
		sprintf(topoStr,"%ld\t%ld\t%ld\n",(*treeH)[i].topoIndex,(*treeH)[i].branchLeft,(*treeH)[i].branchRight);
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	
done:
	// 
	fclose(fp);
	if(err) {	
		printError("Error writing topology");
	}
	return err;
}

OSErr GridMap_c::SaveAsNetCDF(char *path)
{	
	OSErr err = 0;	
	int status, ncid, ver_dim, top_dim, dag_dim, edge_dim, top_dimid[2], landwater_dimid[1], edge_dimid[2], boundary_count_dimid[1], dag_dimid[2], lat_dimid[1], lon_dimid[1], depth_dimid[1];
	int mesh2_node_x_id, mesh2_node_y_id, mesh2_depth_id, mesh2_face_links_id, mesh2_face_nodes_id, mesh2_dagtree_id, mesh2_landwater_id, mesh2_edge_id, mesh2_boundary_count_id;
	int i, j, mesh2_id, two_dim, three_dim, dimid[1];
	double *lat_vals=0,*lon_vals=0,*depth_vals=0;
	long *dag_vals=0,*top_vals=0,*neighbor_vals=0, *landwater_vals=0, *edge_vals=0, *boundary_count_vals=0;
	long dimension = 2, startIndex = 0, flagValues = -1, flagRange = -8;
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,index,index1,index2,boundaryIndex=0;
	
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts=0;
	long boundaryFlagRange[] = {1, 2}, boundaryFlagValues[] = {1,2};
	
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	char errmsg[256];
	
	triGrid = dynamic_cast<TTriGridVel*>(this->fGrid);
	if (!triGrid) {printError("There is no topology to export"); return -1;}
	dagTree = triGrid->GetDagTree();
	if (dagTree) 
	{
		ptsH = dagTree->GetPointsHdl();
		topH = dagTree->GetTopologyHdl();
		treeH = dagTree->GetDagTreeHdl();
	}
	else 
	{
		printError("There is no topology to export");
		return -1;
	}
	depthsH = triGrid->GetDepths();
	if(!ptsH || !topH || !treeH || !depthsH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	boundaryTypeH = GetWaterBoundaries();
	boundarySegmentsH = GetBoundarySegs();
	boundaryPointsH = GetBoundaryPoints();	// if no boundaryPointsH just a special case
	if (!boundaryTypeH || !boundarySegmentsH /*|| !boundaryPointsH*/) 
	{printError("No map info to export"); err=-1; goto done;}
	else
	{
		// any issue with trying to write out non-existent fields?
	}
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	//numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	numBranches = dagTree->fNumBranches;
	nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
	if (boundaryPointsH) nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	// all the boundary points
	else nBoundaryPts = INDEXH(boundarySegmentsH,nBoundarySegs-1)+1;
	
	status = nc_create(path, NC_CLOBBER, &ncid);
	//sprintf(errmsg,"the path is %s\n",path);
	//printNote(errmsg);
	if (status != NC_NOERR) {err = -1; goto done;}
	//ncid = nccreate(path,NC_CLOBBER);
	//if (status != NC_NOERR) {err = -1; goto done;}
	// for some reason the nc_create command gives an error, have to use nccreate with no error checking
	//status = nc_create(path, NC_NOCLOBBER, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_open(path, NC_WRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	// need to open, put into define mode? - create automatically sets up for 
	//status = nc_redef(ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_def_dim(ncid,"nMesh2_node",nver, &ver_dim);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_def_dim(ncid,"nMesh2_face",numTriangles, &top_dim);
	status = nc_def_dim(ncid,"nMesh2_DAGTree",numBranches, &dag_dim);
	status = nc_def_dim(ncid,"nMesh2_edge",nBoundaryPts,&edge_dim);
	status = nc_def_dim(ncid,"Two",2,&two_dim);
	status = nc_def_dim(ncid,"Three",3,&three_dim);
    //nc_cur.createDimension('nMesh2_node',nNodes)
    //nc_cur.createDimension('nMesh2_face',nFaces)
    //if nDAG>0: #Don't want to create an unlimited dimension for nonexistent variables.
	//nc_cur.createDimension('nMesh2_DAGTree',nDAG)
    //if nBoundSeg>0:
	//nMesh2_edge=BoundarySegments[-1] #nEdges
	//nc_cur.createDimension('nMesh2_edge',nMesh2_edge)
    //nc_cur.createDimension('Two',2)
    //nc_cur.createDimension('Three',3)
    
	top_dimid[0] = top_dim;
	top_dimid[1] = three_dim;
	lat_dimid[0] = ver_dim;
	lon_dimid[0] = ver_dim;
	depth_dimid[0] = ver_dim;
	
    /*Mesh topology.*/
	status = nc_def_var(ncid, "Mesh2", NC_LONG, 0, top_dimid, &mesh2_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_put_att_long (ncid, mesh2_id, "valid_range", NC_DOUBLE, 2, rh_range);
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_id, "long_name", strlen("Topology data of 2D unstructured mesh"), "Topology data of 2D unstructured mesh");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_id, "cf_role", strlen("mesh_topology"), "mesh_topology");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_id, "dimension", NC_LONG, 1, &dimension);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_id, "coordinates",strlen("Mesh2_node_x Mesh2_node_y"), "Mesh2_node_x Mesh2_node_y");
	if (status != NC_NOERR) {err = -1; goto done;}
 	status = nc_put_att_text (ncid, mesh2_id, "face_node_connectivity",strlen("Mesh2_face_nodes"), "Mesh2_face_nodes");
	if (status != NC_NOERR) {err = -1; goto done;}
 	status = nc_put_att_text (ncid, mesh2_id, "face_face_connectivity",strlen("Mesh2_face_links"), "Mesh2_face_links");
	if (status != NC_NOERR) {err = -1; goto done;}
   //nc_Mesh2=nc_cur.createVariable('Mesh2',np.int)
    //nc_Mesh2.cf_role = "mesh_topology"
    //nc_Mesh2.long_name = "Topology data of 2D unstructured mesh"
    //nc_Mesh2.dimension = 2
    //nc_Mesh2.node_coordinates = 'Mesh2_node_x Mesh2_node_y'
    //nc_Mesh2.face_node_connectivity = 'Mesh2_face_nodes'
   // nc_Mesh2.face_face_connectivity = 'Mesh2_face_links'# optional attribute (This is defined differently than in the present UGRID model.)
   // if nBoundSeg>0:
       // nc_Mesh2.edge_node_connectivity = 'Mesh2_edge_nodes'
		
	status = nc_def_var(ncid, "Mesh2_face_nodes", NC_LONG, 2, top_dimid, &mesh2_face_nodes_id);
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_face_nodes=nc_cur.createVariable('Mesh2_face_nodes',np.int,('nMesh2_face','Three'),zlib=True)
	status = nc_put_att_text (ncid, mesh2_face_nodes_id, "long_name", strlen("Maps every triangular face to its three corner nodes."), "Maps every triangular face to its three corner nodes.");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_face_nodes_id, "cf_role", strlen("face_node_connectivity"), "face_node_connectivity");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_face_nodes_id, "start_index", NC_LONG, 1, &startIndex);
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_face_nodes.cf_role = "face_node_connectivity"
    //nc_Mesh2_face_nodes.long_name = "Maps every triangular face to its three corner nodes."
    //nc_Mesh2_face_nodes.start_index = 0
    //nc_Mesh2_face_nodes[:]=np.column_stack([Mesh2_face_nodes_1,Mesh2_face_nodes_2,Mesh2_face_nodes_3])	// does this just set up the array?
    //The "Mesh2_edge_nodes" variable has been omitted, since CUR files don't make use of edges in and of themselves.  This could be calculated, if it'll be put to use.
    
	status = nc_def_var(ncid, "Mesh2_face_links", NC_LONG, 2, top_dimid, &mesh2_face_links_id);
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_face_links=nc_cur.createVariable('Mesh2_face_links',np.int,('nMesh2_face','Three'),zlib=True) #Mesh2_face_links is size Nx3, not Nx2 like in the draft UGRID model.
	status = nc_put_att_text (ncid, mesh2_face_links_id, "long_name", strlen("Indicates which other faces neighbor each face."), "Indicates which other faces neighbor each face.");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_face_links_id, "cf_role", strlen("face_face_connectivity"), "face_face_connectivity");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_face_links_id, "start_index", NC_LONG, 1, &startIndex);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_face_links_id, "flag_values", NC_LONG, 1, &flagValues);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_face_links_id, "flag_meanings", strlen("boundary"), "boundary");
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_face_links.cf_role = "face_face_connectivity"
    //nc_Mesh2_face_links.long_name = "Indicates which other faces neighbor each face."
    //nc_Mesh2_face_links.start_index = 0
    //nc_Mesh2_face_links.flag_values = -1
    //nc_Mesh2_face_links.flag_meanings = "boundary"
    //nc_Mesh2_face_links[:]=np.column_stack([Mesh2_face_link_1,Mesh2_face_link_2,Mesh2_face_link_3])	// does this just set up the array?
    
    /*Mesh node coordinates.*/
	status = nc_def_var(ncid, "Mesh2_node_x", NC_DOUBLE, 1, lon_dimid, &mesh2_node_x_id);
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_node_x=nc_cur.createVariable('Mesh2_node_x',np.double,('nMesh2_node',),zlib=True)
	status = nc_put_att_text (ncid, mesh2_node_x_id, "standard_name",strlen("longitude"), "longitude");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_node_x_id, "long_name",strlen("Longitude of 2D mesh nodes."), "Longitude of 2D mesh nodes.");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_node_x_id, "units",strlen("degrees_east"), "degrees_east");
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_node_x.standard_name = "longitude"
    //nc_Mesh2_node_x.long_name = "Longitude of 2D mesh nodes."
    //nc_Mesh2_node_x.units = "degrees_east"
    
	status = nc_def_var(ncid, "Mesh2_node_y", NC_DOUBLE, 1, lat_dimid, &mesh2_node_y_id);
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_node_y=nc_cur.createVariable('Mesh2_node_y',np.double,('nMesh2_node',),zlib=True)
	status = nc_put_att_text (ncid, mesh2_node_y_id, "standard_name",strlen("latitude"), "latitude");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_node_y_id, "long_name",strlen("Latitude of 2D mesh nodes."), "Latitude of 2D mesh nodes.");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_node_y_id, "units",strlen("degrees_north"), "degrees_north");
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_node_y.standard_name = "latitude"
    //nc_Mesh2_node_y.long_name = "Latitude of 2D mesh nodes."
    //nc_Mesh2_node_y.units = "degrees_north"
    
    /*Vertical coordinate.*/
	status = nc_def_var(ncid, "Mesh2_depth", NC_DOUBLE, 1, depth_dimid, &mesh2_depth_id);
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_depth=nc_cur.createVariable('Mesh2_depth',np.double,('nMesh2_node',),zlib=True)
    //nc_Mesh2_depth.standard_name = "sea_floor_depth_below_sea_level" #Is this the definition of depth we're really using?
	//nc_Mesh2_depth.long_name = "Depth of 2D mesh nodes." #No formal long name defined.
 	status = nc_put_att_text (ncid, mesh2_depth_id, "standard_name",strlen("sea_floor_depth_below_sea_level"), "sea_floor_depth_below_sea_level");
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_put_att_text (ncid, mesh2_depth_id, "long_name",strlen("sea_floor_depth_below_sea_level"), "sea_floor_depth_below_sea_level");
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_depth_id, "units",strlen("m"), "m");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_depth_id, "positive",strlen("down"), "down");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_depth_id, "mesh",strlen("Mesh2"), "Mesh2");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_depth_id, "location",strlen("node"), "node");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_depth_id, "coordinates",strlen("Mesh2_node_x Mesh2_node_y"), "Mesh2_node_x Mesh2_node_y");
	if (status != NC_NOERR) {err = -1; goto done;}
    //nc_Mesh2_depth.units = "m"
    //nc_Mesh2_depth.positive = "down"
    //nc_Mesh2_depth.mesh = "Mesh2"
    //nc_Mesh2_depth.location = "node"
    //nc_Mesh2_depth.coordinates = "Mesh2_node_x Mesh2_node_y"
    
    
    /*Mesh face variables.*/
	// this is for the velocities which we do not need
	
    /*DAGtree (custom)*/
    flagValues = -8;
    dag_dimid[0] = dag_dim;
    dag_dimid[1] = three_dim;
	status = nc_def_var(ncid, "Mesh2_DAGtree", NC_LONG, 2, dag_dimid, &mesh2_dagtree_id);
	if (status != NC_NOERR) {err = -1; goto done;}
 	status = nc_put_att_text (ncid, mesh2_dagtree_id, "long_name", strlen("Topological tree ordering for directional (directed) acyclic graph of edges."), "Topological tree ordering for directional (directed) acyclic graph of edges.");
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_put_att_text (ncid, mesh2_dagtree_id, "cf_role", strlen("directional_acyclic_graph_tree"), "directional_acyclic_graph_tree");
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_dagtree_id, "start_index", NC_LONG, 1, &startIndex);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_dagtree_id, "flag_values", NC_LONG, 1, &flagValues);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_dagtree_id, "flag_range", NC_LONG, 1, &flagRange);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_dagtree_id, "flag_meanings", strlen("end-of-branch"), "end-of-branch");
	if (status != NC_NOERR) {err = -1; goto done;}
	//if nDAG>0:
	//nc_Mesh2_DAGtree=nc_cur.createVariable('Mesh2_DAGtree',np.int,('nMesh2_DAGTree','Three'),zlib=True)
	//#nc_Mesh2_DAGtree.cf_role = "directional_acyclic_graph_tree" #This isn't a CF role, so it has been commented out.
	//nc_Mesh2_DAGtree.long_name = "Topological tree ordering for directional (directed) acyclic graph of edges."
	//nc_Mesh2_DAGtree.start_index = 0
	//nc_Mesh2_DAGtree.flag_range = -8
	//nc_Mesh2_DAGtree.flag_values = -8
	//nc_Mesh2_DAGtree.flag_meanings = "end-of-branch"
	//nc_Mesh2_DAGtree[:]=np.column_stack([DAGTree_nSeg,DAGTree_branch_left,DAGTree_branch_right])
    
	//int Mesh2_boundary_nodes(Two, nMesh2_boundary) ;
	//Mesh2_boundary_nodes:cf_role = "boundary_node_connectivity" ;
	//Mesh2_boundary_nodes:long_name = "Maps every edge of each boundary to the two nodes that it connects." ;
	//Mesh2_boundary_nodes:start_index = 1. ;

    /*Boundary segments (custom).*/
	// change to boundary_nodes
    edge_dimid[0] = edge_dim;
    edge_dimid[1] = two_dim;
    startIndex = 1;
	status = nc_def_var(ncid, "nMesh2_boundary", NC_LONG, 2, edge_dimid, &mesh2_edge_id);
	if (status != NC_NOERR) {err = -1; goto done;}
 	status = nc_put_att_text (ncid, mesh2_edge_id, "long_name", strlen("Maps every edge of each boundary to the two nodes that it connects."), "Maps every edge of each boundary to the two nodes that it connects.");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_edge_id, "cf_role", strlen("boundary_node_connectivity"), "boundary_node_connectivity");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_edge_id, "start_index", NC_LONG, 1, &startIndex);
	if (status != NC_NOERR) {err = -1; goto done;}
    //if nBoundSeg>0:
	
	//Only define edges if they're part of a boundary.  Leave all other (triangular mesh) edges undefined.
	//nc_Mesh2_edge_nodes=nc_cur.createVariable('Mesh2_edge_nodes',np.int,('nMesh2_edge','Two'),zlib=True)
	//nc_Mesh2_edge_nodes.cf_role = "edge_node_connectivity"
	//nc_Mesh2_edge_nodes.long_name = "Maps each boundary edge to the two nodes that it connects."
	//nc_Mesh2_edge_nodes.start_index = 1 #Since extended .CUR files use 1 as the starting index for the nodes specified in their BoundarySegments data block, I've kept this convention here.

	
	/*Extended .CUR files specify the ending nodes of "boundary" polygons,
	 which are assumed to close with the first node in each "boundary."
	 Include the node indices necessary to specify those closing edges.*/
	//idxEdges=np.zeros((nMesh2_edge,2),dtype=np.int)
	//idxEdges[0:nMesh2_edge-1,:]=np.column_stack([np.arange(1,BoundarySegments[-1],dtype=np.int),np.arange(2,BoundarySegments[-1]+1,dtype=np.int)])
	//idxEdges[BoundarySegments-1,0]=BoundarySegments
	//idxEdges[BoundarySegments-1,1]=idxEdges[np.insert(BoundarySegments[0:-1],0,0),0]
	//nc_Mesh2_edge_nodes[:]=idxEdges
	
	/*Custom (flag) variables.*/
    boundary_count_dimid[0] = edge_dim;
    startIndex = 1;
	status = nc_def_var(ncid, "Mesh2_edge_boundary_count", NC_LONG, 1, edge_dimid, &mesh2_boundary_count_id);
	if (status != NC_NOERR) {err = -1; goto done;}
 	status = nc_put_att_text (ncid, mesh2_boundary_count_id, "long_name", strlen("Defines which boundary the edge is a part of."), "Defines which boundary the edge is a part of.");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_boundary_count_id, "cf_role", strlen("edge_type"), "edge_type");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_boundary_count_id, "start_index", NC_LONG, 1, &startIndex);
	if (status != NC_NOERR) {err = -1; goto done;}

	//nc_Mesh2_edge_boundary_count=nc_cur.createVariable('Mesh2_edge_boundary_count',np.int,('nMesh2_edge',),zlib=True)
	//nc_Mesh2_edge_boundary_count.cf_role = "edge_type"
	//nc_Mesh2_edge_boundary_count.long_name = "Defines which boundary the edge is a part of."
	//nc_Mesh2_edge_boundary_count.start_index = 1
	//Number the edges by which boundary they belong to... first, second, third....

	//# In extended .CUR files the first boundary described by the
	//# BoundarySegments block is the "outer" boundary and its nodes are
	//# listed in counterclockwise order.  Subsequent boundaries are "inner"
	//# boundaries (e.g., islands) and their nodes are listed in clockwise
	//# order in the .CUR.  Here each boundary is tested for clockwise vs.
	//# counterclockwise orientation and the boundary edges flagged
	//# accordingly as either inner or outer.

	//nc_Mesh2_edge_boundary_type_InOrOut=nc_cur.createVariable('Mesh2_edge_boundary_type_InOrOut',np.int,('nMesh2_edge',),zlib=True)
	//nc_Mesh2_edge_boundary_type_InOrOut.cf_role = "edge_type"
	//nc_Mesh2_edge_boundary_type_InOrOut.long_name = "Specifies whether the edge is part of an inner or outer boundary."
	//nc_Mesh2_edge_boundary_type_InOrOut.flag_range = np.arange(0,2,dtype=np.int)
	//nc_Mesh2_edge_boundary_type_InOrOut.flag_values = np.arange(0,2,dtype=np.int)
	//nc_Mesh2_edge_boundary_type_InOrOut.flag_meanings = "outer_boundary inner_boundary"

	//if nWaterBound>0:
	//nc_Mesh2_edge_boundary_type_WaterORLand=nc_cur.createVariable('Mesh2_edge_boundary_type_WaterORLand',np.int,('nMesh2_edge',),zlib=True)
	//nc_Mesh2_edge_boundary_type_WaterORLand.cf_role = "edge_type"
	//nc_Mesh2_edge_boundary_type_WaterORLand.long_name = "Specifies whether the edge represents land or water."
	//nc_Mesh2_edge_boundary_type_WaterORLand.flag_range = np.arange(0,2,dtype=np.int)
	//nc_Mesh2_edge_boundary_type_WaterORLand.flag_values = np.arange(0,2,dtype=np.int)
	//nc_Mesh2_edge_boundary_type_WaterORLand.flag_meanings = "land_boundary water_boundary"
	//Mesh2_edge_boundary_type_WaterORLand=np.zeros(nMesh2_edge,dtype=np.int) #Preallocate with zeros, since in most cases the majority of boundary edges will be land.
	//Mesh2_edge_boundary_type_WaterORLand[WaterBoundaries-1]=1 #The water boundary values are the zero-based indices of the end nodes of the water boundary edges.

	//int Mesh2_boundary_types(nMesh2_boundary) ;
	//Mesh2_boundary_types:cf_role = "boundary_type" ;
	//Mesh2_boundary_types:long_name = "Classification flag for every edge of each boundary." ;
	//Mesh2_boundary_types:location = "boundary" ;
	//Mesh2_boundary_types:flag_range = 0., 1. ;
	//Mesh2_boundary_types:flag_values = 0., 1. ;
	//Mesh2_boundary_types:flag_meanings = "closed_boundary open_boundary" ;
  
	// Water boundaries (custom).
	landwater_dimid[0] = edge_dim;

	// change to boundary type - boundary_types, 0 closed, 1 open
 	status = nc_def_var(ncid, "Mesh2_boundary_types", NC_LONG, 1, landwater_dimid, &mesh2_landwater_id);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_put_att_text (ncid, mesh2_landwater_id, "long_name", strlen("Classification flag for every edge of each boundary."), "Classification flag for every edge of each boundary.");
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_put_att_text (ncid, mesh2_landwater_id, "cf_role", strlen("edge_type"), "edge_type");
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_landwater_id, "location", strlen("boundary"), "boundary");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_landwater_id, "flag_values", NC_LONG, 2, boundaryFlagValues);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_long (ncid, mesh2_landwater_id, "flag_range", NC_LONG, 2, boundaryFlagRange);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, mesh2_landwater_id, "flag_meanings", strlen("closed_boundary open_boundary"), "closed_boundary open_boundary");
	if (status != NC_NOERR) {err = -1; goto done;}

    /*Set file attributes.*/
	status = nc_put_att_text (ncid, NC_GLOBAL, "Conventions",strlen("UGRID CF"), "UGRID CF");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, NC_GLOBAL, "Title",strlen("Grid Map"), "Grid Map");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, NC_GLOBAL, "Institution",strlen("NOAA/NOS/ERD"), "NOAA/NOS/ERD");
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_put_att_text (ncid, NC_GLOBAL, "References",strlen("General NOAA Operational Modeling Environment"), "General NOAA Operational Modeling Environment");
	if (status != NC_NOERR) {err = -1; goto done;}
    //setattr( nc_cur, "Conventions", "UGRID CF" )
    //setattr( nc_cur, "Title", "CATS Currents" )
    //setattr( nc_cur, "Institution", "USA/DOC/NOAA/NOS/ERD/TSSB" )
    //setattr( nc_cur, "References", "brian.zelenke@noaa.gov" )
    
	lat_vals = new double[nver]; 
	if(!lat_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	lon_vals = new double[nver]; 
	if(!lon_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	depth_vals = new double[nver]; 
	if(!depth_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	dag_vals = new long[3*numBranches]; 
	if(!dag_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	top_vals = new long[3*numTriangles]; 
	if(!top_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	neighbor_vals = new long[3*numTriangles]; 
	if(!neighbor_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	landwater_vals = new long[nBoundaryPts]; 
	if(!landwater_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	edge_vals = new long[2*nBoundaryPts]; 
	if(!edge_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}
	boundary_count_vals = new long[nBoundaryPts]; 
	if(!boundary_count_vals) {TechError("GridMap::SaveAsNetCDF()", "new[]", 0); err = memFullErr; goto done;}

	for (i=0;i<nver;i++)
	{
		lat_vals[i] = INDEXH(ptsH,i).v / 1000000.;
		lon_vals[i] = INDEXH(ptsH,i).h / 1000000.;
		depth_vals[i] = INDEXH(depthsH,i);
	}
	for (i=0;i<numBranches;i++)
	{
		dag_vals[i*3] = (*treeH)[i].topoIndex;
		dag_vals[i*3+1] = (*treeH)[i].branchLeft;
		dag_vals[i*3+2] = (*treeH)[i].branchRight;
	}
	for(i=0;i<numTriangles;i++)
	{
		top_vals[i*3] = (*topH)[i].vertex1;
		top_vals[i*3+1] = (*topH)[i].vertex2;
		top_vals[i*3+2] = (*topH)[i].vertex3;
		neighbor_vals[i*3] = (*topH)[i].adjTri1;
		neighbor_vals[i*3+1] = (*topH)[i].adjTri2;
		neighbor_vals[i*3+2] = (*topH)[i].adjTri3;
	}
	if (boundaryPointsH)
	{
		boundaryIndex = 0;
		for(theSeg = 0; theSeg < nSegs; theSeg++)
		{
			startver = theSeg == 0? 0: (*boundarySegmentsH)[theSeg-1] + 1;
			endver = (*boundarySegmentsH)[theSeg]+1;
			index1 = (*boundaryPointsH)[startver];
			index2 = (*boundaryPointsH)[startver+1];
			edge_vals[boundaryIndex*2] = index1;
			edge_vals[boundaryIndex*2+1] = index2;
			for(j = startver + 1; j < endver-1; j++) // endver-1
			{
				if ((*boundaryTypeH)[j]==2)	// a water boundary
				{
					landwater_vals[boundaryIndex] = 2;
				}
				else
				{
					landwater_vals[boundaryIndex] = 1;	// land
				}
				boundary_count_vals[boundaryIndex] = theSeg+1;
				boundaryIndex+=1;
				index = (*boundaryPointsH)[j];
				index2 = (*boundaryPointsH)[j+1];
				edge_vals[boundaryIndex*2] = index;
				edge_vals[boundaryIndex*2+1] = index2;
			}
			if ((*boundaryTypeH)[endver-1]==2)	// a water boundary
			{
				landwater_vals[boundaryIndex] = 2;
			}
			else
			{
				landwater_vals[boundaryIndex] = 1;	// land
			}
			boundary_count_vals[boundaryIndex] = theSeg+1;
			boundaryIndex+=1;
			if ((*boundaryTypeH)[startver]==2)	// a water boundary
			{
				landwater_vals[boundaryIndex] = 2;
			}
			else
			{
				landwater_vals[boundaryIndex] = 1;	// land
			}
			boundary_count_vals[boundaryIndex] = theSeg+1;
			index = (*boundaryPointsH)[endver-1];
			//index2 = (*fBoundaryPointsH)[j+1];
			edge_vals[boundaryIndex*2] = index;
			edge_vals[boundaryIndex*2+1] = index1;	// original start point
			boundaryIndex+=1;
		}
		//sprintf(errmsg, "Number of boundary points = %d\n",boundaryIndex);
		//printNote(errmsg);
	}
	else 
	{
		boundaryIndex = 0;
		for(theSeg = 0; theSeg < nSegs; theSeg++)
		{
			startver = theSeg == 0? 0: (*boundarySegmentsH)[theSeg-1] + 1;
			endver = (*boundarySegmentsH)[theSeg]+1;

			edge_vals[boundaryIndex*2] = startver;
			edge_vals[boundaryIndex*2+1] = startver + 1;
			// startver to end ver back to startver
			for(j = startver + 1; j < endver-1; j++)	// endver-1
			{
				if ((*boundaryTypeH)[j]==2)	// a water boundary
					landwater_vals[boundaryIndex] = 2;
				else
				{
					landwater_vals[boundaryIndex] = 1;	// land
				}
				boundary_count_vals[boundaryIndex] = theSeg+1;
				boundaryIndex+=1;
				edge_vals[boundaryIndex*2] = j;
				edge_vals[boundaryIndex*2+1] = j + 1;
			 }
			
			if ((*boundaryTypeH)[endver-1]==2)	// a water boundary
				landwater_vals[boundaryIndex] = 2;
			else 
			{
				landwater_vals[boundaryIndex] = 1;
			}
			boundary_count_vals[boundaryIndex] = theSeg+1;
			boundaryIndex+=1;
			if ((*boundaryTypeH)[startver]==2)	// a water boundary
				landwater_vals[boundaryIndex] = 2;
			else 
			{
				landwater_vals[boundaryIndex] = 1;
			}
			boundary_count_vals[boundaryIndex] = theSeg+1;

			edge_vals[boundaryIndex*2] = endver-1;
			edge_vals[boundaryIndex*2+1] = startver;
			
			boundaryIndex+=1;
			//sprintf(errmsg, "Number of boundary points = %d\n",boundaryIndex);
			//printNote(errmsg);
		}
	}

	status = nc_enddef(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store lat 
	status = nc_put_var_double(ncid, mesh2_node_y_id, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store lon 
	status = nc_put_var_double(ncid, mesh2_node_x_id, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store depth 
	status = nc_put_var_double(ncid, mesh2_depth_id, depth_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store dagtree 
	status = nc_put_var_long(ncid, mesh2_dagtree_id, dag_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store triangles 
	status = nc_put_var_long(ncid, mesh2_face_nodes_id, top_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store neighbors 
	status = nc_put_var_long(ncid, mesh2_face_links_id, neighbor_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store landwater 
	status = nc_put_var_long(ncid, mesh2_landwater_id, landwater_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store edges 
	status = nc_put_var_long(ncid, mesh2_edge_id, edge_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// store boundaries 
	status = nc_put_var_long(ncid, mesh2_boundary_count_id, boundary_count_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
done:
	if (err)
	{
		printError("Error writing out netcdf file");
	}
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (dag_vals) delete [] dag_vals;
	if (top_vals) delete [] top_vals;
	if (neighbor_vals) delete [] neighbor_vals;
	if (landwater_vals) delete [] landwater_vals;
	if (edge_vals) delete [] edge_vals;
	if (boundary_count_vals) delete [] boundary_count_vals;
	return err;
}

OSErr GridMap_c::TextRead(char *path)
{
	OSErr err = 0;
	char s[256], fileName[256];
	char nameStr[256];
	char errmsg[256];

	WorldRect bounds = theWorld;
	short gridType;

	if (IsNetCDFFile(path, &gridType))
	{
		strcpy(s, path);
		SplitPathFile(s, fileName);
		strcat (nameStr, fileName);

		if (gridType != REGULAR &&
			gridType != REGULAR_SWAFS)
		{
			FLOATH depthPtsH = 0;
			WORLDPOINTFH vertexPtsH = 0;
			long numRows = 0, numCols = 0, numNodes = 0, numTri = 0, numBoundaryPts = 0;

			if (gridType == CURVILINEAR) {
				DOUBLEH maskH = 0;

				err = this->GetPointsAndMask(path, &maskH, &vertexPtsH, &depthPtsH, &numRows, &numCols);	//Text read
				if (!err)
					err = this->SetUpCurvilinearGrid(maskH, numRows, numCols, vertexPtsH, depthPtsH, errmsg);	//Reorder points

				if (maskH) {
					DisposeHandle((Handle)maskH);
					maskH = 0;
				}
			}
			else if (gridType == TRIANGULAR) {
				// check if topology is included
				LONGPTR bndry_indices = 0, bndry_nums = 0, bndry_type = 0, tri_verts = 0, tri_neighbors = 0;

				err = this->GetPointsAndBoundary(path, &vertexPtsH, &depthPtsH, &numNodes,
												 &bndry_indices, &bndry_nums, &bndry_type, &numBoundaryPts,
												 &tri_verts, &tri_neighbors, &numTri);	//Text read
				if (!err) {
					// separate points and boundary
					if (numTri == 0)
						err = this->SetUpTriangleGrid(numNodes, numTri, vertexPtsH, depthPtsH, bndry_indices, bndry_nums, bndry_type, numBoundaryPts);	//Reorder points
					else
						err = this->SetUpTriangleGrid2(numNodes,numTri, vertexPtsH, depthPtsH, bndry_indices, bndry_nums, bndry_type, numBoundaryPts,tri_verts, tri_neighbors);	//Reorder points
				}

				if (bndry_indices)
					delete [] bndry_indices;
				if (bndry_nums)
					delete [] bndry_nums;
				if (bndry_type)
					delete [] bndry_type;
				if (tri_verts)
					delete [] tri_verts;
				if (tri_neighbors)
					delete [] tri_neighbors;
			}

			if (vertexPtsH) {
				DisposeHandle((Handle)vertexPtsH);
				vertexPtsH = 0;
			}
			if (depthPtsH) {
				DisposeHandle((Handle)depthPtsH);
				depthPtsH = 0;
			}
		}
		else {
			err = true;
			sprintf(errmsg, "File %s is a current file and should be input as a universal mover.", fileName);
			printNote(errmsg);
		}
	}
	else if (IsCATS3DFile(path))	// for any CATS?
	{
		strcpy(s,path);
		SplitPathFile (s, fileName);
		strcat (nameStr, fileName);
		
		//if (gMap)
		{
			err = this->ReadCATSMap(path);	
			if(err) 
			{
				return err;
			}
		
		}
	}
	else
	{
		{	// check if isTopologyFile()
			err = this -> ReadTopology(path);
			if(err) 
			{
				//return err;
			}
		}
		if (err)
		{
			sprintf(errmsg,"File %s is not a recognizable map file.",fileName);
			printError(errmsg);
		}
	}
	return err;
	
}


void GridMap_c::DrawBoundaries(Rect r)
{
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j;
	long x,y;
	//Point pt;
	Boolean offQuickDrawPlane = false;
	
	long penWidth = 3;
	long halfPenWidth = penWidth/2;
	
	//PenNormal();
	//RGBColor sc;
	//GetForeColor(&sc);
	
	// to support new curvilinear algorithm
	if (fBoundaryPointsH)
	{
		DrawBoundaries2(r);
		return;
	}
	
	LongPointHdl ptsHdl = GetPointsHdl();	
	if(!ptsHdl) return;
	
/*#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif*/
	
	// have each seg be a polygon with a fill option - land only, maybe fill with a pattern?
	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		
		/*pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
		MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		for(j = startver + 1; j < endver; j++)
		{
			if ((*fBoundaryTypeH)[j]==2)	// a water boundary
				RGBForeColor(&colors[BLUE]);
			else// add option to change color, light or dark depending on which is easier to see , see premerge GNOME_beta
			{
				RGBForeColor(&colors[BROWN]);	// land
			}
			pt = GetQuickDrawPt((*ptsHdl)[j].h,(*ptsHdl)[j].v,&r,&offQuickDrawPlane);
			if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
			{
				MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			}
			else
				MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
			RGBForeColor(&colors[BLUE]);
		else
		{
			RGBForeColor(&colors[BROWN]);	// land
		}
		pt = GetQuickDrawPt((*ptsHdl)[startver].h,(*ptsHdl)[startver].v,&r,&offQuickDrawPlane);
		if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
		{
			MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		else
			MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);*/
	}
	
/*#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	RGBForeColor(&sc);*/
}

void GridMap_c::DrawBoundaries2(Rect r)
{
	// should combine into original DrawBoundaries, just check for fBoundaryPointsH
	//PenNormal();
	//RGBColor sc;
	//GetForeColor(&sc);
	
	long nSegs = GetNumBoundarySegs();	
	long theSeg,startver,endver,j;
	long x,y,index1,index;
	Point pt;
	Boolean offQuickDrawPlane = false;
	
	long penWidth = 3;
	long halfPenWidth = penWidth/2;
	
	LongPointHdl ptsHdl = GetPointsHdl();	
	if(!ptsHdl) return;
	
/*#ifdef MAC
	PenSize(penWidth,penWidth);
#else
	PenStyle(BLACK,penWidth);
#endif*/
	
	
	for(theSeg = 0; theSeg < nSegs; theSeg++)
	{
		startver = theSeg == 0? 0: (*fBoundarySegmentsH)[theSeg-1] + 1;
		endver = (*fBoundarySegmentsH)[theSeg]+1;
		index1 = (*fBoundaryPointsH)[startver];
		//pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
		//MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		for(j = startver + 1; j < endver; j++)
		{
			index = (*fBoundaryPointsH)[j];
     		/*if ((*fBoundaryTypeH)[j]==2)	// a water boundary
				RGBForeColor(&colors[BLUE]);
			else
				RGBForeColor(&colors[BROWN]);	// land
			pt = GetQuickDrawPt((*ptsHdl)[index].h,(*ptsHdl)[index].v,&r,&offQuickDrawPlane);
			if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[j]==1))
			{
				MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
			}
			else
				MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);*/
		}
		/*if ((*fBoundaryTypeH)[startver]==2)	// a water boundary
			RGBForeColor(&colors[BLUE]);
		else
			RGBForeColor(&colors[BROWN]);	// land
		pt = GetQuickDrawPt((*ptsHdl)[index1].h,(*ptsHdl)[index1].v,&r,&offQuickDrawPlane);
		if(!drawingLandBitMap  || (drawingLandBitMap && (*fBoundaryTypeH)[startver]==1))
		{
			MyLineTo(pt.h-halfPenWidth,pt.v-halfPenWidth);
		}
		else
			MyMoveTo(pt.h-halfPenWidth,pt.v-halfPenWidth);*/
	}

/*#ifdef MAC
	PenSize(1,1);
#else
	PenStyle(BLACK,1);
#endif
	RGBForeColor(&sc);*/
}


