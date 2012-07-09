/*
 *  NetCDFWindMoverCurv_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "NetCDFWindMoverCurv_c.h"
#include "CROSS.H"

NetCDFWindMoverCurv_c::NetCDFWindMoverCurv_c (TMap *owner, char *name) : NetCDFWindMover_c(owner, name)
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
}
LongPointHdl NetCDFWindMoverCurv_c::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

long NetCDFWindMoverCurv_c::GetVelocityIndex(WorldPoint wp)
{
	long index = -1;
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	return index;
}

LongPoint NetCDFWindMoverCurv_c::GetVelocityIndices(WorldPoint wp)
{
	LongPoint indices={-1,-1};
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	return indices;
}

Boolean NetCDFWindMoverCurv_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{	// code goes here, this is triangle code, not curvilinear
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	LongPoint indices;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!bShowArrows && !bShowGrid) return 0;
	err = dynamic_cast<NetCDFWindMoverCurv *>(this) -> SetInterval(errmsg);
	if(err) return false;
	
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (index < 0) return 0;
		indices = this->GetVelocityIndices(wp.p);
	}
	
	// Check for constant current 
	if((/*OK*/dynamic_cast<NetCDFWindMoverCurv *>(this)->GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			velocity.u = INDEXH(fStartData.dataHdl,index).u;
			velocity.v = INDEXH(fStartData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fWindScale * lengthU;
	lengthS = this->fWindScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
				this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
				this->className, uStr, sStr);
	}
	
	return true;
}

WorldPoint3D NetCDFWindMoverCurv_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec windVelocity;
	OSErr err = 0;
	char errmsg[256];
	
	
	//return deltaPoint;
	// might want to check for fFillValue and set velocity to zero - shouldn't be an issue unless we interpolate
	if(!fIsOptimizedForStep) 
	{
		err = dynamic_cast<NetCDFWindMoverCurv *>(this) -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	
	// Check for constant wind 
	//if(GetNumTimesInFile()==1)
	if(/*OK*/dynamic_cast<NetCDFWindMoverCurv *>(this)->GetNumTimesInFile()==1 || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			windVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			windVelocity.v = INDEXH(fStartData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			windVelocity.u = 0.;
			windVelocity.v = 0.;
		}
	}
	else // time varying wind 
	{
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			windVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			windVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			windVelocity.u = 0.;
			windVelocity.v = 0.;
		}
	}
	
scale:
	
	windVelocity.u *= fWindScale; // may want to allow some sort of scale factor, though should be in file
	windVelocity.v *= fWindScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		err = AddUncertainty(setIndex,leIndex,&windVelocity);
	}
	
	windVelocity.u *=  (*theLE).windage;
	windVelocity.v *=  (*theLE).windage;
	
	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat  =  (windVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}


long NetCDFWindMoverCurv_c::CheckSurroundingPoints(LONGH maskH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long neighbor;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows - 1) ? row + 1 : fNumRows - 1;
	jEnd = (col < fNumCols - 1) ? col + 1 : fNumCols - 1;
	// don't allow diagonals for now,they could be separate small islands 
	/*for (i = iStart; i< iEnd+1; i++)
	 {
	 for (j = jStart; j< jEnd+1; j++)
	 {	
	 if (i==row && j==col) continue;
	 neighbor = INDEXH(maskH, i*fNumCols + j);
	 if (neighbor >= 3 && neighbor < lowestLandIndex)
	 lowestLandIndex = neighbor;
	 }
	 }*/
	for (i = iStart; i< iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols + col);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols + j);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	return lowestLandIndex;
}
Boolean NetCDFWindMoverCurv_c::ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd;
	long neighbor;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows - 1) ? row + 1 : fNumRows - 1;
	jEnd = (col < fNumCols - 1) ? col + 1 : fNumCols - 1;
	/*for (i = iStart; i < iEnd+1; i++)
	 {
	 for (j = jStart; j < jEnd+1; j++)
	 {	
	 if (i==row && j==col) continue;
	 neighbor = INDEXH(maskH, i*fNumCols + j);
	 // eventually should use a land mask or fill value to identify land
	 if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==0. && INDEXH(velocityH,i*fNumCols+j).v==0.)) return true;
	 }
	 }*/
	for (i = iStart; i < iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols + col);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+col).u==fFillValue && INDEXH(velocityH,i*fNumCols+col).v==fFillValue)) return true;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols + j);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || (INDEXH(velocityH,row*fNumCols+j).u==fFillValue && INDEXH(velocityH,row*fNumCols+j).v==fFillValue)) return true;
	}
	return false;
}

Boolean NetCDFWindMoverCurv_c::InteriorLandPoint(LONGH maskH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd;
	long neighbor;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows_ext - 1) ? row + 1 : fNumRows_ext - 1;
	jEnd = (col < fNumCols_ext - 1) ? col + 1 : fNumCols_ext - 1;
	/*for (i = iStart; i < iEnd+1; i++)
	 {
	 if (i==row) continue;
	 neighbor = INDEXH(maskH, i*fNumCols_ext + col);
	 if (neighbor < 3)	// water point
	 return false;
	 }
	 for (j = jStart; j< jEnd+1; j++)
	 {	
	 if (j==col) continue;
	 neighbor = INDEXH(maskH, row*fNumCols_ext + j);
	 if (neighbor < 3)	// water point
	 return false;
	 }*/
	//for (i = iStart; i < iEnd+1; i++)
	// point is in lower left corner of grid box (land), so only check 3 other quadrants of surrounding 'square'
	for (i = row; i < iEnd+1; i++)
	{
		//for (j = jStart; j< jEnd+1; j++)
		for (j = jStart; j< jEnd; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + j);
			if (neighbor < 3 /*&& neighbor != -1*/)	// water point
				return false;
			//if (row==1 && INDEXH(maskH,j)==1) return false;
		}
	}
	return true;
}

Boolean NetCDFWindMoverCurv_c::ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) 
{
	long iStart, iEnd, jStart, jEnd;
	long i, j, neighbor, landPolyNum;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows_ext - 1) ? row + 1 : fNumRows_ext - 1;
	jEnd = (col < fNumCols_ext - 1) ? col + 1 : fNumCols_ext - 1;
	
	landPolyNum = INDEXH(maskH, row*fNumCols_ext + col);
	for (i = iStart; i< iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols_ext + col);
		if (neighbor >= 3 && neighbor < landPolyNum) 
		{
			*lowerPolyNum = neighbor;
			return true;
		}
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols_ext + j);
		if (neighbor >= 3 && neighbor < landPolyNum) 
		{
			*lowerPolyNum = neighbor;
			return true;
		}
	}
	// don't allow diagonals for now, they could be separate small islands
	/*for (i = iStart; i< iEnd+1; i++)
	 {
	 for (j = jStart; j< jEnd+1; j++)
	 {	
	 if (i==row && j==col) continue;
	 neighbor = INDEXH(maskH, i*fNumCols_ext + j);
	 if (neighbor >= 3 && neighbor < landPolyNum) 
	 {
	 *lowerPolyNum = neighbor;
	 return true;
	 }
	 }
	 }*/
	return false;
}

void NetCDFWindMoverCurv_c::ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin)
{	// merges adjoining land blocks and then renumbers any higher numbered land blocks
	long i,j,val;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{	
			val = INDEXH(maskH,i*fNumCols_ext+j);
			if (val==landBlockToMerge) INDEXH(maskH,i*fNumCols_ext+j) = landBlockToJoin;
			if (val>landBlockToMerge) INDEXH(maskH,i*fNumCols_ext+j) -= 1;
		}
	}
}

OSErr NetCDFWindMoverCurv_c::NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo, long *numIslands) 
{
	OSErr err = 0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long i, j, n, landPolyNum = 1, lowestSurroundingNum = 0;
	long islandNum, maxIslandNum=3;
	LONGH maskH = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	*islandNumberH = 0;
	
	if (!maskH || !maskH2) {err = memFullErr; goto done;}
	// use surface velocity values at time zero
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j) == -1)// 1 water, -1 land
			{
				if (i==0 || i==fNumRows-1 || j==0 || j==fNumCols-1)
				{
					INDEXH(maskH,i*fNumCols+j) = 3;	// set outer boundary to 3
				}
				else
				{
					if (landPolyNum==1)
					{	// Land point
						INDEXH(maskH,i*fNumCols+j) = landPolyNum+3;
						landPolyNum+=3;
					}
					else
					{
						// check for nearest land poly number
						if (lowestSurroundingNum = CheckSurroundingPoints(maskH,i,j)>=3)
						{
							INDEXH(maskH,i*fNumCols+j) = lowestSurroundingNum;
						}
						else
						{
							INDEXH(maskH,i*fNumCols+j) = landPolyNum;
							landPolyNum += 1;
						}
					}
				}
			}
			else
			{
				if (i==0 || i==fNumRows-1 || j==0 || j==fNumCols-1)
					INDEXH(maskH,i*fNumCols+j) = 1;	// Open water boundary
				else if (ThereIsAdjacentLand2(maskH,velocityH,i,j))
					INDEXH(maskH,i*fNumCols+j) = 2;	// Water boundary, not open water
				else
					INDEXH(maskH,i*fNumCols+j) = 0;	// Interior water point
			}
		}
	}
	// extend grid by one row/col up/right since velocities correspond to lower left corner of a grid box
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (i==0) 
			{
				if (j!=fNumCols)
					INDEXH(maskH2,j) = INDEXH(maskH,j);	// flag for extra boundary point
				else
					INDEXH(maskH2,j) = INDEXH(maskH,j-1);	
				
			}
			else if (i!=0 && j==fNumCols) 
				INDEXH(maskH2,i*fNumCols_ext+fNumCols) = INDEXH(maskH,(i-1)*fNumCols+fNumCols-1);
			else 
			{	
				INDEXH(maskH2,i*fNumCols_ext+j) = INDEXH(maskH,(i-1)*fNumCols+j);
			}
		}
	}
	
	// set original top/right boundaries to interior water points 
	// probably don't need to do this since we aren't paying attention to water types anymore
	for (j=1;j<fNumCols_ext-1;j++)	 
	{
		if (INDEXH(maskH2,fNumCols_ext+j)==1) INDEXH(maskH2,fNumCols_ext+j) = 2;
	}
	for (i=1;i<fNumRows_ext-1;i++)
	{
		if (INDEXH(maskH2,i*fNumCols_ext+fNumCols-1)==1) INDEXH(maskH2,i*fNumCols_ext+fNumCols-1) = 2;
	}
	// now merge any contiguous land blocks (max of landPolyNum)
	// as soon as find one, all others of that number change, and every higher landpoint changes
	// repeat until nothing changes
startLoop:
	{
		long lowerPolyNum = 0;
		for (i=0;i<fNumRows_ext;i++)
		{
			for (j=0;j<fNumCols_ext;j++)
			{
				if (INDEXH(maskH2,i*fNumCols_ext+j) < 3) continue;	// water point
				if (ThereIsALowerLandNeighbor(maskH2,&lowerPolyNum,i,j))
				{
					ResetMaskValues(maskH2,INDEXH(maskH2,i*fNumCols_ext+j),lowerPolyNum);
					goto startLoop;
				}
				if ((i==0 || i==fNumRows_ext-1 || j==0 || j==fNumCols_ext-1) && INDEXH(maskH2,i*fNumCols_ext+j)>3)
				{	// shouldn't get here
					ResetMaskValues(maskH2,INDEXH(maskH2,i*fNumCols_ext+j),3);
					goto startLoop;
				}
			}
		}
	}
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{	// note, the numbers start at 3
			islandNum = INDEXH(maskH2,i*fNumCols_ext+j);
			if (islandNum < 3) continue;	// water point
			if (islandNum > maxIslandNum) maxIslandNum = islandNum;
		}
	}
	*islandNumberH = maskH2;
	*numIslands = maxIslandNum;
done:
	if (err) 
	{
		printError("Error numbering islands for map boundaries");
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	if (maskH) {DisposeHandle((Handle)maskH); maskH = 0;}
	return err;
}

/*OSErr NetCDFWindMoverCurv_c::ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg) 
 {
 long i, j, n, ntri, numVerdatPts=0;
 long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
 long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
 long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
 long nSegs, segNum = 0, numIslands, rectIndex; 
 long iIndex, jIndex, index, currentIndex, startIndex; 
 long triIndex1, triIndex2, waterCellNum=0;
 long ptIndex = 0, cellNum = 0;
 Boolean foundPt = false, isOdd;
 OSErr err = 0;
 
 LONGH landWaterInfo = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
 LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
 
 LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
 LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
 GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
 
 TopologyHdl topo=0;
 LongPointHdl pts=0;
 VelocityFH velH = 0;
 DAGTreeStruct tree;
 WorldRect triBounds;
 
 LONGH boundaryPtsH = 0;
 LONGH boundaryEndPtsH = 0;
 LONGH waterBoundaryPtsH = 0;
 Boolean** segUsed = 0;
 SegInfoHdl segList = 0;
 
 TTriGridVel *triGrid = nil;
 tree.treeHdl = 0;
 TDagTree *dagTree = 0;
 
 
 if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
 
 for (i=0;i<fNumRows;i++)
 {
 for (j=0;j<fNumCols;j++)
 {
 // eventually will need to have a land mask, for now assume fillValue represents land
 //if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
 if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
 {
 INDEXH(landWaterInfo,i*fNumCols+j) = -1;	// may want to mark each separate island with a unique number
 }
 else
 {
 INDEXH(landWaterInfo,i*fNumCols+j) = 1;
 INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -2;	// water box
 INDEXH(ptIndexHdl,i*fNumCols_ext+j+1) = -2;
 INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j) = -2;
 INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1) = -2;
 }
 }
 }
 
 for (i=0;i<fNumRows_ext;i++)
 {
 for (j=0;j<fNumCols_ext;j++)
 {
 if (INDEXH(ptIndexHdl,i*fNumCols_ext+j) == -2)
 {
 INDEXH(ptIndexHdl,i*fNumCols_ext+j) = ptIndex;	// count up grid points
 ptIndex++;
 }
 else
 INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -1;
 }
 }
 
 for (i=0;i<fNumRows;i++)
 {
 for (j=0;j<fNumCols;j++)
 {
 if (INDEXH(landWaterInfo,i*fNumCols+j)>0)
 {
 INDEXH(gridCellInfo,i*fNumCols+j).cellNum = cellNum;
 cellNum++;
 INDEXH(gridCellInfo,i*fNumCols+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols_ext+j);
 INDEXH(gridCellInfo,i*fNumCols+j).topRight = INDEXH(ptIndexHdl,i*fNumCols_ext+j+1);
 INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j);
 INDEXH(gridCellInfo,i*fNumCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1);
 }
 else INDEXH(gridCellInfo,i*fNumCols+j).cellNum = -1;
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
 if(pts == nil)
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
 iIndex = n/fNumCols_ext;
 jIndex = n%fNumCols_ext;
 if (iIndex==0)
 {
 if (jIndex<fNumCols)
 {
 dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
 fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
 dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
 fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
 }
 else
 {
 dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
 dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
 fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
 dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
 dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
 fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
 }
 }
 else 
 {
 if (jIndex<fNumCols)
 {
 fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
 fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
 u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
 v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
 }
 else
 {
 dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
 fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
 dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
 fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
 }
 }
 vertex.v = (long)(fLat*1e6);
 vertex.h = (long)(fLong*1e6);
 
 fDepth = 1.;
 INDEXH(pts,i) = vertex;
 }
 else { // for outputting a verdat the last line should be all zeros
 index = 0;
 fLong = fLat = fDepth = 0.0;
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
 for (i=0;i<fNumRows;i++)
 {
 for (j=0;j<fNumCols;j++)
 {
 if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
 continue;
 waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
 triIndex1 = 2*waterCellNum;
 triIndex2 = 2*waterCellNum+1;
 // top/left tri in rect
 (*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
 (*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
 (*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
 if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
 (*topo)[triIndex1].adjTri1 = -1;
 else
 {
 (*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
 }
 (*topo)[triIndex1].adjTri2 = triIndex2;
 if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
 (*topo)[triIndex1].adjTri3 = -1;
 else
 {
 (*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
 }
 // bottom/right tri in rect
 (*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
 (*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
 (*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
 if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
 (*topo)[triIndex2].adjTri1 = -1;
 else
 {
 (*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
 }
 (*topo)[triIndex2].adjTri2 = triIndex1;
 if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
 (*topo)[triIndex2].adjTri3 = -1;
 else
 {
 (*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
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
 if (this -> moverMap != model -> uMap) goto setFields;	// don't try to create a map
 /////////////////////////////////////////////////
 // go through topo look for -1, and list corresponding boundary sides
 // then reorder as contiguous boundary segments - need to group boundary rects by islands
 // will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
 
 nSegs = 2*ntri; //number of -1's in topo
 boundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryPtsH));
 boundaryEndPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryEndPtsH));
 waterBoundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**waterBoundaryPtsH));
 segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
 segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
 // first go through rectangles and group by island
 // do this before making dagtree, 
 MySpinCursor(); // JLM 8/4/99
 err = NumberIslands(&maskH2, velocityH, landWaterInfo, &numIslands);	// numbers start at 3 (outer boundary)
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
 iIndex = rectIndex/fNumCols_ext;
 jIndex = rectIndex%fNumCols_ext;
 if (jIndex>0 && INDEXH(maskH2,iIndex*fNumCols_ext + jIndex-1)>=3)
 {
 (*segList)[segNum].isWater = 0;
 (*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols_ext + jIndex-1);	
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
 iIndex = rectIndex/fNumCols_ext;
 jIndex = rectIndex%fNumCols_ext;
 if (jIndex<fNumCols && INDEXH(maskH2,iIndex*fNumCols_ext + jIndex+1)>=3)
 {
 (*segList)[segNum].isWater = 0;
 (*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols_ext + jIndex+1);	
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
 iIndex = rectIndex/fNumCols_ext;
 jIndex = rectIndex%fNumCols_ext;
 if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*fNumCols_ext + jIndex)>=3)
 {
 (*segList)[segNum].isWater = 0;
 (*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*fNumCols_ext + jIndex);
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
 iIndex = rectIndex/fNumCols_ext;
 jIndex = rectIndex%fNumCols_ext;
 if (iIndex<fNumRows && INDEXH(maskH2,(iIndex+1)*fNumCols_ext + jIndex)>=3)
 {
 (*segList)[segNum].isWater = 0;
 (*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols_ext + jIndex);		// this should be the neighbor's value
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
 (*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
 (*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
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
 // clean up handles and set grid without a map
 if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
 if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
 if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
 goto setFields;
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
 
 fVerdatToNetCDFH = verdatPtsH;
 
 /////////////////////////////////////////////////
 
 triGrid = new TTriGridVel;
 if (!triGrid)
 {		
 err = true;
 TechError("Error in NetCDFWindMoverCurv::ReorderPoints()","new TTriGridVel",err);
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
 
 pts = 0;	// because fGrid is now responsible for it
 topo = 0; // because fGrid is now responsible for it
 velH = 0; // because fGrid is now responsible for it
 tree.treeHdl = 0; // because fGrid is now responsible for it
 velH = 0; // because fGrid is now responsible for it
 
 // probably will assume wind goes on a map
 //if (waterBoundaryPtsH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
 //{
 //PtCurMap *map = CreateAndInitPtCurMap(fPathName,triBounds); // the map bounds are the same as the grid bounds
 //if (!map) {err=-1; goto done;}
 // maybe move up and have the map read in the boundary information
 //map->SetBoundarySegs(boundaryEndPtsH);	
 //map->SetWaterBoundaries(waterBoundaryPtsH);
 //map->SetBoundaryPoints(boundaryPtsH);
 
 // *newMap = map;
 //}
 //else
 //{
 if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH=0;}
 if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH=0;}
 if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH=0;}
 //}
 
 /////////////////////////////////////////////////
 done:
 if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
 if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
 if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
 if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
 if (segList) {DisposeHandle((Handle)segList); segList = 0;}
 
 if(err)
 {
 if(!errmsg[0])
 strcpy(errmsg,"An error occurred in NetCDFWindMoverCurv::ReorderPoints");
 printError(errmsg); 
 if(pts) {DisposeHandle((Handle)pts); pts=0;}
 if(topo) {DisposeHandle((Handle)topo); topo=0;}
 if(velH) {DisposeHandle((Handle)velH); velH=0;}
 if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
 
 if(fGrid)
 {
 fGrid ->Dispose();
 delete fGrid;
 fGrid = 0;
 }
 if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
 if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
 if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
 if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
 if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
 
 if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
 if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
 if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
 }
 return err;
 }*/
// simplify for wind data - no map needed, no mask 
OSErr NetCDFWindMoverCurv_c::ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
	OSErr err = 0;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			// eventually will need to have a land mask, for now assume fillValue represents land
			//if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
			if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = 1;
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*fNumCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1) = -2;
			}
		}
	}
	
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*fNumCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)>0)
			{
				INDEXH(gridCellInfo,i*fNumCols+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*fNumCols+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols_ext+j);
				INDEXH(gridCellInfo,i*fNumCols+j).topRight = INDEXH(ptIndexHdl,i*fNumCols_ext+j+1);
				INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j);
				INDEXH(gridCellInfo,i*fNumCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1);
			}
			else INDEXH(gridCellInfo,i*fNumCols+j).cellNum = -1;
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
	if(pts == nil)
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
			iIndex = n/fNumCols_ext;
			jIndex = n%fNumCols_ext;
			if (iIndex==0)
			{
				if (jIndex<fNumCols)
				{
					dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
					fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
					fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
				}
				else
				{
					dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
				}
			}
			else 
			{
				if (jIndex<fNumCols)
				{
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
					u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
					v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
				}
				else
				{
					dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
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
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
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
	
	fVerdatToNetCDFH = verdatPtsH;
	
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFWindMoverCurv::ReorderPoints()","new TTriGridVel",err);
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
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFWindMoverCurv::ReorderPoints");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	return err;
}

OSErr NetCDFWindMoverCurv_c::GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp)
{
	float dLat, dLon, dLat1, dLon1, dLat2, dLon2, fLat, fLong;
	
	if (iIndex<0 || jIndex>fNumCols) return -1;
	if (iIndex==0)	// along the outer top or right edge need to add on dlat/dlon
	{					// velocities at a gridpoint correspond to lower left hand corner of a grid box, draw in grid center
		if (jIndex<fNumCols)
		{
			dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
			fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
			dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
			fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
		}
		else
		{
			dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
			dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
			fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1) - (INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
			dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
			dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
			fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1) - (INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
		}
	}
	else 
	{
		if (jIndex<fNumCols)
		{
			fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
			fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
		}
		else
		{
			dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
			fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
			dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
			fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
		}
	}
	(*wp).pLat = (long)(fLat*1e6);
	(*wp).pLong = (long)(fLong*1e6);
	
	return noErr;
}