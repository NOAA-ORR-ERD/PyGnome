/*
 *  GridMapUtils.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridMapUtils.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif



long CheckSurroundingPoints(LONGH maskH, long numRows, long  numCols, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long neighbor;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < numRows - 1) ? row + 1 : numRows - 1;
	jEnd = (col < numCols - 1) ? col + 1 : numCols - 1;
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
		neighbor = INDEXH(maskH, i*numCols + col);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*numCols + j);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	return lowestLandIndex;
}

//Boolean NetCDFMoverCurv_c::ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long numRows, long  numCols, long row, long col) 
Boolean ThereIsAdjacentLand2(LONGH maskH, DOUBLEH landmaskH, long numRows, long  numCols, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long neighbor;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < numRows - 1) ? row + 1 : numRows - 1;
	jEnd = (col < numCols - 1) ? col + 1 : numCols - 1;
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
		neighbor = INDEXH(maskH, i*numCols + col);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		//if (neighbor >= 3 || (INDEXH(velocityH,i*numCols+col).u==fFillValue && INDEXH(velocityH,i*numCols+col).v==fFillValue)) return true;
		if (neighbor >= 3 || INDEXH(landmaskH,i*numCols+col)==0) return true;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*numCols + j);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		//if (neighbor >= 3 || (INDEXH(velocityH,row*numCols+j).u==fFillValue && INDEXH(velocityH,row*numCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || INDEXH(landmaskH,row*numCols+j)==0) return true;
	}
	return false;
}

Boolean InteriorLandPoint(LONGH maskH, long numRows, long numCols, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd;
	long neighbor;
	long numRows_ext = numRows+1, numCols_ext = numCols+1;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < numRows_ext - 1) ? row + 1 : numRows_ext - 1;
	jEnd = (col < numCols_ext - 1) ? col + 1 : numCols_ext - 1;
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
			neighbor = INDEXH(maskH, i*numCols_ext + j);
			if (neighbor < 3 /*&& neighbor != -1*/)	// water point
				return false;
			//if (row==1 && INDEXH(maskH,j)==1) return false;
		}
	}
	return true;
}

Boolean ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long numRows, long numCols, long row, long col) 
{
	long iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long i, j, neighbor, landPolyNum;
	long numRows_ext = numRows+1, numCols_ext = numCols+1;
	
	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < numRows_ext - 1) ? row + 1 : numRows_ext - 1;
	jEnd = (col < numCols_ext - 1) ? col + 1 : numCols_ext - 1;
	
	landPolyNum = INDEXH(maskH, row*numCols_ext + col);
	for (i = iStart; i< iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*numCols_ext + col);
		if (neighbor >= 3 && neighbor < landPolyNum) 
		{
			*lowerPolyNum = neighbor;
			return true;
		}
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*numCols_ext + j);
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

void ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin,long numRows,long numCols)
{	// merges adjoining land blocks and then renumbers any higher numbered land blocks
	long i,j,val;
	long numRows_ext = numRows+1, numCols_ext = numCols+1;
	
	for (i=0;i<numRows_ext;i++)
	{
		for (j=0;j<numCols_ext;j++)
		{	
			val = INDEXH(maskH,i*numCols_ext+j);
			if (val==landBlockToMerge) INDEXH(maskH,i*numCols_ext+j) = landBlockToJoin;
			if (val>landBlockToMerge) INDEXH(maskH,i*numCols_ext+j) -= 1;
		}
	}
}

//OSErr NetCDFMoverCurv_c::NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo, long numRows, long  numCols, long *numIslands) 
OSErr NumberIslands(LONGH *islandNumberH, DOUBLEH landmaskH,LONGH landWaterInfo, long numRows, long  numCols, long *numIslands) 
{
	OSErr err = 0;
	long numRows_ext = numRows+1, numCols_ext = numCols+1;
	long nv = numRows * numCols, nv_ext = numRows_ext*numCols_ext;
	long i, j, n, landPolyNum = 1, lowestSurroundingNum = 0;
	long islandNum, maxIslandNum=3;
	LONGH maskH = (LONGH)_NewHandleClear(numRows * numCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	*islandNumberH = 0;
	
	if (!maskH || !maskH2) {err = memFullErr; goto done;}
	// use surface velocity values at time zero
	for (i=0;i<numRows;i++)
	{
		for (j=0;j<numCols;j++)
		{
			if (INDEXH(landWaterInfo,i*numCols+j) == -1)// 1 water, -1 land
			{
				if (i==0 || i==numRows-1 || j==0 || j==numCols-1)
				{
					INDEXH(maskH,i*numCols+j) = 3;	// set outer boundary to 3
				}
				else
				{
					if (landPolyNum==1)
					{	// Land point
						INDEXH(maskH,i*numCols+j) = landPolyNum+3;
						landPolyNum+=3;
					}
					else
					{
						// check for nearest land poly number
						if ((lowestSurroundingNum = CheckSurroundingPoints(maskH, numRows, numCols, i, j)) >= 3)
						{
							INDEXH(maskH,i*numCols+j) = lowestSurroundingNum;
						}
						else
						{
							INDEXH(maskH,i*numCols+j) = landPolyNum;
							landPolyNum += 1;
						}
					}
				}
			}
			else
			{
				if (i==0 || i==numRows-1 || j==0 || j==numCols-1)
					INDEXH(maskH,i*numCols+j) = 1;	// Open water boundary
				//else if (ThereIsAdjacentLand2(maskH,velocityH,numRows,numCols,i,j))
				else if (ThereIsAdjacentLand2(maskH,landmaskH,numRows,numCols,i,j))
					INDEXH(maskH,i*numCols+j) = 2;	// Water boundary, not open water
				else
					INDEXH(maskH,i*numCols+j) = 0;	// Interior water point
			}
		}
	}
	// extend grid by one row/col up/right since velocities correspond to lower left corner of a grid box
	for (i=0;i<numRows_ext;i++)
	{
		for (j=0;j<numCols_ext;j++)
		{
			if (i==0) 
			{
				if (j!=numCols)
					INDEXH(maskH2,j) = INDEXH(maskH,j);	// flag for extra boundary point
				else
					INDEXH(maskH2,j) = INDEXH(maskH,j-1);	
				
			}
			else if (i!=0 && j==numCols) 
				INDEXH(maskH2,i*numCols_ext+numCols) = INDEXH(maskH,(i-1)*numCols+numCols-1);
			else 
			{	
				INDEXH(maskH2,i*numCols_ext+j) = INDEXH(maskH,(i-1)*numCols+j);
			}
		}
	}
	
	// set original top/right boundaries to interior water points 
	// probably don't need to do this since we aren't paying attention to water types anymore
	for (j=1;j<numCols_ext-1;j++)	 
	{
		if (INDEXH(maskH2,numCols_ext+j)==1) INDEXH(maskH2,numCols_ext+j) = 2;
	}
	for (i=1;i<numRows_ext-1;i++)
	{
		if (INDEXH(maskH2,i*numCols_ext+numCols-1)==1) INDEXH(maskH2,i*numCols_ext+numCols-1) = 2;
	}
	// now merge any contiguous land blocks (max of landPolyNum)
	// as soon as find one, all others of that number change, and every higher landpoint changes
	// repeat until nothing changes
startLoop:
	{
		long lowerPolyNum = 0;
		for (i=0;i<numRows_ext;i++)
		{
			for (j=0;j<numCols_ext;j++)
			{
				if (INDEXH(maskH2,i*numCols_ext+j) < 3) continue;	// water point
				if (ThereIsALowerLandNeighbor(maskH2,&lowerPolyNum,numRows,numCols,i,j))
				{
					ResetMaskValues(maskH2,INDEXH(maskH2,i*numCols_ext+j),lowerPolyNum,numRows,numCols);
					goto startLoop;
				}
				if ((i==0 || i==numRows_ext-1 || j==0 || j==numCols_ext-1) && INDEXH(maskH2,i*numCols_ext+j)>3)
				{	// shouldn't get here
					ResetMaskValues(maskH2,INDEXH(maskH2,i*numCols_ext+j),3,numRows,numCols);
					goto startLoop;
				}
			}
		}
	}
	for (i=0;i<numRows_ext;i++)
	{
		for (j=0;j<numCols_ext;j++)
		{	// note, the numbers start at 3
			islandNum = INDEXH(maskH2,i*numCols_ext+j);
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
