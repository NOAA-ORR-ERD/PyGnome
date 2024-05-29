/*
 *  GridMapUtils.h
 *  gnome
 *
 *  Created by Generic Programmer on 10/21/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef __GridMapUtils__
#define __GridMapUtils__

#include "Basics.h"
#include "TypeDefs.h"
//#include "my_build_list.h"


long 				CheckSurroundingPoints(LONGH maskH, long numRows, long  numCols, long row, long col) ;
Boolean 			InteriorLandPoint(LONGH maskH, long numRows, long  numCols, long row, long col); 
//Boolean 			ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long numRows, long  numCols, long row, long col) ;
Boolean 			ThereIsAdjacentLand2(LONGH maskH, DOUBLEH landmaskH, long numRows, long  numCols, long row, long col) ;
Boolean 			ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long numRows, long  numCols, long row, long col) ;
void 				ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin,long numRows,long numCols);
//OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);
OSErr 				NumberIslands(LONGH *islandNumberH, DOUBLEH landmaskH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);

#endif
