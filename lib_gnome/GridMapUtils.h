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
#include "RectUtils.h"
#include "ClassID_c.h"
#include "my_build_list.h"


#ifdef pyGNOME
#include "Mover_c.h"
#include "GridVel_c.h"
#include "TriGridVel_c.h"
#define TMover Mover_c
#define TGridVel GridVel_c
#define TTriGridVel TriGridVel_c
#else
#include "GridVel.h"
#endif

long 				CheckSurroundingPoints(LONGH maskH, long numRows, long  numCols, long row, long col) ;
Boolean 			InteriorLandPoint(LONGH maskH, long numRows, long  numCols, long row, long col); 
//Boolean 			ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long numRows, long  numCols, long row, long col) ;
Boolean 			ThereIsAdjacentLand2(LONGH maskH, DOUBLEH landmaskH, long numRows, long  numCols, long row, long col) ;
Boolean 			ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long numRows, long  numCols, long row, long col) ;
void 				ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin,long numRows,long numCols);
//OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);
OSErr 				NumberIslands(LONGH *islandNumberH, DOUBLEH landmaskH,LONGH landWaterInfo,long numRows,long numCols,long *numIslands);


#undef TMover
#undef TGridVel
#undef TTriGridVel
#endif
