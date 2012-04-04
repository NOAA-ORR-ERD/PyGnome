/*
 *  RectGridVeL_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 3/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "RectGridVeL_c.h"
#include "RectUtils.h"
#include "MemUtils.h"
#include "GEOMETRY.H"

RectGridVel_c::RectGridVel_c(void)
{
	fGridHdl = 0;
	fNumRows = 0;
	fNumCols = 0;
}


void RectGridVel_c::Dispose ()
{
	if (fGridHdl)
	{
		DisposeHandle((Handle)fGridHdl);
		fGridHdl = nil;
	}
	GridVel_c::Dispose ();
}

long RectGridVel_c::NumVelsInGridHdl(void)
{
	long numInHdl = 0;
	if (fGridHdl) numInHdl = _GetHandleSize((Handle)fGridHdl)/sizeof(**fGridHdl);
	
	return numInHdl;
}

void RectGridVel_c::SetBounds(WorldRect bounds)
{
	// if we read on old style OSSM cur file, we take the bounds from the map
	// (The map calls SetBounds with its bounds)
	// BUT if we read a new style grid file, we already know the lat long and don't want the map overriding it 
	// so ignore the call to this function in that case
	if(EqualWRects(fGridBounds,emptyWorldRect))
	{
		fGridBounds = bounds; // we haven't set the bounds, take their value
	}
	else
	{
		// ignore their value, we already know the bounds
	}
}


VelocityRec RectGridVel_c::GetPatValue(WorldPoint p)
{
	
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, fGridBounds.loLong, fGridBounds.loLat, fGridBounds.hiLong, fGridBounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	//	gridP = WorldToScreenPoint(p, bounds, CATSgridRect);
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	
	if (!fGridHdl || colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ velocity.u = 0.0; velocity.v = 0.0; return velocity; }
	
	return INDEXH (fGridHdl, rowNum * fNumCols + colNum);
	
}

VelocityRec RectGridVel_c::GetSmoothVelocity(WorldPoint p)
{
	Point gridP;
	long rowNum, colNum;
	VelocityRec	velocity, velNew;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, fGridBounds.loLong, fGridBounds.loLat, fGridBounds.hiLong, fGridBounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	velocity = GetPatValue (p);
	
	if (colNum > 0 && colNum < fNumCols - 1 &&
		rowNum > 0 && rowNum < fNumRows - 1)
	{
		VelocityRec		topV, leftV, bottomV, rightV;
		
		topV    = INDEXH (fGridHdl, rowNum + 1 * fNumCols + colNum);
		bottomV = INDEXH (fGridHdl, rowNum - 1 * fNumCols + colNum);
		leftV   = INDEXH (fGridHdl, rowNum     * fNumCols + colNum - 1);
		rightV  = INDEXH (fGridHdl, rowNum     * fNumCols + colNum + 1);
		
		velNew.u = .5 * velocity.u + .125 * (topV.u + bottomV.u + leftV.u + rightV.u);
		velNew.v = .5 * velocity.v + .125 * (topV.v + bottomV.v + leftV.v + rightV.v);
	}
	else
		velNew = velocity;
	
	return velNew;
}