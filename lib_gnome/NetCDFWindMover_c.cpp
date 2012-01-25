/*
 *  NetCDFWindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "NetCDFWindMover_c.h"
#include "CROSS.H"

long NetCDFWindMover_c::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFWindMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return -1; }
	
	return rowNum * fNumCols + colNum;
}

LongPoint NetCDFWindMover_c::GetVelocityIndices(WorldPoint p) 
{
	long rowNum, colNum;
	LongPoint indices = {-1,-1};
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return indices; }
	
	//return rowNum * fNumCols + colNum;
	indices.h = colNum;
	indices.v = rowNum;
	return indices;
}


/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize netcdf currents
double NetCDFWindMover_c::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (index>=0)
	{
		if (fStartData.dataHdl) u = INDEXH(fStartData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFWindMover_c::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		if (fEndData.dataHdl) u = INDEXH(fEndData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFWindMover_c::GetStartVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fStartData.dataHdl) v = INDEXH(fStartData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

double NetCDFWindMover_c::GetEndVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fEndData.dataHdl) v = INDEXH(fEndData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

OSErr NetCDFWindMover_c::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX)
		*startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

OSErr NetCDFWindMover_c::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX)
		*endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

Boolean NetCDFWindMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	LongPoint indices;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!bShowArrows && !bShowGrid) return 0;
	err = /*CHECK*/dynamic_cast<NetCDFWindMover *>(this) -> SetInterval(errmsg);
	if(err) return false;
	
	if(/*OK*/dynamic_cast<NetCDFWindMover *>(this)->GetNumTimesInFile()>1)
		//&& loaded && !err)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) /*return false;*/
		{
			if ((time > startTime || time < startTime) && fAllowExtrapolationOfWinds)
			{
				timeAlpha = 1;
			}
			else
				return false;
		}
		else
			timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}
	//if (loaded && !err)
	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular
		
		indices = this->GetVelocityIndices(wp.p);
		
		if (index >= 0)
		{
			// Check for constant current 
			if(/*OK*/dynamic_cast<NetCDFWindMover *>(this)->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				velocity.u = this->GetStartUVelocity(index);
				velocity.v = this->GetStartVVelocity(index);
			}
			else // time varying current
			{
				velocity.u = timeAlpha*this->GetStartUVelocity(index) + (1-timeAlpha)*this->GetEndUVelocity(index);
				velocity.v = timeAlpha*this->GetStartVVelocity(index) + (1-timeAlpha)*this->GetEndVVelocity(index);
			}
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fWindScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	//sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
	//	this->className, uStr, sStr);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
			this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
	
	return true;
}


OSErr NetCDFWindMover_c::PrepareForModelStep()
{
	OSErr err = this->UpdateUncertainty();
	
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (!bActive) return noErr;
	
	err = /*CHECK*/dynamic_cast<NetCDFWindMover *>(this) -> SetInterval(errmsg); // SetInterval checks to see that the time interval is loaded
	if (err) goto done;	// again don't want to have error if outside time interval
	
	fIsOptimizedForStep = true;	// is this needed?
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFWindMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	
	return err;
}

void NetCDFWindMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}


WorldPoint3D NetCDFWindMover_c::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double 	dLong, dLat;
	WorldPoint3D	deltaPoint ={0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double timeAlpha;
	long index; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec windVelocity;
	OSErr err = noErr;
	char errmsg[256];
	
	// if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	
	if(!fIsOptimizedForStep) 
	{
		err = /*CHECK*/dynamic_cast<NetCDFWindMover *>(this) -> SetInterval(errmsg);	// ok, but don't print error message heref
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
	
	// Check for constant wind 
	if( ( /*OK*/dynamic_cast<NetCDFWindMover *>(this)->GetNumTimesInFile()==1 && !( dynamic_cast<NetCDFWindMover *>(this)->GetNumFiles() > 1 ) ) ||
	   (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
		//if(GetNumTimesInFile()==1)
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
		if (/*OK*/dynamic_cast<NetCDFWindMover *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
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
	
	//scale:
	
	windVelocity.u *= fWindScale; 
	windVelocity.v *= fWindScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		err = AddUncertainty(setIndex,leIndex,&windVelocity);
	}
	
	windVelocity.u *=  (*theLE).windage;
	windVelocity.v *=  (*theLE).windage;
	
	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat =   (windVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

Seconds NetCDFWindMover_c::GetTimeValue(long index)
{
	if (index<0) printError("Access violation in NetCDFWindMover::GetTimeValue()");
	Seconds time = (*fTimeHdl)[index] + fTimeShift;
	return time;
}

