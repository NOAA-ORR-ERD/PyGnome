/*
 *  GridWindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridWindMover_c.h"
#include "GridWindMover.h"
#include "GridVel.h"
#include "CROSS.H"

long GridWindMover_c::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect gridLRect, geoRect;
	ScaleRec	thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of GridWindMover
	
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

OSErr GridWindMover_c::PrepareForModelRun()
{
	return WindMover_c::PrepareForModelRun();
}

OSErr GridWindMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	OSErr err = 0;
	if (uncertain)
	{
		Seconds elapsed_time = model_time - fModelStartTime;
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}

	
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (!bActive) return noErr;
	
	err = dynamic_cast<GridWindMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
	
	if (err) goto done;	// might not want to have error if outside time interval
	
	fIsOptimizedForStep = true;	// is this needed?
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridWindMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	
	return err;
}

void GridWindMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}


WorldPoint3D GridWindMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double dLong, dLat;
	WorldPoint3D deltaPoint ={0,0,0.};
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
		err = dynamic_cast<GridWindMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
	
	// Check for constant wind 
	if(dynamic_cast<GridWindMover *>(this)->GetNumTimesInFile()==1)
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
		if (dynamic_cast<GridWindMover *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		//startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
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
