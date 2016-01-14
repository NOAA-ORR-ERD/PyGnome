/*
 *  GridCurMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridCurMover_c.h"
#include "GridCurMover.h"
#include "CROSS.H"

OSErr GridCurMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	//err = this -> UpdateUncertainty();
	//if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty
	
	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(velocity->u*velocity->u + velocity->v * velocity->v);
		
		
		u = velocity->u;
		v = velocity->v;
		
		//if(lengthS < fVar.uncertMinimumInMPS)
		if(lengthS < fEddyV0)	// reusing the variable for now...
		{
			// use a diffusion  ??
			printError("nonzero UNCERTMIN is unimplemented");
			//err = -1;
		}
		else
		{	// normal case, just use cross and down stuff
			alpha = unrec.downStream;
			beta = unrec.crossStream;
			
			velocity->u = u*(1+alpha)+v*beta;
			velocity->v = v*(1+alpha)-u*beta;	
		}
	}
	else 
	{
		TechError("GridCurMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}
OSErr GridCurMover_c::PrepareForModelRun()
{
	return CurrentMover_c::PrepareForModelRun();
}

OSErr GridCurMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (bIsFirstStep)
		fModelStartTime = model_time;
	
	//check to see that the time interval is loaded and set if necessary
	if (!bActive) return noErr;
	err = dynamic_cast<GridCurMover *>(this) -> SetInterval(errmsg, model_time);	// AH 07/17/2012
	
	if(err) goto done;
	
	if (uncertain)
	{
		Seconds elapsed_time = model_time - fModelStartTime;	// code goes here, how to set start time
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}

	fOptimize.isOptimizedForStep = true;	// don't  use CATS eddy diffusion stuff, follow ptcur
	//fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/model->GetTimeStep()); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	//fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridCurMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	return err;
}

void GridCurMover_c::ModelStepIsDone()
{
	fOptimize.isOptimizedForStep = false;
	bIsFirstStep = false;
}



WorldPoint3D GridCurMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long index; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	if(!fOptimize.isOptimizedForStep) 
	{
		err = dynamic_cast<GridCurMover *>(this) -> SetInterval(errmsg, model_time);	// AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	
	index = GetVelocityIndex(refPoint); 
	
	// Check for constant current 
	if(dynamic_cast<GridCurMover *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<GridCurMover *>(this)->GetNumFiles()>1))
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (dynamic_cast<GridCurMover *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		//startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	
	//scale:
	
	//scaledPatVelocity.u *= fVar.curScale; // may want to allow some sort of scale factor
	//scaledPatVelocity.v *= fVar.curScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

VelocityRec GridCurMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("GridCurMover::GetScaledPatValue is unimplemented");
	return v;
}

VelocityRec GridCurMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("GridCurMover::GetPatValue is unimplemented");
	return v;
}

long GridCurMover_c::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of GridCurMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return -1; }
	
	//return INDEXH (fGridHdl, rowNum * fNumCols + colNum);
	return rowNum * fNumCols + colNum;
}

/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize gridcur currents
double GridCurMover_c::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (fStartData.dataHdl && index>=0)
		u = INDEXH(fStartData.dataHdl,index).u;
	return u;
}

double GridCurMover_c::GetEndUVelocity(long index)
{
	double u = 0;
	if (fEndData.dataHdl && index>=0)
		u = INDEXH(fEndData.dataHdl,index).u;
	return u;
}

double GridCurMover_c::GetStartVVelocity(long index)
{
	double v = 0;
	if (fStartData.dataHdl && index >= 0)
		v = INDEXH(fStartData.dataHdl,index).v;
	return v;
}

double GridCurMover_c::GetEndVVelocity(long index)
{
	double v = 0;
	if (fEndData.dataHdl && index >= 0)
		v = INDEXH(fEndData.dataHdl,index).v;
	return v;
}

OSErr GridCurMover_c::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX && fTimeDataHdl)
		*startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
	else return -1;
	return 0;
}

OSErr GridCurMover_c::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX && fTimeDataHdl)
		*endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
	else return -1;
	return 0;
}

Boolean GridCurMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	long timeDataInterval,numTimesInFile = dynamic_cast<GridCurMover *>(this)->GetNumTimesInFile();
	Boolean intervalLoaded = dynamic_cast<GridCurMover *>(this) -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
	
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	
	if (!intervalLoaded || numTimesInFile == 0) return false; // no data don't try to show velocity
	if(numTimesInFile>1)
		//&& loaded && !err)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) return false;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}
	//if (loaded && !err)
	{
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear
		
		if (index >= 0)
		{
			// Check for constant current 
			if(numTimesInFile==1)
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
	//lengthS = this->fVar.curScale * lengthU;
	lengthS = lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);
	
	return true;
}