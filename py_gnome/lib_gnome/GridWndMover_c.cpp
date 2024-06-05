/*
 *  GridWndMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "GridWndMover_c.h"
#include "GridWndMover.h"
#include "GridVel.h"
#include "CROSS.H"

long GridWndMover_c::GetVelocityIndex(WorldPoint p) 
{
    long rowNum, colNum;
    VelocityRec	velocity;

    LongRect gridLRect, geoRect;
    ScaleRec	thisScaleRec;

    // fNumRows, fNumCols members of GridWndMover
    TRectGridVel* rectGrid = (TRectGridVel*)fGrid;

    WorldRect bounds = rectGrid->GetBounds();

    SetLRect(&gridLRect, 0, fNumRows, fNumCols, 0);
    SetLRect(&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	

    GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);

    colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
    rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;

    if (colNum < 0 || colNum >= fNumCols ||
           rowNum < 0 || rowNum >= fNumRows) {
        return -1;
    }

    return rowNum * fNumCols + colNum;
}


OSErr GridWndMover_c::PrepareForModelRun()
{
	return WindMover_c::PrepareForModelRun();
}


OSErr GridWndMover_c::PrepareForModelStep(const Seconds& model_time,
                                          const Seconds& time_step,
                                          bool uncertain,
                                          int numLESets, int* LESetsSizesList)
{
	char errmsg[256];
	OSErr err = 0;

	errmsg[0] = 0;

	if (uncertain) {
		Seconds elapsed_time = model_time - fModelStartTime;
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}

	if (!bActive) return noErr;

	err = dynamic_cast<GridWndMover *>(this)->SetInterval(errmsg, model_time);

	// might not want to have error if outside time interval
	if (err) goto done;

	fIsOptimizedForStep = true;	// is this needed?

done:
    if (err) {
        if (!errmsg[0]) {
            strcpy(errmsg,
                   "An error occurred in GridWndMover::PrepareForModelStep");
        }

        printError(errmsg);
	}

	return err;
}


void GridWndMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}


WorldPoint3D GridWndMover_c::GetMove(const Seconds& model_time,
                                     Seconds timeStep,
                                     long setIndex, long leIndex,
                                     LERec *theLE, LETYPE leType)
{
	OSErr err = noErr;
	char errmsg[256];

	double dLong, dLat;
	double timeAlpha;
	long index; 

	Seconds startTime, endTime;
	Seconds time = model->GetModelTime();

	WorldPoint3D deltaPoint ={0, 0, 0.};
	WorldPoint refPoint = (*theLE).p;	

	VelocityRec windVelocity;

	// wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	// if ((*theLE).z > 0) return deltaPoint;

	if (!fIsOptimizedForStep) {
		err = dynamic_cast<GridWndMover *>(this)->SetInterval(errmsg, model_time);

		if (err) return deltaPoint;
	}

	index = GetVelocityIndex(refPoint);  // regular grid

	// Check for constant wind
    if (dynamic_cast<GridWndMover *>(this)->GetNumTimesInFile() == 1) {
        // Calculate the interpolated velocity at the point
        if (index >= 0) {
            	windVelocity.u = INDEXH(fStartData.dataHdl, index).u;
            	windVelocity.v = INDEXH(fStartData.dataHdl, index).v;
        }
        else {
            	windVelocity.u = 0.;
            	windVelocity.v = 0.;
        }
    }
    else {
        // time varying wind
        // Calculate the time weight factor
        if (dynamic_cast<GridWndMover *>(this)->GetNumFiles() > 1 && fOverLap)
            startTime = fOverLapStartTime;
        else
            startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;

        //startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
        endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
        timeAlpha = (endTime - time)/(double)(endTime - startTime);

        // Calculate the interpolated velocity at the point
        if (index >= 0) {
            windVelocity.u = timeAlpha * INDEXH(fStartData.dataHdl, index).u + (1 - timeAlpha) * INDEXH(fEndData.dataHdl, index).u;
            windVelocity.v = timeAlpha * INDEXH(fStartData.dataHdl, index).v + (1 - timeAlpha) * INDEXH(fEndData.dataHdl, index).v;
        }
        else {
            windVelocity.u = 0.;
            windVelocity.v = 0.;
        }
	}

	//scale:

    windVelocity.u *= fWindScale;
    windVelocity.v *= fWindScale;
    
    if (leType == UNCERTAINTY_LE) {
        err = AddUncertainty(setIndex, leIndex, &windVelocity);
    }

    windVelocity.u *= (*theLE).windage;
    windVelocity.v *= (*theLE).windage;

    dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
    dLat =   (windVelocity.v / METERSPERDEGREELAT) * timeStep;

    deltaPoint.p.pLong = dLong * 1000000;
    deltaPoint.p.pLat  = dLat  * 1000000;

    return deltaPoint;
}
