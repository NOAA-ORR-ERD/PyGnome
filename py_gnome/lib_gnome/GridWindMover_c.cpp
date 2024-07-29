/*
 *  GridWindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridWindMover_c.h"
#ifndef pyGNOME
#include "CROSS.H"
#include "TimeGridVel_c.h"
#include "GridWndMover.h"
#else
#include "Replacements.h"
#endif

#ifndef pyGNOME
GridWindMover_c::GridWindMover_c(TMap *owner, char *name) : WindMover_c(owner, name)
{
    if (!name || !name[0])
        this->SetClassName("Gridded Wind");
    else
        SetClassName(name); // short file name

	// use wind defaults for uncertainty
	bShowGrid = false;
	bShowArrows = false;

	fIsOptimizedForStep = false;

	fUserUnits = kMetersPerSec;	
	fWindScale = 1.;
	fArrowScale = 10.;

	timeGrid = 0;
}
#endif


GridWindMover_c::GridWindMover_c() : WindMover_c()
{
	fIsOptimizedForStep = false;
	
	fUserUnits = kMetersPerSec;	
	fWindScale = 1.;
	fArrowScale = 10.;
	
	timeGrid = 0;	
}


void GridWindMover_c::Dispose()
{
	if (timeGrid) {
		timeGrid->Dispose();
		delete timeGrid; // this causes a crash...
		timeGrid = nil;
	}
	
	WindMover_c::Dispose();
}


OSErr GridWindMover_c::PrepareForModelRun()
{
	return WindMover_c::PrepareForModelRun();
}


OSErr GridWindMover_c::PrepareForModelStep(const Seconds& model_time,
                                           const Seconds& time_step,
                                           bool uncertain,
                                           int numLESets, int* LESetsSizesList)
{
	OSErr err = 0;

	char errmsg[256];
	errmsg[0] = 0;

	if (bIsFirstStep)
		fModelStartTime = model_time;

	if (!bActive) return noErr;

	if (!timeGrid) return -1;

	err = timeGrid->SetInterval(errmsg, model_time);

	if (err) goto done;	// again don't want to have error if outside time interval

	if (uncertain) {
		// so uncertainty starts at time zero + uncertain_time_delay,
		// rather than a time step later
		Seconds elapsed_time = model_time + time_step - fModelStartTime;
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}

	fIsOptimizedForStep = true;	// is this needed?

done:
    if (err) {
        if (!errmsg[0])
            strcpy(errmsg,
                   "An error occurred in GridWindMover::PrepareForModelStep");

        printError(errmsg);
    }
	
	return err;
}


void GridWindMover_c::ModelStepIsDone()
{
	bIsFirstStep = false;
	fIsOptimizedForStep = false;
}


OSErr GridWindMover_c::get_move(int n, Seconds model_time, Seconds step_len,
                                WorldPoint3D *ref, WorldPoint3D *delta,
                                double *windages, short *LE_status,
                                LEType spillType, long spill_ID)
{
	if (!ref || !delta || !windages) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}

	// For LEType spillType, check to make sure it is within the valid values
	if (spillType < FORECAST_LE || spillType > UNCERTAINTY_LE) {
		// cout << "Invalid spillType.\n";
		return 2;
	}

	LERec rec, *prec;
	prec = &rec;

	WorldPoint3D zero_delta = {0, 0, 0.};

	for (int i = 0; i < n; i++) {

		// only operate on LE if the status is in water
		if (LE_status[i] != OILSTAT_INWATER) {
			delta[i] = zero_delta;
			continue;
		}

		rec.p = ref[i].p;
		rec.z = ref[i].z;
		rec.windage = windages[i];	// define the windage for the current LE

		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong *= 1000000;

		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);

		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}


WorldPoint3D GridWindMover_c::GetMove(const Seconds& model_time,
                                      Seconds timeStep,
                                      long setIndex, long leIndex,
                                      LERec *theLE, LETYPE leType)
{
	OSErr err = noErr;
	char errmsg[256];

	double 	dLong, dLat;

	WorldPoint3D	deltaPoint = {0, 0, 0.};
	WorldPoint3D refPoint;	

	VelocityRec windVelocity;

	errmsg[0] = 0;

	// wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	if ((*theLE).z > 0) return deltaPoint;

	if (!fIsOptimizedForStep) {
		err = timeGrid->SetInterval(errmsg, model_time);

		if (err) return deltaPoint;
	}

	refPoint.p = (*theLE).p;	
	refPoint.z = (*theLE).z;
	windVelocity = timeGrid->GetScaledPatValue(model_time, refPoint);

	windVelocity.u *= fWindScale; 
	windVelocity.v *= fWindScale; 

	if (leType == UNCERTAINTY_LE) {
		err = AddUncertainty(setIndex, leIndex, &windVelocity);
	}

	windVelocity.u *= (*theLE).windage;
	windVelocity.v *= (*theLE).windage;

	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.p.pLat);
	dLat =   (windVelocity.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;

	return deltaPoint;
}


OSErr GridWindMover_c::TextRead(char *path, char *topFilePath) 
{
	// this code is for curvilinear grids
	OSErr err = 0;
	char fileNamesPath[256], filePath[256];

	short gridType, selectedUnits;
	Boolean isNetCDFPathsFile = false;
	TimeGridVel *newTimeGrid = nil;

	memset(fileNamesPath, 0, 256);
	memset(filePath, 0, 256);

	// this gets altered in IsNetCDFPathsFile, eventually change that function
	strcpy(filePath, path);

	if (IsNetCDFFile(path, &gridType) ||
		IsNetCDFPathsFile(filePath, &isNetCDFPathsFile, fileNamesPath, &gridType))
	{
		if (gridType == CURVILINEAR) {
			newTimeGrid = new TimeGridWindCurv();
		}
		else {
			newTimeGrid = new TimeGridWindRect();
		}

		if (newTimeGrid) {
			err = newTimeGrid->TextRead(filePath, topFilePath);
			if (err) return err;

			this->SetTimeGrid(newTimeGrid);
		}

		if (isNetCDFPathsFile) {
			err = timeGrid->ReadInputFileNames(fileNamesPath);
			if (err) return err;

			timeGrid->DisposeAllLoadedData();
		}

		return err;
	}

	// All other file formats are line-formatted text files
	vector<string> linesInFile;
	if (ReadLinesInFile(path, linesInFile)) {
		linesInFile = rtrim_empty_lines(linesInFile);
	}
	else
		return -1; // we failed to read in the file.

	if (IsGridWindFile(linesInFile, &selectedUnits)) {
		newTimeGrid = new TimeGridCurRect();
		if (newTimeGrid) {
			dynamic_cast<TimeGridCurRect*>(newTimeGrid)->fUserUnits = selectedUnits;

			err = newTimeGrid->TextRead(path, "");
			if (err) goto Error;

			this->SetTimeGrid(newTimeGrid);

			if (!err) {
				timeGrid->DisposeAllLoadedData();
			}
		}
	}
	else {
		err = -1;
		return err;
	}

Error:
	return err;
}


OSErr GridWindMover_c::GetScaledVelocities(Seconds model_time,
                                           VelocityFRec *velocities)
{
    return timeGrid->GetScaledVelocities(model_time, velocities);
}


LongPointHdl GridWindMover_c::GetPointsHdl(void)
{
	if ((timeGrid->fVar.gridType) == CURVILINEAR) {
		return timeGrid->fGrid->GetPointsHdl();
	}
	else {
		return timeGrid->GetPointsHdl();
	}
}


GridCellInfoHdl GridWindMover_c::GetCellDataHdl(void)
{
    return timeGrid->GetCellData();
}


WORLDPOINTH	GridWindMover_c::GetCellCenters(void)
{
    // should rename this function...
    return timeGrid->GetCellCenters();
}


long GridWindMover_c::GetNumTriangles(void)
{
    long numTriangles = 0;
    TopologyHdl topoH = timeGrid->fGrid->GetTopologyHdl();

    if (topoH)
        numTriangles = _GetHandleSize((Handle)topoH) / sizeof(**topoH);

    return numTriangles;
}


long GridWindMover_c::GetNumPoints(void)
{
	long numPoints = 0;

	numPoints = (timeGrid->fNumRows) * (timeGrid->fNumCols);	

	return numPoints;
}
