/*
 *  GridCurrentMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridCurrentMover_c.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include <math.h>
#include <float.h>

#ifndef pyGNOME
#include "CROSS.H"
#include "GridCurMover.h"
#else
#include "Replacements.h"
#endif

using std::cout;

#ifndef pyGNOME
GridCurrentMover_c::GridCurrentMover_c (TMap *owner, char *name) : CurrentMover_c(owner, name), Mover_c(owner, name)
{
	timeGrid = 0;
	memset(&fUncertainParams,0,sizeof(fUncertainParams));
	//fArrowScale = 1.;
	//fArrowDepth = 0;
	if (gNoaaVersion)
	{
		fUncertainParams.alongCurUncertainty = .5;
		fUncertainParams.crossCurUncertainty = .25;
		fUncertainParams.durationInHrs = 24.0;
	}
	else
	{
		fUncertainParams.alongCurUncertainty = 0.;
		fUncertainParams.crossCurUncertainty = 0.;
		fUncertainParams.durationInHrs = 0.;
	}
	//fVar.uncertMinimumInMPS = .05;
	fUncertainParams.uncertMinimumInMPS = 0.0;
	fCurScale = 1.0;
	fUncertainParams.startTimeInHrs = 0.0;
	//fGridType = TWO_D; // 2D default
	//fVar.maxNumDepths = 1;	// 2D default - may always be constant for netCDF files
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fUncertainParams.alongCurUncertainty; 
	fUpCurUncertainty = fUncertainParams.alongCurUncertainty; 	
	fRightCurUncertainty = fUncertainParams.crossCurUncertainty;  
	fLeftCurUncertainty = -fUncertainParams.crossCurUncertainty; 
	fDuration=fUncertainParams.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fUncertainParams.startTimeInHrs*3600.);
	//
	
	fIsOptimizedForStep = false;
		
	SetClassName (name); // short file name
	
	fAllowVerticalExtrapolationOfCurrents = false;
	fMaxDepthForExtrapolation = 0.;	// assume 2D is just surface
	
}
#endif

GridCurrentMover_c::GridCurrentMover_c () : CurrentMover_c()
{
	timeGrid = 0;
	memset(&fUncertainParams,0,sizeof(fUncertainParams));

	//if (gNoaaVersion)
	{
		fUncertainParams.alongCurUncertainty = .5;
		fUncertainParams.crossCurUncertainty = .25;
		fUncertainParams.durationInHrs = 24.0;
	}
	/*else
	{
		fVar.alongCurUncertainty = 0.;
		fVar.crossCurUncertainty = 0.;
		fVar.durationInHrs = 0.;
	}*/
	//fVar.uncertMinimumInMPS = .05;
	fUncertainParams.uncertMinimumInMPS = 0.0;
	fCurScale = 1.0;
	fUncertainParams.startTimeInHrs = 0.0;
	//fGridType = TWO_D; // 2D default
	//fVar.maxNumDepths = 1;	// 2D default - may always be constant for netCDF files
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fUncertainParams.alongCurUncertainty; 
	fUpCurUncertainty = fUncertainParams.alongCurUncertainty; 	
	fRightCurUncertainty = fUncertainParams.crossCurUncertainty;  
	fLeftCurUncertainty = -fUncertainParams.crossCurUncertainty; 
	fDuration=fUncertainParams.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fUncertainParams.startTimeInHrs*3600.);
	//
	
	fIsOptimizedForStep = false;
	
	//SetClassName (name); // short file name
	
	fAllowVerticalExtrapolationOfCurrents = false;
	fMaxDepthForExtrapolation = 0.;	// assume 2D is just surface
	
}

void GridCurrentMover_c::Dispose ()
{
	if (timeGrid)
	{
		timeGrid -> Dispose();
		delete timeGrid;	// this causes a crash...
		timeGrid = nil;
	}
	
	CurrentMover_c::Dispose ();
}

OSErr GridCurrentMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty
		
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(velocity->u*velocity->u + velocity->v * velocity->v);
		
		
		u = velocity->u;
		v = velocity->v;
		
		if(lengthS < fUncertainParams.uncertMinimumInMPS)
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
		TechError("GridCurrentMover_c::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}


OSErr GridCurrentMover_c::PrepareForModelRun()
{
	return CurrentMover_c::PrepareForModelRun();
}


OSErr GridCurrentMover_c::PrepareForModelStep(const Seconds &model_time, const Seconds &time_step,
											  bool uncertain, int numLESets, int *LESetsSizesList)
{
	OSErr err = 0;
	char errmsg[256];
	
	errmsg[0] = 0;
	
	if (bIsFirstStep)
		fModelStartTime = model_time;

	if (!bActive)
		return noErr;
	
	if (!timeGrid)
		return -1;

	err = timeGrid->SetInterval(errmsg, model_time);
	if (err)
		goto done;
	
	if (uncertain)
	{
		Seconds elapsed_time = model_time - fModelStartTime;	// code goes here, how to set start time
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}
	fIsOptimizedForStep = true;
	
done:
	
	if (err) {
		if (!errmsg[0])
			strcpy(errmsg, "An error occurred in GridCurrentMover_c::PrepareForModelStep");
		printError(errmsg); 
	}	

	return err;
}


void GridCurrentMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
	bIsFirstStep = false;
}


OSErr GridCurrentMover_c::get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID) {

	if(!ref || !delta) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	

	// For LEType spillType, check to make sure it is within the valid values
	if( spillType < FORECAST_LE || spillType > UNCERTAINTY_LE)
	{
		// cout << "Invalid spillType.\n";
		return 2;
	}
	
	LERec* prec;
	LERec rec;
	prec = &rec;
	
	WorldPoint3D zero_delta ={0,0,0.};
	
	for (int i = 0; i < n; i++) {
		
		// only operate on LE if the status is in water
		if( LE_status[i] != OILSTAT_INWATER)
		{
			delta[i] = zero_delta;
			continue;
		}
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;
		
		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D GridCurrentMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint3D refPoint;	
	double dLong, dLat;
	
	VelocityRec scaledPatVelocity = {0.,0.};
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	if(!fIsOptimizedForStep) 
	{
		err = timeGrid->SetInterval(errmsg, model_time); 
		
		if (err) return deltaPoint;
	}

	refPoint.p = (*theLE).p;	
	refPoint.z = (*theLE).z;
	scaledPatVelocity = timeGrid->GetScaledPatValue(model_time, refPoint);

	scaledPatVelocity.u *= fCurScale;
	scaledPatVelocity.v *= fCurScale;
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.p.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

OSErr GridCurrentMover_c::TextRead(char *path, char *topFilePath) 
{
	// this code is for curvilinear grids
	OSErr err = 0;
	short gridType, selectedUnits;
	char fileNamesPath[256], filePath[256];
	Boolean isNetCDFPathsFile = false;
	TimeGridVel *newTimeGrid = nil;

	memset(fileNamesPath, 0, 256);
	memset(filePath, 0, 256);
	strcpy(filePath,path);	// this gets altered in IsNetCDFPathsFile, eventually change that function

	//  NetCDF is a binary file format, so we will continue to just pass in a path.
	if (IsNetCDFFile(path, &gridType) ||
		IsNetCDFPathsFile(filePath, &isNetCDFPathsFile, fileNamesPath, &gridType))
	{
		if (gridType == CURVILINEAR) {
			newTimeGrid = new TimeGridVelCurv();
		}
		else if (gridType == TRIANGULAR) {
			newTimeGrid = new TimeGridVelTri();
		}
		else {
			newTimeGrid = new TimeGridVelRect();
		}

		if (newTimeGrid) {
			// TODO: This would be more efficient if IsNetCDFFile() would leave the file
			//       open and pass back an active ncid
			err = newTimeGrid->TextRead(filePath, topFilePath);
			if(err) return err;
			this->SetTimeGrid(newTimeGrid);
		}

		if (isNetCDFPathsFile) {
			char errmsg[256];

			err = timeGrid->ReadInputFileNames(fileNamesPath);
			if (err)
				return err;

			timeGrid->DisposeAllLoadedData();
			//err = ((NetCDFMover*)newMover)->SetInterval(errmsg);	// if set interval here will get error if times are not in model range
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

	if (IsPtCurFile(linesInFile))
	{
		char errmsg[256];

		newTimeGrid = new TimeGridCurTri();
		if (newTimeGrid)
		{
			//err = this->InitMover(newTimeGrid);
			//if(err) goto Error;
			// do this in two steps since we need the uncertainty parameters for the grid mover
			err = dynamic_cast<TimeGridCurTri*>(newTimeGrid)->ReadHeaderLines(path,&fUncertainParams);
			err = newTimeGrid->TextRead(path,"");
			if(err) return err;

			// set the TCurrentMover uncertainty parameters 
			// only along and across can be set in the file (uncertainty min is not implemented so should stay zero
			fDownCurUncertainty = -fUncertainParams.alongCurUncertainty; 
			fUpCurUncertainty = fUncertainParams.alongCurUncertainty; 	

			fRightCurUncertainty = fUncertainParams.crossCurUncertainty;  
			fLeftCurUncertainty = -fUncertainParams.crossCurUncertainty; 
			
			this->SetTimeGrid(newTimeGrid);
			if (!err) /// JLM 5/3/10
			{
				//char errmsg[256];
				//err = timeGrid->ReadInputFileNames(fileNamesPath);
				timeGrid->DisposeAllLoadedData();
				//if(!err) err = timeGrid->SetInterval(errmsg,model->GetModelTime()); // if set interval here will get error if times are not in model range
			}
		}
	}
	else if (IsGridCurTimeFile(linesInFile, &selectedUnits))
	{
		//cerr << "we are opening a GridCurTimeFile..." << "'" << path << "'" << endl;
		char errmsg[256];
		newTimeGrid = new TimeGridCurRect();
		//timeGrid = new TimeGridVel();
		if (newTimeGrid)
		{			
			//err = this->InitMover(timeGrid);
			//if(err) goto Error;
			dynamic_cast<TimeGridCurRect*>(newTimeGrid)->fUserUnits = selectedUnits;
			err = newTimeGrid->TextRead(path,"");
			if(err) goto Error;
			this->SetTimeGrid(newTimeGrid);
			if (!err /*&& isNetCDFPathsFile*/) /// JLM 5/3/10
			{
				//char errmsg[256];
				//err = timeGrid->ReadInputFileNames(fileNamesPath);
				/*if(!err)*/ timeGrid->DisposeAllLoadedData();
				//if(!err) err = timeGrid->SetInterval(errmsg,model->GetModelTime()); // if set interval here will get error if times are not in model range
			}
		}
	}
	else {
		err = -1; return err;
	}

	Error: // JLM 	 10/27/98
	//if(newMover) {newMover->Dispose();delete newMover;newMover = 0;};
	//return 0;
	return err;

}


VelocityRec GridCurrentMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("GridCurrentMover_c::GetScaledPatValue is unimplemented");
	return v;
}


VelocityRec GridCurrentMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("GridCurrentMover_c::GetPatValue is unimplemented");
	return v;
}


