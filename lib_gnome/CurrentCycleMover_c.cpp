/*
 *  CurrentCycleMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "CurrentCycleMover_c.h"
#include "GridCurrentMover_c.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include <math.h>
#include <float.h>

#ifndef pyGNOME
#include "CROSS.H"
#include "GridCurMover.h"
#include "TShioTimeValue.h"
#else
#include "Replacements.h"
#endif

using std::cout;

#ifndef pyGNOME
CurrentCycleMover_c::CurrentCycleMover_c (TMap *owner, char *name) : GridCurrentMover_c(owner, name), CurrentMover_c(owner, name)
{
	timeDep = 0;
	bTimeFileActive = true;
	fPatternStartPoint = MaxFlood;	// this should be user input
}
#endif

CurrentCycleMover_c::CurrentCycleMover_c () : GridCurrentMover_c()
{
	timeDep = 0;
	bTimeFileActive = true;
	fPatternStartPoint = MaxFlood;	// this should be user input
	refP.pLat = 0;
	refP.pLong = 0;
}

void CurrentCycleMover_c::Dispose ()
{
#ifndef pyGNOME
	//DeleteTimeDep ();	// will pygnome be handling this?
	if (timeDep)
	{
		timeDep -> Dispose ();
		delete timeDep;
		timeDep = nil;
	}
#endif
	
	GridCurrentMover_c::Dispose ();
}

OSErr CurrentCycleMover_c::PrepareForModelRun()
{
	return GridCurrentMover_c::PrepareForModelRun();
}


OSErr CurrentCycleMover_c::PrepareForModelStep(const Seconds &model_time, const Seconds &time_step,
											  bool uncertain, int numLESets, int *LESetsSizesList)
{
	OSErr err = 0;
	char errmsg[256];
	
	errmsg[0] = 0;
	
	short ebbFloodType;
	long offset;
	float fraction;
	fraction = 0; offset = 0;	// for now

	if (!bActive)
		return noErr;
	
	if (bIsFirstStep)
	{
		timeGrid -> fModelStartTime = model_time;
		// get and apply our time file scale factor
		if (timeDep && bTimeFileActive) {
			// VelocityRec errVelocity={1,1};
			// JLM 11/22/99, if there are no time file values, use zero not 1
			VelocityRec errVelocity = {0, 1}, timeValue = {1.,1.};
	
			err = timeDep->GetTimeValue(model_time, &timeValue); // AH 07/10/2012
			if (err)
				timeValue = errVelocity;
		}
	
	}

	//printNote("Got Here - prepareformodelstep\n");		
	if (timeDep && bTimeFileActive) 
	{
	//printNote("Got Here - timedep prepareformodelstep\n");	
//#ifndef pyGNOME	
		dynamic_cast<TShioTimeValue*> (timeDep) -> GetLocationInTideCycle(model_time,&ebbFloodType,&fraction);
//#else
		//timeDep -> GetLocationInTideCycle(model_time,&ebbFloodType,&fraction);
//#endif
	//printNote("Got Here2 - timedep prepareformodelstep\n");		
		//timeDep->GetLocationInTideCycle(&ebbFloodType,&fraction);
		if (ebbFloodType>=fPatternStartPoint)
		{
			offset = ebbFloodType - fPatternStartPoint; // pass the fraction and offset to the timegrid
		}
		else
		{
			offset = ebbFloodType+4 - fPatternStartPoint;
		}
	}
	timeGrid -> SetTimeCycleInfo(fraction,offset);
	return GridCurrentMover_c::PrepareForModelStep(model_time, time_step, uncertain, numLESets, LESetsSizesList);
	
	// figure out location in tide cycle ...
	
	/*if (bIsFirstStep)
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
			strcpy(errmsg, "An error occurred in GridTideCurrentMover_c::PrepareForModelStep");
		printError(errmsg); 
	}	

	return err;*/
}


void CurrentCycleMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
	bIsFirstStep = false;
}


OSErr CurrentCycleMover_c::get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID) {

	//char errmsg[256];
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
		
	//sprintf(errmsg,"rec.p.pLat = %lf, rec.p.pLong = %lf\n",rec.p.pLat, rec.p.pLong);
	//printNote(errmsg);
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;
		
		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D CurrentCycleMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint3D refPoint;	
	double dLong, dLat;
	
	VelocityRec scaledPatVelocity = {0.,0.}, timeValue = {1.,1.};
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
	//sprintf(errmsg,"ref pt lat = %lf, ref pt long = %lf\n",refPoint.p.pLat, refPoint.p.pLong);
	//printNote(errmsg);
	
	scaledPatVelocity = timeGrid->GetScaledPatValue(model_time, refPoint);

	//sprintf(errmsg,"pat vel u = %lf, v = %lf\n",scaledPatVelocity.u, scaledPatVelocity.v);
	//printNote(errmsg);
	
	scaledPatVelocity.u *= fCurScale;
	scaledPatVelocity.v *= fCurScale;
	
	//sprintf(errmsg,"pat vel scaled u = %lf, scaled v = %lf\n",scaledPatVelocity.u, scaledPatVelocity.v);
	//printNote(errmsg);
	
	// get and apply our time file scale factor
	if (timeDep && bTimeFileActive) {
		// VelocityRec errVelocity={1,1};
		// JLM 11/22/99, if there are no time file values, use zero not 1
		VelocityRec errVelocity = {0, 1};

		err = timeDep->GetTimeValue(model_time, &timeValue); // AH 07/10/2012
		if (err)
			timeValue = errVelocity;
	}
	
	scaledPatVelocity.u *= myfabs(timeValue.u); // magnitude contained in u field only
	scaledPatVelocity.v *= myfabs(timeValue.u); 	// multiplying tide by tide, don't want to change phase
	//scaledPatVelocity.u *= timeValue.u; // magnitude contained in u field only
	//scaledPatVelocity.v *= timeValue.u; // magnitude contained in u field only

	//sprintf(errmsg,"pat vel tide = %lf, model time = %ld\n",timeValue.u, model_time);
	//printNote(errmsg);
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.p.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;

	//sprintf(errmsg,"dlong = %lf,dlat = %lf\n",dLong, dLat);
	//printNote(errmsg);
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}
// let the sub class do this ...
OSErr CurrentCycleMover_c::TextRead(char *path, char *topFilePath) 
{
	OSErr err = 0;
	short gridType, selectedUnits;
	char fileNamesPath[256], filePath[256];
	Boolean isNetCDFPathsFile = false;
	TimeGridVel *newTimeGrid = nil;

	//printNote("Got Here CCM\n");
	memset(fileNamesPath, 0, 256);
	memset(filePath, 0, 256);
	strcpy(filePath,path);	// this gets altered in IsNetCDFPathsFile, eventually change that function

	// code goes here, check if IsTideCurCyleMover...
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
			if (!err) newTimeGrid->bIsCycleMover = true;
			//err = this->InitMover(newTimeGrid);	// dummy variables for now
			err = newTimeGrid->TextRead(filePath, topFilePath);
		//printNote("Got Here CCM Text Read\n");
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
		//if (!err) timeGrid->bIsCycleMover = true;
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
	if (!err) timeGrid->bIsCycleMover = true;
	return err;

}

VelocityRec CurrentCycleMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("CurrentCycleMover_c::GetScaledPatValue is unimplemented");
	return v;
}


VelocityRec CurrentCycleMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("CurrentCycleMover_c::GetPatValue is unimplemented");
	return v;
}


