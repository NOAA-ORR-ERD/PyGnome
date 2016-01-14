/*
 *  IceWindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "IceWindMover_c.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include <math.h>
#include <float.h>

#ifndef pyGNOME
#include "CROSS.H"
#include "GridWindMover.h"
#else
#include "Replacements.h"
#endif

using std::cout;

#ifndef pyGNOME
IceWindMover_c::IceWindMover_c (TMap *owner, char *name) : GridWindMover_c(owner, name), WindMover_c(owner, name)
{
	if(!name || !name[0]) this->SetClassName("Gridded Wind");
	else 	SetClassName (name); // short file name
	
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

IceWindMover_c::IceWindMover_c () : GridWindMover_c()
{
	//fIsOptimizedForStep = false;
	
	//fUserUnits = kMetersPerSec;	
	//fWindScale = 1.;
	//fArrowScale = 10.;
	
	//timeGrid = 0;
	
}

void IceWindMover_c::Dispose ()
{
	if (timeGrid)
	{
		timeGrid -> Dispose();
		delete timeGrid;	// this causes a crash...
		timeGrid = nil;
	}
	
	GridWindMover_c::Dispose ();
}

/*OSErr IceWindMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
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
		TechError("IceWindMover_c::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}*/


OSErr IceWindMover_c::PrepareForModelRun()
{
	return GridWindMover_c::PrepareForModelRun();
}


OSErr IceWindMover_c::PrepareForModelStep(const Seconds &model_time, const Seconds &time_step,
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
			strcpy(errmsg, "An error occurred in IceWindMover_c::PrepareForModelStep");
		printError(errmsg); 
	}	

	return err;
}


void IceWindMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
	bIsFirstStep = false;
}


OSErr IceWindMover_c::get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID) {

	if(!ref || !delta || !windages) {
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
		rec.windage = windages[i];	// define the windage for the current LE
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;
		
		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D IceWindMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint3D refPoint;	
	double dLong, dLat;
	
	VelocityRec windVelocity = {0.,0.};
	OSErr err = 0;
	char errmsg[256];
	
	errmsg[0] = 0;
	
	if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface

	if(!fIsOptimizedForStep) 
	{
		err = timeGrid->SetInterval(errmsg, model_time); 
		
		if (err) return deltaPoint;
	}

	refPoint.p = (*theLE).p;	
	refPoint.z = (*theLE).z;
	windVelocity = timeGrid->GetScaledPatValue(model_time, refPoint);

	windVelocity.u *= fWindScale; 
	windVelocity.v *= fWindScale; 
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&windVelocity);
	}
	
	windVelocity.u *=  (*theLE).windage;
	windVelocity.v *=  (*theLE).windage;
	
	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.p.pLat);
	dLat  =  (windVelocity.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

OSErr IceWindMover_c::TextRead(char *path, char *topFilePath) 
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
			newTimeGrid = new TimeGridWindIce();	// need to handle different grids...
		}
		/*else if (gridType == TRIANGULAR) {
			newTimeGrid = new TimeGridVelTri();
		}
		else {
			newTimeGrid = new TimeGridVelRect();
		}*/

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
	/*vector<string> linesInFile;
	if (ReadLinesInFile(path, linesInFile)) {
		linesInFile = rtrim_empty_lines(linesInFile);
	}
	else
		return -1; // we failed to read in the file.

	//if (IsGridWindFile(path, &selectedUnits))	// check if gui gnome need this
	if (IsGridWindFile(linesInFile, &selectedUnits))
	{
		//char errmsg[256];
		newTimeGrid = new TimeGridCurRect();
		//timeGrid = new TimeGridVel();
		if (newTimeGrid) {
			//err = this->InitMover(timeGrid);
			//if(err) goto Error;
			dynamic_cast<TimeGridCurRect*>(newTimeGrid)->fUserUnits = selectedUnits;
			err = newTimeGrid->TextRead(path,"");
			if(err) goto Error;
			this->SetTimeGrid(newTimeGrid);
			if (!err) {
				//char errmsg[256];
				//err = timeGrid->ReadInputFileNames(fileNamesPath);
				//if(!err)
				timeGrid->DisposeAllLoadedData();
				//if(!err) err = timeGrid->SetInterval(errmsg,model->GetModelTime()); // if set interval here will get error if times are not in model range
			}
		}
	}
	else {
		err = -1; return err;
	}*/

	Error: // JLM 	 10/27/98
	//if(newMover) {newMover->Dispose();delete newMover;newMover = 0;};
	//return 0;
	return err;

}

OSErr IceWindMover_c::GetIceFields(Seconds model_time, double *ice_fraction, double *ice_thickness)
{
	return dynamic_cast<TimeGridWindIce_c *>(timeGrid)->GetIceFields(model_time, ice_thickness, ice_fraction);
}

// for now just use gridcurrentmovers function, eventually may add a separate ice grid
/*OSErr IceWindMover_c::GetScaledVelocities(Seconds model_time, VelocityFRec *velocities)
{
	return timeGrid->GetScaledVelocities(model_time, velocities);
}*/

OSErr IceWindMover_c::GetIceVelocities(Seconds model_time, VelocityFRec *ice_velocities)
{
	return dynamic_cast<TimeGridWindIce_c *>(timeGrid)->GetIceVelocities(model_time, ice_velocities);
}

OSErr IceWindMover_c::GetMovementVelocities(Seconds model_time, VelocityFRec *velocities)
{	// this function gets velocities based on ice coverage 
	// high coverage use ice_velocity, low coverage use current_velocity, in between interpolate
	return dynamic_cast<TimeGridWindIce_c *>(timeGrid)->GetMovementVelocities(model_time, velocities);
}


