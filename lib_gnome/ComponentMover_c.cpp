/*
 *  ComponentMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ComponentMover_c.h"
#include "StringFunctions.h"
//#include "OUTILS.H"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

#ifndef pyGNOME	
ComponentMover_c::ComponentMover_c (TMap *owner, char *name) : CurrentMover_c (owner, name)
{
	pattern1 = nil;
	pattern2 = nil;
	bPat1Open = true;
	bPat2Open = true;
	timeFile = nil;
	
	memset(&refP,0,sizeof(refP));
	bRefPointOpen = false;
	pat1Angle = 0;
	pat2Angle = pat1Angle + 90;
	
	pat1Speed = 10;
	pat2Speed = 10;
	
	pat1SpeedUnits = kMetersPerSec;
	pat2SpeedUnits = kMetersPerSec;
	
	pat1ScaleToValue = .1;
	pat2ScaleToValue = .1;
	
	//scaleBy = WINDSTRESS;	// changed default 5/26/00
	scaleBy = NONE;	// default for dialog is WINDSTRESS, but for location files is WINDSPEED 5/29/00
	// both are set when they are first encountered
	memset(&fOptimize,0,sizeof(fOptimize));
	
	timeMoverCode = kLinkToNone;
	windMoverName [0] = 0;
	
	bUseAveragedWinds = false;
	bExtrapolateWinds = false;
	bUseMainDialogScaleFactor = false;
	fScaleFactorAveragedWinds = 1.;
	fPowerFactorAveragedWinds = 1.;
	fPastHoursToAverage = 24;
	fAveragedWindsHdl = 0;
	
	return;
}
#endif
ComponentMover_c::ComponentMover_c () : CurrentMover_c ()
{
	pattern1 = nil;
	pattern2 = nil;
	//bPat1Open = true;
	//bPat2Open = true;
	timeFile = nil;
	
	memset(&refP,0,sizeof(refP));
	//bRefPointOpen = false;
	pat1Angle = 0;
	pat2Angle = pat1Angle + 90;
	
	pat1Speed = 10;
	pat2Speed = 10;
	
	pat1SpeedUnits = kMetersPerSec;
	pat2SpeedUnits = kMetersPerSec;
	
	pat1ScaleToValue = .1;
	pat2ScaleToValue = .1;
	
	//scaleBy = WINDSTRESS;	// changed default 5/26/00
	scaleBy = NONE;	// default for dialog is WINDSTRESS, but for location files is WINDSPEED 5/29/00
	// both are set when they are first encountered
	memset(&fOptimize,0,sizeof(fOptimize));
	
	timeMoverCode = kLinkToNone;
	windMoverName [0] = 0;
	
	bUseAveragedWinds = false;
	bExtrapolateWinds = false;
	bUseMainDialogScaleFactor = false;
	fScaleFactorAveragedWinds = 1.;
	fPowerFactorAveragedWinds = 1.;
	fPastHoursToAverage = 24;
#ifndef pyGNOME
	fAveragedWindsHdl = 0;
#else
	fAveragedWindVelocity.u = fAveragedWindVelocity.v = 0;
	timeMoverCode = kLinkToTimeFile; // maybe just leave as NONE ?
#endif
	
	return;
}

void ComponentMover_c::Dispose ()
{
	if (pattern1) {
		pattern1 -> Dispose();
		delete pattern1;
		pattern1 = nil;
	}
	if (pattern2) {
		pattern2 -> Dispose();
		delete pattern2;
		pattern2 = nil;
	}
	//For pyGnome, let python/cython manage memory for this object.	
#ifndef pyGNOME
	if (timeFile)
	{
		timeFile -> Dispose ();
		delete timeFile;
		timeFile = nil;
	}
	if (fAveragedWindsHdl)
	{
		DisposeHandle((Handle)fAveragedWindsHdl);
		fAveragedWindsHdl = 0;
	}
#endif
	CurrentMover_c::Dispose ();
}

void ComponentMover_c::ModelStepIsDone()
{
	this -> fOptimize.isFirstStep = false;
	memset(&fOptimize,0,sizeof(fOptimize));
	bIsFirstStep = false;
}

OSErr ComponentMover_c::PrepareForModelRun()
{
	this -> fOptimize.isFirstStep = true;
	return CurrentMover_c::PrepareForModelRun();
}

OSErr ComponentMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)

{
	char errmsg[256];
	OSErr err = 0;

	err = CurrentMover_c::PrepareForModelStep(model_time, time_step, uncertain, numLESets, LESetsSizesList); // note: this calls UpdateUncertainty()	// AH 07/10/2012
	
	errmsg[0]=0;
	
	err = SetOptimizeVariables (errmsg, model_time, time_step);
	
	// code goes here, jump to done?
	//if (err) goto done;
	
	this -> fOptimize.isOptimizedForStep = true;
	//this -> fOptimize.isFirstStep = (model_time == start_time);
	
	// code goes here, I think this is redundant
	/*if (this -> fOptimize.isFirstStep)
	{	
		if (bUseAveragedWinds)
		{
			if (fAveragedWindsHdl)
			{	// should dispose at end of run??
				DisposeHandle((Handle)fAveragedWindsHdl);
				fAveragedWindsHdl = 0;
			}
			err = CalculateAveragedWindsHdl(errmsg);
			if (err) {if (!errmsg[0]) strcpy(errmsg,"There is a problem with the averaged winds. Please check your inputs.");}
		}
	}*/
done:
	if (err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TComponentMover::PrepareForModelStep");
		printError(errmsg); 
	}
	return err;
}
#ifdef pyGNOME
OSErr ComponentMover_c::CalculateAveragedWindsVelocity(const Seconds& model_time, char *errmsg)
{
	OSErr err = 0;
	long j;
	VelocityRec avValue;
	VelocityRec wVel = {0.,0.};
	double pat1Theta = PI * -(0.5 + (pat1Angle / 180.0));
	double pat2Theta = PI * -(0.5 + (pat2Angle / 180.0));
	WorldPoint3D refPoint3D = {0,0,0.};
	VelocityRec pat1ValRef;
	double pat1ValRefLength;
	
	refPoint3D.p = refP;
	pat1ValRef = pattern1 -> GetPatValue (refPoint3D);
	pat1ValRefLength = sqrt (pat1ValRef.u * pat1ValRef.u + pat1ValRef.v * pat1ValRef.v);

	strcpy(errmsg,"");
	
	long averageTimeSteps;
	double averageSpeed=0.,averageDir = 0;
	Seconds timeToGetAverageFor = model_time;
	Seconds startPastTime  = timeToGetAverageFor - fPastHoursToAverage * 3600;
	//INDEXH(fAveragedWindsHdl, i).time = model -> GetStartTime () + i * model -> GetTimeStep();
	//averageTimeSteps = fPastHoursToAverage+1; // for now, will change to match model time steps...
	averageTimeSteps = fPastHoursToAverage; // for now, will change to match model time steps...
	// code goes here, may want to change to GetStartTime, GetEndTime, then check out of range
	if (timeFile) timeFile->CheckStartTime(startPastTime);
	else {err = -1; strcpy(errmsg,"There is no wind data to average"); return err;}
	if (err==-1) 
	{
		if (bExtrapolateWinds)
		{	// GetTimeValue() already extrapolates
			err = 0;
		}
		else
		{strcpy(errmsg,"There is not enough data in your wind file for the averaging"); return err;}
		//printError("There is not enough data in your wind file for the averaging"); goto done;
	}
	if (err==-2) 
	{
		//strcpy(errmsg,"No point in averaging for constant wind."); goto done;
		//fPastHoursToAverage=0; err=0;	// allow constant wind, only need one value though 
		//printError("No point in averaging for constant wind."); goto done;
		averageTimeSteps = 1; 
		startPastTime = timeToGetAverageFor;	// this doesn't really matter with constant wind
		err = 0;
	}
	//if (forTime > INDEXH(timeValues, n - 1).time) 
	if (fPastHoursToAverage==0) averageTimeSteps = 1;	// just use the straight wind
	for (j=0;j<averageTimeSteps;j++)
	{
		Seconds timeToAddToAverage = startPastTime + j*3600; // eventually this will be time step...
		double		windSpeedToScale, windDir,theta;
		// get the time file / wind mover value for this time
		// get the mover first then repeat using it for the times..., but make sure get time value gets a value...
		// check first value - 24, last value else will just use first/last value 
		// also check if it's not a time file...
		// check here that time is in the handle...
		timeFile-> GetTimeValue (timeToAddToAverage, &wVel);	
		
		//windSpeedToScale = sqrt(wVel.u*wVel.u + wVel.v*wVel.v);
		// code goes here, take the component first, then average ?
		windSpeedToScale = wVel.u * cos (pat1Theta) + wVel.v * sin (pat1Theta);
		//averageSpeed += (windSpeedToScale) * fScaleFactorAveragedWinds / pat1ValRefLength; //windSpeedToScale; //?? need the dot product too
		//averageSpeed += (windSpeedToScale); //windSpeedToScale; //?? need the dot product too - seems this was done twice?
		//windDir = fmod(atan2(wVel.u,wVel.v)*180/PI+360,360); // floating point remainder
		windDir = atan2(wVel.u,wVel.v); // 
		//windDir = fmod(-180,360); // not sure what above does...
		//theta = fmod(theta+180,360); // rotate the vector cause wind is FROM this direction
		//r=sqrt(u*u+v*v);
		//	windDir = 0;
		averageSpeed = averageSpeed + windSpeedToScale; // need to divide by averageTimeSteps
		averageDir = averageDir + windDir;
		//averageDir = averageDir + windSpeedToScale; // need to divide by averageTimeSteps
		// if add up wind dir make sure it's -180 to 180 - not necessary
	}
	averageSpeed = averageSpeed / averageTimeSteps;
	// apply power and scale - is this the right order?
	if (averageSpeed<0) averageSpeed = -1. * pow(fabs(averageSpeed),fPowerFactorAveragedWinds);
	else
	/*if (fPowerFactorAveragedWinds!=1.)*/  averageSpeed = pow(averageSpeed,fPowerFactorAveragedWinds); 
	//for now apply the scale factor in SetOptimizeVariables()
	//averageSpeed = averageSpeed*fScaleFactorAveragedWinds; 
	// code goes here bUseMainDialogScaleFactor = true do this way leave out fSFAW, = false just use fSFAW
	averageDir = averageDir / averageTimeSteps;
	//avValue.u = averageSpeed*sin(averageDir/*+PI*/);	// not sure about the pi
	//avValue.v = averageSpeed*cos(averageDir/*+PI*/);
	avValue.u = averageSpeed;	// not sure about the pi
	avValue.v = 0;	// not sure about the pi
}
#else
OSErr ComponentMover_c::CalculateAveragedWindsHdl(char *errmsg)
{
	OSErr err = 0;
	long i, j, numTimeSteps = (model -> GetEndTime () - model -> GetStartTime ()) / model -> GetTimeStep() + 1;
	VelocityRec value, avValue;
	TMover 		*mover;
	VelocityRec wVel = {0.,0.};
	Boolean		bFound = false;
	double pat1Theta = PI * -(0.5 + (pat1Angle / 180.0));
	double pat2Theta = PI * -(0.5 + (pat2Angle / 180.0));
	WorldPoint3D refPoint3D = {0,0,0.};
	VelocityRec pat1ValRef;
	double pat1ValRefLength;
	
	refPoint3D.p = refP;
	pat1ValRef = pattern1 -> GetPatValue (refPoint3D);
	pat1ValRefLength = sqrt (pat1ValRef.u * pat1ValRef.u + pat1ValRef.v * pat1ValRef.v);

	strcpy(errmsg,"");
	
	// calculate handle size - number of time steps, end time - start time / time step + 1
	// for each time step, start at 24 hrs (or whatever) before and average wind at each step up to current
	// if no values that far back give an error
	// put time and value in the handle
	// if error delete
	
	// code goes here, might want to extend handle if model run time is changed, or recreate?
	// then should delete handle in case it still exists...
	if (fAveragedWindsHdl)
	{	// should dispose at end of run??
		DisposeHandle((Handle)fAveragedWindsHdl);
		fAveragedWindsHdl = 0;
	}
	fAveragedWindsHdl = (TimeValuePairH)_NewHandleClear(sizeof(TimeValuePair)*numTimeSteps);
	if (!fAveragedWindsHdl)
	{ TechError("TComponentMover::CalculateAveragedWindsHdl()", "_NewHandleClear()", 0); return -1; }
	
	// get the wind mover, if it's constant or nonexistent this is an error, should be checked in the dialog												  
	if (timeMoverCode == kLinkToWindMover)
	{
		long 	/*	i, j,*/ m, n;
		//	double 		length, theirLengthSq, myLengthSq, dotProduct;
		//	VelocityRec theirVelocity,myVelocity;
		//VelocityRec wVel = {0.,0.};
		TMap		*map;
		//	TMover 		*mover;
		//	Boolean		bFound = false;
		
		for (j = 0, m = model -> mapList -> GetItemCount() ; j < m && !bFound ; j++) 
		{
			model -> mapList -> GetListItem((Ptr)&map, j);
			
			for (i = 0, n = map -> moverList -> GetItemCount() ; i < n ; i++) 
			{
				map -> moverList -> GetListItem((Ptr)&mover, i);
				if (mover -> GetClassID() != TYPE_WINDMOVER) continue;
				if (!strcmp(mover -> className, windMoverName)) 
				{
					// JLM, note: we are implicitly matching by file name above
					//	((TWindMover*) mover) -> GetTimeValue (model -> modelTime, &wVel);
					bFound = true;
					break;
				}
			}
			
			if (!bFound)
			{
				map = model -> uMap;
				for (i = 0, n = map -> moverList -> GetItemCount() ; i < n ; i++) 
				{
					map -> moverList -> GetListItem((Ptr)&mover, i);
					if (mover -> GetClassID() != TYPE_WINDMOVER) continue;
					if (!strcmp(mover -> className, windMoverName)) 
					{
						// JLM, note: we are implicitly matching by file name above
						//((TWindMover*) mover) -> GetTimeValue (model -> modelTime, &wVel);
						bFound = true;
						break;
					}
				}
			}
		}
		
		if (!bFound)
		{
			mover =  (TMover*) (model->GetWindMover(false));	// see if there is a wind at all and use it
			if (!mover) {
				strcpy(windMoverName, "");	// clear out any value
				strcpy(errmsg,"There is no wind time file associated with the component mover");
				//printError("There is no wind time file associated with the component mover");
				return -1;
			}
			strcpy(windMoverName, mover->className);	// link the wind to the component mover
			//print error, also check if it's a constant wind or times out of range
		}
		// alert code goes here if mover is not found
	}
	else
	{
		strcpy(errmsg,"There is no wind time file associated with the component mover");
		//	printError("There is no wind time file associated with the component mover");
		return -1;
	}
	
	// check wind time file exists for whole length of run, including the past averaging hours
	for (i = 0 ; i < numTimeSteps ; i++) 
	{
		long averageTimeSteps;
		double averageSpeed=0.,averageDir = 0;
		Seconds timeToGetAverageFor = model -> GetStartTime () + i * model -> GetTimeStep();
		Seconds startPastTime  = timeToGetAverageFor - fPastHoursToAverage * 3600;
		INDEXH(fAveragedWindsHdl, i).time = model -> GetStartTime () + i * model -> GetTimeStep();
		//averageTimeSteps = fPastHoursToAverage+1; // for now, will change to match model time steps...
		averageTimeSteps = fPastHoursToAverage; // for now, will change to match model time steps...
		// code goes here, may want to change to GetStartTime, GetEndTime, then check out of range
		if (i==0) 	err = dynamic_cast<TWindMover *>(mover)->CheckStartTime(startPastTime); //if (forTime < INDEXH(timeValues, 0).time) 
		if (err==-1) 
		{
			if (bExtrapolateWinds)
			{	// GetTimeValue() already extrapolates
				err = 0;
			}
			else
			{strcpy(errmsg,"There is not enough data in your wind file for the averaging"); goto done;}
			//printError("There is not enough data in your wind file for the averaging"); goto done;
		}
		if (err==-2) 
		{
			//strcpy(errmsg,"No point in averaging for constant wind."); goto done;
			//fPastHoursToAverage=0; err=0;	// allow constant wind, only need one value though 
			//printError("No point in averaging for constant wind."); goto done;
			averageTimeSteps = 1; 
			startPastTime = timeToGetAverageFor;	// this doesn't really matter with constant wind
			err = 0;
		}
		//if (forTime > INDEXH(timeValues, n - 1).time) 
		
		if (fPastHoursToAverage==0) averageTimeSteps = 1;	// just use the straight wind
		for (j=0;j<averageTimeSteps;j++)
		{
			Seconds timeToAddToAverage = startPastTime + j*3600; // eventually this will be time step...
			double		windSpeedToScale, windDir,theta;
			// get the time file / wind mover value for this time
			// get the mover first then repeat using it for the times..., but make sure get time value gets a value...
			// check first value - 24, last value else will just use first/last value 
			// also check if it's not a time file...
			// make sure in the GetMove to GetTimeValue from the averaged handle
			// check here that time is in the handle...
 			dynamic_cast<TWindMover *>(mover)-> GetTimeValue (timeToAddToAverage, &wVel);	// minus AH 07/10/2012
			
			//windSpeedToScale = sqrt(wVel.u*wVel.u + wVel.v*wVel.v);
			// code goes here, take the component first, then average ?
			windSpeedToScale = wVel.u * cos (pat1Theta) + wVel.v * sin (pat1Theta);
			//averageSpeed += (windSpeedToScale) * fScaleFactorAveragedWinds / pat1ValRefLength; //windSpeedToScale; //?? need the dot product too
			//averageSpeed += (windSpeedToScale); //windSpeedToScale; //?? need the dot product too - seems this was done twice?
			//windDir = fmod(atan2(wVel.u,wVel.v)*180/PI+360,360); // floating point remainder
			windDir = atan2(wVel.u,wVel.v); // 
			//windDir = fmod(-180,360); // not sure what above does...
			//theta = fmod(theta+180,360); // rotate the vector cause wind is FROM this direction
			//r=sqrt(u*u+v*v);
			//	windDir = 0;
			averageSpeed = averageSpeed + windSpeedToScale; // need to divide by averageTimeSteps
			averageDir = averageDir + windDir;
			//averageDir = averageDir + windSpeedToScale; // need to divide by averageTimeSteps
			// if add up wind dir make sure it's -180 to 180 - not necessary
		}
		averageSpeed = averageSpeed / averageTimeSteps;
		// apply power and scale - is this the right order?
		if (averageSpeed<0) averageSpeed = -1. * pow(myfabs(averageSpeed),fPowerFactorAveragedWinds);
		else
		/*if (fPowerFactorAveragedWinds!=1.)*/  averageSpeed = pow(averageSpeed,fPowerFactorAveragedWinds); 
		//for now apply the scale factor in SetOptimizeVariables()
		//averageSpeed = averageSpeed*fScaleFactorAveragedWinds; 
		// code goes here bUseMainDialogScaleFactor = true do this way leave out fSFAW, = false just use fSFAW
		averageDir = averageDir / averageTimeSteps;
		//avValue.u = averageSpeed*sin(averageDir/*+PI*/);	// not sure about the pi
		//avValue.v = averageSpeed*cos(averageDir/*+PI*/);
		avValue.u = averageSpeed;	// not sure about the pi
		avValue.v = 0;	// not sure about the pi
		INDEXH(fAveragedWindsHdl, i).value = avValue;// translate back to u,v
		
	}
done:
	if (err)
	{
		if (fAveragedWindsHdl)
		{	// should dispose at end of run??
			DisposeHandle((Handle)fAveragedWindsHdl);
			fAveragedWindsHdl = 0;
		}
	}
	return err;
}
#endif

void ComponentMover_c::SetTimeFile(TOSSMTimeValue *newTimeFile)
{
	timeFile = newTimeFile;
}

OSErr ComponentMover_c::SetOptimizeVariables (char *errmsg, const Seconds& model_time, const Seconds& time_step)
{
	VelocityRec	vVel, hVel, wVel = {0.,0.};
	OSErr		err = noErr;
	Boolean 	useEddyUncertainty = false;	
	VelocityRec	ref1Wind, ref2Wind,pat1ValRef, pat2ValRef;
	double		dot1,dot2;
	double		pat1SpeedMPS,pat2SpeedMPS,pat1Theta,pat2Theta;
	double		pat1ValRefLength,pat2ValRefLength;
	double		windSpeedToScale;
	WorldPoint3D refPoint3D = {0,0,0.};
	
	strcpy(errmsg,"");
	
	wVel.u = wVel.v = 0;
	refPoint3D.p = refP;
	
	// get the time file / wind mover value for this time
	
#ifndef pyGNOME
	if (timeMoverCode == kLinkToWindMover)
	{
		long 		i, j, m, n;
		double 		length, theirLengthSq, myLengthSq, dotProduct;
		VelocityRec theirVelocity,myVelocity;
		Boolean		bFound = false;
		
		if (bUseAveragedWinds)
		{
			// routine to get the time value from the averaged winds handle
			if (fAveragedWindsHdl)
			{	// 
				//get averaged value from the handle, figure out the time step index
				//	long index = (long) ((model -> GetModelTime() - model->GetStartTime()) / model->GetTimeStep())
				//	wvel = INDEXH(fAveragedWindsHdl,index); // check index is in range
				err = GetAveragedWindValue(model_time, time_step, &wVel);
				if (err) 
				{
					err = CalculateAveragedWindsHdl(errmsg);
					if (err) 
					{
						if (!errmsg[0]) {strcpy(errmsg,"There is a problem with the averaged winds. Please check your inputs.");} return -1;
						//printError("There is a problem with the averaged winds. Please check your inputs."); return -1;
					}
					else 
					{
						err = GetAveragedWindValue(model_time, time_step, &wVel);
						if (err) 
						{
							if (!errmsg[0]) {strcpy(errmsg,"There is a problem with the averaged winds. Please check your inputs.");} return -1;
							//printError("There is a problem with the averaged winds. Please check your inputs."); return -1;
						}
					}
				}
			}
			else
			{
				// try to calculate
				//	printError("No averaged winds handle"); // hmm, can't return an error...
				err = CalculateAveragedWindsHdl(errmsg);
				if (err) 
				{
					//printError("There is a problem with the averaged winds. Please check your inputs."); return -1;
					if (!errmsg[0]) {strcpy(errmsg,"There is a problem with the averaged winds. Please check your inputs.");} return -1;
				}
				else 
				{
					err = GetAveragedWindValue(model_time, time_step, &wVel);
					if (err)
					{ 
						if (!errmsg[0]) {strcpy(errmsg,"There is a problem with the averaged winds. Please check your inputs.");} return -1;
						//printError("There is a problem with the averaged winds. Please check your inputs."); return -1;
					}
				}
			}
			//return;
		}
		else
		{
			TMap	*map = 0;
			TMover 	*mover = 0;
			for (j = 0, m = model -> mapList -> GetItemCount() ; j < m && !bFound ; j++) {
				model -> mapList -> GetListItem((Ptr)&map, j);
				
				for (i = 0, n = map -> moverList -> GetItemCount() ; i < n ; i++) {
					map -> moverList -> GetListItem((Ptr)&mover, i);
					if (mover -> GetClassID() != TYPE_WINDMOVER) continue;
					if (!strcmp(mover -> className, windMoverName)) {
						// JLM, note: we are implicitly matching by file name above
						dynamic_cast<TWindMover *>(mover)-> GetTimeValue (model_time, &wVel);	// minus AH 07/10/2012
						bFound = true;
						break;
					}
				}
				
				if (!bFound)
				{
					map = model -> uMap;
					for (i = 0, n = map -> moverList -> GetItemCount() ; i < n ; i++) {
						map -> moverList -> GetListItem((Ptr)&mover, i);
						if (mover -> GetClassID() != TYPE_WINDMOVER) continue;
						if (!strcmp(mover -> className, windMoverName)) {
							// JLM, note: we are implicitly matching by file name above
							dynamic_cast<TWindMover *>(mover)-> GetTimeValue (model_time, &wVel);	// minus AH 07/10/2012
							
							bFound = true;
							break;
						}
					}
				}

			}
			if (!bFound)
			{
				mover =  (TMover*) (model->GetWindMover(false));	// see if there is a wind at all and use it
				/*if (!mover) {
					strcpy(errmsg,"There is no wind time file associated with the component mover");
					//printError("There is no wind time file associated with the component mover");
					return -1;
				}*/
				if (mover) 
				{
					strcpy(windMoverName, mover->className);	// link the wind to the component mover
					dynamic_cast<TWindMover *>(mover)-> GetTimeValue (model_time, &wVel);	// minus AH 07/10/2012
				}
				else strcpy(windMoverName, "");	// clear out any value

			}
			// alert code goes here if mover is not found
		}
	}
#else
	//else {
	// use timeFile (for python do it this way)
		if (timeFile)
		{
			if (bUseAveragedWinds)
			{
				err = CalculateAveragedWindsVelocity(model_time,errmsg);
				wVel = fAveragedWindVelocity;	// don't need to store?
			}
			else
				timeFile->GetTimeValue(model_time, &wVel);	// this needs to be defined
		}	
	//}
#endif	
	// code goes here, option for averaged winds to set a scale or use the main dialog scale, would pat1ValScale/pat2ValScale just be averaged wind value? 
	
	//if (bUseAveragedWinds && bUseMainDialogScaleFactor)
	windSpeedToScale = sqrt(wVel.u*wVel.u + wVel.v*wVel.v);
	
	pat1SpeedMPS = pat1Speed * speedconversion (pat1SpeedUnits);
	pat2SpeedMPS = pat2Speed * speedconversion (pat2SpeedUnits);
	
	pat1Theta = PI * -(0.5 + (pat1Angle / 180.0));
	pat2Theta = PI * -(0.5 + (pat2Angle / 180.0));
	
	ref1Wind.u = cos (pat1Theta) * pat1SpeedMPS;
	ref1Wind.v = sin (pat1Theta) * pat1SpeedMPS;
	ref2Wind.u = cos (pat2Theta) * pat2SpeedMPS;
	ref2Wind.v = sin (pat2Theta) * pat2SpeedMPS;
	
	dot1 = wVel.u*ref1Wind.u + wVel.v*ref1Wind.v;
	dot2 = wVel.u*ref2Wind.u + wVel.v*ref2Wind.v;
	
	pat1ValRef = pattern1 -> GetPatValue (refPoint3D);
	pat1ValRefLength = sqrt (pat1ValRef.u * pat1ValRef.u + pat1ValRef.v * pat1ValRef.v);
	
	// code goes here, some restriction on when WINDSTRESS scaling can be used?
	if (!bUseAveragedWinds)	// usemaindialogscalefactor different...
	{	
		if (scaleBy == WINDSPEED)
			fOptimize.pat1ValScale = dot1 * pat1ScaleToValue / (pat1SpeedMPS * pat1SpeedMPS * pat1ValRefLength);
		else // scaleBy == WINDSTRESS
			fOptimize.pat1ValScale = dot1 * pat1ScaleToValue * windSpeedToScale / (pat1SpeedMPS * pat1SpeedMPS * pat1SpeedMPS * pat1ValRefLength);
	}
	else
		// I think the value needs to be normalized, then match Matt's windage with a scale factor - 3,4 or whatever windage is (not .03)
		// at this point the normalization means the same scale factor is used (I think Matt is dividing by ref val rather than multiplying...)
		//fOptimize.pat1ValScale = (wVel.u * cos (pat1Theta) + wVel.v * sin (pat1Theta)) * fScaleFactorAveragedWinds; //windSpeedToScale; //?? need the dot product too
	{		//fOptimize.pat1ValScale = (wVel.u * cos (pat1Theta) + wVel.v * sin (pat1Theta)) * fScaleFactorAveragedWinds / pat1ValRefLength; //windSpeedToScale; //?? need the dot product too
		
		if (bUseMainDialogScaleFactor)	// wind speed/ wind stress issue?
			//fOptimize.pat1ValScale = wVel.u * pat1ScaleToValue / (pat1SpeedMPS * pat1SpeedMPS * pat1ValRefLength); //stored component in average handle
			fOptimize.pat1ValScale = wVel.u * pat1ScaleToValue / (pat1SpeedMPS * pat1ValRefLength); //stored component in average handle, don't multiply by pat1SpeedMPS in Average Wind calc
		else
			//fOptimize.pat1ValScale = wVel.u * fScaleFactorAveragedWinds / pat1ValRefLength; //stored component in average handle
			fOptimize.pat1ValScale = wVel.u * fScaleFactorAveragedWinds; // Bushy says just multiply by the factor, don't use any reference point stuff
		//fOptimize.pat1ValScale = wVel.u * fScaleFactorAveragedWinds * pat1SpeedMPS / pat1ValRefLength; //stored component in average handle
	}
	if (pattern2) 
	{
		pat2ValRef = pattern2 -> GetPatValue (refPoint3D);
		pat2ValRefLength = sqrt (pat2ValRef.u * pat2ValRef.u + pat2ValRef.v * pat2ValRef.v);
		if (!bUseAveragedWinds || (bUseAveragedWinds && bUseMainDialogScaleFactor))	// averaged winds shouldn't be an issue here
		{
			if (scaleBy == WINDSPEED)
				fOptimize.pat2ValScale = dot2 * pat2ScaleToValue / (pat2SpeedMPS * pat2SpeedMPS * pat2ValRefLength);
			else // scaleBy == WINDSTRESS
				fOptimize.pat2ValScale = dot2 * pat2ScaleToValue * windSpeedToScale / (pat2SpeedMPS * pat2SpeedMPS * pat2SpeedMPS * pat2ValRefLength);
		}
		else	// assuming there is no pat2 for averaged winds option...
			fOptimize.pat2ValScale = (wVel.u * cos (pat2Theta) + wVel.v * sin (pat2Theta)) * fScaleFactorAveragedWinds; //windSpeedToScale; //?? need the dot product too
		
	}
	else fOptimize.pat2ValScale = 0;
	
	return noErr;
}

OSErr ComponentMover_c::get_move(int n, Seconds model_time, Seconds step_len,
							WorldPoint3D *ref, WorldPoint3D *delta, short *LE_status,
							LEType spillType, long spill_ID)
{
	if(!delta || !ref) {
		return 1;
	}
	
	// For LEType spillType, check to make sure it is within the valid values
	if (spillType < FORECAST_LE ||
		spillType > UNCERTAINTY_LE)
	{
		return 2;
	}
	
	LERec* prec;
	LERec rec;
	prec = &rec;
	
	WorldPoint3D zero_delta = { {0, 0}, 0.};
	
	for (int i = 0; i < n; i++) {
		if ( LE_status[i] != OILSTAT_INWATER) {
			delta[i] = zero_delta;
			continue;
		}
		
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1e6;
		rec.p.pLong *= 1e6;
		
		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1e6;
		delta[i].p.pLong /= 1e6;
	}
	
	return noErr;
}

WorldPoint3D ComponentMover_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double 		dLat, dLong;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	WorldPoint3D	refPoint3D = {0,0,0.};
	VelocityRec	finalVel, pat1Val,pat2Val;
	OSErr err = 0;
	char errmsg[256];
	
	refPoint3D.p = refPoint;
	pat1Val = pattern1 -> GetPatValue (refPoint3D);
	if (pattern2) pat2Val = pattern2 -> GetPatValue (refPoint3D);
	else {pat2Val.u = pat2Val.v = 0;}
	
	if (!fOptimize.isOptimizedForStep)
	{
		err = SetOptimizeVariables (errmsg, model_time, timeStep);
		if (err) return deltaPoint;
	}
	
	
	finalVel.u = pat1Val.u * fOptimize.pat1ValScale + pat2Val.u * fOptimize.pat2ValScale;
	finalVel.v = pat1Val.v * fOptimize.pat1ValScale + pat2Val.v * fOptimize.pat2ValScale;
	
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&finalVel,timeStep);
	}
	
	dLong = ((finalVel.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat =   (finalVel.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

Boolean ComponentMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char str1[32], str2[32], str3[32];
	double length1=0., length2=0., length3 = 0.;
	VelocityRec	pat1Val={0.,0.}, pat2Val={0.,0.}, finalVel;
	OSErr err = 0;
	char errmsg[256];
	
	if (this -> pattern1) {
		pat1Val = this -> pattern1 -> GetPatValue (wp);
		length1 = sqrt(pat1Val.u * pat1Val.u + pat1Val.v * pat1Val.v);
		StringWithoutTrailingZeros(str1,length1,6);
	}
	if (this -> pattern2) {
		pat2Val = this -> pattern2 -> GetPatValue (wp);
		length2 = sqrt(pat2Val.u * pat2Val.u + pat2Val.v * pat2Val.v);
		StringWithoutTrailingZeros(str2,length2,6);
	}
#ifndef pyGNOME	// for now this is not used in pyGNOME, will deal with it later if it becomes an issue
	if (!(this->fOptimize.isOptimizedForStep))
	{
		err = this->SetOptimizeVariables (errmsg, model->GetModelTime(), model->GetTimeStep());
		if (err) return false;
	}
#endif	
	finalVel.u = pat1Val.u * this->fOptimize.pat1ValScale + pat2Val.u * this->fOptimize.pat2ValScale;
	finalVel.v = pat1Val.v * this->fOptimize.pat1ValScale + pat2Val.v * this->fOptimize.pat2ValScale;
	length3 = sqrt(finalVel.u * finalVel.u + finalVel.v * finalVel.v);
	
	StringWithoutTrailingZeros(str1,length1 * this->fOptimize.pat1ValScale,4);
	StringWithoutTrailingZeros(str2,length2 * this->fOptimize.pat2ValScale,4);
	StringWithoutTrailingZeros(str3,length3,4);
	sprintf(diagnosticStr, " [grid: %s, pat1: %s m/s, pat2: %s m/s, total: %s m/s]",
			"component", str1, str2, str3);
	return true;
	
}

OSErr ComponentMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep)
{
	
	double u,v,lengthS,alpha,beta;
	LEUncertainRec unrec;
	
	OSErr err = 0;
	
	//err = this -> UpdateUncertainty();
	//if(err) return err;

	if(!fUncertaintyListH || !fLESetSizesH) 
		return 0; // this is our clue to not add uncertainty
	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(patVelocity->u*patVelocity->u + patVelocity->v * patVelocity->v);
		
		u = patVelocity->u;
		v = patVelocity->v;

		if(lengthS>1e-6) // so we don't divide by zero
		{	
			
			alpha = unrec.downStream;
			beta = unrec.crossStream;
			
			patVelocity->u = u*(1+alpha)+v*beta;
			patVelocity->v = v*(1+alpha)-u*beta;	
		}
	}
	else 
	{
		TechError("TComponentMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		patVelocity->u=patVelocity->v=0;
	}
	return err;
}

#ifndef pyGNOME
OSErr ComponentMover_c::GetAveragedWindValue(Seconds time, const Seconds& time_step, VelocityRec *avValue)
{
	long index, numValuesInHdl;
	VelocityRec avWindValue = {0.,0.};
	Seconds avTime;
	
	*avValue = avWindValue;
	
	index = (long)((time - fModelStartTime)/time_step);
	numValuesInHdl = _GetHandleSize((Handle)fAveragedWindsHdl)/sizeof(**fAveragedWindsHdl);
	if (index<0 || index >= numValuesInHdl) {return -1;}	// may want to recalculate
	avTime = INDEXH(fAveragedWindsHdl, index).time;
	if (avTime != time) return -1;
	*avValue = INDEXH(fAveragedWindsHdl, index).value;// translate back to u,v

	return noErr;
}
#endif
#ifdef pyGNOME
OSErr ComponentMover_c::TextRead(char *cats_path1, char *cats_path2) 
{
	OSErr err = 0;
	TTriGridVel *triGrid = 0;
	
	TCATSMover *newCATSMover1 = 0;
	TCATSMover *newCATSMover2 = 0;
	
	if (!cats_path1[0]) return -1;
	
	newCATSMover1 = new TCATSMover;
	if (!newCATSMover1)
	{ 
		TechError("CreateAndInitCurrentsMover()", "new TCATSMover()", 0);
		return -1;
	}
	err = newCATSMover1->TextRead(cats_path1);
	if (!err) pattern1 = newCATSMover1;

	if (cats_path2[0]) 
	{
		newCATSMover2 = new TCATSMover;
		if (!newCATSMover2)
		{ 
			TechError("CreateAndInitCurrentsMover()", "new TCATSMover()", 0);
			return -1;
		}
		err = newCATSMover2->TextRead(cats_path2);
		if (!err) pattern2 = newCATSMover2;
	}
	
	return err;	
}
#endif

