/*
 *  GridWindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridWindMover_c.h"
#include "CROSS.H"
//#include "netcdf.h"

GridWindMover_c::GridWindMover_c(TMap *owner,char* name) : WindMover_c(owner, name)
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
	//fFillValue = -1e+34;
	
	//fTimeShift = 0; // assume file is in local time
	
	timeGrid = 0;
	//fAllowExtrapolationInTime = false;
	
}


/////////////////////////////////////////////////
OSErr GridWindMover_c::PrepareForModelRun()
{
	return WindMover_c::PrepareForModelRun();
}

OSErr GridWindMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	OSErr err = 0;
	if(uncertain) 
	{
		Seconds elapsed_time = model_time - fModelStartTime;
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}
	
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (!bActive) return noErr;
	
	if (!timeGrid) return -1;
	
	err = timeGrid -> SetInterval(errmsg, model_time); 
	
	if (err) goto done;	// again don't want to have error if outside time interval
	
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


OSErr GridWindMover_c::get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID) {	

	if(!ref || !delta || !windages) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	
	// code goes here, windage...
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

WorldPoint3D GridWindMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double 	dLong, dLat;
	WorldPoint3D	deltaPoint ={0,0,0.};
	WorldPoint3D refPoint;	
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
		err = timeGrid -> SetInterval(errmsg, model_time); // AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	
	refPoint.p = (*theLE).p;	
	refPoint.z = (*theLE).z;
	windVelocity = timeGrid->GetScaledPatValue(model_time, refPoint);

	//windVelocity.u *= fWindScale; 
	//windVelocity.v *= fWindScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		err = AddUncertainty(setIndex,leIndex,&windVelocity);
	}
	
	windVelocity.u *=  (*theLE).windage;
	windVelocity.v *=  (*theLE).windage;
	
	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.p.pLat);
	dLat =   (windVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}



