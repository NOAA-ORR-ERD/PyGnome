/*
 *  RiseVelocity_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "RiseVelocity_c.h"
#include "CompFunctions.h"
#include "GEOMETRY.H"
#include "Units.h"

//#ifndef pyGNOME
//#include "TVectMap.h"
//#endif

using std::cout;

RiseVelocity_c::RiseVelocity_c () : Mover_c()
{
	Init();
}
#ifndef pyGNOME
RiseVelocity_c::RiseVelocity_c (TMap *owner, char *name) : Mover_c (owner, name)
{
	Init();
	SetClassName (name);
}
#endif
// Initialize local variables the same way for all constructors
void RiseVelocity_c::Init()
{
	water_density=1020.; 
	water_viscosity = 1.e-6;
}

OSErr RiseVelocity_c::PrepareForModelRun()
{
	//this -> fOptimize.isFirstStep = true;	// may need this, but no uncertainty at this point
	return noErr;
}
OSErr RiseVelocity_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	//this -> fOptimize.isOptimizedForStep = true;
	//this -> fOptimize.value = sqrt(6.*(fDiffusionCoefficient/10000.)*time_step)/METERSPERDEGREELAT; // in deg lat
	//this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*time_step)/METERSPERDEGREELAT; // in deg lat
	return noErr;
}

void RiseVelocity_c::ModelStepIsDone()
{
	//if (this -> fOptimize.isFirstStep == true) this -> fOptimize.isFirstStep = false;
	//memset(&fOptimize,0,sizeof(fOptimize));
}


OSErr RiseVelocity_c::get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* rise_velocity, double* density, long* droplet_size, short* LE_status, LEType spillType, long spill_ID) {	
	
	// JS Ques: Is this required? Could cy/python invoke this method without well defined numpy arrays?
	if(!delta || !ref || !rise_velocity || !density || !droplet_size) {
		//cout << "worldpoints arrays not provided! returning.\n";
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
		rec.riseVelocity = rise_velocity[i];	// define the rise_velocity for the current LE
		rec.density = density[i];	// define the density for the current LE
		rec.dropletSize = droplet_size[i];	// define the droplet_size for the current LE
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	// really only need this for the latitude
		//rec.p.pLong*= 1000000;
		
		delta[i] = this->GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D RiseVelocity_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	//double		dLong, dLat, z, verticalDiffusionCoefficient, mixedLayerDepth=10.;
	WorldPoint3D	deltaPoint = {0,0,0.};
	//WorldPoint refPoint = (*theLE).p;	
	double g = 9.8;
	
	// for now not implementing uncertainty
	
	// do we check if z = 0 here?
	if (isnan((*theLE).riseVelocity)) (*theLE).riseVelocity = (2.*g/9.)*(1.-(*theLE).density/water_density)*((*theLE).dropletSize*1e-6/2.)*((*theLE).dropletSize*1e-6/2)/water_viscosity;	
	deltaPoint.z = -1. * ((*theLE).riseVelocity*CMTOMETERS)*timeStep;	// assuming we add dz to the point, check units...

	
	return deltaPoint;
}

