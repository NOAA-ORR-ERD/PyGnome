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

using namespace std;

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
	water_density = 1020.;
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


OSErr RiseVelocity_c::get_move(int n, unsigned long model_time, unsigned long step_len,
							   WorldPoint3D *ref, WorldPoint3D *delta,
							   double *rise_velocity, double *density, double *droplet_size,
							   short *LE_status, LEType spillType, long spill_ID)
{
	// JS Ques: Is this required? Could cy/python invoke this method without well defined numpy arrays?
	if (!delta || !ref || !rise_velocity || !density || !droplet_size) {
		cerr << "(delta, ref, rise_velocity, density, droplet_size) = ("
			 << delta << "," << ref << "," << rise_velocity << ","
			 << density << "," << droplet_size << ")" << endl;
		return 1;
	}

	// For LEType spillType, check to make sure it is within the valid values
	if (spillType < FORECAST_LE || spillType > UNCERTAINTY_LE) {
		cerr << "Invalid spillType." << endl;
		return 2;
	}

	LERec rec;
	WorldPoint3D zero_delta = { {0, 0}, 0.};

	for (int i = 0; i < n; i++) {
		if (LE_status[i] != OILSTAT_INWATER) {
			delta[i] = zero_delta;
			continue;
		}

		rec.p = ref[i].p;
		rec.z = ref[i].z;

		rec.riseVelocity = rise_velocity[i];
		rec.density = density[i];
		rec.dropletSize = (long)droplet_size[i];	// code goes here, droplet size long or double?

		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1e6;	// really only need this for the latitude
		//rec.p.pLong*= 1000000;

		delta[i] = this->GetMove(model_time, step_len, spill_ID, i, &rec, spillType);

		delta[i].p.pLat /= 1e6;
		delta[i].p.pLong /= 1e6;
	}

	return noErr;
}


WorldPoint3D RiseVelocity_c::GetMove(const Seconds &model_time, Seconds timeStep,
									 long setIndex, long leIndex, LERec *theLE, LETYPE leType)
{
	WorldPoint3D deltaPoint = { {0, 0}, 0.};
	double g = 9.8;
	//double dLong, dLat, z, verticalDiffusionCoefficient, mixedLayerDepth = 10.;
	//WorldPoint refPoint = (*theLE).p;	
	Boolean useGaltAlgorithm = false;
	// for now not implementing uncertainty


	//code goes here, move rise velocity calculation outside mover
	if (!isnan((*theLE).riseVelocity)){deltaPoint.z = -1. * ((*theLE).riseVelocity)*timeStep; return deltaPoint;}
		
	if (useGaltAlgorithm)
	{
		double dfac = g*(water_density-(*theLE).density*1000) / water_density;	// buoyancy term
		double d1 = .002, d2 = .01;	// upper limit for Stoke's law cut; lower limit for Form drag
		water_viscosity = 1.3e-6;	// water viscosity in mks units (for temp = 10 deg C)
		double scaled_dropletSize = ((*theLE).dropletSize - d1) / (d2 - d1);	// droplet diameter scaled to (d2-d1) interval
		double a1, a2, a3, a4, c0, c1, c2, c3;
		
		if (scaled_dropletSize < 0.)
		{
			double ds = scaled_dropletSize * (d2 -d1) + d1;
			(*theLE).riseVelocity = dfac*ds*ds / (18.*water_viscosity);
		}
		if (scaled_dropletSize >= 1.0)
		{
			double df = scaled_dropletSize * (d2 -d1) + d1;
			(*theLE).riseVelocity = sqrt(8.0*dfac*df/3.0);
		}
		a1 = dfac*d1*d1 / (18.*water_viscosity);
		a2 = a1*(d2-d1) / (2.0*d1);
		a3 = sqrt(8.0*dfac*d2/3.0);
		a4 = a3*(d2-d1) / (2.0*d2);
		c3 = 2.0*a1 + a2 - 2.0*a3 + a4;
		c2 = -3.0*a1 -2.0*a2 + 3.0*a3 - a4;
		c1 = a2;
		c0 = a1;
		
		(*theLE).riseVelocity = ((c3*scaled_dropletSize + c2)*scaled_dropletSize + c1)*scaled_dropletSize + c0;
									
		
	}
	else 
	{

		// do we check if z = 0 here?
		//if (isnan((*theLE).riseVelocity)) (*theLE).riseVelocity = (2.*g/9.)*(1.-(*theLE).density/water_density)*((*theLE).dropletSize*1e-6/2.)*((*theLE).dropletSize*1e-6/2)/water_viscosity;	
		if (isnan((*theLE).riseVelocity)) {
			(*theLE).riseVelocity = (2. * g / 9.) *
									(1. - (*theLE).density * 1000 / water_density) *
									((*theLE).dropletSize * 1e-6 / 2.) *
									((*theLE).dropletSize * 1e-6 / 2) /
									water_viscosity;
		}
	}
	//deltaPoint.z = -1. * ((*theLE).riseVelocity*CMTOMETERS)*timeStep;	// assuming we add dz to the point, check units...
	deltaPoint.z = -1. * ((*theLE).riseVelocity)*timeStep;	// assuming we add dz to the point, check units...

	return deltaPoint;
}

