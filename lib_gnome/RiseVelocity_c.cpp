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
#include "Replacements.h"

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
	//water_density = 1020.; // Oceanic; 1010 - Estuary; 1000 - Fresh, units  kg / m^3
	//water_viscosity = 1.e-6;	// Ns/m^2
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

OSErr get_rise_velocity(int n, double *rise_velocity, double *le_density, double *le_droplet_size, double water_viscosity, double water_density)
{
	for (int i=0; i < n; i++)
	{
		rise_velocity[i] = GetRiseVelocity(le_density[i], le_droplet_size[i], water_viscosity, water_density);
	}
	return noErr;
}

double GetRiseVelocity(double le_density, double le_droplet_size, double water_viscosity, double water_density)
{
	double g = 9.8, riseVelocity=0.;
	
	double dfac = g*(water_density-le_density) / water_density;	// buoyancy term
	double d1 = .0002, d2 = .01;	// upper limit for Stoke's law cut; lower limit for Form drag
	//water_viscosity = 1.3e-6;	// water viscosity in mks units (for temp = 10 deg C)
	double scaled_dropletSize = (le_droplet_size - d1) / (d2 - d1);	// droplet diameter scaled to (d2-d1) interval
	double a1, a2, a3, a4, c0, c1, c2, c3;
	
	// check inputs are ok? (water_density>le_density...)
	if (scaled_dropletSize < 0.)	// special case where d is in Stoke's drift range
	{
		double ds = scaled_dropletSize * (d2 - d1) + d1;
		riseVelocity = dfac*ds*ds / (18.*water_viscosity);
	}
	else if (scaled_dropletSize >= 1.0)	// special case where d is in form drag range
	{
		double df = scaled_dropletSize * (d2 - d1) + d1;
		riseVelocity = sqrt(8.0*dfac*df/3.0);
	}
	else 
	{
		a1 = dfac*d1*d1 / (18.*water_viscosity);
		a2 = a1*(d2-d1) / (2.0*d1);
		a3 = sqrt(8.0*dfac*d2/3.0);
		a4 = a3*(d2-d1) / (2.0*d2);
		c3 = 2.0*a1 + a2 - 2.0*a3 + a4;
		c2 = -3.0*a1 -2.0*a2 + 3.0*a3 - a4;
		c1 = a2;
		c0 = a1;
		
		riseVelocity = ((c3*scaled_dropletSize + c2)*scaled_dropletSize + c1)*scaled_dropletSize + c0;
	}

	return riseVelocity;
}

double GetRiseVelocityDerivative(double oil_density, double water_viscosity, double water_density, double droplet_size)
{
	double g = 9.8, dfac = g * (water_density-oil_density) / water_density;	// buoyancy term
	double a1, a2, a3, a4, c1, c2, c3;
	double d1 = .0002, d2 = .01;	// upper limit for Stoke's law cut; lower limit for Form drag
	
	if (droplet_size<0.)	// special case where d is in stokes drift range
	{
		double ds = droplet_size * (d2-d1) + d1;
		return 2.0*dfac*ds / (18.*water_viscosity);
	}
	if (droplet_size>=1.0)	// special case where d is in form drag range
	{
		double df = droplet_size * (d2-d1) + d1;
		return sqrt(8.0*dfac*df / 3.0) / (2.0*df);
	}
	a1 = dfac*d1*d1 / (18.*water_viscosity);
	a2 = a1*(d2-d1) / (2.0*d1);
	a3 = sqrt(8.0*dfac*d2 / 3.0);
	a4 = a3 * (d2-d1) / (2.0*d2);
	c3 = 2.0 * a1 + a2 - 2.0*a3 + a4;
	c2 = -3.0*a1 - 2.0*a2 + 3.0*a3 - a4;
	c1 = a2;
	
	return (3.0*c3*droplet_size + 2.0*c2)*droplet_size + c1;
	
}

double GetDropletSize(double riseVelocity, double water_viscosity, double water_density, double oil_density)
{
	double y0 = .5, y1 = 1.0, chk = 1.0;
	int i=0;
	
	while(chk>.00005)
	{
		y1 = y0 + (riseVelocity - GetRiseVelocity(oil_density, water_viscosity, water_density, y0)) 
					/ GetRiseVelocityDerivative(oil_density, water_viscosity, water_density, y0);
		chk = fabs(y1-y0);
		i++;
		if (i>25) break;
	}
	return y1;
}

OSErr RiseVelocity_c::get_move(int n, unsigned long model_time, unsigned long step_len,
							   WorldPoint3D *ref, WorldPoint3D *delta,
							   double *rise_velocity,
							   short *LE_status, LEType spillType, long spill_ID)
{
	// JS Ques: Is this required? Could cy/python invoke this method without well defined numpy arrays?
	if (!delta || !ref || !rise_velocity) {
		cerr << "(delta, ref, rise_velocity) = ("
			 << delta << "," << ref << "," << rise_velocity << ")" << endl;
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

		// let's do the multiply by 1000000 here - this is what gnome expects
		//rec.p.pLat *= 1e6;	// not using the lat, lon at this point
		//rec.p.pLong*= 1000000;

		delta[i] = this->GetMove(model_time, step_len, spill_ID, i, &rec, spillType);

		//delta[i].p.pLat /= 1e6;
		//delta[i].p.pLong /= 1e6;
	}

	return noErr;
}


WorldPoint3D RiseVelocity_c::GetMove(const Seconds &model_time, Seconds timeStep,
									 long setIndex, long leIndex, LERec *theLE, LETYPE leType)
{
	WorldPoint3D deltaPoint = { {0, 0}, 0.};
	// for now not implementing uncertainty

	deltaPoint.z = -1. * ((*theLE).riseVelocity)*timeStep;	// assuming we add dz to the point, check units (assuming m/s)

	return deltaPoint;
}

