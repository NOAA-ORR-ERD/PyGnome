/*
 *  RandomVertical_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "RandomVertical_c.h"
#include "CompFunctions.h"
#include "GEOMETRY.H"
#include "Units.h"

#ifndef pyGNOME
#include "TVectMap.h"
#endif

using std::cout;

RandomVertical_c::RandomVertical_c () : Mover_c()
{
	Init();
}
#ifndef pyGNOME
RandomVertical_c::RandomVertical_c (TMap *owner, char *name) : Mover_c (owner, name)
{
	Init();
	SetClassName (name);
}
#endif
// Initialize local variables the same way for all constructors
void RandomVertical_c::Init()
{
	fVerticalDiffusionCoefficient = 5; //  cm**2/sec	
	fVerticalBottomDiffusionCoefficient = .11; //  cm**2/sec, Bushy suggested a larger default	
	//fHorizontalDiffusionCoefficient = 126; //  cm**2/sec	
	bUseDepthDependentDiffusion = false;
	//memset(&fOptimize,0,sizeof(fOptimize));
}

OSErr RandomVertical_c::PrepareForModelRun()
{
	//this -> fOptimize.isFirstStep = true;	// may need this, but no uncertainty at this point
	return noErr;
}
OSErr RandomVertical_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	//this -> fOptimize.isOptimizedForStep = true;
	//this -> fOptimize.value = sqrt(6.*(fDiffusionCoefficient/10000.)*time_step)/METERSPERDEGREELAT; // in deg lat
	//this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*time_step)/METERSPERDEGREELAT; // in deg lat
	return noErr;
}

void RandomVertical_c::ModelStepIsDone()
{
	//if (this -> fOptimize.isFirstStep == true) this -> fOptimize.isFirstStep = false;
	//memset(&fOptimize,0,sizeof(fOptimize));
}


OSErr RandomVertical_c::get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID) {	
	
	// JS Ques: Is this required? Could cy/python invoke this method without well defined numpy arrays?
	if(!delta || !ref) {
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
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	// really only need this for the latitude
		//rec.p.pLong*= 1000000;
		
		delta[i] = this->GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D RandomVertical_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double		dLong, dLat, z, verticalDiffusionCoefficient, mixedLayerDepth=10.;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand1,rand2, r, w;
	//double 	diffusionCoefficient;
	
	//if (deltaPoint.z == 0) return deltaPoint;	// allow for surface LEs ?
	
	// for now not implementing uncertainty
	
	/*if ((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT || (*theLE).z>0)	// only apply vertical diffusion if there are particles below surface
	{
		double g = 9.8, buoyancy = 0.;
		double horizontalDiffusionCoefficient, verticalDiffusionCoefficient;
		double mixedLayerDepth=10., totalLEDepth, breakingWaveHeight=1., depthAtPoint=INFINITE_DEPTH;
		double karmen = .4, rho_a = 1.29, rho_w = 1030., dragCoeff, tau, uStar;
		float water_density=1020.,water_viscosity = 1.e-6,eps = 1.e-6;
		TWindMover *wind = model -> GetWindMover(false);
		Boolean alreadyLeaked = false;
		Boolean subsurfaceSpillStartPosition = !((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT);
		Boolean chemicalSpill = ((*theLE).pollutantType==CHEMICAL);
		// diffusion coefficient is O(1) vs O(100000) for horizontal / vertical diffusion
		// vertical is 3-5 cm^2/s, divide by sqrt of 10^5
		PtCurMap *map = GetPtCurMap();
		
		//defaults
		//fWaterDensity = 1020;
		//fMixedLayerDepth = 10.;	//meters
		//fBreakingWaveHeight = 1.;	// meters
		//depthAtPoint=5000.;	// allow no bathymetry
		if (map) breakingWaveHeight = map->GetBreakingWaveHeight();	// meters
		if (map) mixedLayerDepth = map->fMixedLayerDepth;	// meters
		if (bUseDepthDependentDiffusion)
		{	
			VelocityRec windVel;
			double vel;
			if (wind) err = wind -> GetTimeValue(model_time,&windVel);	// minus AH 07/10/2012
			if (err || !wind) 
			{
				//printNote("Depth dependent diffusion requires a wind");
				vel = 5;	// instead should have a minimum diffusion coefficient 5cm2/s
			}
			else 
				vel = sqrt(windVel.u*windVel.u + windVel.v*windVel.v);	// m/s
			dragCoeff = (.8+.065*vel)*.001;
			tau = rho_a*dragCoeff*vel*vel;
			uStar = sqrt(tau/rho_w);
			//verticalDiffusionCoefficient = sqrt(2.*(.4*.00138*500/10000.)*timeStep);	// code goes here, use wind speed, other data
			if ((*theLE).z <= 1.5 * breakingWaveHeight)
				//verticalDiffusionCoefficient = sqrt(2.*(karmen*uStar*1.5/10000.)*timeStep);	
				//verticalDiffusionCoefficient = sqrt(2.*(karmen*uStar*1.5)*timeStep);	
				verticalDiffusionCoefficient = sqrt(2.*(karmen*uStar*1.5*breakingWaveHeight)*timeStep);	
			else if ((*theLE).z <= mixedLayerDepth)
				//verticalDiffusionCoefficient = sqrt(2.*(karmen*uStar*(*theLE).z/10000.)*timeStep);	
				verticalDiffusionCoefficient = sqrt(2.*(karmen*uStar*(*theLE).z)*timeStep);	
			else if ((*theLE).z > mixedLayerDepth)
			{
				//verticalDiffusionCoefficient = sqrt(2.*.000011/10000.*timeStep);
				// code goes here, allow user to set this - is this different from fVerticalBottomDiffusionCoefficient which is used for leaking?
				verticalDiffusionCoefficient = sqrt(2.*.000011*timeStep);
				alreadyLeaked = true;
			}
		}
		else*/
		{
			if ((*theLE).z > mixedLayerDepth /*&& !chemicalSpill*/)
			{
				//verticalDiffusionCoefficient = sqrt(2.*.000011/10000.*timeStep);	
				verticalDiffusionCoefficient = sqrt(2.*.000011*timeStep);	// particles that leaked through
				//alreadyLeaked = true;
			}
			else
				verticalDiffusionCoefficient = sqrt(2.*(fVerticalDiffusionCoefficient/10000.)*timeStep);
		}
		GetRandomVectorInUnitCircle(&rand1,&rand2);
		r = sqrt(rand1*rand1+rand2*rand2);
		w = sqrt(-2*log(r)/r);
		// both rand1*w and rand2*w are normal random vars
		deltaPoint.z = rand1*w*verticalDiffusionCoefficient;
		//z = deltaPoint.z;
				
	//}
	//else
		//deltaPoint.z = 0.;	
	
	return deltaPoint;
}

