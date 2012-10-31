/*
 *  Random3D_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Random3D_c.h"
#include "CompFunctions.h"
#include "GEOMETRY.H"
#include "Units.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

TRandom3D *sharedRMover3D;

Random3D_c::Random3D_c (TMap *owner, char *name) : Mover_c(owner, name), Random_c (owner, name)
{
	//fDiffusionCoefficient = 100000; //  cm**2/sec 
	//memset(&fOptimize,0,sizeof(fOptimize));
	fVerticalDiffusionCoefficient = 5; //  cm**2/sec	
	//fVerticalBottomDiffusionCoefficient = .01; //  cm**2/sec, what to use as default?	
	fVerticalBottomDiffusionCoefficient = .11; //  cm**2/sec, Bushy suggested a larger default	
	fHorizontalDiffusionCoefficient = 126; //  cm**2/sec	
	bUseDepthDependentDiffusion = false;
	SetClassName (name);
	//fUncertaintyFactor = 2;		// default uncertainty mult-factor
}
OSErr Random3D_c::PrepareForModelRun()
{
	this -> fOptimize.isFirstStep = true;
	return noErr;
}

OSErr Random3D_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6*(fDiffusionCoefficient/10000)*time_step)/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6*(fDiffusionCoefficient/10000)*time_step)/METERSPERDEGREELAT; // in deg lat
	//this -> fOptimize.isFirstStep = (model_time == start_time);
	return noErr;
}

void Random3D_c::ModelStepIsDone()
{
	this -> fOptimize.isFirstStep = false;
	memset(&fOptimize,0,sizeof(fOptimize));
}


WorldPoint3D Random3D_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double		dLong, dLat, z;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand1,rand2,r,w;
	double 	diffusionCoefficient;
	OSErr err = 0;
	
	if ((*theLE).z==0 && !((*theLE).dispersionStatus==HAVE_DISPERSED) && !((*theLE).dispersionStatus==HAVE_DISPERSED_NAT))	
	{
		if(!this->fOptimize.isOptimizedForStep)  
		{
			this -> fOptimize.value =  sqrt(6*(fDiffusionCoefficient/10000)*timeStep)/METERSPERDEGREELAT; // in deg lat
			this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6*(fDiffusionCoefficient/10000)*timeStep)/METERSPERDEGREELAT; // in deg lat
		}
		if (leType == UNCERTAINTY_LE)
			diffusionCoefficient = this -> fOptimize.uncertaintyValue;
		else
			diffusionCoefficient = this -> fOptimize.value;
		
		if(this -> fOptimize.isFirstStep)
		{
			GetRandomVectorInUnitCircle(&rand1,&rand2);
		}
		else
		{
			rand1 = GetRandomFloat(-1.0, 1.0);
			rand2 = GetRandomFloat(-1.0, 1.0);
		}
		
		dLong = (rand1 * diffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
		dLat  = rand2 * diffusionCoefficient;
		
		// code goes here
		// note: could add code to make it a circle the first step
		
		deltaPoint.p.pLong = dLong * 1000000;
		deltaPoint.p.pLat  = dLat  * 1000000;
	}
	// at first step LE.z is still zero, but the move has dispersed the LE
	
	//if ((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT)	// only apply vertical diffusion if there are particles below surface
	if ((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT || (*theLE).z>0)	// only apply vertical diffusion if there are particles below surface
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
		//breakingWaveHeight = ((PtCurMap*)moverMap)->fBreakingWaveHeight;	// meters
		//breakingWaveHeight = ((PtCurMap*)moverMap)->GetBreakingWaveHeight();	// meters
		//mixedLayerDepth = ((PtCurMap*)moverMap)->fMixedLayerDepth;	// meters

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
		else
		{
			if ((*theLE).z > mixedLayerDepth /*&& !chemicalSpill*/)
			{
				//verticalDiffusionCoefficient = sqrt(2.*.000011/10000.*timeStep);	
				verticalDiffusionCoefficient = sqrt(2.*.000011*timeStep);	// particles that leaked through
				alreadyLeaked = true;
			}
			else
				verticalDiffusionCoefficient = sqrt(2.*(fVerticalDiffusionCoefficient/10000.)*timeStep);
		}
		GetRandomVectorInUnitCircle(&rand1,&rand2);
		r = sqrt(rand1*rand1+rand2*rand2);
		w = sqrt(-2*log(r)/r);
		// both rand1*w and rand2*w are normal random vars
		deltaPoint.z = rand1*w*verticalDiffusionCoefficient;
		z = deltaPoint.z;
		/*if (bUseDepthDependentDiffusion && (*theLE).z <= mixedLayerDepth) 
		 {
		 // code goes here, to get sign need to calculate dC/dz
		 // this is prohibitively slow
		 float *depthSlice = 0;
		 LongPoint lp;
		 long triNum, depthBin=0;
		 TDagTree *dagTree = 0;
		 TTriGridVel3D* triGrid = ((PtCurMap*)moverMap)->GetGrid(true);	
		 if (!triGrid) return deltaPoint; // some error alert, no depth info to check
		 dagTree = triGrid -> GetDagTree();
		 if(!dagTree)	return deltaPoint;
		 depthBin = (long)ceil((*theLE).z);
		 lp.h = (*theLE).p.pLong;
		 lp.v = (*theLE).p.pLat;
		 triNum = dagTree -> WhatTriAmIIn(lp);
		 if (triNum > -1) err = ((PtCurMap*)moverMap)->CreateDepthSlice(triNum,&depthSlice);
		 if(!err && depthBin < depthSlice[0])
		 {
		 if (depthSlice[depthBin+1] < depthSlice[depthBin])
		 // probably should check bin LE would end up in (and those in between?)
		 deltaPoint.z += karmen*uStar*timeStep;	// Yasuo Onishi at NAS requested this additional term
		 else
		 {
		 if (depthBin<=1 || depthSlice[depthBin-1] < depthSlice[depthBin])
		 deltaPoint.z -= karmen*uStar*timeStep;	// Yasuo Onishi at NAS requested this additional term
		 }
		 }
		 if (depthSlice) delete [] depthSlice; depthSlice = 0; 
		 //if (rand1>0)
		 //deltaPoint.z += karmen*uStar*timeStep;	// Yasuo Onishi at NAS requested this additional term
		 //else
		 //deltaPoint.z -= karmen*uStar*timeStep;	// Yasuo Onishi at NAS requested this additional term
		 
		 z = deltaPoint.z;
		 }*/
		
		//horizontalDiffusionCoefficient = sqrt(2.*(fDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT;
		horizontalDiffusionCoefficient = sqrt(2.*(fHorizontalDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT;
		dLong = (rand1 * w * horizontalDiffusionCoefficient )/ LongToLatRatio3 (refPoint.pLat);
		dLat  = rand2 * w * horizontalDiffusionCoefficient;		
		
		// code goes here, option to add some uncertainty to horizontal diffusivity
		deltaPoint.p.pLong = dLong * 1000000;
		deltaPoint.p.pLat  = dLat  * 1000000;
		
		if (map) water_density = map->fWaterDensity/1000.;	// kg/m^3 to g/cm^3
		//water_density = ((PtCurMap*)moverMap)->fWaterDensity/1000.;	// kg/m^3 to g/cm^3
		// check that depth at point is greater than mixed layer depth, else bottom threshold is the depth
		if (map) depthAtPoint = map->DepthAtPoint(refPoint);	// or check rand instead
		//depthAtPoint = ((PtCurMap*)moverMap)->DepthAtPoint(refPoint);	// or check rand instead
		if (depthAtPoint < mixedLayerDepth && depthAtPoint > 0) mixedLayerDepth = depthAtPoint;
		// may want to put in an option to turn buoyancy on and off
		if ((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT)	// no buoyancy for bottom releases of chemicals - what about oil?
			// actually for bottom spill pollutant is not dispersed so there would be no assigned dropletSize unless LE gets redispersed
			buoyancy = (2.*g/9.)*(1.-(*theLE).density/water_density)*((*theLE).dropletSize*1e-6/2.)*((*theLE).dropletSize*1e-6/2)/water_viscosity;
		else if (chemicalSpill)
		{
			double defaultDropletSize = 70;	//for now use largest droplet size for worst case scenario
			buoyancy = (2.*g/9.)*(1.-(*theLE).density/water_density)*(defaultDropletSize*1e-6/2.)*(defaultDropletSize*1e-6/2)/water_viscosity;
		}
		// also should handle non-dispersed subsurface spill
		deltaPoint.z = deltaPoint.z - buoyancy*timeStep;	// double check sign
		totalLEDepth = (*theLE).z+deltaPoint.z;
		// don't let bottom spill diffuse down
		if (subsurfaceSpillStartPosition && (*theLE).z >= depthAtPoint && deltaPoint.z > 0)
		{
			deltaPoint.z = 0; return deltaPoint;
		}
		// if LE has gone above surface redisperse
		/*if (chemicalSpill)
		 {
		 float eps = .00001;
		 deltaPoint.z = GetRandomFloat(0+eps,depthAtPoint-eps) - (*theLE).z;
		 return deltaPoint;
		 }
		 else
		 {
		 deltaPoint.z = GetRandomFloat(0,depthAtPoint) - (*theLE).z;
		 }*/
		if (totalLEDepth<=0) 
		{	// for non-dispersed subsurface spills, allow oil/chemical to resurface
			//if (chemicalSpill)
			if(subsurfaceSpillStartPosition)
			{
			 //deltaPoint.z = GetRandomFloat(0+eps,depthAtPoint-eps) - (*theLE).z;
			 //deltaPoint.z = GetRandomFloat(0+eps,1.) - (*theLE).z;
			 deltaPoint.z = - (*theLE).z;
			 return deltaPoint;
			}
			// need to check if it was a giant step and if so throw le randomly back into mixed layer
			if (abs(deltaPoint.z) > mixedLayerDepth/2. /*|| chemicalSpill*/)	// what constitutes a giant step??
			{
				//deltaPoint.z = GetRandomFloat(0,mixedLayerDepth) - (*theLE).z;
				deltaPoint.z = GetRandomFloat(eps,mixedLayerDepth) - (*theLE).z;
				return deltaPoint;
			}
			deltaPoint.z = -(*theLE).z;	// cancels out old value, since will add deltaPoint.z back to theLE.z on return
			model->ReDisperseOil(theLE,breakingWaveHeight);	// trouble if LE has already moved to shoreline
			if ((*theLE).z <= 0) 
			{	// if there was a problem just reflect
				//deltaPoint.z = -(rand1*w*verticalDiffusionCoefficient - buoyancy*timeStep);
				deltaPoint.z = 0;	// shouldn't happen
			}
			else
				deltaPoint.z += (*theLE).z;	// resets to dispersed value
			return deltaPoint;
		}
		if (!alreadyLeaked && depthAtPoint > 0 && totalLEDepth > depthAtPoint)
		{
			if (subsurfaceSpillStartPosition)
			{
				// reflect above bottom
				deltaPoint.z = depthAtPoint - (totalLEDepth - depthAtPoint) - (*theLE).z; 
				return deltaPoint;
			}
			// put randomly into water column
			//deltaPoint.z = GetRandomFloat(0,depthAtPoint) - (*theLE).z;
			deltaPoint.z = GetRandomFloat(eps,depthAtPoint-eps) - (*theLE).z;
			return deltaPoint;
		}
		if (alreadyLeaked && depthAtPoint > 0 && totalLEDepth > depthAtPoint)
		{
			// reflect above bottom
			deltaPoint.z = depthAtPoint - (totalLEDepth - depthAtPoint) - (*theLE).z; // reflect about mixed layer depth
			return deltaPoint;
		}
		// don't let all LEs leak, bounce up a certain percentage - r = sqrt(kz_top/kz_bot)
		if ((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT && !alreadyLeaked)	// not relevant for bottom spills
		{
			/*if (totalLEDepth>mixedLayerDepth && (!bUseDepthDependentDiffusion || fVerticalBottomDiffusionCoefficient == 0)) // don't allow leaking
			 {
			 deltaPoint.z = mixedLayerDepth - (totalLEDepth - mixedLayerDepth) - (*theLE).z; // reflect about mixed layer depth
			 // below we re-check in case the reflection caused LE to go out of bounds the other way		
			 totalLEDepth = (*theLE).z+deltaPoint.z;
			 }*/
			//if (totalLEDepth>mixedLayerDepth && bUseDepthDependentDiffusion && fVerticalBottomDiffusionCoefficient > 0) // allow leaking
			if (totalLEDepth>mixedLayerDepth) // allow leaking
			{
				double x, reflectRatio = 0., verticalBottomDiffusionCoefficient;
				verticalBottomDiffusionCoefficient = sqrt(2.*(fVerticalBottomDiffusionCoefficient/10000.)*timeStep);
				if (verticalBottomDiffusionCoefficient>0) reflectRatio = sqrt(verticalDiffusionCoefficient/verticalBottomDiffusionCoefficient); // should be > 1
				x = GetRandomFloat(0, 1.0);
				if(x <= reflectRatio/(reflectRatio+1) || fVerticalBottomDiffusionCoefficient == 0 || totalLEDepth > depthAtPoint) // percent to reflect
				{
					deltaPoint.z = mixedLayerDepth - (totalLEDepth - mixedLayerDepth) - (*theLE).z; // reflect about mixed layer depth
					// below we re-check in case the reflection caused LE to go out of bounds the other way		
					totalLEDepth = (*theLE).z+deltaPoint.z;
				}
			}
		}
		// check if leaked les have gone through bottom, otherwise they'll be bumped up to the bottom 1m
		// code goes here, check if a bottom spill le has gone below the bottom
		
		//if (totalLEDepth>=depthAtPoint)
		// redisperse if LE comes to surface
		if (totalLEDepth<=0) 
		{
			//deltaPoint.z = -totalLEDepth - (*theLE).z; // reflect LE
			// code goes here, this should be outside since changing LE.z in movement grid stuff screwy
			// actually it's not the real LErec, so should be ok
			//deltaPoint.z = -(*theLE).z;	// cancels out old value, since will add deltaPoint.z back to theLE.z on return
			/*model->ReDisperseOil(theLE,breakingWaveHeight);	// trouble if LE has already moved to shoreline
			 if ((*theLE).z <= 0) 
			 {
			 deltaPoint.z = -rand1*w*verticalDiffusionCoefficient;
			 }
			 else
			 deltaPoint.z += (*theLE).z;*/	// resets to dispersed value
			// must have been a giant step if reflection sent it over the surface, should put back randomly into mixed layer
			//deltaPoint.z = GetRandomFloat(0,mixedLayerDepth) - (*theLE).z;
			deltaPoint.z = GetRandomFloat(eps,mixedLayerDepth-eps) - (*theLE).z;
		}
	}
	else
		deltaPoint.z = 0.;	
	
	return deltaPoint;
}