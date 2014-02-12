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
	fMixedLayerDepth = 10.; // meters
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


OSErr RandomVertical_c::get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID) {
	
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
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;
		
		delta[i] = this->GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

double GetDepthAtPoint(WorldPoint p)
{
	//will need bathymetry information
	double depthAtPt = INFINITE_DEPTH;
	return depthAtPt;
}

WorldPoint3D RandomVertical_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double	dLong, dLat, z = 0;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand;
	OSErr err = 0;

	//if ((*theLE).z==0)	return deltaPoint;
	// will need a flag to check if LE is supposed to stay on the surface or can be diffused below	
	// will want to be able to set mixed layer depth, but have a local value that can be changed
	if ((*theLE).z>0)	// only apply vertical diffusion if there are particles below surface
	{
		double verticalDiffusionCoefficient;
		double mixedLayerDepth=fMixedLayerDepth, totalLEDepth, depthAtPoint=INFINITE_DEPTH;
		float eps = 1.e-6;
		// diffusion coefficient is O(1) vs O(100000) for horizontal / vertical diffusion
		// vertical is 3-5 cm^2/s, divide by sqrt of 10^4
		
		// instead diffuse particles above MLD, reflect if necessary and then apply bottom diffusion to all particles
		// then check top and bottom. Still need to consider what to do with large steps - put randomly into mixed layer / water column ?
		depthAtPoint = GetDepthAtPoint(refPoint);
		if (depthAtPoint <= 0) depthAtPoint = INFINITE_DEPTH;	// this should be taken care of in GetDepthAtPoint	
		// if (depthAtPoint < eps) // should this be an error?
		if ((*theLE).z<=mixedLayerDepth)
		{
			if (fVerticalDiffusionCoefficient==0) return deltaPoint;	
			verticalDiffusionCoefficient = sqrt(6.*(fVerticalDiffusionCoefficient/10000.)*timeStep);
			rand = GetRandomFloat(-1.0, 1.0);
			deltaPoint.z = rand*verticalDiffusionCoefficient;
			//z = deltaPoint.z;	// will add this on to the next move
			
			// check that depth at point is greater than mixed layer depth, else bottom threshold is the depth
			//depthAtPoint = GetDepthAtPoint(refPoint);	
			
			if (depthAtPoint < mixedLayerDepth) mixedLayerDepth = depthAtPoint;
				
			// also should handle non-dispersed subsurface spill
			totalLEDepth = (*theLE).z+deltaPoint.z;
			
			if (totalLEDepth>mixedLayerDepth) 
			{
				deltaPoint.z = mixedLayerDepth - (totalLEDepth - mixedLayerDepth) - (*theLE).z; // reflect about mixed layer depth
				// check if went above surface and put randomly into mixed layer
				if ((*theLE).z+deltaPoint.z <= 0) deltaPoint.z = GetRandomFloat(eps,mixedLayerDepth) - (*theLE).z;	
					// or just let it go and deal with it later? then it will go into full water column...
			}
		}
		z = deltaPoint.z;	// will add this on to the next move
		if (mixedLayerDepth==depthAtPoint) /*return deltaPoint*/goto dochecks;	// in this case don't need to do anything more
		// now apply below mixed layer depth diffusion to all particles above and below
		if (fVerticalBottomDiffusionCoefficient==0/* && z==0*/) /*return deltaPoint*/goto dochecks;	// don't return until do checks
		verticalDiffusionCoefficient = sqrt(6.*(fVerticalBottomDiffusionCoefficient/10000.)*timeStep);
		rand = GetRandomFloat(-1.0, 1.0);
		deltaPoint.z = rand*verticalDiffusionCoefficient;
		
		z = z + deltaPoint.z;	// add move to previous move if any
		totalLEDepth = (*theLE).z+z;
		// if LE has gone above surface reflect
dochecks:
		if (totalLEDepth==0) 
		{	
			deltaPoint.z = eps - (*theLE).z; 
			return deltaPoint;
		}
		if (totalLEDepth<0) 
		{	
			deltaPoint.z = - totalLEDepth - (*theLE).z;	// reflect below surface
			totalLEDepth = (*theLE).z + deltaPoint.z;
			if (totalLEDepth > depthAtPoint) 
				deltaPoint.z = GetRandomFloat(eps,depthAtPoint-eps) - (*theLE).z;
			return deltaPoint;
		}
		if (totalLEDepth==depthAtPoint) 
		{	
			deltaPoint.z = (depthAtPoint - eps) - (*theLE).z; 
			return deltaPoint;
		}
		if (totalLEDepth > depthAtPoint) 
		{
			// reflect above bottom
			deltaPoint.z = depthAtPoint - (totalLEDepth - depthAtPoint) - (*theLE).z; 
			totalLEDepth = (*theLE).z + deltaPoint.z;
			if (totalLEDepth <= 0) 
				// put randomly into water column
				deltaPoint.z = GetRandomFloat(eps,depthAtPoint-eps) - (*theLE).z;
			return deltaPoint;
		}
		else
			deltaPoint.z = z;
	}
	else
		deltaPoint.z = 0.;	
	
	return deltaPoint;
}
/*
// Box Muller algorithm - needs a factor adjustment since the random value range is limited by the sqrt(-2log(r)/r) calculation (can't have r>1)
// We might want to revisit this sometime if we want an algorithm for generating normally distributed random numbers
WorldPoint3D RandomVertical_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double	dLong, dLat, z = 0;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand1,rand2,r,w;
	OSErr err = 0;

	//if ((*theLE).z==0)	return deltaPoint;
	// will need a flag to check if LE is supposed to stay on the surface or can be diffused below	
	// will want to be able to set mixed layer depth, but have a local value that can be changed
	if ((*theLE).z>0)	// only apply vertical diffusion if there are particles below surface
	{
		double verticalDiffusionCoefficient;
		double mixedLayerDepth=fMixedLayerDepth, totalLEDepth, depthAtPoint=INFINITE_DEPTH;
		float eps = 1.e-6;
		// diffusion coefficient is O(1) vs O(100000) for horizontal / vertical diffusion
		// vertical is 3-5 cm^2/s, divide by sqrt of 10^4
		
		// instead diffuse particles above MLD, reflect if necessary and then apply bottom diffusion to all particles
		// then check top and bottom. Still need to consider what to do with large steps - put randomly into mixed layer / water column ?
		depthAtPoint = GetDepthAtPoint(refPoint);
		if (depthAtPoint <= 0) depthAtPoint = INFINITE_DEPTH;	// this should be taken care of in GetDepthAtPoint	
		// if (depthAtPoint < eps) // should this be an error?
		if ((*theLE).z<=mixedLayerDepth)
		{
			if (fVerticalDiffusionCoefficient==0) return deltaPoint;	
			verticalDiffusionCoefficient = sqrt(2.*(fVerticalDiffusionCoefficient/10000.)*timeStep);
			GetRandomVectorInUnitCircle(&rand1,&rand2);
			r = sqrt(rand1*rand1+rand2*rand2);
			w = sqrt(-2*log(r)/r);
			// both rand1*w and rand2*w are normal random vars
			deltaPoint.z = rand1*w*verticalDiffusionCoefficient;
			//z = deltaPoint.z;	// will add this on to the next move
			
			// check that depth at point is greater than mixed layer depth, else bottom threshold is the depth
			//depthAtPoint = GetDepthAtPoint(refPoint);	
			
			if (depthAtPoint < mixedLayerDepth) mixedLayerDepth = depthAtPoint;
				
			// also should handle non-dispersed subsurface spill
			totalLEDepth = (*theLE).z+deltaPoint.z;
			
			if (totalLEDepth>mixedLayerDepth) 
			{
				deltaPoint.z = mixedLayerDepth - (totalLEDepth - mixedLayerDepth) - (*theLE).z; // reflect about mixed layer depth
				// check if went above surface and put randomly into mixed layer
				if ((*theLE).z+deltaPoint.z <= 0) deltaPoint.z = GetRandomFloat(eps,mixedLayerDepth) - (*theLE).z;	
					// or just let it go and deal with it later? then it will go into full water column...
			}
		}
		z = deltaPoint.z;	// will add this on to the next move
		if (mixedLayerDepth==depthAtPoint) goto dochecks;	// in this case don't need to do anything more
		// now apply below mixed layer depth diffusion to all particles above and below
		if (fVerticalBottomDiffusionCoefficient==0) goto dochecks;	// don't return until do checks
		verticalDiffusionCoefficient = sqrt(2.*(fVerticalBottomDiffusionCoefficient/10000.)*timeStep);
		GetRandomVectorInUnitCircle(&rand1,&rand2);
		r = sqrt(rand1*rand1+rand2*rand2);
		w = sqrt(-2*log(r)/r);
		// both rand1*w and rand2*w are normal random vars
		deltaPoint.z = rand1*w*verticalDiffusionCoefficient;
		
		z = z + deltaPoint.z;	// add move to previous move if any
		totalLEDepth = (*theLE).z+z;
		// if LE has gone above surface reflect
dochecks:
		if (totalLEDepth==0) 
		{	
			deltaPoint.z = eps - (*theLE).z; 
			return deltaPoint;
		}
		if (totalLEDepth<0) 
		{	
			deltaPoint.z = - totalLEDepth - (*theLE).z;	// reflect below surface
			totalLEDepth = (*theLE).z + deltaPoint.z;
			if (totalLEDepth > depthAtPoint) 
				deltaPoint.z = GetRandomFloat(eps,depthAtPoint-eps) - (*theLE).z;
			return deltaPoint;
		}
		if (totalLEDepth==depthAtPoint) 
		{	
			deltaPoint.z = (depthAtPoint - eps) - (*theLE).z; 
			return deltaPoint;
		}
		if (totalLEDepth > depthAtPoint) 
		{
			// reflect above bottom
			deltaPoint.z = depthAtPoint - (totalLEDepth - depthAtPoint) - (*theLE).z; 
			totalLEDepth = (*theLE).z + deltaPoint.z;
			if (totalLEDepth <= 0) 
				// put randomly into water column
				deltaPoint.z = GetRandomFloat(eps,depthAtPoint-eps) - (*theLE).z;
			return deltaPoint;
		}
		else
			deltaPoint.z = z;
	}
	else
		deltaPoint.z = 0.;	
	
	return deltaPoint;
}*/

