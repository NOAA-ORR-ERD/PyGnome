/*
 *  Random_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Random_c.h"
#include "CompFunctions.h"
#include "GEOMETRY.H"
#include "Units.h"

#ifdef MAC
#ifdef MPW
#pragma SEGMENT RANDOM_C
#endif
#endif

#ifndef pyGNOME
#include "TModel.h"
#include "TVectMap.h"
extern TModel *model;
#else
#include "Map_c.h"
#include "Model_c.h"
#include "VectMap_c.h"
#define TMap Map_c
extern Model_c *model;
#endif

using std::cout;

Random_c::Random_c (TMap *owner, char *name) : Mover_c (owner, name)
{
	fDiffusionCoefficient = 100000; //  cm**2/sec 
	memset(&fOptimize,0,sizeof(fOptimize));
	SetClassName (name);
	fUncertaintyFactor = 2;		// default uncertainty mult-factor
	bUseDepthDependent = false;
}

OSErr Random_c::PrepareForModelRun()
{
	this -> fOptimize.isFirstStep = true;
	return noErr;
}
OSErr Random_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain)
{
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6.*(fDiffusionCoefficient/10000.)*time_step)/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*time_step)/METERSPERDEGREELAT; // in deg lat
	//this -> fOptimize.isFirstStep = (model_time == start_time);
	return noErr;
}

void Random_c::ModelStepIsDone()
{
	this -> fOptimize.isFirstStep = false;
	memset(&fOptimize,0,sizeof(fOptimize));
}


OSErr Random_c::get_move(int n, unsigned long model_time, unsigned long step_len, char *ref_ra, char *wp_ra) {	
	
	if(!wp_ra) {
		cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	
	this->PrepareForModelStep(model_time, step_len, false);

	WorldPoint3D delta;
	WorldPoint3D *wp;
	WorldPoint3D *ref;
	
	ref = (WorldPoint3D*)ref_ra;
	wp = (WorldPoint3D*)wp_ra;
	
	for (int i = 0; i < n; i++) {
		LERec rec;
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		
		delta = this->GetMove(model_time, step_len, 0, i, &rec, FORECAST_LE);
		
		wp[i].p.pLat += delta.p.pLat / 1000000;
		wp[i].p.pLong += delta.p.pLong / 1000000;
		wp[i].z += delta.z;
	}
	
	this->ModelStepIsDone();
	return noErr;
}

WorldPoint3D Random_c::GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double		dLong, dLat;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand1,rand2;
	double 	diffusionCoefficient;
	
	//if (deltaPoint.z > 0) return deltaPoint;	// only use for surface LEs ?
	
	if (bUseDepthDependent)
	{
		float depth=0.;
		double localDiffusionCoefficient, factor;
#ifndef pyGNOME
		VectorMap_c* vMap = GetNthVectorMap(0);	// get first vector map
		if (vMap) depth = vMap->DepthAtPoint(refPoint);
#endif
		// logD = 1+exp(1-1/.1H) 
		if (depth==0)	// couldn't find the point in dagtree, maybe a different default?
			factor = 1;
		else
			factor = 1 + exp(1 - 1/(.1*depth));
		if (depth>20)
			//localDiffusionCoefficient = pow(10,factor);
			localDiffusionCoefficient = pow(10.,factor);
		else
			localDiffusionCoefficient = 0;
		this -> fOptimize.value =  sqrt(6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		/*if (depth<20)
		 {
		 localDiffusionCoefficient = 0;
		 this -> fOptimize.value =  sqrt(6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		 this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		 }
		 else
		 {
		 localDiffusionCoefficient = 0;
		 this -> fOptimize.value =  sqrt(6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		 this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(localDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		 }*/
		// MoverMap->GetGrid();	// mover map will likely be universal map
		// need to get the bathymetry then set diffusion based on > 20 O(1000), < 20 O(100)
		// figure out where LE is, interpolate to get depth (units?)
	}
	if(!this->fOptimize.isOptimizedForStep && !bUseDepthDependent)  
	{
		this -> fOptimize.value =  sqrt(6.*(fDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
		this -> fOptimize.uncertaintyValue =  sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*timeStep)/METERSPERDEGREELAT; // in deg lat
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
	
	return deltaPoint;
}

#undef TMap
