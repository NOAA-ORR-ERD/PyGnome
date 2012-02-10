/*
 *  Random_c.cpp
 *  gnome
 *
 *  Created by Alex Hadjilambris on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "Random_c.h"
#include "CROSS.H"
#include <iostream>

#ifdef MAC
#ifdef MPW
#pragma SEGMENT RANDOM_C
#endif
#endif

extern TModel *model;

OSErr Random_c::PrepareForModelStep()
{
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6.*(fDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6.*(fDiffusionCoefficient/10000.)*model->GetTimeStep())/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	return noErr;
}

void Random_c::ModelStepIsDone()
{
	memset(&fOptimize,0,sizeof(fOptimize));
}

WorldPoint3D Random_c::GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double		dLong, dLat;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	float rand1,rand2;
	double 	diffusionCoefficient;
	
	//if (deltaPoint.z > 0) return deltaPoint;	// only use for surface LEs ?
	
	if (bUseDepthDependent)
	{
		float depth;
		double localDiffusionCoefficient, factor;
		//TVectorMap* vMap = GetNthVectorMap(0);	// get first vector map
		//if (vMap) depth = vMap->DepthAtPoint(refPoint);
		// logD = 1+exp(1-1/.1H) 
		depth = 0;
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
	
	deltaPoint.p.pLong = dLong;
	deltaPoint.p.pLat  = dLat;
	return deltaPoint;
}
