
#include "Cross.h"

TRandom3D *sharedRMover3D;


#ifdef MAC
#ifdef MPW
#pragma SEGMENT TRANDOM3D
#endif
#endif

///////////////////////////////////////////////////////////////////////////

TRandom3D::TRandom3D (TMap *owner, char *name) : TMover(owner, name), TRandom (owner, name)
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


long TRandom3D::GetListLength()
{
	long count = 1;
	
	if (bOpen) {
		count += 2;
		if(model->IsUncertain())count++;
		count++;	// vertical diffusion coefficient
	}
	
	return count;
}

ListItem TRandom3D::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { this, 0, indent, 0 };
	char valStr[32],valStr2[32];
	
	if (n == 0) {
		item.index = I_RANDOMNAME;
		item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		sprintf(text, "Random3D: \"%s\"", className);
		if(!bActive)*style = italic; // JLM 6/14/10
		
		return item;
	}
	
	n -= 1;
	item.indent++;
	
	if (bOpen) {
		if (n == 0) {
			item.index = I_RANDOMACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
		
		n -= 1;
		
		if (n == 0) {
			item.index = I_RANDOMAREA;
			StringWithoutTrailingZeros(valStr,fDiffusionCoefficient,0);
			sprintf(text, "%s cm**2/sec (surface)", valStr);
			
			return item;
		}
		
		n -= 1;
		
		if(model->IsUncertain())
		{
			if (n == 0) {
				item.index = I_RANDOMUFACTOR;
				StringWithoutTrailingZeros(valStr,fUncertaintyFactor,0);
				sprintf(text, "Uncertainty factor: %s", valStr);
				
				return item;
			}
	
			n -= 1;
		}
		
		if (n == 0) {
			item.index = I_RANDOMVERTAREA;
			StringWithoutTrailingZeros(valStr,fVerticalDiffusionCoefficient,0);
			StringWithoutTrailingZeros(valStr2,fHorizontalDiffusionCoefficient,0);
			if (bUseDepthDependentDiffusion)
				sprintf(text, "vert = f(z), %s cm**2/sec (horiz)", valStr2);
			else
				sprintf(text, "%s cm**2/sec (vert), %s cm**2/sec (horiz)", valStr, valStr2);		

			return item;
		}
		
		n -= 1;
		
	}
	
	item.owner = 0;
	
	return item;
}

Boolean TRandom3D::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_RANDOMNAME: bOpen = !bOpen; return TRUE;
			case I_RANDOMACTIVE: bActive = !bActive; 
				model->NewDirtNotification(); return TRUE;
		}
	
	if (doubleClick)
		if(item.index==I_RANDOMAREA)
			TRandom::ListClick(item,inBullet,doubleClick);
		else
			Random3DSettingsDialog(this, this -> moverMap);
	
	// do other click operations...
	
	return FALSE;
}

/*Boolean TRandom3D::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_RANDOMNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
			}
			break;
		default:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case SETTINGSBUTTON: return TRUE;
				case DELETEBUTTON: return FALSE;
			}
			break;
	}
	
	return TMover::FunctionEnabled(item, buttonID);
}*/

OSErr TRandom3D::SettingsItem(ListItem item)
{
	switch (item.index) {
		default:
			return Random3DSettingsDialog(this, this -> moverMap);
	}
	
	return 0;
}

OSErr TRandom3D::DeleteItem(ListItem item)
{
	if (item.index == I_RANDOMNAME)
		return moverMap -> DropMover (this);
	
	return 0;
}

OSErr TRandom3D::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM
	/*char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val = 0;
		OSErr  err;
		err = message->GetParameterAsDouble("coverage",&val); // old style
		if(err) err = message->GetParameterAsDouble("Coefficient",&val);
		if(!err)
		{	
			if(val >= 0)// do we have any other  max or min limits ?
			{
				this->fDiffusionCoefficient = val;
				model->NewDirtNotification();// tell model about dirt
			}
		}
		///
		err = message->GetParameterAsDouble("Uncertaintyfactor",&val);
		if(!err)
		{	
			if(val >= 1.0)// do we have any other  max or min limits ?
			{
				this->fUncertaintyFactor = val;
				model->NewDirtNotification();// tell model about dirt
			}
		}
		///
		
	}*/
	/////////////////////////////////////////////////
	// we have no sub-guys that that need us to pass this message 
	/////////////////////////////////////////////////

	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TRandom::CheckAndPassOnMessage(message);
}

/////////////////////////////////////////////////

OSErr TRandom3D::PrepareForModelStep()
{
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6*(fDiffusionCoefficient/10000)*model->GetTimeStep())/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.uncertaintyValue = sqrt(fUncertaintyFactor*6*(fDiffusionCoefficient/10000)*model->GetTimeStep())/METERSPERDEGREELAT; // in deg lat
	this -> fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	return noErr;
}

void TRandom3D::ModelStepIsDone()
{
	memset(&fOptimize,0,sizeof(fOptimize));
}

/////////////////////////////////////////////////

WorldPoint3D TRandom3D::GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
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
		double mixedLayerDepth, totalLEDepth, breakingWaveHeight, depthAtPoint;
		double karmen = .4, rho_a = 1.29, rho_w = 1030., dragCoeff, tau, uStar;
		float water_density,water_viscosity = 1.e-6,eps = 1.e-6;
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
		if (map) breakingWaveHeight = map->GetBreakingWaveHeight();	// meters
		if (map) mixedLayerDepth = map->fMixedLayerDepth;	// meters
		if (bUseDepthDependentDiffusion)
		{	
			VelocityRec windVel;
			double vel;
			if (wind) err = wind -> GetTimeValue(model->GetModelTime(),&windVel);
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
			/*if (chemicalSpill)
			{
					//deltaPoint.z = GetRandomFloat(0+eps,depthAtPoint-eps) - (*theLE).z;
					deltaPoint.z = GetRandomFloat(0+eps,1.) - (*theLE).z;
					return deltaPoint;
			}*/
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

#define TRandom3D_FileVersion 2
//#define TRandom3D_FileVersion 1
OSErr TRandom3D::Write(BFPB *bfpb)
{
	long version = TRandom3D_FileVersion;
	ClassID id = GetClassID ();
	OSErr err = 0;
	
	if (err = TMover::Write(bfpb)) return err;
	
	StartReadWriteSequence("TRandom3D::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	
	if (err = WriteMacValue(bfpb, fDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, fUncertaintyFactor)) return err;
	
	if (err = WriteMacValue(bfpb, fVerticalDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, fHorizontalDiffusionCoefficient)) return err;

	//if (err = WriteMacValue(bfpb, fVerticalBottomDiffusionCoefficient)) return err;
	if (err = WriteMacValue(bfpb, bUseDepthDependentDiffusion)) return err;

	return 0;
}

OSErr TRandom3D::Read(BFPB *bfpb) 
{
	long version;
	ClassID id;
	OSErr err = 0;
	
	if (err = TMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("TRandom3D::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("TRandom3D::Read()", "id != GetClassID", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > TRandom3D_FileVersion) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &fDiffusionCoefficient)) return err;
	if (err = ReadMacValue(bfpb, &fUncertaintyFactor)) return err;
	
	if (gMearnsVersion || version > 1)
	{
	if (err = ReadMacValue(bfpb, &fVerticalDiffusionCoefficient)) return err;
	if (err = ReadMacValue(bfpb, &fHorizontalDiffusionCoefficient)) return err;
	//if (err = ReadMacValue(bfpb, &fVerticalDiffusionCoefficient)) return err;
	}
	if (version>1)
		if (err = ReadMacValue(bfpb, &bUseDepthDependentDiffusion)) return err;
	
	return 0;
}

static PopInfoRec RandomPopTable[] = {
		{ M28b, nil, M28bINPUTTYPE, 0, pRANDOMINPUTTYPE, 0, 1, FALSE, nil }
	};

void ShowHideRandomDialogItems(DialogPtr dialog)
{
	Boolean showHorizItems, showVertItems, showCurrentWindItems;
	short typeOfInfoSpecified = GetPopSelection(dialog, M28bINPUTTYPE);

	Boolean depthDep  = GetButton (dialog, M28bDEPTHDEPENDENT); 
	
	if (depthDep)
	{
		ShowHideDialogItem(dialog, M28bINPUTTYPE, false);
		typeOfInfoSpecified = 2;
	}
	else
		ShowHideDialogItem(dialog, M28bINPUTTYPE, true); 
	

	switch (typeOfInfoSpecified)
	{
		//default:
		//case Input eddy diffusion values:
		case 1:
			showHorizItems=TRUE;
			showVertItems=TRUE;
			showCurrentWindItems=FALSE;
			break;
		//case Input current and wind speed:
		case 2:
			showCurrentWindItems=FALSE;
			showHorizItems=TRUE;
			showVertItems=FALSE;
			break;
		case 3:
			showCurrentWindItems=TRUE;
			showHorizItems=FALSE;
			showVertItems=FALSE;
			break;
	}
	ShowHideDialogItem(dialog, M28bDIFFUSION, showHorizItems ); 
	ShowHideDialogItem(dialog, M28bUFACTOR, showHorizItems); 
	ShowHideDialogItem(dialog, M28bFROST1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONLABEL1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONUNITS1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bUNCERTAINTYLABEL1, showHorizItems); 
	ShowHideDialogItem(dialog, M28bHORIZONTALLABEL, showHorizItems); 

	ShowHideDialogItem(dialog, M28bVERTDIFFUSION, showVertItems); 
	ShowHideDialogItem(dialog, M28bVERTUFACTOR, showVertItems); 
	ShowHideDialogItem(dialog, M28bFROST2, showVertItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONLABEL2, showVertItems); 
	ShowHideDialogItem(dialog, M28bDIFFUSIONUNITS2, showVertItems); 
	ShowHideDialogItem(dialog, M28bUNCERTAINTYLABEL2, showVertItems); 
	ShowHideDialogItem(dialog, M28bVERTICALLABEL, showVertItems); 

	ShowHideDialogItem(dialog, M28bWINDSPEEDLABEL, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bWINDSPEED, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bCURRENTSPEEDLABEL, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bCURRENTSPEED, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bCURRENTSPEEDUNITS, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bWINDSPEEDUNITS, showCurrentWindItems); 
	ShowHideDialogItem(dialog, M28bFROST3, showCurrentWindItems); 

	//ShowHideDialogItem(dialog, M28bBOTKZLABEL, depthDep); 
	//ShowHideDialogItem(dialog, M28bBOTKZ, depthDep); 
	//ShowHideDialogItem(dialog, M28bBOTKZUNITS, depthDep); 
}

OSErr M28bInit (DialogPtr dialog, VOIDPTR data)
// new random diffusion dialog init
{
	SetDialogItemHandle(dialog, M28bHILITEDEFAULT, (Handle)FrameDefault);
	SetDialogItemHandle(dialog, M28bFROST1, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M28bFROST2, (Handle)FrameEmbossed);
	SetDialogItemHandle(dialog, M28bFROST3, (Handle)FrameEmbossed);

	RegisterPopTable (RandomPopTable, sizeof (RandomPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog (M28b, dialog);
	
	SetPopSelection (dialog, M28bINPUTTYPE, 1);

	mysetitext(dialog, M28bNAME, sharedRMover3D -> className);
	//MySelectDialogItemText (dialog, M28bNAME, 0, 100);

	SetButton(dialog, M28bACTIVE, sharedRMover3D -> bActive);
	SetButton(dialog, M28bDEPTHDEPENDENT, sharedRMover3D -> bUseDepthDependentDiffusion);

	//Float2EditText(dialog, M28bDIFFUSION, sharedRMover3D->fDiffusionCoefficient, 0);
	Float2EditText(dialog, M28bDIFFUSION, sharedRMover3D->fHorizontalDiffusionCoefficient, 0);

	Float2EditText(dialog, M28bUFACTOR, sharedRMover3D->fUncertaintyFactor, 0);
	
	Float2EditText(dialog, M28bVERTDIFFUSION, sharedRMover3D->fVerticalDiffusionCoefficient, 0);

	Float2EditText(dialog, M28bBOTKZ, sharedRMover3D->fVerticalBottomDiffusionCoefficient, 0);

	//Float2EditText(dialog, M28bUFACTOR, sharedRMover3D->fUncertaintyFactor, 0);
	Float2EditText(dialog, M28bVERTUFACTOR, sharedRMover3D->fUncertaintyFactor, 0);
	
	ShowHideRandomDialogItems(dialog);
	MySelectDialogItemText(dialog, M28bDIFFUSION,0,255);
	
	return 0;
}

short M28bClick (DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
// old random diffusion dialog
{
	char	name [kMaxNameLen];
	double uncertaintyFactor,vertStep; 
	double U=0,W=0,botKZ=0;
	short typeOfInput;
	long menuID_menuItem;
	
	switch (itemNum) {
		case M28bOK:
		{
			PtCurMap *map = GetPtCurMap();
			//PtCurMap *map = (PtCurMap*) (sharedRMover3D -> GetMoverMap());
			// uncertaintyFactor enforce >= 1.0
			uncertaintyFactor = EditText2Float(dialog, M28bUFACTOR);
			if(uncertaintyFactor <1.0)
			{
				printError("The uncertainty factor must be >= 1.0");
				MySelectDialogItemText (dialog, M28bUFACTOR, 0, 255);
				break;
			}
			mygetitext(dialog, M28bNAME, name, kMaxNameLen - 1);		// get the mover's nameStr
			sharedRMover3D -> SetClassName (name);
			sharedRMover3D -> SetActive (GetButton(dialog, M28bACTIVE));
			sharedRMover3D -> bUseDepthDependentDiffusion = GetButton(dialog, M28bDEPTHDEPENDENT);
			
			typeOfInput = GetPopSelection(dialog, M28bINPUTTYPE);
			if (sharedRMover3D -> bUseDepthDependentDiffusion) 
			{
				typeOfInput = 2;	// why??
				/*botKZ = EditText2Float(dialog,M28bBOTKZ);
				if (botKZ == 0)
				{
					printError("You must enter a value for the vertical diffusion coefficient on the bottom");
					MySelectDialogItemText(dialog, M28bBOTKZ,0,255);
					break;
				}*/
				//sharedRMover3D -> fVerticalBottomDiffusionCoefficient = EditText2Float(dialog, M28bBOTKZ);
			}
			sharedRMover3D -> fVerticalBottomDiffusionCoefficient = EditText2Float(dialog, M28bBOTKZ);
			
			if (typeOfInput==1)
			{
				//sharedRMover3D -> fDiffusionCoefficient = EditText2Float(dialog, M28DIFFUSION);
				sharedRMover3D -> fHorizontalDiffusionCoefficient = EditText2Float(dialog, M28bDIFFUSION);
				sharedRMover3D -> fUncertaintyFactor = uncertaintyFactor;
				sharedRMover3D -> fVerticalDiffusionCoefficient = EditText2Float(dialog, M28bVERTDIFFUSION);
				//sharedRMover3D -> fVerticalUncertaintyFactor = uncertaintyFactor;
			}
			else if (typeOfInput==2)
			{
				//sharedRMover3D -> fDiffusionCoefficient = EditText2Float(dialog, M28DIFFUSION);
				sharedRMover3D -> fHorizontalDiffusionCoefficient = EditText2Float(dialog, M28bDIFFUSION);
				sharedRMover3D -> fUncertaintyFactor = uncertaintyFactor;
				sharedRMover3D -> fVerticalDiffusionCoefficient = sharedRMover3D -> fHorizontalDiffusionCoefficient/6.88;
				//sharedRMover3D -> fVerticalUncertaintyFactor = uncertaintyFactor;
			}
			else if (typeOfInput==3)
			{
				U = EditText2Float(dialog,M28bCURRENTSPEED);
				if (U == 0)
				{
					printError("You must enter a value for the current velocity");
					MySelectDialogItemText(dialog, M28bCURRENTSPEED,0,255);
					break;
				}
				W = EditText2Float(dialog,M28bWINDSPEED);
				if (W == 0)
				{
					printError("You must enter a value for the wind velocity");
					MySelectDialogItemText(dialog, M28bWINDSPEED,0,255);
					break;
				}
				sharedRMover3D -> fHorizontalDiffusionCoefficient = (272.8*U + 21.1*W); //cm^2/s - Note the conversion from m^2/s is done by leaving out a 10^-4 factor
				sharedRMover3D -> fVerticalDiffusionCoefficient = (39.7*U + 3.1*W);	//cm^2/s
			}
			vertStep = sqrt(6*(sharedRMover3D -> fVerticalDiffusionCoefficient/10000)*model->GetTimeStep()); // in meters
			// compare to mixed layer depth and warn if within a certain percentage - 
			if (vertStep > map->fMixedLayerDepth)
				printNote("The combination of large vertical diffusion coefficient and choice of timestep will likely result in particles moving vertically on the order of the size of the mixed layer depth. They will be randomly placed in the mixed layer if reflection fails.");
			

			return M28bOK;
		}

		case M28bCANCEL: return M28bCANCEL;
		
		case M28bACTIVE:
			ToggleButton(dialog, M28bACTIVE);
			break;
			
		case M28bDEPTHDEPENDENT:
			ToggleButton(dialog, M28bDEPTHDEPENDENT);
			sharedRMover3D -> bUseDepthDependentDiffusion = GetButton(dialog,M28bDEPTHDEPENDENT);
			ShowHideRandomDialogItems(dialog);
			break;
			
		case M28bDIFFUSION:
			CheckNumberTextItem(dialog, itemNum, false); //  don't allow decimals
			break;

		case M28bWINDSPEED:
		case M28bCURRENTSPEED:
		case M28bBOTKZ:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;

		case M28bUFACTOR:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;

		case M28bVERTDIFFUSION:
			CheckNumberTextItem(dialog, itemNum, true); //  allow decimals
			break;

		case M28bVERTUFACTOR:
			CheckNumberTextItem(dialog, itemNum, true); //   allow decimals
			break;

		case M28bINPUTTYPE:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowHideRandomDialogItems(dialog);
			if (GetPopSelection(dialog, M28bINPUTTYPE)==3)
				MySelectDialogItemText(dialog, M28bCURRENTSPEED,0,255);
			else
				MySelectDialogItemText(dialog, M28bDIFFUSION,0,255);
			break;

	}
	
	return 0;
}

OSErr Random3DSettingsDialog(TRandom3D *mover, TMap *owner)
{
	short item;
	TRandom3D *newMover = 0;
	OSErr err = 0;
	
	if (!mover) {
		newMover = new TRandom3D(owner, "3D Diffusion");
		if (!newMover)
			{ TechError("RandomSettingsDialog()", "new TRandom3D()", 0); return -1; }
		
		if (err = newMover->InitMover()) { delete newMover; return err; }
		
		sharedRMover3D = newMover;
	}
	else
		sharedRMover3D = mover;
	
	item = MyModalDialog(M28b, mapWindow, 0, M28bInit, M28bClick);
	
	if (item == M28bOK) model->NewDirtNotification();

	if (newMover) {
		if (item == M28bOK) {
			if (err = owner->AddMover(newMover, 0))
				{ newMover->Dispose(); delete newMover; return -1; }
		}
		else {
			newMover->Dispose();
			delete newMover;
		}
	}
	
	return 0;
}

