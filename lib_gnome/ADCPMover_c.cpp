/*
 *  ADCPMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADCPMover_c.h"
#include "CROSS.H"

OSErr ADCPMover_c::ComputeVelocityScale()
{	// this function computes and sets this->refScale
	// returns Error when the refScale is not defined
	// or in allowable range.  
	// Note it also sets the refScale to 0 if there is an error
#define MAXREFSCALE  1.0e6  // 1 million times is too much
#define MIN_UNSCALED_REF_LENGTH 1.0E-5 // it's way too small
	long i, j, m, n;
	double length, theirLengthSq, myLengthSq, dotProduct;
	VelocityRec theirVelocity,myVelocity;
	TMap *map;
	ADCPMover *mover;
	
	/*if (this->timeDep && this->timeDep->fFileType==HYDROLOGYFILE)
	 {
	 this->refScale = this->timeDep->fScaleFactor;
	 return noErr;
	 }*/
	
	
	switch (scaleType) {
		case SCALE_NONE: this->refScale = 1; return noErr;
		case SCALE_CONSTANT:
			myVelocity = GetPatValue(refP);
			length = sqrt(myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
			/// check for too small lengths
			if(fabs(scaleValue) > length*MAXREFSCALE
			   || length < MIN_UNSCALED_REF_LENGTH)
			{ this->refScale = 0;return -1;} // unable to compute refScale
			this->refScale = scaleValue / length; 
			return noErr;
		case SCALE_OTHERGRID:
			for (j = 0, m = model -> mapList -> GetItemCount() ; j < m ; j++) {
				model -> mapList -> GetListItem((Ptr)&map, j);
				
				for (i = 0, n = map -> moverList -> GetItemCount() ; i < n ; i++) {
					map -> moverList -> GetListItem((Ptr)&mover, i);
					if (mover -> GetClassID() != TYPE_ADCPMOVER) continue;
					if (!strcmp(mover -> className, scaleOtherFile)) {
						// JLM, note: we are implicitly matching by file name above
						
						// JLM: This code left out the possibility of a time file
						//velocity = mover -> GetPatValue(refP);
						//velocity.u *= mover -> refScale;
						//velocity.v *= mover -> refScale;
						// so use GetScaledPatValue() instead
						theirVelocity = mover -> GetScaledPatValue(refP,nil);
						
						theirLengthSq = (theirVelocity.u * theirVelocity.u + theirVelocity.v * theirVelocity.v);
						// JLM, we need to adjust the movers pattern 
						myVelocity = GetPatValue(refP);
						myLengthSq = (myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
						// next problem is that the scale 
						// can be negative, we would have to look at the angle between 
						// these guys
						
						///////////////////////
						// JLM wonders if we should use a refScale 
						// that will give us the projection of their 
						// vector onto our vector instead of 
						// matching lengths.
						// Bushy etc may have a reason for the present method
						//
						// code goes here
						// ask about the proper method
						///////////////////////
						
						// JLM,  check for really small lengths
						if(theirLengthSq > myLengthSq*MAXREFSCALE*MAXREFSCALE
						   || myLengthSq <  MIN_UNSCALED_REF_LENGTH*MIN_UNSCALED_REF_LENGTH)
						{ this->refScale = 0;return -1;} // unable to compute refScale
						
						dotProduct = myVelocity.u * theirVelocity.u + myVelocity.v * theirVelocity.v;
						this->refScale = sqrt(theirLengthSq / myLengthSq);
						if(dotProduct < 0) this->refScale = -(this->refScale);
						return noErr;
					}
				}
			}
			break;
	}
	
	this->refScale = 0;
	return -1;
}

OSErr ADCPMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty)
{
	/// 5/12/99 only add the eddy uncertainty when told to
	
	double u,v,lengthS,alpha,beta,v0,gammaScale;
	LEUncertainRec unrec;
	float rand1,rand2;
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty
	
	
	if(useEddyUncertainty)
	{
		if(this -> fOptimize.isFirstStep)
		{
			GetRandomVectorInUnitCircle(&rand1,&rand2);
		}
		else
		{
			rand1 = GetRandomFloat(-1.0, 1.0);
			rand2 = GetRandomFloat(-1.0, 1.0);
		}
	}
	else
	{	// no need to calculate these when useEddyUncertainty is false
		rand1 = 0;
		rand2 = 0;
	}
	
	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(patVelocity->u*patVelocity->u + patVelocity->v * patVelocity->v);
		
		
		u = patVelocity->u;
		v = patVelocity->v;
		
		if(!this -> fOptimize.isOptimizedForStep)  this -> fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/timeStep); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
		
		v0 = this -> fEddyV0;		 //meters /second
		
		if(lengthS>1e-6) // so we don't divide by zero
		{	
			if(useEddyUncertainty) gammaScale = this -> fOptimize.value * v0 /(lengthS * (v0+lengthS));
			else  gammaScale = 0.0;
			
			alpha = unrec.downStream + gammaScale * rand1;
			beta = unrec.crossStream + gammaScale * rand2;
			
			patVelocity->u = u*(1+alpha)+v*beta;
			patVelocity->v = v*(1+alpha)-u*beta;	
		}
		else
		{	// when lengthS is too small, ignore the downstream and cross stream and only use diffusion uncertainty	
			if(useEddyUncertainty) { // provided we are supposed to
				patVelocity->u = this -> fOptimize.value * rand1;
				patVelocity->v = this -> fOptimize.value * rand2;
			}
		}
	}
	else 
	{
		TechError("ADCPMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		patVelocity->u=patVelocity->v=0;
	}
	return err;
}



OSErr ADCPMover_c::PrepareForModelStep()
{
	OSErr err =0;
	
	if (err = CurrentMover_c::PrepareForModelStep()) return err; // note: this calls UpdateUncertainty()
	
	err = this -> ComputeVelocityScale();// JLM, will this do it ???
	
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/model->GetTimeStep()); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	this -> fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	
	if (err) 
		printError("An error occurred in ADCPMover::PrepareForModelStep");
	return err;
}

void ADCPMover_c::ModelStepIsDone()
{
	memset(&fOptimize,0,sizeof(fOptimize));
}


WorldPoint3D ADCPMover_c::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	Boolean useEddyUncertainty = false;	
	double 		dLong, dLat;
	WorldPoint3D	deltaPoint={0,0,0.};
	WorldPoint refPoint = (*theLE).p;
	WorldPoint3D thisPoint;	
	VelocityRec scaledPatVelocity = {0.,0.};
	
	thisPoint.p = (*theLE).p;
	thisPoint.z = (*theLE).z;
	scaledPatVelocity = this->GetVelocityAtPoint(thisPoint);
	if(leType == UNCERTAINTY_LE)
	{
		AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
	}
	dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	//deltaPoint.p.z = scaledPatVelocity.w * timeStep;
	
	return deltaPoint;
}

VelocityRec ADCPMover_c::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
{	
	VelocityRec	patVelocity, timeValue = {1, 1};
	float lengthSquaredBeforeTimeFactor;
	OSErr err = 0;
	long i, n, depthIndex1, depthIndex2;
	float depthAtPoint = 0., totalDepth = 1000.;
	
	ADCPTimeValue *thisTimeDep;
	// will need to interpolate
	for (i = 0; i < timeDepList -> GetItemCount (); i++)
	{
		timeDepList -> GetListItem ((Ptr) &thisTimeDep, i);
		if(thisTimeDep && thisTimeDep->bActive)
		{	// eventually will want to interpolate based on where p is
			thisTimeDep->GetDepthIndices(depthAtPoint, totalDepth, &depthIndex1, &depthIndex2);
			// need to get top/bottom depths if necessary, calculate scale
			// if (moverMap->IAm(TYPE_PTCURMAP)) totalDepth = DepthAtPoint(p);
			err = thisTimeDep -> GetTimeValue (model -> GetModelTime(), &timeValue); 
			/*if (depthIndex1 != UNASSIGNEDINDEX)	
			 err = thisTimeDep->GetTimeValueAtDepth(depthIndex, model->GetModelTime(), &timeValue);
			 if (depthIndex2 != UNASSIGNEDINDEX)	
			 err = thisTimeDep->GetTimeValueAtDepth(depthIndex, model->GetModelTime(), &timeValue);*/
		}
	}
	
	patVelocity = GetPatValue (p);
	
	//patVelocity.u *= refScale; 
	//patVelocity.v *= refScale; 
	
	patVelocity.u *= timeValue.u; 
	patVelocity.v *= timeValue.v;
	
	return patVelocity;
}

VelocityRec ADCPMover_c::GetVelocityAtPoint(WorldPoint3D p)
{	// change this to  take WorldPoint3D, no eddy
	VelocityRec	patVelocity, timeValue = {0, 0}, topTimeValue = {0,0}, bottomTimeValue = {0,0};
	float lengthSquaredBeforeTimeFactor;
	OSErr err = 0;
	long i, n, depthIndex1, depthIndex2, numActiveStations = 0;
	float depthAtPoint = 0., totalDepth = 0., distInKm, interpolatedU = 0., interpolatedV = 0.;
	double depthAtBin1=0., depthAtBin2, depthAlpha = 1., sumOfDists = 0., weight=1.;
	WorldPoint stationPosition;
	
	ADCPTimeValue *thisTimeDep;
	// will need to interpolate - this will get redone for multiple stations
	if (moverMap->IAm(TYPE_PTCURMAP)) 
	{
		totalDepth = /*CHECK*/(dynamic_cast<PtCurMap *>(moverMap))->DepthAtPoint(p.p);
		depthAtPoint = p.z;
	}
	for (i = 0; i < timeDepList -> GetItemCount (); i++)
	{
		timeDepList -> GetListItem ((Ptr) &thisTimeDep, i);
		if(thisTimeDep && thisTimeDep->bActive)
		{	
			stationPosition = thisTimeDep->GetStationPosition();
			distInKm = DistanceBetweenWorldPoints(p.p,stationPosition);
			if (distInKm>0) sumOfDists = sumOfDists + 1./distInKm*distInKm;
			numActiveStations++;
		}
	}
	if (numActiveStations==0) return timeValue;
	for (i = 0; i < timeDepList -> GetItemCount (); i++)
	{
		timeDepList -> GetListItem ((Ptr) &thisTimeDep, i);
		if(thisTimeDep && thisTimeDep->bActive)
		{	// eventually will want to interpolate based on where p is
			// calculate distance from p to each of the stations for weight factor, also will need a zone of influence
			if (numActiveStations>1)
			{
				stationPosition = thisTimeDep->GetStationPosition();
				distInKm = DistanceBetweenWorldPoints(p.p,stationPosition);
				if (distInKm>0) weight = (1./distInKm*distInKm)/sumOfDists;
			}
			if (fBinToUse > 0)
			{
				if (thisTimeDep->GetSensorOrientation()==2)
					depthIndex1 = fBinToUse-1;
				else depthIndex1 = thisTimeDep->GetNumBins() - fBinToUse;
				depthIndex2=UNASSIGNEDINDEX;
			}
			else
				thisTimeDep->GetDepthIndices(depthAtPoint, totalDepth, &depthIndex1, &depthIndex2);
			if (depthIndex2!=UNASSIGNEDINDEX)
			{
				depthAtBin2 = thisTimeDep -> GetBinDepth(depthIndex2);
				depthAtBin1 = thisTimeDep -> GetBinDepth(depthIndex1);
				depthAlpha = (depthAtBin2 - p.z)/(double)(depthAtBin2 - depthAtBin1);
			}
			// need to get top/bottom depths if necessary, calculate scale
			//err = thisTimeDep -> GetTimeValue (model -> GetModelTime(), &timeValue); 
			if (depthIndex1 != UNASSIGNEDINDEX)
			{	
				err = thisTimeDep->GetTimeValueAtDepth(depthIndex1, model->GetModelTime(), &topTimeValue);
				if (!err && depthIndex2 != UNASSIGNEDINDEX)	
				{
					err = thisTimeDep->GetTimeValueAtDepth(depthIndex2, model->GetModelTime(), &bottomTimeValue);
				}
				if (!err)
				{
					timeValue.u = depthAlpha*topTimeValue.u+(1-depthAlpha)*bottomTimeValue.u;
					timeValue.v = depthAlpha*topTimeValue.v+(1-depthAlpha)*bottomTimeValue.v;
					//if (numActiveStations>1)
					{
						interpolatedU = interpolatedU + timeValue.u*weight;
						interpolatedV = interpolatedV + timeValue.v*weight;
					}
				}
				
			}
		}
	}
	
	patVelocity = GetPatValue (p.p);
	
	//patVelocity.u *= refScale; 
	//patVelocity.v *= refScale; 
	
	if (numActiveStations>1)
	{
		patVelocity.u *= interpolatedU; 
		patVelocity.v *= interpolatedV;
	}
	else
	{
		patVelocity.u *= timeValue.u; 
		patVelocity.v *= timeValue.v;
	}
	
	return patVelocity;
}


VelocityRec ADCPMover_c::GetPatValue(WorldPoint p)
{
	//return fGrid->GetPatValue(p);
	VelocityRec patValue = {1.,1.};
	return patValue;
}

Boolean ADCPMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	Boolean useEddyUncertainty = false;	
	double spillStartDepth = 0.;
	
	if (moverMap->IAm(TYPE_PTCURMAP))
		spillStartDepth = /*OK*/(dynamic_cast<PtCurMap *>(moverMap))->GetSpillStartDepth();
	
	wp.z = spillStartDepth;
	
	//velocity = this->GetPatValue(wp.p);
	velocity = GetVelocityAtPoint(wp);
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->refScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);
	return true;
	
}

OSErr ADCPMover_c::AddTimeDep(ADCPTimeValue *theTimeDep, short where)
{
	OSErr err = 0;
	if (err = timeDepList->AppendItem((Ptr)&theTimeDep))
	{ TechError("ADCPMover::AddTimeDep()", "AppendItem()", err); return err; }
	
	SelectListItemOfOwner(theTimeDep);
	
	return 0;
}

OSErr ADCPMover_c::DropTimeDep(ADCPTimeValue *theTimeDep)
{
	long i;
	OSErr err = 0;
	
	if (timeDepList->IsItemInList((Ptr)&theTimeDep, &i))
		if (err = timeDepList->DeleteItem(i))
		{ TechError("ADCPMover::DropTimeDep()", "DeleteItem()", err); return err; }
	
	return 0;
}
ADCPTimeValue*	ADCPMover_c::AddADCP(OSErr *err)
{	// might send in path for first adcp
	*err = 0;
	char tempStr[128], shortFileName[64], givenPath[256], givenFileName[64], s[256], fileName[64], path[256];
	short unitsIfKnownInAdvance = 0;
	Boolean askForFile = true;
	ADCPTimeValue *timeValObj = 0;
	
	Point where = CenteredDialogUpLeft(M38c);;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	
	//if(askForFile || !givenPath || !givenFileName)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return 0;
		strcpy(path, reply.fullPath);
#else
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) return 0;
		
		my_p2cstr(reply.fName);
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, path);
#else
		strcpy(path, reply.fName);
#endif
#endif		
		strcpy (s, path);
		SplitPathFile (s, fileName);
	}
	/*else
	 {	// don't ask user, we were provided with the path
	 strcpy(path,givenPath);
	 strcpy(fileName,givenFileName);
	 }*/
	
	if (IsADCPFile(path))
	{
		timeValObj = new ADCPTimeValue(dynamic_cast<ADCPMover *>(this));
		//timeDep = new ADCPTimeValue(this);
		
		if (!timeValObj)
		{ TechError("TextRead()", "new ADCPTimeValue()", 0); return nil; }
		
		*err = timeValObj->InitTimeFunc();
		if(*err) {delete timeValObj; timeValObj = nil; return nil;}  
		
		*err = timeValObj->ReadTimeValues2 (path, M19REALREAL, unitsIfKnownInAdvance);
		if(*err) { delete timeValObj; timeValObj = nil; return nil;}
		timeValObj->SetTimeFileName(fileName);
		//return timeValObj;
		//AddTimeDep(timeValObj,0);
	}	
	// code goes here, add code for OSSMHeightFiles, need scale factor to calculate derivative
	else
	{
		sprintf(tempStr,"File %s is not a recognizable ADCP time file.",shortFileName);
		printError(tempStr);
	}
	return timeValObj;
}

