/*
 *  CATSMover3D_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/29/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "CATSMover3D_c.h"
#include "StringFunctions.h"
#include "PtCurMap_c.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

PtCurMap* GetPtCurMap(void);

/*OSErr CATSMover3D_c::ComputeVelocityScale()
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
 TCATSMover3D *mover;
 
 
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
 if (mover -> GetClassID() != TYPE_CATSMOVER) continue;
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
 
 OSErr CATSMover3D_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty)
 {
 /// 5/12/99 only add the eddy uncertainty when told to
 
 double u,v,lengthS,alpha,beta,gamma,v0,gammaScale;
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
 {	// when lengthS is too small , ignore the downstream and cross stream and only use diffusion uncertainty	
 if(useEddyUncertainty) { // provided we are supposed to
 patVelocity->u = this -> fOptimize.value * rand1;
 patVelocity->v = this -> fOptimize.value * rand2;
 }
 }
 }
 else 
 {
 TechError("TCATSMover3D::AddUncertainty()", "fUncertaintyListH==nil", 0);
 patVelocity->u=patVelocity->v=0;
 }
 return err;
 }*/



OSErr CATSMover3D_c::PrepareForModelStep(const Seconds& model_time, const Seconds& start_time, const Seconds& time_step, bool uncertain)

{
	OSErr err =0;
	CurrentMover_c::PrepareForModelStep(model_time, start_time, time_step, uncertain); // note: this calls UpdateUncertainty()
	
	err = this -> ComputeVelocityScale();// JLM, will this do it ???
	
	this -> fOptimize.isOptimizedForStep = true;
	this -> fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/time_step); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	this -> fOptimize.isFirstStep = (model_time == start_time);
	
	if (this -> fOptimize.isFirstStep)
	{
		PtCurMap* ptCurMap = dynamic_cast<PtCurMap *>(moverMap);
		/*OK*/(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
		/*OK*/(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
		((dynamic_cast<TTriGridVel3D*>(fGrid)))->ClearOutputHandles();
		if (fRefinedGrid) fRefinedGrid->ClearOutputHandles();
	}
	
	return err;
}

/*void CATSMover3D_c::ModelStepIsDone()
 {
 memset(&fOptimize,0,sizeof(fOptimize));
 }
 
 
 WorldPoint3D CATSMover3D_c::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
 {
 Boolean useEddyUncertainty = false;	
 double 		dLong, dLat;
 WorldPoint3D	deltaPoint={0,0,0.};
 WorldPoint refPoint = (*theLE).p;	
 VelocityRec scaledPatVelocity = this->GetScaledPatValue(refPoint,&useEddyUncertainty);
 if(leType == UNCERTAINTY_LE)
 {
 AddUncertainty(setIndex,leIndex,&scaledPatVelocity,timeStep,useEddyUncertainty);
 }
 dLong = ((scaledPatVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
 dLat  =  (scaledPatVelocity.v / METERSPERDEGREELAT) * timeStep;
 
 deltaPoint.p.pLong = dLong * 1000000;
 deltaPoint.p.pLat  = dLat  * 1000000;
 
 return deltaPoint;
 }
 
 VelocityRec CATSMover3D_c::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
 {
 /// 5/12/99 JLM, we only add the eddy uncertainty when the vectors are big enough when the timeValue is 1 
 // This is in response to the Prince William sound problem where 5 patterns are being added together
 VelocityRec	patVelocity, timeValue = {1, 1};
 TOSSMTimeValue*  scalingTimeDep = nil;
 float lengthSquaredBeforeTimeFactor;
 
 if(!this -> fOptimize.isOptimizedForStep && this->scaleType == SCALE_OTHERGRID) 
 {	// we need to update refScale
 this -> ComputeVelocityScale();
 }
 
 // get and apply our time file scale factor
 if (timeDep && bTimeFileActive)
 {
 // VelocityRec errVelocity={1,1};
 // JLM 11/22/99, if there are no time file values, use zero not 1
 VelocityRec errVelocity={0,1}; 
 err = timeDep -> GetTimeValue (model -> GetModelTime(), &timeValue); 
 if(err) timeValue = errVelocity;
 }
 
 patVelocity = GetPatValue (p);
 //	patVelocity = GetSmoothVelocity (p);
 
 patVelocity.u *= refScale; 
 patVelocity.v *= refScale; 
 
 if(useEddyUncertainty)
 { // if they gave us a pointer to a boolean fill it in, otherwise don't
 lengthSquaredBeforeTimeFactor = patVelocity.u*patVelocity.u + patVelocity.v*patVelocity.v;
 if(lengthSquaredBeforeTimeFactor < (this -> fEddyV0 * this -> fEddyV0)) *useEddyUncertainty = false; 
 else *useEddyUncertainty = true;
 }
 
 patVelocity.u *= timeValue.u; // magnitude contained in u field only
 patVelocity.v *= timeValue.u; // magnitude contained in u field only
 
 return patVelocity;
 }
 
 
 VelocityRec CATSMover3D_c::GetPatValue(WorldPoint p)
 {
 return fGrid->GetPatValue(p);	// don't use refined grid for movement
 }
 
 VelocityRec CATSMover3D_c::GetSmoothVelocity (WorldPoint p)
 {
 return fGrid->GetSmoothVelocity(p); // don't use refined grid for movement
 }
 */
Boolean CATSMover3D_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],verStr[32];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	long segNo,verNum;
	PtCurMap *map = GetPtCurMap();
	if (map==nil) return false;
	
	((PtCurMap_c*)map)->FindNearestBoundary(wp.p,&verNum,&segNo); //AH 03/19/2012 (this needs to be fixed.)
	
	velocity = this->GetPatValue(wp);
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->refScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	if(verNum > -1) 
	{
		StringWithoutTrailingZeros(verStr,verNum,4);
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s, ptNum: %s]",
				this->className, uStr, sStr, verStr);
	}
	else
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
				this->className, uStr, sStr);
	return true;
	
}


/*void CATSMover3D_c::DeleteTimeDep () 
 {
 if (timeDep)
 {
 timeDep -> Dispose ();
 delete timeDep;
 timeDep = nil;
 }
 
 return;
 }*/

//LongPointHdl CATSMover3D_c::GetPointsHdl(Boolean useRefinedGrid)
LongPointHdl CATSMover3D_c::GetPointsHdl()
{
	TTriGridVel3D* triGrid;
	//if (fRefinedGrid && useRefinedGrid) 
	//triGrid = fRefinedGrid;
	//else 
	triGrid = dynamic_cast<TTriGridVel3D*>(fGrid);
	return triGrid -> GetPointsHdl();
}

