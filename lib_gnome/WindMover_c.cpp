/*
 *  WindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifdef MAC
#ifdef MPW
#include <QDOffscreen.h>
#pragma SEGMENT WINDMOVER_C
#endif
#endif
#include "WindMover_c.h"
#include "MemUtils.h"
#include "GEOMETRY.H"
#include "CompFunctions.h"
#include "OUTILS.H"

#ifdef pyGNOME
#include "OSSMTimeValue_c.h"
#include "Map_c.h"
#include "LEList_c.h"
#include "Model_c.h"
#include "PtCurMap_c.h"
#include "Replacements.h"
#else
#include "CROSS.H"
#include "TOSSMTimeValue.h"
#include "TMap.h"
#endif

// Seconds gTapWindOffsetInSeconds = 0;	minus AH 06/20/2012

using std::cout;

WindMover_c::WindMover_c () { 
	
	timeDep = nil;
	
	fUncertainStartTime = 0;
	fDuration = 3*3600; // 3 hours
	
	fWindUncertaintyList = 0;
	fLESetSizes = 0;
	
	fSpeedScale = 2;
	fAngleScale = .4;
	fMaxSpeed = 30; //mps
	fMaxAngle = 60; //degrees
	fSigma2 =0;
	fSigmaTheta =  0; 
	//conversion = 1.0;// JLM , I think this field should be removed
	bUncertaintyPointOpen=false;
	bSubsurfaceActive = false;
	fGamma = 1.;
	breaking_wave_height = 1.;	// meters
	mixed_layer_depth = 10.;	// meters
	
	fIsConstantWind = false;
	fConstantValue.u = fConstantValue.v = 0.0;
	
	memset(&fWindBarbRect,0,sizeof(fWindBarbRect)); 
	bShowWindBarb = true;
	tap_offset = 0; // AH 06/20/2012
}

WindMover_c::WindMover_c(TMap *owner,char* name) : Mover_c(owner, name)
{
	if(!name || !name[0]) this->SetClassName("Variable Wind"); // JLM , a default useful in the wizard
	timeDep = nil;
	
	fUncertainStartTime = 0;
	fDuration = 3*3600; // 3 hours
	
	fWindUncertaintyList = 0;
	fLESetSizes = 0;
	
	fSpeedScale = 2;
	fAngleScale = .4;
	fMaxSpeed = 30; //mps
	fMaxAngle = 60; //degrees
	fSigma2 =0;
	fSigmaTheta =  0; 
	//conversion = 1.0;// JLM , I think this field should be removed
	bTimeFileOpen = FALSE;
	bUncertaintyPointOpen=false;
	bSubsurfaceActive = false;
	fGamma = 1.;
	breaking_wave_height = 1.;	// meters
	mixed_layer_depth = 10.;	// meters
	
	fIsConstantWind = false;
	fConstantValue.u = fConstantValue.v = 0.0;
	
	memset(&fWindBarbRect,0,sizeof(fWindBarbRect)); 
	bShowWindBarb = true;
	tap_offset = 0;		// AH 06/20/2012
}
void rndv(float *rndv1,float *rndv2)
{
	float cosArg = 2 * PI * GetRandomFloat(0.0,1.0);
	float srt = sqrt(-2 * log(GetRandomFloat(.001,.999)));
	
	*rndv1 = srt * cos(cosArg);
	*rndv2 = srt * sin(cosArg);
}

// Routine to check if random variables selected for
// the variable wind speed and direction are within acceptable
// limits. 0 means ok for angle and speed in anglekey and speedkey
static Boolean TermsLessThanMax(float cosTerm,float sinTerm,
								double speedMax,double angleMax,
								double sigma2, double sigmaTheta)
{
	//	float x,sqs;
	//	sqs = sqrt(speedMax*speedMax - sigma2);
	//	x = (speedMax-sqs) * cosTerm + sqrt(sqs);	
	// JLM 9/16/98  x and sqs were not being used
	return abs(sigmaTheta * sinTerm) <= angleMax;// && (x <= sqrt(3*speedMax));
}



void WindMover_c::DisposeUncertainty()
{
	fTimeUncertaintyWasSet = 0;
	
	if (fWindUncertaintyList)
	{
		DisposeHandle ((Handle) fWindUncertaintyList);
		fWindUncertaintyList = nil;
	}
	
	if (fLESetSizes)
	{
		DisposeHandle ((Handle) fLESetSizes);
		fLESetSizes = 0;
	}
}



void WindMover_c::UpdateUncertaintyValues(Seconds elapsedTime)
{
	long i,j,n;
	float cosTerm,sinTerm;
	
	fTimeUncertaintyWasSet = elapsedTime;
	
	if(!fWindUncertaintyList) return;
	
	n= _GetHandleSize((Handle)fWindUncertaintyList)/sizeof(LEWindUncertainRec);
	
	for(i=0;i<n;i++)
	{
		rndv(&cosTerm,&sinTerm);
		for(j=0;j<10;j++)
		{
			if(TermsLessThanMax(cosTerm,sinTerm,
								fMaxSpeed,fMaxAngle,fSigma2,fSigmaTheta))break;
			rndv(&cosTerm,&sinTerm);
		}
		
		(*fWindUncertaintyList)[i].randCos=cosTerm;
		(*fWindUncertaintyList)[i].randSin = sinTerm;
	}
}


#ifndef pyGNOME

OSErr WindMover_c::AllocateUncertainty()
{
	long i,j,n,numrec;
	TLEList *list;
	LEWindUncertainRecH h;
	OSErr err=0;
	CMyList	*LESetsList = model->LESetsList;
	
	this->DisposeUncertainty(); // get rid of any old values
	if(!LESetsList)return noErr;
	
	// code goes here, fMaxSpeed is unused !!!  JLM 9/29/98
	//if(timeDep)
	//{
	// may notbe called after user changes the wind speed.
	// this object should protect itself.
	//fMaxSpeed=timeDep->GetMaxValue(); 
	//if(fMaxSpeed == -1)fMaxSpeed = 30;
	//}
	
	
	n = LESetsList->GetItemCount();
	if(!(fLESetSizes = (LONGH)_NewHandle(sizeof(long)*n)))goto errHandler;
	
	for (i = 0,numrec=0; i < n ; i++) {
		(*fLESetSizes)[i]=numrec;
		LESetsList->GetListItem((Ptr)&list, i);
		if(list->GetLEType()==UNCERTAINTY_LE) // JLM 9/10/98
			numrec += list->GetLECount();
	}
	if(!(fWindUncertaintyList = 
		 (LEWindUncertainRecH)_NewHandle(sizeof(LEWindUncertainRec)*numrec)))goto errHandler;
	
	return noErr;
errHandler:
	this->DisposeUncertainty(); // get rid of any values allocated
	TechError("TWindMover::AllocateUncertainty()", "_NewHandle()", 0);
	return memFullErr;
}


OSErr WindMover_c::UpdateUncertainty(void)
{
	OSErr err = noErr;
	long i,n;
	Boolean needToReInit = false;
	Seconds elapsedTime =  model->GetModelTime() - model->GetStartTime();
	Boolean bAddUncertainty = (elapsedTime >= fUncertainStartTime) && model->IsUncertain();
	// JLM, this is elapsedTime >= fUncertainStartTime because elapsedTime is the value at the start of the step
	
	if(!bAddUncertainty)
	{	// we will not be adding uncertainty
		// make sure fWindUncertaintyList  && fLESetSizes are unallocated
		if(fWindUncertaintyList) this->DisposeUncertainty();
		return 0;
	}
	
	if(!fWindUncertaintyList || !fLESetSizes)
		needToReInit = true;
	
	if(elapsedTime < fTimeUncertaintyWasSet) 
	{	// the model reset the time without telling us
		needToReInit = true;
	}
	
	if(fLESetSizes)
	{	// check the LE sets are still the same, JLM 9/18/98
		TLEList *list;
		long numrec;
		n = model->LESetsList->GetItemCount();
		i = _GetHandleSize((Handle)fLESetSizes)/sizeof(long);
		if(n != i) needToReInit = true;
		else
		{
			for (i = 0,numrec=0; i < n ; i++) {
				if(numrec != (*fLESetSizes)[i])
				{
					needToReInit = true;
					break;
				}
				model->LESetsList->GetListItem((Ptr)&list, i);
				if(list->GetLEType()==UNCERTAINTY_LE) // JLM 9/10/98
					numrec += list->GetLECount();
			}
		}
		
	}
	
	
	// question JLM, should fSigma2 change only when the duration value is exceeded ??
	// or every step as it does now ??
	fSigma2 = fSpeedScale * .315 * pow(elapsedTime-fUncertainStartTime,.147);
	fSigma2 = fSigma2*fSigma2/2;
	//fSigmaTheta = fAngleScale * 2.73 * sqrt(sqrt(elapsedTime-fUncertainStartTime));
	fSigmaTheta = fAngleScale * 2.73 * sqrt(sqrt(double(elapsedTime-fUncertainStartTime)));
	
	if(needToReInit)
	{
		err = this->AllocateUncertainty();
		if(!err) this->UpdateUncertaintyValues(elapsedTime);
		if(err) return err;
	}
	else if(elapsedTime >= fTimeUncertaintyWasSet + fDuration) // we exceeded the persistance, time to update
	{	
		this->UpdateUncertaintyValues(elapsedTime);
	}
	return err;
}
#else

OSErr WindMover_c::UpdateUncertainty(void) { return 0; }

#endif	// AH 06/20/2012 (this does not affect stand alone behavior.

OSErr WindMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *patVel)
{
	VelocityRec tempV = *patVel;
	double sqs,m,dtheta,x,w,s,t,costheta,sintheta;
	double norm;
	LEWindUncertainRec unrec;
	OSErr err = 0;
	
	Boolean useOldCode = false; //OptionKeyDown(); // code goes here, take this out
	
	if(!fWindUncertaintyList || !fLESetSizes) 
		return 0; // this is our clue to not add uncertainty
	
	norm = sqrt(tempV.v*tempV.v + tempV.u*tempV.u);
	if(abs(norm) < 1e-6)
		return 0;
	
	unrec=(*fWindUncertaintyList)[(*fLESetSizes)[setIndex]+leIndex];
	w=norm;
	s = w*w-fSigma2;
	
	if(s >0)
	{
		sqs = sqrt(s);
		m = sqrt(sqs); 
	}
	else
	{
		sqs = 0;
		// code goes here, check this
		if(useOldCode) 
			m = sqrt(w); // old incorrect code
		else 
			m = 0; // new code
	}
	
	// code goes here
	if(useOldCode) 
		s = w-sqs;  // old incorrect code 
	else
		s = sqrt(w-sqs); // new code
	
	x = unrec.randCos*s + m;
	
	w = x*x;
	
	dtheta = unrec.randSin*fSigmaTheta*PI/180;
	costheta = cos(dtheta);
	sintheta = sin(dtheta);
	
	w = w/(costheta < .001 ? .001 : costheta); // compensate for projection vector effect
	
	// Scale pattern velocity to have norm w
	t=w/norm;
	tempV.u *= t;
	tempV.v *= t;
	
	// Rotate velocity by dtheta
	patVel->u = tempV.u * costheta - tempV.v * sintheta;
	patVel->v = tempV.v * costheta + tempV.u * sintheta;
	
	return err;
}

void WindMover_c::ClearWindValues() 
{ 	// have timedep throw away its list 
	// but don't delete timeDep
	if (timeDep)
	{
		timeDep -> Dispose();
	}
	// also clear the constant wind values
	fConstantValue.u = fConstantValue.v = 0.0;
}

void WindMover_c::DeleteTimeDep() 
{
	if (timeDep)
	{
		timeDep -> Dispose();
		delete timeDep;
		timeDep = nil;
	}
}

OSErr WindMover_c::PrepareForModelStep(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, const Seconds& time_step, bool uncertain)
{
	OSErr err = 0;
	if (uncertain)
		err = this->UpdateUncertainty();
	if (err) printError("An error occurred in TWindMover::PrepareForModelStep");
	return err;
}

OSErr WindMover_c::CheckStartTime (Seconds time)
{
	OSErr err = 0;
	if(fIsConstantWind) return -2;	// value is same for all time
	else
	{	// variable wind
		if (this -> timeDep)
		{
			err = timeDep -> CheckStartTime(time);
		}
	}
	return err;
}

OSErr WindMover_c::GetTimeValue(const Seconds& start_time, const Seconds& stop_time, const Seconds& current_time, VelocityRec *value)
{
	VelocityRec	timeValue = { 0, 0 };
	OSErr err = 0;
	// get and apply time file scale factor
	if(fIsConstantWind) timeValue = fConstantValue;
	else
	{	// variable wind
		if (this -> timeDep)
		{
			// note : constant wind always uses the first record
//			err = timeDep -> GetTimeValue(time, &timeValue);	// minus AH 07/10/2012
			err = timeDep -> GetTimeValue(start_time, stop_time, current_time, &timeValue);	// AH 07/10/2012
			
		}
	}
	*value = timeValue;
	return err;
}

OSErr WindMover_c::get_move(int n, long start_time, long stop_time, long model_time, long step_len, char *ref_ra, char *wp_ra, char *wind_ra, char *dispersion_ra, double f_sigma_2, double f_sigma_theta, double breaking_wave, double mix_layer, char *uncertain_ra, char* time_vals, int num_times) {	

// AH 06/20/2012:	
// unfortunately, because we determine the size of the handle by a small region at the base of the array,
// we have to recreate the container for any set of values passed, so that we're compatible with the existing
// _GetHandleSize() logic. maybe we can talk about this next meeting.
	

	TimeValuePairH time_val_hdl = 0;
	
	if(!uncertain_ra) {
		cout << "uncertainty values not provided! returning.\n";
		return 1;
	}

	if(!wp_ra) {
		cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	
	if(!time_vals) {
		cout << "time values array not provided! returning.\n";
		return 1;
	} else {
		if(num_times == 1) {
			fIsConstantWind = true;
			fConstantValue = ((TimeValuePair*)time_vals)->value;
		} else {
#ifdef pyGNOME
			try {
				time_val_hdl = (TimeValuePairH)_NewHandle(sizeof(TimeValuePair)*num_times);
				memcpy(*time_val_hdl, time_vals, sizeof(TimeValuePair)*num_times);
				timeDep = new OSSMTimeValue_c(this, time_val_hdl, kCMS);	// should we have to instantiate this every time we call the mover?
			} catch(...) {
				cout << "cannot create time values handle in windmover::get_move. returning.\n";
				if(time_val_hdl)
					_DisposeHandle((Handle)time_val_hdl);
				return 1;
			}
#endif
		}
	}
	// and so on.
	
	this->breaking_wave_height = breaking_wave;
	this->mixed_layer_depth = mix_layer;
	this->fSigma2 = f_sigma_2;
	this->fSigmaTheta = f_sigma_theta;
	this->tap_offset = 0;
	this->bSubsurfaceActive = true;

	try {
		this->fWindUncertaintyList = (LEWindUncertainRecH)_NewHandle(sizeof(LEWindUncertainRec)*n);
		memcpy(*fWindUncertaintyList, uncertain_ra, sizeof(LEWindUncertainRec)*n);
		this->fLESetSizes = (LONGH)_NewHandle(sizeof(long));
		DEREFH(this->fLESetSizes)[0] = 0;
	} catch(...) {
		cout << "cannot create uncertainty handle in windmover::get_move. returning.\n";
		if(this->fWindUncertaintyList)
			_DisposeHandle((Handle)this->fWindUncertaintyList);
		return 1;
	}
	
	WorldPoint3D delta;
	WorldPoint3D *ref;
	WorldPoint3D *wp;
	double *windages;
	short *disp_ra;
	ref = (WorldPoint3D*)ref_ra;
	wp = (WorldPoint3D*)wp_ra;
	windages = (double*)wind_ra;
	disp_ra = (short*)dispersion_ra;
	
	for (int i = 0; i < n; i++) {
		LERec rec;
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		rec.windage = windages[i];
		rec.dispersionStatus = disp_ra[i];
		
		delta = this->GetMove(start_time, stop_time, model_time, step_len, 0, i, &rec, UNCERTAINTY_LE);
		
		wp[i].p.pLat += delta.p.pLat / 1000000;
		wp[i].p.pLong += delta.p.pLong / 1000000;
		wp[i].z += delta.z;
	}
	if(timeDep)
		delete timeDep;
	if(time_val_hdl)
		_DisposeHandle((Handle)time_val_hdl);
	if(this->fLESetSizes)
		_DisposeHandle((Handle)this->fLESetSizes);
	if(this->fWindUncertaintyList)
		_DisposeHandle((Handle)this->fWindUncertaintyList);
	return noErr;
}

// AH 06/20/2012:
// maybe we really don't need to two functions at all.
// consider using variable length arguments..

// ++

OSErr WindMover_c::get_move(int n, long start_time, long stop_time, long model_time, long step_len, char *ref_ra, char *wp_ra, char *wind_ra, char *dispersion_ra, double breaking_wave, double mix_layer, char* time_vals, int num_times) {	
	
	// AH 06/20/2012:	
	// unfortunately, because we determine the size of the handle by a small region at the base of the array,
	// we have to recreate the container for any set of values passed, so that we're compatible with the existing
	// _GetHandleSize() logic. maybe we can talk about this next meeting.
	
	
	TimeValuePairH time_val_hdl = 0;
	
	if(!wp_ra) {
		cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	
	if(!time_vals) {
		cout << "time values array not provided! returning.\n";
		return 1;
	} else {
		if(num_times == 1) {
			fIsConstantWind = true;
			fConstantValue = ((TimeValuePair*)time_vals)->value;
		} else {
#ifdef pyGNOME
			try {
				time_val_hdl = (TimeValuePairH)_NewHandle(sizeof(TimeValuePair)*num_times);
				memcpy(*time_val_hdl, time_vals, sizeof(TimeValuePair)*num_times);
				timeDep = new OSSMTimeValue_c(this, time_val_hdl, kCMS);	// should we have to instantiate this every time we call the mover?
			} catch(...) {
				cout << "cannot create time values handle in windmover::get_move. returning.\n";
				if(time_val_hdl)
					_DisposeHandle((Handle)time_val_hdl);
				return 1;
			}
#endif
		}
	}
	// and so on.
	
	this->breaking_wave_height = breaking_wave;
	this->mixed_layer_depth = mix_layer;
	this->tap_offset = 0;
	this->bSubsurfaceActive = true;
	
	WorldPoint3D delta;
	WorldPoint3D *wp;
	WorldPoint3D *ref;
	double *windages;
	short *disp_ra;
	ref = (WorldPoint3D*)ref_ra;
	wp = (WorldPoint3D*)wp_ra;
	windages = (double*)wind_ra;
	disp_ra = (short*)dispersion_ra;
	
	for (int i = 0; i < n; i++) {
		LERec rec;
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		rec.windage = windages[i];
		rec.dispersionStatus = disp_ra[i];
		
		delta = this->GetMove(start_time, stop_time, model_time, step_len, 0, i, &rec, FORECAST_LE);
		
		wp[i].p.pLat += delta.p.pLat / 1000000;
		wp[i].p.pLong += delta.p.pLong / 1000000;
		wp[i].z += delta.z;
	}
	if(timeDep)
		delete timeDep;
	if(time_val_hdl)
		_DisposeHandle((Handle)time_val_hdl);
	return noErr;
}

// ..



WorldPoint3D WindMover_c::GetMove(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double 	dLong, dLat;
	VelocityRec	patVelocity, timeValue = { 0, 0 };
	WorldPoint3D	deltaPoint ={0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	OSErr err = noErr;
	
	
	// if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	if (((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT || (*theLE).z>0) && !bSubsurfaceActive) return deltaPoint;
	
	// code goes here, check LE type - plankton will have no windage
	
	// get and apply time file scale factor
	// code goes here, use some sort of average of past winds for dispersed oil
// 	err = this -> GetTimeValue (model_time + this->tap_offset,&timeValue);	 // minus AH 07/10/2012
	err = this -> GetTimeValue (start_time, stop_time, model_time + this->tap_offset,&timeValue);	// AH 07/10/2012
	
	if(err)  return deltaPoint;
	
	// separate algorithm for dispersed oil
	if ((*theLE).dispersionStatus==HAVE_DISPERSED || (*theLE).dispersionStatus==HAVE_DISPERSED_NAT)
	{
		//return deltaPoint;
		//code goes here, check if depth at point is less than mixed layer depth or breaking wave depth
		// shouldn't happen if other checks are done...
		double f=0, z = (*theLE).z,angle, u_mag; 
/*
		PtCurMap *map = GetPtCurMap();	// in theory should be moverMap, unless universal...
		if (!map) printError("Programmer error - TWindMover::GetWindageMove()");
		//breakingWaveHeight = map->GetBreakingWaveHeight();
		breakingWaveHeight = map->GetBreakingWaveHeight();
*/	// minus AH 06/20/2012
		
		// AH 06/20/2012: The value should have been computed by the time we've gotten here
	
		if (breaking_wave_height==0) breaking_wave_height = 1;	// need to have a default or give an error
		
		// mixedLayerDepth = map->fMixedLayerDepth;		minus AH 06/20/2012
		if (z<=fGamma*breaking_wave_height*1.5)
		{
			f = 2./3.;	// note, setting fGamma = 0 does not reduce subsurface windage effect
			// at this point only making it inactive will do the trick
		}
		else if (z<=mixed_layer_depth)
		{
			f = 2.*(1 - (log(z/(breaking_wave_height*1.5))/log(mixed_layer_depth/(breaking_wave_height*1.5))))/3.;
		}
		else
			f = 0.; // for depth dependent diffusion, z could get below mixed layer depth
		
		//angle = atan2(timeValue.v,timeValue.u);
		angle = atan2(	timeValue.u,timeValue.v); // measured from north
		
		//timeValue.u *= .03 * sin(angle) * f;
		//timeValue.v *= .03 * cos(angle) * f;
		u_mag = sqrt(timeValue.u*timeValue.u + timeValue.v*timeValue.v);
		timeValue.u = u_mag * .03 * sin(angle) * f;	// should use average wind
		timeValue.v = u_mag * .03 * cos(angle) * f;
		//timeValue.u = abs(timeValue.u) * .03 * sin(angle) * f;	// should use average wind
		//timeValue.v = abs(timeValue.v) * .03 * cos(angle) * f;
	}
	else
	{
		if(leType == UNCERTAINTY_LE)
		{
			err = AddUncertainty(setIndex,leIndex,&timeValue);
		}
		
		timeValue.u *=  (*theLE).windage;
		timeValue.v *=  (*theLE).windage;
	}
	
	dLong = ((timeValue.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat =   (timeValue.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}
