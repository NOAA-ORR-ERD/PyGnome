/*
 *  WindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 10/18/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "WindMover_c.h"
#include "MemUtils.h"
#include "GEOMETRY.H"
#include "CompFunctions.h"
//#include "OUTILS.H"

#ifdef pyGNOME
#include "OSSMTimeValue_c.h"
#include "Replacements.h"
#define TOSSMTimeValue OSSMTimeValue_c
#else
#include "CROSS.H"
#include "TOSSMTimeValue.h"
#include "TMap.h"
#endif

using std::cout;

WindMover_c::WindMover_c() : Mover_c()
{
	Init();
}

#ifndef pyGNOME
WindMover_c::WindMover_c(TMap *owner,char* name) : Mover_c(owner, name)
{
	if (!name || !name[0]) this->SetClassName("Variable Wind"); // JLM , a default useful in the wizard

	Init();	// initialize the local vars
}
#endif

void WindMover_c::Init()
{
	timeDep = nil;

	fModelStartTime = 0;
	fUncertainStartTime = 0;
	fDuration = 3*3600; // 3 hours

	fWindUncertaintyList = 0;
	fLESetSizes = 0;

	fSpeedScale = 2;
	fMaxSpeed = 30; //mps

	fAngleScale = .4;
	fMaxAngle = 60; //degrees

	fSigma2 =0;
	fSigmaTheta =  0; 
	fGamma = 1.;

	fUncertaintyDiffusion = 0;

	bTimeFileOpen = FALSE;
	bIsFirstStep = false;
	bUncertaintyPointOpen=false;
	bSubsurfaceActive = false;

	fIsConstantWind = false;
	fConstantValue.u = fConstantValue.v = 0.0;

	bShowWindBarb = true;

	memset(&fWindBarbRect, 0, sizeof(fWindBarbRect)); 
}


void WindMover_c::Dispose()
{
#ifndef pyGNOME
	// When invoked from python/cython, it OSSMTimeValue_c object is managed by cython
	// and cython also tries to delete it when the object goes out of scope. Since it tries
	// to delete an object that has already been deleted, it crashes. For pyGnome, let python/cython
	// manage memory for this object.
	DeleteTimeDep ();
#endif
	this->DisposeUncertainty();
	
	Mover_c::Dispose ();
}


void rndv(float *rndv1,float *rndv2)
{
	float cosArg = 2 * PI * GetRandomFloat(0.0, 1.0);
	float srt = sqrt(-2 * log(GetRandomFloat(.001, .999)));
	
	*rndv1 = srt * cos(cosArg);
	*rndv2 = srt * sin(cosArg);
}


// Routine to check if random variables selected for
// the variable wind speed and direction are within acceptable
// limits. 0 means ok for angle and speed in anglekey and speedkey
static Boolean TermsLessThanMax(float cosTerm, float sinTerm,
                                double speedMax, double angleMax,
                                double sigma2, double sigmaTheta)
{
	return fabs(sigmaTheta * sinTerm) <= angleMax;  // && (x <= sqrt(3*speedMax));
}


void WindMover_c::DisposeUncertainty()
{
	fTimeUncertaintyWasSet = 0;
	
	if (fWindUncertaintyList) {
		DisposeHandle ((Handle) fWindUncertaintyList);
		fWindUncertaintyList = nil;
	}
	
	if (fLESetSizes) {
		DisposeHandle ((Handle) fLESetSizes);
		fLESetSizes = 0;
	}
}


void WindMover_c::UpdateUncertaintyValues(Seconds elapsedTime)
{
	long i, j, n;
	float cosTerm, sinTerm;

	fTimeUncertaintyWasSet = elapsedTime;

	if (!fWindUncertaintyList) return;

	n = _GetHandleSize((Handle)fWindUncertaintyList) / sizeof(LEWindUncertainRec);

	for (i = 0; i < n; i++) {
		rndv(&cosTerm,&sinTerm);

		for(j = 0; j < 10; j++) {
			if (TermsLessThanMax(cosTerm,sinTerm,
								fMaxSpeed,fMaxAngle,fSigma2,fSigmaTheta)) break;

			rndv(&cosTerm,&sinTerm);
		}
		
		(*fWindUncertaintyList)[i].randCos = cosTerm;
		(*fWindUncertaintyList)[i].randSin = sinTerm;
	}
}


OSErr WindMover_c::ReallocateUncertainty(int numLEs, short* statusCodes)
{
    // remove off map LEs

	long i, numrec = 0, uncertListSize, numLESetsStored;
	//OSErr err = 0;

	if (numLEs == 0 || ! statusCodes) return -1;	// shouldn't happen

	if (!fWindUncertaintyList || !fLESetSizes) return 0;	// assume uncertainty is not on

	// check that (*fLESetSizesH)[0]==numLEs and size of fLESetSizesH == 1
	uncertListSize = _GetHandleSize((Handle)fWindUncertaintyList) / sizeof(LEWindUncertainRec);
	numLESetsStored = _GetHandleSize((Handle)fLESetSizes) / sizeof(long);

	if (uncertListSize != numLEs) return -1;

	if (numLESetsStored != 1) return -1;

	for (i = 0; i < numLEs ; i++) {
		if ( statusCodes[i] == OILSTAT_TO_BE_REMOVED) {
			// for OFF_MAPS, EVAPORATED, etc
			//continue;
		}
		else {
			(*fWindUncertaintyList)[numrec] = (*fWindUncertaintyList)[i];
			numrec++;
		}
	}

	if (numrec == 0) {
		this->DisposeUncertainty();
		return noErr;
	}

	if (numrec < uncertListSize) {
		//(*fLESetSizes)[0] = numrec;
		//(*fLESetSizes)[0] = 0;	// index into array, should never change
		_SetHandleSize((Handle)fWindUncertaintyList,
		               numrec * sizeof(LEWindUncertainRec)); 
	}

	return noErr;
}


OSErr WindMover_c::AllocateUncertainty(int numLESets, int* LESetsSizesList)
{
	// only passing in uncertainty list information
	long i, numrec = 0;
	//OSErr err = 0;

	this->DisposeUncertainty(); // get rid of any old values

	// shouldn't happen - if we get here there should be an uncertainty set,
	// unless there is no spill...
	//if (numLESets == 0) return -1;

	if (!(fLESetSizes = (LONGH)_NewHandle(sizeof(long)*numLESets))) goto errHandler;

	for (i = 0,numrec=0; i < numLESets ; i++) {
        // this is really storing an index to the fWindUncertaintyList
        (*fLESetSizes)[i] = numrec;
        numrec += LESetsSizesList[i];
	}

	if(!(fWindUncertaintyList = 
		 (LEWindUncertainRecH)_NewHandle(sizeof(LEWindUncertainRec)*numrec))) goto errHandler;

	return noErr;

errHandler:

	this->DisposeUncertainty(); // get rid of any values allocated

	TechError("TWindMover_c::AllocateUncertainty()", "_NewHandle()", 0);

	return memFullErr;
}


OSErr WindMover_c::UpdateUncertainty(const Seconds& elapsedTime,
                                     int numLESets,
                                     int *LESetsSizesList)
{
	OSErr err = noErr;
	long i, j;

	Boolean needToReInit = false, needToReAllocate = false;
	Boolean bAddUncertainty = (elapsedTime >= fUncertainStartTime);

    // JLM, this is elapsedTime >= fUncertainStartTime because elapsedTime
    // is the value at the start of the step

	if (!bAddUncertainty) {
		// we will not be adding uncertainty
		// make sure fWindUncertaintyList  && fLESetSizes are unallocated
		if (fWindUncertaintyList) this->DisposeUncertainty();

		return 0;
	}

	if (!fWindUncertaintyList || !fLESetSizes)
		needToReInit = true;

	if (elapsedTime < fTimeUncertaintyWasSet) {
		// the model reset the time without telling us
		needToReInit = true;
	}

	if (fLESetSizes) {
		// check the LE sets are still the same, JLM 9/18/98
		// code goes here, if LEs were added instead of needToReInit use needToReAllocate - save uncertainty if duration has not been exceeded
		long numrec, uncertListSize = 0, numLESetsStored;

		numLESetsStored = _GetHandleSize((Handle)fLESetSizes) / sizeof(long);

		if (numLESets != numLESetsStored) {
		    needToReInit = true;
		}
		else {
			for (i = 0,numrec=0; i < numLESets ; i++) {
				if (numrec != (*fLESetSizes)[i]) {
					needToReInit = true;
					break;
				}
				numrec += LESetsSizesList[i];
			}

			uncertListSize = _GetHandleSize((Handle)fWindUncertaintyList) / sizeof(LEWindUncertainRec); 
			if (numrec != uncertListSize) {
			    // this should not happen for gui gnome
#ifdef pyGNOME
				if (numrec > uncertListSize)
					needToReAllocate = true;
				else 
					needToReInit = true;
#else
				needToReInit = true;
#endif
				//break;
			}// need to check
		}

		if (needToReAllocate) {
			// move to separate function, and probably should combine with 
			float cosTerm, sinTerm;

			_SetHandleSize((Handle)fWindUncertaintyList, numrec * sizeof(LEWindUncertainRec));

			//for pyGNOME there should only be one uncertainty spill so fLESetSizes has only 1 value which is zero and doesn't need to be updated.
#ifdef pyGNOME
			if (numLESets != 1 || numLESetsStored != 1) {
			    printError("num uncertainty spills not equal 1\n");
			    return -1;
			}
#endif

			// this shouldn't happen
			if (needToReInit) printNote("Uncertainty arrays are being reset\n");

			// but would also need to update fSigmas - maybe move this section lower
			for (i = uncertListSize; i < numrec; i++) {
				rndv(&cosTerm,&sinTerm);

				for(j = 0; j < 10; j++) {
					if (TermsLessThanMax(cosTerm, sinTerm,
										fMaxSpeed, fMaxAngle,
										fSigma2, fSigmaTheta)) break;

					rndv(&cosTerm, &sinTerm);
				}
				
				(*fWindUncertaintyList)[i].randCos = cosTerm;
				(*fWindUncertaintyList)[i].randSin = sinTerm;
			}
		}
	}
	
	// question JLM, should fSigma2 change only when the duration value is exceeded ??
	// or every step as it does now ??
	fSigma2 = fSpeedScale * .315 * pow(elapsedTime - fUncertainStartTime, .147);
	fSigma2 = fSigma2 * fSigma2 / 2;

	fSigmaTheta = fAngleScale * 2.73 * sqrt(sqrt(double(elapsedTime - fUncertainStartTime)));
	
	if (needToReInit) {
		err = this->AllocateUncertainty(numLESets, LESetsSizesList);

		if (!err) this->UpdateUncertaintyValues(elapsedTime);

		if (err) return err;
	}
	else if (elapsedTime >= fTimeUncertaintyWasSet + fDuration) {
        // we exceeded the persistance, time to update
        this->UpdateUncertaintyValues(elapsedTime);
	}

	return err;
}


OSErr WindMover_c::AddUncertainty(long setIndex, long leIndex,
                                  VelocityRec *patVel)
{
	OSErr err = 0;

	double sqs, m, dtheta, x, w, s, t, costheta, sintheta;
	double norm;
	float rand1,rand2/*, eddyDiffusion = 100000, value*/;

	VelocityRec tempV = *patVel;
	LEWindUncertainRec unrec;

	if (!fWindUncertaintyList || !fLESetSizes) 
		return 0; // this is our clue to not add uncertainty

	norm = sqrt(tempV.v * tempV.v + tempV.u * tempV.u);
	if (norm < 1) {
		// try some small diffusion rather than nothing 2/13/13
		rand1 = GetRandomFloat(-1.0, 1.0);
		rand2 = GetRandomFloat(-1.0, 1.0);

		patVel->u = tempV.u + fUncertaintyDiffusion * rand1;
		patVel->v = tempV.v + fUncertaintyDiffusion * rand2;

		return 0;
	}

	unrec = (*fWindUncertaintyList)[(*fLESetSizes)[setIndex] + leIndex];
	w = norm;
	s = w * w - fSigma2;

	if (s > 0) {
		sqs = sqrt(s);
		m = sqrt(sqs); 
	}
	else {
		sqs = 0;
		m = 0; // new code
	}

	s = sqrt(w - sqs);
	x = unrec.randCos * s + m;
	w = x * x;

	dtheta = unrec.randSin * fSigmaTheta * PI / 180;
	costheta = cos(dtheta);
	sintheta = sin(dtheta);

	w = w / (costheta < .001 ? .001 : costheta); // compensate for projection vector effect

	// Scale pattern velocity to have norm w
	t = w / norm;
	tempV.u *= t;
	tempV.v *= t;

	// Rotate velocity by dtheta
	patVel->u = tempV.u * costheta - tempV.v * sintheta;
	patVel->v = tempV.v * costheta + tempV.u * sintheta;

	return err;
}


void WindMover_c::ClearWindValues() 
{
 	// have timedep throw away its list 
	// but don't delete timeDep
	if (timeDep) {
		timeDep->Dispose();
	}

	// also clear the constant wind values
	fConstantValue.u = fConstantValue.v = 0.0;
}


void WindMover_c::DeleteTimeDep() 
{
	if (timeDep) {
		timeDep->Dispose();
		delete timeDep;
		timeDep = nil;
	}
}


OSErr WindMover_c::PrepareForModelRun()
{
	bIsFirstStep = true;
	this->DisposeUncertainty();

	return noErr;
}


OSErr WindMover_c::PrepareForModelStep(const Seconds& model_time,
                                       const Seconds& time_step,
                                       bool uncertain,
                                       int numLESets,
                                       int* LESetsSizesList)
{
	OSErr err = 0;

	if (bIsFirstStep)
		fModelStartTime = model_time;

	if (uncertain) {
        	// so uncertainty starts at time zero + uncertain_time_delay,
        	// rather than a time step later
        Seconds elapsed_time = model_time + time_step - fModelStartTime;

		double eddyDiffusion = 1000000;

		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);

        // in m/s, note: DIVIDED by timestep because this is later multiplied
        // by the timestep
		fUncertaintyDiffusion = sqrt(6 * (eddyDiffusion / 10000) / time_step);
	}

	err = this->GetTimeValue(model_time, &this->current_time_value);	
#ifndef pyGNOME
	if (err) printError("An error occurred in TWindMover::PrepareForModelStep");
#endif

	return err;
}


OSErr WindMover_c::GetTimeValue(const Seconds &current_time,
                                VelocityRec *value)
{
	VelocityRec	timeValue = { 0, 0 };
	OSErr err = 0;

    // get and apply time file scale factor
    if (fIsConstantWind) {
        timeValue = fConstantValue;
    }
    else if (this->timeDep)
    {
        // variable wind
        // note : constant wind always uses the first record
        err = timeDep->GetTimeValue(current_time, &timeValue);	// minus AH 07/10/2012
	}

	*value = timeValue;

	return err;
}


void WindMover_c::ModelStepIsDone()
{
	bIsFirstStep = false;
}


OSErr WindMover_c::CheckStartTime (Seconds time)
{
	OSErr err = 0;

    if (fIsConstantWind) {
        // value is same for all time
        return -2;
    }
	else if (this->timeDep) {
        // variable wind
        err = timeDep->CheckStartTime(time);
	}

	return err;
}


// JS 10/8/12: Updated so the input arguments are not char * 
// NOTE: Some of the input arrays (ref, windages) should be const since you don't want the method to change them;
// however, haven't gotten const to work well with cython yet so just be careful when changing the input data
OSErr WindMover_c::get_move(int n, Seconds model_time, Seconds step_len,
                            WorldPoint3D* ref, WorldPoint3D* delta,
                            double* windages,
                            short* LE_status, LEType spillType, long spill_ID)
{
	// JS Ques: Is this required? Could cy/python invoke this method without well defined numpy arrays?
	if (!delta || !ref || !windages) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}

	// For LEType spillType, check to make sure it is within the valid values
	if ( spillType < FORECAST_LE || spillType > UNCERTAINTY_LE) {
		// cout << "Invalid spillType.\n";
		return 2;
	}

	LERec* prec;
	LERec rec;
	prec = &rec;

	WorldPoint3D zero_delta ={{0,0},0.};

	for (int i = 0; i < n; i++) {
		// only operate on LE if the status is in water
		if ( LE_status[i] != OILSTAT_INWATER) {
			delta[i] = zero_delta;
			continue;
		}

		rec.p = ref[i].p;
		rec.z = ref[i].z;
		rec.windage = windages[i];	// define the windage for the current LE

		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	// really only need this for the latitude
		//rec.p.pLong*= 1000000;

		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);

		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}

	return noErr;
}


WorldPoint3D WindMover_c::GetMove(const Seconds& model_time, Seconds timeStep,
                                  long setIndex, long leIndex,
                                  LERec *theLE, LETYPE leType)
{
	OSErr err = noErr;

	double 	dLong, dLat;

	VelocityRec	timeValue = { 0, 0 };
	WorldPoint3D	deltaPoint ={{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	

	if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface

	timeValue.u = this->current_time_value.u;	// AH 07/16/2012
	timeValue.v = this->current_time_value.v;

	if (err)  return deltaPoint;

	if (leType == UNCERTAINTY_LE) {
		err = AddUncertainty(setIndex,leIndex,&timeValue);
	}

	timeValue.u *= (*theLE).windage;
	timeValue.v *= (*theLE).windage;

	dLong = ((timeValue.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3(refPoint.pLat);
	dLat = (timeValue.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;

	return deltaPoint;
}


void WindMover_c::SetTimeDep (TOSSMTimeValue *newTimeDep) 
{ 
	timeDep = newTimeDep;
}


WindMover_c::~WindMover_c()
{
	 Dispose ();
}
