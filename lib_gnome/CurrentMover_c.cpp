/*
 *  CurrentMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/23/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "CurrentMover_c.h"
#include "CompFunctions.h"
#include "MemUtils.h"

#ifdef pyGNOME
#include "LEList_c.h"
#include "OLEList_c.h"
#include "Model_c.h"
#include "Replacements.h"
#else
#include "CROSS.H"
#include "TLEList.h"
#include "TOLEList.h"
#include "TModel.h"
extern TModel *model;
#endif

CurrentMover_c::CurrentMover_c (TMap *owner, char *name) : Mover_c(owner, name)
{
	// set fields of our base class
	fDuration=48*3600; //48 hrs as seconds 
	fUncertainStartTime = 0;
	fTimeUncertaintyWasSet = 0;
	
	fDownCurUncertainty = -.3;  // 30%
	fUpCurUncertainty = .3; 	
	fRightCurUncertainty = .1;  // 10%
	fLeftCurUncertainty= -.1; 
	
	fLESetSizesH = 0;
	fUncertaintyListH = 0;
	
	bIAmPartOfACompoundMover = false;
	bIAmA3DMover = false;
	
	bIsFirstStep = false;
	fModelStartTime = 0;
}

CurrentMover_c::CurrentMover_c () : Mover_c()
{
	// set fields of our base class
	fDuration=48*3600; //48 hrs as seconds 
	fUncertainStartTime = 0;
	fTimeUncertaintyWasSet = 0;
	
	fDownCurUncertainty = -.3;  // 30%
	fUpCurUncertainty = .3; 	
	fRightCurUncertainty = .1;  // 10%
	fLeftCurUncertainty= -.1; 
	
	fLESetSizesH = 0;
	fUncertaintyListH = 0;
	
	bIAmPartOfACompoundMover = false;
	bIAmA3DMover = false;
	
	bIsFirstStep = false;
	fModelStartTime = 0;
}


void CurrentMover_c::UpdateUncertaintyValues(Seconds elapsedTime)
{
	long i,n;
	
	fTimeUncertaintyWasSet = elapsedTime;
	
	if(!fUncertaintyListH) return;
	
	n = _GetHandleSize((Handle)fUncertaintyListH)/sizeof(**fUncertaintyListH);
	
	
	for(i=0;i<n;i++)
	{
		LEUncertainRec localCopy;
		memset(&localCopy,0,sizeof(LEUncertainRec));
		
		if(fDownCurUncertainty<fUpCurUncertainty)
		{
			localCopy.downStream = 
			GetRandomFloat(fDownCurUncertainty,fUpCurUncertainty);
		}
		else
		{
			localCopy.downStream = 
			GetRandomFloat(fUpCurUncertainty,fDownCurUncertainty);
		}
		if(fLeftCurUncertainty<fRightCurUncertainty)
		{
			localCopy.crossStream = 
			GetRandomFloat(fLeftCurUncertainty,fRightCurUncertainty);
		}
		else
		{
			localCopy.crossStream = 
			GetRandomFloat(fRightCurUncertainty,fLeftCurUncertainty);
		}
		INDEXH(fUncertaintyListH,i) = localCopy;
		
	}	
}

OSErr CurrentMover_c::AllocateUncertainty(int numLESets, int* LESetsSizesList)	// only passing in uncertainty list information
{
	long i,j,numrec=0;
	OSErr err=0;
	
	this->DisposeUncertainty(); // get rid of any old values
	
	if (numLESets == 0) return -1;	// shouldn't happen - if we get here there should be an uncertainty set
	
	if(!(fLESetSizesH = (LONGH)_NewHandle(sizeof(long)*numLESets)))goto errHandler;
	
	for (i = 0,numrec=0; i < numLESets ; i++) {
		(*fLESetSizesH)[i]=numrec;
		numrec += LESetsSizesList[i];
	}
	if(!(fUncertaintyListH = 
		 (LEUncertainRecH)_NewHandle(sizeof(LEUncertainRec)*numrec)))goto errHandler;
	
	return noErr;
errHandler:
	this->DisposeUncertainty(); // get rid of any values allocated
	TechError("TCurrentMover::AllocateUncertainty()", "_NewHandle()", 0);
	return memFullErr;
}

OSErr CurrentMover_c::UpdateUncertainty(const Seconds& elapsedTime, int numLESets, int* LESetsSizesList)
{
	OSErr err = noErr;
	long i;
	Boolean needToReInit = false;
	
	//Boolean bAddUncertainty = (elapsedTime >= fUncertainStartTime) && model->IsUncertain();
	Boolean bAddUncertainty = (elapsedTime >= fUncertainStartTime);
	// JLM, this is elapsedTime >= fUncertainStartTime because elapsedTime is the value at the start of the step
	
	if(!bAddUncertainty)
	{	// we will not be adding uncertainty
		// make sure fWindUncertaintyList  && fLESetSizesH are unallocated
		if(fUncertaintyListH) this->DisposeUncertainty();
		return 0;
	}
	
	if(!fUncertaintyListH || !fLESetSizesH)
		needToReInit = true;
	
	if(elapsedTime < fTimeUncertaintyWasSet) 
	{	// the model reset the time without telling us
		needToReInit = true;
	}
	
	if(fLESetSizesH)
	{	// check the LE sets are still the same, JLM 9/18/98
		TLEList *list;
		long numrec;
		i = _GetHandleSize((Handle)fLESetSizesH)/sizeof(long);
		if(numLESets != i) needToReInit = true;
		else
		{
			for (i = 0,numrec=0; i < numLESets ; i++) {
				if(numrec != (*fLESetSizesH)[i])
				{
					needToReInit = true;
					break;
				}
				numrec += LESetsSizesList[i];
			}
		}
		
	}
	
	if(needToReInit)
	{
		err = this->AllocateUncertainty(numLESets,LESetsSizesList);
		if(!err) this->UpdateUncertaintyValues(elapsedTime);
		if(err) return err;
	}
	else if(elapsedTime >= fTimeUncertaintyWasSet + fDuration) // we exceeded the persistance, time to update
	{	
		this->UpdateUncertaintyValues(elapsedTime);
	}
	
	return err;
}

OSErr CurrentMover_c::PrepareForModelRun()
{
	bIsFirstStep = true;
	this->DisposeUncertainty();
	return noErr;
}

OSErr CurrentMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	OSErr err = 0;
	if (bIsFirstStep)
		fModelStartTime = model_time;
	if (uncertain)
	{
		Seconds elapsed_time = model_time - fModelStartTime;	// code goes here, how to set start time
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}
	if (err) printError("An error occurred in TCurrentMover::PrepareForModelStep");
	return err;
}


void CurrentMover_c::DisposeUncertainty()
{
	fTimeUncertaintyWasSet = 0;
	
	if (fUncertaintyListH)
	{
		DisposeHandle ((Handle) fUncertaintyListH);
		fUncertaintyListH = nil;
	}
	
	if (fLESetSizesH)
	{
		DisposeHandle ((Handle) fLESetSizesH);
		fLESetSizesH = 0;
	}
}