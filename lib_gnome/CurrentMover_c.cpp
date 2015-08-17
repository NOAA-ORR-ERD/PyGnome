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
#include "Replacements.h"
#else
#include "CROSS.H"
#endif

#ifndef pyGNOME
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
#endif
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


void CurrentMover_c::Dispose ()
{
	
	this->DisposeUncertainty();
	
	Mover_c::Dispose ();
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

OSErr CurrentMover_c::ReallocateUncertainty(int numLEs, short* statusCodes)	// remove off map LEs
{
	long i,numrec=0,uncertListSize,numLESetsStored;
	OSErr err=0;
	
	if (numLEs == 0 || ! statusCodes) return -1;	// shouldn't happen
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0;	// assume uncertainty is not on

	// check that (*fLESetSizesH)[0]==numLEs and size of fLESetSizesH == 1
	uncertListSize = _GetHandleSize((Handle)fUncertaintyListH)/sizeof(LEUncertainRec);
	numLESetsStored = _GetHandleSize((Handle)fLESetSizesH)/sizeof(long);
	
	if (uncertListSize != numLEs) return -1;
	if (numLESetsStored != 1) return -1;
	
	for (i = 0; i < numLEs ; i++) {
		if( statusCodes[i] == OILSTAT_TO_BE_REMOVED)	// for OFF_MAPS, EVAPORATED, etc
		{
			continue;
		}
		else {
			(*fUncertaintyListH)[numrec] = (*fUncertaintyListH)[i];
			numrec++;
		}
	}
	
	if (numrec == 0)
	{	
		this->DisposeUncertainty();
		return noErr;
	}
	
	if (numrec < uncertListSize)
	{
		//(*fLESetSizesH)[0] = numrec;
		//(*fLESetSizesH)[0] = 0;	// index into array, should never change
		_SetHandleSize((Handle)fUncertaintyListH,numrec*sizeof(LEUncertainRec)); 
	}
	
	return noErr;
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
	Boolean needToReInit = false, needToReAllocate = false;
	
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
		//TLEList *list;
		long numrec, uncertListSize = 0, numLESetsStored;;
		numLESetsStored = _GetHandleSize((Handle)fLESetSizesH)/sizeof(long);
		if(numLESets != numLESetsStored) needToReInit = true;
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
			uncertListSize = _GetHandleSize((Handle)fUncertaintyListH)/sizeof(LEUncertainRec); 
			if (numrec != uncertListSize)// this should not happen for gui gnome
			{
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
		
		if (needToReAllocate)
		{	// move to separate function, and probably should combine with 
			//char errmsg[256] = "";
			_SetHandleSize((Handle)fUncertaintyListH, numrec*sizeof(LEUncertainRec));
			//sprintf(errmsg,"Num LEs to Allocate = %ld, previous Size = %ld\n",numrec,uncertListSize);
			//printNote(errmsg);
			//for pyGNOME there should only be one uncertainty spill so fLESetSizes has only 1 value which is zero and doesn't need to be updated.
#ifdef pyGNOME
			if (numLESets != 1 || numLESetsStored != 1) {printError("num uncertainty spills not equal 1\n"); return -1;}
#endif
			if (needToReInit) printNote("Uncertainty arrays are being reset\n");	// this shouldn't happen
			//if(elapsedTime >= fTimeUncertaintyWasSet + fDuration) // we exceeded the persistance, time to update - either update whole list or just add on
			for(i=uncertListSize;i<numrec;i++)
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
