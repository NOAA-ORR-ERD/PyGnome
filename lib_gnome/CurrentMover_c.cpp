/*
 *  CurrentMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/23/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "CurrentMover_c.h"
#include "Cross.h"
#include "GridVel.h"
#include "OUtils.h"
#include "Uncertainty.h"

#ifdef pyGNOME
#define TMap Map_c
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


OSErr CurrentMover_c::AllocateUncertainty()
{
	long i,j,n,numrec;
	TOLEList *list;
	LEUncertainRecH h;
	OSErr err=0;
	CMyList	*LESetsList = model->LESetsList;
	
	this->DisposeUncertainty(); // get rid of any old values
	if(!LESetsList)return noErr;
	
	n = LESetsList->GetItemCount();
	if(!(fLESetSizesH = (LONGH)_NewHandle(sizeof(long)*n)))goto errHandler;
	
	for (i = 0,numrec=0; i < n ; i++) {
		(*fLESetSizesH)[i]=numrec;
		LESetsList->GetListItem((Ptr)&list, i);
		if(list->GetLEType()==UNCERTAINTY_LE) // JLM 9/10/98
			numrec += list->GetLECount();
	}
	if(!(fUncertaintyListH = 
		 (LEUncertainRecH)_NewHandle(sizeof(LEUncertainRec)*numrec)))goto errHandler;
	
	return noErr;
errHandler:
	this->DisposeUncertainty(); // get rid of any values allocated
	TechError("TCurrentMover::AllocateUncertainty()", "_NewHandle()", 0);
	return memFullErr;
}

OSErr CurrentMover_c::UpdateUncertainty(void)
{
	OSErr err = noErr;
	long i,n;
	Boolean needToReInit = false;
	Seconds elapsedTime =  model->GetModelTime() - model->GetStartTime();
	
	
	Boolean bAddUncertainty = (elapsedTime >= fUncertainStartTime) && model->IsUncertain();
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
		n = model->LESetsList->GetItemCount();
		i = _GetHandleSize((Handle)fLESetSizesH)/sizeof(long);
		if(n != i) needToReInit = true;
		else
		{
			for (i = 0,numrec=0; i < n ; i++) {
				if(numrec != (*fLESetSizesH)[i])
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


OSErr CurrentMover_c::PrepareForModelStep()
{
	OSErr err = this->UpdateUncertainty();
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