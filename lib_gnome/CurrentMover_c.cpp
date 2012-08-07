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

#ifdef pyGNOME
OSErr CurrentMover_c::AllocateUncertainty(void) { return 0; }
#else
OSErr CurrentMover_c::AllocateUncertainty()
{
	long i,j,n,numrec;
	//TOLEList *list; AH 04/12/2012: This should be the more basic list type. (Every time we call on AppendItem() we're passing TLELists, not TOLELists. Please correct me if I'm wrong here.)
	TLEList *list;
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
#endif

#ifdef pyGNOME
OSErr CurrentMover_c::UpdateUncertainty(void) { return 0; }
#else
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
#endif

OSErr CurrentMover_c::PrepareForModelRun()
{
	return noErr;
}

OSErr CurrentMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain)
{
	OSErr err = 0;
	if (uncertain)
		err = this->UpdateUncertainty();
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