/*
 *  CompoundMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "CompoundMover_c.h"
#include "MemUtils.h"
#include "CompFunctions.h"
#include "StringFunctions.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"

#endif

extern CMyList *sMapList;
extern TCompoundMap	*sSharedDialogCompoundMap;

CompoundMover_c::CompoundMover_c (TMap *owner, char *name) : CurrentMover_c (owner, name), Mover_c(owner, name)
{
	moverList = 0;
	
	bMoversOpen = TRUE;
	
	return;
}


Boolean CompoundMover_c::IAmA3DMover()
{
	long i,n;
	TMover *mover = 0;
	n = moverList->GetItemCount() ;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		if (mover->IAmA3DMover()) return true;
		//listLength += mover->GetListLength();
	}
	return false;
}

OSErr CompoundMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep)
{
	//probably don't need this since will use individual currents' uncertainties
	double u,v,lengthS,alpha,beta;
	LEUncertainRec unrec;
	
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	
	if(!fUncertaintyListH || !fLESetSizesH) 
		return 0; // this is our clue to not add uncertainty
	
	/*if(fUncertaintyListH && fLESetSizesH)
	 {
	 unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
	 lengthS = sqrt(patVelocity->u*patVelocity->u + patVelocity->v * patVelocity->v);
	 
	 u = patVelocity->u;
	 v = patVelocity->v;
	 
	 if(lengthS>1e-6) // so we don't divide by zero
	 {	
	 
	 alpha = unrec.downStream;
	 beta = unrec.crossStream;
	 
	 patVelocity->u = u*(1+alpha)+v*beta;
	 patVelocity->v = v*(1+alpha)-u*beta;	
	 }
	 }
	 else 
	 {
	 TechError("TCompoundMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
	 patVelocity->u=patVelocity->v=0;
	 }*/
	return err;
}

void CompoundMover_c::ModelStepIsDone()
{
	long i, n;
	TMover *mover;
	//memset(&fOptimize,0,sizeof(fOptimize));
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		mover->ModelStepIsDone();
	}
	
}

OSErr CompoundMover_c::PrepareForModelStep()
{
	char errmsg[256];
	OSErr err = 0;
	//OSErr err = TCurrentMover::PrepareForModelStep(); // note: this calls UpdateUncertainty()
	
	errmsg[0]=0;
	
	// code goes here, jump to done?
	//if (err) goto done;
	long i, n;
	TMover *mover;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) {
		moverList->GetListItem((Ptr)&mover, i);
		err = mover->PrepareForModelStep();
		if (err) goto done;
	}
	
	if (model->GetModelTime() == model->GetStartTime())	// first step
	{
		if (moverMap->IAm(TYPE_COMPOUNDMAP))
		{
			//TCompoundMap* compoundMap = (TCompoundMap*)moverMap;
			/*OK*/(dynamic_cast<TCompoundMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<TCompoundMap *>(moverMap))->fContourDepth1;	
			/*OK*/(dynamic_cast<TCompoundMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<TCompoundMap *>(moverMap))->fContourDepth2;	
			//if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)
			//((TTriGridVel3D*)fGrid)->ClearOutputHandles();	// this gets done by the individual movers
		}
	}
	//this -> fOptimize.isOptimizedForStep = true;
	//this -> fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	
done:
	if (err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TCompoundMover::PrepareForModelStep");
		printError(errmsg); 
	}
	return err;
}

OSErr CompoundMover_c::AddMover(TMover *theMover, short where)
{
	OSErr err = 0;
	if (!moverList) return -1;
	if (err = moverList->AppendItem((Ptr)&theMover))
	{ TechError("TCompoundMover::AddMover()", "AppendItem()", err); return err; }
#ifndef pyGNOME	
	SetDirty (true);
	
	SelectListItemOfOwner(theMover);
	//SelectListItemOfOwner(this);
#endif
	return 0;
}


WorldPoint3D CompoundMover_c::GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double 		dLat, dLong;
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	VelocityRec	finalVel = {0.,0.}, pat1Val,pat2Val;
	long	i, n;
	TMover *mover;	
	OSErr err = 0;
	char errmsg[256];
	
	
	// figure out which pattern has priority, see if LE is on the grid, check next,...
	//if (bMoversOpen)
	{
		n = moverList->GetItemCount() ;
		//listLength += n;
		for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
		{	// movers should be listed in priority order
			moverList->GetListItem((Ptr)&mover, i);
			//check if LE is on the mover's grid
			if (!mover -> IsActive ()) continue; // to next mover
			deltaPoint = mover->GetMove(timeStep,setIndex,leIndex,theLE,leType);
			if (deltaPoint.p.pLong == 0 && deltaPoint.p.pLat == 0 && deltaPoint.z == 0.)
				continue;
			else break;
			//listLength += mover->GetListLength();
		}
	}
	// any sort of scaling to add ??
	
	/*pat1Val = pattern1 -> GetPatValue (refPoint);
	 if (pattern2) pat2Val = pattern2 -> GetPatValue (refPoint);
	 else {pat2Val.u = pat2Val.v = 0;}
	 */
	
	/*if(leType == UNCERTAINTY_LE)
	 {
	 AddUncertainty(setIndex,leIndex,&finalVel,timeStep);
	 }
	 
	 dLong = ((finalVel.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	 dLat =   (finalVel.v / METERSPERDEGREELAT) * timeStep;
	 
	 deltaPoint.p.pLong = dLong * 1000000;
	 deltaPoint.p.pLat  = dLat  * 1000000;*/
	
	return deltaPoint;
}

Boolean CompoundMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32], sStr[32];
	double lengthU=0., lengthS=0.;
	long	i, n;
	TMover *mover;
	OSErr err = 0;
	Boolean bVel = false;
	
	n = moverList->GetItemCount() ;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order
		moverList->GetListItem((Ptr)&mover, i);
		//check if LE is on the mover's grid
		//deltaPoint = mover->GetMove(timeStep,setIndex,leIndex,&theLE,leType);
		//if (deltaPoint.p.pLong == 0 && deltaPoint.p.pLat == 0 && deltaPoint.z == 0.)
		//continue;
		bVel = mover->VelocityStrAtPoint(wp,diagnosticStr);
		if (bVel)
		{
			return true;
		}
		else continue;
	}
	// else return 0
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);
	
	return true;
	
}
LongPointHdl CompoundMover_c::GetPointsHdl()
{
	long i,n;
	TMover *mover = 0;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order - which one to use?
		moverList->GetListItem((Ptr)&mover, i);
		if (mover) return mover->GetPointsHdl();
	}
	return nil;
}
TTriGridVel* CompoundMover_c::GetGrid(Boolean wantRefinedGrid)
{
	long i,n;
	TMover *mover = 0;
	TTriGridVel* triGrid = 0;	
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order - which one to use?
		moverList->GetListItem((Ptr)&mover, i);
		if (mover)
		{
			/*OK*/triGrid = dynamic_cast<TTriGridVel*>(((dynamic_cast<NetCDFMover *>(mover)) -> fGrid));
			if (triGrid) return triGrid;
		}
		//return mover->GetPointsHdl();
	}
	return triGrid;
}
TTriGridVel3D* CompoundMover_c::GetGrid3D(Boolean wantRefinedGrid)
{
	long i,n;
	TMover *mover = 0;
	TTriGridVel3D* triGrid = 0;	
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order - which one to use?
		moverList->GetListItem((Ptr)&mover, i);
		if (mover)
		{
			/*OK*/triGrid = dynamic_cast<TTriGridVel3D*>(((dynamic_cast<NetCDFMover *>(mover)) -> fGrid));
			if (triGrid) return triGrid;
		}
		//return mover->GetPointsHdl();
	}
	return triGrid;
}

TTriGridVel3D* CompoundMover_c::GetGrid3DFromMoverIndex(long moverIndex)
{
	long i,n;
	TMover *mover = 0;
	TTriGridVel3D* triGrid = 0;	
	
	n = moverList->GetItemCount();
	if (moverIndex<0 || moverIndex>=n) return nil;
	//for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	//{	// movers should be listed in priority order - which one to use?
	moverList->GetListItem((Ptr)&mover, moverIndex);
	if (mover)
	{	// could be TCATS or PtCUr?
		/*OK*/triGrid = dynamic_cast<TTriGridVel3D*>(((dynamic_cast<NetCDFMover *>(mover)) -> fGrid));
		if (triGrid) return triGrid;
	}
	//return mover->GetPointsHdl();
	//}
	return triGrid;
}
TCurrentMover* CompoundMover_c::Get3DCurrentMover()
{
	TMover *thisMover = nil;
	long i,d;
	for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	{
		this -> moverList -> GetListItem ((Ptr) &thisMover, i);
		// might want to be specific since this could allow CATSMovers...
		if(thisMover -> IAm(TYPE_PTCURMOVER) || thisMover -> IAm(TYPE_TRICURMOVER) || thisMover -> IAm(TYPE_CATSMOVER3D)
		   || thisMover -> IAm(TYPE_NETCDFMOVERCURV) || thisMover -> IAm(TYPE_NETCDFMOVERTRI)) return dynamic_cast<TCurrentMover*>(thisMover);
	}
	return nil;
}

TCurrentMover* CompoundMover_c::Get3DCurrentMoverFromIndex(long moverIndex)
{
	TMover *thisMover = nil;
	long i,d= this -> moverList -> GetItemCount ();
	if (moverIndex < 0 || moverIndex >= d) return nil;
	//for (i = 0, d = this -> moverList -> GetItemCount (); i < d; i++)
	//{
	this -> moverList -> GetListItem ((Ptr) &thisMover, moverIndex);
	// might want to be specific since this could allow CATSMovers...
	if(thisMover -> IAm(TYPE_PTCURMOVER) || thisMover -> IAm(TYPE_TRICURMOVER) || thisMover -> IAm(TYPE_CATSMOVER3D)
	   || thisMover -> IAm(TYPE_NETCDFMOVERCURV) || thisMover -> IAm(TYPE_NETCDFMOVERTRI)) return dynamic_cast<TCurrentMover*>(thisMover);
	//}
	return nil;
}

float CompoundMover_c::GetArrowDepth()
{
	long i,n;
	float depth = 0.;
	TMover *mover = 0;
	for (i = 0, n = moverList->GetItemCount() ; i < n ; i++) 
	{	// movers should be listed in priority order
		moverList->GetListItem((Ptr)&mover, i);
		//should have separate arrowDepth for combined mover ?, otherwise which one to use?
		return mover->GetArrowDepth();
	}
	return depth;
}
