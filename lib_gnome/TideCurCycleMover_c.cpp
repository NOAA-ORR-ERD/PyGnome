/*
 *  TideCurCycleMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TideCurCycleMover_c.h"
#include "TideCurCycleMover.h"
#include "CROSS.H"

#ifndef pyGNOME
extern TModel *model;
#else
extern Model_c *model;
#endif

LongPointHdl TideCurCycleMover_c::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

OSErr TideCurCycleMover_c::ComputeVelocityScale()
{	// this function computes and sets this->refScale
	// returns Error when the refScale is not defined
	// or in allowable range.  
	// Note it also sets the refScale to 0 if there is an error
#define MAXREFSCALE  1.0e6  // 1 million times is too much
#define MIN_UNSCALED_REF_LENGTH 1.0E-5 // it's way too small
	//long i, j, m, n;
	double length;
	//VelocityRec myVelocity;
	//TMap *map;
	//TCATSMover *mover;
	WorldPoint3D	deltaPoint = {0,0,0.};
	//WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long ptIndex1,ptIndex2,ptIndex3; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	/*	if (this->timeDep && !this->timeDep->fFileType==SHIOCURRENTSFILE)
	 {
	 this->refScale = 0;
	 return -1;
	 }*/
	
	
	this->refScale = 1;
	return 0;
	// probably don't need this code	
	switch (scaleType) {
		case SCALE_NONE: this->refScale = 1; return noErr;
		case SCALE_CONSTANT:
			if(!this -> fOptimize.isOptimizedForStep) 
			{
				err = /*CHECK*/dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg);
				if (err) 
				{
					this->refScale = 0;
					return err;	// set to zero, avoid any accidental access violation
				}
			}
			
			// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
			interpolationVal = fGrid -> GetInterpolationValues(refP);
			
			if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
			{
				// this is only section that's different from ptcur
				ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
				ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
				ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			}
			else
			{
				this->refScale = 0;
				return -1;	// set to zero, avoid any accidental access violation
			}
			
			// Check for constant current 
			if(/*OK*/dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()==1)
			{
				// Calculate the interpolated velocity at the point
				if (interpolationVal.ptIndex1 >= 0) 
				{
					scaledPatVelocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
					+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
					+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
					scaledPatVelocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
					+interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
					+interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);
				}
				else	// if negative corresponds to negative ntri, set vel to zero
				{
					scaledPatVelocity.u = 0.;
					scaledPatVelocity.v = 0.;
				}
			}
			else // time varying current 
			{
				// Calculate the time weight factor
				if (fTimeAlpha==-1)
				{
					Seconds relTime = time - model->GetStartTime();
					startTime = (*fTimeHdl)[fStartData.timeIndex];
					endTime = (*fTimeHdl)[fEndData.timeIndex];
					//timeAlpha = (endTime - time)/(double)(endTime - startTime);
					timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
				}
				else
					timeAlpha=fTimeAlpha;
				
				// Calculate the interpolated velocity at the point
				if (interpolationVal.ptIndex1 >= 0) 
				{
					scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
					+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
					+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
					scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
					+interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
					+interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);
				}
				else	// if negative corresponds to negative ntri, set vel to zero
				{
					scaledPatVelocity.u = 0.;
					scaledPatVelocity.v = 0.;
				}
			}
			
			//myVelocity = GetPatValue(refP);
			//length = sqrt(myVelocity.u * myVelocity.u + myVelocity.v * myVelocity.v);
			length = sqrt(scaledPatVelocity.u * scaledPatVelocity.u + scaledPatVelocity.v * scaledPatVelocity.v);
			/// check for too small lengths
			if(fabs(scaleValue) > length*MAXREFSCALE
			   || length < MIN_UNSCALED_REF_LENGTH)
			{ this->refScale = 0;return -1;} // unable to compute refScale
			this->refScale = scaleValue / length; 
			return noErr;
		default:
			break;
	}
	
	this->refScale = 0;
	return -1;
}

OSErr TideCurCycleMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	err = this -> UpdateUncertainty();
	if(err) return err;
	
	if(!fUncertaintyListH || !fLESetSizesH) return 0; // this is our clue to not add uncertainty
	
	
	if(fUncertaintyListH && fLESetSizesH)
	{
		unrec=(*fUncertaintyListH)[(*fLESetSizesH)[setIndex]+leIndex];
		lengthS = sqrt(velocity->u*velocity->u + velocity->v * velocity->v);
		
		
		u = velocity->u;
		v = velocity->v;
		
		//if(lengthS < fVar.uncertMinimumInMPS)
		if(lengthS < fEddyV0)	// reusing the variable for now...
		{
			// use a diffusion  ??
			printError("nonzero UNCERTMIN is unimplemented");
			//err = -1;
		}
		else
		{	// normal case, just use cross and down stuff
			alpha = unrec.downStream;
			beta = unrec.crossStream;
			
			velocity->u = u*(1+alpha)+v*beta;
			velocity->v = v*(1+alpha)-u*beta;	
		}
	}
	else 
	{
		TechError("TideCurCycleMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}

OSErr TideCurCycleMover_c::PrepareForModelStep()
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	//check to see that the time interval is loaded and set if necessary
	if (!bActive) return noErr;
	err = /*CHECK*/dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg);
	if(err) goto done;
	
	fOptimize.isOptimizedForStep = true;	// don't  use CATS eddy diffusion stuff, follow ptcur
	//fOptimize.value = sqrt(6*(fEddyDiffusion/10000)/model->GetTimeStep()); // in m/s, note: DIVIDED by timestep because this is later multiplied by the timestep
	//fOptimize.isFirstStep = (model->GetModelTime() == model->GetStartTime());
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TideCurCycleMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	return err;
}

void TideCurCycleMover_c::ModelStepIsDone()
{
	fOptimize.isOptimizedForStep = false;
}


Boolean TideCurCycleMover_c::IsDryTriangle(long index1, long index2, long index3, float timeAlpha)
{
	Boolean isDry = false;
	VelocityRec vel1, vel2, vel3;
	// check timeAlpha = -1 (no time file, using fake times), 1 (start only), 0 (end only)
	if (timeAlpha == 1)
	{
		vel1.u = INDEXH(fStartData.dataHdl,index1).u;
		vel2.u = INDEXH(fStartData.dataHdl,index2).u;
		vel3.u = INDEXH(fStartData.dataHdl,index3).u;
		if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
			return true;
	}
	else if (timeAlpha == 0)
	{
		vel1.u = INDEXH(fEndData.dataHdl,index1).u;
		vel2.u = INDEXH(fEndData.dataHdl,index2).u;
		vel3.u = INDEXH(fEndData.dataHdl,index3).u;
		if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
			return true;
	}
	else if (timeAlpha > 0 && timeAlpha < 1)
	{
		vel1.u = INDEXH(fStartData.dataHdl,index1).u;
		vel2.u = INDEXH(fStartData.dataHdl,index2).u;
		vel3.u = INDEXH(fStartData.dataHdl,index3).u;
		if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
		{
			vel1.u = INDEXH(fEndData.dataHdl,index1).u;
			vel2.u = INDEXH(fEndData.dataHdl,index2).u;
			vel3.u = INDEXH(fEndData.dataHdl,index3).u;
			if (vel1.u == fDryValue && vel2.u == fDryValue && vel3.u == fDryValue)
				return true;
		}
		else
			return false;
	}
	return isDry;
}

Boolean TideCurCycleMover_c::IsDryTri(long triIndex)
{
	Boolean isDry = false;
	long index1, index2, index3, ptIndex1, ptIndex2, ptIndex3;
	double timeAlpha = 1;
	Seconds time = model->GetModelTime(), startTime, endTime;
	TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	TopologyHdl topH = triGrid->GetTopologyHdl();
	
	index1 = (*topH)[triIndex].vertex1;
	index2 = (*topH)[triIndex].vertex2;
	index3 = (*topH)[triIndex].vertex3;
	ptIndex1 =  (*fVerdatToNetCDFH)[index1];	
	ptIndex2 =  (*fVerdatToNetCDFH)[index2];
	ptIndex3 =  (*fVerdatToNetCDFH)[index3];
	
	if(/*OK*/dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()>1)
	{
		// Calculate the time weight factor
		if (fTimeAlpha==-1)
		{
			Seconds relTime = time - model->GetStartTime();
			startTime = (*fTimeHdl)[fStartData.timeIndex];
			endTime = (*fTimeHdl)[fEndData.timeIndex];
			//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
		}
		else
			timeAlpha = fTimeAlpha;
	}
	
	isDry = IsDryTriangle(ptIndex1, ptIndex2, ptIndex3, timeAlpha);
	return isDry;
}

VelocityRec TideCurCycleMover_c::GetStartVelocity(long index, Boolean *isDryPt)
{
	VelocityRec vel = {0,0};
	*isDryPt = false;
	if (index>=0)
	{
		vel.u = INDEXH(fStartData.dataHdl,index).u;
		vel.v = INDEXH(fStartData.dataHdl,index).v;
	}
	if (vel.u == fDryValue || vel.v == fDryValue) 
	{
		*isDryPt = true; 
		vel.u = 0; 
		vel.v = 0;
	}
	return vel;
}

VelocityRec TideCurCycleMover_c::GetEndVelocity(long index, Boolean *isDryPt)
{
	VelocityRec vel = {0,0};
	*isDryPt = false;
	if (index>=0)
	{
		vel.u = INDEXH(fEndData.dataHdl,index).u;
		vel.v = INDEXH(fEndData.dataHdl,index).v;
	}
	if (vel.u == fDryValue || vel.v == fDryValue) 
	{
		*isDryPt = true; 
		vel.u = 0; 
		vel.v = 0;
	}
	return vel;
}

double TideCurCycleMover_c::GetStartUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		u = INDEXH(fStartData.dataHdl,index).u;
	}
	if (u == fDryValue) {u = 0;}
	return u;
}

double TideCurCycleMover_c::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		u = INDEXH(fEndData.dataHdl,index).u;
	}
	if (u == fDryValue) {u = 0;}
	return u;
}

double TideCurCycleMover_c::GetStartVVelocity(long index)
{
	double v = 0;
	if (index>=0)
	{
		v = INDEXH(fStartData.dataHdl,index).v;
	}
	if (v == fDryValue) {v = 0;}
	return v;
}

double TideCurCycleMover_c::GetEndVVelocity(long index)
{
	double v = 0;
	if (index>=0)
	{
		v = INDEXH(fEndData.dataHdl,index).v;
	}
	if (v == fDryValue) {v = 0;}
	return v;
}


WorldPoint3D TideCurCycleMover_c::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	// see PtCurMover::GetMove - will depend on what is in netcdf files and how it's stored
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long ptIndex1,ptIndex2,ptIndex3; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity, timeValue = {1, 1};
	Boolean useEddyUncertainty = false, isDry = false;	
	OSErr err = 0;
	char errmsg[256];
	
	(*theLE).leCustomData = 0;
	
	if(!this -> fOptimize.isOptimizedForStep) 
	{
		err = /*CHECK*/dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	interpolationVal = fGrid -> GetInterpolationValues(refPoint);
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
	}
	else
		return deltaPoint;	// set to zero, avoid any accidental access violation
	
	// Check for constant current 
	if(/*OK*/dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()==1)
	{
		// check if all 3 values are dry and if so set LE.customdata = -1;, else 0;
		// set u,v=0 for dry values, on return check if customdata=-1 and then don't move at all this step
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,1);
			if (isDry) 
			{ 
				(*theLE).leCustomData = -1;
				return deltaPoint;
			}
			scaledPatVelocity.u = interpolationVal.alpha1*GetStartUVelocity(ptIndex1)
			+interpolationVal.alpha2*GetStartUVelocity(ptIndex2)
			+interpolationVal.alpha3*GetStartUVelocity(ptIndex3);
			scaledPatVelocity.v = interpolationVal.alpha1*GetStartVVelocity(ptIndex1)
			+interpolationVal.alpha2*GetStartVVelocity(ptIndex2)
			+interpolationVal.alpha3*GetStartVVelocity(ptIndex3);
			/*scaledPatVelocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
			 +interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
			 +interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			 scaledPatVelocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
			 +interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
			 +interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);*/
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (fTimeAlpha==-1)
		{
			Seconds relTime = time - model->GetStartTime();
			startTime = (*fTimeHdl)[fStartData.timeIndex];
			endTime = (*fTimeHdl)[fEndData.timeIndex];
			//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
		}
		else
			timeAlpha = fTimeAlpha;
		
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,timeAlpha);
			if (isDry) 
			{ 
				(*theLE).leCustomData = -1; 
				return deltaPoint;
			}
			/*scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
			 +interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
			 +interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			 scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
			 +interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
			 +interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);*/
			scaledPatVelocity.u = interpolationVal.alpha1*(timeAlpha*GetStartUVelocity(ptIndex1) + (1-timeAlpha)*GetEndUVelocity(ptIndex1))
			+interpolationVal.alpha2*(timeAlpha*GetStartUVelocity(ptIndex2) + (1-timeAlpha)*GetEndUVelocity(ptIndex2))
			+interpolationVal.alpha3*(timeAlpha*GetStartUVelocity(ptIndex3) + (1-timeAlpha)*GetEndUVelocity(ptIndex3));
			scaledPatVelocity.v = interpolationVal.alpha1*(timeAlpha*GetStartVVelocity(ptIndex1) + (1-timeAlpha)*GetEndVVelocity(ptIndex1))
			+interpolationVal.alpha2*(timeAlpha*GetStartVVelocity(ptIndex2) + (1-timeAlpha)*GetEndVVelocity(ptIndex2))
			+interpolationVal.alpha3*(timeAlpha*GetStartVVelocity(ptIndex3) + (1-timeAlpha)*GetEndVVelocity(ptIndex3));
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	
scale:
	
	// get and apply our time file scale factor
	if (timeDep && bTimeFileActive)
	{
		// VelocityRec errVelocity={1,1};
		// JLM 11/22/99, if there are no time file values, use zero not 1
		VelocityRec errVelocity={0,1};
		err = timeDep -> GetTimeValue (model -> GetModelTime(), &timeValue); 
		if(err) timeValue = errVelocity;
	}
	
	scaledPatVelocity.u *= abs(timeValue.u); // magnitude contained in u field only
	scaledPatVelocity.v *= abs(timeValue.u); 	// multiplying tide by tide, don't want to change phase
	//scaledPatVelocity.u = timeValue.u; // magnitude contained in u field only
	//scaledPatVelocity.v = timeValue.v; 	// multiplying tide by tide, don't want to change phase
	
	//scaledPatVelocity.u *= fVar.curScale; // may want to allow some sort of scale factor, though should be in file
	//scaledPatVelocity.v *= fVar.curScale; 
	
	
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

VelocityRec TideCurCycleMover_c::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("TideCurCycleMover::GetScaledPatValue is unimplemented");
	return v;
}

VelocityRec TideCurCycleMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("TideCurCycleMover::GetPatValue is unimplemented");
	return v;
}

/*long TideCurCycleMover_c::GetVelocityIndex(WorldPoint p) 
 {
 long rowNum, colNum;
 VelocityRec	velocity;
 
 LongRect		gridLRect, geoRect;
 ScaleRec		thisScaleRec;
 
 TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of TideCurCycleMover
 
 WorldRect bounds = rectGrid->GetBounds();
 
 SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
 SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
 GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
 
 colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
 rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
 
 
 if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
 
 { return -1; }
 
 //return INDEXH (fGridHdl, rowNum * fNumCols + colNum);
 return rowNum * fNumCols + colNum;
 }*/

/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize gridcur currents
/*double TideCurCycleMover_c::GetStartUVelocity(long index)
 {	// 
 double u = 0;
 if (fStartData.dataHdl && index>=0)
 u = INDEXH(fStartData.dataHdl,index).u;
 return u;
 }
 
 double TideCurCycleMover_c::GetEndUVelocity(long index)
 {
 double u = 0;
 if (fEndData.dataHdl && index>=0)
 u = INDEXH(fEndData.dataHdl,index).u;
 return u;
 }
 
 double TideCurCycleMover_c::GetStartVVelocity(long index)
 {
 double v = 0;
 if (fStartData.dataHdl && index >= 0)
 v = INDEXH(fStartData.dataHdl,index).v;
 return v;
 }
 
 double TideCurCycleMover_c::GetEndVVelocity(long index)
 {
 double v = 0;
 if (fEndData.dataHdl && index >= 0)
 v = INDEXH(fEndData.dataHdl,index).v;
 return v;
 }
 
 OSErr TideCurCycleMover_c::GetStartTime(Seconds *startTime)
 {
 OSErr err = 0;
 *startTime = 0;
 if (fStartData.timeIndex != UNASSIGNEDINDEX && fTimeHdl)
 *startTime = (*fTimeHdl)[fStartData.timeIndex];
 else return -1;
 return 0;
 }
 
 OSErr TideCurCycleMover_c::GetEndTime(Seconds *endTime)
 {
 OSErr err = 0;
 *endTime = 0;
 if (fEndData.timeIndex != UNASSIGNEDINDEX && fTimeHdl)
 *endTime = (*fTimeHdl)[fEndData.timeIndex];
 else return -1;
 return 0;
 }*/

Boolean TideCurCycleMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	Boolean isDry = false;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	err = /*CHECK*/dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg);
	if(err) return false;
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	interpolationVal = fGrid -> GetInterpolationValues(wp.p);
	if (interpolationVal.ptIndex1<0) return false;
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
		ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
		ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
	}
	
	// Check for constant current 
	if(/*OK*/dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,1);
			
			velocity.u = interpolationVal.alpha1*GetStartUVelocity(ptIndex1)
			+interpolationVal.alpha2*GetStartUVelocity(ptIndex2)
			+interpolationVal.alpha3*GetStartUVelocity(ptIndex3);
			velocity.v = interpolationVal.alpha1*GetStartUVelocity(ptIndex1)
			+interpolationVal.alpha2*GetStartUVelocity(ptIndex2)
			+interpolationVal.alpha3*GetStartUVelocity(ptIndex3);
			/*velocity.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).u)
			 +interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).u)
			 +interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).u );
			 velocity.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,ptIndex1).v)
			 +interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,ptIndex2).v)
			 +interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,ptIndex3).v);*/
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if(fTimeAlpha==-1)
		{
			Seconds relTime = time - model->GetStartTime();
			startTime = (*fTimeHdl)[fStartData.timeIndex];
			endTime = (*fTimeHdl)[fEndData.timeIndex];
			//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			timeAlpha = (endTime - relTime)/(double)(endTime - startTime);
		}
		else
			timeAlpha = fTimeAlpha;
		
		// Calculate the interpolated velocity at the point
		if (interpolationVal.ptIndex1 >= 0) 
		{
			isDry = IsDryTriangle(ptIndex1,ptIndex2,ptIndex3,timeAlpha);
			
			velocity.u = interpolationVal.alpha1*(timeAlpha*GetStartUVelocity(ptIndex1) + (1-timeAlpha)*GetEndUVelocity(ptIndex1))
			+interpolationVal.alpha2*(timeAlpha*GetStartUVelocity(ptIndex2) + (1-timeAlpha)*GetEndUVelocity(ptIndex2))
			+interpolationVal.alpha3*(timeAlpha*GetStartUVelocity(ptIndex3) + (1-timeAlpha)*GetEndUVelocity(ptIndex3));
			velocity.v = interpolationVal.alpha1*(timeAlpha*GetStartVVelocity(ptIndex1) + (1-timeAlpha)*GetEndVVelocity(ptIndex1))
			+interpolationVal.alpha2*(timeAlpha*GetStartVVelocity(ptIndex2) + (1-timeAlpha)*GetEndVVelocity(ptIndex2))
			+interpolationVal.alpha3*(timeAlpha*GetStartVVelocity(ptIndex3) + (1-timeAlpha)*GetEndVVelocity(ptIndex3));
			/*velocity.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).u)
			 +interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).u)
			 +interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).u);
			 velocity.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex1).v)
			 +interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex2).v)
			 +interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex3).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex3).v);*/
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	//velocity.u *= fVar.curScale; 
	//velocity.v *= fVar.curScale; 
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->refScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (!isDry)	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
						this->className, uStr, sStr);
	else sprintf(diagnosticStr, " [grid: %s, dry value]",
				 this->className);
	
	return true;
}


OSErr TideCurCycleMover_c::ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts) 
{
	OSErr err = 0;
	char errmsg[256];
	long i, n, nv = fNumNodes;
	long currentBoundary;
	long numVerdatPts = 0, numVerdatBreakPts = 0;
	
	LONGH vertFlagsH = (LONGH)_NewHandleClear(nv * sizeof(**vertFlagsH));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	LONGH verdatBreakPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatBreakPtsH));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	Boolean addOne = false;	// for debugging
	
	// write out verdat file for debugging
	/*FILE *outfile = 0;
	 char name[32], path[256],m[300];
	 SFReply reply;
	 Point where = CenteredDialogUpLeft(M55);
	 char ibmBackwardsTypeStr[32] = "";
	 strcpy(name,"NewVerdat.dat");
	 errmsg[0]=0;
	 
	 #ifdef MAC
	 sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
	 #else
	 sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	 M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	 #endif
	 if (!reply.good) {err = -1; goto done;}
	 
	 my_p2cstr(reply.fName);
	 #ifdef MAC
	 GetFullPath (reply.vRefNum, 0, (char *) "", path);
	 strcat (path, ":");
	 strcat (path, (char *) reply.fName);
	 #else
	 strcpy(path, reply.fName);
	 #endif
	 //strcpy(sExportSelectedTriPath, path); // remember the path for the user
	 SetWatchCursor();
	 sprintf(m, "Exporting VERDAT to %s...",path);
	 DisplayMessage("NEXTMESSAGETEMP");
	 DisplayMessage(m);*/
	/////////////////////////////////////////////////
	
	
	if (!vertFlagsH || !verdatPtsH || !verdatBreakPtsH) {err = memFullErr; goto done;}
	
	// put boundary points into verdat list
	
	// code goes here, double check that the water boundary info is also reordered
	currentBoundary=1;
	if (bndry_nums[0]==0) addOne = true;	// for debugging
	for (i = 0; i < numBoundaryPts; i++)
	{	
		short islandNum, index;
		index = bndry_indices[i];
		islandNum = bndry_nums[i];
		if (addOne) islandNum++;	// for debugging
		INDEXH(vertFlagsH,index-1) = 1;	// note that point has been used
		INDEXH(verdatPtsH,numVerdatPts++) = index-1;	// add to verdat list
		if (islandNum>currentBoundary)
		{
			// for verdat file indices are really point numbers, subtract one for actual index
			INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = i;	// passed a break point
			currentBoundary++;
		}
	}
	INDEXH(verdatBreakPtsH,numVerdatBreakPts++) = numBoundaryPts;
	
	// add the rest of the points to the verdat list (these points are the interior points)
	for(i = 0; i < nv; i++) {
		if(INDEXH(vertFlagsH,i) == 0)	
		{
			INDEXH(verdatPtsH,numVerdatPts++) = i;
			INDEXH(vertFlagsH,i) = 0; // mark as used
		}
	}
	if (numVerdatPts!=nv) 
	{
		printNote("Not all vertex points were used");
		// shrink handle
		_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
	}
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	/*outfile=fopen(path,"w");
	 if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	 fprintf(outfile,"DOGS\tMETERS\n");*/
	
	for (i=0; i<=numVerdatPts; i++)
	{
		long index;
		float fLong, fLat, fDepth;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			fLat = INDEXH(fVertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(fVertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
		}
		else { // the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		/////////////////////////////////////////////////
	}
	// figure out the bounds
	triBounds = voidWorldRect;
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		if(numVerdatPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numVerdatPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &triBounds);
			}
		}
	}
	
	// write out the number of chains
	/*fprintf(outfile,"%ld\n",numVerdatBreakPts);
	 
	 // now write out out the break points
	 for(i = 0; i < numVerdatBreakPts; i++ )
	 {
	 fprintf(outfile,"%ld\n",INDEXH(verdatBreakPtsH,i));
	 }
	 /////////////////////////////////////////////////
	 
	 fclose(outfile);*/
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
		//if (err = maketriangles2(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts,verdatPtsH,fNumCols_ext))
		goto done;
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TideCurCycleMover::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to create dag tree.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(totalDepthH);	// used by PtCurMap to check vertical movement
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//totalDepthH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
	numBoundaryPts = INDEXH(verdatBreakPtsH,numVerdatBreakPts-1)+1;
	waterBoundariesH = (LONGH)_NewHandle(sizeof(long)*numBoundaryPts);
	if (!waterBoundariesH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
	}
	
	if (waterBoundariesH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fPathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(verdatBreakPtsH);	
		map->SetWaterBoundaries(waterBoundariesH);
		
		*newMap = map;
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
	}
	
	/////////////////////////////////////////////////
	fVerdatToNetCDFH = verdatPtsH;	// this should be resized
	
done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TideCurCycleMover::ReorderPoints");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (*newMap) 
		{
			(*newMap)->Dispose();
			delete *newMap;
			*newMap=0;
		}
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	}
	return err;
}