/*
 *  TideCurCycleMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TideCurCycleMover_c.h"
#include "MemUtils.h"
#include "my_build_list.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include "netcdf.h"

#ifndef pyGNOME
#include "TideCurCycleMover.h"
#include "TShioTimeValue.h"
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

#ifndef pyGNOME
TideCurCycleMover_c::TideCurCycleMover_c (TMap *owner, char *name) : CATSMover_c(owner, name)
{
	fTimeHdl = 0;
	
	fUserUnits = kUndefined;
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -.5; 
	fUpCurUncertainty = .5; 	
	fRightCurUncertainty = .25;  
	fLeftCurUncertainty = -.25; 
	fDuration=24*3600.; //24 hrs as seconds 
	fUncertainStartTime = 0.; // seconds
	fEddyV0 = 0.0;	// fVar.uncertMinimumInMPS
	//SetClassName (name); // short file name
	
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	fNumNodes = 0;
	
	//fPatternStartPoint = 2;	// some default
	fPatternStartPoint = MaxFlood;	// this should be user input
	fTimeAlpha = -1;
	
	fFillValue = -1e+34;
	fDryValue = -1e+34;
	//fDryValue = 999;
	
	fTopFilePath[0] = 0;	// don't seem to need this
}
#endif
TideCurCycleMover_c::TideCurCycleMover_c () : CATSMover_c()
{
	fTimeHdl = 0;
	
	fUserUnits = kUndefined;
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -.5; 
	fUpCurUncertainty = .5; 	
	fRightCurUncertainty = .25;  
	fLeftCurUncertainty = -.25; 
	fDuration=24*3600.; //24 hrs as seconds 
	fUncertainStartTime = 0.; // seconds
	fEddyV0 = 0.0;	// fVar.uncertMinimumInMPS
	//SetClassName (name); // short file name
	
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	fNumNodes = 0;
	
	//fPatternStartPoint = 2;	// some default
	fPatternStartPoint = MaxFlood;	// this should be user input
	fTimeAlpha = -1;
	
	fFillValue = -1e+34;
	fDryValue = -1e+34;
	//fDryValue = 999;
	
	fTopFilePath[0] = 0;	// don't seem to need this
}

void TideCurCycleMover_c::Dispose ()
{
	/*if (fGrid)
	 {
	 fGrid -> Dispose();
	 delete fGrid;
	 fGrid = nil;
	 }*/
	
	if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);
	
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	
	
	CATSMover_c::Dispose ();
}


LongPointHdl TideCurCycleMover_c::GetPointsHdl()
{
	return (dynamic_cast<TTriGridVel*>(fGrid)) -> GetPointsHdl();
}

OSErr TideCurCycleMover_c::ComputeVelocityScale(const Seconds& model_time)
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
	
	//Seconds time = model_time;	// AH 07/09/2012
	
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
	
	/*switch (scaleType) {
		case SCALE_NONE: this->refScale = 1; return noErr;
		case SCALE_CONSTANT:
			if(!this -> fOptimize.isOptimizedForStep) 
			{
				err = dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
				
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
			if(dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()==1)
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
					//Seconds relTime = time - model->GetStartTime();	// minus AH 07/10/2012
					Seconds relTime = time - start_time;	// AH 07/10/2012
					
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
	return -1;*/
}

OSErr TideCurCycleMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	//err = this -> UpdateUncertainty();
	//if(err) return err;
	
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
OSErr TideCurCycleMover_c::PrepareForModelRun()
{
	fOptimize.isFirstStep = true;
	return CurrentMover_c::PrepareForModelRun();
}
OSErr TideCurCycleMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;

	//if (fOptimize.isFirstStep) model_start_time = model_time;	// use fModelStartTime in current mover

	//check to see that the time interval is loaded and set if necessary
	if (bIsFirstStep)
	{
		VelocityRec dummyValue;
		fModelStartTime = model_time;
		if (timeDep) err = timeDep->GetTimeValue(model_time,&dummyValue);
	}
	if (!bActive) return noErr;
	//err = dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
	err = SetInterval(errmsg, model_time);
	
	if(err) goto done;
	
	if (uncertain)
	{
		Seconds elapsed_time = model_time - fModelStartTime;	// code goes here, how to set start time
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}
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
	fOptimize.isFirstStep = false;
	fOptimize.isOptimizedForStep = false;
	bIsFirstStep = false;
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
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid);
	TopologyHdl topH = triGrid->GetTopologyHdl();
	
	index1 = (*topH)[triIndex].vertex1;
	index2 = (*topH)[triIndex].vertex2;
	index3 = (*topH)[triIndex].vertex3;
	ptIndex1 =  (*fVerdatToNetCDFH)[index1];	
	ptIndex2 =  (*fVerdatToNetCDFH)[index2];
	ptIndex3 =  (*fVerdatToNetCDFH)[index3];
	
	if(dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()>1)
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


WorldPoint3D TideCurCycleMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	// see PtCurMover::GetMove - will depend on what is in netcdf files and how it's stored
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long ptIndex1,ptIndex2,ptIndex3; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model_time;
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity, timeValue = {1, 1};
	Boolean useEddyUncertainty = false, isDry = false;	
	OSErr err = 0;
	char errmsg[256];
	
	(*theLE).leCustomData = 0;
	
	if(!this -> fOptimize.isOptimizedForStep) 
	{
		err = dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
		
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
	if(dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()==1)
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
			//Seconds relTime = time - model->GetStartTime();
			Seconds relTime = time - fModelStartTime;
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
		//err = timeDep -> GetTimeValue (model -> GetModelTime(), &timeValue);	// AH 07/10/2012
		err = timeDep -> GetTimeValue (model_time, &timeValue);	// AH 07/10/2012
		//err = timeDep -> GetTimeValue (start_time, stop_time, model_time, &timeValue);	// AH 07/10/2012
		
		if(err) timeValue = errVelocity;
	}
	
	scaledPatVelocity.u *= myfabs(timeValue.u); // magnitude contained in u field only
	scaledPatVelocity.v *= myfabs(timeValue.u); 	// multiplying tide by tide, don't want to change phase
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

VelocityRec TideCurCycleMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
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
	err = dynamic_cast<TideCurCycleMover *>(this) -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
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
	if(dynamic_cast<TideCurCycleMover *>(this)->GetNumTimesInFile()==1)
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
			//Seconds relTime = time - model->GetStartTime();
			Seconds relTime; 
			if (fModelStartTime==0)	// prepareformodelstep hasn't been called yet, just show the first data set
				relTime = (*fTimeHdl)[0];
			else 
				relTime = time - fModelStartTime;
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
	if (lengthS > 1000000 || this->refScale==0) return true;	// if bad data in file causes a crash
	
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
		// it seems this should be an error...
		err = -1;
		goto done;
		// shrink handle
		//_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(long));
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


Boolean TideCurCycleMover_c::CheckInterval(long &timeDataInterval, const Seconds& model_time)
{
	Seconds time =  model_time; // AH 07/17/2012
	
	long i,numTimes;
	
	
	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists
	
	// check for constant current
	if (numTimes==1) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
			return true;
		else
			return false;
	}
	
	
	
	//#define MinBeforeFlood  0
	//#define MaxFlood  1
	//#define MinBeforeEbb  2
	//#define MaxEbb  3
	// need to know how tidal current pattern is divided up
	// starts with max flood, then ..., what is known?
	// each interval here corresponds to 3 in the pattern, assuming we have 12 times
	// should have a fPatternStartPoint - maxFlood or whatever
	{
		TShioTimeValue *timeFile = dynamic_cast<TShioTimeValue*>(dynamic_cast<TideCurCycleMover *>(this)->GetTimeDep());
		long numTimes = GetNumTimesInFile(), offset, index1, index2;
		float  numSegmentsPerFloodEbb = numTimes/4.;
		if (!timeFile || !bTimeFileActive)
		{
			//time = time - model->GetStartTime();
			if (fModelStartTime==0)	// haven't called prepareformodelstep yet, so get the first (or could set it...)
				time = (*fTimeHdl)[0];
			else
				time = time - fModelStartTime;
			fTimeAlpha = -1;
			// what used to be here
			if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
			{
				if (time>=(*fTimeHdl)[fStartData.timeIndex] && time<=(*fTimeHdl)[fEndData.timeIndex])
				{	// we already have the right interval loaded
					timeDataInterval = fEndData.timeIndex;
					return true;
				}
			}
			
			for (i=0;i<numTimes;i++) 
			{	// find the time interval
				if (time>=(*fTimeHdl)[i] && time<=(*fTimeHdl)[i+1])
				{
					timeDataInterval = i+1; // first interval is between 0 and 1, and so on
					return false;
				}
			}	
			// don't allow time before first or after last
			if (time<(*fTimeHdl)[0]) 
				timeDataInterval = 0;
			if (time>(*fTimeHdl)[numTimes-1]) 
				timeDataInterval = numTimes;
			return false;	// need to decide what to do here
		}
		else
		{
			short ebbFloodType;
			float fraction;
			timeFile->GetLocationInTideCycle(model_time,&ebbFloodType,&fraction);
			if (ebbFloodType>=fPatternStartPoint)
			{
				offset = ebbFloodType - fPatternStartPoint;
			}
			else
			{
				offset = ebbFloodType+4 - fPatternStartPoint;
			}
			index1 = floor(numSegmentsPerFloodEbb * (fraction + offset));	// round, floor?
			index2 = ceil(numSegmentsPerFloodEbb * (fraction + offset));	// round, floor?
			timeDataInterval = index2;	// first interval is between 0 and 1, and so on
			if (fraction==0) timeDataInterval++;
			fTimeAlpha = timeDataInterval-numSegmentsPerFloodEbb * (fraction + offset);
			// check if index2==numTimes, then need to cycle back to zero
			// check if index1==index2, then only need one file loaded
			if (index2==12) index2=0;
			if(fStartData.timeIndex==index1 && fEndData.timeIndex==index2)
			{
				return true;			
			}
			return false;
		}
		// check if interval already loaded, so fStartData.timeIndex == interval-1, fEndData.timeIndex = interval
		//return 0; 
	}
	
	return false;
	
}

void TideCurCycleMover_c::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void TideCurCycleMover_c::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long TideCurCycleMover_c::GetNumTimesInFile()
{
	long numTimes = 0;
	
	if (fTimeHdl) numTimes = _GetHandleSize((Handle)fTimeHdl)/sizeof(**fTimeHdl);
	return numTimes;     
}


OSErr TideCurCycleMover_c::SetInterval(char *errmsg, const Seconds& model_time)
{
	long timeDataInterval;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	// AH 07/17/2012
	
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
	
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
	
	// check for constant current 
	if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0)
	{	// before the first step in the file
		// shouldn't happen
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}
	if(timeDataInterval == numTimesInFile)
	{	// before the first step in the file
		if(fTimeAlpha>=0/*timeDep*/)
			indexOfEnd = 0;	// start over
		else
		{
			err = -1;
			//strcpy(errmsg,"Time outside of interval being modeled");
			strcpy(errmsg,"There are no more tidal patterns.");
			goto done;
		}
	}
	// load the two intervals
	{
		DisposeLoadedData(&fStartData);
		
		if(indexOfStart == fEndData.timeIndex) // passing into next interval
		{
			fStartData = fEndData;
			ClearLoadedData(&fEndData);
		}
		else
		{
			DisposeLoadedData(&fEndData);
		}
		
		//////////////////
		
		if(fStartData.dataHdl == 0 && indexOfStart >= 0) 
		{ // start data is not loaded
			err = this -> ReadTimeData(indexOfStart,&fStartData.dataHdl,errmsg);
			if(err) goto done;
			fStartData.timeIndex = indexOfStart;
		}	
		
		if(indexOfEnd < numTimesInFile && indexOfEnd != UNASSIGNEDINDEX)  // not past the last interval and not constant current
		{
			err = this -> ReadTimeData(indexOfEnd,&fEndData.dataHdl,errmsg);
			if(err) goto done;
			fEndData.timeIndex = indexOfEnd;
		}
	}
	
done:	
	if(err)
	{
		if(!errmsg[0])strcpy(errmsg,"Error in TideCurCycleMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;
	
}

OSErr TideCurCycleMover_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{	// - needs to be updated once triangle grid format is set
	OSErr err = 0;
	long i;
	char path[256], outPath[256]; 
	int status, ncid, numdims;
	int curr_ucmp_id, curr_vcmp_id;
	static size_t curr_index[] = {0,0,0};
	static size_t curr_count[3];
	float *curr_uvals,*curr_vvals, fill_value, dry_value = 0;
	long totalNumberOfVels = fNumNodes;
	VelocityFH velH = 0;
	long numNodes = fNumNodes;
	
	errmsg[0]=0;
	
	strcpy(path,fPathName);
	if (!path || !path[0]) return -1;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_ndims(ncid, &numdims);	// in general it's not the total number of dimensions but the number the variable depends on
	if (status != NC_NOERR) {err = -1; goto done;}
	
	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	//curr_count[1] = 1;	// depth
	//curr_count[2] = numNodes;
	
	// check for sigma or zgrid dimension
	if (numdims>=6)	// should check what the dimensions are
	{
		curr_count[1] = 1;	// depth
		//curr_count[1] = depthlength;	// depth
		curr_count[2] = numNodes;
	}
	else
	{
		curr_count[1] = numNodes;	
	}
	curr_uvals = new float[numNodes]; 
	if(!curr_uvals) {TechError("TideCurCycleMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	curr_vvals = new float[numNodes]; 
	if(!curr_vvals) {TechError("TideCurCycleMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	
	status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);// missing_value vs _FillValue
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_att_float(ncid, curr_ucmp_id, "missing_value", &fill_value);// missing_value vs _FillValue
	//if (status != NC_NOERR) {err = -1; goto done;}
	if (status != NC_NOERR) fill_value=-9999.;
	status = nc_get_att_float(ncid, curr_ucmp_id, "dry_value", &dry_value);// missing_value vs _FillValue
	if (status != NC_NOERR) {/*err = -1; goto done;*/}  
	else fDryValue = dry_value;
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) {err = memFullErr; goto done;}
	for (i=0;i<totalNumberOfVels;i++)
	{
		// really need to store the fill_value data and check for it when moving or drawing
		if (curr_uvals[i]==0.||curr_vvals[i]==0.)
			curr_uvals[i] = curr_vvals[i] = 1e-06;
		if (curr_uvals[i]==fill_value)
			curr_uvals[i]=0.;
		if (curr_vvals[i]==fill_value)
			curr_vvals[i]=0.;
		// for now until we decide what to do with the dry value flag
		//if (curr_uvals[i]==dry_value)
		//curr_uvals[i]=0.;
		//if (curr_vvals[i]==dry_value)
		//curr_vvals[i]=0.;
		INDEXH(velH,i).u = curr_uvals[i];	// need units
		INDEXH(velH,i).v = curr_vvals[i];
	}
	*velocityH = velH;
	fFillValue = fill_value;
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading current data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) delete [] curr_uvals;
	if (curr_vvals) delete [] curr_vvals;
	return err;
}
