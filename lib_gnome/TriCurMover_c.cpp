/*
 *  TriCurMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriCurMover_c.h"
#include "MemUtils.h"
#include "CompFunctions.h"
#include "StringFunctions.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

TriCurMover_c::TriCurMover_c (TMap *owner, char *name) : CurrentMover_c(owner, name), Mover_c(owner, name)
{
	memset(&fVar,0,sizeof(fVar));
	fVar.arrowScale = 5;
	fVar.arrowDepth = 0;
	fVar.alongCurUncertainty = .5;
	fVar.crossCurUncertainty = .25;
	//fVar.uncertMinimumInMPS = .05;
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	fVar.durationInHrs = 24.0;
	fVar.numLandPts = 0; // default that boundary velocities are given
	fVar.maxNumDepths = 1; // 2D default
	fVar.gridType = TWO_D; // 2D default
	fVar.bLayerThickness = 0.; // FREESLIP default
	//
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);
	//
	fGrid = 0;
	fTimeDataHdl = 0;
	fIsOptimizedForStep = false;
	//fOverLap = false;		// for multiple files case
	//fOverLapStartTime = 0;
	
	memset(&fInputValues,0,sizeof(fInputValues));
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fDepthsH = 0;
	fDepthDataInfo = 0;
	//fInputFilesHdl = 0;	// for multiple files case
	
	bShowDepthContourLabels = false;
	bShowDepthContours = false;
	
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
	
	SetClassName (name); // short file name
	
}
OSErr TriCurMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
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
		
		if(lengthS < fVar.uncertMinimumInMPS)
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
		TechError("TriCurMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}



OSErr TriCurMover_c::PrepareForModelStep(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, const Seconds& time_step, bool uncertain)
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	/*if (!bActive) return 0; 
	 
	 err = this -> SetInterval(errmsg);
	 if(err) goto done;
	 
	 fIsOptimizedForStep = true;*/
	
	if (model_time == start_time)	// first step
	{
		//PtCurMap* ptCurMap = (PtCurMap*)moverMap;
		PtCurMap* ptCurMap = GetPtCurMap();
		if (ptCurMap)
		{
			(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
			(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
			(dynamic_cast<TTriGridVel3D*>(fGrid))->ClearOutputHandles();
		}
	}
	
	if (!bActive) return 0; 
	
// 	err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);	// minus AH 07/17/2012
	err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg, start_time, model_time);	// AH 07/17/2012
	
	if(err) goto done;
	
	fIsOptimizedForStep = true;
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TriCurMover::PrepareForModelStep");
		printError(errmsg); 
	}
	return err;
}

void TriCurMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}


/*long TriCurMover::GetNumFiles()
 {
 long numFiles = 0;
 
 if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
 return numFiles;     
 }*/

long TriCurMover_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

LongPointHdl TriCurMover_c::GetPointsHdl()
{
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid);
	return triGrid -> GetPointsHdl();
}

TopologyHdl TriCurMover_c::GetTopologyHdl()
{
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid);
	return triGrid -> GetTopologyHdl();
}

long TriCurMover_c::WhatTriAmIIn(WorldPoint wp)
{
	LongPoint lp;
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid);
	TDagTree *dagTree = triGrid->GetDagTree();
	lp.h = wp.pLong;
	lp.v = wp.pLat;
	return dagTree -> WhatTriAmIIn(lp);
}

WorldPoint3D TriCurMover_c::GetMove(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	// figure out which depth values the LE falls between
	// since velocities are at centers no need to interpolate, use value over whole triangle
	// and some sort of check on the returned indices, what to do if one is below bottom?
	// for sigma model might have different depth values at each point
	// for multilayer they should be the same, so only one interpolation would be needed
	// others don't have different velocities at different depths so no interpolation is needed
	
	WorldPoint3D deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depth = (*theLE).z;
	long depthIndex1, depthIndex2, velDepthIndex1, velDepthIndex2 = -1;
	double topDepth, bottomDepth, depthAlpha;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec scaledPatVelocity = {0.,0.};
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	long triNum, numDepths = 0, totalDepth = 0;
	
	if(!fIsOptimizedForStep) 
	{
//		err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);	// minus AH 07/17/2012
		err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg, start_time, model_time); // AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	
	triNum = WhatTriAmIIn(refPoint);
	if (triNum < 0) return deltaPoint;	// probably an error
	
	if (fDepthDataInfo) totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(fDepthsH);	
	
	GetDepthIndices(triNum,depth,&depthIndex1,&depthIndex2);
	if (depthIndex1 == -1) return deltaPoint;
	
	if(dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/)
	{
		if (depthIndex1!=-1)
		{
			if (depthIndex2!=-1 && numDepths > 0 && totalDepth > 0) 
			{
				topDepth = INDEXH(fDepthsH,depthIndex1);
				bottomDepth = INDEXH(fDepthsH,depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				scaledPatVelocity.u = depthAlpha*(INDEXH(fStartData.dataHdl,depthIndex1).u)
				+ (1-depthAlpha)*(INDEXH(fStartData.dataHdl,depthIndex2).u);
				scaledPatVelocity.v = depthAlpha*(INDEXH(fStartData.dataHdl,depthIndex1).v)
				+ (1-depthAlpha)*(INDEXH(fStartData.dataHdl,depthIndex2).v);
			}
			else
			{
				scaledPatVelocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u; 
				scaledPatVelocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v; 
			}
		}
		
	}
	
	else // time varying current 
	{
		// Calculate the time weight factor
		/*if (GetNumFiles()>1 && fOverLap)
		 startTime = fOverLapStartTime;
		 else*/
		startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		if (depthIndex1!=-1)
		{
			if (depthIndex2!=-1 && numDepths > 0 && totalDepth > 0) 
			{
				topDepth = INDEXH(fDepthsH,depthIndex1);
				bottomDepth = INDEXH(fDepthsH,depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				scaledPatVelocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u)
				+ (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
				scaledPatVelocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v)
				+ (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
			}
			else
			{
				scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u; 
				scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v; 
			}
		}
	}
	
	scaledPatVelocity.u *= fVar.curScale; 
	scaledPatVelocity.v *= fVar.curScale; 
	
	
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

VelocityRec TriCurMover_c::GetScaledPatValue(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("TriCurMover::GetScaledPatValue is unimplemented");
	return v;
}


VelocityRec TriCurMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("TriCurMover::GetPatValue is unimplemented");
	return v;
}

Boolean TriCurMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth/*, totalDepth=0*/;
	long depthIndex1,depthIndex2;	// default to -1?
	//WorldPoint refPoint = wp.p;
	
	long triNum, ptIndex; 
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
//	err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);	// minus AH 07/17/2012
	err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg, model->GetStartTime(), model->GetModelTime()); // AH 07/17/2012
	
	if(err) return false;
	
	
	//triNum = WhatTriAmIIn(refPoint);
	triNum = WhatTriAmIIn(wp.p);
	if (triNum < 0) return false;	// probably an error
	
	//ptIndex = triNum*fVar.maxNumDepths; 
	//ptIndex = (*fDepthDataInfo)[triNum].indexToDepthData;
	
	GetDepthIndices(triNum,fVar.arrowDepth,&depthIndex1,&depthIndex2);
	
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return false;	// no value for this point at chosen depth, should show zero?
	
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		//if (fDepthDataInfo) totalDepth = INDEXH(fDepthDataInfo,i).totalDepth;	// depth from input file (?) at triangle center
		//else {printError("Problem with depth data in TriCurMover::Draw"); return false;}
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthsH,depthIndex1);
		bottomDepth = INDEXH(fDepthsH,depthIndex2);
		depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}
	
	// Check for constant current 
	//if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
	if(dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		//if (ptIndex >= 0) 
		if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
		{
			velocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u;
			velocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v;
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
			velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
			//velocity.u = 0.;
			//velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		//if (GetNumFiles()>1 && fOverLap)
		//startTime = fOverLapStartTime;
		//else
		startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		//if (ptIndex >= 0) 
		if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
		{
			velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
			velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
		}
		else	// if negative corresponds to negative ntri, set vel to zero
		{
			//velocity.u = 0.;
			//velocity.v = 0.;
			velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
			velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
			velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
			velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
		}
	}
	//velocity.u *= fVar.curScale; 
	//velocity.v *= fVar.curScale; 
	
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fVar.curScale * lengthU;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
			this->className, uStr, sStr);
	
	return true;
}

OSErr TriCurMover_c::GetTriangleCentroid(long trinum, LongPoint *p)
{	
	long x[3],y[3];
	OSErr err = (dynamic_cast<TTriGridVel3D*>(fGrid))->GetTriangleVertices(trinum,x,y);
	p->v = (y[0]+y[1]+y[2])/3;
	p->h =(x[0]+x[1]+x[2])/3;
	return err;
}

float TriCurMover_c::GetMaxDepth(void)
{
	long i,numDepths;
	float depth, maxDepth = -9999.0;
	
	if (!fDepthDataInfo) return 0; // some error alert, no depth info to check
	
	numDepths = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	for (i=0;i<numDepths;i++)
	{
		depth = INDEXH(fDepthDataInfo,i).totalDepth;	// depth at triangle
		if (depth > maxDepth) 
			maxDepth = depth;
	}
	return maxDepth;
}

long TriCurMover_c::CreateDepthSlice(long triNum, float **depthSlice)	
//long TriCurMover::CreateDepthSlice(long triNum, float *depthSlice)	
{	// show depth concentration profile at selected triangle
	long i,n,listIndex,numDepthsToPlot=0;
	PtCurMap *map = GetPtCurMap();
	if (!map) return -1;
	TTriGridVel3D* triGrid = (TTriGridVel3D*) map -> GetGrid3D(true);	// since output data is stored on the refined grid need to used it here
	OSErr err = 0;
	double timeAlpha,depthAlpha;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	float topDepth,bottomDepth,totalDepth=0;
	float *depthSliceArray = 0;
	float inchesX, inchesY;
	long timeDataInterval;
	Boolean loaded;
	char errmsg[256];
	if (!triGrid) return -1;
	
	if (triNum < 0)
	{
		return -1;
	}
	
	if (fDepthDataInfo) totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	else {printError("Problem with depth data in TriCurMover::CreateDepthSlice"); return -1;}
	
	numDepthsToPlot = floor(totalDepth)+1;	// split into 1m increments to track vertical slice
	
	if (*depthSlice)
	{delete [] *depthSlice; *depthSlice = 0;}
	
	//if (depthSlice)
	//{delete [] depthSlice; depthSlice = 0;}
	
	depthSliceArray = new float[numDepthsToPlot+1];
	if (!depthSliceArray) {TechError("TriCurMover::CreateDepthSlice()", "new[]", 0); err = memFullErr; goto done;}
	
	depthSliceArray[0]=numDepthsToPlot;	//store size here, maybe store triNum too
	for (i=0;i<numDepthsToPlot;i++)
	{
		depthSliceArray[i+1]=0;
	}
	
//	err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);	// minus AH 07/17/2012
	err = dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg, model->GetStartTime(), model->GetModelTime()); // AH 07/17/2012
	
	if(err) return -1;
	
//	loaded = dynamic_cast<TriCurMover *>(this) -> CheckInterval(timeDataInterval);	// minus AH 07/17/2012
	loaded = dynamic_cast<TriCurMover *>(this) -> CheckInterval(timeDataInterval, model->GetStartTime(), model->GetModelTime());	// AH 07/17/2012
	
	if(!loaded) return -1;
	
	// Check for time varying current 
	if(dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()>1 /*|| GetNumFiles()>1*/)
	{
		// Calculate the time weight factor
		//if (GetNumFiles()>1 && fOverLap)
		//startTime = fOverLapStartTime;
		//else
		startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
	}
	for(i = 0; i < numDepthsToPlot; i++)
	{
		// get the value at each triangle center and draw an arrow
		//long ptIndex = (*fDepthDataInfo)[i].indexToDepthData;
		//WorldPoint wp;
		//Point p,p2;
		VelocityRec velocity = {0.,0.};
		long depthIndex1,depthIndex2;	// default to -1?
		
		GetDepthIndices(triNum,(float)i,&depthIndex1,&depthIndex2);
		
		if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
			continue;	// no value for this point at chosen depth
		
		if (depthIndex2!=UNASSIGNEDINDEX)
		{
			// Calculate the depth weight factor
			topDepth = INDEXH(fDepthsH,depthIndex1);
			bottomDepth = INDEXH(fDepthsH,depthIndex2);
			depthAlpha = (bottomDepth - (float)i)/(double)(bottomDepth - topDepth);
		}
		
		//p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
		
		// Check for constant current 
		if(dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()==1)
		{
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = INDEXH(fStartData.dataHdl,depthIndex1).u;
				velocity.v = INDEXH(fStartData.dataHdl,depthIndex1).v;
			}
			else 	// below surface velocity
			{
				velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).u;
				velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,depthIndex2).v;
			}
		}
		else // time varying current
		{
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u;
				velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v;
			}
			else	// below surface velocity
			{
				velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
				velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
				velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
				velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
			}
		}
		inchesX = (velocity.u * fVar.curScale) /*/ fVar.arrowScale*/;
		inchesY = (velocity.v * fVar.curScale)/* / fVar.arrowScale*/;
		depthSliceArray[i+1] = sqrt(inchesX*inchesX + inchesY*inchesY);
	}
	
done:
	//(*depthSlice) = depthSliceArray;
	if (err) 
	{
		if (depthSliceArray)
		{delete [] depthSliceArray; depthSliceArray = 0;}
		return err;
	}
	(*depthSlice) = depthSliceArray;
	//depthSlice = depthSliceArray;
	return numDepthsToPlot;
}

void TriCurMover_c::GetDepthIndices(long triIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long j,indexToDepthData = (*fDepthDataInfo)[triIndex].indexToDepthData;
	long numDepths = (*fDepthDataInfo)[triIndex].numDepths;
	float totalDepth = (*fDepthDataInfo)[triIndex].totalDepth;
	
	if (triIndex < 0) {*depthIndex1 = UNASSIGNEDINDEX; *depthIndex2 = UNASSIGNEDINDEX; return;}
	
	switch(fVar.gridType) 
	{
		case TWO_D:	// no depth data
			*depthIndex1 = indexToDepthData;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
		case BAROTROPIC:	// values same throughout column, but limit on total depth
			if (depthAtPoint <= totalDepth)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			else
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		case MULTILAYER: //
			//break;
		case SIGMA: // 
			if (depthAtPoint <= totalDepth && fDepthsH) // check data exists at chosen/LE depth for this point
			{
				for(j=0;j<numDepths-1;j++)
				{
					if(INDEXH(fDepthsH,indexToDepthData+j)<depthAtPoint &&
					   depthAtPoint<=INDEXH(fDepthsH,indexToDepthData+j+1))
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
					}
					else if(INDEXH(fDepthsH,indexToDepthData+j)==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
					}
				}
				if(INDEXH(fDepthsH,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
				}
				else if(INDEXH(fDepthsH,indexToDepthData+numDepths-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepths-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				}
				else if(INDEXH(fDepthsH,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
			break;
		default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}
}

OSErr TriCurMover_c::CalculateVerticalGrid(LongPointHdl ptsH, FLOATH totalDepthH, TopologyHdl topH, long numTri, FLOATH sigmaLevelsH, long numSigmaLevels) 
{
	long i,j,index=0,numDepths;
	long ptIndex1,ptIndex2,ptIndex3;
	float depth1,depth2,depth3;
	double depthAtPoint;	
	FLOATH depthsH = 0;
	DepthDataInfoH depthDataInfo = 0;
	OSErr err = 0;
	
	if (fVar.gridType == TWO_D) // may want an option to handle 2D here
	{	
		if (numSigmaLevels != 0) {printError("2D grid can't have sigma levels"); return -1;}
	}
	if (!totalDepthH || !topH) return -1;
	
	numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	
	if (fVar.gridType != TWO_D)
	{
		if (!sigmaLevelsH) return -1;
		depthsH = (FLOATH)_NewHandle(sizeof(float)*numSigmaLevels*numTri);
		if(!depthsH) {TechError("TriCurMover::CalculateVerticalGrid()", "_NewHandle()", 0); err = memFullErr; goto done;}
	}
	
	depthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**depthDataInfo)*numTri);
	if(!depthDataInfo){TechError("TriCurMover::CalculateVerticalGrid()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	numDepths = numSigmaLevels;
	if (numDepths==0) numDepths = 1;
	
	for (i=0;i<numTri;i++)
	{
		// get the index into the pts handle for each vertex
		
		ptIndex1 = (*topH)[i].vertex1;
		ptIndex2 = (*topH)[i].vertex2;
		ptIndex3 = (*topH)[i].vertex3;
		
		depth1 = (*totalDepthH)[ptIndex1];
		depth2 = (*totalDepthH)[ptIndex2];
		depth3 = (*totalDepthH)[ptIndex3];
		
		depthAtPoint = (depth1 + depth2 + depth3) / 3.;
		
		(*depthDataInfo)[i].totalDepth = depthAtPoint;
		
		(*depthDataInfo)[i].indexToDepthData = index;
		(*depthDataInfo)[i].numDepths = numDepths;
		
		for (j=0;j<numSigmaLevels;j++)
		{
			(*depthsH)[index+j] = (*sigmaLevelsH)[j] * depthAtPoint;
			
		}
		index+=numDepths;
	}
	fDepthsH = depthsH;
	fDepthDataInfo = depthDataInfo;
	
done:	
	
	if(err) 
	{
		if(depthDataInfo) {DisposeHandle((Handle)depthDataInfo); depthDataInfo = 0;}
		if(depthsH) {DisposeHandle((Handle)depthsH); depthsH = 0;}
	}
	return err;		
}


OSErr TriCurMover_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	char s[256], path[256]; 
	long i,j,line = 0;
	long offset,lengthToRead;
	CHARH h = 0;
	char *sectionOfFile = 0;
	char *strToMatch = 0;
	long len,numScanned;
	VelocityFH velH = 0;
	long totalNumberOfVels = 0;
	
	LongPointHdl ptsHdl = 0;
	//TopologyHdl topoH = GetTopologyHdl();
	//TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	
	OSErr err = 0;
	DateTimeRec time;
	Seconds timeSeconds;
	//long numPoints, numDepths; 
	long numTris;
	errmsg[0]=0;
	
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	
	lengthToRead = (*fTimeDataHdl)[index].lengthOfData;
	offset = (*fTimeDataHdl)[index].fileOffsetToStartOfData;
	
	if (fDepthDataInfo)
		numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	//if(topoH)
	//numTris = _GetHandleSize((Handle)topoH)/sizeof(**topoH);
	else 
	{err=-1; goto done;} // no data
	
	h = (CHARH)_NewHandle(lengthToRead+1);
	if(!h){TechError("TriCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	_HLock((Handle)h);
	sectionOfFile = *h;			
	
	err = ReadSectionOfFile(0,0,path,offset,lengthToRead,sectionOfFile,0);
	if(err || !h) 
	{
		char firstPartOfLine[128];
		sprintf(errmsg,"Unable to open data file:%s",NEWLINESTRING);
		strncpy(firstPartOfLine,path,120);
		strcpy(firstPartOfLine+120,"...");
		strcat(errmsg,firstPartOfLine);
		goto done;
	}
	sectionOfFile[lengthToRead] = 0; // make it a C string
	
	//numDepths = fVar.maxNumDepths;
	// for now we will always have a full set of velocities
	totalNumberOfVels = (*fDepthDataInfo)[numTris-1].indexToDepthData+(*fDepthDataInfo)[numTris-1].numDepths;
	//totalNumberOfVels = numTris*numDepths;
	if(totalNumberOfVels<numTris) {err=-1; goto done;} // must have at least full set of 2D velocity data
	velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*totalNumberOfVels);
	if(!velH){TechError("TriCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	strToMatch = "[TIME]";
	len = strlen(strToMatch);
	NthLineInTextOptimized (sectionOfFile, line = 0, s, 256);
	if(!strncmp(s,strToMatch,len)) 
	{
		numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("TriCurMover::ReadTimeData()", "sscanf() == 5", 0); goto done; }
		// check for constant current
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
			//if (time.year == time.month == time.day == time.hour == time.minute == -1) 
		{
			timeSeconds = CONSTANTCURRENT;
		}
		else // time varying current
		{
			if (time.year < 1900)					// two digit date, so fix it
			{
				if (time.year >= 40 && time.year <= 99)	
					time.year += 1900;
				else
					time.year += 2000;					// correct for year 2000 (00 to 40)
			}
			
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}
		
		// check time is correct
		if (timeSeconds!=(*fTimeDataHdl)[index].time)
		{ err = -1;  strcpy(errmsg,"Can't read data - times in the file have changed."); goto done; }
		line++;
	}
	
	
	for(i=0;i<numTris;i++) // interior points
	{
		VelocityRec vel;
		char *startScan;
		long scanLength,stringIndex=0;
		long numDepths = (*fDepthDataInfo)[i].numDepths;	// allow for variable depths/velocites
		//long numDepths = fVar.maxNumDepths;
		
		char *s1 = new char[numDepths*64];
		if(!s1) {TechError("TriCurMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		
		NthLineInTextOptimized (sectionOfFile, line, s1, numDepths*64);
		//might want to check that the number of lines matches the number of triangles (ie there is data at every triangle)
		startScan = &s1[stringIndex];
		
		for(j=0;j<numDepths;j++) 
		{
			err = ScanVelocity(startScan,&vel,&scanLength); 
			// ScanVelocity is faster than scanf, but doesn't handle scientific notation. Try a scanf on error.
			if (err)
			{
				if(err!=-2 || sscanf(&s1[stringIndex],lfFix("%lf%lf"),&vel.u,&vel.v) < 2)
				{
					char firstPartOfLine[128];
					sprintf(errmsg,"Unable to read velocity data from line %ld:%s",line,NEWLINESTRING);
					strncpy(firstPartOfLine,s1,120);
					strcpy(firstPartOfLine+120,"...");
					strcat(errmsg,firstPartOfLine);
					delete[] s1; s1=0;
					goto done;
				}
				err = 0;
			}
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u = vel.u; 
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v = vel.v; 
			//(*velH)[i*numDepths+j].u = vel.u; 
			//(*velH)[i*numDepths+j].v = vel.v; 
			stringIndex += scanLength;
			startScan = &s1[stringIndex];
		}
		line++;
		delete[] s1; s1=0;
	}
	*velocityH = velH;
	
done:
	
	if(h) {
		_HUnlock((Handle)h); 
		DisposeHandle((Handle)h); 
		h = 0;
	}
	
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in TriCurMover::ReadTimeData");
		//printError(errmsg); // This alert causes a freeze up...
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	return err;
	
}


OSErr TriCurMover_c::SetInterval(char *errmsg, const Seconds& start_time, const Seconds& model_time)
{
	long timeDataInterval=0;
//	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);	// minus AH 07/17/2012
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval, start_time, model_time);	// AH 07/17/2012
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
	
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
	
	// check for constant current 
	//if(numTimesInFile==1 && !(GetNumFiles()>1))	//or if(timeDataInterval==-1) 
	if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile)
	{	// before the first step in the file
		/*if (GetNumFiles()>1)
		 {
		 //if ((err = CheckAndScanFile(errmsg)) || fOverLap) goto done;	// overlap is special case
		 intervalLoaded = this -> CheckInterval(timeDataInterval);
		 indexOfStart = timeDataInterval-1;
		 indexOfEnd = timeDataInterval;
		 numTimesInFile = this -> GetNumTimesInFile();
		 }
		 else*/
		{
			err = -1;
			strcpy(errmsg,"Time outside of interval being modeled");
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
		if(!errmsg[0])strcpy(errmsg,"Error in TriCurMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;
	
}

Boolean TriCurMover_c::CheckInterval(long &timeDataInterval, const Seconds& start_time, const Seconds& model_time)
{
//	Seconds time = model->GetModelTime();	// minus AH 07/17/2012
	Seconds time = model_time; // AH 07/17/2012
	
	long i,numTimes;
	
	
	numTimes = this -> GetNumTimesInFile(); 
	
	// check for constant current
	if (numTimes==1 /*&& !(GetNumFiles()>1)*/) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
			return true;
		else
			return false;
	}
	
	if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
	{
		if (time>=(*fTimeDataHdl)[fStartData.timeIndex].time && time<=(*fTimeDataHdl)[fEndData.timeIndex].time)
		{	// we already have the right interval loaded
			timeDataInterval = fEndData.timeIndex;
			return true;
		}
	}
	
	/*if (GetNumFiles()>1 && fOverLap)
	 {	
	 if (time>=fOverLapStartTime && time<=(*fTimeDataHdl)[fEndData.timeIndex].time)
	 return true;	// we already have the right interval loaded, time is in between two files
	 else fOverLap = false;
	 }*/
	
	for (i=0;i<numTimes;i++) 
	{	// find the time interval
		if (time>=(*fTimeDataHdl)[i].time && time<=(*fTimeDataHdl)[i+1].time)
		{
			timeDataInterval = i+1; // first interval is between 0 and 1, and so on
			return false;
		}
	}	
	// don't allow time before first or after last
	if (time<(*fTimeDataHdl)[0].time) 
		timeDataInterval = 0;
	if (time>(*fTimeDataHdl)[numTimes-1].time) 
		timeDataInterval = numTimes;
	return false;
	
}

long TriCurMover_c::GetNumTimesInFile()
{
	long numTimes = 0;
	
	if (fTimeDataHdl) numTimes = _GetHandleSize((Handle)fTimeDataHdl)/sizeof(**fTimeDataHdl);
	return numTimes;     
}


void TriCurMover_c::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void TriCurMover_c::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}


