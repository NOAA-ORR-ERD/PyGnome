/*
 *  TriCurMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 11/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriCurMover_c.h"
#include "CROSS.H"

#ifdef pyGNOME
#define TMap Map_c
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



OSErr TriCurMover_c::PrepareForModelStep()
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	/*if (!bActive) return 0; 
	 
	 err = this -> SetInterval(errmsg);
	 if(err) goto done;
	 
	 fIsOptimizedForStep = true;*/
	
	if (model->GetModelTime() == model->GetStartTime())	// first step
	{
		//PtCurMap* ptCurMap = (PtCurMap*)moverMap;
		PtCurMap* ptCurMap = GetPtCurMap();
		if (ptCurMap)
		{
			/*OK*/(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
			/*OK*/(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
			((TTriGridVel3D*)fGrid)->ClearOutputHandles();
		}
	}
	
	if (!bActive) return 0; 
	
	err = /*CHECK*/dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);
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
	TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	return triGrid -> GetPointsHdl();
}

TopologyHdl TriCurMover_c::GetTopologyHdl()
{
	TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	return triGrid -> GetTopologyHdl();
}

long TriCurMover_c::WhatTriAmIIn(WorldPoint wp)
{
	LongPoint lp;
	TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	TDagTree *dagTree = triGrid->GetDagTree();
	lp.h = wp.pLong;
	lp.v = wp.pLat;
	return dagTree -> WhatTriAmIIn(lp);
}

WorldPoint3D TriCurMover_c::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
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
		err = /*CHECK*/dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	
	triNum = WhatTriAmIIn(refPoint);
	if (triNum < 0) return deltaPoint;	// probably an error
	
	if (fDepthDataInfo) totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(fDepthsH);	
	
	GetDepthIndices(triNum,depth,&depthIndex1,&depthIndex2);
	if (depthIndex1 == -1) return deltaPoint;
	
	if(/*OK*/dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/)
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

VelocityRec TriCurMover_c::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
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
	err = /*CHECK*/dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);
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
	if(/*OK*/dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()==1)
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
	OSErr err = ((TTriGridVel3D*)fGrid)->GetTriangleVertices(trinum,x,y);
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
	
	err = /*CHECK*/dynamic_cast<TriCurMover *>(this) -> SetInterval(errmsg);
	if(err) return -1;
	
	loaded = /*CHECK*/dynamic_cast<TriCurMover *>(this) -> CheckInterval(timeDataInterval);
	if(!loaded) return -1;
	
	// Check for time varying current 
	if(/*OK*/dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()>1 /*|| GetNumFiles()>1*/)
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
		if(/*OK*/dynamic_cast<TriCurMover *>(this)->GetNumTimesInFile()==1)
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