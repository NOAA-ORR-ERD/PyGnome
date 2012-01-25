/*
 *  NetCDFMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "NetCDFMover_c.h"
#include "netcdf.h"
#include "CROSS.H"

#ifdef pyGNOME
#define TMap Map_c
#endif

NetCDFMover_c::NetCDFMover_c (TMap *owner, char *name) : CurrentMover_c(owner, name), Mover_c(owner, name)
{
	memset(&fVar,0,sizeof(fVar));
	fVar.arrowScale = 1.;
	fVar.arrowDepth = 0;
	if (gNoaaVersion)
	{
		fVar.alongCurUncertainty = .5;
		fVar.crossCurUncertainty = .25;
		fVar.durationInHrs = 24.0;
	}
	else
	{
		fVar.alongCurUncertainty = 0.;
		fVar.crossCurUncertainty = 0.;
		fVar.durationInHrs = 0.;
	}
	//fVar.uncertMinimumInMPS = .05;
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	//fVar.durationInHrs = 24.0;
	fVar.gridType = TWO_D; // 2D default
	fVar.maxNumDepths = 1;	// 2D default - may always be constant for netCDF files
	
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);
	//
	fGrid = 0;
	fTimeHdl = 0;
	fDepthLevelsHdl = 0;	// depth level, sigma, or sc_r
	fDepthLevelsHdl2 = 0;	// Cs_r
	hc = 1.;	// what default?
	
	bShowDepthContours = false;
	bShowDepthContourLabels = false;
	
	fTimeShift = 0;	// assume file is in local time
	fIsOptimizedForStep = false;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;
	
	fFillValue = -1e+34;
	fIsNavy = false;	
	
	/*fOffset_u = 0.;
	 fOffset_v = 0.;
	 fCurScale_u = 1.;
	 fCurScale_v = 1.;*/
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fDepthsH = 0;
	fDepthDataInfo = 0;
	fInputFilesHdl = 0;	// for multiple files case
	
	SetClassName (name); // short file name
	
	fNumDepthLevels = 1;	// default surface current only
	
	fAllowExtrapolationOfCurrentsInTime = false;
	fAllowVerticalExtrapolationOfCurrents = false;
	fMaxDepthForExtrapolation = 0.;	// assume 2D is just surface
	
	memset(&fLegendRect,0,sizeof(fLegendRect)); 
}


OSErr NetCDFMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
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
		TechError("NetCDFMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}

OSErr NetCDFMover_c::PrepareForModelStep()
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (model->GetModelTime() == model->GetStartTime())	// first step
	{
		if (/*OK*/dynamic_cast<NetCDFMover *>(this)->IAm(TYPE_NETCDFMOVERCURV) || dynamic_cast<NetCDFMover *>(this)->IAm(TYPE_NETCDFMOVERTRI))
		{
			//PtCurMap* ptCurMap = (PtCurMap*)moverMap;
			//PtCurMap* ptCurMap = GetPtCurMap();
			//if (ptCurMap)
			if (moverMap->IAm(TYPE_PTCURMAP))
			{
				(/*OK*/dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
				(/*OK*/dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
				if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)
					((TTriGridVel3D*)fGrid)->ClearOutputHandles();
			}
		}
	}
	if (!bActive) return noErr;
	
	err = /*CHECK*/dynamic_cast<NetCDFMover *>(this) -> SetInterval(errmsg); // SetInterval checks to see that the time interval is loaded
	if (err) goto done;
	
	fIsOptimizedForStep = true;
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	return err;
}

void NetCDFMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}


long NetCDFMover_c::GetNumDepthLevels()
{
	long numDepthLevels = 0;
	
	if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	else
	{
		long numDepthLevels = 0;
		OSErr err = 0;
		char path[256], outPath[256];
		int status, ncid, sigmaid, sigmavarid;
		size_t sigmaLength=0;
		//if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
		//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
		strcpy(path,fVar.pathName);
		if (!path || !path[0]) return -1;
		
		status = nc_open(path, NC_NOWRITE, &ncid);
		if (status != NC_NOERR) /*{err = -1; goto done;}*/
		{
#if TARGET_API_MAC_CARBON
			err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
			status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
			if (status != NC_NOERR) {err = -1; return -1;}
		}
		//if (status != NC_NOERR) {/*err = -1; goto done;*/return -1;}
		status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
		if (status != NC_NOERR) 
		{
			numDepthLevels = 1;	// check for zgrid option here
		}	
		else
		{
			status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
			if (status != NC_NOERR) {numDepthLevels = 1;}	// require variable to match the dimension
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {numDepthLevels = 1;}	// error in file
			//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
			//fVar.maxNumDepths = sigmaLength;
			numDepthLevels = sigmaLength;
			//status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
			//if (status != NC_NOERR) {err = -1; goto done;}
			// once depth is read in 
		}
	}
	return numDepthLevels;     
}

long NetCDFMover_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

WorldPoint3D NetCDFMover_c::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index; 
	long depthIndex1,depthIndex2;	// default to -1?
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	Boolean useVelSubsurface = false;	// don't seem to need this, just return 
	char errmsg[256];
	
	if(!fIsOptimizedForStep) 
	{
		err = /*CHECK*/dynamic_cast<NetCDFMover *>(this) -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
	
	if ((*theLE).z>0 && fVar.gridType==TWO_D)
	{		
		if (fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= (*theLE).z) useVelSubsurface = true;
		else
		{	// may allow 3D currents later
			deltaPoint.p.pLong = 0.;
			deltaPoint.p.pLat = 0.;
			deltaPoint.z = 0;
			return deltaPoint; 
		}
	}
	
	GetDepthIndices(0,(*theLE).z,&depthIndex1,&depthIndex2);
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}
	
	// Check for constant current 
	if((/*OK*/dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMover *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
		//if(GetNumTimesInFile()==1 && !(GetNumFiles()>1))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				//scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
				//scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
				scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else
			{
				scaledPatVelocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
				scaledPatVelocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
			}
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (/*OK*/dynamic_cast<NetCDFMover *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				//scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
				//scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
				scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else	// below surface velocity
			{
				scaledPatVelocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
				scaledPatVelocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
				scaledPatVelocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
				scaledPatVelocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
			}
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	
scale:
	
	scaledPatVelocity.u *= fVar.curScale; 
	scaledPatVelocity.v *= fVar.curScale; 
	//scaledPatVelocity.u *= fCurScale_u; 
	//scaledPatVelocity.v *= fCurScale_v; 
	
	//if (scaledPatVelocity.u != 0) scaledPatVelocity.u += fOffset_u; 
	//if (scaledPatVelocity.v != 0) scaledPatVelocity.v += fOffset_v; 
	
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

VelocityRec NetCDFMover_c::GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("NetCDFMover::GetScaledPatValue is unimplemented");
	return v;
}

Seconds NetCDFMover_c::GetTimeValue(long index)
{
	if (index<0) printError("Access violation in NetCDFMover::GetTimeValue()");
	Seconds time = (*fTimeHdl)[index] + fTimeShift;
	return time;
}

VelocityRec NetCDFMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("NetCDFMover::GetPatValue is unimplemented");
	return v;
}

long NetCDFMover_c::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	double dRowNum, dColNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	//SetLRect (&gridLRect, 0, fNumRows-1, fNumCols-1, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	//colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	//rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	//dColNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	//dRowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	//dColNum = round((p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5);
	//dRowNum = round((p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5);
	dColNum = (p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5;
	dRowNum = (p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5;
	//if (dColNum<0) dColNum = -1; if (dRowNum<0) dRowNum = -1;
	//colNum = dColNum;
	//rowNum = dRowNum;
	colNum = round(dColNum);
	rowNum = round(dRowNum);
	
	//if (colNum < 0 || colNum >= fNumCols-1 || rowNum < 0 || rowNum >= fNumRows-1)
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return -1; }
	
	return rowNum * fNumCols + colNum;
	//return rowNum * (fNumCols-1) + colNum;
}

LongPoint NetCDFMover_c::GetVelocityIndices(WorldPoint p) 
{
	long rowNum, colNum;
	double dRowNum, dColNum;
	LongPoint indices = {-1,-1};
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	//colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	//rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	dColNum = round((p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5);
	dRowNum = round((p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5);
	//if (dColNum<0) dColNum = -1; if (dRowNum<0) dRowNum = -1;
	colNum = dColNum;
	rowNum = dRowNum;
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return indices; }
	
	//return rowNum * fNumCols + colNum;
	indices.h = colNum;
	indices.v = rowNum;
	return indices;
}


/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize netcdf currents
double NetCDFMover_c::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (index>=0)
	{
		if (fStartData.dataHdl) u = INDEXH(fStartData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFMover_c::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		if (fEndData.dataHdl) u = INDEXH(fEndData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFMover_c::GetStartVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fStartData.dataHdl) v = INDEXH(fStartData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

double NetCDFMover_c::GetEndVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fEndData.dataHdl) v = INDEXH(fEndData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

OSErr NetCDFMover_c::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX)
		*startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

OSErr NetCDFMover_c::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX)
		*endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

double NetCDFMover_c::GetDepthAtIndex(long depthIndex, double totalDepth)
{	// really can combine and use GetDepthAtIndex - could move to base class
	double depth = 0;
	float sc_r, Cs_r;
	if (fVar.gridType == SIGMA_ROMS)
	{
		sc_r = INDEXH(fDepthLevelsHdl,depthIndex);
		Cs_r = INDEXH(fDepthLevelsHdl2,depthIndex);
		//depth = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
		depth = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
	}
	else
		depth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth; // times totalDepth
	
	return depth;
}

Boolean NetCDFMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	Boolean useVelSubsurface = false;
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index;
	LongPoint indices;
	long depthIndex1,depthIndex2;	// default to -1?
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!fVar.bShowArrows && !fVar.bShowGrid) return 0;
	err = /*CHECK*/dynamic_cast<NetCDFMover *>(this) -> SetInterval(errmsg);
	if(err) return false;
	
	if (fVar.arrowDepth>0 && fVar.gridType==TWO_D)
	{		
		if (fAllowVerticalExtrapolationOfCurrents && fMaxDepthForExtrapolation >= fVar.arrowDepth) useVelSubsurface = true;
		else
		{
			velocity.u = 0.;
			velocity.v = 0.;
			goto CalcStr;
		}
	}
	
	GetDepthIndices(0,fVar.arrowDepth,&depthIndex1,&depthIndex2);
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
		bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
		depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}
	
	if(/*OK*/dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()>1)
		//&& loaded && !err)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) /*return false;*/
		{
			if ((time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
			{
				timeAlpha = 1;
			}
			else
				return false;
		}
		else
			timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}
	//if (loaded && !err)
	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular
		
		indices = this->GetVelocityIndices(wp.p);
		
		if (index >= 0)
		{
			// Check for constant current 
			if(/*OK*/dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
					//velocity.u = this->GetStartUVelocity(index);
					//velocity.v = this->GetStartVVelocity(index);
					velocity.u = this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols);
					velocity.v = this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols);
				}
				else
				{
					velocity.u = depthAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*this->GetStartUVelocity(index+depthIndex2*fNumRows*fNumCols);
					velocity.v = depthAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*this->GetStartVVelocity(index+depthIndex2*fNumRows*fNumCols);
				}
			}
			else // time varying current
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
					//velocity.u = timeAlpha*this->GetStartUVelocity(index) + (1-timeAlpha)*this->GetEndUVelocity(index);
					//velocity.v = timeAlpha*this->GetStartVVelocity(index) + (1-timeAlpha)*this->GetEndVVelocity(index);
					velocity.u = timeAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex1*fNumRows*fNumCols);
					velocity.v = timeAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex1*fNumRows*fNumCols);
				}
				else	// below surface velocity
				{
					velocity.u = depthAlpha*(timeAlpha*this->GetStartUVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex1*fNumRows*fNumCols));
					velocity.u += (1-depthAlpha)*(timeAlpha*this->GetStartUVelocity(index+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndUVelocity(index+depthIndex2*fNumRows*fNumCols));
					velocity.v = depthAlpha*(timeAlpha*this->GetStartVVelocity(index+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex1*fNumRows*fNumCols));
					velocity.v += (1-depthAlpha)*(timeAlpha*this->GetStartVVelocity(index+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*this->GetEndVVelocity(index+depthIndex2*fNumRows*fNumCols));
				}
			}
		}
	}
	
CalcStr:
	
	/*if (this->fOffset_u != 0 && velocity.u!=0&& velocity.v!=0) 
	 {
	 velocity.u = this->fCurScale_u * velocity.u + this->fOffset_u; 
	 velocity.v = this->fCurScale_v * velocity.v + this->fOffset_v;
	 lengthU = lengthS = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	 }
	 else
	 {*/
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fVar.curScale * lengthU;
	//}
	//if (this->fVar.offset != 0 && lengthS!=0) lengthS += this->fVar.offset;
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
				this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
				this->className, uStr, sStr);
	}
	
	return true;
}


float NetCDFMover_c::GetMaxDepth()
{
	float maxDepth = 0;
	if (fDepthsH)
	{
		float depth=0;
		long i,numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
		for (i=0;i<numDepths;i++)
		{
			depth = INDEXH(fDepthsH,i);
			if (depth > maxDepth) 
				maxDepth = depth;
		}
		return maxDepth;
	}
	else
	{
		long numDepthLevels = /*CHECK*/dynamic_cast<NetCDFMover *>(this)->GetNumDepthLevelsInFile();
		if (numDepthLevels<=0) return maxDepth;
		if (fDepthLevelsHdl) maxDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	}
	return maxDepth;
}

float NetCDFMover_c::GetTotalDepth(WorldPoint wp, long triNum)
{	// z grid only 
#pragma unused(wp)
#pragma unused(triNum)
	long numDepthLevels = /*CHECK*/dynamic_cast<NetCDFMover *>(this)->GetNumDepthLevelsInFile();
	float totalDepth = 0;
	if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	return totalDepth;
}

void NetCDFMover_c::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData = 0;
	long numDepthLevels = /*CHECK*/dynamic_cast<NetCDFMover *>(this)->GetNumDepthLevelsInFile();
	float totalDepth = 0;
	
	
	if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	else
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}
	/*	switch(fVar.gridType) 
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
	 case SIGMA: // */
	if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
	{
		long j;
		for(j=0;j<numDepthLevels-1;j++)
		{
			if(INDEXH(fDepthLevelsHdl,indexToDepthData+j)<depthAtPoint &&
			   depthAtPoint<=INDEXH(fDepthLevelsHdl,indexToDepthData+j+1))
			{
				*depthIndex1 = indexToDepthData+j;
				*depthIndex2 = indexToDepthData+j+1;
			}
			else if(INDEXH(fDepthLevelsHdl,indexToDepthData+j)==depthAtPoint)
			{
				*depthIndex1 = indexToDepthData+j;
				*depthIndex2 = UNASSIGNEDINDEX;
			}
		}
		if(INDEXH(fDepthLevelsHdl,indexToDepthData)==depthAtPoint)	// handles single depth case
		{
			*depthIndex1 = indexToDepthData;
			*depthIndex2 = UNASSIGNEDINDEX;
		}
		else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)<depthAtPoint)
		{
			*depthIndex1 = indexToDepthData+numDepthLevels-1;
			*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
		}
		else if(INDEXH(fDepthLevelsHdl,indexToDepthData)>depthAtPoint)
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
	//break;
	/*default:
	 *depthIndex1 = UNASSIGNEDINDEX;
	 *depthIndex2 = UNASSIGNEDINDEX;
	 break;
	 }*/
}