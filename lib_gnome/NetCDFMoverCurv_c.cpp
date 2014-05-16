/*
 *  NetCDFMoverCurv_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

#include "NetCDFMoverCurv_c.h"
#include "netcdf.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include "DagTree.h"
#include "DagTreeIO.h"

NetCDFMoverCurv_c::NetCDFMoverCurv_c (TMap *owner, char *name) : NetCDFMover_c(owner, name)
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	bIsCOOPSWaterMask = false;
}	

Boolean NetCDFMoverCurv_c::IsCOOPSFile()
{
	OSErr err = 0;
	Boolean isCOOPSFile = false;
	long i,j,k;
	char path[256], outPath[256];
	int status, ncid;
	int mask_id;
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR)
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; return false;}
	}

	status = nc_inq_varid(ncid, "coops_mask", &mask_id);
	if (status != NC_NOERR)	{isCOOPSFile = false;}
	else {isCOOPSFile = true; bIsCOOPSWaterMask = true;}

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; return isCOOPSFile;}

	return 	isCOOPSFile;

}

LongPointHdl NetCDFMoverCurv_c::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

long NetCDFMoverCurv_c::GetVelocityIndex(WorldPoint wp)
{
	long index = -1;
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		if (bIsCOOPSWaterMask)
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp,fVerdatToNetCDFH,fNumCols);// curvilinear grid
		else
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	return index;
}

LongPoint NetCDFMoverCurv_c::GetVelocityIndices(WorldPoint wp)
{
	LongPoint indices={-1,-1};
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		if (bIsCOOPSWaterMask)
			indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(wp,fVerdatToNetCDFH,fNumCols);// curvilinear grid
		else
			indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	return indices;
}

Boolean NetCDFMoverCurv_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{	// code goes here, this is triangle code, not curvilinear
	char uStr[32],vStr[32],sStr[32],depthStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth, totalDepth = 0.;
	long index = 0;
	LongPoint indices;
	
	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;
	long depthIndex1,depthIndex2;	// default to -1?
	
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!fVar.bShowArrows && !fVar.bShowGrid) return 0;
	err = dynamic_cast<NetCDFMoverCurv *>(this) -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
	if(err) return false;
	
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		if (bIsCOOPSWaterMask)
		{
			//code goes here, put in interpolation
			interpolationVal = fGrid -> GetInterpolationValues(wp.p);
			if (interpolationVal.ptIndex1<0) return false;
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols);// curvilinear grid
		}
		else
			index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (index < 0) return 0;
		indices = this->GetVelocityIndices(wp.p);
	}
	else return 0;
	totalDepth = GetTotalDepth(wp.p,index);
	GetDepthIndices(index,fVar.arrowDepth,totalDepth,&depthIndex1,&depthIndex2);
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return false;	// no value for this point at chosen depth - should show as 0,0 or nothing?
	
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		/*if (fDepthsH)
		 {
		 totalDepth = INDEXH(fDepthsH,index);
		 }
		 else 
		 {
		 totalDepth = 0;
		 }*/
		//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
		//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
		topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
		bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth); // times totalDepth
		//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
		//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
		if (totalDepth == 0) depthAlpha = 1;
		else
			depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
	}
	
	// Check for constant current 
	if((dynamic_cast<NetCDFMoverCurv *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMoverCurv *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//velocity.u = INDEXH(fStartData.dataHdl,index).u;
			//velocity.v = INDEXH(fStartData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				velocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}	
			else
			{
				velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u;
				velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v;
			}
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
				velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
				velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
			}
			else
			{
				velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u);
				velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).u);
				velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v);
				velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex2*fNumRows*fNumCols).v);
			}
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v) * fFileScaleFactor;
	//lengthS = this->fWindScale * lengthU;
	//lengthS = this->fVar.curScale * lengthU;
	//lengthS = this->fVar.curScale * fFileScaleFactor * lengthU;
	lengthS = this->fVar.curScale * lengthU;
	if (lengthS > 1000000 || this->fVar.curScale==0) return true;	// if bad data in file causes a crash
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	if (indices.h >= 0 && fNumRows-indices.v-1 >=0 && indices.h < fNumCols && fNumRows-indices.v-1 < fNumRows)
	{
		//sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
		//this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
		StringWithoutTrailingZeros(uStr,fFileScaleFactor*velocity.u,4);
		StringWithoutTrailingZeros(vStr,fFileScaleFactor*velocity.v,4);
		StringWithoutTrailingZeros(depthStr,depthIndex1,4);
		sprintf(diagnosticStr, " [grid: %s, u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld]",
				this->className, uStr, vStr, fNumRows-indices.v-1, indices.h);
		//if (depthIndex1>0 || !(depthIndex2==UNASSIGNEDINDEX))
		if (fVar.gridType!=TWO_D)
			sprintf(diagnosticStr, " [grid: %s, u vel: %s m/s, v vel: %s m/s], file indices : [%ld, %ld, %ld], total depth : %lf",
					this->className, uStr, vStr, fNumRows-indices.v-1, indices.h, depthIndex1, totalDepth);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
				this->className, uStr, sStr);
	}
	
	return true;
}

WorldPoint3D NetCDFMoverCurv_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depthAlpha, depth = (*theLE).z;
	float topDepth, bottomDepth;
	long index = -1, depthIndex1, depthIndex2; 
	float totalDepth; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	InterpolationVal interpolationVal;
	OSErr err = 0;
	char errmsg[256];
	
	// might want to check for fFillValue and set velocity to zero - shouldn't be an issue unless we interpolate
	if(!fIsOptimizedForStep) 
	{
		err = dynamic_cast<NetCDFMoverCurv *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		if (bIsCOOPSWaterMask)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			interpolationVal = fGrid -> GetInterpolationValues(refPoint);
			if (interpolationVal.ptIndex1<0) return deltaPoint;
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	if (index < 0) return deltaPoint;
	
	totalDepth = GetTotalDepth(refPoint,index);
	if (index>=0)
		GetDepthIndices(index,depth,totalDepth,&depthIndex1,&depthIndex2);	// if not ?? point is off grid but not beached (map mismatch)
	else 
		return deltaPoint;
	if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
		return deltaPoint;	// no value for this point at chosen depth - should this be an error? question of show currents below surface vs an actual LE moving
	
	if (depthIndex2!=UNASSIGNEDINDEX)
	{
		// Calculate the depth weight factor
		/*if (fDepthsH)
		 {
		 totalDepth = INDEXH(fDepthsH,index);
		 }
		 else 
		 {
		 totalDepth = 0;
		 }*/
		//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
		//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
		topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
		bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth);
		//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
		//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
		if (totalDepth == 0) depthAlpha = 1;
		else
			depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
	}
	
	// Check for constant current 
	if((dynamic_cast<NetCDFMoverCurv *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMoverCurv *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			//scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
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
		if (dynamic_cast<NetCDFMoverCurv *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0 && depthIndex1 >= 0) 
		{
			//scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
			if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
			{
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
	
	scaledPatVelocity.u *= fVar.curScale; // may want to allow some sort of scale factor, though should be in file
	scaledPatVelocity.v *= fVar.curScale; 
	scaledPatVelocity.u *= fFileScaleFactor; // may want to allow some sort of scale factor, though should be in file
	scaledPatVelocity.v *= fFileScaleFactor; 
	
	
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


float NetCDFMoverCurv_c::GetTotalDepthFromTriIndex(long triNum)
{
	long index1, index2, index3, index4, numDepths;
	OSErr err = 0;
	float totalDepth = 0;
	Boolean useTriNum = true;
	WorldPoint refPoint = {0.,0.};
	
	if (fVar.gridType == SIGMA_ROMS)	// should always be true
	{
		//if (triNum < 0) useTriNum = false;
		err = ((TTriGridVel*)fGrid)->GetRectCornersFromTriIndexOrPoint(&index1, &index2, &index3, &index4, refPoint, triNum, useTriNum, fVerdatToNetCDFH, fNumCols+1);
		
		if (err) return 0;
		if (fDepthsH)
		{	// issue with extended grid not having depths - probably need to rework that idea
			long numCorners = 4;
			numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
			if (index1<numDepths && index1>=0) totalDepth += INDEXH(fDepthsH,index1); else numCorners--;
			if (index2<numDepths && index2>=0) totalDepth += INDEXH(fDepthsH,index2); else numCorners--;
			if (index3<numDepths && index3>=0) totalDepth += INDEXH(fDepthsH,index3); else numCorners--;
			if (index4<numDepths && index4>=0) totalDepth += INDEXH(fDepthsH,index4); else numCorners--;
			if (numCorners>0)
				totalDepth = totalDepth/(float)numCorners;
		}
	}
	//else totalDepth = INDEXH(fDepthsH,ptIndex);
	return totalDepth;
	
}
float NetCDFMoverCurv_c::GetTotalDepth2(WorldPoint refPoint)
{
	long index;
	float totalDepth = 0;
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		if (bIsCOOPSWaterMask)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			InterpolationVal interpolationVal = fGrid -> GetInterpolationValues(refPoint);
			if (interpolationVal.ptIndex1<0) return totalDepth;
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else
			index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (fDepthsH) totalDepth = INDEXH(fDepthsH,index);
	}
	return totalDepth;
}

float NetCDFMoverCurv_c::GetTotalDepth(WorldPoint refPoint,long ptIndex)
{
	long index1, index2, index3, index4, numDepths;
	OSErr err = 0;
	float totalDepth = 0;
	Boolean useTriNum = false;
	long triNum = 0;
	
	if (fVar.gridType == SIGMA_ROMS)
	{
		//if (triNum < 0) useTriNum = false;
		if (bIsCOOPSWaterMask)
			err = ((TTriGridVel*)fGrid)->GetRectCornersFromTriIndexOrPoint(&index1, &index2, &index3, &index4, refPoint, triNum, useTriNum, fVerdatToNetCDFH, fNumCols);
		else 
			err = ((TTriGridVel*)fGrid)->GetRectCornersFromTriIndexOrPoint(&index1, &index2, &index3, &index4, refPoint, triNum, useTriNum, fVerdatToNetCDFH, fNumCols+1);
		
		//if (err) return 0;
		if (err) return -1;
		if (fDepthsH)
		{	// issue with extended grid not having depths - probably need to rework that idea
			long numCorners = 4;
			numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
			if (index1<numDepths && index1>=0) totalDepth += INDEXH(fDepthsH,index1); else numCorners--;
			if (index2<numDepths && index2>=0) totalDepth += INDEXH(fDepthsH,index2); else numCorners--;
			if (index3<numDepths && index3>=0) totalDepth += INDEXH(fDepthsH,index3); else numCorners--;
			if (index4<numDepths && index4>=0) totalDepth += INDEXH(fDepthsH,index4); else numCorners--;
			if (numCorners>0)
				totalDepth = totalDepth/(float)numCorners;
		}
	}
	else 
	{
		double depthAtPoint = 0;
		long ptIndex1,ptIndex2,ptIndex3; 
		float depth1,depth2,depth3;
		InterpolationVal interpolationVal;
		FLOATH depthsHdl = 0;
		
		interpolationVal = ((TTriGridVel*)fGrid)->GetInterpolationValues(refPoint);
		if (fDepthsH) 
		{
			totalDepth = INDEXH(fDepthsH,ptIndex); 
			return totalDepth;
		}
		if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)
			//depthsHdl = ((TTriGridVel3D*)fGrid)->GetDepths();
			depthsHdl = dynamic_cast<TriGridVel3D_c *>(fGrid) -> GetDepths();
		if (!depthsHdl || interpolationVal.ptIndex1<0 ) 
			//return -1;	// some error alert, no depth info to check
			return totalDepth;	// assume 2D
		//if (fDepthsH)
		{
			depth1 = (*depthsHdl)[interpolationVal.ptIndex1];
			depth2 = (*depthsHdl)[interpolationVal.ptIndex2];
			depth3 = (*depthsHdl)[interpolationVal.ptIndex3];
			totalDepth = interpolationVal.alpha1*depth1 + interpolationVal.alpha2*depth2 + interpolationVal.alpha3*depth3;
		}
	}	
	//else 
	//{
		//if (fDepthsH) totalDepth = INDEXH(fDepthsH,ptIndex);
	//}
	return totalDepth;
	
}
void NetCDFMoverCurv_c::GetDepthIndices(long ptIndex, float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2)
{
	// probably eventually switch to NetCDFMover only
	long indexToDepthData = 0;
	long numDepthLevels = GetNumDepthLevelsInFile();
	//float totalDepth = 0;
	//FLOATH depthsH = ((TTriGridVel3D*)fGrid)->GetDepths();
	
	/*if (fDepthsH)
	 {
	 totalDepth = INDEXH(fDepthsH,ptIndex);
	 }
	 else*/
	if (totalDepth==0)
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}
	
	if (fDepthLevelsHdl && numDepthLevels>0) 
	{
		/*if (fVar.gridType==MULTILAYER)
			totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);*/
		// otherwise it's SIGMA_ROMS
	}
	else
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}
	switch(fVar.gridType) 
	{	// function should not be called for TWO_D, haven't used BAROTROPIC yet
			/*case TWO_D:	// no depth data
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
			 break;*/
			//case MULTILAYER: //
		/*case MULTILAYER: //
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// if depths are measured from the bottom this is confusing
				long j;
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
			break;*/
		case MULTILAYER: // 
			if (depthAtPoint<0)
			{	// what is this?
				//*depthIndex1 = indexToDepthData+numDepthLevels-1;
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				return;
			}
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// is sigma always 0-1 ?
				long j;
				float depthAtLevel, depthAtNextLevel;
				for(j=0;j<numDepthLevels-1;j++)
				{
					depthAtLevel = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					depthAtNextLevel = INDEXH(fDepthLevelsHdl,indexToDepthData+j+1);
					if(depthAtLevel<depthAtPoint &&
					   depthAtPoint<=depthAtNextLevel)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
						return;
					}
					else if(depthAtLevel==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
						return;
					}
				}
				if(INDEXH(fDepthLevelsHdl,indexToDepthData)==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
					return;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
					return;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
					return;
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
				return;
			}
			break;
			//break;
		case SIGMA: // 
			// code goes here, add SIGMA_ROMS, using z[k,:,:] = hc * (sc_r-Cs_r) + Cs_r * depth
			if (depthAtPoint<0)
			{	// keep in mind for grids with values at the bottom (rather than mid-cell) they may all be zero
				*depthIndex1 = indexToDepthData+numDepthLevels-1;
				*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				return;
			}
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// is sigma always 0-1 ?
				long j;
				float sigma, sigmaNext, depthAtLevel, depthAtNextLevel;
				for(j=0;j<numDepthLevels-1;j++)
				{
					sigma = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					sigmaNext = INDEXH(fDepthLevelsHdl,indexToDepthData+j+1);
					depthAtLevel = sigma * totalDepth;
					depthAtNextLevel = sigmaNext * totalDepth;
					if(depthAtLevel<depthAtPoint &&
					   depthAtPoint<=depthAtNextLevel)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = indexToDepthData+j+1;
						return;
					}
					else if(depthAtLevel==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
						return;
					}
				}
				if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth==depthAtPoint)	// handles single depth case
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX;
					return;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)*totalDepth<depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
					return;
				}
				else if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
					return;
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
				return;
			}
			//break;
		case SIGMA_ROMS: // 
			// code goes here, add SIGMA_ROMS, using z[k,:,:] = hc * (sc_r-Cs_r) + Cs_r * depth
			//WorldPoint wp; 
			//long triIndex;
			//totalDepth = GetTotalDepth(wp,triIndex);
			if (depthAtPoint<0)
			{
				*depthIndex1 = indexToDepthData;
				*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
				return;
			}
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{	// is sigma always 0-1 ?
				long j;
				float sc_r, sc_r2, Cs_r, Cs_r2, depthAtLevel, depthAtNextLevel;
				//for(j=0;j<numDepthLevels-1;j++)
				for(j=numDepthLevels-1;j>0;j--)
				{
					// sc and Cs are negative so need abs value
					/*float sc_r = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					 float sc_r2 = INDEXH(fDepthLevelsHdl,indexToDepthData+j+1);
					 float Cs_r = INDEXH(fDepthLevelsHdl2,indexToDepthData+j);
					 float Cs_r2 = INDEXH(fDepthLevelsHdl2,indexToDepthData+j+1);*/
					sc_r = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
					sc_r2 = INDEXH(fDepthLevelsHdl,indexToDepthData+j-1);
					Cs_r = INDEXH(fDepthLevelsHdl2,indexToDepthData+j);
					Cs_r2 = INDEXH(fDepthLevelsHdl2,indexToDepthData+j-1);
					//depthAtLevel = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
					//depthAtNextLevel = abs(hc * (sc_r2-Cs_r2) + Cs_r2 * totalDepth);
					depthAtLevel = myfabs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
					depthAtNextLevel = myfabs(totalDepth*(hc*sc_r2+totalDepth*Cs_r2))/(totalDepth+hc);
					if(depthAtLevel<depthAtPoint &&
					   depthAtPoint<=depthAtNextLevel)
					{
						*depthIndex1 = indexToDepthData+j;
						//*depthIndex2 = indexToDepthData+j+1;
						*depthIndex2 = indexToDepthData+j-1;
						return;
					}
					else if(depthAtLevel==depthAtPoint)
					{
						*depthIndex1 = indexToDepthData+j;
						*depthIndex2 = UNASSIGNEDINDEX;
						return;
					}
				}
				//if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth==depthAtPoint)	// handles single depth case
				if(GetDepthAtIndex(indexToDepthData+numDepthLevels-1,totalDepth)==depthAtPoint)	// handles single depth case
					//if(GetTopDepth(indexToDepthData+numDepthLevels-1,totalDepth)==depthAtPoint)	// handles single depth case
				{
					//*depthIndex1 = indexToDepthData;
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX;
					return;
				}
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)*totalDepth<depthAtPoint)
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth<depthAtPoint)	// 0 is bottom
				else if(GetDepthAtIndex(indexToDepthData,totalDepth)<depthAtPoint)	// 0 is bottom
					//else if(GetBottomDepth(indexToDepthData,totalDepth)<depthAtPoint)	// 0 is bottom
				{
					*depthIndex1 = indexToDepthData;
					*depthIndex2 = UNASSIGNEDINDEX; //BOTTOM, for now just extrapolate lowest depth value (at bottom case?)
					return;
				}
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData)*totalDepth>depthAtPoint)
				//else if(INDEXH(fDepthLevelsHdl,indexToDepthData+numDepthLevels-1)*totalDepth>depthAtPoint)
				else if(GetDepthAtIndex(indexToDepthData+numDepthLevels-1,totalDepth)>depthAtPoint)
					//else if(GetTopDepth(indexToDepthData+numDepthLevels-1,totalDepth)>depthAtPoint)
				{
					*depthIndex1 = indexToDepthData+numDepthLevels-1;
					*depthIndex2 = UNASSIGNEDINDEX; //TOP, for now just extrapolate highest depth value
					return;
				}
			}
			else // no data at this point
			{
				*depthIndex1 = UNASSIGNEDINDEX;
				*depthIndex2 = UNASSIGNEDINDEX;
				return;
			}
			break;
		default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}
}


//OSErr NetCDFMoverCurv_c::ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg) 
OSErr NetCDFMoverCurv_c::ReorderPoints(DOUBLEH landmaskH, TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long iIndex,jIndex,index,currentIndex,startIndex; 
	long triIndex1,triIndex2,waterCellNum=0;
	long ptIndex = 0,cellNum = 0,diag = 1;
	Boolean foundPt = false, isOdd;
	OSErr err = 0;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	
	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;
	
	//TTriGridVel *triGrid = nil;
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	
	// write out verdat file for debugging
	//FILE *outfile = 0;
	//char name[32], path[256],m[300];
	//strcpy(name,"NewVerdat.dat");
	//errmsg[0]=0;
	
	//err = AskUserForSaveFilename(name,path,".dat",true);
	//if(err) return USERCANCEL; 
	//strcpy(sExportSelectedTriPath, path); // remember the path for the user
	//SetWatchCursor();
	//sprintf(m, "Exporting VERDAT to %s...",path);
	//DisplayMessage("NEXTMESSAGETEMP");
	//DisplayMessage(m);
	/////////////////////////////////////////////////
	
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			// eventually will need to have a land mask
			//if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
			//if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
			if (INDEXH(landmaskH,i*fNumCols+j)==0)	// land point
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				//float dLat = INDEXH(fVertexPtsH,i*fNumCols+j).pLat;
				//float dLon = INDEXH(fVertexPtsH,i*fNumCols+j).pLong;
				//long index = i*fNumCols+j+1;
				//float dZ = 1.;
				//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, dLon, dLat, dZ);	
				INDEXH(landWaterInfo,i*fNumCols+j) = 1;
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*fNumCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1) = -2;
			}
		}
	}
	
	//fclose(outfile);
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*fNumCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)>0)
			{
				INDEXH(gridCellInfo,i*fNumCols+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*fNumCols+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols_ext+j);
				INDEXH(gridCellInfo,i*fNumCols+j).topRight = INDEXH(ptIndexHdl,i*fNumCols_ext+j+1);
				INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j);
				INDEXH(gridCellInfo,i*fNumCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1);
			}
			else INDEXH(gridCellInfo,i*fNumCols+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv_ext;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			//INDEXH(verdatPtsH,INDEXH(ptIndexHdl,i)) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/fNumCols_ext;
			jIndex = n%fNumCols_ext;
			if (iIndex==0)
			{
				if (jIndex<fNumCols)
				{
					dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
					fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
					fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
				}
				else
				{
					dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
				}
			}
			else 
			{
				if (jIndex<fNumCols)
				{
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
					//u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
					//v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
				}
				else
				{
					dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f,%.2f,%.2f\n", index, fLong, fLat, fDepth, u, v);	
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		/////////////////////////////////////////////////
		
	}
	//fclose(outfile);
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
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
			}
		}
	}
	// wondering if for regular grids using the curvilinear format it would be better to have 
	// the diagonals not all be the same - it seems to cause the dagtree to not work
	// would also need to change the island numbering stuff to make sure boundary gets done correctly
	/*for (i=0;i<fNumRows;i++)
	 {
	 for (j=0;j<fNumCols;j++)
	 {
	 if (diag>0)
	 {
	 if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
	 {diag = diag*-1; continue;}
	 waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
	 triIndex1 = 2*waterCellNum;
	 triIndex2 = 2*waterCellNum+1;
	 // top/left tri in rect
	 (*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
	 (*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
	 (*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
	 if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
	 (*topo)[triIndex1].adjTri1 = -1;
	 else
	 {
	 (*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
	 }
	 (*topo)[triIndex1].adjTri2 = triIndex2;
	 if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
	 (*topo)[triIndex1].adjTri3 = -1;
	 else
	 {
	 (*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
	 }
	 // bottom/right tri in rect
	 (*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
	 (*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
	 (*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
	 if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
	 (*topo)[triIndex2].adjTri1 = -1;
	 else
	 {
	 (*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
	 }
	 (*topo)[triIndex2].adjTri2 = triIndex1;
	 if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
	 (*topo)[triIndex2].adjTri3 = -1;
	 else
	 {
	 (*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
	 }
	 }
	 else
	 {
	 if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
	 {diag = diag*-1; continue;}
	 waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
	 triIndex1 = 2*waterCellNum;
	 triIndex2 = 2*waterCellNum+1;
	 // bot/left tri in rect
	 (*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
	 (*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
	 (*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
	 if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
	 (*topo)[triIndex1].adjTri1 = -1;
	 else
	 {
	 (*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
	 }
	 (*topo)[triIndex1].adjTri2 = triIndex2;
	 if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
	 (*topo)[triIndex1].adjTri3 = -1;
	 else
	 {
	 (*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
	 }
	 // top/right tri in rect
	 (*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
	 (*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
	 (*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
	 if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
	 (*topo)[triIndex2].adjTri1 = -1;
	 else
	 {
	 (*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
	 }
	 (*topo)[triIndex2].adjTri2 = triIndex1;
	 if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
	 (*topo)[triIndex2].adjTri3 = -1;
	 else
	 {
	 (*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
	 }
	 }
	 diag = -1.*diag;
	 }
	 }*/
	
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	/*MySpinCursor(); // JLM 8/4/99
	 tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	 MySpinCursor(); // JLM 8/4/99
	 if (errmsg[0])	
	 {err = -1; goto done;} 
	 // sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	 _SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));*/
	/////////////////////////////////////////////////
	if (this -> moverMap != model -> uMap) goto setFields;	// don't try to create a map
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
	
	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**waterBoundaryPtsH));
	flagH = (LONGH)_NewHandleClear(nv_ext * sizeof(**flagH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Numbering Islands");
	MySpinCursor(); // JLM 8/4/99
	//err = NumberIslands(&maskH2, velocityH, landWaterInfo, fNumRows, fNumCols, &numIslands);	// numbers start at 3 (outer boundary)
	err = NumberIslands(&maskH2, landmaskH, landWaterInfo, fNumRows, fNumCols, &numIslands);	// numbers start at 3 (outer boundary)
	MySpinCursor(); // JLM 8/4/99
	if (err) goto done;
	for (i=0;i<ntri;i++)
	{
		if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
		// the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
		// odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
		if ((*topo)[i].adjTri1 == -1)
		{
			// add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
			(*segList)[segNum].pt1 = (*topo)[i].vertex2;
			(*segList)[segNum].pt2 = (*topo)[i].vertex3;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check left rectangle for L/W border 
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (jIndex>0 && INDEXH(maskH2,iIndex*fNumCols_ext + jIndex-1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols_ext + jIndex-1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			else 
			{	
				// check right rectangle for L/W border convert back to row/col
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (jIndex<fNumCols && INDEXH(maskH2,iIndex*fNumCols_ext + jIndex+1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols_ext + jIndex+1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			segNum++;
		}
		
		if ((*topo)[i].adjTri3 == -1)
		{
			// add segment pt 1 - pt 2 to list
			// odd top, even bottom
			(*segList)[segNum].pt1 = (*topo)[i].vertex1;
			(*segList)[segNum].pt2 = (*topo)[i].vertex2;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check top rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*fNumCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*fNumCols_ext + jIndex);
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;
				}
			}
			else 
			{
				// check bottom rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/fNumCols_ext;
				jIndex = rectIndex%fNumCols_ext;
				if (iIndex<fNumRows && INDEXH(maskH2,(iIndex+1)*fNumCols_ext + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols_ext + jIndex);		// this should be the neighbor's value
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;		
				}
			}
			segNum++;
		}
	}
	nSegs = segNum;
	_SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	_SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	// go through list of segments, and make list of boundary segments
	// as segment is taken mark so only use each once
	// get a starting point, add the first and second to the list
	islandNum = 3;
findnewstartpoint:
	if (islandNum > numIslands) 
	{
		_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	}
	foundPt = false;
	for (i=0;i<nSegs;i++)
	{
		if ((*segUsed)[i]) continue;
		waterStartPoint = nBoundaryPts;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		(*flagH)[(*segList)[i].pt1] = 1;
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		(*flagH)[(*segList)[i].pt2] = 1;
		currentIndex = (*segList)[i].pt2;
		startIndex = (*segList)[i].pt1;
		currentIsland = (*segList)[i].islandNumber;	
		foundPt = true;
		(*segUsed)[i] = true;
		break;
	}
	if (!foundPt)
	{
		printNote("Lost trying to set boundaries");
		// clean up handles and set grid without a map
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		goto setFields;
	}
	
findnextpoint:
	for (i=0;i<nSegs;i++)
	{
		// look for second point of the previous selected segment, add the second to point list
		if ((*segUsed)[i]) continue;
		if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		index = (*segList)[i].pt1;
		if (index == currentIndex)	// found next point
		{
			currentIndex = (*segList)[i].pt2;
			(*segUsed)[i] = true;
			if (currentIndex == startIndex) // completed a segment
			{
				islandNum++;
				(*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				(*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				goto findnewstartpoint;
			}
			else
			{
				(*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				(*flagH)[(*segList)[i].pt2] = 1;
				(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				nBoundaryPts++;
				goto findnextpoint;
			}
		}
	}
	// shouldn't get here unless there's a problem...
	_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
	
setFields:	
	
	fVerdatToNetCDFH = verdatPtsH;
	
	
	{	//for debugging
		/////////////////////////////////////////////////
		// write out the file
		/////////////////////////////////////////////////
		/*outfile=fopen(path,"w");
		 if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
		 fprintf(outfile,"DOGS\tMETERS\n");
		 
		 float fLong, fLat, fDepth = 1.0;
		 long index, index1, startver, endver, count = 0;
		 LongPoint vertex;
		 
		 for(i = 0; i < nEndPts; i++)
		 {
		 // boundary points may be used in more than one segment, this will mess up verdat 
		 startver = i == 0? 0: (*boundaryEndPtsH)[i-1] + 1;
		 endver = (*boundaryEndPtsH)[i]+1;
		 index1 = (*boundaryPtsH)[startver];
		 vertex = (*pts)[index1];
		 fLong = ((float)vertex.h) / 1e6;
		 fLat = ((float)vertex.v) / 1e6;
		 count++;
		 fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", count, fLong, fLat, fDepth);	
		 for(j = startver + 1; j < endver; j++)
		 {
		 index = (*boundaryPtsH)[j];
		 vertex = (*pts)[index];
		 fLong = ((float)vertex.h) / 1e6;
		 fLat = ((float)vertex.v) / 1e6;
		 count++;
		 fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", count, fLong, fLat, fDepth);	
		 }
		 }
		 for (i = 0; i < numVerdatPts; i++)
		 {
		 if ((*flagH)[i] == 1) continue;
		 count++;
		 vertex = (*pts)[i];
		 fLong = ((float)vertex.h) / 1e6;
		 fLat = ((float)vertex.v) / 1e6;
		 fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", count, fLong, fLat, fDepth);	
		 }
		 fprintf(outfile, "0,0.,0.,0.\n");	
		 
		 // write out the number of boundary segments
		 fprintf(outfile,"%ld\n",nEndPts);
		 
		 // now write out out the break points
		 for(i = 0; i < nEndPts; i++ )
		 {
		 fprintf(outfile,"%ld\n",INDEXH(boundaryEndPtsH,i)+1);
		 }
		 /////////////////////////////////////////////////
		 
		 fclose(outfile);*/
	}
	/////////////////////////////////////////////////
	
	//triGrid = new TTriGridVel;
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverCurv::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}
	
	//fGrid = (TTriGridVel*)triGrid;
	fGrid = (TTriGridVel3D*)triGrid;
	
	triGrid -> SetBounds(triBounds); 
	
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
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
	
	if (waterBoundaryPtsH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundaryEndPtsH);	
		map->SetWaterBoundaries(waterBoundaryPtsH);
		map->SetBoundaryPoints(boundaryPtsH);
		
		*newMap = map;
	}
	else
	{
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH=0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH=0;}
	}
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
	if (segList) {DisposeHandle((Handle)segList); segList = 0;}
	if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverCurv::ReorderPoints");
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
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
		
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}
	return err;
}

// simplify for codar data - no map needed, no mask 
//OSErr NetCDFMoverCurv_c::ReorderPointsNoMask(VelocityFH velocityH, TMap **newMap, char* errmsg) 
OSErr NetCDFMoverCurv_c::ReorderPointsNoMask(TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
	long indexOfStart = 0;
	OSErr err = 0;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	
	//TTriGridVel *triGrid = nil;
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	VelocityFH velocityH = 0;
	// write out verdat file for debugging
	/*FILE *outfile = 0;
	 char name[32], path[256],m[300];
	 strcpy(name,"NewVerdat.dat");
	 errmsg[0]=0;
	 
	 err = AskUserForSaveFilename(name,path,".dat",true);
	 if(err) return USERCANCEL; 
	 
	 SetWatchCursor();
	 sprintf(m, "Exporting VERDAT to %s...",path);
	 DisplayMessage("NEXTMESSAGETEMP");
	 DisplayMessage(m);*/
	/////////////////////////////////////////////////
	
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);	// try to use velocities to set grid

	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			// eventually will need to have a land mask, for now assume fillValue represents land
			if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
				// if use fill_value need to be sure to check for it in GetMove and VelocityStrAtPoint
				//if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				INDEXH(landWaterInfo,i*fNumCols+j) = 1;
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -2;	// water box
				INDEXH(ptIndexHdl,i*fNumCols_ext+j+1) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j) = -2;
				INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1) = -2;
			}
		}
	}
	
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (INDEXH(ptIndexHdl,i*fNumCols_ext+j) == -2)
			{
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*fNumCols_ext+j) = -1;
		}
	}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)>0)
			{
				INDEXH(gridCellInfo,i*fNumCols+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*fNumCols+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols_ext+j);
				INDEXH(gridCellInfo,i*fNumCols+j).topRight = INDEXH(ptIndexHdl,i*fNumCols_ext+j+1);
				INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j);
				INDEXH(gridCellInfo,i*fNumCols+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols_ext+j+1);
			}
			else INDEXH(gridCellInfo,i*fNumCols+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv_ext;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			//INDEXH(verdatPtsH,INDEXH(ptIndexHdl,i)) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	index = 0;
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			//index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/fNumCols_ext;
			jIndex = n%fNumCols_ext;
			if (iIndex==0)
			{
				if (jIndex<fNumCols)
				{
					dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
					fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
					dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
					fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
				}
				else
				{
					dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
					dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
					fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
					dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
					dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
					fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
				}
			}
			else 
			{
				if (jIndex<fNumCols)
				{
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
					u = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).u;
					v = INDEXH(velocityH,(iIndex-1)*fNumCols+jIndex).v;
				}
				else
				{
					dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
					fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
					dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
					fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
				}
			}
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth, u, v);	
		//if (u!=0. && v!=0.) {index=index+1; fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);}	
		/////////////////////////////////////////////////
		
	}
	//fclose(outfile);
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
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*fNumCols+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols+j).topRight;
			if (j==fNumCols-1 || INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==fNumRows-1 || INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols+j).cellNum * 2;
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	fVerdatToNetCDFH = verdatPtsH;
	
	/////////////////////////////////////////////////
	
	//triGrid = new TTriGridVel;
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverCurv::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel3D*)triGrid;
	//fGrid = (TTriGridVel*)triGrid;
	
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
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverCurv::ReorderPoints");
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
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
	return err;
}

//OSErr NetCDFMoverCurv_c::ReorderPointsCOOPSMask(VelocityFH velocityH, TMap **newMap, char* errmsg) 
OSErr NetCDFMoverCurv_c::ReorderPointsCOOPSMask(DOUBLEH landmaskH, TMap **newMap, char* errmsg) 
{
	OSErr err = 0;
	long i,j,k;
	char path[256], outPath[256];
	char *velUnits=0; 
	int status, ncid, numdims;
	int mask_id, uv_ndims;
	//static size_t mask_index[] = {0,0};
	//static size_t mask_count[2];
	//double *landmask = 0, *mylandmask=0;
	//double debug_mask;
	long latlength = fNumRows, numtri = 0;
	long lonlength = fNumCols;
	Boolean isLandMask = true;
	float fDepth1, fLat1, fLong1;
	long index1=0;
	
	errmsg[0]=0;
	*newMap = 0;

	long n, ntri, numVerdatPts=0;
	long fNumRows_minus1 = fNumRows-1, fNumCols_minus1 = fNumCols-1;
	long nv = fNumRows * fNumCols;
	long nCells = fNumRows_minus1 * fNumCols_minus1;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;

	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long currentIndex,startIndex; 
	long diag = 1;
	Boolean foundPt = false, isOdd;
	
	LONGH landWaterInfo = (LONGH)_NewHandleClear(nCells * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nCells * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	
	LONGH boundaryPtsH = 0;
	LONGH boundaryEndPtsH = 0;
	LONGH waterBoundaryPtsH = 0;
	Boolean** segUsed = 0;
	SegInfoHdl segList = 0;
	LONGH flagH = 0;
	
	//TTriGridVel *triGrid = nil;
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	// write out verdat file for debugging
	/* FILE *outfile = 0;
	 char name[32], verdatpath[256],m[300];
	 strcpy(name,"NewVerdat.dat");
	 errmsg[0]=0;
	 
	 err = AskUserForSaveFilename(name,verdatpath,".dat",true);
	 if(err) return USERCANCEL; 
	 
	 SetWatchCursor();
	 sprintf(m, "Exporting VERDAT to %s...",verdatpath);
	 DisplayMessage("NEXTMESSAGETEMP");
	 DisplayMessage(m);*/
	/////////////////////////////////////////////////
	
	strcpy(path,fVar.pathName);
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
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	if (!landmaskH) return -1;
	/*mask_count[0] = latlength;
	mask_count[1] = lonlength;
	
	status = nc_inq_varid(ncid, "coops_mask", &mask_id);
	if (status != NC_NOERR)	{isLandMask = false;}
	if (isLandMask)
	{
		landmask = new double[latlength*lonlength]; 
		if(!landmask) {TechError("NetCDFMoverCurv::ReorderPointsCOOPSMask()", "new[]", 0); err = memFullErr; goto done;}
		mylandmask = new double[latlength*lonlength]; 
		if(!mylandmask) {TechError("NetCDFMoverCurv::ReorderPointsCOOPSMask()", "new[]", 0); err = memFullErr; goto done;}
	}
	if (isLandMask)
	{
		//status = nc_get_vara_float(ncid, mask_id, angle_index, angle_count, landmask);
		status = nc_get_vara_double(ncid, mask_id, mask_index, mask_count, landmask);
		if (status != NC_NOERR) {err = -1; goto done;}
	}*/
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}
	
	/*outfile=fopen(verdatpath,"w");
	if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	fprintf(outfile,"DOGS\tMETERS\n");*/

						
	/*for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			debug_mask = landmask[(latlength-i-1)*lonlength+j];
			//if (debug_mask == 1.1) numtri++;
			if (debug_mask > 0) 
			{
				numtri++;
			}
			// eventually will need to have a land mask, for now assume fillValue represents land
			if (landmask[(latlength-i-1)*fNumCols+j]==0)	// land point
			{
			}
			else
			{
				index1++;
				//fLat1 = INDEXH(fVertexPtsH,(iIndex)*fNumCols+jIndex).pLat;
				//fLong1 = INDEXH(fVertexPtsH,(iIndex)*fNumCols+jIndex).pLong;
				fLat1 = INDEXH(fVertexPtsH,(i)*fNumCols+j).pLat;
				fLong1 = INDEXH(fVertexPtsH,(i)*fNumCols+j).pLong;
			
			fDepth1 = 1.;				
			//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index1, fLong1, fLat1, fDepth1);	
			}

		}
	}*/
	//fclose(outfile);
	/*for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			mylandmask[i*lonlength+j] = landmask[(latlength-i-1)*lonlength+j];
		}
	}*/
	index1 = 0;
	for (i=0;i<fNumRows-1;i++)
	{
		for (j=0;j<fNumCols-1;j++)
		{
			//if (landmask[i*fNumCols_minus1+j]==0 && landmask[i*fNumCols_minus1+j]=0)	// land point
			//if (mylandmask[i*fNumCols+j]==0)	// land point
			if (INDEXH(landmaskH,i*fNumCols+j)==0)	// land point
			{
				INDEXH(landWaterInfo,i*fNumCols_minus1+j) = -1;	// may want to mark each separate island with a unique number
			}
			else
			{
				//if (landmask[(latlength-i)*fNumCols+j]==0 || landmask[(latlength-i-1)*fNumCols+j+1]==0 || landmask[(latlength-i)*fNumCols+j+1]==0)
				//if (mylandmask[(i+1)*fNumCols+j]==0 || mylandmask[i*fNumCols+j+1]==0 || mylandmask[(i+1)*fNumCols+j+1]==0)
				if (INDEXH(landmaskH,(i+1)*fNumCols+j)==0 || INDEXH(landmaskH,i*fNumCols+j+1)==0 || INDEXH(landmaskH,(i+1)*fNumCols+j+1)==0)
				{
					INDEXH(landWaterInfo,i*fNumCols_minus1+j) = -1;	// may want to mark each separate island with a unique number
				}
				else
				{
					INDEXH(landWaterInfo,i*fNumCols_minus1+j) = 1;
					INDEXH(ptIndexHdl,i*fNumCols+j) = -2;	// water box
					INDEXH(ptIndexHdl,i*fNumCols+j+1) = -2;
					INDEXH(ptIndexHdl,(i+1)*fNumCols+j) = -2;
					INDEXH(ptIndexHdl,(i+1)*fNumCols+j+1) = -2;
				}
			}
		}
	}
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(ptIndexHdl,i*fNumCols+j) == -2)
			{
				INDEXH(ptIndexHdl,i*fNumCols+j) = ptIndex;	// count up grid points
				ptIndex++;
			}
			else
				INDEXH(ptIndexHdl,i*fNumCols+j) = -1;
		}
	}
	
	for (i=0;i<fNumRows-1;i++)
	{
		for (j=0;j<fNumCols-1;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols_minus1+j)>0)
			{
				INDEXH(gridCellInfo,i*fNumCols_minus1+j).cellNum = cellNum;
				cellNum++;
				INDEXH(gridCellInfo,i*fNumCols_minus1+j).topLeft = INDEXH(ptIndexHdl,i*fNumCols+j);
				INDEXH(gridCellInfo,i*fNumCols_minus1+j).topRight = INDEXH(ptIndexHdl,i*fNumCols+j+1);
				INDEXH(gridCellInfo,i*fNumCols_minus1+j).bottomLeft = INDEXH(ptIndexHdl,(i+1)*fNumCols+j);
				INDEXH(gridCellInfo,i*fNumCols_minus1+j).bottomRight = INDEXH(ptIndexHdl,(i+1)*fNumCols+j+1);
			}
			else INDEXH(gridCellInfo,i*fNumCols_minus1+j).cellNum = -1;
		}
	}
	ntri = cellNum*2;	// each water cell is split into two triangles
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology)))){err = memFullErr; goto done;}	
	for (i=0;i<nv;i++)
	{
		if (INDEXH(ptIndexHdl,i) != -1)
		{
			INDEXH(verdatPtsH,numVerdatPts) = i;
			numVerdatPts++;
		}
	}
	_SetHandleSize((Handle)verdatPtsH,numVerdatPts*sizeof(**verdatPtsH));
	pts = (LongPointHdl)_NewHandle(sizeof(LongPoint)*(numVerdatPts));
	if(pts == nil)
	{
		strcpy(errmsg,"Not enough memory to triangulate data.");
		return -1;
	}
	
	/////////////////////////////////////////////////
	// write out the file
	/////////////////////////////////////////////////
	/*outfile=fopen(verdatpath,"w");
	if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	fprintf(outfile,"DOGS\tMETERS\n");*/
	index = 0;
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		double val, u=0., v=0.;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	// since velocities are defined at the lower left corner of each grid cell
			// need to add an extra row/col at the top/right of the grid
			// set lat/lon based on distance between previous two points 
			// these are just for boundary/drawing purposes, velocities are set to zero
			index = i+1;
			n = INDEXH(verdatPtsH,i);
			iIndex = n/fNumCols;
			jIndex = n%fNumCols;
			//fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
			//fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
			fLat = INDEXH(fVertexPtsH,(iIndex)*fNumCols+jIndex).pLat;
			fLong = INDEXH(fVertexPtsH,(iIndex)*fNumCols+jIndex).pLong;
			/*if (landmask[(latlength-iIndex-1)*fNumCols+jIndex]==0)	// land point
			{
				//index1++;
				u = INDEXH(velocityH,(iIndex)*fNumCols+jIndex).u;
				v = INDEXH(velocityH,(iIndex)*fNumCols+jIndex).v;
			}*/
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			fDepth = 1.;
			INDEXH(pts,i) = vertex;
		}
		else { // for outputting a verdat the last line should be all zeros
			index = 0;
			fLong = fLat = fDepth = 0.0;
		}
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth, u, v);	
		//fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);	
		//if (u!=0. && v!=0.) {index=index+1; fprintf(outfile, "%ld,%.6f,%.6f,%.6f\n", index, fLong, fLat, fDepth);}	
		/////////////////////////////////////////////////
		
	}
	//fclose(outfile);
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
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	
	/////////////////////////////////////////////////
	for (i=0;i<fNumRows_minus1;i++)
	{
		for (j=0;j<fNumCols_minus1;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols_minus1+j)==-1)
				continue;
			waterCellNum = INDEXH(gridCellInfo,i*fNumCols_minus1+j).cellNum;	// split each cell into 2 triangles
			triIndex1 = 2*waterCellNum;
			triIndex2 = 2*waterCellNum+1;
			// top/left tri in rect
			(*topo)[triIndex1].vertex1 = INDEXH(gridCellInfo,i*fNumCols_minus1+j).topRight;
			(*topo)[triIndex1].vertex2 = INDEXH(gridCellInfo,i*fNumCols_minus1+j).topLeft;
			(*topo)[triIndex1].vertex3 = INDEXH(gridCellInfo,i*fNumCols_minus1+j).bottomLeft;
			if (j==0 || INDEXH(gridCellInfo,i*fNumCols_minus1+j-1).cellNum == -1)
				(*topo)[triIndex1].adjTri1 = -1;
			else
			{
				(*topo)[triIndex1].adjTri1 = INDEXH(gridCellInfo,i*fNumCols_minus1+j-1).cellNum * 2 + 1;
			}
			(*topo)[triIndex1].adjTri2 = triIndex2;
			if (i==0 || INDEXH(gridCellInfo,(i-1)*fNumCols_minus1+j).cellNum==-1)
				(*topo)[triIndex1].adjTri3 = -1;
			else
			{
				(*topo)[triIndex1].adjTri3 = INDEXH(gridCellInfo,(i-1)*fNumCols_minus1+j).cellNum * 2 + 1;
			}
			// bottom/right tri in rect
			(*topo)[triIndex2].vertex1 = INDEXH(gridCellInfo,i*fNumCols_minus1+j).bottomLeft;
			(*topo)[triIndex2].vertex2 = INDEXH(gridCellInfo,i*fNumCols_minus1+j).bottomRight;
			(*topo)[triIndex2].vertex3 = INDEXH(gridCellInfo,i*fNumCols_minus1+j).topRight;
			if (j==fNumCols-2 || INDEXH(gridCellInfo,i*fNumCols_minus1+j+1).cellNum == -1)
				(*topo)[triIndex2].adjTri1 = -1;
			else
			{
				(*topo)[triIndex2].adjTri1 = INDEXH(gridCellInfo,i*fNumCols_minus1+j+1).cellNum * 2;
			}
			(*topo)[triIndex2].adjTri2 = triIndex1;
			if (i==fNumRows-2 || INDEXH(gridCellInfo,(i+1)*fNumCols_minus1+j).cellNum == -1)
				(*topo)[triIndex2].adjTri3 = -1;
			else
			{
				(*topo)[triIndex2].adjTri3 = INDEXH(gridCellInfo,(i+1)*fNumCols_minus1+j).cellNum * 2;
			}
		}
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Dag Tree");
	MySpinCursor(); // JLM 8/4/99
	tree = MakeDagTree(topo, (LongPoint**)pts, errmsg); 
	MySpinCursor(); // JLM 8/4/99
	if (errmsg[0])	
	{err = -1; goto done;} 
	// sethandle size of the fTreeH to be tree.fNumBranches, the rest are zeros
	_SetHandleSize((Handle)tree.treeHdl,tree.numBranches*sizeof(DAG));
	/////////////////////////////////////////////////
	
	/////////////////////////////////////////////////
	if (this -> moverMap != model -> uMap) goto setFields;	// don't try to create a map
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints
	
	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv * sizeof(**waterBoundaryPtsH));
	flagH = (LONGH)_NewHandleClear(nv * sizeof(**flagH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Numbering Islands");
	MySpinCursor(); // JLM 8/4/99
	//err = NumberIslands(&maskH2, velocityH, landWaterInfo, fNumRows_minus1, fNumCols_minus1, &numIslands);	// numbers start at 3 (outer boundary)
	err = NumberIslands(&maskH2, landmaskH, landWaterInfo, fNumRows_minus1, fNumCols_minus1, &numIslands);	// numbers start at 3 (outer boundary)
	MySpinCursor(); // JLM 8/4/99
	if (err) goto done;
	//numIslands++;	// this is a special case for CBOFS, right now the only coops_mask example
	for (i=0;i<ntri;i++)
	{
		if ((i+1)%2==0) isOdd = 0; else isOdd = 1;
		// the middle neighbor triangle is always the other half of the rectangle so can't be land or outside the map
		// odd - left/top, even - bottom/right the 1-2 segment is top/bot, the 2-3 segment is right/left
		if ((*topo)[i].adjTri1 == -1)
		{
			// add segment pt 2 - pt 3 to list, need points, triNum and whether it's L/W boundary (boundary num)
			(*segList)[segNum].pt1 = (*topo)[i].vertex2;
			(*segList)[segNum].pt2 = (*topo)[i].vertex3;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check left rectangle for L/W border 
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/fNumCols;
				jIndex = rectIndex%fNumCols;
				if (jIndex>0 && INDEXH(maskH2,iIndex*fNumCols + jIndex-1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols + jIndex-1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			else 
			{	
				// check right rectangle for L/W border convert back to row/col
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/fNumCols;
				jIndex = rectIndex%fNumCols;
				//if (jIndex<fNumCols && INDEXH(maskH2,iIndex*fNumCols + jIndex+1)>=3)
				if (jIndex<fNumCols_minus1 && INDEXH(maskH2,iIndex*fNumCols + jIndex+1)>=3)
				{
					(*segList)[segNum].isWater = 0;
					//(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols + jIndex+1);	
					(*segList)[segNum].islandNumber = INDEXH(maskH2,iIndex*fNumCols + jIndex+1);	
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;	
				}
			}
			segNum++;
		}
		
		if ((*topo)[i].adjTri3 == -1)
		{
			// add segment pt 1 - pt 2 to list
			// odd top, even bottom
			(*segList)[segNum].pt1 = (*topo)[i].vertex1;
			(*segList)[segNum].pt2 = (*topo)[i].vertex2;
			// check which land block this segment borders and mark the island
			if (isOdd) 
			{
				// check top rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex3);	// to get back into original grid for L/W info - use maskH2
				iIndex = rectIndex/fNumCols;
				jIndex = rectIndex%fNumCols;
				if (iIndex>0 && INDEXH(maskH2,(iIndex-1)*fNumCols + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex-1)*fNumCols + jIndex);
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;
				}
			}
			else 
			{
				// check bottom rectangle for L/W border
				rectIndex = INDEXH(verdatPtsH,(*topo)[i].vertex1);
				iIndex = rectIndex/fNumCols;
				jIndex = rectIndex%fNumCols;
				//if (iIndex<fNumRows && INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex)>=3)
				if (iIndex<fNumRows_minus1 && INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex)>=3)
				{
					(*segList)[segNum].isWater = 0;
					//(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex);		// this should be the neighbor's value
					(*segList)[segNum].islandNumber = INDEXH(maskH2,(iIndex+1)*fNumCols + jIndex);		// this should be the neighbor's value
				}
				else
				{
					(*segList)[segNum].isWater = 1;
					(*segList)[segNum].islandNumber = 1;		
				}
			}
			segNum++;
		}
	}
	nSegs = segNum;
	_SetHandleSize((Handle)segList,nSegs*sizeof(**segList));
	_SetHandleSize((Handle)segUsed,nSegs*sizeof(**segUsed));
	// go through list of segments, and make list of boundary segments
	// as segment is taken mark so only use each once
	// get a starting point, add the first and second to the list
	islandNum = 3;
findnewstartpoint:
	if (islandNum > numIslands) 
	{
		_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
		_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
		_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
		goto setFields;	// off by 2 - 0,1,2 are water cells, 3 and up are land
	}
	foundPt = false;
	for (i=0;i<nSegs;i++)
	{
		if ((*segUsed)[i]) continue;
		waterStartPoint = nBoundaryPts;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt1;
		(*flagH)[(*segList)[i].pt1] = 1;
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
		(*flagH)[(*segList)[i].pt2] = 1;
		currentIndex = (*segList)[i].pt2;
		startIndex = (*segList)[i].pt1;
		currentIsland = (*segList)[i].islandNumber;	
		foundPt = true;
		(*segUsed)[i] = true;
		break;
	}
	if (!foundPt)
	{
		printNote("Lost trying to set boundaries");
		// clean up handles and set grid without a map
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
		goto setFields;
	}
	
findnextpoint:
	for (i=0;i<nSegs;i++)
	{
		// look for second point of the previous selected segment, add the second to point list
		if ((*segUsed)[i]) continue;
		if ((*segList)[i].islandNumber > 3 && (*segList)[i].islandNumber != currentIsland) continue;
		if ((*segList)[i].islandNumber > 3 && currentIsland <= 3) continue;
		index = (*segList)[i].pt1;
		if (index == currentIndex)	// found next point
		{
			currentIndex = (*segList)[i].pt2;
			(*segUsed)[i] = true;
			if (currentIndex == startIndex) // completed a segment
			{
				islandNum++;
				(*boundaryEndPtsH)[nEndPts++] = nBoundaryPts-1;
				(*waterBoundaryPtsH)[waterStartPoint] = (*segList)[i].isWater+1;	// need to deal with this
				goto findnewstartpoint;
			}
			else
			{
				(*boundaryPtsH)[nBoundaryPts] = (*segList)[i].pt2;
				(*flagH)[(*segList)[i].pt2] = 1;
				(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
				nBoundaryPts++;
				goto findnextpoint;
			}
		}
	}
	// shouldn't get here unless there's a problem...
	_SetHandleSize((Handle)boundaryPtsH,nBoundaryPts*sizeof(**boundaryPtsH));
	_SetHandleSize((Handle)waterBoundaryPtsH,nBoundaryPts*sizeof(**waterBoundaryPtsH));
	_SetHandleSize((Handle)boundaryEndPtsH,nEndPts*sizeof(**boundaryEndPtsH));
	
setFields:	
	
	fVerdatToNetCDFH = verdatPtsH;
	
	//fVerdatToNetCDFH = verdatPtsH;
	
	/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverCurv::ReorderPointsCOOPSMask()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel3D*)triGrid;
	
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
	
	if (waterBoundaryPtsH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundaryEndPtsH);	
		map->SetWaterBoundaries(waterBoundaryPtsH);
		map->SetBoundaryPoints(boundaryPtsH);
		
		*newMap = map;
	}
	else
	{
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH=0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH=0;}
	}
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	
	/////////////////////////////////////////////////
done:
	if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
	if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
	if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
	if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
	if (segList) {DisposeHandle((Handle)segList); segList = 0;}
	if (flagH) {DisposeHandle((Handle)flagH); flagH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverCurv::ReorderPointsCOOPSMask");
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
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}

		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH = 0;}
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH = 0;}
	}

	


	return err;	
}

OSErr NetCDFMoverCurv_c::GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp)
{
	float dLat, dLon, dLat1, dLon1, dLat2, dLon2, fLat, fLong;
	
	if (iIndex<0 || jIndex>fNumCols) return -1;
	if (iIndex==0)	// along the outer top or right edge need to add on dlat/dlon
	{					// velocities at a gridpoint correspond to lower left hand corner of a grid box, draw in grid center
		if (jIndex<fNumCols)
		{
			dLat = INDEXH(fVertexPtsH,fNumCols+jIndex).pLat - INDEXH(fVertexPtsH,jIndex).pLat;
			fLat = INDEXH(fVertexPtsH,jIndex).pLat - dLat;
			dLon = INDEXH(fVertexPtsH,fNumCols+jIndex).pLong - INDEXH(fVertexPtsH,jIndex).pLong;
			fLong = INDEXH(fVertexPtsH,jIndex).pLong - dLon;
		}
		else
		{
			dLat1 = (INDEXH(fVertexPtsH,jIndex-1).pLat - INDEXH(fVertexPtsH,jIndex-2).pLat);
			dLat2 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLat;
			fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1)-(INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
			dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
			dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
			fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1)-(INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
		}
	}
	else 
	{
		if (jIndex<fNumCols)
		{
			fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLat;
			fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex).pLong;
		}
		else
		{
			dLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLat;
			fLat = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLat + dLat;
			dLon = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-2).pLong;
			fLong = INDEXH(fVertexPtsH,(iIndex-1)*fNumCols+jIndex-1).pLong + dLon;
		}
	}
	(*wp).pLat = (long)(fLat*1e6);
	(*wp).pLong = (long)(fLong*1e6);
	
	return noErr;
}

long NetCDFMoverCurv_c::GetNumDepthLevels()
{
	// should have only one version of this for all grid types, but will have to redo the regular grid stuff with depth levels
	// and check both sigma grid and multilayer grid (and maybe others)
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
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "sc_r", &sigmavarid);
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {numDepthLevels = 1;}	// require variable to match the dimension
			else numDepthLevels = sigmaLength;
		}
		else
		{
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {numDepthLevels = 1;}	// error in file
			//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
			//fVar.maxNumDepths = sigmaLength;
			else numDepthLevels = sigmaLength;
			//status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
			//if (status != NC_NOERR) {err = -1; goto done;}
			// once depth is read in 
		}
	}
	
	//done:
	return numDepthLevels;     
}


OSErr NetCDFMoverCurv_c::GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH)
{	// may want to input a time index to do time varying profiles for CDOG
	// what if file has u,v but not temp, sal?
	// should have only one version of this for all grid types, but will have to redo the regular grid stuff with depth levels
	// and check both sigma grid and multilayer grid (and maybe others)
	DepthValuesSetH depthLevelsH=0;
	float depth_val, lat_val, lon_val, *sigma_vals=0, *hydrodynamicField_vals=0, debugVal = 0;
	long i, j, index, numDepthLevels = 0, jIndex, iIndex;
	static size_t sigma_count;
	static size_t curr_index[] = {0,0,0,0}, depth_index[] = {0,0};
	static size_t curr_count[4];
	int curr_ucmp_id, curr_vcmp_id, temp_id, sal_id;
	char path[256], outPath[256]; 
	OSErr err = 0;
	int status, ncid, sigmaid, sigmavarid, depthid, numdims, latid, lonid;
	size_t sigmaLength=0,ptIndex=0,pt_count;
	LongPoint indices;
	//if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (bIsCOOPSWaterMask)
		indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
		else
		indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	else return nil;
	iIndex = indices.v;
	iIndex = fNumRows-iIndex-1;
	jIndex = indices.h;
	// really want to get the i, j values to index the file - need to be sure the lat is reordered						
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	//status = nc_open(fVar.pathName, NC_NOWRITE, &ncid);
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// code goes here, support the s-coordinate grid (generalized sigma) used in ROMS
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR) 
	{
		err = -1;
		goto done;
	}	
	status = nc_inq_varid(ncid, "sigma", &sigmavarid); 
	if (status != NC_NOERR) {err = -1; goto done;}	// require variable to match the dimension
	status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
	if (status != NC_NOERR) {err = -1; goto done;}	// error in file
	//fVar.gridType = SIGMA;	// in theory we should track this on initial read...
	//fVar.maxNumDepths = sigmaLength;
	sigma_vals = new float[sigmaLength];
	if (!sigma_vals) {err = memFullErr; goto done;}
	depthLevelsH = (DepthValuesSetH)_NewHandleClear(sigmaLength*sizeof(**depthLevelsH));
	if (!depthLevelsH) {err = memFullErr; goto done;}
	sigma_count = sigmaLength;
	status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//curr_index[0] = index;	// time 
	curr_index[0] = timeIndex;	// time - for now just use first (0), eventually want a whole set
	curr_count[0] = 1;	// take one at a time
	if (numdims>=4)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		//curr_count[2] = latlength;
		//curr_count[3] = lonlength;
		curr_count[1] = sigmaLength;	// depth
		curr_count[2] = 1;
		curr_count[3] = 1;
		curr_index[2] = iIndex;	// point closest to spill lat - I think this needs to be reversed
		curr_index[3] = jIndex;	// point closest to spill lon
		depth_index[0] = iIndex;	// point closest to spill lat - I think this needs to be reversed
		depth_index[1] = jIndex;	// point closest to spill lon
	}
	else
	{
		err = -1;
		goto done;
	}
	
	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//  get depth at specific point
	status = nc_get_var1_float(ncid, depthid, depth_index, &depth_val);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_varid(ncid, "lat", &latid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_varid(ncid, "lon", &lonid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//  for testing get lat at specific point
	status = nc_get_var1_float(ncid, latid, depth_index, &lat_val);
	if (status != NC_NOERR) {err = -1; goto done;}
	//  for testing get lon at specific point
	status = nc_get_var1_float(ncid, lonid, depth_index, &lon_val);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].depth = depth_val*sigma_vals[i];
	}
	
	
	// reuse the same array for u,v,temp,sal - check for w or set it to zero
	hydrodynamicField_vals = new float[sigmaLength];
	if (!hydrodynamicField_vals) {err = memFullErr; goto done;}
	
	status = nc_inq_varid(ncid, "U", &curr_ucmp_id);
	if (status != NC_NOERR)
	{
		status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
		if (status != NC_NOERR)
		{
			status = nc_inq_varid(ncid, "water_u", &curr_ucmp_id);
			if (status != NC_NOERR)
			{err = -1; goto done;}
		}
		//{err = -1; goto done;}
	}
	status = nc_inq_varid(ncid, "V", &curr_vcmp_id);
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "water_v", &curr_vcmp_id);
			if (status != NC_NOERR)
			{err = -1; goto done;}
		}
		//{err = -1; goto done;}
	}
	status = nc_inq_varid(ncid, "temp", &temp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "salt", &sal_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	else {if (sal_id==0) sal_id = temp_id+1;}	// for some reason returns zero rather than ten for GOM
	status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].value.u = hydrodynamicField_vals[i];
	}
	status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].value.v = hydrodynamicField_vals[i];
		(*depthLevelsH)[i].w = 0.;
	}
	status = nc_get_vara_float(ncid, temp_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].temp = hydrodynamicField_vals[i];
	}
	status = nc_get_vara_float(ncid, sal_id, curr_index, curr_count, hydrodynamicField_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<sigmaLength;i++)
	{
		(*depthLevelsH)[i].sal = hydrodynamicField_vals[i];
	}
	// should check units and scale factors here (GOM uses m/s and 1.0)
	// don't think we need to check fill or missing values since model should have data everywhere there isn't a mask
	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);
	//if (status != NC_NOERR) {status = nc_get_att_float(ncid, curr_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){err = -1; goto done;}*/}	// don't require
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
done:
	if  (err) 
	{
		printError("Problem exporting hydrodynamic profile at spill site");
		if (depthLevelsH)  {DisposeHandle((Handle)depthLevelsH); depthLevelsH=0;}
	}
	if (sigma_vals) delete [] sigma_vals;
	if (hydrodynamicField_vals) delete [] hydrodynamicField_vals;
	
	*profilesH =  depthLevelsH;    
	return err;
}

