/*
 *  NetCDFMoverTri_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/2/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

#include "NetCDFMoverTri_c.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include "my_build_list.h"
#include "netcdf.h"

NetCDFMoverTri_c::NetCDFMoverTri_c (TMap *owner, char *name) : NetCDFMoverCurv_c(owner, name)
{
	fNumNodes = 0;
	fNumEles = 0;
	bVelocitiesOnTriangles = false;
}


LongPointHdl NetCDFMoverTri_c::GetPointsHdl()
{
	return (dynamic_cast<TTriGridVel*>(fGrid)) -> GetPointsHdl();
}

Boolean NetCDFMoverTri_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double topDepth, bottomDepth, depthAlpha, timeAlpha;
	VelocityRec pt1interp = {0.,0.}, pt2interp = {0.,0.}, pt3interp = {0.,0.};
	long index, amtOfDepthData = 0, triIndex;
	
	long ptIndex1=-1,ptIndex2=-1,ptIndex3=-1; 
	long pt1depthIndex1, pt1depthIndex2, pt2depthIndex1, pt2depthIndex2, pt3depthIndex1, pt3depthIndex2;
	InterpolationVal interpolationVal;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!fVar.bShowArrows && !fVar.bShowGrid) return 0;

	err = dynamic_cast<NetCDFMoverTri *>(this) -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
	if(err) return false;
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	if (!bVelocitiesOnTriangles)
	{
		interpolationVal = fGrid -> GetInterpolationValues(wp.p);
		if (interpolationVal.ptIndex1<0) return false;
	}
	else
	{
		LongPoint lp;
		TDagTree *dagTree = 0;
		dagTree = ((dynamic_cast<TTriGridVel3D*>(fGrid))) -> GetDagTree();
		if(!dagTree) return false;
		lp.h = wp.p.pLong;
		lp.v = wp.p.pLat;
		triIndex = dagTree -> WhatTriAmIIn(lp);
		interpolationVal.ptIndex1 = -1;
	}
	
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  interpolationVal.ptIndex1;	
		ptIndex2 =  interpolationVal.ptIndex2;
		ptIndex3 =  interpolationVal.ptIndex3;
		if (fVerdatToNetCDFH)
		{
			ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
		}

		// probably want to extend to show the velocity at level that is being shown
		if (fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 	if (amtOfDepthData>0)
	 	{
			GetDepthIndices(ptIndex1,fVar.arrowDepth,&pt1depthIndex1,&pt1depthIndex2);	
			GetDepthIndices(ptIndex2,fVar.arrowDepth,&pt2depthIndex1,&pt2depthIndex2);	
			GetDepthIndices(ptIndex3,fVar.arrowDepth,&pt3depthIndex1,&pt3depthIndex2);	
		}
		else
		{	// old version that didn't use fDepthDataInfo, must be 2D
	 		pt1depthIndex1 = ptIndex1;	pt1depthIndex2 = -1;
	 		pt2depthIndex1 = ptIndex2;	pt2depthIndex2 = -1;
	 		pt3depthIndex1 = ptIndex3;	pt3depthIndex2 = -1;
		} 
	}
	else
	{
		if (!bVelocitiesOnTriangles)
			return false;
	}
	
	// Check for constant current 
	if((dynamic_cast<NetCDFMoverTri *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMoverTri *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	{
		if (bVelocitiesOnTriangles)
		{
			pt1depthIndex1 = -1;
			pt2depthIndex1 = -1;
			pt3depthIndex1 = -1;
			if (triIndex > 0)
			{
				pt1interp.u = INDEXH(fStartData.dataHdl,triIndex).u; 
				pt1interp.v = INDEXH(fStartData.dataHdl,triIndex).v; 
			}
		}
		// Calculate the interpolated velocity at the point
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v);
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v); 
			}
		}
		
		
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		if (bVelocitiesOnTriangles)
		{
			pt1depthIndex1 = -1;
			pt2depthIndex1 = -1;
			pt3depthIndex1 = -1;
			if (triIndex > 0)
			{
				pt1interp.u = INDEXH(fStartData.dataHdl,triIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).u; 
				pt1interp.v = INDEXH(fStartData.dataHdl,triIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).v; 
			}
		}
		
		// Calculate the interpolated velocity at the point
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v); 
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1)
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v); 
			}
		}
		
		
		
	}
	
	velocity.u = pt1interp.u + pt2interp.u + pt3interp.u; 
	velocity.v = pt1interp.v + pt2interp.v + pt3interp.v; 
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v) * fFileScaleFactor;
	lengthS = this->fVar.curScale * lengthU;
	if (lengthS > 1000000 || this->fVar.curScale==0) return true;	// if bad data in file causes a crash
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	if (interpolationVal.ptIndex1 >= 0 && ptIndex1>=0 && ptIndex2>=0 && ptIndex3>=0)
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices to triangle vertices : [%ld, %ld, %ld]",
				this->className, uStr, sStr, ptIndex1, ptIndex2, ptIndex3);
	}
	else
	{
		sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
				this->className, uStr, sStr);
	}
	return true;
}

WorldPoint3D NetCDFMoverTri_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depth = (*theLE).z;
	long ptIndex1,ptIndex2,ptIndex3,triIndex; 
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	InterpolationVal interpolationVal;
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	char errmsg[256];
	
	if(!fIsOptimizedForStep) 
	{
		err = SetInterval(errmsg, model_time); // AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	
	// Get the interpolation coefficients, alpha1,ptIndex1,alpha2,ptIndex2,alpha3,ptIndex3
	if (!bVelocitiesOnTriangles)
		interpolationVal = fGrid -> GetInterpolationValues(refPoint);
	else
	{
		LongPoint lp;
		TDagTree *dagTree = 0;
		dagTree = (dynamic_cast<TTriGridVel3D*>(fGrid)) -> GetDagTree();
		if(!dagTree) return deltaPoint;
		lp.h = refPoint.pLong;
		lp.v = refPoint.pLat;
		triIndex = dagTree -> WhatTriAmIIn(lp);
		interpolationVal.ptIndex1 = -1;
	}
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  interpolationVal.ptIndex1;	
		ptIndex2 =  interpolationVal.ptIndex2;
		ptIndex3 =  interpolationVal.ptIndex3;
		if (fVerdatToNetCDFH)
		{
			ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
		}
	}
	else
	{
		if (!bVelocitiesOnTriangles)
			return deltaPoint;	// set to zero, avoid any accidental access violation
	}
	
	// code goes here, need interpolation in z if LE is below surface
	// what kind of weird things can triangles do below the surface ??
	if (/*depth>0 &&*/ interpolationVal.ptIndex1 >= 0) 
	{
		scaledPatVelocity = GetMove3D(interpolationVal,depth);
		goto scale;
	}						
	if (depth > 0) return deltaPoint;	// set subsurface spill with no subsurface velocity

	// Check for constant current 
	if((dynamic_cast<NetCDFMoverTri *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMoverTri *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
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
			if (bVelocitiesOnTriangles && triIndex > 0)
			{
				scaledPatVelocity.u = INDEXH(fStartData.dataHdl,triIndex).u;
				scaledPatVelocity.v = INDEXH(fStartData.dataHdl,triIndex).v;
			}
			else
			{
				scaledPatVelocity.u = 0.;
				scaledPatVelocity.v = 0.;
			}
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (dynamic_cast<NetCDFMoverTri *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
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
			if (bVelocitiesOnTriangles && triIndex > 0)
			{
				scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,triIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).u;
				scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,triIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,triIndex).v;
			}
			else
			{
				scaledPatVelocity.u = 0.;
				scaledPatVelocity.v = 0.;
			}
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

VelocityRec NetCDFMoverTri_c::GetMove3D(InterpolationVal interpolationVal,float depth)
{
	// figure out which depth values the LE falls between
	// will have to interpolate in lat/long for both levels first
	// and some sort of check on the returned indices, what to do if one is below bottom?
	// for sigma model might have different depth values at each point
	// for multilayer they should be the same, so only one interpolation would be needed
	// others don't have different velocities at different depths so no interpolation is needed
	// in theory the surface case should be a subset of this case, may eventually combine
	
	long pt1depthIndex1, pt1depthIndex2, pt2depthIndex1, pt2depthIndex2, pt3depthIndex1, pt3depthIndex2;
	long ptIndex1, ptIndex2, ptIndex3, amtOfDepthData = 0;
	double topDepth, bottomDepth, depthAlpha, timeAlpha;
	VelocityRec pt1interp = {0.,0.}, pt2interp = {0.,0.}, pt3interp = {0.,0.};
	VelocityRec scaledPatVelocity = {0.,0.};
	Seconds startTime, endTime, time = model->GetModelTime();
	
	if (interpolationVal.ptIndex1 >= 0)  // if negative corresponds to negative ntri
	{
		// this is only section that's different from ptcur
		ptIndex1 =  interpolationVal.ptIndex1;	
		ptIndex2 =  interpolationVal.ptIndex2;
		ptIndex3 =  interpolationVal.ptIndex3;
		if (fVerdatToNetCDFH)
		{
			ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
		}
	}
	else
		return scaledPatVelocity;
	
	if (fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
 	if (amtOfDepthData>0)
 	{
		GetDepthIndices(ptIndex1,depth,&pt1depthIndex1,&pt1depthIndex2);	
		GetDepthIndices(ptIndex2,depth,&pt2depthIndex1,&pt2depthIndex2);	
		GetDepthIndices(ptIndex3,depth,&pt3depthIndex1,&pt3depthIndex2);	
	}
 	else
 	{	// old version that didn't use fDepthDataInfo, must be 2D
 		pt1depthIndex1 = ptIndex1;	pt1depthIndex2 = -1;
 		pt2depthIndex1 = ptIndex2;	pt2depthIndex2 = -1;
 		pt3depthIndex1 = ptIndex3;	pt3depthIndex2 = -1;
 	}
	
 	// the contributions from each point will default to zero if the depth indicies
	// come back negative (ie the LE depth is out of bounds at the grid point)
	if(dynamic_cast<NetCDFMoverTri *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMoverTri *>(this)->GetNumFiles()>1) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	{
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(INDEXH(fStartData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(INDEXH(fStartData.dataHdl,pt2depthIndex1).v);
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(INDEXH(fStartData.dataHdl,pt3depthIndex1).v); 
			}
		}
	}
	
	else // time varying current 
	{
		// Calculate the time weight factor
		if (dynamic_cast<NetCDFMoverTri *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex];
		endTime = (*fTimeHdl)[fEndData.timeIndex];
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		if (pt1depthIndex1!=-1)
		{
			if (pt1depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt1depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt1depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt1interp.u = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).u));
				pt1interp.v = depthAlpha*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex2).v));
			}
			else
			{
				pt1interp.u = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).u); 
				pt1interp.v = interpolationVal.alpha1*(timeAlpha*INDEXH(fStartData.dataHdl,pt1depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt1depthIndex1).v); 
			}
		}
		
		if (pt2depthIndex1!=-1)
		{
			if (pt2depthIndex2!=-1) 
			{
				topDepth = INDEXH(fDepthsH,pt2depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt2depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt2interp.u = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).u));
				pt2interp.v = depthAlpha*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex2).v));
			}
			else
			{
				pt2interp.u = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).u); 
				pt2interp.v = interpolationVal.alpha2*(timeAlpha*INDEXH(fStartData.dataHdl,pt2depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt2depthIndex1).v); 
			}
		}
		
		if (pt3depthIndex1!=-1) 
		{
			if (pt3depthIndex2!=-1)
			{
				topDepth = INDEXH(fDepthsH,pt3depthIndex1);	
				bottomDepth = INDEXH(fDepthsH,pt3depthIndex2);
				depthAlpha = (bottomDepth - depth)/(double)(bottomDepth - topDepth);
				pt3interp.u = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).u));
				pt3interp.v = depthAlpha*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v))
				+ (1-depthAlpha)*(interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex2).v));
			}
			else
			{
				pt3interp.u = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).u); 
				pt3interp.v = interpolationVal.alpha3*(timeAlpha*INDEXH(fStartData.dataHdl,pt3depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,pt3depthIndex1).v); 
			}
		}
	}
	scaledPatVelocity.u = pt1interp.u + pt2interp.u + pt3interp.u;
	scaledPatVelocity.v = pt1interp.v + pt2interp.v + pt3interp.v;
	
	return scaledPatVelocity;
}


float NetCDFMoverTri_c::GetTotalDepth(WorldPoint refPoint, long triNum)
{
#pragma unused(refPoint)
	float totalDepth = 0;
	if (fDepthDataInfo) 
	{
		//indexToDepthData = (*fDepthDataInfo)[ptIndex].indexToDepthData;
		//numDepths = (*fDepthDataInfo)[ptIndex].numDepths;
		totalDepth = (*fDepthDataInfo)[triNum].totalDepth;
	}
	return totalDepth; // this should be an error
}
// probably eventually switch to NetCDFMover only

void NetCDFMoverTri_c::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData;
	long numDepths;
	float totalDepth;
	
	if (fDepthDataInfo) 
	{
		indexToDepthData = (*fDepthDataInfo)[ptIndex].indexToDepthData;
		numDepths = (*fDepthDataInfo)[ptIndex].numDepths;
		totalDepth = (*fDepthDataInfo)[ptIndex].totalDepth;
	}
	else
		return; // this should be an error
	
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
			break;
		case SIGMA: // should rework the sigma to match Gnome_beta's simpler method
			if (depthAtPoint <= totalDepth) // check data exists at chosen/LE depth for this point
			{
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
			break;
		default:
			*depthIndex1 = UNASSIGNEDINDEX;
			*depthIndex2 = UNASSIGNEDINDEX;
			break;
	}
}



OSErr NetCDFMoverTri_c::ReorderPoints2(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri, Boolean isCCW) 
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
	DAGTreeStruct tree;
	
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	WorldRect triBounds;
	LONGH waterBoundariesH=0;
	LONGH boundaryPtsH = 0;
	
	TTriGridVel3D *triGrid = nil;
	
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
		//short islandNum, index;
		long islandNum, index;
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
		//INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
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
	
	numVerdatPts = nv;	//for now, may reorder later
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
	
	//numVerdatPts = nv;	//for now, may reorder later
	for (i=0; i<=numVerdatPts; i++)
	{
		//long index;
		float fLong, fLat/*, fDepth*/;
		LongPoint vertex;
		
		if(i < numVerdatPts) 
		{	
			//index = i+1;
			//n = INDEXH(verdatPtsH,i);
			n = i;	// for now, not sure if need to reorder
			fLat = INDEXH(fVertexPtsH,n).pLat;	// don't need to store fVertexPtsH, just pass in and use here
			fLong = INDEXH(fVertexPtsH,n).pLong;
			vertex.v = (long)(fLat*1e6);
			vertex.h = (long)(fLong*1e6);
			
			//fDepth = 1.;	// this will be set from bathymetry, just a fudge here for outputting a verdat
			INDEXH(pts,i) = vertex;
		}
		else { // the last line should be all zeros
			//index = 0;
			//fLong = fLat = fDepth = 0.0;
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
	//fprintf(outfile,"%ld\n",numVerdatBreakPts);
	
	// now write out out the break points
	/*for(i = 0; i < numVerdatBreakPts; i++ )
	 {
	 fprintf(outfile,"%ld\n",INDEXH(verdatBreakPtsH,i));
	 }*/
	/////////////////////////////////////////////////
	
	//fclose(outfile);
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	//if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
	if(!(topo = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology))))goto done;	
	
	// point and triangle indices should start with zero
	for(i = 0; i < 3*ntri; i ++)
	{
		/*if (tri_neighbors[i]==0)
		 tri_neighbors[i]=-1;
		 else */
		tri_neighbors[i] = tri_neighbors[i] - 1;
		tri_verts[i] = tri_verts[i] - 1;
	}
	for(i = 0; i < ntri; i ++)
	{	// topology data needs to be CCW
		//long debugTest = tri_verts[i];
		(*topo)[i].vertex1 = tri_verts[i];
		//debugTest = tri_verts[i+ntri];
		if (isCCW)
			(*topo)[i].vertex2 = tri_verts[i+ntri];
		else
			(*topo)[i].vertex3 = tri_verts[i+ntri];
		//debugTest = tri_verts[i+2*ntri];
		if (isCCW)
			(*topo)[i].vertex3 = tri_verts[i+2*ntri];
		else
			(*topo)[i].vertex2 = tri_verts[i+2*ntri];
		//debugTest = tri_neighbors[i];
		(*topo)[i].adjTri1 = tri_neighbors[i];
		//debugTest = tri_neighbors[i+ntri];
		if (isCCW)
			(*topo)[i].adjTri2 = tri_neighbors[i+ntri];
		else
			(*topo)[i].adjTri3 = tri_neighbors[i+ntri];
		//debugTest = tri_neighbors[i+2*ntri];
		if (isCCW)
			(*topo)[i].adjTri3 = tri_neighbors[i+2*ntri];
		else
			(*topo)[i].adjTri2 = tri_neighbors[i+2*ntri];
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
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}
	
	//fGrid = (TTriGridVel*)triGrid;
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
	//if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);	// should be set in TextRead
	
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
	boundaryPtsH = (LONGH)_NewHandleClear(numBoundaryPts * sizeof(**boundaryPtsH));
	if (!boundaryPtsH) {err = memFullErr; goto done;}
	
	for (i=0;i<numBoundaryPts;i++)
	{
		INDEXH(waterBoundariesH,i)=1;	// default is land
		if (bndry_type[i]==1)	
			INDEXH(waterBoundariesH,i)=2;	// water boundary, this marks start point rather than end point...
		INDEXH(boundaryPtsH,i) = bndry_indices[i]-1;
	}
	
	if (waterBoundariesH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	{
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
		if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(verdatBreakPtsH);	
		map->SetWaterBoundaries(waterBoundariesH);
		map->SetBoundaryPoints(boundaryPtsH);
		
		*newMap = map;
	}
	else
	{
		if (waterBoundariesH) {DisposeHandle((Handle)waterBoundariesH); waterBoundariesH=0;}
		if (verdatBreakPtsH) {DisposeHandle((Handle)verdatBreakPtsH); verdatBreakPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	
	/////////////////////////////////////////////////
	//fVerdatToNetCDFH = verdatPtsH;	// this should be resized
	
done:
	if (err) printError("Error reordering gridpoints into verdat format");
	if (vertFlagsH) {DisposeHandle((Handle)vertFlagsH); vertFlagsH = 0;}
	if (verdatPtsH) {DisposeHandle((Handle)verdatPtsH); verdatPtsH = 0;}
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReorderPoints");
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
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH = 0;}
	}
	return err;
}

OSErr NetCDFMoverTri_c::ReorderPoints(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts) 
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
	
	TTriGridVel3D *triGrid = nil;
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
		//short islandNum, index;
		long islandNum, index;
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
	//outfile=fopen(path,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	
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
	//fprintf(outfile,"%ld\n",numVerdatBreakPts);
	
	// now write out out the break points
	/*for(i = 0; i < numVerdatBreakPts; i++ )
	 {
	 fprintf(outfile,"%ld\n",INDEXH(verdatBreakPtsH,i));
	 }*/
	/////////////////////////////////////////////////
	
	//fclose(outfile);
	// shrink handle
	_SetHandleSize((Handle)verdatBreakPtsH,numVerdatBreakPts*sizeof(long));
	for(i = 0; i < numVerdatBreakPts; i++ )
	{
		INDEXH(verdatBreakPtsH,i)--;
	}
	
	DisplayMessage("NEXTMESSAGETEMP");
	DisplayMessage("Making Triangles");
	// use new maketriangles to force algorithm to avoid 3 points in the same row or column
	MySpinCursor(); // JLM 8/4/99
	if (err = maketriangles(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts))
		//if (err = maketriangles2(&topo,pts,numVerdatPts,verdatBreakPtsH,numVerdatBreakPts,verdatPtsH,fNumCols_ext))
		goto done;
	
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
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReorderPoints()","new TTriGridVel" ,err);
		goto done;
	}
	
	//fGrid = (TTriGridVel*)triGrid;
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
	if (topo) fNumEles = _GetHandleSize((Handle)topo)/sizeof(**topo);
	
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
		PtCurMap *map = CreateAndInitPtCurMap(fVar.pathName,triBounds); // the map bounds are the same as the grid bounds
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
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReorderPoints");
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

long NetCDFMoverTri_c::GetNumDepthLevels()
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
	//if (status != NC_NOERR) {/*err = -1; goto done;*/return -1;}
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
		if (status != NC_NOERR) {err = -1; return -1;}
	}
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
	
	//done:
	return numDepthLevels;     
}


