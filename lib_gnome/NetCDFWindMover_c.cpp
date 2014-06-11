/*
 *  NetCDFWindMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "NetCDFWindMover_c.h"
#include "CROSS.H"
#include "netcdf.h"

NetCDFWindMover_c::NetCDFWindMover_c(TMap *owner,char* name) : WindMover_c(owner, name)
{
	if(!name || !name[0]) this->SetClassName("NetCDF Wind");
	else 	SetClassName (name); // short file name
	
	// use wind defaults for uncertainty
	bShowGrid = false;
	bShowArrows = false;
	
	fGrid = 0;
	fTimeHdl = 0;
	fIsOptimizedForStep = false;
	
	fUserUnits = kMetersPerSec;	
	fWindScale = 1.;
	fArrowScale = 10.;
	fFillValue = -1e+34;
	
	fTimeShift = 0; // assume file is in local time
	
	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fAllowExtrapolationOfWinds = false;
	
	fOverLap = false;
	fOverLapStartTime = 0;
	fInputFilesHdl = 0; 
}


long NetCDFWindMover_c::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFWindMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return -1; }
	
	return rowNum * fNumCols + colNum;
}

LongPoint NetCDFWindMover_c::GetVelocityIndices(WorldPoint p) 
{
	long rowNum, colNum;
	LongPoint indices = {-1,-1};
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of NetCDFMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	colNum = p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset;
	rowNum = p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset;
	
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return indices; }
	
	//return rowNum * fNumCols + colNum;
	indices.h = colNum;
	indices.v = rowNum;
	return indices;
}


/////////////////////////////////////////////////
// routines for ShowCoordinates() to recognize netcdf currents
double NetCDFWindMover_c::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (index>=0)
	{
		if (fStartData.dataHdl) u = INDEXH(fStartData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFWindMover_c::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		if (fEndData.dataHdl) u = INDEXH(fEndData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFWindMover_c::GetStartVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fStartData.dataHdl) v = INDEXH(fStartData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

double NetCDFWindMover_c::GetEndVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fEndData.dataHdl) v = INDEXH(fEndData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

OSErr NetCDFWindMover_c::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX)
		*startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

OSErr NetCDFWindMover_c::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX)
		*endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

Boolean NetCDFWindMover_c::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{
	char uStr[32],sStr[32],errmsg[256];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;
	
	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	LongPoint indices;
	
	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!bShowArrows && !bShowGrid) return 0;
	err = dynamic_cast<NetCDFWindMover *>(this) -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
	if(err) return false;
	
	if(dynamic_cast<NetCDFWindMover *>(this)->GetNumTimesInFile()>1)
	{
		if (err = this->GetStartTime(&startTime)) return false;	// should this stop us from showing any velocity?
		if (err = this->GetEndTime(&endTime)) /*return false;*/
		{
			if ((time > startTime || time < startTime) && fAllowExtrapolationOfWinds)
			{
				timeAlpha = 1;
			}
			else
				return false;
		}
		else
			timeAlpha = (endTime - time)/(double)(endTime - startTime);	
	}

	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular
		
		indices = this->GetVelocityIndices(wp.p);
		
		if (index >= 0)
		{
			// Check for constant current 
			if(dynamic_cast<NetCDFWindMover *>(this)->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				velocity.u = this->GetStartUVelocity(index);
				velocity.v = this->GetStartVVelocity(index);
			}
			else // time varying current
			{
				velocity.u = timeAlpha*this->GetStartUVelocity(index) + (1-timeAlpha)*this->GetEndUVelocity(index);
				velocity.v = timeAlpha*this->GetStartVVelocity(index) + (1-timeAlpha)*this->GetEndVVelocity(index);
			}
		}
	}
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	lengthS = this->fWindScale * lengthU;
	if (lengthS > 1000000 || this->fWindScale==0) return true;	// if bad data in file causes a crash
	
	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
			this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);
	
	return true;
}
OSErr NetCDFWindMover_c::PrepareForModelRun()
{
	return WindMover_c::PrepareForModelRun();
}

OSErr NetCDFWindMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	OSErr err = 0;

	if (bIsFirstStep)
		fModelStartTime = model_time;
	if(uncertain) 
	{
		//Seconds elapsed_time = model_time - fModelStartTime;
		Seconds elapsed_time = model_time + time_step - fModelStartTime;	// so uncertainty starts at time zero + uncertain_time_delay, rather than a time step later
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}
	
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (!bActive) return noErr;
	
	err = dynamic_cast<NetCDFWindMover *>(this) -> SetInterval(errmsg, model_time);
	
	if (err) goto done;	// again don't want to have error if outside time interval
	
	fIsOptimizedForStep = true;	// is this needed?
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFWindMover::PrepareForModelStep");
		printError(errmsg); 
	}	
	
	return err;
}

void NetCDFWindMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}


WorldPoint3D NetCDFWindMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double 	dLong, dLat;
	WorldPoint3D	deltaPoint ={0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double timeAlpha;
	long index; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec windVelocity;
	OSErr err = noErr;
	char errmsg[256];
	
	if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	
	if(!fIsOptimizedForStep) 
	{
		err = dynamic_cast<NetCDFWindMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
		
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
	
	// Check for constant wind 
	if( ( dynamic_cast<NetCDFWindMover *>(this)->GetNumTimesInFile()==1 && !( dynamic_cast<NetCDFWindMover *>(this)->GetNumFiles() > 1 ) ) ||
	   (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			windVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			windVelocity.v = INDEXH(fStartData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			windVelocity.u = 0.;
			windVelocity.v = 0.;
		}
	}
	else // time varying wind 
	{
		// Calculate the time weight factor
		if (dynamic_cast<NetCDFWindMover *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			windVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			windVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			windVelocity.u = 0.;
			windVelocity.v = 0.;
		}
	}
	
	windVelocity.u *= fWindScale; 
	windVelocity.v *= fWindScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		err = AddUncertainty(setIndex,leIndex,&windVelocity);
	}
	
	windVelocity.u *=  (*theLE).windage;
	windVelocity.v *=  (*theLE).windage;
	
	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat =   (windVelocity.v / METERSPERDEGREELAT) * timeStep;
	
	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;
	
	return deltaPoint;
}

Seconds NetCDFWindMover_c::GetTimeValue(long index)
{
	if (index<0) printError("Access violation in NetCDFWindMover::GetTimeValue()");
	Seconds time = (*fTimeHdl)[index] + fTimeShift;
	return time;
}

OSErr NetCDFWindMover_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{	
	// regular grid wind format
	OSErr err = 0;
	long i,j;
	char path[256], outPath[256]; 
	int status, ncid, numdims, numvars, uv_ndims;
	int wind_ucmp_id, wind_vcmp_id, sigma_id;
	static size_t wind_index[] = {0,0,0,0};
	static size_t wind_count[4];
	double *wind_uvals=0,*wind_vvals=0, fill_value = -1e+10;
	long totalNumberOfVels = fNumRows * fNumCols;
	VelocityFH velH = 0;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	double scale_factor = 1.;
	Boolean bHeightIncluded = false;
	
	errmsg[0]=0;
	
	strcpy(path,fPathName);
	if (!path || !path[0]) return -1;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR)
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	wind_index[0] = index;	// time 
	wind_count[0] = 1;	// take one at a time
	
	if (numdims>=4)
	{	// won't be using the heights, just need to know how to read the file
		status = nc_inq_dimid(ncid, "sigma", &sigma_id);	//3D
		if (status != NC_NOERR) 
		{
			/*status = nc_inq_dimid(ncid, "height", &sigma_id);	//3D - need to check sigma values in TextRead...
			 if (status != NC_NOERR) bHeightIncluded = false;
			 else bHeightIncluded = true;*/
			bHeightIncluded = false;
		}
		else bHeightIncluded = true;
	}
	
	if (bHeightIncluded)
	{
		wind_count[1] = 1;	// depth - height here, is this necessary?
		wind_count[2] = latlength;
		wind_count[3] = lonlength;
	}
	else
	{
		wind_count[1] = latlength;	
		wind_count[2] = lonlength;
	}
	
	wind_uvals = new double[latlength*lonlength]; 
	if(!wind_uvals) {TechError("NetCDFWindMover::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
	wind_vvals = new double[latlength*lonlength]; 
	if(!wind_vvals) {TechError("NetCDFWindMover::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
	
	// code goes here, change key word to wind_u,v
	status = nc_inq_varid(ncid, "air_u", &wind_ucmp_id);	
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "UX", &wind_ucmp_id);	// for Lucas's Pac SSH LAS server data
		if (status != NC_NOERR) {err = -1; /*goto done;*/ goto LAS;}	// broader check for variable names coming out of LAS
	}
	status = nc_inq_varid(ncid, "air_v", &wind_vcmp_id);	// what if only input one at a time (u,v separate movers)?
	if (status != NC_NOERR)
	{
		status = nc_inq_varid(ncid, "VY", &wind_vcmp_id);	// for Lucas's Pac SSH LAS server data
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	
LAS:
	if (err)
	{
		Boolean bLASStyleNames = false;
		char uname[NC_MAX_NAME],vname[NC_MAX_NAME],varname[NC_MAX_NAME];
		err = 0;
		status = nc_inq_nvars(ncid, &numvars);
		if (status != NC_NOERR) {err = -1; goto done;}
		for (i=0;i<numvars;i++)
		{
			//if (i == recid) continue;
			status = nc_inq_varname(ncid,i,varname);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (varname[0]=='U' || varname[0]=='u' /*|| strstrnocase(varname,"EVEL")*/)	// careful here, could end up with wrong u variable (like u_curr for example)
			{
				wind_ucmp_id = i; bLASStyleNames = true;
				strcpy(uname,varname);
			}
			if (varname[0]=='V' || varname[0]=='v' /*|| strstrnocase(varname,"NVEL")*/)
			{
				wind_vcmp_id = i; bLASStyleNames = true;
				strcpy(vname,varname);
			}
		}
		if (!bLASStyleNames){err = -1; goto done;}
	}
	
	status = nc_inq_varndims(ncid, wind_ucmp_id, &uv_ndims);
	if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {wind_count[1] = latlength; wind_count[2] = lonlength;}}	// could have more dimensions than are used in u,v
	if (uv_ndims==4) {wind_count[1] = 1;wind_count[2] = latlength;wind_count[3] = lonlength;}
	
	
	status = nc_get_vara_double(ncid, wind_ucmp_id, wind_index, wind_count, wind_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_double(ncid, wind_vcmp_id, wind_index, wind_count, wind_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_att_double(ncid, wind_ucmp_id, "_FillValue", &fill_value);	// should get this in text_read and store, but will have to go short to float and back
	if (status != NC_NOERR) 
	{
		status = nc_get_att_double(ncid, wind_ucmp_id, "FillValue", &fill_value); /*if (status != NC_NOERR) {err = -1; goto done;}}*/	// require fill value
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, wind_ucmp_id, "missing_value", &fill_value);} /*if (status != NC_NOERR) {err = -1; goto done;}*/
	}	// require fill value
	//if (status != NC_NOERR) {err = -1; goto done;}	// don't require fill value
	status = nc_get_att_double(ncid, wind_ucmp_id, "scale_factor", &scale_factor);
	//if (status != NC_NOERR) {err = -1; goto done;}	// don't require scale factor
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) {err = memFullErr; goto done;}
	for (i=0;i<latlength;i++)
	{
		for (j=0;j<lonlength;j++)
		{
			if (wind_uvals[(latlength-i-1)*lonlength+j]==fill_value)	// should store in wind array and check before drawing or moving
				wind_uvals[(latlength-i-1)*lonlength+j]=0.;
			if (wind_vvals[(latlength-i-1)*lonlength+j]==fill_value)
				wind_vvals[(latlength-i-1)*lonlength+j]=0.;
			INDEXH(velH,i*lonlength+j).u = (float)wind_uvals[(latlength-i-1)*lonlength+j];
			INDEXH(velH,i*lonlength+j).v = (float)wind_vvals[(latlength-i-1)*lonlength+j];
		}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	fWindScale = scale_factor;
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading wind data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (wind_uvals) delete [] wind_uvals;
	if (wind_vvals) delete [] wind_vvals;
	return err;
}

