/*
 *  TimeGridWind_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/5/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "TimeGridVel_c.h"
#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#include "OUTILS.H"
#endif

#include "netcdf.h"

/*Boolean IsGridWindFile(char *path,short *selectedUnitsP)
{
	
	Boolean	bIsValid = false;
	OSErr	err = noErr;
	long line;
	char	strLine [512];
	char	firstPartOfFile [512];
	long lenToRead,fileLength;
	short selectedUnits = kUndefined, numScanned;
	char unitsStr[64], gridwindStr[64];
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) return false;
	
	lenToRead = _min(512,fileLength);
	
	err = ReadSectionOfFile(0,0,path,0,lenToRead,firstPartOfFile,0);
	firstPartOfFile[lenToRead-1] = 0; // make sure it is a cString
	if (!err)
	{
		NthLineInTextNonOptimized (firstPartOfFile, line = 0, strLine, 512);
		if (strstr(strLine,"[GRIDWIND"))
		{
			bIsValid = true;
			*selectedUnitsP = selectedUnits;
			numScanned = sscanf(strLine,"%s%s",gridwindStr,unitsStr);
			if(numScanned != 2) { selectedUnits = kUndefined; goto done; }
			RemoveLeadingAndTrailingWhiteSpace(unitsStr);
			selectedUnits = StrToSpeedUnits(unitsStr);// note we are not supporting cm/sec in gnome
		}
	}
	
done:
	if(bIsValid)
	{
		*selectedUnitsP = selectedUnits;
	}
	return bIsValid;
}*/

TimeGridWindRect_c::TimeGridWindRect_c() : TimeGridVel_c()
{
	
}

void TimeGridWindRect_c::Dispose()
{
	TimeGridVel_c::Dispose ();
}


VelocityRec TimeGridWindRect_c::GetScaledPatValue(const Seconds& model_time, WorldPoint3D refPoint)
{	// pull out the getpatval part
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index; 
	long depthIndex1,depthIndex2;	// default to -1?
	Seconds startTime,endTime;

	VelocityRec	windVelocity = {0.,0.};
	OSErr err = 0;
	
	index = GetVelocityIndex(refPoint.p);  // regular grid
	
	// Check for constant wind 
	if( ( GetNumTimesInFile()==1 && !( GetNumFiles() > 1 ) ) ||
	   (fEndData.timeIndex == UNASSIGNEDINDEX && model_time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
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
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - model_time)/(double)(endTime - startTime);
		
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
	
scale:
	
	//code goes here, deal with scale factor - file vs dialog (offer both?)
	
	windVelocity.u *= fVar.fileScaleFactor; 
	windVelocity.v *= fVar.fileScaleFactor; 
	
	
	return windVelocity;
}

OSErr TimeGridWindRect_c::TextRead(const char *path, const char *topFilePath)
{
	// this code is for regular grids
	OSErr err = 0;
	long i,j, numScanned;
	int status, ncid, latid, lonid, recid, timeid, numdims;
	int latvarid, lonvarid;
	size_t latLength, lonLength, recs, t_len, t_len2;
	double startLat,startLon,endLat,endLon,dLat,dLon,timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0, month[10];	
	WorldRect bounds;
	double *lat_vals=0,*lon_vals=0;
	TRectGridVel *rectGrid = nil;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0;
	static size_t pt_count[2];
	Seconds startTime, startTime2;
	double timeConversion = 1.;
	char fileName[256],s[256],*modelTypeStr=0;
	char  outPath[256];

	char errmsg[256];
	errmsg[0] = 0;
	
    if (!path || !path[0])
		return 0;
	//cerr << "TimeGridWindRect_c::TextRead(): path = " << path << endl;


	strcpy(fVar.pathName,path);
	
	strcpy(s,path);
#ifndef pyGNOME
	SplitPathFile (s, fileName);
#else
	SplitPathFileName (s, fileName);
#endif
	//strcpy(fFileName, fileName);	// maybe use a name from the file
	strcpy(fVar.userName, fileName); // maybe use a name from the file
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	/*if (status != NC_NOERR)
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}*/
	
	status = nc_inq_dimid(ncid, "time", &recid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR || recid==-1) {err = -1; goto done;}
	}
	
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {status = nc_inq_varid(ncid, "TIME", &timeid);if (status != NC_NOERR) {err = -1; goto done;} /*timeid = recid;*/} 	// for Ferret files, everything is in CAPS
	
	/////////////////////////////////////////////////
	
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		err = -1; goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];
		
		timeUnits = new char[t_len+1];
		status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		StringSubstitute(timeUnits, 'T', ' ');
		StringSubstitute(timeUnits, 'Z', ' ');
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
						  unitStr, junk, &time.year, &time.month, &time.day,
						  &time.hour, &time.minute, &time.second) ;
		if (numScanned==5)	
		{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned==7) // has two extra time entries ??	
			time.second = 0;
		else if (numScanned<8)	
			//if (numScanned!=8)	
		{ err = -1; TechError("TimeGridWindRect_c::TextRead()", "sscanf() == 8", 0); goto done; }
		DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
		if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
			timeConversion = 3600.;
		else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
			timeConversion = 60.;
		else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
			timeConversion = 1.;
		else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
			timeConversion = 24*3600.;
	} 
	
	// check for Navy model name
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) {}	
	else 
	{
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {}	 
		else
		{
			modelTypeStr[t_len2] = '\0';
			//strcpy(fFileName, modelTypeStr); // maybe use a name from the file
		}
	}
	status = nc_inq_dimid(ncid, "lat", &latid); 
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "LAT", &latid);	if (status != NC_NOERR) {err = -1; goto LAS;}	// this is for SSH files which have LAS/ferret style caps
	}
	status = nc_inq_varid(ncid, "lat", &latvarid); 
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "LAT", &latvarid);	if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_dimlen(ncid, latid, &latLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimid(ncid, "lon", &lonid);	
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "LON", &lonid);	if (status != NC_NOERR) {err = -1; goto done;}	// this is for SSH files which have LAS/ferret style caps
	}
	status = nc_inq_varid(ncid, "lon", &lonvarid);	
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "LON", &lonvarid);	if (status != NC_NOERR) {err = -1; goto done;}
	}
	status = nc_inq_dimlen(ncid, lonid, &lonLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
LAS:
	// check number of dimensions - 2D or 3D
	// allow more flexibility with dimension names
	if (err)
	{
		Boolean bLASStyleNames = false;
		char latname[NC_MAX_NAME],lonname[NC_MAX_NAME],dimname[NC_MAX_NAME];
		err = 0;
		status = nc_inq_ndims(ncid, &numdims);
		if (status != NC_NOERR) {err = -1; goto done;}
		for (i=0;i<numdims;i++)
		{
			if (i == recid) continue;
			status = nc_inq_dimname(ncid,i,dimname);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (strstrnocase(dimname,"LON"))
			{
				lonid = i; bLASStyleNames = true;
				strcpy(lonname,dimname);
			}
			if (strstrnocase(dimname,"LAT"))
			{
				latid = i; bLASStyleNames = true;
				strcpy(latname,dimname);
			}
		}
		if (bLASStyleNames)
		{
			status = nc_inq_varid(ncid, latname, &latvarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, latid, &latLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, lonname, &lonvarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, lonid, &lonLength);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		else
		{err = -1; goto done;}
		
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	
	lat_vals = new double[latLength]; 
	lon_vals = new double[lonLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_double(ncid, latvarid, &ptIndex, &pt_count[0], lat_vals);
	if (status != NC_NOERR) {err=-1; goto done;}
	status = nc_get_vara_double(ncid, lonvarid, &ptIndex, &pt_count[1], lon_vals);
	if (status != NC_NOERR) {err=-1; goto done;}
	
	latIndex = 0;
	lonIndex = 0;
	status = nc_get_var1_double(ncid, latvarid, &latIndex, &startLat);
	if (status != NC_NOERR) {err=-1; goto done;}
	status = nc_get_var1_double(ncid, lonvarid, &lonIndex, &startLon);
	if (status != NC_NOERR) {err=-1; goto done;}
	latIndex = latLength-1;
	lonIndex = lonLength-1;
	status = nc_get_var1_double(ncid, latvarid, &latIndex, &endLat);
	if (status != NC_NOERR) {err=-1; goto done;}
	status = nc_get_var1_double(ncid, lonvarid, &lonIndex, &endLon);
	if (status != NC_NOERR) {err=-1; goto done;}
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {err = -1; goto done;}
		INDEXH(fTimeHdl,i) = startTime2+(long) (timeVal*timeConversion);	// which start time where?
		if (i==0) startTime = startTime2+(long) (timeVal*timeConversion);
	}
	dLat = (endLat - startLat) / (latLength - 1);
	dLon = (endLon - startLon) / (lonLength - 1);
	
	bounds.loLat = ((startLat-dLat/2.))*1e6;
	bounds.hiLat = ((endLat+dLat/2.))*1e6;
	if (startLon>180.)
	{
		bounds.loLong = (((startLon-dLon/2.)-360.))*1e6;
		bounds.hiLong = (((endLon+dLon/2.)-360.))*1e6;
	}
	else
	{
		bounds.loLong = ((startLon-dLon/2.))*1e6;
		bounds.hiLong = ((endLon+dLon/2.))*1e6;
	}
	rectGrid = new TRectGridVel;
	if (!rectGrid)
	{		
		err = true;
		TechError("Error in TimeGridWindRect_c::TextRead()","new TRectGridVel" ,err);
		goto done;
	}
	
	fNumRows = latLength;
	fNumCols = lonLength;
	fGrid = (TGridVel*)rectGrid;
	
	rectGrid -> SetBounds(bounds); 
	this->SetGridBounds(bounds);
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	
done:
	if (err)
	{
		printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	}
	
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (modelTypeStr) delete [] modelTypeStr;
	if (timeUnits) delete [] timeUnits;
	return err;
}

OSErr TimeGridWindRect_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
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
	
	//strcpy(path,fPathName);
	strcpy(path,fVar.pathName);
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
	if(!wind_uvals) {TechError("TimeGridWindRect::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	wind_vvals = new double[latlength*lonlength]; 
	if(!wind_vvals) {TechError("TimeGridWindRect::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	
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
			if (wind_uvals[(latlength-i-1)*lonlength+j]==fill_value)
				wind_uvals[(latlength-i-1)*lonlength+j]=0.;
			if (wind_vvals[(latlength-i-1)*lonlength+j]==fill_value)
				wind_vvals[(latlength-i-1)*lonlength+j]=0.;
			if (isnan(wind_uvals[(latlength-i-1)*lonlength+j])) 
				wind_uvals[(latlength-i-1)*lonlength+j]=0.;
			if (isnan(wind_vvals[(latlength-i-1)*lonlength+j])) 
				wind_vvals[(latlength-i-1)*lonlength+j]=0.;
			INDEXH(velH,i*lonlength+j).u = (float)wind_uvals[(latlength-i-1)*lonlength+j];
			INDEXH(velH,i*lonlength+j).v = (float)wind_vvals[(latlength-i-1)*lonlength+j];
		}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	//fWindScale = scale_factor;
	fVar.fileScaleFactor = scale_factor;
	
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

TimeGridWindCurv_c::TimeGridWindCurv_c () : TimeGridWindRect_c()
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
}

void TimeGridWindCurv_c::Dispose ()
{
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	
	TimeGridWindRect_c::Dispose ();
}

long TimeGridWindCurv_c::GetVelocityIndex(WorldPoint wp)
{
	long index = -1;
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	return index;
}

LongPoint TimeGridWindCurv_c::GetVelocityIndices(WorldPoint wp)
{
	LongPoint indices={-1,-1};
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		indices = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndicesFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	return indices;
}

OSErr TimeGridWindCurv_c::get_values(int n, Seconds model_time, WorldPoint3D* ref, VelocityRec* vels) {

	if(!ref || !vels) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}	
	
	WorldPoint3D rec;
	
	VelocityRec zero_vel ={0.,0.};
	
	for (int i = 0; i < n; i++) {
		
		// will get all values and let movers figure out which ones to use
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;

		vels[i] = GetScaledPatValue(model_time, rec);
		
		//delta[i].p.pLat /= 1000000;
		//delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

VelocityRec TimeGridWindCurv_c::GetScaledPatValue(const Seconds& model_time, WorldPoint3D refPoint)
{
	double timeAlpha;
	long index = -1; 
	Seconds startTime,endTime;
	VelocityRec windVelocity;
	OSErr err = 0;
	
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	
	// Check for constant wind 
	if(GetNumTimesInFile()==1 || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime)  || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
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
		startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - model_time)/(double)(endTime - startTime);
		
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
	
scale:
	
	//windVelocity.u *= fWindScale; // may want to allow some sort of scale factor, though should be in file
	//windVelocity.v *= fWindScale; 
	windVelocity.u *= fVar.fileScaleFactor; // may want to allow some sort of scale factor, though should be in file
	windVelocity.v *= fVar.fileScaleFactor; 
	
	return windVelocity;
	
}



// this code is for curvilinear grids
OSErr TimeGridWindCurv_c::TextRead(const char *path, const char *topFilePath) // don't want a map
{
	OSErr err = 0;
	char s[256], topPath[256];
	char recname[NC_MAX_NAME];
	char dimname[NC_MAX_NAME];
	char fileName[256];
	char outPath[256];

	char errmsg[256];
	errmsg[0] = 0;

	long i, j, numScanned, indexOfStart = 0;
	int status, ncid, latIndexid, lonIndexid, latid, lonid, recid, timeid, numdims;
	size_t latLength, lonLength, recs, t_len, t_len2;
	static size_t timeIndex, ptIndex[2] = {0, 0};
	static size_t pt_count[2];

	float timeVal;
	double timeConversion = 1.;
	Seconds startTime, startTime2;

	char *timeUnits = 0, month[10];
	float *lat_vals = 0, *lon_vals = 0, yearShift = 0.;
	char *modelTypeStr = 0;
	WORLDPOINTFH vertexPtsH = 0;

	// for now keep code around but probably don't need Navy curvilinear wind
	Boolean bTopFile = false, fIsNavy = false;

	if (!path || !path[0])
		return 0;
	//cerr << "TimeGridWindCurv_c::TextRead(): path = " << path << endl;

	strcpy(fVar.pathName, path);
	strcpy(s, path);

#ifndef pyGNOME
	SplitPathFile (s, fileName);
#else
	SplitPathFileName (s, fileName);
#endif

	//strcpy(fFileName, fileName); // maybe use a name from the file
	strcpy(fVar.userName, fileName); // maybe use a name from the file
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) {
		err = -1;
		goto done;
	}

	// check number of dimensions - 2D or 3D
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) {fIsNavy = false; /*goto done;*/}	
	else 
	{
		fIsNavy = true;
		// may only need to see keyword is there, since already checked grid type
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {fIsNavy = false; goto done;}	
		modelTypeStr[t_len2] = '\0';
		
		//strcpy(fFileName, modelTypeStr); 
		strcpy(fVar.userName, modelTypeStr); 
	}
	
	//if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "time", &recid); //Navy
		if (status != NC_NOERR) {
			status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
			if (status != NC_NOERR) {
				err = -1;
				goto done;
			}
		}			
	}

	//if (fIsNavy)
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) 
	{	
		status = nc_inq_varid(ncid, "ProjectionHr", &timeid); 
		if (status != NC_NOERR) {err = -1; goto done;}
	}			
	
	//if (!fIsNavy)
	//status = nc_inq_attlen(ncid, recid, "units", &t_len);	// recid is the dimension id not the variable id
	//else	// LAS has them in order, and time is unlimited, but variable/dimension names keep changing so leave this way for now
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) {
		//timeUnits = 0;	// files should always have this info
		//timeConversion = 3600.;		// default is hours
		//startTime2 = model->GetStartTime();	// default to model start time
		err = -1;
		goto done;
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];

		timeUnits = new char[t_len+1];
		//if (!fIsNavy)
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);	// recid is the dimension id not the variable id
		//else

		status = nc_get_att_text(ncid, timeid, "units", timeUnits);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}

		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		StringSubstitute(timeUnits, 'T', ' ');
		StringSubstitute(timeUnits, 'Z', ' ');
		
		numScanned = sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
							unitStr, junk, &time.year, &time.month, &time.day,
							&time.hour, &time.minute, &time.second);
		if (numScanned == 5) {
			time.hour = 0;
			time.minute = 0;
			time.second = 0;
		}
		else if (numScanned == 7) {
			// has two extra time entries ??
			time.second = 0;
		}
		else if (numScanned < 8) {
			//timeUnits = 0;	// files should always have this info
			//timeConversion = 3600.;		// default is hours
			//startTime2 = model->GetStartTime();	// default to model start time
			err = -1;
			TechError("TimeGridWindCurv_c::TextRead()", "sscanf() == 8", 0);
			goto done;
		}
		else {
			// code goes here, trouble with the DAYS since 1900 format, since converts to seconds since 1904
			if (time.year == 1900) {
				time.year += 40;
				time.day += 1; /*for the 1900 non-leap yr issue*/
				yearShift = 40.;
			}
			DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
			if (!strcmpnocase(unitStr, "HOURS") || !strcmpnocase(unitStr, "HOUR"))
				timeConversion = 3600.;
			else if (!strcmpnocase(unitStr, "MINUTES") || !strcmpnocase(unitStr, "MINUTE"))
				timeConversion = 60.;
			else if (!strcmpnocase(unitStr, "SECONDS") || !strcmpnocase(unitStr, "SECOND"))
				timeConversion = 1.;
			else if (!strcmpnocase(unitStr, "DAYS") || !strcmpnocase(unitStr, "DAY"))
				timeConversion = 24. * 3600.;
		}
	} 
	
	if (fIsNavy) {
		status = nc_inq_dimid(ncid, "gridy", &latIndexid); //Navy
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		status = nc_inq_dimid(ncid, "gridx", &lonIndexid);	//Navy
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		// option to use index values?
		status = nc_inq_varid(ncid, "grid_lat", &latid);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		status = nc_inq_varid(ncid, "grid_lon", &lonid);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
	}
	else
	{
		for (i = 0; i < numdims; i++) {
			if (i == recid)
				continue;

			status = nc_inq_dimname(ncid, i, dimname);
			if (status != NC_NOERR) {
				err = -1;
				goto done;
			}

			if (!strncmpnocase(dimname, "X", 1) ||
				!strncmpnocase(dimname, "LON", 3) ||
				!strncmpnocase(dimname, "nx", 2))
			{
				lonIndexid = i;
			}
			if (!strncmpnocase(dimname, "Y", 1) ||
				!strncmpnocase(dimname, "LAT", 3) ||
				!strncmpnocase(dimname, "ny", 2))
			{
				latIndexid = i;
			}
		}
		
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}
		
		status = nc_inq_varid(ncid, "LATITUDE", &latid);
		if (status != NC_NOERR) {
			status = nc_inq_varid(ncid, "lat", &latid);
			if (status != NC_NOERR) {
				status = nc_inq_varid(ncid, "latitude", &latid);
				if (status != NC_NOERR) {
					err = -1;
					goto done;
				}
			}
		}
		status = nc_inq_varid(ncid, "LONGITUDE", &lonid);
		if (status != NC_NOERR) {
			status = nc_inq_varid(ncid, "lon", &lonid);
			if (status != NC_NOERR) {
				status = nc_inq_varid(ncid, "longitude", &lonid);
				if (status != NC_NOERR) {
					err = -1;
					goto done;
				}
			}
		}
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(latLength * lonLength * sizeof(WorldPointF));
	if (!vertexPtsH) {
		err = memFullErr;
		goto done;
	}

	lat_vals = new float[latLength * lonLength];
	lon_vals = new float[latLength * lonLength];
	if (!lat_vals || !lon_vals) {
		err = memFullErr;
		goto done;
	}

	status = nc_get_vara_float(ncid, latid, ptIndex, pt_count, lat_vals);
	if (status != NC_NOERR) {
		err = -1;
		goto done;
	}

	status = nc_get_vara_float(ncid, lonid, ptIndex, pt_count, lon_vals);
	if (status != NC_NOERR) {
		err = -1;
		goto done;
	}

	for (i = 0; i < latLength; i++) {
		for (j = 0; j < lonLength; j++) {
			//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
			//lat_vals[(latLength-i-1)*lonLength+j]=0.;
			//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
			//lon_vals[(latLength-i-1)*lonLength+j]=0.;
			INDEXH(vertexPtsH,i * lonLength + j).pLat = lat_vals[(latLength - i - 1) * lonLength + j];
			INDEXH(vertexPtsH,i * lonLength + j).pLong = lon_vals[(latLength - i - 1) * lonLength + j];
		}
	}
	fVertexPtsH	 = vertexPtsH;
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {
		err = -1;
		goto done;
	}

	if (recs <= 0) {
		strcpy(errmsg, "No times in file. Error opening NetCDF wind file");
		err = -1;
		goto done;
	}
	
	fTimeHdl = (Seconds**)_NewHandleClear(recs * sizeof(Seconds));
	if (!fTimeHdl) {
		err = memFullErr;
		goto done;
	}

	for (i = 0; i < recs; i++) {
		Seconds newTime;

		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;

		//if (!fIsNavy)
		//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		//else

		status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {
			err = -1;
			goto done;
		}

		newTime = RoundDateSeconds(round(startTime2 + timeVal * timeConversion));
		//INDEXH(fTimeHdl,i) = startTime2+(long)(timeVal*timeConversion -yearShift*3600.*24.*365.25);	// which start time where?
		//if (i==0) startTime = startTime2+(long)(timeVal*timeConversion -yearShift*3600.*24.*365.25);
		INDEXH(fTimeHdl, i) = newTime - yearShift * 3600. * 24. * 365.25;	// which start time where?
		if (i == 0)
			startTime = newTime - yearShift * 3600. * 24. * 365.25;
	}
	
	fNumRows = latLength;
	fNumCols = lonLength;
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {
		err = -1;
		goto done;
	}

	// for now ask for an ascii file, output from Topology save option
	//{if (topFilePath[0]) {err = (dynamic_cast<TimeGridWindCurv*>(this))->ReadTopology(topFilePath); goto done;}}
	if (topFilePath[0]) {
		err = ReadTopology(topFilePath);
		goto done;
	}

	err = ReorderPoints(errmsg);	
	
done:
	if (err) {
		printNote("Error opening NetCDF wind file");
		if(fGrid) {
			fGrid->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (vertexPtsH) {
			DisposeHandle((Handle)vertexPtsH);
			vertexPtsH = 0;
			fVertexPtsH = 0;
		}
	}
	
	if (timeUnits)
		delete [] timeUnits;
	if (lat_vals)
		delete [] lat_vals;
	if (lon_vals)
		delete [] lon_vals;
	if (modelTypeStr)
		delete [] modelTypeStr;

	return err;
}


OSErr TimeGridWindCurv_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	OSErr err = 0;
	long i,j;
	char path[256], outPath[256]; 
	char *velUnits=0;
	int status, ncid, numdims;
	int wind_ucmp_id, wind_vcmp_id, angle_id, uv_ndims;
	static size_t wind_index[] = {0,0,0,0}, angle_index[] = {0,0};
	static size_t wind_count[4], angle_count[2];
	size_t velunit_len;
	float *wind_uvals = 0,*wind_vvals = 0, fill_value=-1e-72, velConversion=1.;
	short *wind_uvals_Navy = 0,*wind_vvals_Navy = 0, fill_value_Navy;
	float *angle_vals = 0;
	long totalNumberOfVels = fNumRows * fNumCols;
	VelocityFH velH = 0;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	float scale_factor = 1.,angle = 0.,u_grid,v_grid;
	Boolean bRotated = true, fIsNavy = false, bIsNWSSpeedDirData = false;
	
	errmsg[0]=0;
	
	//strcpy(path,fPathName);
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	//if (status != NC_NOERR) {err = -1; goto done;}
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
	if (numdims>=4)	// should check what the dimensions are, CO-OPS uses sigma
	{
		wind_count[1] = 1;	// depth
		wind_count[2] = latlength;
		wind_count[3] = lonlength;
	}
	else
	{
		wind_count[1] = latlength;	
		wind_count[2] = lonlength;
	}
	angle_count[0] = latlength;
	angle_count[1] = lonlength;
	
	//wind_count[0] = latlength;		// a fudge for the PWS format which has u(lat,lon) not u(time,lat,lon)
	//wind_count[1] = lonlength;
	
	if (fIsNavy)
	{
		// need to check if type is float or short, if float no scale factor?
		wind_uvals = new float[latlength*lonlength]; 
		if(!wind_uvals) {TechError("TimeGridWindCurv_c::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		wind_vvals = new float[latlength*lonlength]; 
		if(!wind_vvals) {TechError("TimeGridWindCurv_c::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		
		angle_vals = new float[latlength*lonlength]; 
		if(!angle_vals) {TechError("TimeGridWindCurv_c::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		status = nc_inq_varid(ncid, "air_gridu", &wind_ucmp_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "air_gridv", &wind_vcmp_id);	
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_get_vara_float(ncid, wind_ucmp_id, wind_index, wind_count, wind_uvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_vara_float(ncid, wind_vcmp_id, wind_index, wind_count, wind_vvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_att_float(ncid, wind_ucmp_id, "_FillValue", &fill_value);
		//if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_att_float(ncid, wind_ucmp_id, "scale_factor", &scale_factor);
		//if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "grid_orient", &angle_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_vara_float(ncid, angle_id, angle_index, angle_count, angle_vals);
		if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
	}
	else
	{
		wind_uvals = new float[latlength*lonlength]; 
		if(!wind_uvals) {TechError("TimeGridWindCurv_c::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		wind_vvals = new float[latlength*lonlength]; 
		if(!wind_vvals) {TechError("TimeGridWindCurv_c::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		status = nc_inq_varid(ncid, "air_u", &wind_ucmp_id);
		if (status != NC_NOERR)
		{
			status = nc_inq_varid(ncid, "u", &wind_ucmp_id);
			if (status != NC_NOERR)
			{
				status = nc_inq_varid(ncid, "U", &wind_ucmp_id);
				if (status != NC_NOERR)
				{
					status = nc_inq_varid(ncid, "WindSpd_SFC", &wind_ucmp_id);
					if (status != NC_NOERR)
					{err = -1; goto done;}
					bIsNWSSpeedDirData = true;
				}
				//{err = -1; goto done;}
			}
			//{err = -1; goto done;}
		}
		if (bIsNWSSpeedDirData)
		{
			status = nc_inq_varid(ncid, "WindDir_SFC", &wind_vcmp_id);
			if (status != NC_NOERR)
			{err = -2; goto done;}
		}
		else
		{
			status = nc_inq_varid(ncid, "air_v", &wind_vcmp_id);
			if (status != NC_NOERR) 
			{
				status = nc_inq_varid(ncid, "v", &wind_vcmp_id);
				if (status != NC_NOERR) 
				{
					status = nc_inq_varid(ncid, "V", &wind_vcmp_id);
					if (status != NC_NOERR)
					{err = -1; goto done;}
				}
				//{err = -1; goto done;}
			}
		}
		
		status = nc_inq_varndims(ncid, wind_ucmp_id, &uv_ndims);
		if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {wind_count[1] = latlength; wind_count[2] = lonlength;}}	// could have more dimensions than are used in u,v
		
		status = nc_get_vara_float(ncid, wind_ucmp_id, wind_index, wind_count, wind_uvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_vara_float(ncid, wind_vcmp_id, wind_index, wind_count, wind_vvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_att_float(ncid, wind_ucmp_id, "_FillValue", &fill_value);
		if (status != NC_NOERR) 
		{
			status = nc_get_att_float(ncid, wind_ucmp_id, "Fill_Value", &fill_value);
			if (status != NC_NOERR)
			{
				status = nc_get_att_float(ncid, wind_ucmp_id, "fillValue", &fill_value);// nws 2.5km
				if (status != NC_NOERR)
				{
					status = nc_get_att_float(ncid, wind_ucmp_id, "missing_value", &fill_value);
				}
				/*if (status != NC_NOERR)*//*err = -1; goto done;*/}}	// don't require
		//if (status != NC_NOERR) {err = -1; goto done;}	// don't require
	}	
	
	status = nc_inq_attlen(ncid, wind_ucmp_id, "units", &velunit_len);
	if (status == NC_NOERR)
	{
		velUnits = new char[velunit_len+1];
		status = nc_get_att_text(ncid, wind_ucmp_id, "units", velUnits);
		if (status == NC_NOERR)
		{
			velUnits[velunit_len] = '\0'; 
			if (!strcmpnocase(velUnits,"knots"))
				velConversion = KNOTSTOMETERSPERSEC;
			else if (!strcmpnocase(velUnits,"m/s"))
				velConversion = 1.0;
		}
	}
	
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) {err = memFullErr; goto done;}
	//for (i=0;i<totalNumberOfVels;i++)
	for (i=0;i<latlength;i++)
	{
		for (j=0;j<lonlength;j++)
		{
			if (wind_uvals[(latlength-i-1)*lonlength+j]==fill_value)
				wind_uvals[(latlength-i-1)*lonlength+j]=0.;
			if (wind_vvals[(latlength-i-1)*lonlength+j]==fill_value)
				wind_vvals[(latlength-i-1)*lonlength+j]=0.;
			if (isnan(wind_uvals[(latlength-i-1)*lonlength+j])) 
				wind_uvals[(latlength-i-1)*lonlength+j]=0.;
			if (isnan(wind_vvals[(latlength-i-1)*lonlength+j])) 
				wind_vvals[(latlength-i-1)*lonlength+j]=0.;
			if (fIsNavy)
			{
				u_grid = (float)wind_uvals[(latlength-i-1)*lonlength+j];
				v_grid = (float)wind_vvals[(latlength-i-1)*lonlength+j];
				if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
				INDEXH(velH,i*lonlength+j).u = u_grid*cos(angle*PI/180.)-v_grid*sin(angle*PI/180.);
				INDEXH(velH,i*lonlength+j).v = u_grid*sin(angle*PI/180.)+v_grid*cos(angle*PI/180.);
			}
			else if (bIsNWSSpeedDirData)
			{
				//INDEXH(velH,i*lonlength+j).u = KNOTSTOMETERSPERSEC * wind_uvals[(latlength-i-1)*lonlength+j] * sin ((PI/180.) * wind_vvals[(latlength-i-1)*lonlength+j]);	// need units
				//INDEXH(velH,i*lonlength+j).v = KNOTSTOMETERSPERSEC * wind_uvals[(latlength-i-1)*lonlength+j] * cos ((PI/180.) * wind_vvals[(latlength-i-1)*lonlength+j]);
				// since direction is from rather than to need to switch the sign
				//INDEXH(velH,i*lonlength+j).u = -1. * KNOTSTOMETERSPERSEC * wind_uvals[(latlength-i-1)*lonlength+j] * sin ((PI/180.) * wind_vvals[(latlength-i-1)*lonlength+j]);	// need units
				//INDEXH(velH,i*lonlength+j).v = -1. * KNOTSTOMETERSPERSEC * wind_uvals[(latlength-i-1)*lonlength+j] * cos ((PI/180.) * wind_vvals[(latlength-i-1)*lonlength+j]);
				INDEXH(velH,i*lonlength+j).u = -1. * velConversion * wind_uvals[(latlength-i-1)*lonlength+j] * sin ((PI/180.) * wind_vvals[(latlength-i-1)*lonlength+j]);	// need units
				INDEXH(velH,i*lonlength+j).v = -1. * velConversion * wind_uvals[(latlength-i-1)*lonlength+j] * cos ((PI/180.) * wind_vvals[(latlength-i-1)*lonlength+j]);
			}
			else
			{
				// Look for a land mask, but do this if don't find one - float mask(lat,lon) - 1,0 which is which?
				//if (wind_uvals[(latlength-i-1)*lonlength+j]==0. && wind_vvals[(latlength-i-1)*lonlength+j]==0.)
				//wind_uvals[(latlength-i-1)*lonlength+j] = wind_vvals[(latlength-i-1)*lonlength+j] = 1e-06;
				
				// just leave fillValue as velocity for new algorithm - comment following lines out
				// should eliminate the above problem, assuming fill_value is a land mask
				// leave for now since not using a map...use the entire grid
				/////////////////////////////////////////////////
				
				INDEXH(velH,i*lonlength+j).u = /*KNOTSTOMETERSPERSEC**/velConversion*wind_uvals[(latlength-i-1)*lonlength+j];	// need units
				INDEXH(velH,i*lonlength+j).v = /*KNOTSTOMETERSPERSEC**/velConversion*wind_vvals[(latlength-i-1)*lonlength+j];
			}
		}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	
	//fWindScale = scale_factor;	// hmm, this forces a reset of scale factor each time, overriding any set by hand
	fVar.fileScaleFactor = scale_factor;	// hmm, this forces a reset of scale factor each time, overriding any set by hand
	
done:
	if (err)
	{
		if (err==-2)
			strcpy(errmsg,"Error reading wind data from NetCDF file");
		else
			strcpy(errmsg,"Error reading wind direction data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		//printNote("Error opening NetCDF file");
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (wind_uvals) {delete [] wind_uvals; wind_uvals = 0;}
	if (wind_vvals) {delete [] wind_vvals; wind_vvals = 0;}
	if (angle_vals) {delete [] angle_vals; angle_vals = 0;}
	return err;
}

OSErr TimeGridWindCurv_c::ExportTopology(char* path)
{
	// export NetCDF curvilinear info so don't have to regenerate each time
	// move to NetCDFWindMover so Tri can use it too
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;	// should we bother with the map stuff? 
	FILE *fp = fopen(path, "w");
	//BFPB bfpb;
	//PtCurMap *map = GetPtCurMap();
	
	triGrid = dynamic_cast<TTriGridVel*>(this->fGrid);
	if (!triGrid) {printError("There is no topology to export"); return -1;}
	dagTree = triGrid->GetDagTree();
	if (dagTree) 
	{
		ptsH = dagTree->GetPointsHdl();
		topH = dagTree->GetTopologyHdl();
		treeH = dagTree->GetDagTreeHdl();
	}
	else 
	{
		printError("There is no topology to export");
		return -1;
	}
	if(!ptsH || !topH || !treeH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	//if (moverMap->IAm(TYPE_PTCURMAP))
	/*if (map)
	{
		//boundaryTypeH = (dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		//boundarySegmentsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		//boundaryPointsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundaryPoints();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		boundaryPointsH = map->GetBoundaryPoints();
		if (!boundaryTypeH || !boundarySegmentsH || !boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
	}*/
	
	//(void)hdelete(0, 0, path);
	//if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	//{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	//if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	//{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else {printError("There is no transpose array"); err = -1; goto done;}
	sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	for(i=0;i<n;i++)
	{	
		sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
		//strcpy(buffer,topoStr);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
	sprintf(hdrStr,"Vertices\t%ld\n",nver);	// total vertices
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	sprintf(hdrStr,"%ld\t%ld\n",nver,nver);	// junk line
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	for(i=0;i<nver;i++)
	{	
		x = (*ptsH)[i].h/1000000.0;
		y =(*ptsH)[i].v/1000000.0;
		//sprintf(topoStr,"%ld\t%lf\t%lf\t%lf\n",i+1,x,y,(*gDepths)[i]);
		//sprintf(topoStr,"%ld\t%lf\t%lf\n",i+1,x,y);
		sprintf(topoStr,"%lf\t%lf\n",x,y);
		//strcpy(buffer,topoStr);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	//code goes here, boundary points - an optional handle, only for curvilinear case
	
	/*if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]);
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
	
	nBoundarySegs = 0;
	if (boundaryTypeH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundaryTypeH)/sizeof(long);	// should be same size as previous handle
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2) nWaterBoundaries++;
		}
		sprintf(hdrStr,"WaterBoundaries\t%ld\t%ld\n",nWaterBoundaries,nBoundarySegs);	
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			if ((*boundaryTypeH)[i]==2)
				//sprintf(topoStr,"%ld\n",(*boundaryTypeH)[i]);
			{
				sprintf(topoStr,"%ld\n",i);
				strcpy(buffer,topoStr);
				if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
			}
		}
	}
	nBoundaryPts = 0;
	if (boundaryPointsH) 
	{
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	// should be same size as previous handle
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	// when reading in subtracts 1
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}*/
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	sprintf(hdrStr,"Topology\t%ld\n",numTriangles);
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	for(i = 0; i< numTriangles;i++)
	{
		v1 = (*topH)[i].vertex1;
		v2 = (*topH)[i].vertex2;
		v3 = (*topH)[i].vertex3;
		n1 = (*topH)[i].adjTri1;
		n2 = (*topH)[i].adjTri2;
		n3 = (*topH)[i].adjTri3;
		sprintf(topoStr, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
				v1, v2, v3, n1, n2, n3);
		
		/////
		//strcpy(buffer,topoStr);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	
	numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	sprintf(hdrStr,"DAGTree\t%ld\n",dagTree->fNumBranches);
	//strcpy(buffer,hdrStr);
	//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	fwrite(hdrStr,sizeof(char),strlen(hdrStr),fp);
	
	for(i = 0; i<dagTree->fNumBranches; i++)
	{
		sprintf(topoStr,"%ld\t%ld\t%ld\n",(*treeH)[i].topoIndex,(*treeH)[i].branchLeft,(*treeH)[i].branchRight);
		//strcpy(buffer,topoStr);
		//if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		fwrite(topoStr,sizeof(char),strlen(topoStr),fp);
	}
	
done:
	// 
	//FSCloseBuf(&bfpb);
	fclose(fp);
	if(err) {	
		printError("Error writing topology");
		//(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}

// simplify for wind data - no map needed, no mask 
OSErr TimeGridWindCurv_c::ReorderPoints(char* errmsg) 
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
	//LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	
	LONGH ptIndexHdl = (LONGH)_NewHandleClear(nv_ext * sizeof(**ptIndexHdl));
	LONGH verdatPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**verdatPtsH));
	GridCellInfoHdl gridCellInfo = (GridCellInfoHdl)_NewHandleClear(nv * sizeof(**gridCellInfo));
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect triBounds;
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	VelocityFH velocityH = 0;
	
	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH /*|| !maskH2*/) {err = memFullErr; goto done;}
	
	err = ReadTimeData(indexOfStart,&velocityH,errmsg);	// try to use velocities to set grid
	if (err) return err;
	
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			// eventually will need to have a land mask, for now assume fillValue represents land
			//if (INDEXH(velocityH,i*fNumCols+j).u==0 && INDEXH(velocityH,i*fNumCols+j).v==0)	// land point
			if (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)	// land point
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
	
	for (i=0; i<=numVerdatPts; i++)	// make a list of grid points that will be used for triangles
	{
		float fLong, fLat, fDepth, dLon, dLat, dLon1, dLon2, dLat1, dLat2;
		//double val, u=0., v=0.;
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
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in TimeGridWindCurv_c::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
	this->SetGridBounds(triBounds);
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
			strcpy(errmsg,"An error occurred in TimeGridWindCurv_c::ReorderPoints");
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
		//if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
	return err;
}


// import NetCDF curvilinear info so don't have to regenerate
OSErr TimeGridWindCurv_c::ReadTopology(vector<string> &linesInFile)
{
	OSErr err = 0;
	string currentLine;

	char s[1024], errmsg[256];

	long i, numPoints, numTopoPoints, line = 0, numPts;

	CHARH f = 0;
	TopologyHdl topo = 0;
	LongPointHdl pts = 0;
	FLOATH depths = 0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;

	TTriGridVel *triGrid = 0;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;

	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs = 0, waterBoundaries = 0, boundaryPts = 0;

	errmsg[0] = 0;

	MySpinCursor();

	// No header
	// start with transformation array and vertices
	currentLine = linesInFile[(line)++];
	if (IsTransposeArrayHeaderLine(currentLine, numPts)) {
		err = ReadTransposeArray(linesInFile, &line, &fVerdatToNetCDFH, numPts, errmsg);
		if (err) {
			strcpy(errmsg, "Error in ReadTransposeArray");
			goto done;
		}
	}
	else {
		err = -1;
		strcpy(errmsg, "Error in Transpose header line");
		goto done;
	}

	err = ReadTVertices(linesInFile, &line, &pts, &depths, errmsg);
	if (err)
		goto done;

	if (pts) {
		LongPoint thisLPoint;

		numPts = _GetHandleSize((Handle)pts) / sizeof(LongPoint);
		if (numPts > 0) {
			WorldPoint wp;
			for (i = 0; i < numPts; i++) {
				thisLPoint = INDEXH(pts, i);

				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;

				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}

	MySpinCursor();

	//code goes here, boundary points
	currentLine = linesInFile[(line)++];
	if (IsBoundarySegmentHeaderLine(currentLine, numBoundarySegs)) {
		// Boundary data from CATs
		MySpinCursor();

		if (numBoundarySegs > 0)
			err = ReadBoundarySegs(linesInFile, &line, &boundarySegs, numBoundarySegs, errmsg);
		if (err)
			goto done;

		currentLine = linesInFile[(line)++];
	}

	MySpinCursor(); // JLM 8/4/99

	if (IsWaterBoundaryHeaderLine(currentLine, numWaterBoundaries, numBoundaryPts)) {
		// Boundary types from CATs
		MySpinCursor();

		if (numBoundaryPts > 0)
			err = ReadWaterBoundaries(linesInFile, &line,
									  &waterBoundaries,
									  numWaterBoundaries, numBoundaryPts, errmsg);
		if (err)
			goto done;
		currentLine = linesInFile[(line)++];
	}

	MySpinCursor(); // JLM 8/4/99

	if (IsBoundaryPointsHeaderLine(currentLine, numBoundaryPts)) {
		// Boundary data from CATs
		MySpinCursor();

		if (numBoundaryPts > 0)
			err = ReadBoundaryPts(linesInFile, &line,
								  &boundaryPts,
								  numBoundaryPts, errmsg);
		if (err)
			goto done;
		currentLine = linesInFile[(line)++];
	}

	MySpinCursor(); // JLM 8/4/99

	if (IsTTopologyHeaderLine(currentLine, numTopoPoints)) {
		// Topology from CATs
		MySpinCursor();

		err = ReadTTopologyBody(linesInFile, &line,
								&topo, &velH,
								errmsg, numTopoPoints, FALSE);
		if (err)
			goto done;
		currentLine = linesInFile[(line)++];
	}
	else {
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}

	MySpinCursor(); // JLM 8/4/99


	if (IsTIndexedDagTreeHeaderLine(currentLine, numPoints)) {
		// DagTree from CATs
		MySpinCursor();

		err = ReadTIndexedDagTreeBody(linesInFile, &line,
									  &tree,
									  errmsg, numPoints);
		if (err)
			goto done;
	}
	else {
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg, "Error in dag tree header line");
		if (err)
			goto done;
	}

	MySpinCursor(); // JLM 8/4/99

	/////////////////////////////////////////////////
	// if map information is in the file just toss it
	/*if (waterBoundaries && (this -> moverMap == model -> uMap))
	 {
	 //PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
	 PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
	 if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
	 // maybe move up and have the map read in the boundary information
	 map->SetBoundarySegs(boundarySegs);
	 map->SetWaterBoundaries(waterBoundaries);

	 *newMap = map;
	 }*/

	{	// wind will always be on another map
		if (waterBoundaries) {
			DisposeHandle((Handle)waterBoundaries);
			waterBoundaries = 0;
		}
		if (boundarySegs) {
			DisposeHandle((Handle)boundarySegs);
			boundarySegs = 0;
		}
		if (boundaryPts) {
			DisposeHandle((Handle)boundaryPts);
			boundaryPts = 0;
		}
	}

	/*if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	 {
	 if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
	 if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
	 }*/

	/////////////////////////////////////////////////
	triGrid = new TTriGridVel;
	if (!triGrid) {
		err = true;
		TechError("Error in TimeGridWindCurv_c::ReadTopology()", "new TTriGridVel", err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;

	triGrid->SetBounds(bounds);
	//triGrid -> SetDepths(depths);

	dagTree = new TDagTree(pts, topo, tree.treeHdl, velH, tree.numBranches);
	if (!dagTree) {
		err = -1;
		printError("Unable to read Extended Topology file.");
		goto done;
	}

	triGrid->SetDagTree(dagTree);

	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//depths = 0;

done:

	if (depths) {
		DisposeHandle((Handle)depths);
		depths = 0;
	}
	if (f) {
		_HUnlock((Handle)f);
		DisposeHandle((Handle)f);
		f = 0;
	}

	if (err) {
		if (!errmsg[0])
			strcpy(errmsg, "An error occurred in TimeGridWindCurv_c::ReadTopology");
		printError(errmsg);
		if (pts) {
			DisposeHandle((Handle)pts);
			pts = 0;
		}
		if (topo) {
			DisposeHandle((Handle)topo);
			topo = 0;
		}
		if (velH) {
			DisposeHandle((Handle)velH);
			velH = 0;
		}
		if (depths) {
			DisposeHandle((Handle)depths);
			depths = 0;
		}
		if (tree.treeHdl) {
			DisposeHandle((Handle)tree.treeHdl);
			tree.treeHdl = 0;
		}
		if (fGrid) {
			fGrid->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if (waterBoundaries) {
			DisposeHandle((Handle)waterBoundaries);
			waterBoundaries = 0;
		}
		if (boundarySegs) {
			DisposeHandle((Handle)boundarySegs);
			boundarySegs = 0;
		}
		if (boundaryPts) {
			DisposeHandle((Handle)boundaryPts);
			boundaryPts = 0;
		}
	}

	return err;
}


// import NetCDF curvilinear info so don't have to regenerate
OSErr TimeGridWindCurv_c::ReadTopology(const char *path)
{
	vector<string> linesInFile;

	ReadLinesInFile(path, linesInFile);
	return ReadTopology(linesInFile);
}


OSErr TimeGridWindCurv_c::GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp)
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
			fLat = 2*(INDEXH(fVertexPtsH,jIndex-1).pLat + dLat1) - (INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLat+dLat2);
			dLon1 = INDEXH(fVertexPtsH,fNumCols+jIndex-1).pLong - INDEXH(fVertexPtsH,jIndex-1).pLong;
			dLon2 = (INDEXH(fVertexPtsH,fNumCols+jIndex-2).pLong - INDEXH(fVertexPtsH,jIndex-2).pLong);
			fLong = 2*(INDEXH(fVertexPtsH,jIndex-1).pLong-dLon1) - (INDEXH(fVertexPtsH,jIndex-2).pLong-dLon2);
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


TimeGridWindIce_c::TimeGridWindIce_c () : TimeGridWindCurv_c(), TimeGridVel_c()
{
	memset(&fStartDataIce,0,sizeof(fStartDataIce));
	fStartDataIce.timeIndex = UNASSIGNEDINDEX; 
	fStartDataIce.dataHdl = 0; 
	memset(&fEndDataIce,0,sizeof(fEndDataIce));
	fEndDataIce.timeIndex = UNASSIGNEDINDEX;
	fEndDataIce.dataHdl = 0;
	
	memset(&fStartDataThickness,0,sizeof(fStartDataThickness));
	fStartDataThickness.timeIndex = UNASSIGNEDINDEX; 
	fStartDataThickness.dataHdl = 0; 
	memset(&fEndDataThickness,0,sizeof(fEndDataThickness));
	fEndDataThickness.timeIndex = UNASSIGNEDINDEX;
	fEndDataThickness.dataHdl = 0;
	
	memset(&fStartDataFraction,0,sizeof(fStartDataFraction));
	fStartDataFraction.timeIndex = UNASSIGNEDINDEX; 
	fStartDataFraction.dataHdl = 0; 
	memset(&fEndDataFraction,0,sizeof(fEndDataFraction));
	fEndDataFraction.timeIndex = UNASSIGNEDINDEX;
	fEndDataFraction.dataHdl = 0;
	
}

void TimeGridWindIce_c::Dispose ()
{
	if(fStartDataIce.dataHdl)DisposeLoadedData(&fStartDataIce); 
	if(fEndDataIce.dataHdl)DisposeLoadedData(&fEndDataIce); 
	if(fStartDataThickness.dataHdl)DisposeLoadedData(&fStartDataThickness);
	if(fEndDataThickness.dataHdl)DisposeLoadedData(&fEndDataThickness);
	if(fStartDataFraction.dataHdl)DisposeLoadedData(&fStartDataFraction);
	if(fEndDataFraction.dataHdl)DisposeLoadedData(&fEndDataFraction);
	
	TimeGridWindCurv_c::Dispose ();
}

/*void TimeGridWindIce_c::DisposeLoadedFieldData(LoadedFieldData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}*/

void TimeGridWindIce_c::DisposeLoadedStartData()
{
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fStartDataIce.dataHdl)DisposeLoadedData(&fStartDataIce);
	if(fStartDataThickness.dataHdl)DisposeLoadedData(&fStartDataThickness);
	if(fStartDataFraction.dataHdl)DisposeLoadedData(&fStartDataFraction);
}

void TimeGridWindIce_c::DisposeLoadedEndData()
{
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData); 
	if(fEndDataIce.dataHdl)DisposeLoadedData(&fEndDataIce);
	if(fEndDataThickness.dataHdl)DisposeLoadedData(&fEndDataThickness);
	if(fEndDataFraction.dataHdl)DisposeLoadedData(&fEndDataFraction);
}

void TimeGridWindIce_c::ShiftInterval()
{
	fStartData = fEndData;
	fStartDataIce = fEndDataIce;
	fStartDataThickness = fEndDataThickness;
	fStartDataFraction = fEndDataFraction;
	ClearLoadedEndData();
	
}

/*void TimeGridWindIce_c::ClearLoadedData(LoadedFieldData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}*/

void TimeGridWindIce_c::ClearLoadedEndData()
{
	if(fEndData.dataHdl)ClearLoadedData(&fEndData); 
	if(fEndDataIce.dataHdl)ClearLoadedData(&fEndDataIce);
	if(fEndDataThickness.dataHdl)ClearLoadedData(&fEndDataThickness);
	if(fEndDataFraction.dataHdl)ClearLoadedData(&fEndDataFraction);
	
}

OSErr TimeGridWindIce_c::SetInterval(char *errmsg, const Seconds& model_time)
{
	OSErr err = 0;

	long timeDataInterval = 0;
	Boolean intervalLoaded = this->CheckInterval(timeDataInterval, model_time);
	long indexOfStart = timeDataInterval - 1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this->GetNumTimesInFile();

	errmsg[0] = 0;

	if (intervalLoaded)
		return 0;

	// check for constant current 
	if (numTimesInFile == 1 && !(GetNumFiles() > 1))
		//or if(timeDataInterval==-1)
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}

	if (timeDataInterval == 0 && fAllowExtrapolationInTime) {
		indexOfStart = 0;
		indexOfEnd = -1;
	}

	//cerr << "timeDataInterval: " << timeDataInterval << endl;
	if (timeDataInterval == 0 || timeDataInterval == numTimesInFile)
	{

		//cerr << "GetNumFiles(): " << GetNumFiles() << endl;
		// before the first step in the file
		if (GetNumFiles() > 1) {
			if ((err = CheckAndScanFile(errmsg, model_time)) || fOverLap)
				goto done;
			
			intervalLoaded = this->CheckInterval(timeDataInterval, model_time);
			
			indexOfStart = timeDataInterval - 1;
			indexOfEnd = timeDataInterval;
			numTimesInFile = this->GetNumTimesInFile();
			if (fAllowExtrapolationInTime &&
				(timeDataInterval == numTimesInFile || timeDataInterval == 0))
			{
				if (intervalLoaded)
					return 0;
				indexOfEnd = -1;
				if (timeDataInterval == 0)
					indexOfStart = 0;	// if we allow extrapolation we need to load the first time
			}
		}
		else {
			if(fTimeAlpha>=0 && timeDataInterval == numTimesInFile)
				indexOfEnd = 0;	// start over
			else if (fAllowExtrapolationInTime && timeDataInterval == numTimesInFile) {
				fStartData.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fStartDataIce.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fStartDataThickness.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fStartDataFraction.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
				fEndDataIce.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
				fEndDataThickness.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
				fEndDataFraction.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else if (fAllowExtrapolationInTime && timeDataInterval == 0) {
				fStartData.timeIndex = 0;//check if time > last model time in all files
				fStartDataIce.timeIndex = 0;//check if time > last model time in all files
				fStartDataFraction.timeIndex = 0;//check if time > last model time in all files
				fStartDataThickness.timeIndex = 0;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
				fEndDataIce.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
				fEndDataFraction.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
				fEndDataThickness.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else {
				err = -1;
				strcpy(errmsg,"Time outside of interval being modeled");
				goto done;
			}
		}
		// code goes here, if time > last time in files allow user to continue
		// leave last two times loaded? move last time to start and nothing for end?
		// redefine as constant or just check if time > last time and some flag set
		// careful with timeAlpha, really just want to use the last time but needs to be loaded
		// want to check so that don't reload at every step, should recognize last time is ok
	}
	//else // load the two intervals
	{
		//DisposeLoadedData(&fStartData);
		DisposeLoadedStartData();
		
		if(indexOfStart == fEndData.timeIndex) // passing into next interval
		{
			ShiftInterval();
			//ClearAllLoadedEndData();
			//ClearLoadedData(&fEndData);
		}
		else
		{
			//DisposeLoadedData(&fEndData);
			DisposeLoadedEndData();
		}
		
		//////////////////
		
		if(fStartData.dataHdl == 0 && indexOfStart >= 0) 
		{ // start data is not loaded
			err = this -> ReadTimeData(indexOfStart,&fStartData.dataHdl,errmsg);
			err = this -> ReadTimeDataIce(indexOfStart,&fStartDataIce.dataHdl,errmsg);
			err = this -> ReadTimeDataFields(indexOfStart,&fStartDataThickness.dataHdl,&fStartDataFraction.dataHdl,errmsg);
			if(err) goto done;
			fStartData.timeIndex = indexOfStart;
			fStartDataIce.timeIndex = indexOfStart;
			fStartDataThickness.timeIndex = indexOfStart;
			fStartDataFraction.timeIndex = indexOfStart;
		}	
		
		if(indexOfEnd < numTimesInFile && indexOfEnd != UNASSIGNEDINDEX)  // not past the last interval and not constant current
		{
			err = this -> ReadTimeData(indexOfEnd,&fEndData.dataHdl,errmsg);
			err = this -> ReadTimeDataIce(indexOfEnd,&fEndDataIce.dataHdl,errmsg);
			err = this -> ReadTimeDataFields(indexOfEnd,&fEndDataThickness.dataHdl,&fEndDataFraction.dataHdl,errmsg);
			if(err) goto done;
			fEndData.timeIndex = indexOfEnd;
			fEndDataIce.timeIndex = indexOfEnd;
			fEndDataThickness.timeIndex = indexOfEnd;
			fEndDataFraction.timeIndex = indexOfEnd;
		}
	}
	
done:	
	if(err)
	{
		if(!errmsg[0])strcpy(errmsg,"Error in TimeGridWindIce::SetInterval()");
		//DisposeLoadedData(&fStartData);
		//DisposeLoadedData(&fEndData);
		DisposeLoadedStartData();
		DisposeLoadedEndData();
	}
	return err;
	
}



OSErr TimeGridWindIce_c::CheckAndScanFile(char *errmsg, const Seconds& model_time)
{
	OSErr err = 0;
	Seconds time = model_time, startTime, endTime, lastEndTime, testTime, firstStartTime; 
	
	long i, numFiles = GetNumFiles();

	errmsg[0] = 0;

	if (fEndData.timeIndex!=UNASSIGNEDINDEX)
		testTime = (*fTimeHdl)[fEndData.timeIndex];	// currently loaded end time
	
	firstStartTime = (*fInputFilesHdl)[0].startTime + fTimeShift;
	for (i = 0; i < numFiles; i++)
	{
		startTime = (*fInputFilesHdl)[i].startTime + fTimeShift;
		endTime = (*fInputFilesHdl)[i].endTime + fTimeShift;
		if (startTime<=time&&time<=endTime && !(startTime==endTime))
		{
			//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			DisposeTimeHdl();
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl);	
			
			// code goes here, check that start/end times match
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
			fOverLap = false;
			return err;
		}
		if (startTime==endTime && startTime==time)	// one time in file, need to overlap
		{
			long fileNum;
			if (i<numFiles-1)
				fileNum = i+1;
			else
				fileNum = i;
			fOverLapStartTime = (*fInputFilesHdl)[fileNum-1].endTime;	// last entry in previous file
			//DisposeLoadedData(&fStartData);
			DisposeLoadedStartData();
			/*if (fOverLapStartTime==testTime)	// shift end time data to start time data
			 {
			 fStartData = fEndData;
			 ClearLoadedData(&fEndData);
			 }
			 else*/
			{
				//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
				DisposeTimeHdl();
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeHdl);	
				
				//DisposeLoadedData(&fEndData);
				DisposeLoadedEndData();
				strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum-1].pathName);
				err = this->ReadTimeData(GetNumTimesInFile() - 1, &fStartData.dataHdl, errmsg);
				err = this->ReadTimeDataIce(GetNumTimesInFile() - 1, &fStartDataIce.dataHdl, errmsg);
				err = this->ReadTimeDataFields(GetNumTimesInFile() - 1, &fStartDataThickness.dataHdl,  &fStartDataFraction.dataHdl, errmsg);
				if (err)
					return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			fStartDataIce.timeIndex = UNASSIGNEDINDEX;
			fStartDataFraction.timeIndex = UNASSIGNEDINDEX;
			fStartDataThickness.timeIndex = UNASSIGNEDINDEX;
			//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			DisposeTimeHdl();
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeHdl);	
			
			strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			err = this -> ReadTimeDataIce(0,&fEndDataIce.dataHdl,errmsg);
			err = this -> ReadTimeDataFields(0,&fEndDataThickness.dataHdl,&fEndDataFraction.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fEndDataIce.timeIndex = 0;
			fEndDataFraction.timeIndex = 0;
			fEndDataThickness.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		if (i>0 && (lastEndTime<time && time<startTime))
		{
			fOverLapStartTime = (*fInputFilesHdl)[i-1].endTime;	// last entry in previous file
			DisposeLoadedData(&fStartData);
			if (fOverLapStartTime==testTime)	// shift end time data to start time data
			{
				//fStartData = fEndData;
				//ClearLoadedData(&fEndData);
				ShiftInterval();
			}
			else
			{
				//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
				DisposeTimeHdl();
				err = ScanFileForTimes((*fInputFilesHdl)[i-1].pathName,&fTimeHdl);	
				
				//DisposeLoadedData(&fEndData);
				DisposeLoadedEndData();
				strcpy(fVar.pathName,(*fInputFilesHdl)[i-1].pathName);
				err = this->ReadTimeData(GetNumTimesInFile() - 1, &fStartData.dataHdl, errmsg);
				err = this->ReadTimeDataIce(GetNumTimesInFile() - 1, &fStartDataIce.dataHdl, errmsg);
				err = this->ReadTimeDataFields(GetNumTimesInFile() - 1, &fStartDataThickness.dataHdl, &fStartDataFraction.dataHdl, errmsg);
				if (err)
					return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			fStartDataIce.timeIndex = UNASSIGNEDINDEX;
			fStartDataThickness.timeIndex = UNASSIGNEDINDEX;
			fStartDataFraction.timeIndex = UNASSIGNEDINDEX;
			//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			DisposeTimeHdl();
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl);	
			
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			err = this -> ReadTimeDataIce(0,&fEndDataIce.dataHdl,errmsg);
			err = this -> ReadTimeDataFields(0,&fEndDataThickness.dataHdl,&fEndDataFraction.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fEndDataIce.timeIndex = 0;
			fEndDataThickness.timeIndex = 0;
			fEndDataFraction.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		lastEndTime = endTime;
	}
	if (fAllowExtrapolationInTime && time > lastEndTime)
	{
		//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		DisposeTimeHdl();
		err = ScanFileForTimes((*fInputFilesHdl)[numFiles-1].pathName,&fTimeHdl);	
		
		// code goes here, check that start/end times match
		strcpy(fVar.pathName,(*fInputFilesHdl)[numFiles-1].pathName);
		fOverLap = false;
		return err;
	}
	if (fAllowExtrapolationInTime && time < firstStartTime)
	{
		//if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		DisposeTimeHdl();
		err = ScanFileForTimes((*fInputFilesHdl)[0].pathName,&fTimeHdl);	
		
		// code goes here, check that start/end times match
		strcpy(fVar.pathName,(*fInputFilesHdl)[0].pathName);
		fOverLap = false;
		return err;
	}
	strcpy(errmsg,"Time outside of interval being modeled");
	return -1;	
}

double TimeGridWindIce_c::GetStartFieldValue(long index, long field)
{	// 
	double value = 0;
	if (index>=0)
	{
		if (field==1)	// thickness
		{
			if (fStartDataThickness.dataHdl) value = INDEXH(fStartDataThickness.dataHdl,index);
			if (value==fFillValue) value = 0;
		}
		if (field==2)	// fraction
		{
			if (fStartDataFraction.dataHdl) value = INDEXH(fStartDataFraction.dataHdl,index);
			if (value==fFillValue) value = 0;
		}
	}
	return value;
}

double TimeGridWindIce_c::GetEndFieldValue(long index, long field)
{	// 
	double value = 0;
	if (index>=0)
	{
		if (field==1)	// thickness
		{
			if (fEndDataThickness.dataHdl) value = INDEXH(fEndDataThickness.dataHdl,index);
			if (value==fFillValue) value = 0;
		}
		if (field==2)	// fraction
		{
			if (fEndDataFraction.dataHdl) value = INDEXH(fEndDataFraction.dataHdl,index);
			if (value==fFillValue) value = 0;
		}
	}
	return value;
}

/////////////////////////////////////////////////
double TimeGridWindIce_c::GetStartIceUVelocity(long index)
{	// 
	double u = 0;
	if (index>=0)
	{
		if (fStartDataIce.dataHdl) u = INDEXH(fStartDataIce.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double TimeGridWindIce_c::GetEndIceUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		if (fEndDataIce.dataHdl) u = INDEXH(fEndDataIce.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double TimeGridWindIce_c::GetStartIceVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fStartDataIce.dataHdl) v = INDEXH(fStartDataIce.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

double TimeGridWindIce_c::GetEndIceVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fEndDataIce.dataHdl) v = INDEXH(fEndDataIce.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

VelocityRec TimeGridWindIce_c::GetScaledPatValue(const Seconds& model_time, WorldPoint3D refPoint)
{
	double frac_coverage = 0, max_coverage = .8, min_coverage = .2, fracAlpha;
	VelocityRec scaledPatVelocity = {0.,0.}, iceVelocity = {0.,0.}, currentVelocity = {0.,0.};

	frac_coverage = GetDataField(model_time, refPoint, 2);
	//iceVelocity = GetScaledPatValueIce(model_time, refPoint);
	currentVelocity = TimeGridWindCurv_c::GetScaledPatValue(model_time, refPoint);
	
	if (frac_coverage >= max_coverage)
	{
		return iceVelocity;	// return zero
	}
	else if (frac_coverage <= min_coverage)
	{
		return currentVelocity;
	}
	else
	{
		fracAlpha = (.8 - frac_coverage)/(double)(max_coverage - min_coverage);
		//scaledPatVelocity.u = fracAlpha*currentVelocity.u + (1 - fracAlpha)*iceVelocity.u;
		//scaledPatVelocity.v = fracAlpha*currentVelocity.v + (1 - fracAlpha)*iceVelocity.v;
		scaledPatVelocity.u = fracAlpha*currentVelocity.u;	// scale to interpolate between the two cases
		scaledPatVelocity.v = fracAlpha*currentVelocity.v;
	}
	return scaledPatVelocity;
}

VelocityRec TimeGridWindIce_c::GetScaledPatValueIce(const Seconds& model_time, WorldPoint3D refPoint)
{	
	double timeAlpha, depthAlpha, depth = refPoint.z;
	//float topDepth, bottomDepth;
	long index = -1; 
	//long depthIndex1, depthIndex2; 
	//float totalDepth; 
	Seconds startTime,endTime;
	VelocityRec scaledPatVelocity = {0.,0.};
	//InterpolationVal interpolationVal;
	OSErr err = 0;
	
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	
	/*if (fGrid) 
	{
		if (bVelocitiesOnNodes)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			interpolationVal = fGrid -> GetInterpolationValues(refPoint.p);
			if (interpolationVal.ptIndex1<0) return scaledPatVelocity;
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else // for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
			index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}*/
	if (index < 0) return scaledPatVelocity;
	
	//totalDepth = GetTotalDepth(refPoint.p,index);	// may want to know depth relative to ice thickness...
	
	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			//scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			//scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
			scaledPatVelocity.u = INDEXH(fStartDataIce.dataHdl,index).u;
			scaledPatVelocity.v = INDEXH(fStartDataIce.dataHdl,index).v;
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
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - model_time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			//scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;			{
			scaledPatVelocity.u = timeAlpha*INDEXH(fStartDataIce.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndDataIce.dataHdl,index).u;
			scaledPatVelocity.v = timeAlpha*INDEXH(fStartDataIce.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndDataIce.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			scaledPatVelocity.u = 0.;
			scaledPatVelocity.v = 0.;
		}
	}
	
scale:
	
	//scaledPatVelocity.u *= fVar.curScale; // is there a dialog scale factor?
	//scaledPatVelocity.v *= fVar.curScale; 
	scaledPatVelocity.u *= fVar.fileScaleFactor; // may want to allow some sort of scale factor, though should be in file
	scaledPatVelocity.v *= fVar.fileScaleFactor; 
			
	return scaledPatVelocity;
}

double TimeGridWindIce_c::GetDataField(const Seconds& model_time, WorldPoint3D refPoint, long field)
{	
	double timeAlpha, depthAlpha, depth = refPoint.z;
	long index = -1; 
	//float totalDepth; 
	Seconds startTime,endTime;
	double iceDataValue = 0.;
	InterpolationVal interpolationVal;
	OSErr err = 0;
	
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	
	/*if (fGrid) 
	{
		if (bVelocitiesOnNodes)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			interpolationVal = fGrid -> GetInterpolationValues(refPoint.p);
			if (interpolationVal.ptIndex1<0) return iceDataValue;
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else // for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
			index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}*/
	if (index < 0) return iceDataValue;
	
	//totalDepth = GetTotalDepth(refPoint.p,index);	// may want to know depth relative to ice thickness...
	
	// Check for constant current 
	if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && model_time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationInTime))
		//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated data at the point
		if (index >= 0) 
		{
			//scaledPatVelocity.u = INDEXH(fStartData.dataHdl,index).u;
			//scaledPatVelocity.v = INDEXH(fStartData.dataHdl,index).v;
			if (field == 1) iceDataValue = INDEXH(fStartDataThickness.dataHdl,index);
			if (field == 2) iceDataValue = INDEXH(fStartDataFraction.dataHdl,index);
		}
		else	// set value to zero
		{
			iceDataValue = 0.;
		}
	}
	else // time varying current 
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - model_time)/(double)(endTime - startTime);
		
		// Calculate the interpolated data at the point
		if (index >= 0) 
		{
			//scaledPatVelocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			//scaledPatVelocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;			
			if (field == 1) iceDataValue = timeAlpha*INDEXH(fStartDataThickness.dataHdl,index) + (1-timeAlpha)*INDEXH(fEndDataThickness.dataHdl,index);
			if (field == 2) iceDataValue = timeAlpha*INDEXH(fStartDataFraction.dataHdl,index) + (1-timeAlpha)*INDEXH(fEndDataFraction.dataHdl,index);
		}
		else	// set value to zero
		{
			iceDataValue = 0.;
		}
	}
	
scale:
	
	//scaledPatVelocity.u *= fVar.curScale; // is there a dialog scale factor?
	//scaledPatVelocity.v *= fVar.curScale; 
	//scaledPatVelocity.u *= fVar.fileScaleFactor; // may want to allow some sort of scale factor, though should be in file
	//scaledPatVelocity.v *= fVar.fileScaleFactor; 
			
	return iceDataValue;
}

OSErr TimeGridWindIce_c::ReadTimeDataIce(long index,VelocityFH *velocityH, char* errmsg) 
{
	OSErr err = 0;
	char path[256], outPath[256];

	long i, j;
	int status, ncid, numdims;
	int ice_ucmp_id, ice_vcmp_id, angle_id, mask_id, uv_ndims;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	long totalNumberOfVels = fNumRows * fNumCols;	// assume ice is just surface velocities
	size_t velunit_len;
	static size_t ice_index[] = {0, 0, 0}, angle_index[] = {0, 0};
	static size_t ice_count[3], angle_count[2];

	double scale_factor = 1., angle = 0., u_grid, v_grid;
	char *velUnits = 0;
	double *ice_uvals = 0, *ice_vvals = 0, fill_value = -1e+34, test_value = 8e+10;
	double /**landmask = 0,*/ velConversion = 1.;
	double *angle_vals = 0, debug_mask;

	VelocityFH velH = 0;
	Boolean bRotated = true;
	
	errmsg[0] = 0;
	
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	ice_index[0] = index;	// time 
	ice_count[0] = 1;	// take one at a time
	/*if (numdims>=4)	// should check what the dimensions are
	{
		ice_count[1] = 1;	// depth
		//ice_count[1] = numDepths;	// depth
		ice_count[2] = latlength;
		ice_count[3] = lonlength;
	}
	else*/
	{
		ice_count[1] = latlength;	
		ice_count[2] = lonlength;
	}
	angle_count[0] = latlength;
	angle_count[1] = lonlength;
	
	status = nc_inq_varid(ncid, "ang", &angle_id);
	if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
	else
	{
		angle_vals = new double[latlength*lonlength]; 
		if(!angle_vals) {TechError("TimeGridWindIce_c::ReadTimeData()", "new[ ]", 0); err = memFullErr; goto done;}
		status = nc_get_vara_double(ncid, angle_id, angle_index, angle_count, angle_vals);
		if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
	}
	ice_uvals = new double[latlength*lonlength]; 
	if(!ice_uvals) 
	{
		TechError("TimeGridWindIce_c::ReadTimeData()", "new[]", 0); 
		err = memFullErr; 
		goto done;
	}
	ice_vvals = new double[latlength*lonlength]; 
	if(!ice_vvals) 
	{
		TechError("TimeGridWindIce_c::ReadTimeData()", "new[]", 0); 
		err = memFullErr; 
		goto done;
	}

	status = nc_inq_varid(ncid, "ice_u", &ice_ucmp_id);
	if (status != NC_NOERR)
	{
		status = nc_inq_varid(ncid, "ICE_U", &ice_ucmp_id);
		if (status != NC_NOERR)
		{
			err = -1; goto done;
		}
	}
	status = nc_inq_varid(ncid, "ice_v", &ice_vcmp_id);
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "ICE_V", &ice_vcmp_id);
		if (status != NC_NOERR) 
		{
			err = -1; goto done;
		}
	}
	//status = nc_inq_varndims(ncid, ice_ucmp_id, &uv_ndims);
	//if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {ice_count[1] = latlength; ice_count[2] = lonlength;}}	// could have more dimensions than are used in u,v

	status = nc_get_vara_double(ncid, ice_ucmp_id, ice_index, ice_count, ice_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_get_vara_double(ncid, ice_vcmp_id, ice_index, ice_count, ice_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_attlen(ncid, ice_ucmp_id, "units", &velunit_len);
	if (status == NC_NOERR)
	{
		velUnits = new char[velunit_len+1];
		status = nc_get_att_text(ncid, ice_ucmp_id, "units", velUnits);
		if (status == NC_NOERR)
		{
			velUnits[velunit_len] = '\0';
			if (!strcmpnocase(velUnits,"cm/s"))
				velConversion = .01;
			else if (!strcmpnocase(velUnits,"m/s"))
				velConversion = 1.0;
		}
	}
	
	
	status = nc_get_att_double(ncid, ice_ucmp_id, "_FillValue", &fill_value);
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, ice_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, ice_ucmp_id, "FillValue", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, ice_ucmp_id, "missing_value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
	//if (status != NC_NOERR) {err = -1; goto done;}	// don't require
	status = nc_get_att_double(ncid, ice_ucmp_id, "scale_factor", &scale_factor);

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// NOTE: if allow fill_value as NaN need to be sure to check for it wherever fill_value is used
	if (isnan(fill_value))
		fill_value = -9999.;
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) 
	{
		err = memFullErr; 
		goto done;
	}
	//for (i=0;i<totalNumberOfVels;i++)
	for (i=0;i<latlength;i++)
	{
		for (j=0;j<lonlength;j++)
		{
			if (ice_uvals[(latlength-i-1)*lonlength+j]==fill_value || ice_vvals[(latlength-i-1)*lonlength+j]==fill_value)
				ice_uvals[(latlength-i-1)*lonlength+j] = ice_vvals[(latlength-i-1)*lonlength+j] = 0;
			// NOTE: if leave velocity as NaN need to be sure to check for it wherever velocity is used (GetMove,Draw,...)
			if (isnan(ice_uvals[(latlength-i-1)*lonlength+j]) || isnan(ice_vvals[(latlength-i-1)*lonlength+j]))
				ice_uvals[(latlength-i-1)*lonlength+j] = ice_vvals[(latlength-i-1)*lonlength+j] = 0;
			/////////////////////////////////////////////////
			/*if (bRotated)
			{
				u_grid = (double)ice_uvals[(latlength-i-1)*lonlength+j] * velConversion;
				v_grid = (double)ice_vvals[(latlength-i-1)*lonlength+j] * velConversion;
				if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
				INDEXH(velH,i*lonlength+j).u = u_grid*cos(angle)-v_grid*sin(angle);	//in radians
				INDEXH(velH,i*lonlength+j).v = u_grid*sin(angle)+v_grid*cos(angle);
			}
			else*/
			{
				INDEXH(velH,i*lonlength+j).u = ice_uvals[(latlength-i-1)*lonlength+j] * velConversion;	// need units
				INDEXH(velH,i*lonlength+j).v = ice_vvals[(latlength-i-1)*lonlength+j] * velConversion;
			}
		}
	}
	*velocityH = velH;
	fFillValue = fill_value * velConversion;
	
	//if (scale_factor!=1.) fVar.curScale = scale_factor;	// hmm, this forces a reset of scale factor each time, overriding any set by hand
	if (scale_factor!=1.) fVar.fileScaleFactor = scale_factor;
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading ice data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		//printNote("Error opening NetCDF file");
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (ice_uvals) 
	{
		delete [] ice_uvals; 
		ice_uvals = 0;
	}
	if (ice_vvals) 
	{
		delete [] ice_vvals; 
		ice_vvals = 0;
	}
	
	//if (landmask) {delete [] landmask; landmask = 0;}
	if (angle_vals) {delete [] angle_vals; angle_vals = 0;}
	if (velUnits) {delete [] velUnits;}
	return err;
}

OSErr TimeGridWindIce_c::ReadTimeDataFields(long index, DOUBLEH *thicknessH, DOUBLEH *fractionH, char* errmsg) 
{
	OSErr err = 0;
	char path[256], outPath[256];

	long i, j;
	int status, ncid, numdims;
	int data_fraction_id, data_thickness_id, mask_id, uv_ndims;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	long totalNumberOfValues = fNumRows * fNumCols;	// assume ice is just surface values
	size_t unit_len;
	static size_t data_index[] = {0, 0, 0, 0};
	static size_t data_count[4];

	double scale_factor1 = 1., scale_factor2 = 1., u_grid, v_grid;
	char *units = 0;
	double *data_thickness = 0, *data_fraction = 0, fill_value = -1e+34, test_value = 8e+10;
	double /**landmask = 0,*/ unitConversion = 1.;
	double debug_mask;

	DOUBLEH thickH = 0, fracH = 0;
	
	errmsg[0] = 0;
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	data_index[0] = index;	// time 
	data_count[0] = 1;	// take one at a time
	/*if (numdims>=4)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		curr_count[1] = numDepths;	// depth
		curr_count[2] = latlength;
		curr_count[3] = lonlength;
	}
	else*/
	{
		data_count[1] = latlength;	
		data_count[2] = lonlength;
	}
	
	data_thickness = new double[latlength*lonlength]; 
	if(!data_thickness) 
	{
		TechError("TimeGridVelIce_c::ReadTimeData()", "new[]", 0); 
		err = memFullErr; 
		goto done;
	}
	data_fraction = new double[latlength*lonlength]; 
	if(!data_fraction) 
	{
		TechError("TimeGridVelIce_c::ReadTimeData()", "new[]", 0); 
		err = memFullErr; 
		goto done;
	}

	status = nc_inq_varid(ncid, "ice_thickness", &data_thickness_id);
	if (status != NC_NOERR)
	{
		status = nc_inq_varid(ncid, "ICE_U", &data_thickness_id);
		if (status != NC_NOERR)
		{
			err = -1; goto done;
		}
	}
	status = nc_inq_varid(ncid, "ice_fraction", &data_fraction_id);
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "ICE_V", &data_fraction_id);
		if (status != NC_NOERR) 
		{
			err = -1; goto done;
		}
	}

	status = nc_inq_varndims(ncid, data_thickness_id, &uv_ndims);
	if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {data_count[1] = latlength; data_count[2] = lonlength;}}	// could have more dimensions than are used in u,v

	status = nc_get_vara_double(ncid, data_thickness_id, data_index, data_count, data_thickness);
	if (status != NC_NOERR) {err = -1; goto done;}

	status = nc_get_vara_double(ncid, data_fraction_id, data_index, data_count, data_fraction);
	if (status != NC_NOERR) {err = -1; goto done;}

	// what are unit options?
	status = nc_inq_attlen(ncid, data_thickness_id, "units", &unit_len);
	if (status == NC_NOERR)
	{
		units = new char[unit_len+1];
		status = nc_get_att_text(ncid, data_thickness_id, "units", units);
		if (status == NC_NOERR)
		{
			units[unit_len] = '\0';
			if (!strcmpnocase(units,"cm"))
				unitConversion = .01;
			else if (!strcmpnocase(units,"m"))
				unitConversion = 1.0;
		}
	}
	
	
	status = nc_get_att_double(ncid, data_thickness_id, "_FillValue", &fill_value);
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, data_thickness_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, data_thickness_id, "FillValue", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
	if (status != NC_NOERR) {status = nc_get_att_double(ncid, data_thickness_id, "missing_value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
	//if (status != NC_NOERR) {err = -1; goto done;}	// don't require
	status = nc_get_att_double(ncid, data_thickness_id, "scale_factor", &scale_factor1);
	status = nc_get_att_double(ncid, data_fraction_id, "scale_factor", &scale_factor2);

	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	// NOTE: if allow fill_value as NaN need to be sure to check for it wherever fill_value is used
	if (isnan(fill_value))
		fill_value = -9999.;
	
	fracH = (DOUBLEH)_NewHandleClear(totalNumberOfValues * sizeof(double));
	if (!fracH) 
	{
		err = memFullErr; 
		goto done;
	}
	thickH = (DOUBLEH)_NewHandleClear(totalNumberOfValues * sizeof(double));
	if (!thickH) 
	{
		err = memFullErr; 
		goto done;
	}
	//for (i=0;i<totalNumberOfVels;i++)
	for (i=0;i<latlength;i++)
	{
		for (j=0;j<lonlength;j++)
		{
			if (data_thickness[(latlength-i-1)*lonlength+j]==fill_value || data_fraction[(latlength-i-1)*lonlength+j]==fill_value)
				data_thickness[(latlength-i-1)*lonlength+j] = data_fraction[(latlength-i-1)*lonlength+j] = 0;
			// NOTE: if leave velocity as NaN need to be sure to check for it wherever velocity is used (GetMove,Draw,...)
			if (isnan(data_thickness[(latlength-i-1)*lonlength+j]) || isnan(data_fraction[(latlength-i-1)*lonlength+j]))
				data_thickness[(latlength-i-1)*lonlength+j] = data_fraction[(latlength-i-1)*lonlength+j] = 0;
			/////////////////////////////////////////////////
			INDEXH(thickH,i*lonlength+j) = data_thickness[(latlength-i-1)*lonlength+j] * scale_factor1 * unitConversion;	// need units
			INDEXH(fracH,i*lonlength+j) = data_fraction[(latlength-i-1)*lonlength+j] * scale_factor2 * unitConversion;
		}
	}
	*fractionH = fracH;
	*thicknessH = thickH;
	//fFillValue = fill_value;	// do we want to store fill_values / scale_factors ?
	
	// code goes here, will need separate scale factors for each (or just apply it on read?)
	//if (scale_factor!=1.) fVar.fileScaleFactor = scale_factor;
	
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading ice data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		//printNote("Error opening NetCDF file");
		if(fracH) {DisposeHandle((Handle)fracH); fracH = 0;}
		if(thickH) {DisposeHandle((Handle)thickH); thickH = 0;}
	}
	if (data_thickness) 
	{
		delete [] data_thickness; 
		data_thickness = 0;
	}
	if (data_fraction) 
	{
		delete [] data_fraction; 
		data_fraction = 0;
	}
	
	//if (landmask) {delete [] landmask; landmask = 0;}
	if (units) {delete [] units;}
	return err;
}

OSErr TimeGridWindIce_c::GetIceFields(Seconds time, double *thickness, double *fraction)
{	// use for curvilinear
	double timeAlpha;
	Seconds startTime,endTime;
	OSErr err = 0;
	
	long numVertices,i,numTri,numCells,index=-1,triIndex;
	InterpolationVal interpolationVal;
	LongPointHdl ptsHdl = 0;
	TopologyHdl topH ;
	long timeDataInterval;
	Boolean loaded;
	//TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	TTriGridVel* triGrid = (dynamic_cast<TTriGridVel*>(fGrid));

	char errmsg[256];
	errmsg[0] = 0;

	err = this -> SetInterval(errmsg, time);
	if(err) return err;
	loaded = this -> CheckInterval(timeDataInterval, time);	 
	
	if(!loaded) return -1;
	
	//topH = triGrid -> GetTopologyHdl();
	topH = fGrid -> GetTopologyHdl();
	if(topH)
		numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	else 
		numTri = 0;
		
	/*ptsHdl = triGrid -> GetPointsHdl();
	if(ptsHdl)
		numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
	else 
		numVertices = 0;*/
	
	// Check for time varying current 
	if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;

		if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
		{
			timeAlpha = 1;
		}
		else
		{	
			endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
		}
	}
	
	numCells = numTri / 2;
	//for (i = 0 ; i< numTri; i++)
	for (i = 0 ; i< numCells; i++)
	{
		triIndex = i*2;
		/*if (bVelocitiesOnNodes)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			//interpolationVal = triGrid -> GetInterpolationValues(refPoint.p);
			//interpolationVal = triGrid -> GetInterpolationValuesFromIndex(i);
			interpolationVal = triGrid -> GetInterpolationValuesFromIndex(triIndex);
			if (interpolationVal.ptIndex1<0) {thickness[i] = 0;	fraction[i] = 0;}// should this be an error?
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else*/ // for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
			//index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
			//index = triGrid->GetRectIndexFromTriIndex2(i,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
			index = triGrid->GetRectIndexFromTriIndex2(triIndex,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid

		if (index < 0) {thickness[i] = 0;	fraction[i] = 0;}// should this be an error?
		
		// Should check vs fFillValue
		// Check for constant current 
		if(((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha == 1) && index!=-1)
		{
				thickness[i] = GetStartFieldValue(index,1);
				fraction[i] = GetStartFieldValue(index,2);
		}
		else if (index!=-1)// time varying current
		{
			thickness[i] = timeAlpha*GetStartFieldValue(index,1) + (1-timeAlpha)*GetEndFieldValue(index,1);
			fraction[i] = timeAlpha*GetStartFieldValue(index,2) + (1-timeAlpha)*GetEndFieldValue(index,2);
		}
		if (thickness[i] == fFillValue) thickness[i] = 0.;
		if (fraction[i] == fFillValue) fraction[i] = 0.;
		//thickness[i] *= fVar.fileScaleFactor;	// doing this on read
		//fraction[i] *= fVar.fileScaleFactor; // doing this on read
	}
	return err;
}

//OSErr TimeGridVelIce_c::GetIceVelocities(Seconds time, double *u, double *v)
OSErr TimeGridWindIce_c::GetIceVelocities(Seconds time, VelocityFRec *ice_velocity)
{	// use for curvilinear
	double timeAlpha;
	Seconds startTime,endTime;
	OSErr err = 0;

	char errmsg[256];
	errmsg[0] = 0;
	
	long numVertices,i,numTri,index=-1;
	InterpolationVal interpolationVal;
	LongPointHdl ptsHdl = 0;
	TopologyHdl topH ;
	long timeDataInterval;
	Boolean loaded;
	//TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	TTriGridVel* triGrid = (dynamic_cast<TTriGridVel*>(fGrid));
	VelocityFRec velocity;
	
	err = this -> SetInterval(errmsg, time);
	if(err) return err;
	
	loaded = this -> CheckInterval(timeDataInterval, time);	 
	
	if(!loaded) return -1;
	
	//topH = triGrid -> GetTopologyHdl();
	topH = fGrid -> GetTopologyHdl();
	if(topH)
		numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	else 
		numTri = 0;
		
	/*ptsHdl = triGrid -> GetPointsHdl();
	if(ptsHdl)
		numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
	else 
		numVertices = 0;*/
	
	// Check for time varying current 
	if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;

		if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
		{
			timeAlpha = 1;
		}
		else
		{	
			endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
		}
	}
	
	for (i = 0 ; i< numTri; i++)
	{
		/*if (bVelocitiesOnNodes)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			//interpolationVal = triGrid -> GetInterpolationValues(refPoint.p);
			interpolationVal = triGrid -> GetInterpolationValuesFromIndex(i);
			if (interpolationVal.ptIndex1<0) {ice_velocity[i].u = 0;	ice_velocity[i].v = 0;}// should this be an error?
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else*/ // for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
			//index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
			index = triGrid->GetRectIndexFromTriIndex2(i,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid

		if (index < 0) {ice_velocity[i].u = 0;	ice_velocity[i].v = 0;}// should this be an error?
		
		// Should check vs fFillValue
		// Check for constant current 
		if(((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha == 1) && index!=-1)
		{
				velocity.u = GetStartIceUVelocity(index);
				velocity.v = GetStartIceVVelocity(index);
		}
		else if (index!=-1)// time varying current
		{
			velocity.u = timeAlpha*GetStartIceUVelocity(index) + (1-timeAlpha)*GetEndIceUVelocity(index);
			velocity.v = timeAlpha*GetStartIceVVelocity(index) + (1-timeAlpha)*GetEndIceVVelocity(index);
		}
		if (velocity.u == fFillValue) velocity.u = 0.;
		if (velocity.v == fFillValue) velocity.v = 0.;
		/*if ((velocity.u != 0 || velocity.v != 0) && (velocity.u != fFillValue && velocity.v != fFillValue)) // should already have handled fill value issue
		{
			// code goes here, fill up arrays with data
			float inchesX = (velocity.u * refScale * fVar.fileScaleFactor) / arrowScale;
			float inchesY = (velocity.v * refScale * fVar.fileScaleFactor) / arrowScale;
		}*/
		//u[i] = velocity.u * fVar.fileScaleFactor;
		//v[i] = velocity.v * fVar.fileScaleFactor;
		ice_velocity[i].u = velocity.u * fVar.fileScaleFactor;
		ice_velocity[i].v = velocity.v * fVar.fileScaleFactor;
	}
	return err;
}

OSErr TimeGridWindIce_c::GetMovementVelocities(Seconds time, VelocityFRec *movement_velocity)
{	// use for curvilinear
	OSErr err = 0;
	double timeAlpha;
	double frac_coverage = 0, max_coverage = .8, min_coverage = .2, fracAlpha;
	Seconds startTime,endTime;
	
	long numVertices,i,numTri,index=-1;
	InterpolationVal interpolationVal;
	LongPointHdl ptsHdl = 0;
	TopologyHdl topH ;
	long timeDataInterval;
	Boolean loaded;
	//TTriGridVel* triGrid = (TTriGridVel*)fGrid;
	TTriGridVel* triGrid = (dynamic_cast<TTriGridVel*>(fGrid));
	VelocityRec velocity = {0.,0.}, iceVelocity = {0.,0.}, currentVelocity = {0.,0.};
	VelocityRec iceVelocityStart = {0.,0.}, currentVelocityStart = {0.,0.}, iceVelocityEnd = {0.,0.}, currentVelocityEnd = {0.,0.};
	
	char errmsg[256];
	errmsg[0] = 0;

	err = this -> SetInterval(errmsg, time);
	if(err) return err;
	
	loaded = this -> CheckInterval(timeDataInterval, time);	 
	
	if(!loaded) return -1;
	
	//topH = triGrid -> GetTopologyHdl();
	topH = fGrid -> GetTopologyHdl();
	if(topH)
		numTri = _GetHandleSize((Handle)topH)/sizeof(**topH);
	else 
		numTri = 0;
		
	/*ptsHdl = triGrid -> GetPointsHdl();
	if(ptsHdl)
		numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
	else 
		numVertices = 0;*/
	
	// Check for time varying current 
	if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
	{
		// Calculate the time weight factor
		if (GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;

		if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationInTime)
		{
			timeAlpha = 1;
		}
		else
		{	
			endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
		}
	}
	
	for (i = 0 ; i< numTri; i++)
	{
		/*if (bVelocitiesOnNodes)
		{
			//index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols);// curvilinear grid
			//interpolationVal = triGrid -> GetInterpolationValues(refPoint.p);
			interpolationVal = triGrid -> GetInterpolationValuesFromIndex(i);
			if (interpolationVal.ptIndex1<0) {movement_velocity[i].u = 0;	movement_velocity[i].v = 0; return err;}// should this be an error?
			//ptIndex1 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];	
			//ptIndex2 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex2];
			//ptIndex3 =  (*fVerdatToNetCDFH)[interpolationVal.ptIndex3];
			index = (*fVerdatToNetCDFH)[interpolationVal.ptIndex1];
		}
		else */// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
			//index = (dynamic_cast<TTriGridVel*>(fGrid))->GetRectIndexFromTriIndex(refPoint.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
			index = triGrid->GetRectIndexFromTriIndex2(i,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid

		if (index < 0) {movement_velocity[i].u = 0;	movement_velocity[i].v = 0; return err;}// should this be an error?
		
		// Should check vs fFillValue
		// Check for constant current 
		frac_coverage = GetStartFieldValue(index,2);
		fracAlpha = (.8 - frac_coverage)/(double)(max_coverage - min_coverage);
		iceVelocityStart.u = GetStartIceUVelocity(index);
		iceVelocityStart.v = GetStartIceVVelocity(index);
		currentVelocityStart.u = GetStartUVelocity(index);
		currentVelocityStart.v = GetStartVVelocity(index);
		if(((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha == 1) && index!=-1)
		{
			if (frac_coverage >= max_coverage)
			{
				velocity.u = iceVelocityStart.u;
				velocity.v = iceVelocityStart.v;
			}
			else if (frac_coverage <= min_coverage)
			{
				velocity.u = currentVelocityStart.u;
				velocity.v = currentVelocityStart.v;
			}
			else
			{
				velocity.u = fracAlpha*currentVelocityStart.u + (1 - fracAlpha)*iceVelocityStart.u;
				velocity.v = fracAlpha*currentVelocityStart.v + (1 - fracAlpha)*iceVelocityStart.v;
			}
		}
		else if (index!=-1)// time varying current
		{
			iceVelocityEnd.u = GetEndIceUVelocity(index);
			iceVelocityEnd.v = GetEndIceVVelocity(index);
			currentVelocityEnd.u = GetEndUVelocity(index);
			currentVelocityEnd.v = GetEndVVelocity(index);
			if (frac_coverage >= max_coverage)
			{
				velocity.u = timeAlpha*iceVelocityStart.u + (1-timeAlpha)*iceVelocityEnd.u;
				velocity.v = timeAlpha*iceVelocityStart.v + (1-timeAlpha)*iceVelocityEnd.v;
			}
			else if (frac_coverage <= min_coverage)
			{
				velocity.u = timeAlpha*currentVelocityStart.u + (1-timeAlpha)*currentVelocityEnd.u;
				velocity.v = timeAlpha*currentVelocityStart.v + (1-timeAlpha)*currentVelocityEnd.v;
			}
			else
			{
				currentVelocity.u = timeAlpha*currentVelocityStart.u + (1-timeAlpha)*currentVelocityEnd.u;
				currentVelocity.v = timeAlpha*currentVelocityStart.v + (1-timeAlpha)*currentVelocityEnd.v;
				iceVelocity.u = timeAlpha*iceVelocityStart.u + (1-timeAlpha)*iceVelocityEnd.u;
				iceVelocity.v = timeAlpha*iceVelocityStart.v + (1-timeAlpha)*iceVelocityEnd.v;
				velocity.u = fracAlpha*currentVelocity.u + (1 - fracAlpha)*iceVelocity.u;
				velocity.v = fracAlpha*currentVelocity.v + (1 - fracAlpha)*iceVelocity.v;
			}
		}

		movement_velocity[i].u = velocity.u * fVar.fileScaleFactor;
		movement_velocity[i].v = velocity.v * fVar.fileScaleFactor;
	}
	return err;
}

