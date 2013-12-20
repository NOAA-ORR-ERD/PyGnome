/*
 *  NetCDFMoverCurv.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "CROSS.H"
#include "NetCDFMoverCurv.h"
#include "MemUtils.h"
#include "netcdf.h"
#include "DagTreeIO.h"

NetCDFMoverCurv::NetCDFMoverCurv (TMap *owner, char *name) : NetCDFMover(owner, name)
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
	bIsCOOPSWaterMask = false;
}	

void NetCDFMoverCurv::Dispose ()
{
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	
	NetCDFMover::Dispose ();
}


Boolean NetCDFMoverCurv::IAmA3DMover()
{
	if (fVar.gridType != TWO_D) return true;
	return false;
}
#define NetCDFMoverCurvREADWRITEVERSION 1 //JLM

OSErr NetCDFMoverCurv::Write (BFPB *bfpb)
{
	long i, version = NetCDFMoverCurvREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints = 0, numPts = 0, index;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMoverCurv::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	if (fVerdatToNetCDFH) numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}
	
	if (fVertexPtsH) numPts = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPts)) goto done;
	for (i=0;i<numPts;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}
	
done:
	if(err)
		TechError("NetCDFMoverCurv::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr NetCDFMoverCurv::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMoverCurv::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFMoverCurv::Read()", "id != TYPE_NETCDFMOVERCURV", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != NetCDFMoverCurvREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	// for curvilinear
	if(!fVerdatToNetCDFH)
	{TechError("NetCDFMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	// for curvilinear
	if(!fVertexPtsH)
	{TechError("NetCDFMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}
	
done:
	if(err)
	{
		TechError("NetCDFMoverCurv::Read(char* path)", " ", 0); 
		if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
		if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFMoverCurv::CheckAndPassOnMessage(TModelMessage *message)
{
	return NetCDFMover::CheckAndPassOnMessage(message); 
}

OSErr NetCDFMoverCurv::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// this code is for curvilinear grids
	OSErr err = 0;
	long i,j,k, numScanned, indexOfStart = 0;
	int status, ncid, latIndexid, lonIndexid, latid, lonid, recid, timeid, sigmaid, sigmavarid, sigmavarid2, hcvarid, depthid, depthdimid, depthvarid, mask_id, numdims;
	size_t latLength, lonLength, recs, t_len, t_len2, sigmaLength=0;
	float startLat,startLon,endLat,endLon,hc_param=0.;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	char dimname[NC_MAX_NAME], s[256], topPath[256], outPath[256];
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0, sigmaLevelsH=0;
	float yearShift=0.;
	//float *lat_vals=0,*lon_vals=0,timeVal;
	double *lat_vals=0,*lon_vals=0,timeVal;
	float *depth_vals=0,*sigma_vals=0,*sigma_vals2=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex[2]={0,0},sigmaIndex=0;
	static size_t pt_count[2], sigma_count;
	Seconds startTime, startTime2;
	double timeConversion = 1., scale_factor = 1.;
	char errmsg[256] = "";
	char fileName[64],*modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, isLandMask = true, isCoopsMask = false;
	//VelocityFH velocityH = 0;
	static size_t mask_index[] = {0,0};
	static size_t mask_count[2];
	double *landmask = 0; 
	DOUBLEH landmaskH=0;
	//long numTimesInFile = 0;
	
	if (!path || !path[0]) return 0;
	strcpy(fVar.pathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fVar.userName, fileName); // maybe use a name from the file
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
	// check number of dimensions - 2D or 3D
	status = nc_inq_ndims(ncid, &numdims);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_attlen(ncid,NC_GLOBAL,"generating_model",&t_len2);
	if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; /*goto done;*/}}	// will need to split for Navy vs LAS
	else 
	{
		fIsNavy = true;
		// may only need to see keyword is there, since already checked grid type
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {status = nc_inq_attlen(ncid,NC_GLOBAL,"generator",&t_len2); if (status != NC_NOERR) {fIsNavy = false; goto done;}}	// will need to split for regridded or non-Navy cases 
		modelTypeStr[t_len2] = '\0';
		
		strcpy(fVar.userName, modelTypeStr); // maybe use a name from the file
		/*
		 if (!strncmp (modelTypeStr, "SWAFS", 5))
		 fIsNavy = true;
		 else if (!strncmp (modelTypeStr, "fictitious test data", strlen("fictitious test data")))
		 fIsNavy = true;
		 else
		 fIsNavy = false;*/
	}
	
	/*if (fIsNavy)
	 {
	 status = nc_inq_dimid(ncid, "time", &recid); //Navy
	 if (status != NC_NOERR) {err = -1; goto done;}
	 }
	 else
	 {
	 status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
	 if (status != NC_NOERR) {err = -1; goto done;}
	 }*/
	
	status = nc_inq_dimid(ncid, "time", &recid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR || recid==-1) {err = -1; goto done;}
	}
	
	//if (fIsNavy)
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {status = nc_inq_varid(ncid, "TIME", &timeid);if (status != NC_NOERR) {err = -1; goto done;} /*timeid = recid;*/} 	// for Ferret files, everything is in CAPS
	//if (status != NC_NOERR) {/*err = -1; goto done;*/ timeid = recid;} 	// for LAS files, variable names unstable
	
	//if (!fIsNavy)
	//status = nc_inq_attlen(ncid, recid, "units", &t_len);	// recid is the dimension id not the variable id
	//else	// LAS has them in order, and time is unlimited, but variable/dimension names keep changing so leave this way for now
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		//err = -1; goto done;
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
		if (status != NC_NOERR) {err = -1; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
						  unitStr, junk, &time.year, &time.month, &time.day,
						  &time.hour, &time.minute, &time.second) ;
		if (numScanned==5 || numScanned==4)	
		{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned==7)	time.second = 0;
		else if (numScanned<8)	
		//if (numScanned!=8)	
		{ 
			//timeUnits = 0;	// files should always have this info
			//timeConversion = 3600.;		// default is hours
			//startTime2 = model->GetStartTime();	// default to model start time
			err = -1; TechError("NetCDFMoverCurv::TextRead()", "sscanf() == 8", 0); goto done;
		}
		//else
		{
			// code goes here, trouble with the DAYS since 1900 format, since converts to seconds since 1904
			if (time.year ==1900) {time.year += 40; time.day += 1; /*for the 1900 non-leap yr issue*/ yearShift = 40.;}
			DateToSeconds (&time, &startTime2);	// code goes here, which start Time to use ??
			if (!strcmpnocase(unitStr,"HOURS") || !strcmpnocase(unitStr,"HOUR"))
				timeConversion = 3600.;
			else if (!strcmpnocase(unitStr,"MINUTES") || !strcmpnocase(unitStr,"MINUTE"))
				timeConversion = 60.;
			else if (!strcmpnocase(unitStr,"SECONDS") || !strcmpnocase(unitStr,"SECOND"))
				timeConversion = 1.;
			else if (!strcmpnocase(unitStr,"DAYS") || !strcmpnocase(unitStr,"DAY"))
				timeConversion = 24.*3600.;
		}
	} 
	
	if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "gridy", &latIndexid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimid(ncid, "gridx", &lonIndexid);	//Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		// option to use index values?
		status = nc_inq_varid(ncid, "grid_lat", &latid);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "grid_lon", &lonid);
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	else
	{
		for (i=0;i<numdims;i++)
		{
			if (i == recid) continue;
			status = nc_inq_dimname(ncid,i,dimname);
			if (status != NC_NOERR) {err = -1; goto done;}
			//if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3))
			if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3) || !strncmpnocase(dimname,"NX",2))
			{
				lonIndexid = i;
			}
			//if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3))
			if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3) || !strncmpnocase(dimname,"NY",2))
			{
				latIndexid = i;
			}
		}
		
		status = nc_inq_dimlen(ncid, latIndexid, &latLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_inq_dimlen(ncid, lonIndexid, &lonLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_inq_varid(ncid, "LATITUDE", &latid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lat", &latid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_varid(ncid, "LONGITUDE", &lonid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lon", &lonid);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(latLength*lonLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	//lat_vals = new float[latLength*lonLength]; 
	lat_vals = new double[latLength*lonLength]; 
	//lon_vals = new float[latLength*lonLength]; 
	lon_vals = new double[latLength*lonLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	//status = nc_get_vara_float(ncid, latid, ptIndex, pt_count, lat_vals);
	status = nc_get_vara_double(ncid, latid, ptIndex, pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_vara_float(ncid, lonid, ptIndex, pt_count, lon_vals);
	status = nc_get_vara_double(ncid, lonid, ptIndex, pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
			//lat_vals[(latLength-i-1)*lonLength+j]=0.;
			//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
			//lon_vals[(latLength-i-1)*lonLength+j]=0.;
			// grid ordering does matter for creating ptcurmap, assume increases fastest in x/lon, then in y/lat
			INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(latLength-i-1)*lonLength+j];	
			INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(latLength-i-1)*lonLength+j];
			//INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(i)*lonLength+j];	
			//INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(i)*lonLength+j];
		}
	}
	fVertexPtsH	 = vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	//latIndex = 0;
	//lonIndex = 0;
	//status = nc_get_var1_float(ncid, latIndexid, &latIndex, &startLat);	// this won't work for curvilinear case
	//status = nc_get_var1_float(ncid, lonIndexid, &lonIndex, &startLon);
	//latIndex = latLength-1;
	//lonIndex = lonLength-1;
	//status = nc_get_var1_float(ncid, latIndexid, &latIndex, &endLat);	// this won't work for curvilinear case
	//status = nc_get_var1_float(ncid, lonIndexid, &lonIndex, &endLon);
	
	
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	//if (status != NC_NOERR || fIsNavy) {fVar.gridType = TWO_D; /*err = -1; goto done;*/}	// check for zgrid option here
	if (status != NC_NOERR)
	{
		status = nc_inq_dimid(ncid, "levels", &depthdimid); 
		//status = nc_inq_dimid(ncid, "depth", &depthdimid); 
		if (status != NC_NOERR || fIsNavy) 
		{
			fVar.gridType = TWO_D; /*err = -1; goto done;*/
		}	
		else
		{// check for zgrid option here
			fVar.gridType = MULTILAYER; /*err = -1; goto done;*/
			//status = nc_inq_varid(ncid, "depth", &sigmavarid); //Navy
			status = nc_inq_varid(ncid, "depth_levels", &sigmavarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, depthdimid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
	}
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "sc_r", &sigmavarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, "Cs_r", &sigmavarid2);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			fVar.gridType = SIGMA_ROMS;
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_vals2 = new float[sigmaLength];
			if (!sigma_vals2) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_vara_float(ncid, sigmavarid2, &sigmaIndex, &sigma_count, sigma_vals2);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varid(ncid, "hc", &hcvarid);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_var1_float(ncid, hcvarid, &sigmaIndex, &hc_param);
			if (status != NC_NOERR) {err = -1; goto done;}
			//{err = -1; goto done;}
		}
		else
		{
			// code goes here, for SIGMA_ROMS the variable isn't sigma but sc_r and Cs_r, with parameter hc
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			// check if sigmaLength > 1
			fVar.gridType = SIGMA;
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			//sigmaLevelsH = (FLOATH)_NewHandleClear(sigmaLength*sizeof(sigmaLevelsH));
			//if (!sigmaLevelsH) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &sigmaIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		// once depth is read in 
	}
	
	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR || fIsNavy) {fVar.gridType = TWO_D;/*err = -1; goto done;*/}
	else
	{	
		/*if (fVar.gridType==MULTILAYER)
		{
			// for now
			totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
			if (!totalDepthsH) {err = memFullErr; goto done;}
			depth_vals = new float[latLength*lonLength];
			if (!depth_vals) {err = memFullErr; goto done;}
			for (i=0;i<latLength*lonLength;i++)
			{
				INDEXH(totalDepthsH,i)=sigma_vals[sigmaLength-1];
				depth_vals[i] = INDEXH(totalDepthsH,i);
			}
		
		}
		else*/
		{
			totalDepthsH = (FLOATH)_NewHandleClear(latLength*lonLength*sizeof(float));
			if (!totalDepthsH) {err = memFullErr; goto done;}
			depth_vals = new float[latLength*lonLength];
			if (!depth_vals) {err = memFullErr; goto done;}
			status = nc_get_vara_float(ncid, depthid, ptIndex,pt_count, depth_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
		
			status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
			if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
		}
	}
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	if (recs <= 0) {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		//if (!fIsNavy)
		//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		//else
		//status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		// get rid of the seconds since they get garbled in the dialogs
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		INDEXH(fTimeHdl,i) = newTime-yearShift*3600.*24.*365.25;	// which start time where?
		if (i==0) startTime = newTime-yearShift*3600.*24.*365.25;
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion -yearShift*3600.*24.*365.25;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion -yearShift*3600.*24.*365.25 + fTimeShift;
	}
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
			//buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first\n time in the file?",FALSE);
			//if(!gCommandFileErrorLogPath[0])
			if(!gCommandFileRun)	// also may want to skip for location files...
				buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
			switch(buttonSelected){
				case 1: // reset model start time
					//bTopFile = true;
					model->SetModelTime(startTime);
					model->SetStartTime(startTime);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
					break;  
				case 3: // don't reset model start time
					//bTopFile = false;
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
		//model->SetModelTime(startTime);
		//model->SetStartTime(startTime);
		//model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
	}
	
	fNumRows = latLength;
	fNumCols = lonLength;
	
	/*status = nc_inq_varid(ncid, "mask", &mask_id);
	if (status != NC_NOERR)	{isLandMask = false;}
	
	status = nc_inq_varid(ncid, "coops_mask", &mask_id);
	if (status != NC_NOERR)	{isCoopsMask = false;}
	else {isCoopsMask = true; bIsCOOPSWaterMask = true;}
	*/

	mask_count[0] = latLength;
	mask_count[1] = lonLength;
	
	status = nc_inq_varid(ncid, "mask", &mask_id);
	if (status != NC_NOERR)	{isLandMask = false;}

	status = nc_inq_varid(ncid, "coops_mask", &mask_id);	// should only have one or the other
	if (status != NC_NOERR)	{isCoopsMask = false;}
	else {isCoopsMask = true; bIsCOOPSWaterMask = true;}
	
	if (isLandMask || isCoopsMask)
	{	// no need to bother with the handle here...
		// maybe should store the mask? we are using it in ReadTimeValues, do we need to?
		landmask = new double[latLength*lonLength]; 
		if(!landmask) {TechError("NetCDFMoverCurv::TextRead()", "new[]", 0); err = memFullErr; goto done;}
		//mylandmask = new double[latlength*lonlength]; 
		//if(!mylandmask) {TechError("NetCDFMoverCurv::ReoderPointsNoMask()", "new[]", 0); err = memFullErr; goto done;}
		landmaskH = (double**)_NewHandleClear(latLength*lonLength*sizeof(double));
		if(!landmaskH) {TechError("NetCDFMoverCurv::TextRead()", "_NewHandleClear()", 0); err = memFullErr; goto done;}
		status = nc_get_vara_double(ncid, mask_id, mask_index, mask_count, landmask);
		if (status != NC_NOERR) {err = -1; goto done;}

		for (i=0;i<latLength;i++)
		{
			for (j=0;j<lonLength;j++)
			{
				INDEXH(landmaskH,i*lonLength+j) = landmask[(latLength-i-1)*lonLength+j];
			}
		}
	}
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//err = this -> SetInterval(errmsg);
	//if(err) goto done;
	
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	//{if (topFilePath[0]) {strcpy(fTopFilePath,topFilePath); err = ReadTopology(fTopFilePath,newMap); goto done;}}
	{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto depths;}}
	//if (isLandMask/*fIsNavy*//*true*/)	// allow for the LAS files too ?
	if (!gCommandFileRun)
	//if (true)	// allow for the LAS files too ?
	{
		short buttonSelected;
		buttonSelected  = MULTICHOICEALERT(1688,"Do you have an extended topology file to load?",FALSE);
		switch(buttonSelected){
			case 1: // there is an extended top file
				bTopFile = true;
				break;  
			case 3: // no extended top file
				bTopFile = false;
				break;
			case 4: // cancel
				err=-1;// stay at this dialog
				goto done;
		}
	}
	if(bTopFile)
	{
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
					 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		//if (!reply.good) return USERCANCEL;
		if (!reply.good) /*return 0;*/
		{
			/*if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;*/
			//err = dynamic_cast<NetCDFMoverCurv *>(this)->ReorderPoints(velocityH,newMap,errmsg);	
			if (isLandMask) err = dynamic_cast<NetCDFMoverCurv *>(this)->ReorderPoints(landmaskH,newMap,errmsg);	
			else if (isCoopsMask) err = ReorderPointsCOOPSMask(landmaskH,newMap,errmsg);
			else err = ReorderPointsNoMask(newMap,errmsg);
			//else err = ReorderPointsNoMask(velocityH,newMap,errmsg);
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}
		else
			strcpy(topPath, reply.fullPath);
		
#else
		where = CenteredDialogUpLeft(M38c);
		sfpgetfile(&where, "",
				   (FileFilterUPP)0,
				   -1, typeList,
				   (DlgHookUPP)0,
				   &reply, M38c,
				   (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		if (!reply.good) 
		{
			//numTimesInFile = this -> GetNumTimesInFile();	// use recs?
			//if (numTimesInFile>0)
			/*if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;*/
			//err = dynamic_cast<NetCDFMoverCurv *>(this)->ReorderPoints(velocityH,newMap,errmsg);	
			if (isLandMask) err = dynamic_cast<NetCDFMoverCurv *>(this)->ReorderPoints(landmaskH,newMap,errmsg);	
			else if (isCoopsMask) err = ReorderPointsCOOPSMask(landmaskH,newMap,errmsg);
			else err = ReorderPointsNoMask(newMap,errmsg);
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
			//return 0;
		}
		
		my_p2cstr(reply.fName);
		
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
#else
		strcpy(topPath, reply.fName);
#endif
#endif		
		strcpy (s, topPath);
		err = ReadTopology(topPath,newMap);	// newMap here
		goto depths;
		//SplitPathFile (s, fileName);
	}
	
	//numTimesInFile = this -> GetNumTimesInFile();
	//if (numTimesInFile>0)
	/*if (recs>0)
		err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
	else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
	if(err) goto done;*/
	//if (isLandMask) err = ReorderPoints(velocityH,newMap,errmsg);
	if (isLandMask) err = ReorderPoints(landmaskH,newMap,errmsg);
	else if (isCoopsMask) err = ReorderPointsCOOPSMask(landmaskH,newMap,errmsg);
	else err = ReorderPointsNoMask(newMap,errmsg);
	//else err = ReorderPointsNoMask(velocityH,newMap,errmsg);
	//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	
depths:
	if (err) goto done;
	// also translate to fDepthDataInfo and fDepthsH here, using sigma or zgrid info
	
	if (totalDepthsH)
	{
		fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols);
		if(!fDepthsH){TechError("NetCDFMoverCurv::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i=0;i<latLength;i++)
		{
			for (j=0;j<lonLength;j++)
			{
				//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
				//lat_vals[(latLength-i-1)*lonLength+j]=0.;
				//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
				//lon_vals[(latLength-i-1)*lonLength+j]=0.;
				INDEXH(totalDepthsH,i*lonLength+j) = fabs(depth_vals[(latLength-i-1)*lonLength+j]) * scale_factor;	
				INDEXH(fDepthsH,i*lonLength+j) = fabs(depth_vals[(latLength-i-1)*lonLength+j]) * scale_factor;	
			}
		}
		//((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}
	
	fNumDepthLevels = sigmaLength;
	if (sigmaLength>1)
	{
		//status = nc_get_vara_double(ncid, depthvarid, &ptIndex, &pt_count[2], depthLevels);
		//if (status != NC_NOERR) {err=-1; goto done;}
		float sigma = 0;
		fDepthLevelsHdl = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
		if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
		for (i=0;i<sigmaLength;i++)
		{	// decide what to do here, may be upside down for ROMS
			sigma = sigma_vals[i];
			if (sigma_vals[0]==1) 
				INDEXH(fDepthLevelsHdl,i) = (1-sigma);	// in this case velocities will be upside down too...
			else
			{
				if (fVar.gridType == SIGMA_ROMS)
					INDEXH(fDepthLevelsHdl,i) = sigma;
				else
					INDEXH(fDepthLevelsHdl,i) = fabs(sigma);
			}
			
		}
		if (fVar.gridType == SIGMA_ROMS)
		{
			fDepthLevelsHdl2 = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
			if (!fDepthLevelsHdl2) {err = memFullErr; goto done;}
			for (i=0;i<sigmaLength;i++)
			{
				sigma = sigma_vals2[i];
				//if (sigma_vals[0]==1) 
				//INDEXH(fDepthLevelsHdl,i) = (1-sigma);	// in this case velocities will be upside down too...
				//else
				INDEXH(fDepthLevelsHdl2,i) = sigma;
			}
			hc = hc_param;
		}
	}
	
	/*if (totalDepthsH) 
	 {	// use fDepths only if 
	 fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols);
	 if(!fDepthsH){TechError("NetCDFMoverCurv::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	 fDepthsH = totalDepthsH;
	 }*/	// may be null, call it barotropic if depths exist??
	// CalculateVerticalGrid(sigmaLength,sigmaLevelsH,totalDepthsH);	// maybe multigrid
	/*{
	 long j,index = 0;
	 fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**fDepthDataInfo)*fNumRows*fNumCols);
	 if(!fDepthDataInfo){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	 if (fVar.gridType==TWO_D) 
	 {if (totalDepthsH) fDepthsH = totalDepthsH;}	// may be null, call it barotropic if depths exist?? - should copy here
	 // assign arrays
	 else
	 {	//TWO_D grid won't need fDepthsH
	 fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumRows*fNumCols*fVar.maxNumDepths);
	 if(!fDepthsH){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	 
	 }
	 for (i=0;i<fNumRows;i++)
	 {
	 for (j=0;j<fNumCols;j++)
	 {
	 // might want to order all surface depths, all sigma1, etc., but then indexToDepthData wouldn't work
	 // have 2D case, zgrid case as well
	 if (fVar.gridType==TWO_D)
	 {
	 if (totalDepthsH) (*fDepthDataInfo)[i*fNumCols+j].totalDepth = (*totalDepthsH)[i*fNumCols+j];
	 else (*fDepthDataInfo)[i*fNumCols+j].totalDepth = -1;	// no depth data
	 (*fDepthDataInfo)[i*fNumCols+j].indexToDepthData = i*fNumCols+j;
	 (*fDepthDataInfo)[i*fNumCols+j].numDepths = 1;
	 }
	 else
	 {
	 (*fDepthDataInfo)[i*fNumCols+j].totalDepth = (*totalDepthsH)[i*fNumCols+j];
	 (*fDepthDataInfo)[i*fNumCols+j].indexToDepthData = index;
	 (*fDepthDataInfo)[i*fNumCols+j].numDepths = sigmaLength;
	 for (k=0;k<sigmaLength;k++)
	 {
	 //(*fDepthsH)[index+j] = (*totalDepthsH)[i]*(1-(*sigmaLevelsH)[j]);
	 // any other option than 1:0 or 0:1 ?
	 if (sigma_vals[0]==1) (*fDepthsH)[index+j] = (*totalDepthsH)[i*fNumCols+j]*(1-sigma_vals[k]);
	 else (*fDepthsH)[index+j] = (*totalDepthsH)[i*fNumCols+j]*sigma_vals[k];	// really need a check on all values
	 //(*fDepthsH)[j*fNumNodes+i] = totalDepthsH[i]*(1-sigmaLevelsH[j]);
	 }
	 index+=sigmaLength;
	 }
	 }
	 }
	 }*/
	if (totalDepthsH)
	{	// may need to extend the depth grid along with lat/lon grid - not sure what to use for the values though...
		// not sure what map will expect in terms of depths order
		long n,ptIndex,iIndex,jIndex;
		long numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
		//_SetHandleSize((Handle)totalDepthsH,(fNumRows+1)*(fNumCols+1)*sizeof(float));
		_SetHandleSize((Handle)totalDepthsH,numPoints*sizeof(float));
		//for (i=0; i<fNumRows*fNumCols; i++)
		//for (i=0; i<(fNumRows+1)*(fNumCols+1); i++)
		
		/*if (iIndex==0)
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
		 }*/
		
		for (i=0; i<numPoints; i++)
		{	// works okay for simple grid except for far right column (need to extend depths similar to lat/lon)
			// if land use zero, if water use point next to it?
			ptIndex = INDEXH(fVerdatToNetCDFH,i);
			if (bIsCOOPSWaterMask)
			{
				iIndex = ptIndex/(fNumCols);
				jIndex = ptIndex%(fNumCols);
			}
			else {
				iIndex = ptIndex/(fNumCols+1);
				jIndex = ptIndex%(fNumCols+1);
			}

			//iIndex = ptIndex/(fNumCols+1);
			//jIndex = ptIndex%(fNumCols+1);
			if (iIndex>0 && jIndex<fNumCols)
			//if (iIndex>0 && jIndex<fNumCols)
				ptIndex = (iIndex-1)*(fNumCols)+jIndex;
			else
				ptIndex = -1;
			
			//n = INDEXH(fVerdatToNetCDFH,i);
			//if (n<0 || n>= fNumRows*fNumCols) {printError("indices messed up"); err=-1; goto done;}
			//INDEXH(totalDepthsH,i) = depth_vals[n];
			if (ptIndex<0 || ptIndex>= fNumRows*fNumCols) 
			{
				//printError("indices messed up"); 
				//err=-1; goto done;
				//INDEXH(totalDepthsH,i) = 0;	// need to figure out what to do here...
				if (iIndex==0 && jIndex==fNumCols) ptIndex = jIndex-1;
				else if (iIndex==0) ptIndex = jIndex;
				else if (jIndex==fNumCols)ptIndex = (iIndex-1)*fNumCols+jIndex-1;
				if (ptIndex<0 || ptIndex >= fNumRows*fNumCols)
					INDEXH(totalDepthsH,i) = 0;
				else
					INDEXH(totalDepthsH,i) = INDEXH(fDepthsH,ptIndex);	// need to figure out what to do here...
				continue;
			}
			//INDEXH(totalDepthsH,i) = depth_vals[ptIndex];
			INDEXH(totalDepthsH,i) = INDEXH(fDepthsH,ptIndex);
		}
		if (!bIsCOOPSWaterMask)	// code goes here, figure out how to handle depths in this case
			((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}
	
done:
	// code goes here, set bathymetry
	if (err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0; fVertexPtsH = 0;}
		if(sigmaLevelsH) {DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH = 0;}
		if (fDepthLevelsHdl) {DisposeHandle((Handle)fDepthLevelsHdl); fDepthLevelsHdl=0;}
		if (fDepthLevelsHdl2) {DisposeHandle((Handle)fDepthLevelsHdl2); fDepthLevelsHdl2=0;}
	}
	
	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (sigma_vals) delete [] sigma_vals;
	if (modelTypeStr) delete [] modelTypeStr;
	//if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
	return err;
}


OSErr NetCDFMoverCurv::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	OSErr err = 0;
	long i,j,k;
	char path[256], outPath[256];
	char *velUnits=0; 
	int status, ncid, numdims;
	int curr_ucmp_id, curr_vcmp_id, curr_wcmp_id, angle_id, mask_id, uv_ndims;
	static size_t curr_index[] = {0,0,0,0}, angle_index[] = {0,0};
	static size_t curr_count[4], angle_count[2];
	size_t velunit_len;
	//float *curr_uvals = 0,*curr_vvals = 0, fill_value=-1e-72;
	//float *landmask = 0;
	double *curr_uvals = 0,*curr_vvals = 0, *curr_wvals = 0, fill_value=-1e+34, test_value=8e+10;
	double *landmask = 0, velConversion=1.;
	//short *curr_uvals_Navy = 0,*curr_vvals_Navy = 0, fill_value_Navy;
	//float *angle_vals = 0,debug_mask;
	double *angle_vals = 0,debug_mask;
	//long totalNumberOfVels = fNumRows * fNumCols;
	long totalNumberOfVels = fNumRows * fNumCols * fVar.maxNumDepths;
	VelocityFH velH = 0;
	FLOATH wvelH = 0;
	long latlength = fNumRows, numtri = 0;
	long lonlength = fNumCols;
	//float scale_factor = 1.,angle = 0.,u_grid,v_grid;
	double scale_factor = 1.,angle = 0.,u_grid,v_grid;
	long numDepths = fVar.maxNumDepths;	// assume will always have full set of depths at each point for now
	Boolean bRotated = true, isLandMask = true, bIsWVel = false;
	
	errmsg[0]=0;
	
	// write out verdat file for debugging
	/*FILE *outfile = 0;
	 char name[32], verdatpath[256],m[300];
	 SFReply reply;
	 Point where = CenteredDialogUpLeft(M55);
	 Boolean changeExtension = false;	// for now
	 char previousPath[256]="",defaultExtension[3]="";
	 char ibmBackwardsTypeStr[32] = "";
	 strcpy(name,"NewVerdat.dat");
	 //errmsg[0]=0;
	 
	 #if TARGET_API_MAC_CARBON
	 err = AskUserForSaveFilename("verdat.dat",verdatpath,".dat",true);
	 if (err) return USERCANCEL;
	 #else
	 #ifdef MAC
	 sfputfile(&where, "Name:", name, (DlgHookUPP)0, &reply);
	 #else
	 sfpputfile(&where, ibmBackwardsTypeStr, name, (MyDlgHookProcPtr)0, &reply,
	 M55, (ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
	 #endif
	 if (!reply.good) {err = -1; goto done;}
	 
	 my_p2cstr(reply.fName);
	 #ifdef MAC
	 GetFullPath (reply.vRefNum, 0, (char *) "", verdatpath);
	 strcat (verdatpath, ":");
	 strcat (verdatpath, (char *) reply.fName);
	 #else
	 strcpy(verdatpath, reply.fName);
	 #endif
	 #endif
	 //strcpy(sExportSelectedTriPath, verdatpath); // remember the path for the user
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
	
	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	if (numdims>=4)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		curr_count[1] = numDepths;	// depth
		curr_count[2] = latlength;
		curr_count[3] = lonlength;
	}
	else
	{
		curr_count[1] = latlength;	
		curr_count[2] = lonlength;
	}
	angle_count[0] = latlength;
	angle_count[1] = lonlength;
	
	//outfile=fopen(verdatpath,"w");
	//if (!outfile) {err = -1; printError("Unable to open file for writing"); goto done;}
	//fprintf(outfile,"DOGS\tMETERS\n");
	
	if (fIsNavy)
	{
		numDepths = 1;
		// need to check if type is float or short, if float no scale factor?
		curr_uvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_uvals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		curr_vvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_vvals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		angle_vals = new double[latlength*lonlength]; 
		if(!angle_vals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		status = nc_inq_varid(ncid, "water_gridu", &curr_ucmp_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_varid(ncid, "water_gridv", &curr_vcmp_id);	// what if only input one at a time (u,v separate movers)?
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_vara_double(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_vara_double(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_att_double(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
		status = nc_inq_varid(ncid, "grid_orient", &angle_id);
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_get_vara_double(ncid, angle_id, angle_index, angle_count, angle_vals);
		if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
	}
	else
	{
		status = nc_inq_varid(ncid, "mask", &mask_id);
		if (status != NC_NOERR)	{/*err=-1; goto done;*/ isLandMask = false;}
		status = nc_inq_varid(ncid, "ang", &angle_id);
		if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
		else
		{
			angle_vals = new double[latlength*lonlength]; 
			if(!angle_vals) {TechError("GridVel::ReadNetCDFFile()", "new[ ]", 0); err = memFullErr; goto done;}
			status = nc_get_vara_double(ncid, angle_id, angle_index, angle_count, angle_vals);
			if (status != NC_NOERR) {/*err = -1; goto done;*/bRotated = false;}
		}
		curr_uvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_uvals) 
		{
			TechError("GridVel::ReadNetCDFFile()", "new[]", 0); 
			err = memFullErr; 
			goto done;
		}
		//curr_vvals = new float[latlength*lonlength]; 
		curr_vvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_vvals) 
		{
			TechError("GridVel::ReadNetCDFFile()", "new[]", 0); 
			err = memFullErr; 
			goto done;
		}
		curr_wvals = new double[latlength*lonlength*numDepths]; 
		if(!curr_wvals) 
		{
			TechError("GridVel::ReadNetCDFFile()", "new[]", 0); 
			err = memFullErr; 
			goto done;
		}
		/*if (isLandMask)
		{
			//landmask = new float[latlength*lonlength]; 
			landmask = new double[latlength*lonlength]; 
			if(!landmask) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		}*/
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
		status = nc_inq_varid(ncid, "W", &curr_wcmp_id);
		if (status != NC_NOERR)
		{
			status = nc_inq_varid(ncid, "w", &curr_wcmp_id);
			if (status != NC_NOERR)
			{
				status = nc_inq_varid(ncid, "water_w", &curr_wcmp_id);
				if (status != NC_NOERR)
					//{err = -1; goto done;}
					bIsWVel = false;
				else
					bIsWVel = true;
			}
			//{err = -1; goto done;}
		}
		/*if (isLand)
		{
			//status = nc_get_vara_float(ncid, mask_id, angle_index, angle_count, landmask);
			status = nc_get_vara_double(ncid, mask_id, angle_index, angle_count, landmask);
			if (status != NC_NOERR) {err = -1; goto done;}
		}*/
		status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
		if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {curr_count[1] = latlength; curr_count[2] = lonlength;}}	// could have more dimensions than are used in u,v
		//status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		status = nc_get_vara_double(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		//status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		status = nc_get_vara_double(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
		if (status != NC_NOERR) {err = -1; goto done;}
		if (bIsWVel)
		{	
			status = nc_get_vara_double(ncid, curr_wcmp_id, curr_index, curr_count, curr_wvals);
			if (status != NC_NOERR) {err = -1; goto done;}
		}
		status = nc_inq_attlen(ncid, curr_ucmp_id, "units", &velunit_len);
		if (status == NC_NOERR)
		{
			velUnits = new char[velunit_len+1];
			status = nc_get_att_text(ncid, curr_ucmp_id, "units", velUnits);
			if (status == NC_NOERR)
			{
				velUnits[velunit_len] = '\0';
				if (!strcmpnocase(velUnits,"cm/s"))
					velConversion = .01;
				else if (!strcmpnocase(velUnits,"m/s"))
					velConversion = 1.0;
			}
		}
		
		
		//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		status = nc_get_att_double(ncid, curr_ucmp_id, "_FillValue", &fill_value);
		//if (status != NC_NOERR) {status = nc_get_att_float(ncid, curr_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){err = -1; goto done;}*/}	// don't require
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "Fill_Value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "FillValue", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "missing_value", &fill_value);/*if (status != NC_NOERR){fill_value=-1e+32;}{err = -1; goto done;}*/}	// don't require
		//if (status != NC_NOERR) {err = -1; goto done;}	// don't require
		status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
	}	
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
	for (k=0;k<numDepths;k++)
	{
		for (i=0;i<latlength;i++)
		{
			for (j=0;j<lonlength;j++)
			{
				if (fIsNavy)
				{
					if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
						curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
					if (curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
						curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
					u_grid = (double)curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols];
					v_grid = (double)curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols];
					if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
					INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = u_grid*cos(angle*PI/180.)-v_grid*sin(angle*PI/180.);
					INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = u_grid*sin(angle*PI/180.)+v_grid*cos(angle*PI/180.);
				}
				else
				{
					// Look for a land mask, but do this if don't find one - float mask(lat,lon) - 1,0 which is which?
					// Until the files have land masks the work around for NYNJ is to make sure zero is treated as a velocity
					// while for Galveston (and Tampa Bay) zero is a land value, not sure for Lake Erie 
					//if (curr_uvals[(latlength-i-1)*lonlength+j]==0. && curr_vvals[(latlength-i-1)*lonlength+j]==0.)
					//curr_uvals[(latlength-i-1)*lonlength+j] = curr_vvals[(latlength-i-1)*lonlength+j] = 1e-06;
					
					// just leave fillValue as velocity for new algorithm - comment following lines out
					// should eliminate the above problem, assuming fill_value is a land mask
					/*if (curr_uvals[(latlength-i-1)*lonlength+j]==fill_value)
					 curr_uvals[(latlength-i-1)*lonlength+j]=0.;
					 if (curr_vvals[(latlength-i-1)*lonlength+j]==fill_value)
					 curr_vvals[(latlength-i-1)*lonlength+j]=0.;*/
					

					/*if (isLandMask)
					{
						if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value || curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
						if (abs(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols])>test_value || abs(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols])>test_value)
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#ifdef MAC
						//if (__isnan(curr_uvals[(latlength-i-1)*lonlength+j]) || __isnan(curr_vvals[(latlength-i-1)*lonlength+j]))
						//if ((curr_uvals[(latlength-i-1)*lonlength+j])==NAN || (curr_vvals[(latlength-i-1)*lonlength+j])==NAN)
						if (isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#else
						if (_isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || _isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
#endif
						debug_mask = landmask[(latlength-i-1)*lonlength+j];
						//if (debug_mask == 1.1) numtri++;
						if (debug_mask > 0) 
						{
							numtri++;
						}
						//if (landmask[(latlength-i-1)*lonlength+j]<1)	// land
						if (landmask[(latlength-i-1)*lonlength+j]<1 || landmask[(latlength-i-1)*lonlength+j]>8)	// land
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=fill_value;
						//if (landmask[(latlength-i-1)*lonlength+j]<1)
						if (landmask[(latlength-i-1)*lonlength+j]<1 || landmask[(latlength-i-1)*lonlength+j]>8)
							curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=fill_value;
					}
					else
					{*/
						if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value || curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
						// NOTE: if leave velocity as NaN need to be sure to check for it wherever velocity is used (GetMove,Draw,...)
						if (isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]) || isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
							curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] = 0;
						//if (curr_uvals[(latlength-i-1)*lonlength+j]==0 && curr_vvals[(latlength-i-1)*lonlength+j]==0)
						//curr_uvals[(latlength-i-1)*lonlength+j] = curr_vvals[(latlength-i-1)*lonlength+j] = fill_value;
					//}
					/////////////////////////////////////////////////
					if (bRotated)
					{
						u_grid = (double)curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
						v_grid = (double)curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
						if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
						INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = u_grid*cos(angle)-v_grid*sin(angle);	//in radians
						INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = u_grid*sin(angle)+v_grid*cos(angle);
					}
					else
					{
						INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;	// need units
						INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
					}
				}
			}
		}
	}
	*velocityH = velH;
	fFillValue = fill_value * velConversion;
	
	//if (scale_factor!=1.) fVar.curScale = scale_factor;	// hmm, this forces a reset of scale factor each time, overriding any set by hand
	if (scale_factor!=1.) fFileScaleFactor = scale_factor;
	
done:
	if (err)
	{
		strcpy(errmsg,"Error reading current data from NetCDF file");
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		//printNote("Error opening NetCDF file");
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) 
	{
		delete [] curr_uvals; 
		curr_uvals = 0;
	}
	if (curr_vvals) 
	{
		delete [] curr_vvals; 
		curr_vvals = 0;
	}
	if (curr_wvals) 
	{
		delete [] curr_wvals; 
		curr_wvals = 0;
	}

	if (landmask) {delete [] landmask; landmask = 0;}
	if (angle_vals) {delete [] angle_vals; angle_vals = 0;}
	if (velUnits) {delete [] velUnits;}
	return err;
}



void NetCDFMoverCurv::Draw(Rect r, WorldRect view) 
{	// use for curvilinear
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	long depthIndex1,depthIndex2;	// default to -1?
	long amtOfDepthData = 0;
	Rect currentMapDrawingRect = MapDrawingRect();
	WorldRect cmdr;
	
	
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&fColor);
	
	if (fDepthLevelsHdl) amtOfDepthData = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	
	if(fVar.bShowArrows || fVar.bShowGrid)
	{
		Boolean overrideDrawArrows = FALSE;
		/*if (fVar.bShowGrid) 	// make sure to draw grid even if don't draw arrows
		 {
		 ((TTriGridVel*)fGrid)->DrawCurvGridPts(r,view);
		 //return;
		 }*/	// I think this is redundant with the draw triangle (maybe just a diagnostic)
		if (fVar.bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;
			
			err = this -> SetInterval(errmsg, model->GetModelTime());
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	 
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
				//if(GetNumTimesInFile()>1 && loaded && !err)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
					//if (fEndData.timeIndex == UNASSIGNEDINDEX && time != startTime && fAllowExtrapolationOfCurrentsInTime)
				{
					timeAlpha = 1;
				}
				else
				{	//return false;
					endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
					timeAlpha = (endTime - time)/(double)(endTime - startTime);
				}
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
			}
			
			cmdr = ScreenToWorldRect(currentMapDrawingRect, MapDrawingRect(), settings.currentView);	// have a look at this to see how to recognize out of view points
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex=-1,iIndex,jIndex;
				//long ptIndex2=-1,iIndex2,jIndex2;
				WorldPoint wp,wp2;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;				
				float totalDepth=0.;
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				ptIndex = INDEXH(fVerdatToNetCDFH,i);
				
				if (bIsCOOPSWaterMask)
				{
				iIndex = ptIndex/(fNumCols);
				jIndex = ptIndex%(fNumCols);
				}
				else
				{
				iIndex = ptIndex/(fNumCols+1);
				jIndex = ptIndex%(fNumCols+1);
				}
				if (iIndex>0 && jIndex<fNumCols)
					ptIndex = (iIndex-1)*(fNumCols)+jIndex;
				else
				{ptIndex = -1; continue;}
				
				if (bIsCOOPSWaterMask) ptIndex = INDEXH(fVerdatToNetCDFH,i);
	 			if (amtOfDepthData>0 && ptIndex>=0)
				{
					totalDepth = GetTotalDepth(wp,ptIndex);
					if (totalDepth==-1)
					{
						depthIndex1 = -1; depthIndex2 = -1;
					}
					else GetDepthIndices(ptIndex,fVar.arrowDepth,totalDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					//depthIndex1 = ptIndex;
					depthIndex1 = 0;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					/*if (fDepthsH)
					 {
					 totalDepth = INDEXH(fDepthsH,ptIndex);
					 //totalDepth = INDEXH(depthsH,i);
					 }
					 else 
					 {
					 totalDepth = 0;	// error
					 }*/
					//topDepth = INDEXH(fDepthLevelsHdl,depthIndex1)*totalDepth; // times totalDepth
					//bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2)*totalDepth;
					topDepth = GetDepthAtIndex(depthIndex1,totalDepth); // times totalDepth
					bottomDepth = GetDepthAtIndex(depthIndex2,totalDepth);
					//topDepth = GetTopDepth(depthIndex1,totalDepth); // times totalDepth
					//bottomDepth = GetBottomDepth(depthIndex2,totalDepth);
					if (totalDepth == 0) depthAlpha = 1;
					else
						depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}
				// for now draw arrow at midpoint of diagonal of gridbox
				// this will result in drawing some arrows more than once
				if (GetLatLonFromIndex(iIndex-1,jIndex+1,&wp2)!=-1)	// may want to get all four points and interpolate
				{
					wp.pLat = (wp.pLat + wp2.pLat)/2.;
					wp.pLong = (wp.pLong + wp2.pLong)/2.;
				}
				
				if (bIsCOOPSWaterMask)
				{
					wp.pLat = (long)(1e6*INDEXH(fVertexPtsH,ptIndex).pLat);
					wp.pLong = (long)(1e6*INDEXH(fVertexPtsH,ptIndex).pLong);
				}
				if (wp.pLong < cmdr.loLong || wp.pLong > cmdr.hiLong || wp.pLat < cmdr.loLat || wp.pLat > cmdr.hiLat) 
					continue;
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	// should put velocities in center of grid box
				
				// Should check vs fFillValue
				// Check for constant current 
				if(((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha == 1) && ptIndex!=-1)
					//if(GetNumTimesInFile()==1 && ptIndex!=-1)
				{
					//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
					//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						velocity.v = GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v;
					}
					else 	// below surface velocity
					{
						velocity.u = depthAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*GetStartUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols);
						velocity.v = depthAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-depthAlpha)*GetStartVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols);
						//velocity.u = depthAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u+(1-depthAlpha)*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u;
						//velocity.v = depthAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v+(1-depthAlpha)*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v;
					}
				}
				else if (ptIndex!=-1)// time varying current
				{
					// need to rescale velocities for Navy case, store angle
					// should check for fillValue, don't want to try to interpolate in that case
					//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
					//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{	
						velocity.u = timeAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						velocity.v = timeAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols)+(1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols);
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*GetStartUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex1*fNumRows*fNumCols));
						velocity.u += (1-depthAlpha)*(timeAlpha*GetStartUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*GetEndUVelocity(ptIndex+depthIndex2*fNumRows*fNumCols));
						velocity.v = depthAlpha*(timeAlpha*GetStartVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols) + (1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex1*fNumRows*fNumCols));
						velocity.v += (1-depthAlpha)*(timeAlpha*GetStartVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols) + (1-timeAlpha)*GetEndVVelocity(ptIndex+depthIndex2*fNumRows*fNumCols));
						//velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).u);
						//velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).u);
						//velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex1*fNumRows*fNumCols).v);
						//velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex+depthIndex2*fNumRows*fNumCols).v);
					}
					//velocity.u = timeAlpha*GetStartUVelocity(ptIndex) + (1-timeAlpha)*GetEndUVelocity(ptIndex);
					//velocity.v = timeAlpha*GetStartVVelocity(ptIndex) + (1-timeAlpha)*GetEndVVelocity(ptIndex);
				}
				if ((velocity.u != 0 || velocity.v != 0) && (velocity.u != fFillValue && velocity.v != fFillValue)) // should already have handled fill value issue
				{
					float inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
					float inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
					short pixX = inchesX * PixelsPerInchCurrent();
					short pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
	}
	if (fVar.bShowGrid) fGrid->Draw(r,view,wayOffMapPt,fVar.curScale,fVar.arrowScale,fVar.arrowDepth,false,true,fColor);
	if (bShowDepthContours && fVar.gridType!=TWO_D) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);// careful with 3D grid
	
	RGBForeColor(&colors[BLACK]);
}

void NetCDFMoverCurv::DrawContourScale(Rect r, WorldRect view)
{
	Point		p;
	short		h,v,x,y,dY,widestNum=0;
	RGBColor	rgb;
	Rect		rgbrect;
	Rect legendRect = fLegendRect;
	char 		numstr[30],numstr2[30],text[30],errmsg[256];
	long 		i,numLevels,istep=1;
	double	minLevel, maxLevel;
	double 	value;
	float totalDepth = 0;
	long numDepths = 0, numTris = 0, triNum = 0;
	OSErr err = 0;
	PtCurMap *map = GetPtCurMap();
	TTriGridVel3D *triGrid = (TTriGridVel3D*) map->GetGrid3D(false);
	Boolean **triSelected = 0;
	long indexToDepthData = 0, index;
	long numDepthLevels = GetNumDepthLevelsInFile();
	long j;
	float sc_r, sc_r2, Cs_r, Cs_r2, depthAtLevel, depthAtNextLevel;
	
	// code goes here, need separate cases for each grid type - have depth data on points, not triangles...
	long timeDataInterval;
	Boolean loaded;
	
	if (!triGrid) return;
	triSelected = triGrid -> GetTriSelection(false);	// don't init

	//if (fVar.gridType != SIGMA_ROMS) return;
	err = SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
	if(err) return;
	
	loaded = CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
	
	if(!loaded) return;	
	
	//if (!fDepthDataInfo) return;
	//numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);	// depth from input file (?) at triangle center
	numTris = triGrid->GetNumTriangles();
	
	//list which triNum, use selected triangle, scale arrows, list values ??? 
	if (triSelected)
	{
		for (i=0;i<numTris; i++)
		{
			if ((*triSelected)[i]) 
			{
				triNum = i;
				break;
			}
		}
	}
	else
		return;
	//triNum = GetRandom(0,numTris-1);	// show or not show anything ?
	
	// code goes here, probably need different code for each grid type - how to select a grid box?, allow to select triangles on curvilinear grid? different for regular grid	
	//numDepths = INDEXH(fDepthDataInfo,triNum).numDepths;
	//totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
	{
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex2(triNum,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	}
	else return;
	//if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	//else return;
	if (fVar.gridType==SIGMA_ROMS)	// maybe always do it this way...
		totalDepth = GetTotalDepthFromTriIndex(triNum);
	else
	{
		if (fDepthsH)
		{
			totalDepth = INDEXH(fDepthsH, index);
		}
	}
	
	if (totalDepth==0 || numDepthLevels==0) return;
	//SetRGBColor(&rgb,0,0,0);
	TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
#ifdef IBM
	TextFont(kFontIDGeneva); TextSize(6);
#endif
	
	if (gSavingOrPrintingPictFile)
	{
		Rect mapRect;
#ifdef MAC
		mapRect = DrawingRect(settings.listWidth + 1, RIGHTBARWIDTH);
#else
		mapRect = DrawingRect(settings.listWidth, RIGHTBARWIDTH);
#endif
		if (!EqualRects(r,mapRect))
		{
			Boolean bCloserToTop = (legendRect.top - mapRect.top) <= (mapRect.bottom - legendRect.bottom);
			Boolean bCloserToLeft = (legendRect.left - mapRect.left) <= (mapRect.right - legendRect.right);
			if (bCloserToTop)
			{
				legendRect.top = legendRect.top - mapRect.top + r.top;
				legendRect.bottom = legendRect.bottom - mapRect.top + r.top;
			}
			else
			{
				legendRect.top = r.bottom - (mapRect.bottom - legendRect.top);
				legendRect.bottom = r.bottom - (mapRect.bottom - legendRect.bottom);
			}
			if (bCloserToLeft)
			{
				legendRect.left = legendRect.left - mapRect.left + r.left;
				legendRect.right = legendRect.right - mapRect.left + r.left;
			}
			else
			{
				legendRect.left = r.right - (mapRect.right - legendRect.left);
				legendRect.right = r.right - (mapRect.right - legendRect.right);
			}
		}
	}
	else
	{
		if (EmptyRect(&fLegendRect)||!RectInRect2(&legendRect,&r))
		{
			legendRect.top = r.top;
			legendRect.left = r.right - 80;
			//legendRect.bottom = r.top + 120;	// reset after contour levels drawn
			legendRect.bottom = r.top + 90;	// reset after contour levels drawn
			legendRect.right = r.right;	// reset if values go beyond default width
		}
	}
	rgbrect = legendRect;
	EraseRect(&rgbrect);
	
	x = (rgbrect.left + rgbrect.right) / 2;
	//dY = RectHeight(rgbrect) / 12;
	dY = 10;
	y = rgbrect.top + dY / 2;
	MyMoveTo(x - stringwidth("Depth Barbs") / 2, y + dY);
	drawstring("Depth Barbs");
	numtostring(triNum+1,numstr);
	strcpy(numstr2,"Tri Num = ");
	strcat(numstr2,numstr);
	MyMoveTo(x-stringwidth(numstr2) / 2, y + 2*dY);
	drawstring(numstr2);
	widestNum = stringwidth(numstr2);
	
	v = rgbrect.top+45;
	h = rgbrect.left;
	//if (numDepths>20) istep = (long)numDepths/20.;
	//for (i=0;i<numDepths;i++)
	//for (i=0;i<numDepthLevels;i+=istep)
	//for (i=0;i<numDepthLevels;i++)
	//if (fVar.gridType==SIGMA_ROMS)
	{
		for(j=numDepthLevels-1;j>=0;j--)	// also want j==0?
		{
			WorldPoint wp;
			Point p,p2;
			VelocityRec velocity = {0.,0.};
			Boolean offQuickDrawPlane = false;
			long depthIndex1/*, depthIndex2*/;
			Seconds time, startTime, endTime;
			double timeAlpha;
			
			//long velDepthIndex1 = (*fDepthDataInfo)[triNum].indexToDepthData+i;
			//sc_r = INDEXH(fDepthLevelsHdl,indexToDepthData+j);
			//sc_r2 = INDEXH(fDepthLevelsHdl,indexToDepthData+j-1);
			//Cs_r = INDEXH(fDepthLevelsHdl2,indexToDepthData+j);
			//Cs_r2 = INDEXH(fDepthLevelsHdl2,indexToDepthData+j-1);
			//depthAtLevel = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);	// may want this eventually
			//depthAtLevel = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
			
			if (fVar.gridType==SIGMA_ROMS)
				depthIndex1 = indexToDepthData+j;
			else
				depthIndex1 = indexToDepthData+numDepthLevels-j-1;
			//depthIndex2 = UNASSIGNEDINDEX;
			
			if((dynamic_cast<NetCDFMoverCurv *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMoverCurv *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
			{
				if (index >= 0 && depthIndex1 >= 0) 
				{
					velocity.u = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
					velocity.v = INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
				}
			}
			else
			{
				// Calculate the time weight factor
				if (dynamic_cast<NetCDFMoverCurv *>(this)->GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				time = model->GetModelTime();
				endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				timeAlpha = (endTime - time)/(double)(endTime - startTime);
				
				if (index >= 0 && depthIndex1 >= 0) 
				{
					velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).u;
					velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index+depthIndex1*fNumRows*fNumCols).v;
				}
			}
			MyMoveTo(h+40,v+.5);
			
			if ((velocity.u != 0 || velocity.v != 0))
			{
				float inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
				float inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
				short pixX = inchesX * PixelsPerInchCurrent();
				short pixY = inchesY * PixelsPerInchCurrent();
				//p.h = h+20;
				p.h = h+40;
				p.v = v+.5;
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				//MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
			if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
			v = v+9;
		}
	}
	sprintf(text, "Depth: %g m",totalDepth);
	//MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);
	MyMoveTo(h+20, v+5);
	drawstring(text);
	if (stringwidth(text)+20 > widestNum) widestNum = stringwidth(text)+20;
	v = v + 9;
	legendRect.bottom = v+3;
	if (legendRect.right<h+20+widestNum+4) legendRect.right = h+20+widestNum+4;
	else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
		legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
	RGBForeColor(&colors[BLACK]);
 	//MyFrameRect(&legendRect);
	
	if (!gSavingOrPrintingPictFile)
		fLegendRect = legendRect;
	return;
}
/*void NetCDFMoverCurv_c::DrawContourScale(Rect r, WorldRect view)
 {
 Point		p;
 short		h,v,x,y,dY,widestNum=0;
 RGBColor	rgb;
 Rect		rgbrect;
 Rect legendRect = fLegendRect;
 char 		numstr[30],numstr2[30],text[30],errmsg[256];
 long 		i,numLevels,istep=1;
 double	minLevel, maxLevel;
 double 	value;
 float totalDepth = 0;
 long numDepths = 0, numTris = 0, triNum = 0;
 OSErr err = 0;
 PtCurMap *map = GetPtCurMap();
 TTriGridVel3D *triGrid = (TTriGridVel3D*) map->GetGrid3D(false);
 Boolean **triSelected = triGrid -> GetTriSelection(false);	// don't init
 
 // code goes here, need separate cases for each grid type - have depth data on points, not triangles...
 long timeDataInterval;
 Boolean loaded;
 
 err = this -> SetInterval(errmsg);
 if(err) return;
 
 loaded = this -> CheckInterval(timeDataInterval);
 if(!loaded) return;
 
 
 if (!fDepthDataInfo) return;
 numTris = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);	// depth from input file (?) at triangle center
 
 //list which triNum, use selected triangle, scale arrows, list values ??? 
 if (triSelected)
 {
 for (i=0;i<numTris; i++)
 {
 if ((*triSelected)[i]) 
 {
 triNum = i;
 break;
 }
 }
 }
 else
 triNum = GetRandom(0,numTris-1);
 
 // code goes here, probably need different code for each grid type - how to select a grid box?, allow to select triangles on curvilinear grid? different for regular grid	
 numDepths = INDEXH(fDepthDataInfo,triNum).numDepths;
 totalDepth = INDEXH(fDepthDataInfo,triNum).totalDepth;	// depth from input file (?) at triangle center
 
 //SetRGBColor(&rgb,0,0,0);
 TextFont(kFontIDGeneva); TextSize(LISTTEXTSIZE);
 #ifdef IBM
 TextFont(kFontIDGeneva); TextSize(6);
 #endif
 
 if (gSavingOrPrintingPictFile)
 {
 Rect mapRect;
 #ifdef MAC
 mapRect = DrawingRect(settings.listWidth + 1, RIGHTBARWIDTH);
 #else
 mapRect = DrawingRect(settings.listWidth, RIGHTBARWIDTH);
 #endif
 if (!EqualRects(r,mapRect))
 {
 Boolean bCloserToTop = (legendRect.top - mapRect.top) <= (mapRect.bottom - legendRect.bottom);
 Boolean bCloserToLeft = (legendRect.left - mapRect.left) <= (mapRect.right - legendRect.right);
 if (bCloserToTop)
 {
 legendRect.top = legendRect.top - mapRect.top + r.top;
 legendRect.bottom = legendRect.bottom - mapRect.top + r.top;
 }
 else
 {
 legendRect.top = r.bottom - (mapRect.bottom - legendRect.top);
 legendRect.bottom = r.bottom - (mapRect.bottom - legendRect.bottom);
 }
 if (bCloserToLeft)
 {
 legendRect.left = legendRect.left - mapRect.left + r.left;
 legendRect.right = legendRect.right - mapRect.left + r.left;
 }
 else
 {
 legendRect.left = r.right - (mapRect.right - legendRect.left);
 legendRect.right = r.right - (mapRect.right - legendRect.right);
 }
 }
 }
 else
 {
 if (EmptyRect(&fLegendRect)||!RectInRect2(&legendRect,&r))
 {
 legendRect.top = r.top;
 legendRect.left = r.right - 80;
 //legendRect.bottom = r.top + 120;	// reset after contour levels drawn
 legendRect.bottom = r.top + 90;	// reset after contour levels drawn
 legendRect.right = r.right;	// reset if values go beyond default width
 }
 }
 rgbrect = legendRect;
 EraseRect(&rgbrect);
 
 x = (rgbrect.left + rgbrect.right) / 2;
 //dY = RectHeight(rgbrect) / 12;
 dY = 10;
 y = rgbrect.top + dY / 2;
 MyMoveTo(x - stringwidth("Depth Barbs") / 2, y + dY);
 drawstring("Depth Barbs");
 numtostring(triNum+1,numstr);
 strcpy(numstr2,"Tri Num = ");
 strcat(numstr2,numstr);
 MyMoveTo(x-stringwidth(numstr2) / 2, y + 2*dY);
 drawstring(numstr2);
 widestNum = stringwidth(numstr2);
 
 v = rgbrect.top+45;
 h = rgbrect.left;
 //if (numDepths>20) istep = (long)numDepths/20.;
 //for (i=0;i<numDepths;i++)
 for (i=0;i<numDepths;i+=istep)
 {
 WorldPoint wp;
 Point p,p2;
 VelocityRec velocity = {0.,0.};
 Boolean offQuickDrawPlane = false;
 
 long velDepthIndex1 = (*fDepthDataInfo)[triNum].indexToDepthData+i;
 
 velocity.u = INDEXH(fStartData.dataHdl,velDepthIndex1).u;
 velocity.v = INDEXH(fStartData.dataHdl,velDepthIndex1).v;
 
 MyMoveTo(h+40,v+.5);
 
 if ((velocity.u != 0 || velocity.v != 0))
 {
 float inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
 float inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
 short pixX = inchesX * PixelsPerInchCurrent();
 short pixY = inchesY * PixelsPerInchCurrent();
 //p.h = h+20;
 p.h = h+40;
 p.v = v+.5;
 p2.h = p.h + pixX;
 p2.v = p.v - pixY;
 //MyMoveTo(p.h, p.v);
 MyLineTo(p2.h, p2.v);
 MyDrawArrow(p.h,p.v,p2.h,p2.v);
 }
 if (p2.h-h>widestNum) widestNum = p2.h-h;	// also issue of negative velocity, or super large value, maybe scale?
 v = v+9;
 }
 sprintf(text, "Depth: %g m",totalDepth);
 //MyMoveTo(x - stringwidth(text) / 2, y + 3 * dY);
 MyMoveTo(h+20, v+5);
 drawstring(text);
 if (stringwidth(text)+20 > widestNum) widestNum = stringwidth(text)+20;
 v = v + 9;
 legendRect.bottom = v+3;
 if (legendRect.right<h+20+widestNum+4) legendRect.right = h+20+widestNum+4;
 else if (legendRect.right>legendRect.left+80 && h+20+widestNum+4<=legendRect.left+80)
 legendRect.right = legendRect.left+80;	// may want to redraw to recenter the header
 RGBForeColor(&colors[BLACK]);
 //MyFrameRect(&legendRect);
 
 if (!gSavingOrPrintingPictFile)
 fLegendRect = legendRect;
 return;
 }*/
/*double NetCDFMoverCurv_c::GetTopDepth(long depthIndex, double totalDepth)
 {	// really can combine and use GetDepthAtIndex - could move to base class
 double topDepth = 0;
 float sc_r, Cs_r;
 if (fVar.gridType == SIGMA_ROMS)
 {
 sc_r = INDEXH(fDepthLevelsHdl,depthIndex);
 Cs_r = INDEXH(fDepthLevelsHdl2,depthIndex);
 //topDepth = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
 topDepth = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
 }
 else
 topDepth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth; // times totalDepth
 
 return topDepth;
 }
 double NetCDFMoverCurv_c::GetBottomDepth(long depthIndex, double totalDepth)
 {
 double bottomDepth = 0;
 float sc_r, Cs_r;
 if (fVar.gridType == SIGMA_ROMS)
 {
 sc_r = INDEXH(fDepthLevelsHdl,depthIndex);
 Cs_r = INDEXH(fDepthLevelsHdl2,depthIndex);
 //bottomDepth = abs(hc * (sc_r-Cs_r) + Cs_r * totalDepth);
 bottomDepth = abs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
 }
 else
 bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth;
 
 return bottomDepth;
 }*/

/*Boolean IsTransposeArrayHeaderLine(char *s, long* numPts)
{		
	char* strToMatch = "TransposeArray";
	long numScanned, len = strlen(strToMatch);
	if(!strncmpnocase(s,strToMatch,len)) {
		numScanned = sscanf(s+len+1,"%ld",numPts);
		if (numScanned != 1 || *numPts <= 0)
			return FALSE; 
	}
	else
		return FALSE;
	return TRUE; 
}*/
/////////////////////////////////////////////////////////////////
//OSErr NetCDFMoverCurv::ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg)
/*OSErr ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg)
// Note: '*line' must contain the line# at which the vertex data begins
{ // May want to combine this with read vertices if it becomes a mandatory component of PtCur files
	OSErr err=0;
	char s[64];
	long i,numScanned,index;
	LONGH verdatToNetCDFH = 0;
	
	strcpy(errmsg,""); // clear it
	
	verdatToNetCDFH = (LONGH)_NewHandle(sizeof(long)*numPts);
	if(!verdatToNetCDFH){TechError("NetCDFMover::ReadTransposeArray()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	for(i=0;i<numPts;i++)
	{
		NthLineInTextOptimized(*fileBufH, (*line)++, s, 64); 
		numScanned=sscanf(s,"%ld",&index) ;
		if (numScanned!= 1)
		{ err = -1; TechError("NetCDFMover::ReadTransposeArray()", "sscanf() == 1", 0); goto done; }
		(*verdatToNetCDFH)[i] = index;
	}
	*transposeArray = verdatToNetCDFH;
	
done:
	
	if(err) 
	{
		if(verdatToNetCDFH) {DisposeHandle((Handle)verdatToNetCDFH); verdatToNetCDFH=0;}
	}
	return err;		
	
}*/

//OSErr NetCDFMoverCurv::ReadTopology(char* path, TMap **newMap)
OSErr NetCDFMoverCurv::ReadTopology(vector<string> &linesInFile, TMap **newMap)
{
	// import NetCDF curvilinear info so don't have to regenerate
	char s[1024], errmsg[256]/*, s[256], topPath[256]*/;
	long i, numPoints, numTopoPoints, line = 0, numPts;
	string currentLine;
	//CHARH f = 0;
	OSErr err = 0;
	
	TopologyHdl topo=0;
	LongPointHdl pts=0;
	FLOATH depths=0;
	VelocityFH velH = 0;
	DAGTreeStruct tree;
	WorldRect bounds = voidWorldRect;
	
	TTriGridVel3D *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0, boundaryPts=0;
	
	errmsg[0]=0;
	
	/*if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("NetCDFMover::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	*/
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	currentLine = linesInFile[line++];
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	//if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	if(IsTransposeArrayHeaderLine(currentLine,numPts)) // 
	{
		//if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		if (err = ReadTransposeArray(linesInFile,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	
	//if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;
	if(err = ReadTVertices(linesInFile,&line,&pts,&depths,errmsg)) goto done;
	
	if(pts) 
	{
		LongPoint	thisLPoint;
		
		numPts = _GetHandleSize((Handle)pts)/sizeof(LongPoint);
		if(numPts > 0)
		{
			WorldPoint  wp;
			for(i=0;i<numPts;i++)
			{
				thisLPoint = INDEXH(pts,i);
				wp.pLat = thisLPoint.v;
				wp.pLong = thisLPoint.h;
				AddWPointToWRect(wp.pLat, wp.pLong, &bounds);
			}
		}
	}
	MySpinCursor();
	
	currentLine = linesInFile[line++];
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	//if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	if(IsBoundarySegmentHeaderLine(currentLine,numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundarySegs>0)
			//err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
			err = ReadBoundarySegs(linesInFile,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		currentLine = linesInFile[line++];
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	
	//if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	if(IsWaterBoundaryHeaderLine(currentLine,numWaterBoundaries,numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			//err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
			err = ReadWaterBoundaries(linesInFile,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		currentLine = linesInFile[line++];
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Water boundaries header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	//if(IsBoundaryPointsHeaderLine(s,&numBoundaryPts)) // Boundary data from CATs
	if(IsBoundaryPointsHeaderLine(currentLine,numBoundaryPts)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			//err = ReadBoundaryPts(f,&line,&boundaryPts,numBoundaryPts,errmsg);
			err = ReadBoundaryPts(linesInFile,&line,&boundaryPts,numBoundaryPts,errmsg);
		if(err) goto done;
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		currentLine = linesInFile[line++];
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not always needed ? probably always needed for curvilinear
	}
	MySpinCursor(); // JLM 8/4/99
	
	//if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	if(IsTTopologyHeaderLine(currentLine,numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		//err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		err = ReadTTopologyBody(linesInFile,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
		currentLine = linesInFile[line++];
	}
	else
	{
		err = -1; // for now we require TTopology
		strcpy(errmsg,"Error in topology header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	//if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	if(IsTIndexedDagTreeHeaderLine(currentLine,numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		//err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
		err = ReadTIndexedDagTreeBody(linesInFile,&line,&tree,errmsg,numPoints);
		if(err) goto done;
	}
	else
	{
		err = -1; // for now we require TIndexedDagTree
		strcpy(errmsg,"Error in dag tree header line");
		if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99
	
	/////////////////////////////////////////////////
	// if the boundary information is in the file we'll need to create a bathymetry map (required for 3D)
	
	if (waterBoundaries && waterBoundaries && boundaryPts && (this -> moverMap == model -> uMap))
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetBoundaryPoints(boundaryPts);	
		map->SetWaterBoundaries(waterBoundaries);
		
		*newMap = map;
	}
	
	//if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	else	// maybe assume rectangle grids will have map?
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts=0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMover::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel3D*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	//triGrid -> SetDepths(depths);
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		err = -1;
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	
	pts = 0;	// because fGrid is now responsible for it
	topo = 0; // because fGrid is now responsible for it
	tree.treeHdl = 0; // because fGrid is now responsible for it
	velH = 0; // because fGrid is now responsible for it
	//depths = 0;
	
done:
	
	if(depths) {DisposeHandle((Handle)depths); depths=0;}
	/*if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}*/
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFMover::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
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
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	return err;
}

OSErr NetCDFMoverCurv::ReadTopology(const char *path, TMap **newMap)
{
	// note the commented out code would be needed for a topology file in a resource (if we made a location file with a netcdf file...)
	vector<string> linesInFile;
	char outPath[kMaxNameLen];
	//CHARH fileBufH = 0;
	//vector<string> linesInBuffer;
	OSErr err = 0;
	
	//if (!path || !path[0]) return 0;
	
	// this supports reading from resource for location files
	/*if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &fileBufH)) {
		TechError("TideCurCycleMover::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)fileBufH); // JLM 8/4/99
	
	ReadLinesInBuffer(fileBufH, linesInBuffer);	
	err = ReadTopology(linesInBuffer, newMap);
	 */
#ifdef TARGET_API_MAC_CARBON
	 if (IsClassicPath((char*)path))
	 {
		 err = ConvertTraditionalPathToUnixPath(path, outPath, kMaxNameLen) ;
		 if (!err) strcpy((char*)path,outPath);
		 else return err;
	 }
#endif
	 // Note, this doesn't work for resources in Location Files...
	 ReadLinesInFile(path, linesInFile);
	 err = ReadTopology(linesInFile, newMap);
	
//done:
	/*if(fileBufH) 
	{
		_HUnlock((Handle)fileBufH); 
		DisposeHandle((Handle)fileBufH); 
		fileBufH = 0;
	}*/
	return err;
}

OSErr NetCDFMoverCurv::ExportTopology(char* path)
{
	// export NetCDF curvilinear info so don't have to regenerate each time
	// move to NetCDFMover so Tri can use it too
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n, v1,v2,v3,n1,n2,n3;
	double x,y,z=0;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl		treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	BFPB bfpb;
	
	triGrid = (TTriGridVel*)(this->fGrid);
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
	depthsH = ((TTriGridVel3D*)triGrid)->GetDepths();
	if(!ptsH || !topH || !treeH) 
	{
		printError("There is no topology to export");
		return -1;
	}
	if (moverMap->IAm(TYPE_PTCURMAP))
	{
		boundaryTypeH = (dynamic_cast<PtCurMap *>(moverMap))->GetWaterBoundaries();
		boundarySegmentsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundarySegs();
		boundaryPointsH = (dynamic_cast<PtCurMap *>(moverMap))->GetBoundaryPoints();
		if (!boundaryTypeH || !boundarySegmentsH || !boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
	}
	else
	{
		// any issue with trying to write out non-existent fields?
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else {printError("There is no transpose array"); err = -1; goto done;}
	sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<n;i++)
	{	
		sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	nver = _GetHandleSize((Handle)ptsH)/sizeof(**ptsH);
	//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
	sprintf(hdrStr,"Vertices\t%ld\n",nver);	// total vertices
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	sprintf(hdrStr,"%ld\t%ld\n",nver,nver);	// junk line
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	for(i=0;i<nver;i++)
	{	
		x = (*ptsH)[i].h/1000000.0;
		y =(*ptsH)[i].v/1000000.0;
		//sprintf(topoStr,"%ld\t%lf\t%lf\t%lf\n",i+1,x,y,(*gDepths)[i]);
		//sprintf(topoStr,"%ld\t%lf\t%lf\n",i+1,x,y);
		/*if (depthsH) 
		{
			z = (*depthsH)[i];
			sprintf(topoStr,"%lf\t%lf\t%lf\n",x,y,z);
		}
		else*/
			sprintf(topoStr,"%lf\t%lf\n",x,y);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	//boundary points - an optional handle, only for curvilinear case
	
	if (boundarySegmentsH) 
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
	}
	numTriangles = _GetHandleSize((Handle)topH)/sizeof(**topH);
	sprintf(hdrStr,"Topology\t%ld\n",numTriangles);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
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
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	numBranches = _GetHandleSize((Handle)treeH)/sizeof(**treeH);
	sprintf(hdrStr,"DAGTree\t%ld\n",dagTree->fNumBranches);
	strcpy(buffer,hdrStr);
	if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	
	for(i = 0; i<dagTree->fNumBranches; i++)
	{
		sprintf(topoStr,"%ld\t%ld\t%ld\n",(*treeH)[i].topoIndex,(*treeH)[i].branchLeft,(*treeH)[i].branchRight);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
done:
	// 
	FSCloseBuf(&bfpb);
	if(err) {	
		printError("Error writing topology");
		(void)hdelete(0, 0, path); // don't leave them with a partial file
	}
	return err;
}