/*
 *  NetCDFWindMoverCurv.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "NetCDFWindMoverCurv.h"
#include "MemUtils.h"
#include "netcdf.h"
#include "CROSS.H"
#include "DagTreeIO.h"

/////////////////////////////////////////////////
// Curvilinear grid code - separate mover
// read in grid values for first time and set up transformation (dagtree?)
// need to read in lat/lon since won't be evenly spaced
// probably all wind grids will be rectangular?

NetCDFWindMoverCurv::NetCDFWindMoverCurv (TMap *owner, char *name) : NetCDFWindMover(owner, name)
{
	fVerdatToNetCDFH = 0;	
	fVertexPtsH = 0;
}


void NetCDFWindMoverCurv::Dispose ()
{
	if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
	if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	
	NetCDFWindMover::Dispose ();
}


#define NetCDFWindMoverCurvREADWRITEVERSION 1 //JLM

OSErr NetCDFWindMoverCurv::Write (BFPB *bfpb)
{
	long i, version = NetCDFWindMoverCurvREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints, index;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFWindMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("NetCDFWindMoverCurv::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	numPoints = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(**fVerdatToNetCDFH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		index = INDEXH(fVerdatToNetCDFH,i);
		if (err = WriteMacValue(bfpb, index)) goto done;
	}
	
	numPoints = _GetHandleSize((Handle)fVertexPtsH)/sizeof(**fVertexPtsH);
	if (err = WriteMacValue(bfpb, numPoints)) goto done;
	for (i=0;i<numPoints;i++)
	{
		vertex = INDEXH(fVertexPtsH,i);
		if (err = WriteMacValue(bfpb, vertex.pLat)) goto done;
		if (err = WriteMacValue(bfpb, vertex.pLong)) goto done;
	}
	
done:
	if(err)
		TechError("NetCDFWindMoverCurv::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr NetCDFWindMoverCurv::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFWindMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFWindMoverCurv::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFWindMoverCurv::Read()", "id != TYPE_NETCDFWINDMOVERCURV", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != NetCDFWindMoverCurvREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVerdatToNetCDFH = (LONGH)_NewHandleClear(sizeof(long)*numPoints);	// for curvilinear
	if(!fVerdatToNetCDFH)
	{TechError("NetCDFWindMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &index)) goto done;
		INDEXH(fVerdatToNetCDFH, i) = index;
	}
	
	if (err = ReadMacValue(bfpb, &numPoints)) goto done;	
	fVertexPtsH = (WORLDPOINTFH)_NewHandleClear(sizeof(WorldPointF)*numPoints);	// for curvilinear
	if(!fVertexPtsH)
	{TechError("NetCDFWindMoverCurv::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numPoints ; i++) {
		if (err = ReadMacValue(bfpb, &vertex.pLat)) goto done;
		if (err = ReadMacValue(bfpb, &vertex.pLong)) goto done;
		INDEXH(fVertexPtsH, i) = vertex;
	}
	
done:
	if(err)
	{
		TechError("NetCDFWindMoverCurv::Read(char* path)", " ", 0); 
		if(fVerdatToNetCDFH) {DisposeHandle((Handle)fVerdatToNetCDFH); fVerdatToNetCDFH=0;}
		if(fVertexPtsH) {DisposeHandle((Handle)fVertexPtsH); fVertexPtsH=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFWindMoverCurv::CheckAndPassOnMessage(TModelMessage *message)
{
	return NetCDFWindMover::CheckAndPassOnMessage(message); 
}

OSErr NetCDFWindMoverCurv::TextRead(char *path, TMap **newMap, char *topFilePath) // don't want a map  
{
	// this code is for curvilinear grids
	OSErr err = 0;
	long i,j, numScanned, indexOfStart = 0;
	int status, ncid, latIndexid, lonIndexid, latid, lonid, recid, timeid, numdims;
	size_t latLength, lonLength, recs, t_len, t_len2;
	float timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0, month[10];	
	char dimname[NC_MAX_NAME], s[256], topPath[256];
	WORLDPOINTFH vertexPtsH=0;
	float *lat_vals=0,*lon_vals=0,yearShift=0.;
	static size_t timeIndex,ptIndex[2]={0,0};
	static size_t pt_count[2];
	Seconds startTime, startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "",className[256]="";
	char fileName[64],*modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, fIsNavy = false;	// for now keep code around but probably don't need Navy curvilinear wind
	//VelocityFH velocityH = 0;
	char outPath[256];
	
	if (!path || !path[0]) return 0;
	strcpy(fPathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fFileName, fileName); // maybe use a name from the file
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
		
		strcpy(fFileName, modelTypeStr); 
	}
	GetClassName(className);
	if (!strcmp("NetCDF Wind",className))
		SetClassName(fFileName); //first check that name is now the default and not set by command file ("NetCDF Wind")
	
	//if (fIsNavy)
	{
		status = nc_inq_dimid(ncid, "time", &recid); //Navy
		//if (status != NC_NOERR) {err = -1; goto done;}
		if (status != NC_NOERR) 
		{	status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
			if (status != NC_NOERR) {err = -1; goto done;}
		}			
	}
	/*else
	 {
	 status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
	 if (status != NC_NOERR) {err = -1; goto done;}
	 }*/
	
	//if (fIsNavy)
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) 
	{	
		status = nc_inq_varid(ncid, "ProjectionHr", &timeid); 
		if (status != NC_NOERR) {err = -1; goto done;}
	}			
	//	if (status != NC_NOERR) {/*err = -1; goto done;*/timeid=recid;} 
	
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
		if (numScanned==5)	
		{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned==7) // has two extra time entries ??	
			time.second = 0;
		else if (numScanned<8)	
		//else if (numScanned!=8)	
		{ 
			//timeUnits = 0;	// files should always have this info
			//timeConversion = 3600.;		// default is hours
			//startTime2 = model->GetStartTime();	// default to model start time
			err = -1; TechError("NetCDFWindMoverCurv::TextRead()", "sscanf() == 8", 0); goto done;
		}
		else
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
			if (!strncmpnocase(dimname,"X",1) || !strncmpnocase(dimname,"LON",3) || !strncmpnocase(dimname,"nx",2))
			{
				lonIndexid = i;
			}
			if (!strncmpnocase(dimname,"Y",1) || !strncmpnocase(dimname,"LAT",3) || !strncmpnocase(dimname,"ny",2))
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
			if (status != NC_NOERR) 
			{
				status = nc_inq_varid(ncid, "latitude", &latid);
				if (status != NC_NOERR) {err = -1; goto done;}
			}
		}
		status = nc_inq_varid(ncid, "LONGITUDE", &lonid);
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "lon", &lonid);
			if (status != NC_NOERR) 
			{
				status = nc_inq_varid(ncid, "longitude", &lonid);
				if (status != NC_NOERR) {err = -1; goto done;}
			}
		}
	}
	
	pt_count[0] = latLength;
	pt_count[1] = lonLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(latLength*lonLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	lat_vals = new float[latLength*lonLength]; 
	lon_vals = new float[latLength*lonLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_float(ncid, latid, ptIndex, pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, lonid, ptIndex, pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	for (i=0;i<latLength;i++)
	{
		for (j=0;j<lonLength;j++)
		{
			//if (lat_vals[(latLength-i-1)*lonLength+j]==fill_value)	// this would be an error
			//lat_vals[(latLength-i-1)*lonLength+j]=0.;
			//if (lon_vals[(latLength-i-1)*lonLength+j]==fill_value)
			//lon_vals[(latLength-i-1)*lonLength+j]=0.;
			INDEXH(vertexPtsH,i*lonLength+j).pLat = lat_vals[(latLength-i-1)*lonLength+j];	
			INDEXH(vertexPtsH,i*lonLength+j).pLong = lon_vals[(latLength-i-1)*lonLength+j];
		}
	}
	fVertexPtsH	 = vertexPtsH;
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	if (recs<=0) {strcpy(errmsg,"No times in file. Error opening NetCDF wind file"); err =  -1; goto done;}
	
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
		status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {err = -1; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		//INDEXH(fTimeHdl,i) = startTime2+(long)(timeVal*timeConversion -yearShift*3600.*24.*365.25);	// which start time where?
		//if (i==0) startTime = startTime2+(long)(timeVal*timeConversion -yearShift*3600.*24.*365.25);
		INDEXH(fTimeHdl,i) = newTime-yearShift*3600.*24.*365.25;	// which start time where?
		if (i==0) startTime = newTime-yearShift*3600.*24.*365.25;
	}
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
			//buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
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
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//err = this -> SetInterval(errmsg);
	//if(err) goto done;
	
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	//if (fIsNavy)	// for now don't allow for wind files
	{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto done;}}
	if (!gCommandFileRun)
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
		if (!reply.good)/* return USERCANCEL;*/
		{
			/*if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;*/
			err = dynamic_cast<NetCDFWindMoverCurv *>(this)->ReorderPoints(newMap,errmsg);	
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
			/*if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;*/
			err = dynamic_cast<NetCDFWindMoverCurv *>(this)->ReorderPoints(newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	
	 		/*if (err)*/ goto done;
		}
		
		my_p2cstr(reply.fName);
		
#ifdef MAC
		GetFullPath(reply.vRefNum, 0, (char *)reply.fName, topPath);
#else
		strcpy(topPath, reply.fName);
#endif
#endif		
		strcpy (s, topPath);
		err = ReadTopology(topPath,newMap);	
		goto done;
	}
	
	/*if (recs>0)
		err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
	else {strcpy(errmsg,"No times in file. Error opening NetCDF wind file"); err =  -1;}
	if(err) goto done;*/
	err = dynamic_cast<NetCDFWindMoverCurv *>(this)->ReorderPoints(newMap,errmsg);	
	//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	
	
done:
	if (err)
	{
		printNote("Error opening NetCDF wind file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;	fVertexPtsH	 = 0;}
	}
	
	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (modelTypeStr) delete [] modelTypeStr;
	//if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
	return err;
}


OSErr NetCDFWindMoverCurv::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
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
	
	strcpy(path,fPathName);
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
		if(!wind_uvals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		wind_vvals = new float[latlength*lonlength]; 
		if(!wind_vvals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
		
		angle_vals = new float[latlength*lonlength]; 
		if(!angle_vals) {TechError("GridVel::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
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
		if(!wind_uvals) {TechError("NetCDFWindMoverCurv::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		wind_vvals = new float[latlength*lonlength]; 
		if(!wind_vvals) {TechError("NetCDFWindMoverCurv::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
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
			if (fIsNavy)
			{
				if (wind_uvals[(latlength-i-1)*lonlength+j]==fill_value)
					wind_uvals[(latlength-i-1)*lonlength+j]=0.;
				if (wind_vvals[(latlength-i-1)*lonlength+j]==fill_value)
					wind_vvals[(latlength-i-1)*lonlength+j]=0.;
				u_grid = (float)wind_uvals[(latlength-i-1)*lonlength+j];
				v_grid = (float)wind_vvals[(latlength-i-1)*lonlength+j];
				if (bRotated) angle = angle_vals[(latlength-i-1)*lonlength+j];
				INDEXH(velH,i*lonlength+j).u = u_grid*cos(angle*PI/180.)-v_grid*sin(angle*PI/180.);
				INDEXH(velH,i*lonlength+j).v = u_grid*sin(angle*PI/180.)+v_grid*cos(angle*PI/180.);
			}
			else if (bIsNWSSpeedDirData)
			{
				if (wind_uvals[(latlength-i-1)*lonlength+j]==fill_value)
					wind_uvals[(latlength-i-1)*lonlength+j]=0.;
				if (wind_vvals[(latlength-i-1)*lonlength+j]==fill_value)
					wind_vvals[(latlength-i-1)*lonlength+j]=0.;
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
				if (wind_uvals[(latlength-i-1)*lonlength+j]==fill_value)
					wind_uvals[(latlength-i-1)*lonlength+j]=0.;
				if (wind_vvals[(latlength-i-1)*lonlength+j]==fill_value)
					wind_vvals[(latlength-i-1)*lonlength+j]=0.;
				/////////////////////////////////////////////////
				
				INDEXH(velH,i*lonlength+j).u = /*KNOTSTOMETERSPERSEC**/velConversion*wind_uvals[(latlength-i-1)*lonlength+j];	// need units
				INDEXH(velH,i*lonlength+j).v = /*KNOTSTOMETERSPERSEC**/velConversion*wind_vvals[(latlength-i-1)*lonlength+j];
			}
		}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	
	fWindScale = scale_factor;	// hmm, this forces a reset of scale factor each time, overriding any set by hand
	
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


void NetCDFWindMoverCurv::Draw(Rect r, WorldRect view) 
{	// use for curvilinear
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha/*,depthAlpha*/;
	//float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	
	RGBForeColor(&colors[PURPLE]);
	
	if(bShowArrows || bShowGrid)
	{
		if (bShowGrid) 	// make sure to draw grid even if don't draw arrows
		{
			((TTriGridVel*)fGrid)->DrawCurvGridPts(r,view);
			//return;
		}
		if (bShowArrows)
		{ // we have to draw the arrows
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;
			
			err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
			
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying wind 
			if( (GetNumTimesInFile()>1 || GetNumFiles()>1 )&& loaded && !err)
			{
				// Calculate the time weight factor
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
				//timeAlpha = (endTime - time)/(double)(endTime - startTime);
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationOfWinds)
				{
					timeAlpha = 1;
				}
				else
				{	//return false;
					endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
					timeAlpha = (endTime - time)/(double)(endTime - startTime);
				}
			}
			
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				long ptIndex=-1,iIndex,jIndex;
				WorldPoint wp,wp2;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;				
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				ptIndex = INDEXH(fVerdatToNetCDFH,i);
				iIndex = ptIndex/(fNumCols+1);
				jIndex = ptIndex%(fNumCols+1);
				if (iIndex>0 && jIndex<fNumCols)
					ptIndex = (iIndex-1)*(fNumCols)+jIndex;
				else
					ptIndex = -1;
				
				// for now draw arrow at midpoint of diagonal of gridbox
				// this will result in drawing some arrows more than once
				if (dynamic_cast<NetCDFWindMoverCurv *>(this)->GetLatLonFromIndex(iIndex-1,jIndex+1,&wp2)!=-1)	// may want to get all four points and interpolate
				{
					wp.pLat = (wp.pLat + wp2.pLat)/2.;
					wp.pLong = (wp.pLong + wp2.pLong)/2.;
				}
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);	// should put velocities in center of grid box
				
				// Should check vs fFillValue
				// Check for constant wind 
				if(   ((  GetNumTimesInFile()==1 &&!(GetNumFiles()>1)  ) || timeAlpha == 1) && ptIndex!=-1)
				{
					velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
					velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
				}
				else if (ptIndex!=-1)// time varying wind
				{
					// need to rescale velocities for Navy case, store angle
					velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
					velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
				}
				if ((velocity.u != 0 || velocity.v != 0) && (velocity.u != fFillValue && velocity.v != fFillValue))
				{
					float inchesX = (velocity.u * fWindScale) / fArrowScale;
					float inchesY = (velocity.v * fWindScale) / fArrowScale;
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
	if (bShowGrid) fGrid->Draw(r,view,wayOffMapPt,fWindScale,fArrowScale,0.,false,true,fColor);
	
	RGBForeColor(&colors[BLACK]);
}


//OSErr NetCDFWindMoverCurv::ReadTopology(char* path, TMap **newMap)
OSErr NetCDFWindMoverCurv::ReadTopology(vector<string> &linesInFile, TMap **newMap)
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
	
	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	
	long numWaterBoundaries, numBoundaryPts, numBoundarySegs;
	LONGH boundarySegs=0, waterBoundaries=0, boundaryPts=0;
	
	errmsg[0]=0;
	
	
	/*if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("NetCDFWindMover::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	*/
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	currentLine = linesInFile[line++];
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
	//code goes here, boundary points
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
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts=0;}
	}
	/*if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	 {
	 if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
	 if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs=0;}
	 }*/
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFWindMover::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel*)triGrid;
	
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
			strcpy(errmsg,"An error occurred in NetCDFWindMoverCurv::ReadTopology");
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

OSErr NetCDFWindMoverCurv::ReadTopology(const char *path, TMap **newMap)
{
	vector<string> linesInFile;
	char outPath[kMaxNameLen];
	OSErr err = 0;
	
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
	
	return err;
}

OSErr NetCDFWindMoverCurv::ExportTopology(char* path)
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
		sprintf(topoStr,"%lf\t%lf\n",x,y);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	//code goes here, boundary points - an optional handle, only for curvilinear case
	
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
/////////////////////////////////////////////////
// This is a cross between TWindMover and GridCurMover, built off TWindMover
// The uncertainty is from the wind, the reading, storing, accessing, displaying data is from GridCurMover




