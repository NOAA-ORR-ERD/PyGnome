/*
 *  NetCDFMoverTri.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

/////////////////////////////////////////////////
// Triangular grid code - separate mover, could be derived from NetCDFMoverCurv or vice versa
// read in grid values for first time and set up transformation (dagtree?)
// need to read in lat/lon since won't be evenly spaced

#include "CROSS.H"
#include "NetCDFMoverTri.h"
#include "netcdf.h"
#include "DagTreeIO.h"

NetCDFMoverTri::NetCDFMoverTri (TMap *owner, char *name) : NetCDFMoverCurv(owner, name)
{
	fNumNodes = 0;
	fNumEles = 0;
	bVelocitiesOnTriangles = false;
}

void NetCDFMoverTri::Dispose ()
{
	NetCDFMoverCurv::Dispose ();
}

//#define NetCDFMoverTriREADWRITEVERSION 1 //JLM
#define NetCDFMoverTriREADWRITEVERSION 2 //JLM

OSErr NetCDFMoverTri::Write (BFPB *bfpb)
{
	long i, version = NetCDFMoverTriREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	long numPoints = 0, numPts = 0, index;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFMoverCurv::Write (bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMoverTri::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	
	if (err = WriteMacValue(bfpb, fNumNodes)) goto done;
	if (err = WriteMacValue(bfpb, fNumEles)) goto done;
	if (err = WriteMacValue(bfpb, bVelocitiesOnTriangles)) goto done;
	
	
done:
	if(err)
		TechError("NetCDFMoverTri::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr NetCDFMoverTri::Read(BFPB *bfpb)	
{
	long i, version, index, numPoints;
	ClassID id;
	WorldPointF vertex;
	OSErr err = 0;
	
	if (err = NetCDFMoverCurv::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFMoverTri::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFMoverTri::Read()", "id != TYPE_NETCDFMOVERTRI", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version != NetCDFMoverTriREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	if (err = ReadMacValue(bfpb, &fNumNodes)) goto done;	
	
	if (version>1)
	{
		if (err = ReadMacValue(bfpb, &fNumEles)) goto done;
		if (err = ReadMacValue(bfpb, &bVelocitiesOnTriangles)) goto done;
	}
	
	
done:
	if(err)
	{
		TechError("NetCDFMoverTri::Read(char* path)", " ", 0); 
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFMoverTri::CheckAndPassOnMessage(TModelMessage *message)
{
	return NetCDFMoverCurv::CheckAndPassOnMessage(message); 
}


OSErr NetCDFMoverTri::TextRead(char *path, TMap **newMap, char *topFilePath) 
{
	// needs to be updated once triangle grid format is set
	
	OSErr err = 0;
	long i, numScanned;
	int status, ncid, nodeid, nbndid, bndid, neleid, latid, lonid, recid, timeid, sigmaid, sigmavarid, depthid, nv_varid, nbe_varid;
	int curr_ucmp_id, uv_dimid[3], uv_ndims;
	size_t nodeLength, nbndLength, neleLength, recs, t_len, sigmaLength=0;
	float timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0, *topOrder=0;;	
	WORLDPOINTFH vertexPtsH=0;
	FLOATH totalDepthsH=0, sigmaLevelsH=0;
	float *lat_vals=0,*lon_vals=0,*depth_vals=0, *sigma_vals=0;
	long *bndry_indices=0, *bndry_nums=0, *bndry_type=0, *top_verts=0, *top_neighbors=0;
	static size_t latIndex=0,lonIndex=0,timeIndex,ptIndex=0,bndIndex[2]={0,0};
	static size_t pt_count, bnd_count[2], sigma_count,topIndex[2]={0,0}, top_count[2];
	Seconds startTime, startTime2;
	double timeConversion = 1., scale_factor = 1.;
	char errmsg[256] = "";
	char fileName[64],s[256],topPath[256], outPath[256];
	
	char *modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, bTopInfoInFile = false, isCCW = true;
	
	if (!path || !path[0]) return 0;
	strcpy(fVar.pathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fVar.userName, fileName); // maybe use a name from the file
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) 
	{
#if TARGET_API_MAC_CARBON
		err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
		status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
		if (status != NC_NOERR) {err = -1; goto done;}
	}

	status = nc_inq_dimid(ncid, "time", &recid); 
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
		if (status != NC_NOERR) {err = -1; goto done;}
	}
	
	status = nc_inq_varid(ncid, "time", &timeid); 
	if (status != NC_NOERR) {err = -1; goto done;} 
	
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
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);// recid is the dimension id not the variable id
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
		else if (numScanned<8) // has two extra time entries ??	
			//if (numScanned<8) // has two extra time entries ??	
		{ err = -1; TechError("NetCDFMoverTri::TextRead()", "sscanf() == 8", 0); goto done; }
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
	
	status = nc_inq_dimid(ncid, "node", &nodeid); 
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nodeid, &nodeLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimid(ncid, "nbnd", &nbndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "bnd", &bndid);	
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_dimlen(ncid, nbndid, &nbndLength);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	bnd_count[0] = nbndLength;
	bnd_count[1] = 1;
	bndry_indices = new long[nbndLength]; 
	bndry_nums = new long[nbndLength]; 
	bndry_type = new long[nbndLength]; 
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	bndIndex[1] = 1;	// take second point of boundary segments instead, so that water boundaries work out
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_indices);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 2;
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_nums);
	if (status != NC_NOERR) {err = -1; goto done;}
	bndIndex[1] = 3;
	status = nc_get_vara_long(ncid, bndid, bndIndex, bnd_count, bndry_type);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	//status = nc_inq_dimid(ncid, "nele", &neleid);	
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required
	//status = nc_inq_dimlen(ncid, neleid, &neleLength);
	//if (status != NC_NOERR) {err = -1; goto done;}	// not using these right now so not required
	
	status = nc_inq_dimid(ncid, "sigma", &sigmaid); 	
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "zloc", &sigmaid); 	
		if (status != NC_NOERR) 
		{
			fVar.gridType = TWO_D; /*err = -1; goto done;*/
		}
		else
		{	// might change names to depth rather than sigma here
			status = nc_inq_varid(ncid, "zloc", &sigmavarid); //Navy
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
			if (status != NC_NOERR) {err = -1; goto done;}
			fVar.gridType = MULTILAYER;
			fVar.maxNumDepths = sigmaLength;
			sigma_vals = new float[sigmaLength];
			if (!sigma_vals) {err = memFullErr; goto done;}
			sigma_count = sigmaLength;
			status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
			if (status != NC_NOERR) {err = -1; goto done;}
			fDepthLevelsHdl = (FLOATH)_NewHandleClear(sigmaLength * sizeof(float));
			if (!fDepthLevelsHdl) {err = memFullErr; goto done;}
			for (i=0;i<sigmaLength;i++)
			{
				INDEXH(fDepthLevelsHdl,i) = (float)sigma_vals[i];
			}
			fNumDepthLevels = sigmaLength;	//  here also do we want all depths?
			// once depth is read in 
		}
	}	// check for zgrid option here
	else
	{
		status = nc_inq_varid(ncid, "sigma", &sigmavarid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
		status = nc_inq_dimlen(ncid, sigmaid, &sigmaLength);
		if (status != NC_NOERR) {err = -1; goto done;}
		fVar.gridType = SIGMA;
		fVar.maxNumDepths = sigmaLength;
		sigma_vals = new float[sigmaLength];
		if (!sigma_vals) {err = memFullErr; goto done;}
		sigma_count = sigmaLength;
		status = nc_get_vara_float(ncid, sigmavarid, &ptIndex, &sigma_count, sigma_vals);
		if (status != NC_NOERR) {err = -1; goto done;}
		// once depth is read in 
	}
	
	// option to use index values?
	status = nc_inq_varid(ncid, "lat", &latid);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "lon", &lonid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	pt_count = nodeLength;
	vertexPtsH = (WorldPointF**)_NewHandleClear(nodeLength*sizeof(WorldPointF));
	if (!vertexPtsH) {err = memFullErr; goto done;}
	lat_vals = new float[nodeLength]; 
	lon_vals = new float[nodeLength]; 
	if (!lat_vals || !lon_vals) {err = memFullErr; goto done;}
	status = nc_get_vara_float(ncid, latid, &ptIndex, &pt_count, lat_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, lonid, &ptIndex, &pt_count, lon_vals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	status = nc_inq_varid(ncid, "depth", &depthid);	// this is required for sigma or multilevel grids
	if (status != NC_NOERR) {fVar.gridType = TWO_D;/*err = -1; goto done;*/}
	else
	{	
		totalDepthsH = (FLOATH)_NewHandleClear(nodeLength*sizeof(float));
		if (!totalDepthsH) {err = memFullErr; goto done;}
		depth_vals = new float[nodeLength];
		if (!depth_vals) {err = memFullErr; goto done;}
		status = nc_get_vara_float(ncid, depthid, &ptIndex, &pt_count, depth_vals);
		if (status != NC_NOERR) {err = -1; goto done;}
		
		status = nc_get_att_double(ncid, depthid, "scale_factor", &scale_factor);
		if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
		
	}
	
	for (i=0;i<nodeLength;i++)
	{
		INDEXH(vertexPtsH,i).pLat = lat_vals[i];	
		INDEXH(vertexPtsH,i).pLong = lon_vals[i];
	}
	fVertexPtsH	 = vertexPtsH;// get first and last, lat/lon values, then last-first/total-1 = dlat/dlon
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -1; goto done;}
	fTimeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!fTimeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		//status = nc_get_var1_float(ncid, recid, &timeIndex, &timeVal);	// recid is the dimension id not the variable id
		status = nc_get_var1_float(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -1; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		//newTime = startTime2+timeVal*timeConversion;
		INDEXH(fTimeHdl,i) = newTime;	// which start time where?
		if (i==0) startTime = newTime + fTimeShift;
		//INDEXH(fTimeHdl,i) = startTime2+timeVal*timeConversion;	// which start time where?
		//if (i==0) startTime = startTime2+timeVal*timeConversion + fTimeShift;
	}
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{	// might want to move this so time doesn't get changed if user cancels or there is an error
			short buttonSelected;
			if(!gCommandFileRun)	// also may want to skip for location files...
				buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
			else buttonSelected = 1;	// TAP user doesn't want to see any dialogs, always reset (or maybe never reset? or send message to errorlog?)
			switch(buttonSelected){
				case 1: // reset model start time
					model->SetModelTime(startTime);
					model->SetStartTime(startTime);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
					break;  
				case 3: // don't reset model start time
					break;
				case 4: // cancel
					err=-1;// user cancel
					goto done;
			}
		}
	}
	
	fNumNodes = nodeLength;
	
	// check if file has topology in it
	{
		status = nc_inq_varid(ncid, "nv", &nv_varid); //Navy
		if (status != NC_NOERR) {/*err = -1; goto done;*/}
		else
		{
			status = nc_inq_varid(ncid, "nbe", &nbe_varid); //Navy
			if (status != NC_NOERR) {/*err = -1; goto done;*/}
			else 
			{
				bTopInfoInFile = true;
				status = nc_inq_attlen(ncid, nbe_varid, "order", &t_len);
				topOrder = new char[t_len+1];
				status = nc_get_att_text(ncid, nbe_varid, "order", topOrder);
				if (status != NC_NOERR) {isCCW = false;} // for now to suppport old FVCOM
				topOrder[t_len] = '\0'; 
				if (!strncmpnocase (topOrder, "CW", 2))
					isCCW = false;
				else if (!strncmpnocase (topOrder, "CCW", 3))
					isCCW = true;
				// if order is there let it default to true, that will eventually be default
			}
		}
		if (bTopInfoInFile)
		{
			status = nc_inq_dimid(ncid, "nele", &neleid);	
			if (status != NC_NOERR) {err = -1; goto done;}	
			status = nc_inq_dimlen(ncid, neleid, &neleLength);
			if (status != NC_NOERR) {err = -1; goto done;}	
			fNumEles = neleLength;
			top_verts = new long[neleLength*3]; 
			if (!top_verts ) {err = memFullErr; goto done;}
			top_neighbors = new long[neleLength*3]; 
			if (!top_neighbors ) {err = memFullErr; goto done;}
			top_count[0] = 3;
			top_count[1] = neleLength;
			status = nc_get_vara_long(ncid, nv_varid, topIndex, top_count, top_verts);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_get_vara_long(ncid, nbe_varid, topIndex, top_count, top_neighbors);
			if (status != NC_NOERR) {err = -1; goto done;}
			
			//determine if velocities are on triangles
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
			if (status != NC_NOERR) {err = -1; goto done;}
			status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
			if (status != NC_NOERR) {err = -1; goto done;}
			
			status = nc_inq_vardimid (ncid, curr_ucmp_id, uv_dimid);	// see if dimid(1) or (2) == nele or node, depends on uv_ndims
			if (status==NC_NOERR) 
			{
				if (uv_ndims == 3 && uv_dimid[2] == neleid)
				{bVelocitiesOnTriangles = true;}
				else if (uv_ndims == 2 && uv_dimid[1] == neleid)
				{bVelocitiesOnTriangles = true;}
			}
			
		}
	}
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	if (!bndry_indices || !bndry_nums || !bndry_type) {err = memFullErr; goto done;}
	
	
	{if (topFilePath[0]) {err = ReadTopology(topFilePath,newMap); goto depths;}}
	// look for topology in the file
	// for now ask for an ascii file, output from Topology save option
	// need dialog to ask for file
	if (!bTopFile && !gCommandFileRun)
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
			if (bTopInfoInFile/*bVelocitiesOnTriangles*/)
			{	// code goes here, really this is topology included...
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength,isCCW);	 
				//err = ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);	 
				if (err) goto done;
				goto depths;
			}
			else
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
				//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
				if (err) goto done;
	 			goto depths;
			}
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
			if (bTopInfoInFile/*bVelocitiesOnTriangles*/)	// code goes here, really this is topology included
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength,isCCW);	 
				//err = ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength);		 
				if (err) goto done;
				goto depths;
			}
			else
			{
				err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
				if (err) goto done;	
				goto depths;
			}	
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
		if (err) goto done;
		goto depths;
	}
	
	if (bTopInfoInFile/*bVelocitiesOnTriangles*/)
		err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints2(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength,top_verts,top_neighbors,neleLength,isCCW);	 
	else
		err = dynamic_cast<NetCDFMoverTri *>(this)->ReorderPoints(newMap,bndry_indices,bndry_nums,bndry_type,nbndLength);	 
	
depths:
	if (err) goto done;
	// also translate to fDepthDataInfo and fDepthsH here, using sigma or zgrid info
	
	if (totalDepthsH)
	{
		for (i=0; i<fNumNodes; i++)
		{
			long n;			
			n = i;
			if (n<0 || n>= fNumNodes) {printError("indices messed up"); err=-1; goto done;}
			INDEXH(totalDepthsH,i) = depth_vals[n] * scale_factor;
		}
		//((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}
	
	// CalculateVerticalGrid(sigmaLength,sigmaLevelsH,totalDepthsH);	// maybe multigrid
	{
		long j,index = 0;
		fDepthDataInfo = (DepthDataInfoH)_NewHandle(sizeof(**fDepthDataInfo)*fNumNodes);
		if(!fDepthDataInfo){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
		//if (fVar.gridType==TWO_D || fVar.gridType==MULTILAYER) 
		if (fVar.gridType==TWO_D) 
		{
			if (totalDepthsH) 
			{
				fDepthsH = (FLOATH)_NewHandleClear(nodeLength*sizeof(float));
				if (!fDepthsH) {TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
				for (i=0; i<fNumNodes; i++)
				{
					(*fDepthsH)[i] = (*totalDepthsH)[i];
				}
			}
			//fDepthsH = totalDepthsH;	// may be null, call it barotropic if depths exist??
		}	
		// assign arrays
		else
		{	//TWO_D grid won't need fDepthsH
			fDepthsH = (FLOATH)_NewHandle(sizeof(float)*fNumNodes*fVar.maxNumDepths);
			if(!fDepthsH){TechError("NetCDFMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
			
		}
		// code goes here, if velocities on triangles need to interpolate total depth I think, or use this differently
		for (i=0;i<fNumNodes;i++)
		{
			// might want to order all surface depths, all sigma1, etc., but then indexToDepthData wouldn't work
			// have 2D case, zgrid case as well
			if (fVar.gridType==TWO_D)
			{
				if (totalDepthsH) (*fDepthDataInfo)[i].totalDepth = (*totalDepthsH)[i];
				else (*fDepthDataInfo)[i].totalDepth = -1;	// no depth data
				(*fDepthDataInfo)[i].indexToDepthData = i;
				(*fDepthDataInfo)[i].numDepths = 1;
			}
			/*else if (fVar.gridType==MULTILAYER)
			 {
			 if (totalDepthsH) (*fDepthDataInfo)[i].totalDepth = (*totalDepthsH)[i];
			 else (*fDepthDataInfo)[i].totalDepth = -1;	// no depth data, this should be an error I think
			 (*fDepthDataInfo)[i].indexToDepthData = 0;
			 (*fDepthDataInfo)[i].numDepths = sigmaLength;
			 }*/
			else
			{
				(*fDepthDataInfo)[i].totalDepth = (*totalDepthsH)[i];
				(*fDepthDataInfo)[i].indexToDepthData = index;
				(*fDepthDataInfo)[i].numDepths = sigmaLength;
				for (j=0;j<sigmaLength;j++)
				{
					//(*fDepthsH)[index+j] = (*totalDepthsH)[i]*(1-(*sigmaLevelsH)[j]);
					//if (fVar.gridType==MULTILAYER) (*fDepthsH)[index+j] = (*totalDepthsH)[i]*(j);	// check this
					if (fVar.gridType==MULTILAYER) /*(*fDepthsH)[index+j] = (sigma_vals[j]);*/	// check this, measured from the bottom
						// since depth is measured from bottom should recalculate the depths for each point
					{
						if (( (*totalDepthsH)[i] - sigma_vals[sigmaLength - j - 1]) >= 0) 
							(*fDepthsH)[index+j] = (*totalDepthsH)[i] - sigma_vals[sigmaLength - j - 1] ; 
						else (*fDepthsH)[index+j] = (*totalDepthsH)[i]+1;
					}
					else (*fDepthsH)[index+j] = (*totalDepthsH)[i]*(1-sigma_vals[j]);
					//(*fDepthsH)[j*fNumNodes+i] = totalDepthsH[i]*(1-sigmaLevelsH[j]);
				}
				index+=sigmaLength;
			}
		}
	}
	if (totalDepthsH)	// why is this here twice?
	{
		for (i=0; i<fNumNodes; i++)
		{
			long n = i;
			
			if (fVerdatToNetCDFH) n = INDEXH(fVerdatToNetCDFH,i);
			if (n<0 || n>= fNumNodes) {printError("indices messed up"); err=-1; goto done;}
			INDEXH(totalDepthsH,i) = depth_vals[n] * scale_factor;
		}
		((TTriGridVel3D*)fGrid)->SetDepths(totalDepthsH);
	}
	
done:
	if (err)
	{
		if (!errmsg[0]) 
			strcpy(errmsg,"Error opening NetCDF file");
		printNote(errmsg);
		//printNote("Error opening NetCDF file");
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;	fVertexPtsH	 = 0;}
		if(sigmaLevelsH) {DisposeHandle((Handle)sigmaLevelsH); sigmaLevelsH = 0;}
	}
	//printNote("NetCDF triangular grid model current mover is not yet implemented");
	
	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (depth_vals) delete [] depth_vals;
	if (sigma_vals) delete [] sigma_vals;
	if (bndry_indices) delete [] bndry_indices;
	if (bndry_nums) delete [] bndry_nums;
	if (bndry_type) delete [] bndry_type;
	if (topOrder) delete [] topOrder;
	
	return err;
}


OSErr NetCDFMoverTri::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{	// - needs to be updated once triangle grid format is set
	OSErr err = 0;
	long i,j;
	char path[256], outPath[256]; 
	int status, ncid, numdims, uv_ndims;
	int curr_ucmp_id, curr_vcmp_id, uv_dimid[3], nele_id;
	//static size_t curr_index[] = {0,0,0};
	//static size_t curr_count[3];
	static size_t curr_index[] = {0,0,0,0};
	static size_t curr_count[4];
	float *curr_uvals,*curr_vvals, fill_value, dry_value = 0;
	long totalNumberOfVels = fNumNodes * fVar.maxNumDepths, numVelsAtDepthLevel=0;
	VelocityFH velH = 0;
	long numNodes = fNumNodes;
	long numTris = fNumEles;
	long numDepths = fVar.maxNumDepths;	// assume will always have full set of depths at each point for now
	double scale_factor = 1.;
	
	errmsg[0]=0;
	
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
	status = nc_inq_ndims(ncid, &numdims);	// in general it's not the total number of dimensions but the number the variable depends on
	if (status != NC_NOERR) {err = -1; goto done;}
	
	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time
	//curr_count[1] = 1;	// depth
	//curr_count[2] = numNodes;
	
	// check for sigma or zgrid dimension
	if (numdims>=6)	// should check what the dimensions are
	{
		//curr_count[1] = 1;	// depth
		curr_count[1] = numDepths;	// depth
		//curr_count[1] = depthlength;	// depth
		curr_count[2] = numNodes;
	}
	else
	{
		curr_count[1] = numNodes;	
	}
	status = nc_inq_varid(ncid, "u", &curr_ucmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varid(ncid, "v", &curr_vcmp_id);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
	if (status==NC_NOERR){if (numdims < 6 && uv_ndims==3) {curr_count[1] = numDepths; curr_count[2] = numNodes;}}	// could have more dimensions than are used in u,v
	if (status==NC_NOERR){if (numdims >= 6 && uv_ndims==2) {curr_count[1] = numNodes;}}	// could have more dimensions than are used in u,v
	
	status = nc_inq_vardimid (ncid, curr_ucmp_id, uv_dimid);	// see if dimid(1) or (2) == nele or node, depends on uv_ndims
	if (status==NC_NOERR) 
	{
		status = nc_inq_dimid (ncid, "nele", &nele_id);
		if (status==NC_NOERR)
		{
			if (uv_ndims == 3 && uv_dimid[2] == nele_id)
			{bVelocitiesOnTriangles = true; curr_count[2] = numTris;}
			else if (uv_ndims == 2 && uv_dimid[1] == nele_id)
			{bVelocitiesOnTriangles = true; curr_count[1] = numTris;}
		}
	}
	if (bVelocitiesOnTriangles) 
	{
		totalNumberOfVels = numTris * fVar.maxNumDepths;
		numVelsAtDepthLevel = numTris;
	}
	else
		numVelsAtDepthLevel = numNodes;
	//curr_uvals = new float[numNodes]; 
	curr_uvals = new float[totalNumberOfVels]; 
	if(!curr_uvals) {TechError("NetCDFMoverTri::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	//curr_vvals = new float[numNodes]; 
	curr_vvals = new float[totalNumberOfVels]; 
	if(!curr_vvals) {TechError("NetCDFMoverTri::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	
	status = nc_get_vara_float(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_float(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_att_float(ncid, curr_ucmp_id, "_FillValue", &fill_value);// missing_value vs _FillValue
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_att_float(ncid, curr_ucmp_id, "missing_value", &fill_value);// missing_value vs _FillValue
	if (status != NC_NOERR) {/*err = -1; goto done;*/fill_value=-9999.;}
	status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
	//if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_att_float(ncid, curr_ucmp_id, "dry_value", &dry_value);// missing_value vs _FillValue
	if (status != NC_NOERR) {/*err = -1; goto done;*/}  
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec));
	if (!velH) {err = memFullErr; goto done;}
	for (j=0;j<numDepths;j++)
	{
		//for (i=0;i<totalNumberOfVels;i++)
		for (i=0;i<numVelsAtDepthLevel;i++)
			//for (i=0;i<numNodes;i++)
		{
			// really need to store the fill_value data and check for it when moving or drawing
			/*if (curr_uvals[i]==0.||curr_vvals[i]==0.)
			 curr_uvals[i] = curr_vvals[i] = 1e-06;
			 if (curr_uvals[i]==fill_value)
			 curr_uvals[i]=0.;
			 if (curr_vvals[i]==fill_value)
			 curr_vvals[i]=0.;
			 // for now until we decide what to do with the dry value flag
			 if (curr_uvals[i]==dry_value)
			 curr_uvals[i]=0.;
			 if (curr_vvals[i]==dry_value)
			 curr_vvals[i]=0.;
			 INDEXH(velH,i).u = curr_uvals[i];	// need units
			 INDEXH(velH,i).v = curr_vvals[i];*/
			/*if (curr_uvals[j*fNumNodes+i]==0.||curr_vvals[j*fNumNodes+i]==0.)
			 curr_uvals[j*fNumNodes+i] = curr_vvals[j*fNumNodes+i] = 1e-06;
			 if (curr_uvals[j*fNumNodes+i]==fill_value)
			 curr_uvals[j*fNumNodes+i]=0.;
			 if (curr_vvals[j*fNumNodes+i]==fill_value)
			 curr_vvals[j*fNumNodes+i]=0.;*/
			if (curr_uvals[j*numVelsAtDepthLevel+i]==0.||curr_vvals[j*numVelsAtDepthLevel+i]==0.)
				curr_uvals[j*numVelsAtDepthLevel+i] = curr_vvals[j*numVelsAtDepthLevel+i] = 1e-06;
			if (curr_uvals[j*numVelsAtDepthLevel+i]==fill_value)
				curr_uvals[j*numVelsAtDepthLevel+i]=0.;
			if (curr_vvals[j*numVelsAtDepthLevel+i]==fill_value)
				curr_vvals[j*numVelsAtDepthLevel+i]=0.;
			//if (fVar.gridType==MULTILAYER /*sigmaReversed*/)
			/*{
			 INDEXH(velH,(numDepths-j-1)*fNumNodes+i).u = curr_uvals[j*fNumNodes+i];	// need units
			 INDEXH(velH,(numDepths-j-1)*fNumNodes+i).v = curr_vvals[j*fNumNodes+i];	// also need to reverse top to bottom (if sigma is reversed...)
			 }
			 else*/
			{
				//INDEXH(velH,i*numDepths+(numDepths-j-1)).u = curr_uvals[j*fNumNodes+i];	// need units
				//INDEXH(velH,i*numDepths+(numDepths-j-1)).v = curr_vvals[j*fNumNodes+i];	// also need to reverse top to bottom
				INDEXH(velH,i*numDepths+(numDepths-j-1)).u = curr_uvals[j*numVelsAtDepthLevel+i];	// need units
				INDEXH(velH,i*numDepths+(numDepths-j-1)).v = curr_vvals[j*numVelsAtDepthLevel+i];	// also need to reverse top to bottom
			}
		}
	}
	*velocityH = velH;
	fFillValue = fill_value;
	if (scale_factor!=1.) fFileScaleFactor = scale_factor;
	
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

void NetCDFMoverTri::Draw(Rect r, WorldRect view) 
{	// will need to update once triangle format is set
	WorldPoint wayOffMapPt = {-200*1000000,-200*1000000};
	double timeAlpha,depthAlpha;
	float topDepth,bottomDepth;
	Point p;
	Rect c;
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	OSErr err = 0;
	char errmsg[256];
	long amtOfDepthData = 0;
	
	RGBForeColor(&colors[PURPLE]);
	
	if(fDepthDataInfo) amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	
	if(fGrid && (fVar.bShowArrows || fVar.bShowGrid))
	{
		Boolean overrideDrawArrows = FALSE;
		fGrid->Draw(r,view,wayOffMapPt,fVar.curScale,fVar.arrowScale,fVar.arrowDepth,overrideDrawArrows,fVar.bShowGrid,fColor);
		if(fVar.bShowArrows && bVelocitiesOnTriangles == false)
		{ // we have to draw the arrows
			RGBForeColor(&fColor);
			long numVertices,i;
			LongPointHdl ptsHdl = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't need 3D stuff to draw here
			
//			err = this -> SetInterval(errmsg);	// minus AH 07/17/2012
			//err = this -> SetInterval(errmsg, model->GetStartTime(), model->GetModelTime()); // AH 07/17/2012
			err = this -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
			
			if(err) return;
			
//			loaded = this -> CheckInterval(timeDataInterval);	// minus AH 07/17/2012
			//loaded = this -> CheckInterval(timeDataInterval, model->GetStartTime(), model->GetModelTime());	// AH 07/17/2012
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
				//if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
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
			
			for(i = 0; i < numVertices; i++)
			{
			 	// get the value at each vertex and draw an arrow
				LongPoint pt = INDEXH(ptsHdl,i);
				//long ptIndex = INDEXH(fVerdatToNetCDFH,i);
				long index = i;
				if (fVerdatToNetCDFH) index = INDEXH(fVerdatToNetCDFH,i);
				//long ptIndex = (*fDepthDataInfo)[index].indexToDepthData;	// not used ?
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case
				
				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				//amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 			if (amtOfDepthData>0)
				{
					dynamic_cast<NetCDFMoverTri *>(this)->GetDepthIndices(index,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					depthIndex1 = index;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}
				
				/*if (fVar.gridType == MULTILAYER)
				 {
				 if (fDepthLevelsHdl) 
				 {
				 topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
				 bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
				 depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				 }
				 //else //this should be an error
				 depthIndex1 = index + depthIndex1*fNumNodes;
				 if (depthIndex2!=UNASSIGNEDINDEX) depthIndex2 = index + depthIndex2*fNumNodes;
				 }*/
				
				wp.pLat = pt.v;
				wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						//velocity.u = INDEXH(fStartData.dataHdl,ptIndex).u;
						//velocity.v = INDEXH(fStartData.dataHdl,ptIndex).v;
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
						//velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).u;
						//velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,ptIndex).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,ptIndex).v;
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
				if ((velocity.u != 0 || velocity.v != 0))
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
		else if (fVar.bShowArrows && bVelocitiesOnTriangles)
		{ // we have to draw the arrows
			short row, col, pixX, pixY;
			float inchesX, inchesY;
			Point p, p2;
			Rect c;
			WorldPoint wp;
			VelocityRec velocity;
			LongPoint wp1,wp2,wp3;
			Boolean offQuickDrawPlane = false;
			long numVertices,i,numTri;
			LongPointHdl ptsHdl = 0;
			TopologyHdl topH = 0;
			long timeDataInterval;
			Boolean loaded;
			TTriGridVel* triGrid = (TTriGridVel*)fGrid;	// don't need 3D stuff to draw here
			RGBForeColor(&fColor);
		
			err = this -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
			
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
			
			if(!loaded) return;
			
			ptsHdl = triGrid -> GetPointsHdl();
			if(ptsHdl)
				numVertices = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
			else 
				numVertices = 0;
			
			topH = triGrid -> GetTopologyHdl();
			if (topH)
				numTri = _GetHandleSize((Handle)topH)/sizeof(Topology);
			else 
				numTri = 0;
			
			// Check for time varying current 
			if(GetNumTimesInFile()>1 || GetNumFiles()>1)
				//if(GetNumTimesInFile()>1)
			{
				// Calculate the time weight factor
				if (GetNumFiles()>1 && fOverLap)
					startTime = fOverLapStartTime + fTimeShift;
				else
					startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				//startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
				if (fEndData.timeIndex == UNASSIGNEDINDEX && (time > startTime || time < startTime) && fAllowExtrapolationOfCurrentsInTime)
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
			
			//for(i = 0; i < numVertices; i++)
			for(i = 0; i < numTri; i++)
			{
			 	// get the value at each vertex and draw an arrow
				//LongPoint pt = INDEXH(ptsHdl,i);
				//long index = INDEXH(fVerdatToNetCDFH,i);
				WorldPoint wp;
				Point p,p2;
				VelocityRec velocity = {0.,0.};
				Boolean offQuickDrawPlane = false;
				long depthIndex1,depthIndex2;	// default to -1?, eventually use in surface velocity case
				
				//GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				//amtOfDepthData = _GetHandleSize((Handle)fDepthDataInfo)/sizeof(**fDepthDataInfo);
	 			if (amtOfDepthData>0 && !bVelocitiesOnTriangles)	// for now, will have to figure out how depth data is handled
				{
					//GetDepthIndices(index,fVar.arrowDepth,&depthIndex1,&depthIndex2);
					dynamic_cast<NetCDFMoverTri *>(this)->GetDepthIndices(i,fVar.arrowDepth,&depthIndex1,&depthIndex2);
				}
				else
				{	// for old SAV files without fDepthDataInfo
					//depthIndex1 = index;
					depthIndex1 = i;
					depthIndex2 = -1;
				}
				
				if (depthIndex1==UNASSIGNEDINDEX && depthIndex2==UNASSIGNEDINDEX)
					continue;	// no value for this point at chosen depth
				
				if (depthIndex2!=UNASSIGNEDINDEX)
				{
					// Calculate the depth weight factor
					topDepth = INDEXH(fDepthsH,depthIndex1);
					bottomDepth = INDEXH(fDepthsH,depthIndex2);
					depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				}
				
				wp1 = (*ptsHdl)[(*topH)[i].vertex1];
				wp2 = (*ptsHdl)[(*topH)[i].vertex2];
				wp3 = (*ptsHdl)[(*topH)[i].vertex3];
				
				wp.pLong = (wp1.h+wp2.h+wp3.h)/3;
				wp.pLat = (wp1.v+wp2.v+wp3.v)/3;
				//velocity = GetPatValue(wp);
				
				/*if (fVar.gridType == MULTILAYER)
				 {
				 if (fDepthLevelsHdl) 
				 {
				 topDepth = INDEXH(fDepthLevelsHdl,depthIndex1);
				 bottomDepth = INDEXH(fDepthLevelsHdl,depthIndex2);
				 depthAlpha = (bottomDepth - fVar.arrowDepth)/(double)(bottomDepth - topDepth);
				 }
				 //else //this should be an error
				 depthIndex1 = index + depthIndex1*fNumNodes;
				 if (depthIndex2!=UNASSIGNEDINDEX) depthIndex2 = index + depthIndex2*fNumNodes;
				 }*/
				
				//wp.pLat = pt.v;
				//wp.pLong = pt.h;
				
				p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
				
				// Check for constant current 
				if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
				{
					if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
					{
						velocity.u = INDEXH(fStartData.dataHdl,i).u;
						velocity.v = INDEXH(fStartData.dataHdl,i).v;
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
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,i).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,i).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,i).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,i).v;
					}
					else	// below surface velocity
					{
						velocity.u = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).u);
						velocity.u += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).u);
						velocity.v = depthAlpha*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex1).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex1).v);
						velocity.v += (1-depthAlpha)*(timeAlpha*INDEXH(fStartData.dataHdl,depthIndex2).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,depthIndex2).v);
					}
				}
				if ((velocity.u != 0 || velocity.v != 0))
				{
					inchesX = (velocity.u * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
					inchesY = (velocity.v * fVar.curScale * fFileScaleFactor) / fVar.arrowScale;
					pixX = inchesX * PixelsPerInchCurrent();
					pixY = inchesY * PixelsPerInchCurrent();
					p2.h = p.h + pixX;
					p2.v = p.v - pixY;
					MyMoveTo(p.h, p.v);
					MyLineTo(p2.h, p2.v);
					MyDrawArrow(p.h,p.v,p2.h,p2.v);
				}
			}
		}
	}
	if (bShowDepthContours && fVar.gridType!=TWO_D) ((TTriGridVel3D*)fGrid)->DrawDepthContours(r,view,bShowDepthContourLabels);
	
	RGBForeColor(&colors[BLACK]);
}

//OSErr NetCDFMoverTri::ReadTopology(char* path, TMap **newMap)
OSErr NetCDFMoverTri::ReadTopology(vector<string> &linesInFile, TMap **newMap)
{
	// import NetCDF triangle info so don't have to regenerate
	// this is same as curvilinear mover so may want to combine later
	char s[1024], errmsg[256];
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
		TechError("NetCDFMoverTri::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	*/
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	currentLine = linesInFile[line++];
	//if(IsTransposeArrayHeaderLine(s,&numPts)) 
	if(IsTransposeArrayHeaderLine(currentLine,numPts)) 
	{
		//if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		if (err = ReadTransposeArray(linesInFile,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else 
	//{err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
	{
		//if (!bVelocitiesOnTriangles) {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}
		//else line--;
		line--;
	}
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
		currentLine = linesInFile[line++];
		//NthLineInTextOptimized(*f, (line)++, s, 1024); 
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
		//strcpy(errmsg,"Error in Boundary points header line");
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
	
	// check if bVelocitiesOnTriangles and boundaryPts
	if (waterBoundaries && boundarySegs && (this -> moverMap == model -> uMap))
	{
		//PtCurMap *map = CreateAndInitPtCurMap(fVar.userName,bounds); // the map bounds are the same as the grid bounds
		PtCurMap *map = CreateAndInitPtCurMap("Extended Topology",bounds); // the map bounds are the same as the grid bounds
		if (!map) {strcpy(errmsg,"Error creating ptcur map"); goto done;}
		// maybe move up and have the map read in the boundary information
		map->SetBoundarySegs(boundarySegs);	
		map->SetWaterBoundaries(waterBoundaries);
		//if (bVelocitiesOnTriangles && boundaryPts) map->SetBoundaryPoints(boundaryPts);	
		if (boundaryPts) map->SetBoundaryPoints(boundaryPts);	
		
		*newMap = map;
	}
	
	//if (!(this -> moverMap == model -> uMap))	// maybe assume rectangle grids will have map?
	else	// maybe assume rectangle grids will have map?
	{
		if (waterBoundaries) {DisposeHandle((Handle)waterBoundaries); waterBoundaries=0;}
		if (boundarySegs) {DisposeHandle((Handle)boundarySegs); boundarySegs = 0;}
		if (boundaryPts) {DisposeHandle((Handle)boundaryPts); boundaryPts = 0;}
	}
	
	/////////////////////////////////////////////////
	
	
	triGrid = new TTriGridVel3D;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFMoverTri::ReadTopology()","new TTriGridVel" ,err);
		goto done;
	}
	
	fGrid = (TTriGridVel3D*)triGrid;
	
	triGrid -> SetBounds(bounds); 
	
	dagTree = new TDagTree(pts,topo,tree.treeHdl,velH,tree.numBranches); 
	if(!dagTree)
	{
		printError("Unable to read Extended Topology file.");
		goto done;
	}
	
	triGrid -> SetDagTree(dagTree);
	//triGrid -> SetDepths(depths);
	
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
			strcpy(errmsg,"An error occurred in NetCDFMoverTri::ReadTopology");
		printError(errmsg); 
		if(pts) {DisposeHandle((Handle)pts); pts=0;}
		if(topo) {DisposeHandle((Handle)topo); topo=0;}
		if(velH) {DisposeHandle((Handle)velH); velH=0;}
		if(tree.treeHdl) {DisposeHandle((Handle)tree.treeHdl); tree.treeHdl=0;}
		if(depths) {DisposeHandle((Handle)depths); depths=0;}
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

OSErr NetCDFMoverTri::ReadTopology(const char *path, TMap **newMap)
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


/////////////////////////////////////////////////
PtCurMap* GetPtCurMap(void)
{
	long i,n;
	TMap *map;
	PtCurMap *ptCurMap = 0;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_COMPOUNDMAP))
		{
			return dynamic_cast<PtCurMap *>(map);
		} 
		if (map->IAm(TYPE_PTCURMAP)) 
		{
			ptCurMap = dynamic_cast<PtCurMap *>(map);
			return ptCurMap;
			//return (PtCurMap*)map;
		}
	}
	return nil;
}
TMap* Get3DMap(void)
{
	long i,n;
	TMap *map;
	n = model -> mapList->GetItemCount() ;
	for (i=0; i<n; i++)
	{
		model -> mapList->GetListItem((Ptr)&map, i);
		if (map->IAm(TYPE_COMPOUNDMAP))
		{
			return map;
		} 
		if (map->IAm(TYPE_PTCURMAP)) 
		{
			return map;
		}
		if (map->IAm(TYPE_MAP3D)) 
		{
			return map;
		}
	}
	return nil;
}

OSErr NetCDFMoverTri::ExportTopology(char* path)
{
	// export NetCDF triangle info so don't have to regenerate each time
	// same as curvilinear so may want to combine at some point
	OSErr err = 0;
	long numTriangles, numBranches, nver, nBoundarySegs=0, nWaterBoundaries=0, nBoundaryPts;
	long i, n=0, v1,v2,v3,n1,n2,n3;
	double x,y,z=0;
	char buffer[512],hdrStr[64],topoStr[128];
	TopologyHdl topH=0;
	TTriGridVel* triGrid = 0;	
	TDagTree* dagTree = 0;
	LongPointHdl ptsH=0;
	FLOATH depthsH=0;
	DAGHdl treeH = 0;
	LONGH	boundarySegmentsH = 0, boundaryTypeH = 0, boundaryPointsH = 0;
	BFPB bfpb;
	PtCurMap *map = GetPtCurMap();
	
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
	
	//if (moverMap->IAm(TYPE_PTCURMAP))
	if (map)
	{
		//boundaryTypeH = ((PtCurMap*)moverMap)->GetWaterBoundaries();
		//boundarySegmentsH = ((PtCurMap*)moverMap)->GetBoundarySegs();
		boundaryTypeH = map->GetWaterBoundaries();
		boundarySegmentsH = map->GetBoundarySegs();
		if (!boundaryTypeH || !boundarySegmentsH) {printError("No map info to export"); err=-1; goto done;}
		/*if (bVelocitiesOnTriangles) 
		{
			boundaryPointsH = map->GetBoundaryPoints();
			if (!boundaryPointsH) {printError("No map info to export"); err=-1; goto done;}
		}*/
		boundaryPointsH = map->GetBoundaryPoints();
	}
	
	(void)hdelete(0, 0, path);
	if (err = hcreate(0, 0, path, 'ttxt', 'TEXT'))
	{ printError("1"); TechError("WriteToPath()", "hcreate()", err); return err; }
	if (err = FSOpenBuf(0, 0, path, &bfpb, 100000, FALSE))
	{ printError("2"); TechError("WriteToPath()", "FSOpenBuf()", err); return err; }
	
	
	// Write out values
	if (fVerdatToNetCDFH) n = _GetHandleSize((Handle)fVerdatToNetCDFH)/sizeof(long);
	else n = 0;
	//else {printError("There is no transpose array"); err = -1; goto done;}
	//else 
		//{if (!bVelocitiesOnTriangles) {printError("There is no transpose array"); err = -1; goto done;}}
	//if (!bVelocitiesOnTriangles)
	if (n>0)
	{
		sprintf(hdrStr,"TransposeArray\t%ld\n",n);	
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<n;i++)
		{	
			sprintf(topoStr,"%ld\n",(*fVerdatToNetCDFH)[i]);
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
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
		if (depthsH) 
		{
			z = (*depthsH)[i];
			sprintf(topoStr,"%lf\t%lf\t%lf\n",x,y,z);
		}
		else
			sprintf(topoStr,"%lf\t%lf\n",x,y);
		strcpy(buffer,topoStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
	}
	
	if (boundarySegmentsH) 
	{
		nBoundarySegs = _GetHandleSize((Handle)boundarySegmentsH)/sizeof(long);
		//fprintf(outfile,"Vertices\t%ld\t%ld\n",nver,numBoundaryPts);	// total vertices and number of boundary points
		sprintf(hdrStr,"BoundarySegments\t%ld\n",nBoundarySegs);	// total vertices
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundarySegs;i++)
		{	
			//sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]); // when reading in subtracts 1
			sprintf(topoStr,"%ld\n",(*boundarySegmentsH)[i]+1);
			strcpy(buffer,topoStr);
			if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		}
	}
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
		nBoundaryPts = _GetHandleSize((Handle)boundaryPointsH)/sizeof(long);	
		sprintf(hdrStr,"BoundaryPoints\t%ld\n",nBoundaryPts);	// total boundary points
		strcpy(buffer,hdrStr);
		if (err = WriteMacValue(&bfpb, buffer, strlen(buffer))) goto done;
		for(i=0;i<nBoundaryPts;i++)
		{	
			sprintf(topoStr,"%ld\n",(*boundaryPointsH)[i]);	
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
