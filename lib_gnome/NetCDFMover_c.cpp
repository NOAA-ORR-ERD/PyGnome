/*
 *  NetCDFMover_c.cpp
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

#include "NetCDFMover_c.h"
#include "netcdf.h"
#include "CompFunctions.h"
#include "StringFunctions.h"
#include <math.h>
#include <float.h>

using std::cout;

#ifndef pyGNOME
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
	
	fFileScaleFactor = 1.;	// let user set a scale factor in addition to what is in the file
	
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
#endif


OSErr NetCDFMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
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

long NetCDFMover_c::GetNumTimesInFile()
{
	long numTimes = 0;
	
	if (fTimeHdl) numTimes = _GetHandleSize((Handle)fTimeHdl)/sizeof(**fTimeHdl);
	return numTimes;     
}

long NetCDFMover_c::GetNumFiles()
{
	long numFiles = 0;
	
	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}


OSErr NetCDFMover_c::PrepareForModelRun()
{
#ifndef pyGNOME
	if (dynamic_cast<NetCDFMover *>(this)->IAm(TYPE_NETCDFMOVERCURV) || dynamic_cast<NetCDFMover *>(this)->IAm(TYPE_NETCDFMOVERTRI))
	{
		//PtCurMap* ptCurMap = (PtCurMap*)moverMap;
		//PtCurMap* ptCurMap = GetPtCurMap();
		//if (ptCurMap)
		if (moverMap->IAm(TYPE_PTCURMAP))
		{
			(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
			(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
			if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)
				(dynamic_cast<TTriGridVel3D*>(fGrid))->ClearOutputHandles();
		}
	}
#endif
	return CurrentMover_c::PrepareForModelRun();
}


OSErr NetCDFMover_c::PrepareForModelStep(const Seconds& model_time, const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	long timeDataInterval;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	if (bIsFirstStep)
		fModelStartTime = model_time;
	
	if (!bActive) return noErr;
	
	err = dynamic_cast<NetCDFMover *>(this) -> SetInterval(errmsg, model_time); // AH 07/17/2012
	
	if (err) goto done;
	
	if (uncertain)
	{
		Seconds elapsed_time = model_time - fModelStartTime;	// code goes here, how to set start time
		err = this->UpdateUncertainty(elapsed_time, numLESets, LESetsSizesList);
	}
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


Boolean NetCDFMover_c::CheckInterval(long &timeDataInterval, const Seconds& model_time)
{
	Seconds time =  model_time, startTime, endTime;	
	
	long i,numTimes,numFiles = GetNumFiles();
	
	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists
	
	// check for constant current
	if (numTimes==1 && !(GetNumFiles()>1)) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
			return true;
		else
			return false;
	}
	
	if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
	{
		if (time>=((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && time<=((*fTimeHdl)[fEndData.timeIndex] + fTimeShift))
		{	// we already have the right interval loaded
			timeDataInterval = fEndData.timeIndex;
			return true;
		}
	}
	
	if (GetNumFiles()>1 && fOverLap)
	{	
		if (time>=fOverLapStartTime + fTimeShift && time<=(*fTimeHdl)[fEndData.timeIndex] + fTimeShift)
			return true;	// we already have the right interval loaded, time is in between two files
		else fOverLap = false;
	}
	
	//for (i=0;i<numTimes;i++) 
	for (i=0;i<numTimes-1;i++) 
	{	// find the time interval
		if (time>=((*fTimeHdl)[i] + fTimeShift) && time<=((*fTimeHdl)[i+1] + fTimeShift))
		{
			timeDataInterval = i+1; // first interval is between 0 and 1, and so on
			return false;
		}
	}	
	// don't allow time before first or after last
	if (time<((*fTimeHdl)[0] + fTimeShift)) 
	{
		// if number of files > 1 check that first is the one loaded
		timeDataInterval = 0;
		if (numFiles > 0)
		{
			//startTime = (*fInputFilesHdl)[0].startTime + fTimeShift;
			startTime = (*fInputFilesHdl)[0].startTime;
			if ((*fTimeHdl)[0] != startTime)
				return false;
		}
		if (fAllowExtrapolationOfCurrentsInTime && fEndData.timeIndex == UNASSIGNEDINDEX && !(fStartData.timeIndex == UNASSIGNEDINDEX))	// way to recognize last interval is set
		{
			//check if time > last model time in all files
			//timeDataInterval = 1;
			return true;
		}
	}
	if (time>((*fTimeHdl)[numTimes-1] + fTimeShift) )
		// code goes here, check if this is last time in all files and user has set flag to continue
		// then if last time is loaded as start time and nothing as end time this is right interval
	{
		// if number of files > 1 check that last is the one loaded
		timeDataInterval = numTimes;
		if (numFiles > 0)
		{
			//endTime = (*fInputFilesHdl)[numFiles-1].endTime + fTimeShift;
			endTime = (*fInputFilesHdl)[numFiles-1].endTime;
			if ((*fTimeHdl)[numTimes-1] != endTime)
				return false;
		}
		if (fAllowExtrapolationOfCurrentsInTime && fEndData.timeIndex == UNASSIGNEDINDEX && !(fStartData.timeIndex == UNASSIGNEDINDEX))	// way to recognize last interval is set
		{
			//check if time > last model time in all files
			return true;
		}
	}
	return false;
	
}

void NetCDFMover_c::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void NetCDFMover_c::DisposeAllLoadedData()
{
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);
}

void NetCDFMover_c::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}



long NetCDFMover_c::GetNumDepthLevelsInFile()
{
	long numDepthLevels = 0;
	
	if (fDepthLevelsHdl) numDepthLevels = _GetHandleSize((Handle)fDepthLevelsHdl)/sizeof(**fDepthLevelsHdl);
	return numDepthLevels;     
}



OSErr NetCDFMover_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	OSErr err = 0;
	long i,j,k;
	char path[256], outPath[256]; 
	char *velUnits=0; 
	int status, ncid, numdims, uv_ndims, numvars;
	int curr_ucmp_id, curr_vcmp_id, depthid;
	static size_t curr_index[] = {0,0,0,0};
	static size_t curr_count[4];
	size_t velunit_len;
	double *curr_uvals=0,*curr_vvals=0, fill_value, velConversion=1.;
	long totalNumberOfVels = fNumRows * fNumCols;
	VelocityFH velH = 0;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	long depthlength = fNumDepthLevels;	// code goes here, do we want all depths? maybe if movermap is a ptcur map??
	//long depthlength = 1;	// code goes here, do we want all depths?
	double scale_factor = 1.;
	Boolean bDepthIncluded = false;
	
	errmsg[0]=0;
	
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
	
	if (numdims>=4)
	{	// code goes here, do we really want to use all the depths - for big files Gnome can't handle it
		status = nc_inq_dimid(ncid, "depth", &depthid);	//3D
		if (status != NC_NOERR) 
		{
			status = nc_inq_dimid(ncid, "sigma", &depthid);	//3D - need to check sigma values in TextRead...
			if (status != NC_NOERR) bDepthIncluded = false;
			else bDepthIncluded = true;
		}
		else bDepthIncluded = true;
		// code goes here, might want to check other dimensions (lev), or just how many dimensions uv depend on
	}
	
	curr_index[0] = index;	// time 
	curr_count[0] = 1;	// take one at a time

	if (bDepthIncluded)
	{
		if (moverMap->IAm(TYPE_PTCURMAP)) depthlength = fNumDepthLevels;
		//curr_count[1] = 1;	// depth
		curr_count[1] = depthlength;	// depth
		curr_count[2] = latlength;
		curr_count[3] = lonlength;
	}
	else
	{
		curr_count[1] = latlength;	
		curr_count[2] = lonlength;
	}
	
	curr_uvals = new double[latlength*lonlength*depthlength]; 
	if(!curr_uvals) {TechError("NetCDFMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	curr_vvals = new double[latlength*lonlength*depthlength]; 
	if(!curr_vvals) {TechError("NetCDFMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
	
	status = nc_inq_varid(ncid, "water_u", &curr_ucmp_id);
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "curr_ucmp", &curr_ucmp_id); 
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "u", &curr_ucmp_id); // allow u,v since so many people get confused
			if (status != NC_NOERR) {status = nc_inq_varid(ncid, "U", &curr_ucmp_id); if (status != NC_NOERR)	// ferret uses CAPS
			{err = -1; goto LAS;}}	// broader check for variable names coming out of LAS
		}	
	}
	status = nc_inq_varid(ncid, "water_v", &curr_vcmp_id);	// what if only input one at a time (u,v separate movers)?
	if (status != NC_NOERR) 
	{
		status = nc_inq_varid(ncid, "curr_vcmp", &curr_vcmp_id); 
		if (status != NC_NOERR) 
		{
			status = nc_inq_varid(ncid, "v", &curr_vcmp_id); // allow u,v since so many people get confused
			if (status != NC_NOERR) {status = nc_inq_varid(ncid, "V", &curr_vcmp_id); if (status != NC_NOERR)	// ferret uses CAPS
			{err = -1; goto done;}}
		}	
	}
	
LAS:
	if (err)
	{
		Boolean bLASStyleNames = false;
		char uname[NC_MAX_NAME],vname[NC_MAX_NAME],levname[NC_MAX_NAME],varname[NC_MAX_NAME];
		err = 0;
		status = nc_inq_nvars(ncid, &numvars);
		if (status != NC_NOERR) {err = -1; goto done;}
		for (i=0;i<numvars;i++)
		{
			//if (i == recid) continue;
			status = nc_inq_varname(ncid,i,varname);
			if (status != NC_NOERR) {err = -1; goto done;}
			if (varname[0]=='U' || varname[0]=='u' || strstrnocase(varname,"EVEL"))	// careful here, could end up with wrong u variable (like u_wind for example)
			{
				curr_ucmp_id = i; bLASStyleNames = true;
				strcpy(uname,varname);
			}
			if (varname[0]=='V' || varname[0]=='v' || strstrnocase(varname,"NVEL"))
			{
				curr_vcmp_id = i; bLASStyleNames = true;
				strcpy(vname,varname);
			}
			if (strstrnocase(varname,"LEV"))
			{
				depthid = i; bDepthIncluded = true;
				strcpy(levname,varname);
				curr_count[1] = depthlength;	// depth (set to 1)
				curr_count[2] = latlength;
				curr_count[3] = lonlength;
			}
		}
		if (!bLASStyleNames){err = -1; goto done;}
	}
	
	
	status = nc_inq_varndims(ncid, curr_ucmp_id, &uv_ndims);
	if (status==NC_NOERR){if (uv_ndims < numdims && uv_ndims==3) {curr_count[1] = latlength; curr_count[2] = lonlength;}}	// could have more dimensions than are used in u,v
	if (uv_ndims==4) {curr_count[1] = depthlength;curr_count[2] = latlength;curr_count[3] = lonlength;}
	status = nc_get_vara_double(ncid, curr_ucmp_id, curr_index, curr_count, curr_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	status = nc_get_vara_double(ncid, curr_vcmp_id, curr_index, curr_count, curr_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	
	status = nc_inq_attlen(ncid, curr_ucmp_id, "units", &velunit_len);
	if (status == NC_NOERR)
	{
		velUnits = new char[velunit_len+1];
		status = nc_get_att_text(ncid, curr_ucmp_id, "units", velUnits);
		if (status == NC_NOERR)
		{
			velUnits[velunit_len] = '\0'; 
			if (!strcmpnocase(velUnits,"cm/s") ||!strcmpnocase(velUnits,"Centimeters per second") )
				velConversion = .01;
			else if (!strcmpnocase(velUnits,"m/s"))
				velConversion = 1.0;
		}
	}
	
	
	status = nc_get_att_double(ncid, curr_ucmp_id, "_FillValue", &fill_value);	// should get this in text_read and store, but will have to go short to float and back
	if (status != NC_NOERR) 
	{status = nc_get_att_double(ncid, curr_ucmp_id, "FillValue", &fill_value); 
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, curr_ucmp_id, "missing_value", &fill_value); /*if (status != NC_NOERR) {err = -1; goto done;}*/ }}	// require fill value (took this out 12.12.08)
	
	if (isnan(fill_value))
		fill_value=-99999;
	
	status = nc_get_att_double(ncid, curr_ucmp_id, "scale_factor", &scale_factor);
	if (status != NC_NOERR) {/*err = -1; goto done;*/}	// don't require scale factor
	
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}
	
	velH = (VelocityFH)_NewHandleClear(totalNumberOfVels * sizeof(VelocityFRec) * depthlength);
	if (!velH) {err = memFullErr; goto done;}
	for (k=0;k<depthlength;k++)
	{
		for (i=0;i<latlength;i++)
		{
			for (j=0;j<lonlength;j++)
			{
				if (curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)	// should store in current array and check before drawing or moving
					curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
				if (curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]==fill_value)
					curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;

				if (isnan(curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))	// should store in current array and check before drawing or moving
					curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;
				if (isnan(curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]))
					curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols]=0.;

				INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).u = (float)curr_uvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
				INDEXH(velH,i*lonlength+j+k*fNumRows*fNumCols).v = (float)curr_vvals[(latlength-i-1)*lonlength+j+k*fNumRows*fNumCols] * velConversion;
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
		//printNote("Error opening NetCDF file");
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	if (curr_uvals) delete [] curr_uvals;
	if (curr_vvals) delete [] curr_vvals;
	if (velUnits) {delete [] velUnits;}
	return err;
}


OSErr NetCDFMover_c::SetInterval(char *errmsg, const Seconds& model_time)
{
	long timeDataInterval = 0;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	
	
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
	
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
	
	// check for constant current 
	if(numTimesInFile==1 && !(GetNumFiles()>1))	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0 && fAllowExtrapolationOfCurrentsInTime)
	{
		indexOfStart = 0;
		indexOfEnd = -1;
	}

	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile /*|| (timeDataInterval==1 && fAllowExtrapolationOfCurrentsInTime)*/)
	{	// before the first step in the file
		
		if (GetNumFiles()>1)
		{
			if ((err = CheckAndScanFile(errmsg, model_time)) || fOverLap) goto done;	
			
			intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	
			
			indexOfStart = timeDataInterval-1;
			indexOfEnd = timeDataInterval;
			numTimesInFile = this -> GetNumTimesInFile();
			if (fAllowExtrapolationOfCurrentsInTime && (timeDataInterval==numTimesInFile || timeDataInterval == 0))
			{
				if(intervalLoaded) 
					return 0;
				indexOfEnd = -1;
				if (timeDataInterval == 0) indexOfStart = 0;	// if we allow extrapolation we need to load the first time
			}
		}
		else
		{
			if (fAllowExtrapolationOfCurrentsInTime && timeDataInterval == numTimesInFile) 
			{
				fStartData.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else if (fAllowExtrapolationOfCurrentsInTime && timeDataInterval == 0) 
			{
				fStartData.timeIndex = 0;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else
			{
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
		if(!errmsg[0])strcpy(errmsg,"Error in NetCDFMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;
	
}


OSErr NetCDFMover_c::CheckAndScanFile(char *errmsg, const Seconds& model_time)
{
	Seconds time = model_time, startTime, endTime, lastEndTime, testTime, firstStartTime; 
	
	long i,numFiles = GetNumFiles();
	OSErr err = 0;
	
	errmsg[0]=0;
	if (fEndData.timeIndex!=UNASSIGNEDINDEX)
		testTime = (*fTimeHdl)[fEndData.timeIndex];	// currently loaded end time
	
	firstStartTime = (*fInputFilesHdl)[0].startTime + fTimeShift;
	for (i=0;i<numFiles;i++)
	{
		startTime = (*fInputFilesHdl)[i].startTime + fTimeShift;
		endTime = (*fInputFilesHdl)[i].endTime + fTimeShift;
		if (startTime<=time&&time<=endTime && !(startTime==endTime))
		{
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);	
			
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
			DisposeLoadedData(&fStartData);
			/*if (fOverLapStartTime==testTime)	// shift end time data to start time data
			 {
			 fStartData = fEndData;
			 ClearLoadedData(&fEndData);
			 }
			 else*/
			{
				if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeHdl,false);	
				
				DisposeLoadedData(&fEndData);
				strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeHdl,false);	
			
			strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		if (i>0 && (lastEndTime<time && time<startTime))
		{
			fOverLapStartTime = (*fInputFilesHdl)[i-1].endTime;	// last entry in previous file
			DisposeLoadedData(&fStartData);
			if (fOverLapStartTime==testTime)	// shift end time data to start time data
			{
				fStartData = fEndData;
				ClearLoadedData(&fEndData);
			}
			else
			{
				if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[i-1].pathName,&fTimeHdl,false);	
				
				DisposeLoadedData(&fEndData);
				strcpy(fVar.pathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);	
			
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		lastEndTime = endTime;
	}
	if (fAllowExtrapolationOfCurrentsInTime && time > lastEndTime)
	{
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		err = ScanFileForTimes((*fInputFilesHdl)[numFiles-1].pathName,&fTimeHdl,false);	
		
		// code goes here, check that start/end times match
		strcpy(fVar.pathName,(*fInputFilesHdl)[numFiles-1].pathName);
		fOverLap = false;
		return err;
	}
	if (fAllowExtrapolationOfCurrentsInTime && time < firstStartTime)
	{
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		err = ScanFileForTimes((*fInputFilesHdl)[0].pathName,&fTimeHdl,false);	
		
		// code goes here, check that start/end times match
		strcpy(fVar.pathName,(*fInputFilesHdl)[0].pathName);
		fOverLap = false;
		return err;
	}
	strcpy(errmsg,"Time outside of interval being modeled");
	return -1;	
}

/*Seconds RoundDateSeconds(Seconds timeInSeconds)
{
	double	DaysInMonth[13] = {0.0,31.0,28.0,31.0,30.0,31.0,30.0,31.0,31.0,30.0,31.0,30.0,31.0};
	DateTimeRec date;
	Seconds roundedTimeInSeconds;
	// get rid of the seconds since they get garbled in the dialogs
	SecondsToDate(timeInSeconds,&date);
	if (date.second == 0) return timeInSeconds;
	if (date.second > 30) 
	{
		if (date.minute<59) date.minute++;
		else
		{
			date.minute = 0;
			if (date.hour < 23) date.hour++;
			else
			{
				if( (date.year % 4 == 0 && date.year % 100 != 0) || date.year % 400 == 0) DaysInMonth[2]=29.0;
				date.hour = 0;
				if (date.day < DaysInMonth[date.month]) date.day++;
				else
				{
					date.day = 1;
					if (date.month < 12) date.month++;
					else
					{
						date.month = 1;
						if (date.year>2019) {printError("Time outside of model range"); }
						else date.year++;
						date.year++;
					}
				}
			}
		}
	}
	date.second = 0;
	DateToSeconds(&date,&roundedTimeInSeconds);
	return roundedTimeInSeconds;
}*/


OSErr NetCDFMover_c::ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime)
{
	OSErr err = 0;
	long i,numScanned,line=0;
	DateTimeRec time;
	Seconds timeSeconds;
	char s[1024], outPath[256];
	CHARH fileBufH = 0;
	int status, ncid, recid, timeid;
	size_t recs, t_len, t_len2;
	double timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	static size_t timeIndex;
	Seconds startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	Seconds **timeHdl = 0;
	
	status = nc_open(path, NC_NOWRITE, &ncid);
	if (status != NC_NOERR) /*{err = -1; goto done;}*/
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
		if (status != NC_NOERR) {err = -2; goto done;} 
		timeUnits[t_len] = '\0'; // moved this statement before StringSubstitute, JLM 5/2/10
		StringSubstitute(timeUnits, ':', ' ');
		StringSubstitute(timeUnits, '-', ' ');
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
						  unitStr, junk, &time.year, &time.month, &time.day,
						  &time.hour, &time.minute, &time.second) ;
		if (numScanned==5)	
		{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned==7)	time.second = 0;
		else if (numScanned<8)	
			//if (numScanned!=8)	
		{ err = -1; TechError("NetCDFMover::ScanFileForTimes()", "sscanf() == 8", 0); goto done; }
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
	
	status = nc_inq_dim(ncid, recid, recname, &recs);
	if (status != NC_NOERR) {err = -2; goto done;}
	timeHdl = (Seconds**)_NewHandleClear(recs*sizeof(Seconds));
	if (!timeHdl) {err = memFullErr; goto done;}
	for (i=0;i<recs;i++)
	{
		Seconds newTime;
		// possible units are, HOURS, MINUTES, SECONDS,...
		timeIndex = i;
		status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
		if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); err = -2; goto done;}
		newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
		INDEXH(timeHdl,i) = newTime;	// which start time where?
	}
	*timeH = timeHdl;
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -2; goto done;}
	
	
done:
	if (err)
	{
		if (err==-2) {printError("Error reading times from NetCDF file");}
		if (timeHdl) {DisposeHandle((Handle)timeHdl); timeHdl=0;}
	}
	return err;
}
void NetCDFMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
	bIsFirstStep = false;
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
		strcpy(path,fVar.pathName);
		if (!path || !path[0]) return -1;
		
		status = nc_open(path, NC_NOWRITE, &ncid);
		if (status != NC_NOERR) 
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
			numDepthLevels = sigmaLength;
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

OSErr NetCDFMover_c::get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID) {

	if(!ref || !delta) {
		//cout << "worldpoints array not provided! returning.\n";
		return 1;
	}
	

	// For LEType spillType, check to make sure it is within the valid values
	if( spillType < FORECAST_LE || spillType > UNCERTAINTY_LE)
	{
		// cout << "Invalid spillType.\n";
		return 2;
	}
	
	LERec* prec;
	LERec rec;
	prec = &rec;
	
	WorldPoint3D zero_delta ={0,0,0.};
	
	for (int i = 0; i < n; i++) {
		
		// only operate on LE if the status is in water
		if( LE_status[i] != OILSTAT_INWATER)
		{
			delta[i] = zero_delta;
			continue;
		}
		rec.p = ref[i].p;
		rec.z = ref[i].z;
		
		// let's do the multiply by 1000000 here - this is what gnome expects
		rec.p.pLat *= 1000000;	
		rec.p.pLong*= 1000000;
		
		delta[i] = GetMove(model_time, step_len, spill_ID, i, prec, spillType);
		
		delta[i].p.pLat /= 1000000;
		delta[i].p.pLong /= 1000000;
	}
	
	return noErr;
}

WorldPoint3D NetCDFMover_c::GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {{0,0},0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha, depthAlpha;
	float topDepth, bottomDepth;
	long index; 
	long depthIndex1,depthIndex2;	// default to -1?
	Seconds startTime,endTime;
	Seconds time = model_time; 
	
	VelocityRec scaledPatVelocity;
	Boolean useEddyUncertainty = false;	
	OSErr err = 0;
	Boolean useVelSubsurface = false;	// don't seem to need this, just return 
	char errmsg[256];
	
	if(!fIsOptimizedForStep) 
	{
		err = dynamic_cast<NetCDFMover *>(this) -> SetInterval(errmsg, model_time); 
		
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
	if((dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()==1 && !(dynamic_cast<NetCDFMover *>(this)->GetNumFiles()>1)) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfCurrentsInTime))
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
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
		if (dynamic_cast<NetCDFMover *>(this)->GetNumFiles()>1 && fOverLap)
			startTime = fOverLapStartTime + fTimeShift;
		else
			startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
		endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
		timeAlpha = (endTime - time)/(double)(endTime - startTime);
		
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
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

	scaledPatVelocity.u *= fVar.curScale; 
	scaledPatVelocity.v *= fVar.curScale; 
	scaledPatVelocity.u *= fFileScaleFactor; 
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

VelocityRec NetCDFMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
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
	
	TRectGridVel* rectGrid = dynamic_cast<TRectGridVel*>(fGrid);	// fNumRows, fNumCols members of NetCDFMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	dColNum = (p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5;
	dRowNum = (p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5;

	colNum = round(dColNum);
	rowNum = round(dRowNum);
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return -1; }
	
	return rowNum * fNumCols + colNum;
}

LongPoint NetCDFMover_c::GetVelocityIndices(WorldPoint p) 
{
	long rowNum, colNum;
	double dRowNum, dColNum;
	LongPoint indices = {-1,-1};
	VelocityRec	velocity;
	
	LongRect		gridLRect, geoRect;
	ScaleRec		thisScaleRec;
	
	TRectGridVel* rectGrid = dynamic_cast<TRectGridVel*>(fGrid);	// fNumRows, fNumCols members of NetCDFMover
	
	WorldRect bounds = rectGrid->GetBounds();
	
	SetLRect (&gridLRect, 0, fNumRows, fNumCols, 0);
	SetLRect (&geoRect, bounds.loLong, bounds.loLat, bounds.hiLong, bounds.hiLat);	
	GetLScaleAndOffsets (&geoRect, &gridLRect, &thisScaleRec);
	
	dColNum = round((p.pLong * thisScaleRec.XScale + thisScaleRec.XOffset) -.5);
	dRowNum = round((p.pLat  * thisScaleRec.YScale + thisScaleRec.YOffset) -.5);

	colNum = dColNum;
	rowNum = dRowNum;
	
	if (colNum < 0 || colNum >= fNumCols || rowNum < 0 || rowNum >= fNumRows)
		
	{ return indices; }
	
	indices.h = colNum;
	indices.v = rowNum;
	return indices;
}


/////////////////////////////////////////////////
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
		depth = myfabs(totalDepth*(hc*sc_r+totalDepth*Cs_r))/(totalDepth+hc);
	}
	else
		depth = INDEXH(fDepthLevelsHdl,depthIndex)*totalDepth; // times totalDepth
	
	return depth;
}
#ifndef pyGNOME
// this is really a gui function...
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
	err = dynamic_cast<NetCDFMover *>(this) -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
	
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
	
	if(dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()>1)
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

	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular
		
		indices = this->GetVelocityIndices(wp.p);
		
		if (index >= 0)
		{
			// Check for constant current 
			if(dynamic_cast<NetCDFMover *>(this)->GetNumTimesInFile()==1 || timeAlpha == 1)
			{
				if(depthIndex2==UNASSIGNEDINDEX) // surface velocity or special cases
				{
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
	
	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v) * this->fFileScaleFactor;
	lengthS = this->fVar.curScale * lengthU;
	if (lengthS > 1000000 || this->fVar.curScale==0) return true;	// if bad data in file causes a crash
	
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
#endif


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
		long numDepthLevels = dynamic_cast<NetCDFMover *>(this)->GetNumDepthLevelsInFile();
		if (numDepthLevels<=0) return maxDepth;
		if (fDepthLevelsHdl) maxDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	}
	return maxDepth;
}

float NetCDFMover_c::GetTotalDepth(WorldPoint wp, long triNum)
{	// z grid only 
#pragma unused(wp)
#pragma unused(triNum)
	long numDepthLevels = dynamic_cast<NetCDFMover *>(this)->GetNumDepthLevelsInFile();
	float totalDepth = 0;

	if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	return totalDepth;
}

void NetCDFMover_c::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData = 0;
	long numDepthLevels = GetNumDepthLevelsInFile();
	float totalDepth = 0;
	
	
	if (fDepthLevelsHdl && numDepthLevels>0) totalDepth = INDEXH(fDepthLevelsHdl,numDepthLevels-1);
	else
	{
		*depthIndex1 = indexToDepthData;
		*depthIndex2 = UNASSIGNEDINDEX;
		return;
	}

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
}
