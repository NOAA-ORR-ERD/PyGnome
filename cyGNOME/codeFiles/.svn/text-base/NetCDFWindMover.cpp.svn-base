// This is a cross between TWindMover and NetCDFMover, possibly reorganize to build off one or the other
// The uncertainty is from the wind, the reading, storing, accessing, displaying data is from NetCDFMover


#include "Cross.h"
#include "NetCDFMover.h"
#include "netcdf.h"
#include "TWindMover.h"
#include "GridCurMover.h"
#include "Outils.h"
#include "DagTreeIO.h"

#ifdef MAC
#ifdef MPW
//#include <QDOffscreen.h>
#pragma SEGMENT NETCDFWINDMOVER
#endif
#endif

enum {
	   I_NETCDFWINDNAME = 0, I_NETCDFWINDACTIVE, I_NETCDFWINDSHOWGRID, I_NETCDFWINDSHOWARROWS, I_NETCDFWINDUNCERTAIN,
	   I_NETCDFWINDSPEEDSCALE,I_NETCDFWINDANGLESCALE, I_NETCDFWINDSTARTTIME,I_NETCDFWINDDURATION
		};


///////////////////////////////////////////////////////////////////////////

long NetCDFWindMover::GetNumFiles()
{
	long numFiles = 0;

	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

OSErr NetCDFWindMover::CheckAndScanFile(char *errmsg)
{
	Seconds time = model->GetModelTime(), startTime, endTime, lastEndTime, testTime, firstStartTime;
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
			strcpy(fPathName,(*fInputFilesHdl)[i].pathName);
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
				strcpy(fPathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeHdl,false);
			strcpy(fPathName,(*fInputFilesHdl)[fileNum].pathName);
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
				strcpy(fPathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);
			strcpy(fPathName,(*fInputFilesHdl)[i].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		lastEndTime = endTime;
	}
	if (fAllowExtrapolationOfWinds && time > lastEndTime)
	{
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		err = ScanFileForTimes((*fInputFilesHdl)[numFiles-1].pathName,&fTimeHdl,false);
		// code goes here, check that start/end times match
		strcpy(fPathName,(*fInputFilesHdl)[numFiles-1].pathName);
		fOverLap = false;
		return err;
	}
	if (fAllowExtrapolationOfWinds && time < firstStartTime)
	{
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
		err = ScanFileForTimes((*fInputFilesHdl)[0].pathName,&fTimeHdl,false);
		// code goes here, check that start/end times match
		strcpy(fPathName,(*fInputFilesHdl)[0].pathName);
		fOverLap = false;
		return err;
	}
	strcpy(errmsg,"Time outside of interval being modeled");
	return -1;	
	//return err;
}

Boolean NetCDFWindMover::CheckInterval(long &timeDataInterval)
{
	Seconds time =  model->GetModelTime(), startTime, endTime;;
	long i,numTimes,numFiles = GetNumFiles();


	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists

	// check for constant current
	if (numTimes==1 && !(GetNumFiles()>1)) 
	//if (numTimes==1) 
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
		timeDataInterval = 0;
		if (numFiles > 0)
		{
			//startTime = (*fInputFilesHdl)[0].startTime + fTimeShift;
			startTime = (*fInputFilesHdl)[0].startTime;
			if ((*fTimeHdl)[0] != startTime)
				return false;
		}
		if (fAllowExtrapolationOfWinds && fEndData.timeIndex == UNASSIGNEDINDEX && !(fStartData.timeIndex == UNASSIGNEDINDEX))	// way to recognize last interval is set
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
		timeDataInterval = numTimes;
		if (numFiles > 0)
		{
			//endTime = (*fInputFilesHdl)[numFiles-1].endTime + fTimeShift;
			endTime = (*fInputFilesHdl)[numFiles-1].endTime;
			if ((*fTimeHdl)[numTimes-1] != endTime)
				return false;
		}
		if (fAllowExtrapolationOfWinds && fEndData.timeIndex == UNASSIGNEDINDEX && !(fStartData.timeIndex == UNASSIGNEDINDEX))	// way to recognize last interval is set
		{
			//check if time > last model time in all files
			return true;
		}
	}
	return false;
		
}


OSErr NetCDFWindMover::SetInterval(char *errmsg)
{
	long timeDataInterval = 0;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
		
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
		
	// check for constant current 
	//if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	if(numTimesInFile==1 && !(GetNumFiles()>1))	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0 && fAllowExtrapolationOfWinds)
	{
		indexOfStart = 0;
		indexOfEnd = -1;
	}
	/*if(timeDataInterval == 0)
	{	// before the first step in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}
	else if(timeDataInterval == numTimesInFile) 
	{	// past the last information in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}*/
	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile /*|| (timeDataInterval==1 && fAllowExtrapolationOfWinds)*/)
	{	// before the first step in the file

		if (GetNumFiles()>1)
		{
			if ((err = CheckAndScanFile(errmsg)) || fOverLap) goto done;	// overlap is special case
			intervalLoaded = this -> CheckInterval(timeDataInterval);
			indexOfStart = timeDataInterval-1;
			indexOfEnd = timeDataInterval;
			numTimesInFile = this -> GetNumTimesInFile();
			if (fAllowExtrapolationOfWinds && (timeDataInterval==numTimesInFile || timeDataInterval == 0))
			{
				if(intervalLoaded) 
					return 0;
				indexOfEnd = -1;
				if (timeDataInterval == 0) indexOfStart = 0;	// if we allow extrapolation we need to load the first time
			}
		}
		else
		{
			if (fAllowExtrapolationOfWinds && timeDataInterval == numTimesInFile) 
			{
				fStartData.timeIndex = numTimesInFile-1;//check if time > last model time in all files
				fEndData.timeIndex = UNASSIGNEDINDEX;//check if time > last model time in all files
			}
			else if (fAllowExtrapolationOfWinds && timeDataInterval == 0) 
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
		if(!errmsg[0])strcpy(errmsg,"Error in NetCDFWindMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;

}

OSErr NetCDFWindMover::ReadInputFileNames(char *fileNamesPath)
{
	// for netcdf files, header file just has the paths, the start and end times will be read from the files
	long i,numScanned,line=0, numFiles, numLinesInText;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], path[256], outPath[256], classicPath[256];
	CHARH fileBufH = 0;
	PtCurFileInfoH inputFilesHdl = 0;
	int status, ncid, recid, timeid;
	size_t recs, t_len, t_len2;
	double timeVal;
	char recname[NC_MAX_NAME], *timeUnits=0;	
	static size_t timeIndex;
	Seconds startTime2;
	double timeConversion = 1.;
	char errmsg[256] = "";
	
	if (err = ReadFileContents(TERMINATED,0, 0, fileNamesPath, 0, 0, &fileBufH)) goto done;

	numLinesInText = NumLinesInText(*fileBufH);
	numFiles = numLinesInText - 1;	// subtract off the header ("NetCDF Files")
	inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("NetCDFWindMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
	NthLineInTextNonOptimized(*fileBufH, (line)++, s, 1024); 	// header line
	for (i=0;i<numFiles;i++)	// should count files as go along
	{
		NthLineInTextNonOptimized(*fileBufH, (line)++, s, 1024); 	// check it is a [FILE] line
		//strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE]\t"));
		RemoveLeadingAndTrailingWhiteSpace(s);
		strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace((*inputFilesHdl)[i].pathName);
		ResolvePathFromInputFile(fileNamesPath,(*inputFilesHdl)[i].pathName); // JLM 6/8/10
		strcpy(path,(*inputFilesHdl)[i].pathName);
		if((*inputFilesHdl)[i].pathName[0] && FileExists(0,0,(*inputFilesHdl)[i].pathName))
		{
			//
			status = nc_open(path, NC_NOWRITE, &ncid);
			if (status != NC_NOERR) /*{err = -1; goto done;}*/
			{
		#if TARGET_API_MAC_CARBON
				err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
					status = nc_open(outPath, NC_NOWRITE, &ncid);
		#endif
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			//if (status != NC_NOERR) {err = -2; goto done;}
		
			//status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being dimension name
			status = nc_inq_dimid(ncid, "time", &recid); 
			if (status != NC_NOERR) 
			{
				//status = nc_inq_dimid(ncid, "time", &recid); 
				status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
				if (status != NC_NOERR) {err = -2; goto done;}
			}
		
			status = nc_inq_varid(ncid, "time", &timeid); 
			if (status != NC_NOERR) {err = -2; goto done;} 
		
			/////////////////////////////////////////////////
			status = nc_inq_attlen(ncid, timeid, "units", &t_len);
			if (status != NC_NOERR) 
			{
				timeUnits = 0;	// files should always have this info
				timeConversion = 3600.;		// default is hours
				startTime2 = model->GetStartTime();	// default to model start time
				/*err = -2; goto done;*/
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
				else if (numScanned!=8)	
				//if (numScanned!=8)	
					{ err = -1; TechError("NetCDFWindMover::ReadInputFileNames()", "sscanf() == 8", 0); goto done; }
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
			{
				Seconds newTime;
				// possible units are, HOURS, MINUTES, SECONDS,...
				timeIndex = 0;	// first time
				status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
				if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); printError(errmsg); err = -1; goto done;}
				newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
				(*inputFilesHdl)[i].startTime = newTime;
				timeIndex = recs-1;	// last time
				status = nc_get_var1_double(ncid, timeid, &timeIndex, &timeVal);
				if (status != NC_NOERR) {strcpy(errmsg,"Error reading times from NetCDF file"); printError(errmsg); err = -1; goto done;}
				newTime = RoundDateSeconds(round(startTime2+timeVal*timeConversion));
				(*inputFilesHdl)[i].endTime = newTime;
			}
			status = nc_close(ncid);
			if (status != NC_NOERR) {err = -2; goto done;}
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to NetCDF data File does not exist.%s%s",NEWLINESTRING,(*inputFilesHdl)[i].pathName);
			printError(msg);
			err = true;
			goto done;
		}


	}
	fInputFilesHdl = inputFilesHdl;

done:
	if(fileBufH) { DisposeHandle((Handle)fileBufH); fileBufH = 0;}
	if (err)
	{
		if (err==-2) {printError("Error reading netCDF file");}
		if(inputFilesHdl) {DisposeHandle((Handle)inputFilesHdl); inputFilesHdl=0;}
	}
	return err;
}

OSErr NetCDFWindMover::ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime)
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
	//if (status != NC_NOERR) {err = -1; goto done;}

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
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		/*err = -1; goto done;*/
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
		else if (numScanned!=8)	
		//if (numScanned!=8)	
			{ err = -1; TechError("NetCDFWindMover::ScanFileForTimes()", "sscanf() == 8", 0); goto done; }
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

//////////////////////////////// END OF ADDED CODE

long NetCDFWindMover::GetListLength()
{
	long count = 1; // wind name
	long mode = model->GetModelMode();
	long numTimesInFile = GetNumTimesInFile();
	
	if (bOpen) {
		if(mode == ADVANCEDMODE) count += 1; // active
		if(mode == ADVANCEDMODE) count += 1; // showgrid
		if(mode == ADVANCEDMODE) count += 1; // showarrows
		if(mode == ADVANCEDMODE && model->IsUncertain())count++;
		if(mode == ADVANCEDMODE && model->IsUncertain() && bUncertaintyPointOpen)count+=4;
		if (numTimesInFile>0 || GetNumFiles()>1) count +=2;	// start and end times, otherwise it's steady state
	}
	
	return count;
}

ListItem NetCDFWindMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	char valStr[64], dateStr[64];
	long numTimesInFile = GetNumTimesInFile();
	ListItem item = { this, n, indent, 0 };
	long mode = model->GetModelMode();
	
	if (n == 0) {
		item.index = I_NETCDFWINDNAME;
		if (mode == ADVANCEDMODE) item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text,"Wind File: ");
		strcat(text,fFileName);
		if(!bActive)*style = italic; // JLM 6/14/10
	
		return item;
	}
	
	if (bOpen) {
	
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_NETCDFWINDACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
				
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_NETCDFWINDSHOWGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Grid");
			
			return item;
		}
				
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_NETCDFWINDSHOWARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			StringWithoutTrailingZeros(valStr,fArrowScale,6);
			//strcpy(text, "Show Velocity Vectors");
			sprintf(text, "Show Velocities (@ 1 in = %s m/s) ",valStr);
			
			return item;
		}
						
		// release time
		if (GetNumFiles()>1)
		{
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fInputFilesHdl)[0].startTime + fTimeShift;
				Secs2DateString2 (time, dateStr);
				/*if(numTimesInFile>0)*/ sprintf (text, "Start Time: %s", dateStr);
				//else sprintf (text, "Time: %s", dateStr);
				return item;
			}
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fInputFilesHdl)[GetNumFiles()-1].endTime + fTimeShift;				
				Secs2DateString2 (time, dateStr);
				sprintf (text, "End Time: %s", dateStr);
				return item;
			}
		}
		else
		{
		if (numTimesInFile>0)
		{
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fTimeHdl)[0] + fTimeShift;				
				Secs2DateString2 (time, dateStr);
				/*if(numTimesInFile>0)*/ sprintf (text, "Start Time: %s", dateStr);
				//else sprintf (text, "Time: %s", dateStr);
				return item;
			}
		}
		if (numTimesInFile>0)
		{
			if (--n == 0) {
				//item.indent++;
				Seconds time = (*fTimeHdl)[numTimesInFile-1] + fTimeShift;				
				Secs2DateString2 (time, dateStr);
				sprintf (text, "End Time: %s", dateStr);
				return item;
			}
		}
		}

		if(mode == ADVANCEDMODE && model->IsUncertain())
		{
			if (--n == 0) 
			{
				item.indent++;
				item.index = I_NETCDFWINDUNCERTAIN;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				
				return item;
			}
			
			if(bUncertaintyPointOpen)
			{
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDSTARTTIME;
					sprintf(text, "Start Time: %.2f hours",(float)(fUncertainStartTime/3600));
					return item;
				}
			
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDDURATION;
					sprintf(text, "Duration: %.2f hr", (float)(fDuration / 3600.0));
					return item;
				}
				
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDSPEEDSCALE;
					sprintf(text, "Speed Scale: %.2f ", fSpeedScale);
					return item;
				}
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_NETCDFWINDANGLESCALE;
					sprintf(text, "Angle Scale: %.2f ", fAngleScale);
					return item;
				}
			}
		}
	}
	
	item.owner = 0;
	
	return item;
}

Boolean NetCDFWindMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_NETCDFWINDNAME: bOpen = !bOpen; return TRUE;
			case I_NETCDFWINDACTIVE: bActive = !bActive; 
					model->NewDirtNotification(); return TRUE;
		case I_NETCDFWINDSHOWGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_NETCDFWINDSHOWARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_NETCDFWINDUNCERTAIN:bUncertaintyPointOpen = !bUncertaintyPointOpen;return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_NETCDFWINDNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
	}

	if (doubleClick)
	{	
		switch(item.index)
		{
			case I_NETCDFWINDACTIVE:
			case I_NETCDFWINDUNCERTAIN:
			case I_NETCDFWINDSPEEDSCALE:
			case I_NETCDFWINDANGLESCALE:
			case I_NETCDFWINDSTARTTIME:
			case I_NETCDFWINDDURATION:
			case I_NETCDFWINDNAME:
				NetCDFWindSettingsDialog(this, this -> moverMap,false,mapWindow);
				//WindSettingsDialog(this, this -> moverMap,false,mapWindow,false);
				break;
			default:	// why not call this for everything?
				NetCDFWindSettingsDialog(this, this -> moverMap,false,mapWindow);
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean NetCDFWindMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_NETCDFWINDNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON:
					if(model->GetModelMode() < ADVANCEDMODE) return FALSE; // shouldn't happen
					else return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
			}
			break;
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return TWindMover::FunctionEnabled(item, buttonID);
}

OSErr NetCDFWindMover::SettingsItem(ListItem item)
{
	//return NetCDFWindSettingsDialog(this, this -> moverMap,false,mapWindow);
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = ListClick(item,inBullet,doubleClick);
	return 0;
}

OSErr NetCDFWindMover::DeleteItem(ListItem item)
{
	if (item.index == I_NETCDFWINDNAME)
		return moverMap -> DropMover(this);
	
	return 0;
}

NetCDFWindMover::NetCDFWindMover(TMap *owner,char* name) : TWindMover(owner, name)
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
	//fArrowScale = 1.;
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

long NetCDFWindMover::GetVelocityIndex(WorldPoint p) 
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

LongPoint NetCDFWindMover::GetVelocityIndices(WorldPoint p) 
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
double NetCDFWindMover::GetStartUVelocity(long index)
{	// 
	double u = 0;
	if (index>=0)
	{
		if (fStartData.dataHdl) u = INDEXH(fStartData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFWindMover::GetEndUVelocity(long index)
{
	double u = 0;
	if (index>=0)
	{
		if (fEndData.dataHdl) u = INDEXH(fEndData.dataHdl,index).u;
		if (u==fFillValue) u = 0;
	}
	return u;
}

double NetCDFWindMover::GetStartVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fStartData.dataHdl) v = INDEXH(fStartData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

double NetCDFWindMover::GetEndVVelocity(long index)
{
	double v = 0;
	if (index >= 0)
	{
		if (fEndData.dataHdl) v = INDEXH(fEndData.dataHdl,index).v;
		if (v==fFillValue) v = 0;
	}
	return v;
}

OSErr NetCDFWindMover::GetStartTime(Seconds *startTime)
{
	OSErr err = 0;
	*startTime = 0;
	if (fStartData.timeIndex != UNASSIGNEDINDEX)
		*startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

OSErr NetCDFWindMover::GetEndTime(Seconds *endTime)
{
	OSErr err = 0;
	*endTime = 0;
	if (fEndData.timeIndex != UNASSIGNEDINDEX)
		*endTime = (*fTimeHdl)[fEndData.timeIndex] + fTimeShift;
	else return -1;
	return 0;
}

Boolean NetCDFWindMover::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
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
	err = this -> SetInterval(errmsg);
	if(err) return false;

	if(this->GetNumTimesInFile()>1)
	//&& loaded && !err)
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
	//if (loaded && !err)
	{	
		index = this->GetVelocityIndex(wp.p);	// need alternative for curvilinear and triangular

		indices = this->GetVelocityIndices(wp.p);

		if (index >= 0)
		{
			// Check for constant current 
			if(this->GetNumTimesInFile()==1 || timeAlpha == 1)
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

	StringWithoutTrailingZeros(uStr,lengthU,4);
	StringWithoutTrailingZeros(sStr,lengthS,4);
	
	//sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s]",
						//	this->className, uStr, sStr);
	sprintf(diagnosticStr, " [grid: %s, unscaled: %s m/s, scaled: %s m/s], file indices : [%ld, %ld]",
							this->className, uStr, sStr, fNumRows-indices.v-1, indices.h);

	return true;
}

void NetCDFWindMover::Dispose()
{
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}

	if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);
	if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}

	TWindMover::Dispose ();
}

///////////////////////////////////////////////////////////////////////////
OSErr NetCDFWindMover::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM

	/*char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		char str[256];
		OSErr err = 0;
		////////////////
		err = message->GetParameterAsDouble("uncertaintySpeedScale",&val);
		if(!err) this->fSpeedScale = val; 
		////////////////
		err = message->GetParameterAsDouble("uncertaintyAngleScale",&val);
		if(!err) this->fAngleScale = val; 
		////////////////
		model->NewDirtNotification();// tell model about dirt
	}*/
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TWindMover::CheckAndPassOnMessage(message);
}

OSErr NetCDFWindMover::PrepareForModelStep()
{
	OSErr err = this->UpdateUncertainty();

	char errmsg[256];
	
	errmsg[0]=0;

	if (!bActive) return noErr;

	err = this -> SetInterval(errmsg); // SetInterval checks to see that the time interval is loaded
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

void NetCDFWindMover::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}

void NetCDFWindMover::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void NetCDFWindMover::DisposeAllLoadedData()
{
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);
}


void NetCDFWindMover::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long NetCDFWindMover::GetNumTimesInFile()
{
	long numTimes;

	numTimes = _GetHandleSize((Handle)fTimeHdl)/sizeof(**fTimeHdl);
	return numTimes;     
}

WorldPoint3D NetCDFWindMover::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
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
	
	// if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);	// ok, but don't print error message here
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
							
	// Check for constant wind 
	if( ( GetNumTimesInFile()==1 && !( GetNumFiles() > 1 ) ) ||
		(fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds) || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
	//if(GetNumTimesInFile()==1)
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

//scale:

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

Seconds NetCDFWindMover::GetTimeValue(long index)
{
	if (index<0) printError("Access violation in NetCDFWindMover::GetTimeValue()");
	Seconds time = (*fTimeHdl)[index] + fTimeShift;
	return time;
}

//#define NetCDFWindMoverREADWRITEVERSION 1 //JLM	7/10/01
//#define NetCDFWindMoverREADWRITEVERSION 2 //JLM	7/10/01
#define NetCDFWindMoverREADWRITEVERSION 3 //JLM	5/3/10

OSErr NetCDFWindMover::Write(BFPB *bfpb)
{
	long i, version = NetCDFWindMoverREADWRITEVERSION;
	ClassID id = GetClassID ();
	long numTimes = GetNumTimesInFile();
	Seconds time;
	OSErr err = 0;
	
	if (err = TWindMover::Write(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFWindMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;

	if (err = WriteMacValue(bfpb, fSpeedScale)) return err;
	if (err = WriteMacValue(bfpb, fAngleScale)) return err;
	if (err = WriteMacValue(bfpb, fMaxAngle)) return err;
	if (err = WriteMacValue(bfpb, fSigma2)) return err;
	if (err = WriteMacValue(bfpb, fSigmaTheta)) return err;
	
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	if (err = fGrid -> Write (bfpb)) goto done;
	
	if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	if (err = WriteMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	if (err = WriteMacValue(bfpb, fFileName, kPtCurUserNameLen)) return err;

	if (err = WriteMacValue(bfpb, bShowGrid)) return err;
	if (err = WriteMacValue(bfpb, bShowArrows)) return err;
	if (err = WriteMacValue(bfpb, fUserUnits)) return err;
	if (err = WriteMacValue(bfpb, fArrowScale)) return err;
	if (err = WriteMacValue(bfpb, fWindScale)) return err;
	//
	if (err = WriteMacValue(bfpb, fFillValue)) return err;
	//
	if (err = WriteMacValue(bfpb, numTimes)) goto done;
	for (i=0;i<numTimes;i++)
	{
		time = INDEXH(fTimeHdl,i);
		if (err = WriteMacValue(bfpb, time)) goto done;
	}
	if (err = WriteMacValue(bfpb, fTimeShift)) goto done;
	if (err = WriteMacValue(bfpb, fAllowExtrapolationOfWinds)) goto done;

	///////////////////////////////////
	// JLM 5/3/10
	//////////////
	{   ///// start version 3 /////////	
		long numFiles = GetNumFiles();
		if (err = WriteMacValue(bfpb, numFiles)) goto done;
		if (numFiles > 0)
		{
			for (i = 0 ; i < numFiles ; i++) {
				PtCurFileInfo fileInfo = INDEXH(fInputFilesHdl,i);
				if (err = WriteMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
				if (err = WriteMacValue(bfpb, fileInfo.startTime)) goto done;
				if (err = WriteMacValue(bfpb, fileInfo.endTime)) goto done;
			}
			if (err = WriteMacValue(bfpb, fOverLap)) return err;
			if (err = WriteMacValue(bfpb, fOverLapStartTime)) return err;
		}
	}  // end version 3
	////////////////


done:
	if(err)
		TechError("NetCDFWindMover::Write(char* path)", " ", 0); 

	return err;
}

OSErr NetCDFWindMover::Read(BFPB *bfpb)
{
	char c, msg[256], fileName[256], newFileName[64];
	long i, version, numTimes, numPoints;
	ClassID id;
	float val;
	Seconds time;
	Boolean bPathIsValid = true;
	OSErr err = 0;
	
	if (err = TWindMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("NetCDFWindMover::Read()");
	if (err = ReadMacValue(bfpb, &id)) return err;
	if (id != GetClassID ()) { TechError("NetCDFWindMover::Read()", "id != TYPE_NETCDFWINDMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb, &version)) return err;
	if (version > NetCDFWindMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	
	if (err = ReadMacValue(bfpb, &fSpeedScale)) return err;
	if (err = ReadMacValue(bfpb, &fAngleScale)) return err;
	if (err = ReadMacValue(bfpb, &fMaxAngle)) return err;
	if (err = ReadMacValue(bfpb, &fSigma2)) return err;
	if (err = ReadMacValue(bfpb, &fSigmaTheta)) return err;
	
	// read the type of grid used for the NetCDF wind mover (should always be rectgrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{	
		case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		//case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in NetCDFWindMover::Read()."); return -1;
	}

	if (err = fGrid -> Read (bfpb)) goto done;
	
	if (err = ReadMacValue(bfpb, &fNumRows)) goto done;	
	if (err = ReadMacValue(bfpb, &fNumCols)) goto done;	
	if (err = ReadMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	ResolvePath(fPathName); // JLM 6/3/10
	//if (!FileExists(0,0,fPathName)) {/*err=-1;*/ sprintf(msg,"The file path %s is no longer valid.",fPathName); printNote(msg); /*goto done;*/}
	if (!FileExists(0,0,fPathName)) 
	{	// allow user to put file in local directory
		char newPath[kMaxNameLen],/*fileName[64],*/*p;
		strcpy(fileName,"");
		strcpy(newPath,fPathName);
		p = strrchr(newPath,DIRDELIMITER);
		if (p)
		{
			strcpy(fileName,p);
			ResolvePath(fileName);
		}
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{/*err=-1;*/ /*sprintf(msg,"The file path %s is no longer valid.",fPathName); printNote(msg);*/ bPathIsValid = false;/*goto done;*/}
		else
			strcpy(fPathName,fileName);
	}
	if (err = ReadMacValue(bfpb, fFileName, kPtCurUserNameLen)) return err;
	if (!bPathIsValid)
	{	// try other platform
		char delimStr[32] ={DIRDELIMITER,0};		
		strcpy(fileName,delimStr);
		strcat(fileName,fFileName);
		ResolvePath(fileName);
		if (!fileName[0] || !FileExists(0,0,fileName)) 
		{/*err=-1;*/ /*sprintf(msg,"The file path %s is no longer valid.",fPathName); printNote(msg);*/ /*goto done;*/}
		else
		{
			strcpy(fPathName,fileName);
			bPathIsValid = true;
		}
	}
	// otherwise ask the user, trusting that user actually chooses the same data file (should insist name is the same?)
	if(!bPathIsValid)
	{
		Point where;
		OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
		MySFReply reply;
		where = CenteredDialogUpLeft(M38c);
		char newPath[kMaxNameLen], s[kMaxNameLen];
		sprintf(msg,"This save file references a netCDF file which cannot be found.  Please find the file \"%s\".",fPathName);printNote(msg);
#if TARGET_API_MAC_CARBON
		mysfpgetfile(&where, "", -1, typeList,
				   (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
		if (!reply.good) return USERCANCEL;
		strcpy(newPath, reply.fullPath);
		strcpy (s, newPath);
		SplitPathFile (s, newFileName);
		strcpy (fPathName, newPath);
		strcpy (fFileName, newFileName);
#else
		sfpgetfile(&where, "",
					(FileFilterUPP)0,
					-1, typeList,
					(DlgHookUPP)0,
					&reply, M38c,
					(ModalFilterUPP)MakeUPP((ProcPtr)STDFilter, uppModalFilterProcInfo));
		//if (!reply.good) return 0;	// just keep going...
		if (reply.good)
		{
			my_p2cstr(reply.fName);
		#ifdef MAC
			GetFullPath(reply.vRefNum, 0, (char *)reply.fName, newPath);
		#else
			strcpy(newPath, reply.fName);
		#endif
			
			strcpy (s, newPath);
			SplitPathFile (s, newFileName);
			strcpy (fPathName, newPath);
			strcpy (fFileName, newFileName);
		}
#endif
	}
	
	//
	if (err = ReadMacValue(bfpb, &bShowGrid)) return err;
	if (err = ReadMacValue(bfpb, &bShowArrows)) return err;
	if (err = ReadMacValue(bfpb, &fUserUnits)) return err;
	if (err = ReadMacValue(bfpb, &fArrowScale)) return err;
	if (fArrowScale==1) fArrowScale=100;
	if (err = ReadMacValue(bfpb, &fWindScale)) return err;
	//
	if (err = ReadMacValue(bfpb, &fFillValue)) return err;
	//
	if (err = ReadMacValue(bfpb, &numTimes)) goto done;	
	fTimeHdl = (Seconds**)_NewHandleClear(sizeof(Seconds)*numTimes);
	if(!fTimeHdl)
		{TechError("NetCDFWindMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < numTimes ; i++) {
		if (err = ReadMacValue(bfpb, &time)) goto done;
		INDEXH(fTimeHdl, i) = time;
	}
	if (version > 1) {if (err = ReadMacValue(bfpb, &fTimeShift)) goto done;}
	if (version > 1) {if (err = ReadMacValue(bfpb, &fAllowExtrapolationOfWinds)) goto done;}
	
	/////////////////
	if (version > 2) { ///////////////// JLM 5/3/10
		long numFiles;
		if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
		if (numFiles > 0)
		{
			fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
			if(!fInputFilesHdl)
				{TechError("NetCDFMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
			for (i = 0 ; i < numFiles ; i++) {
				PtCurFileInfo fileInfo;
				memset(&fileInfo,0,sizeof(fileInfo));
				if (err = ReadMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
				ResolvePath(fileInfo.pathName); // JLM 6/3/10
				if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
				if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
				INDEXH(fInputFilesHdl,i) = fileInfo;
			}
			if (err = ReadMacValue(bfpb, &fOverLap)) return err;
			if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
		}
	}////////////////
	
done:
	if(err)
	{
		TechError("NetCDFWindMover::Read(char* path)", " ", 0); 
		if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
	}
	return err;
}

///////////////////////////////////////////////////////////////////////////

OSErr NetCDFWindMover::TextRead(char *path) 
{
	// this code is for regular grids
	OSErr err = 0;
	long i,j, numScanned;
	int status, ncid, latid, lonid, recid, timeid;
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
	char errmsg[256] = "";
	char fileName[64],s[256],*modelTypeStr=0;
	char  outPath[256];

	if (!path || !path[0]) return 0;
	strcpy(fPathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fFileName, fileName);	// maybe use a name from the file

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
	
	/*status = nc_inq_unlimdim(ncid, &recid);	// if no unlimited dimension doesn't return error
	//if (status != NC_NOERR) {err = -1; goto done;} 
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "time", &recid); //Navy
		if (status != NC_NOERR) {err = -1; goto done;}
	}*/

	status = nc_inq_dimid(ncid, "time", &recid); //Navy
	if (status != NC_NOERR) 
	{
		status = nc_inq_unlimdim(ncid, &recid);	// issue of time not being unlimited dimension
		if (status != NC_NOERR || recid==-1) {err = -1; goto done;}
	}

	status = nc_inq_varid(ncid, "time", &timeid); 
	//if (status != NC_NOERR) {err = -1; goto done;} 
	if (status != NC_NOERR) {status = nc_inq_varid(ncid, "TIME", &timeid);if (status != NC_NOERR) {err = -1; goto done;} /*timeid = recid;*/} 	// for Ferret files, everything is in CAPS

/////////////////////////////////////////////////
	//status = nc_inq_attlen(ncid, recid, "units", &t_len);	// recid is the dimension id not the variable id
	status = nc_inq_attlen(ncid, timeid, "units", &t_len);
	if (status != NC_NOERR) 
	{
		timeUnits = 0;	// files should always have this info
		timeConversion = 3600.;		// default is hours
		startTime2 = model->GetStartTime();	// default to model start time
		/*err = -1; goto done;*/
	}
	else
	{
		DateTimeRec time;
		char unitStr[24], junk[10];
		
		timeUnits = new char[t_len+1];
		//status = nc_get_att_text(ncid, recid, "units", timeUnits);
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
		else if (numScanned!=8)	
		//if (numScanned!=8)	
			{ err = -1; TechError("NetCDFWindMover::TextRead()", "sscanf() == 8", 0); goto done; }
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
	if (status != NC_NOERR) { /*goto done;*/}	
	else 
	{
		modelTypeStr = new char[t_len2+1];
		status = nc_get_att_text(ncid, NC_GLOBAL, "generating_model", modelTypeStr);
		if (status != NC_NOERR) {/*goto done;*/}	 
		else
		{
			modelTypeStr[t_len2] = '\0';
			strcpy(fFileName, modelTypeStr); // maybe use a name from the file
		}
	}
	status = nc_inq_dimid(ncid, "lat", &latid); 
	if (status != NC_NOERR) 
	{
		status = nc_inq_dimid(ncid, "LAT", &latid);	if (status != NC_NOERR) {err = -1; goto done;}	// this is for SSH files which have LAS/ferret style caps
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
	// probabaly don't want to set based on wind ?
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
			buttonSelected  = MULTICHOICEALERT(1688,"Do you want to reset the model start time to the first time in the file?",FALSE);
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
		TechError("Error in NetCDFWindMover::TextRead()","new TRectGridVel" ,err);
		goto done;
	}

	fNumRows = latLength;
	fNumCols = lonLength;
	fGrid = (TGridVel*)rectGrid;

	rectGrid -> SetBounds(bounds); 
	status = nc_close(ncid);
	if (status != NC_NOERR) {err = -1; goto done;}

	//err = this -> SetInterval(errmsg);	// if outside of time interval, let it go
	//if(err) goto done;

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
	 	 
/////////////////////////////////////////////////

OSErr NetCDFWindMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{	
	// regular grid wind format
	OSErr err = 0;
	long i,j;
	char path[256], outPath[256]; 
	int status, ncid, numdims, numvars, uv_ndims;
	int wind_ucmp_id, wind_vcmp_id, sigma_id;
	static size_t wind_index[] = {0,0,0,0};
	static size_t wind_count[4];
	//float *wind_uvals=0,*wind_vvals=0, fill_value = -1e+10;
	double *wind_uvals=0,*wind_vvals=0, fill_value = -1e+10;
	long totalNumberOfVels = fNumRows * fNumCols;
	VelocityFH velH = 0;
	long latlength = fNumRows;
	long lonlength = fNumCols;
	//float scale_factor = 1.;
	double scale_factor = 1.;
	Boolean bHeightIncluded = false;
	
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
		// code goes here, might want to check other dimensions (lev), or just how many dimensions uv depend on
		//status = nc_inq_dimid(ncid, "sigma", &depthid);	//3D
		//if (status != NC_NOERR) bHeightIncluded = false;
		//else bHeightIncluded = true;
	}

	if (/*numdims==4*/bHeightIncluded)
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

	//wind_uvals = new float[latlength*lonlength]; 
	wind_uvals = new double[latlength*lonlength]; 
	if(!wind_uvals) {TechError("NetCDFWindMover::ReadNetCDFFile()", "new[]", 0); err = memFullErr; goto done;}
	//wind_vvals = new float[latlength*lonlength]; 
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

	
	//status = nc_get_vara_float(ncid, wind_ucmp_id, wind_index, wind_count, wind_uvals);
	status = nc_get_vara_double(ncid, wind_ucmp_id, wind_index, wind_count, wind_uvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_vara_float(ncid, wind_vcmp_id, wind_index, wind_count, wind_vvals);
	status = nc_get_vara_double(ncid, wind_vcmp_id, wind_index, wind_count, wind_vvals);
	if (status != NC_NOERR) {err = -1; goto done;}
	//status = nc_get_att_float(ncid, wind_ucmp_id, "_FillValue", &fill_value);	// should get this in text_read and store, but will have to go short to float and back
	status = nc_get_att_double(ncid, wind_ucmp_id, "_FillValue", &fill_value);	// should get this in text_read and store, but will have to go short to float and back
	//if (status != NC_NOERR) {status = nc_get_att_float(ncid, wind_ucmp_id, "FillValue", &fill_value); /*if (status != NC_NOERR) {err = -1; goto done;}*/}	// require fill value
	if (status != NC_NOERR) 
	{
		status = nc_get_att_double(ncid, wind_ucmp_id, "FillValue", &fill_value); /*if (status != NC_NOERR) {err = -1; goto done;}}*/	// require fill value
		if (status != NC_NOERR) {status = nc_get_att_double(ncid, wind_ucmp_id, "missing_value", &fill_value);} /*if (status != NC_NOERR) {err = -1; goto done;}*/
	}	// require fill value
	//if (status != NC_NOERR) {err = -1; goto done;}	// don't require fill value
	//status = nc_get_att_float(ncid, wind_ucmp_id, "scale_factor", &scale_factor);
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

Boolean NetCDFWindMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant wind, we can say "no"
	if(this->GetNumTimesInFile()==1 && !(GetNumFiles()>1)) depends = false;
	return depends;
}

void NetCDFWindMover::Draw(Rect r, WorldRect view) 
{	// Use this for regular grid
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newGridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of NetCDFWindMover
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	OSErr err = 0;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	

	if (!bShowArrows && !bShowGrid) return;
	
	bounds = rectGrid->GetBounds();
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / fNumCols) / 2;
	dLat = (WRectHeight(bounds) / fNumRows) / 2;
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&fColor);

	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);

	if (bShowArrows)
	{
		err = this -> SetInterval(errmsg);
		if(err && !bShowGrid) return;	// want to show grid even if there's no wind data
		
		loaded = this -> CheckInterval(timeDataInterval);
		if(!loaded && !bShowGrid) return;
	
		if((GetNumTimesInFile()>1 || GetNumFiles()>1) && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime + fTimeShift;
			else
				startTime = (*fTimeHdl)[fStartData.timeIndex] + fTimeShift;
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
	}	
	
	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {

			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newGridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = GetVelocityIndex(wp);	
	
				if (bShowArrows && index >= 0)
				{
					// Check for constant wind pattern 
					if((GetNumTimesInFile()==1 && !(GetNumFiles()>1)) || timeAlpha==1)
					{
						velocity.u = INDEXH(fStartData.dataHdl,index).u;
						velocity.v = INDEXH(fStartData.dataHdl,index).v;
					}
					else // time varying wind
					{
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
					}
				}
			}
	
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bShowGrid && bShowArrows && (velocity.u != 0 || velocity.v != 0)) 
				PaintRect(&c);	// should check fill_value
			if (bShowGrid && !bShowArrows) 
				PaintRect(&c);	// should check fill_value
				
			if (bShowArrows && (velocity.u != 0 || velocity.v != 0))
			{
				inchesX = (velocity.u * fWindScale) / fArrowScale;
				inchesY = (velocity.v * fWindScale) / fArrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
		}
		
	RGBForeColor(&colors[BLACK]);
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
static PopInfoRec NetCDFWindMoverPopTable[] = {
		{ M18b, nil, M18bTIMEZONEPOPUP, 0, pTIMEZONES, 0, 1, FALSE, nil },
		{ M18b, nil, M18ANGLEUNITSPOPUP, 0, pANGLEUNITS, 0, 1, FALSE, nil }
	};

static NetCDFWindMover *sharedWMover;

void ShowNetCDFWindMoverDialogItems(DialogPtr dialog)
{
	Boolean bShowGMTItems = true;
	short timeZone = GetPopSelection(dialog, M18bTIMEZONEPOPUP);
	if (timeZone == 1) bShowGMTItems = false;

	ShowHideDialogItem(dialog, M18bTIMESHIFTLABEL, bShowGMTItems); 
	ShowHideDialogItem(dialog, M18bTIMESHIFT, bShowGMTItems); 
	ShowHideDialogItem(dialog, M18bGMTOFFSETS, bShowGMTItems); 
}

short NetCDFWindClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{	
	long menuID_menuItem;
	switch (itemNum) {
		case M18OK:
		{	
			short timeZone = GetPopSelection(dialog, M18bTIMEZONEPOPUP);
			Seconds timeShift = sharedWMover->fTimeShift;
			long timeShiftInHrs;

			short angleUnits = GetPopSelection(dialog, M18ANGLEUNITSPOPUP);
			
			timeShiftInHrs = EditText2Long(dialog, M18bTIMESHIFT);
			if (timeShiftInHrs < -12 || timeShiftInHrs > 14)	// what should limits be?
			{
				printError("Time offsets must be in the range -12 : 14");
				MySelectDialogItemText(dialog, M18bTIMESHIFT,0,100);
				break;
			}

			mygetitext(dialog, M18bFILENAME, sharedWMover->fFileName, kPtCurUserNameLen-1);
			sharedWMover -> bActive = GetButton(dialog, M18ACTIVE);
			sharedWMover->bShowArrows = GetButton(dialog, M18bSHOWARROWS);
			sharedWMover->fArrowScale = EditText2Float(dialog, M18bARROWSCALE);

			sharedWMover -> fAngleScale = EditText2Float(dialog,M18ANGLESCALE);
			if (angleUnits==2) sharedWMover->fAngleScale *= PI/180.;		
			sharedWMover -> fSpeedScale = EditText2Float(dialog,M18SPEEDSCALE);
			sharedWMover -> fUncertainStartTime = (long) round(EditText2Float(dialog,M18UNCERTAINSTARTTIME)*3600);
						
			sharedWMover -> fDuration = EditText2Float(dialog, M18DURATION) * 3600;
			
			if (timeZone>1) sharedWMover->fTimeShift =(long)( EditText2Float(dialog, M18bTIMESHIFT)*3600);
			else sharedWMover->fTimeShift = 0;	// file is in local time

			//if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
			//if (timeShift != sharedWMover->fTimeShift || sharedWMover->GetTimeValue(0) != model->GetStartTime())
			// code goes here, if decide to use this check GetTimeValue is using new fTimeShift...
				//{
					//Seconds timeValZero = sharedWMover->GetTimeValue(0), startTime = model->GetStartTime();
			if (timeShift != sharedWMover->fTimeShift && sharedWMover->GetTimeValue(0) != model->GetStartTime())
			{
				//model->SetModelTime(model->GetModelTime() - (sNetCDFDialogMover->fTimeShift-timeShift));
				model->SetStartTime(model->GetStartTime() + (sharedWMover->fTimeShift-timeShift));
				//model->SetStartTime(sharedWMover->GetTimeValue(0));
				//sNetCDFDialogMover->SetInterval(errmsg);
				model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
			}
			//}
			sharedWMover->fAllowExtrapolationOfWinds = GetButton(dialog, M18bEXTRAPOLATECHECKBOX);
			// code goes here, check interval?
			return M18OK;
		}
		case M18CANCEL: return M18CANCEL;
		
		case M18ACTIVE:
		case M18bSHOWARROWS:
		case M18bEXTRAPOLATECHECKBOX:
			ToggleButton(dialog, itemNum);
			break;
		
		case M18bARROWSCALE:
		case M18DURATION:
		case M18UNCERTAINSTARTTIME:
		case M18ANGLESCALE:
		case M18SPEEDSCALE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;

		case M18ANGLEUNITSPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			break;

		case M18bTIMESHIFT:
			CheckNumberTextItemAllowingNegative(dialog, itemNum, TRUE);	// decide whether to allow half hours
			break;

		case M18bTIMEZONEPOPUP:
			PopClick(dialog, itemNum, &menuID_menuItem);
			ShowNetCDFWindMoverDialogItems(dialog);
			if (GetPopSelection(dialog, M18bTIMEZONEPOPUP) == 2) MySelectDialogItemText(dialog, M18bTIMESHIFT, 0, 100);
			break;
		
		case M18bGMTOFFSETS:
		{
			char gmtStr[512], gmtStr2[512];
			strcpy(gmtStr,"Standard time offsets from GMT\n\nHawaiian Standard -10 hrs\nAlaska Standard -9 hrs\n");
			strcat(gmtStr,"Pacific Standard -8 hrs\nMountain Standard -7 hrs\nCentral Standard -6 hrs\nEastern Standard -5 hrs\nAtlantic Standard -4 hrs\n");
			strcpy(gmtStr2,"Daylight time offsets from GMT\n\nHawaii always in standard time\nAlaska Daylight -8 hrs\n");
			strcat(gmtStr2,"Pacific Daylight -7 hrs\nMountain Daylight -6 hrs\nCentral Daylight -5 hrs\nEastern Daylight -4 hrs\nAtlantic Daylight -3 hrs");
			//printNote(gmtStr);
			short buttonSelected  = MULTICHOICEALERT2(1691,gmtStr,gmtStr2,TRUE);
			switch(buttonSelected){
				case 1:// ok
					//return -1;// 
					break;  
				case 3: // cancel
					//return -1;// 
					break;
			}
			break;
		}
	}
	
	return noErr;
}

OSErr NetCDFWindInit(DialogPtr dialog, VOIDPTR data)
{
	RegisterPopTable (NetCDFWindMoverPopTable, sizeof (NetCDFWindMoverPopTable) / sizeof (PopInfoRec));
	RegisterPopUpDialog(M18b, dialog);

	SetDialogItemHandle(dialog, M18HILITEDEFAULT, (Handle)FrameDefault);
	
	mysetitext(dialog, M18bFILENAME, sharedWMover->fFileName); // use short file name for now
	SetButton(dialog, M18ACTIVE, sharedWMover->bActive);
	
	SetPopSelection (dialog, M18ANGLEUNITSPOPUP, 1);

	if (sharedWMover->fTimeShift == 0) SetPopSelection (dialog, M18bTIMEZONEPOPUP, 1);
	else SetPopSelection (dialog, M18bTIMEZONEPOPUP, 2);
	//Long2EditText(dialog, M33TIMESHIFT, (long) (-1.*sNetCDFDialogMover->fTimeShift/3600.));
	Float2EditText(dialog, M18bTIMESHIFT, (float)(sharedWMover->fTimeShift)/3600.,1);

	SetButton(dialog, M18bSHOWARROWS, sharedWMover->bShowArrows);
	Float2EditText(dialog, M18bARROWSCALE, sharedWMover->fArrowScale, 6);

	Float2EditText(dialog, M18SPEEDSCALE, sharedWMover->fSpeedScale, 4);
	Float2EditText(dialog, M18ANGLESCALE, sharedWMover->fAngleScale, 4);

	Float2EditText(dialog, M18DURATION, sharedWMover->fDuration / 3600.0, 2);
	Float2EditText(dialog, M18UNCERTAINSTARTTIME, sharedWMover->fUncertainStartTime / 3600.0, 2);
	
	MySelectDialogItemText(dialog, M18SPEEDSCALE, 0, 255);
	//ShowHideDialogItem(dialog,M18ANGLEUNITSPOPUP,false);	// for now don't allow the units option here
	
	SetButton(dialog, M18bEXTRAPOLATECHECKBOX, sharedWMover->fAllowExtrapolationOfWinds);

	ShowNetCDFWindMoverDialogItems(dialog);
	//if (sharedWMover->fTimeShift == 0) MySelectDialogItemText(dialog, M33ALONG, 0, 100);
	//else MySelectDialogItemText(dialog, M18bTIMESHIFT, 0, 100);

	if (sharedWMover->fTimeShift != 0) MySelectDialogItemText(dialog, M18bTIMESHIFT, 0, 100);

	//SetDialogItemHandle(dialog,M18SETTINGSFRAME,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,M18UNCERTAINFRAME,(Handle)FrameEmbossed);

	return 0;
}

OSErr NetCDFWindSettingsDialog(NetCDFWindMover *mover, TMap *owner,Boolean bAddMover,WindowPtr parentWindow)
{ // Note: returns USERCANCEL when user cancels
	OSErr err = noErr;
	short item;
	
	if(!owner && bAddMover) {printError("Programmer error"); return -1;}
		
	sharedWMover = mover;			// existing mover is being edited
	
	if(parentWindow == 0) parentWindow = mapWindow; // JLM 6/2/99
	item = MyModalDialog(1825, parentWindow, 0, NetCDFWindInit, NetCDFWindClick);
	if (item == M18OK)
	{
		if (bAddMover)
		{
			err = owner -> AddMover (sharedWMover, 0);
		}
		model->NewDirtNotification();
	}
	if(item == M18CANCEL)
	{
		err = USERCANCEL;
	}
	
	return err;
}

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

LongPointHdl NetCDFWindMoverCurv::GetPointsHdl()
{
	return ((TTriGridVel*)fGrid) -> GetPointsHdl();
}

long NetCDFWindMoverCurv::GetVelocityIndex(WorldPoint wp)
{
	long index = -1;
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	return index;
}

LongPoint NetCDFWindMoverCurv::GetVelocityIndices(WorldPoint wp)
{
	LongPoint indices={-1,-1};
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		indices = ((TTriGridVel*)fGrid)->GetRectIndicesFromTriIndex(wp,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
	return indices;
}

Boolean NetCDFWindMoverCurv::VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr)
{	// code goes here, this is triangle code, not curvilinear
	char uStr[32],sStr[32],errmsg[64];
	double lengthU, lengthS;
	VelocityRec velocity = {0.,0.};
	OSErr err = 0;

	Seconds startTime, endTime, time = model->GetModelTime();
	double timeAlpha;
	long index;
	LongPoint indices;

	long ptIndex1,ptIndex2,ptIndex3; 
	InterpolationVal interpolationVal;

	// maybe should set interval right after reading the file...
	// then wouldn't have to do it here
	if (!bActive) return 0; 
	if (!bShowArrows && !bShowGrid) return 0;
	err = this -> SetInterval(errmsg);
	if(err) return false;

	if (fGrid) 
	{
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(wp.p,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
		if (index < 0) return 0;
		indices = this->GetVelocityIndices(wp.p);
	}
							
	// Check for constant current 
	if((GetNumTimesInFile()==1 /*&& !(GetNumFiles()>1)*/) || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
	//if(GetNumTimesInFile()==1)
	{
		// Calculate the interpolated velocity at the point
		if (index >= 0) 
		{
			velocity.u = INDEXH(fStartData.dataHdl,index).u;
			velocity.v = INDEXH(fStartData.dataHdl,index).v;
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
		if (index >= 0) 
		{
			velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
			velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
		}
		else	// set vel to zero
		{
			velocity.u = 0.;
			velocity.v = 0.;
		}
	}

	lengthU = sqrt(velocity.u * velocity.u + velocity.v * velocity.v);
	//lengthS = this->fWindScale * lengthU;
	lengthS = this->fWindScale * lengthU;

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

WorldPoint3D NetCDFWindMoverCurv::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	WorldPoint3D	deltaPoint = {0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double dLong, dLat;
	double timeAlpha;
	long index = -1; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec windVelocity;
	OSErr err = 0;
	char errmsg[256];
	
	
	//return deltaPoint;
	// might want to check for fFillValue and set velocity to zero - shouldn't be an issue unless we interpolate
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);
		if (err) return deltaPoint;
	}
	if (fGrid) 
		// for now just use the u,v at left and bottom midpoints of grid box as velocity over entire gridbox
		index = ((TTriGridVel*)fGrid)->GetRectIndexFromTriIndex(refPoint,fVerdatToNetCDFH,fNumCols+1);// curvilinear grid
							
	// Check for constant wind 
	//if(GetNumTimesInFile()==1)
	if(GetNumTimesInFile()==1 || (fEndData.timeIndex == UNASSIGNEDINDEX && time > ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds)  || (fEndData.timeIndex == UNASSIGNEDINDEX && time < ((*fTimeHdl)[fStartData.timeIndex] + fTimeShift) && fAllowExtrapolationOfWinds))
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

scale:

	windVelocity.u *= fWindScale; // may want to allow some sort of scale factor, though should be in file
	windVelocity.v *= fWindScale; 
	
	
	if(leType == UNCERTAINTY_LE)
	{
		 err = AddUncertainty(setIndex,leIndex,&windVelocity);
	}
	
	windVelocity.u *=  (*theLE).windage;
	windVelocity.v *=  (*theLE).windage;

	dLong = ((windVelocity.u / METERSPERDEGREELAT) * timeStep) / LongToLatRatio3 (refPoint.pLat);
	dLat  =  (windVelocity.v / METERSPERDEGREELAT) * timeStep;

	deltaPoint.p.pLong = dLong * 1000000;
	deltaPoint.p.pLat  = dLat  * 1000000;

	return deltaPoint;
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

OSErr NetCDFWindMoverCurv::TextRead(char *path, TMap **newMap) // don't want a map  
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
	char errmsg[256] = "";
	char fileName[64],*modelTypeStr=0;
	Point where;
	OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
	MySFReply reply;
	Boolean bTopFile = false, fIsNavy = false;	// for now keep code around but probably don't need Navy curvilinear wind
	VelocityFH velocityH = 0;
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
		if (numScanned!=8)	
		{ 
			timeUnits = 0;	// files should always have this info
			timeConversion = 3600.;		// default is hours
			startTime2 = model->GetStartTime();	// default to model start time
			/*err = -1; TechError("NetCDFWindMoverCurv::TextRead()", "sscanf() == 8", 0); goto done;*/
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
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}
		else
			strcpy(topPath, reply.fullPath);

		/*{
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = ReorderPoints(velocityH,newMap,errmsg);	
			//err = ReorderPoints(fStartData.dataHdl,newMap,errmsg);	// if u, v input separately only do this once?
	 		goto done;
		}*/
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
			if (recs>0)
				err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
			else {strcpy(errmsg,"No times in file. Error opening NetCDF file"); err =  -1;}
			if(err) goto done;
			err = ReorderPoints(velocityH,newMap,errmsg);	
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

	if (recs>0)
		err = this -> ReadTimeData(indexOfStart,&velocityH,errmsg);
	else {strcpy(errmsg,"No times in file. Error opening NetCDF wind file"); err =  -1;}
	if(err) goto done;
	err = ReorderPoints(velocityH,newMap,errmsg);	
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
		if(vertexPtsH) {DisposeHandle((Handle)vertexPtsH); vertexPtsH = 0;}
	}

	if (timeUnits) delete [] timeUnits;
	if (lat_vals) delete [] lat_vals;
	if (lon_vals) delete [] lon_vals;
	if (modelTypeStr) delete [] modelTypeStr;
	if (velocityH) {DisposeHandle((Handle)velocityH); velocityH = 0;}
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

long NetCDFWindMoverCurv::CheckSurroundingPoints(LONGH maskH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd, lowestLandIndex = 0;
	long neighbor;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows - 1) ? row + 1 : fNumRows - 1;
	jEnd = (col < fNumCols - 1) ? col + 1 : fNumCols - 1;
	// don't allow diagonals for now,they could be separate small islands 
	/*for (i = iStart; i< iEnd+1; i++)
	{
		for (j = jStart; j< jEnd+1; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols + j);
			if (neighbor >= 3 && neighbor < lowestLandIndex)
				lowestLandIndex = neighbor;
		}
	}*/
	for (i = iStart; i< iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols + col);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols + j);
		if (neighbor >= 3 && neighbor < lowestLandIndex)
			lowestLandIndex = neighbor;
	}
	return lowestLandIndex;
}
Boolean NetCDFWindMoverCurv::ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd;
	long neighbor;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows - 1) ? row + 1 : fNumRows - 1;
	jEnd = (col < fNumCols - 1) ? col + 1 : fNumCols - 1;
	/*for (i = iStart; i < iEnd+1; i++)
	{
		for (j = jStart; j < jEnd+1; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols + j);
			// eventually should use a land mask or fill value to identify land
			if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==0. && INDEXH(velocityH,i*fNumCols+j).v==0.)) return true;
		}
	}*/
	for (i = iStart; i < iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols + col);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+col).u==fFillValue && INDEXH(velocityH,i*fNumCols+col).v==fFillValue)) return true;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols + j);
		//if (neighbor >= 3 || (INDEXH(velocityH,i*fNumCols+j).u==fFillValue && INDEXH(velocityH,i*fNumCols+j).v==fFillValue)) return true;
		if (neighbor >= 3 || (INDEXH(velocityH,row*fNumCols+j).u==fFillValue && INDEXH(velocityH,row*fNumCols+j).v==fFillValue)) return true;
	}
	return false;
}

Boolean NetCDFWindMoverCurv::InteriorLandPoint(LONGH maskH, long row, long col) 
{
	long i, j, iStart, iEnd, jStart, jEnd;
	long neighbor;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows_ext - 1) ? row + 1 : fNumRows_ext - 1;
	jEnd = (col < fNumCols_ext - 1) ? col + 1 : fNumCols_ext - 1;
	/*for (i = iStart; i < iEnd+1; i++)
	{
		if (i==row) continue;
		neighbor = INDEXH(maskH, i*fNumCols_ext + col);
		if (neighbor < 3)	// water point
			return false;
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols_ext + j);
		if (neighbor < 3)	// water point
			return false;
	}*/
	//for (i = iStart; i < iEnd+1; i++)
	// point is in lower left corner of grid box (land), so only check 3 other quadrants of surrounding 'square'
	for (i = row; i < iEnd+1; i++)
	{
		//for (j = jStart; j< jEnd+1; j++)
		for (j = jStart; j< jEnd; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + j);
			if (neighbor < 3 /*&& neighbor != -1*/)	// water point
				return false;
			//if (row==1 && INDEXH(maskH,j)==1) return false;
		}
	}
	return true;
}

Boolean NetCDFWindMoverCurv::ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) 
{
	long iStart, iEnd, jStart, jEnd;
	long i, j, neighbor, landPolyNum;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;

	iStart = (row > 0) ? row - 1 : 0;
	jStart = (col > 0) ? col - 1 : 0;
	iEnd = (row < fNumRows_ext - 1) ? row + 1 : fNumRows_ext - 1;
	jEnd = (col < fNumCols_ext - 1) ? col + 1 : fNumCols_ext - 1;
	
	landPolyNum = INDEXH(maskH, row*fNumCols_ext + col);
	for (i = iStart; i< iEnd+1; i++)
	{
			if (i==row) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + col);
			if (neighbor >= 3 && neighbor < landPolyNum) 
			{
				*lowerPolyNum = neighbor;
				return true;
			}
	}
	for (j = jStart; j< jEnd+1; j++)
	{	
		if (j==col) continue;
		neighbor = INDEXH(maskH, row*fNumCols_ext + j);
		if (neighbor >= 3 && neighbor < landPolyNum) 
		{
			*lowerPolyNum = neighbor;
			return true;
		}
	}
	// don't allow diagonals for now, they could be separate small islands
	/*for (i = iStart; i< iEnd+1; i++)
	{
		for (j = jStart; j< jEnd+1; j++)
		{	
			if (i==row && j==col) continue;
			neighbor = INDEXH(maskH, i*fNumCols_ext + j);
			if (neighbor >= 3 && neighbor < landPolyNum) 
			{
				*lowerPolyNum = neighbor;
				return true;
			}
		}
	}*/
	return false;
}

void NetCDFWindMoverCurv::ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin)
{	// merges adjoining land blocks and then renumbers any higher numbered land blocks
	long i,j,val;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{	
			val = INDEXH(maskH,i*fNumCols_ext+j);
			if (val==landBlockToMerge) INDEXH(maskH,i*fNumCols_ext+j) = landBlockToJoin;
			if (val>landBlockToMerge) INDEXH(maskH,i*fNumCols_ext+j) -= 1;
		}
	}
}

OSErr NetCDFWindMoverCurv::NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo, long *numIslands) 
{
	OSErr err = 0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long i, j, n, landPolyNum = 1, lowestSurroundingNum = 0;
	long islandNum, maxIslandNum=3;
	LONGH maskH = (LONGH)_NewHandleClear(fNumRows * fNumCols * sizeof(long));
	LONGH maskH2 = (LONGH)_NewHandleClear(nv_ext * sizeof(long));
	*islandNumberH = 0;
	
	if (!maskH || !maskH2) {err = memFullErr; goto done;}
	// use surface velocity values at time zero
	for (i=0;i<fNumRows;i++)
	{
		for (j=0;j<fNumCols;j++)
		{
			if (INDEXH(landWaterInfo,i*fNumCols+j) == -1)// 1 water, -1 land
			{
				if (i==0 || i==fNumRows-1 || j==0 || j==fNumCols-1)
				{
					INDEXH(maskH,i*fNumCols+j) = 3;	// set outer boundary to 3
				}
				else
				{
					if (landPolyNum==1)
					{	// Land point
						INDEXH(maskH,i*fNumCols+j) = landPolyNum+3;
						landPolyNum+=3;
					}
					else
					{
						// check for nearest land poly number
						if (lowestSurroundingNum = CheckSurroundingPoints(maskH,i,j)>=3)
						{
							INDEXH(maskH,i*fNumCols+j) = lowestSurroundingNum;
						}
						else
						{
							INDEXH(maskH,i*fNumCols+j) = landPolyNum;
							landPolyNum += 1;
						}
					}
				}
			}
			else
			{
				if (i==0 || i==fNumRows-1 || j==0 || j==fNumCols-1)
					INDEXH(maskH,i*fNumCols+j) = 1;	// Open water boundary
				else if (ThereIsAdjacentLand2(maskH,velocityH,i,j))
					INDEXH(maskH,i*fNumCols+j) = 2;	// Water boundary, not open water
				else
					INDEXH(maskH,i*fNumCols+j) = 0;	// Interior water point
			}
		}
	}
	// extend grid by one row/col up/right since velocities correspond to lower left corner of a grid box
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{
			if (i==0) 
			{
				if (j!=fNumCols)
					INDEXH(maskH2,j) = INDEXH(maskH,j);	// flag for extra boundary point
				else
					INDEXH(maskH2,j) = INDEXH(maskH,j-1);	
					
			}
			else if (i!=0 && j==fNumCols) 
				INDEXH(maskH2,i*fNumCols_ext+fNumCols) = INDEXH(maskH,(i-1)*fNumCols+fNumCols-1);
			else 
			{	
				INDEXH(maskH2,i*fNumCols_ext+j) = INDEXH(maskH,(i-1)*fNumCols+j);
			}
		}
	}

	// set original top/right boundaries to interior water points 
	// probably don't need to do this since we aren't paying attention to water types anymore
	for (j=1;j<fNumCols_ext-1;j++)	 
	{
		if (INDEXH(maskH2,fNumCols_ext+j)==1) INDEXH(maskH2,fNumCols_ext+j) = 2;
	}
	for (i=1;i<fNumRows_ext-1;i++)
	{
		if (INDEXH(maskH2,i*fNumCols_ext+fNumCols-1)==1) INDEXH(maskH2,i*fNumCols_ext+fNumCols-1) = 2;
	}
	// now merge any contiguous land blocks (max of landPolyNum)
	// as soon as find one, all others of that number change, and every higher landpoint changes
	// repeat until nothing changes
startLoop:
	{
		long lowerPolyNum = 0;
		for (i=0;i<fNumRows_ext;i++)
		{
			for (j=0;j<fNumCols_ext;j++)
			{
				if (INDEXH(maskH2,i*fNumCols_ext+j) < 3) continue;	// water point
				if (ThereIsALowerLandNeighbor(maskH2,&lowerPolyNum,i,j))
				{
					ResetMaskValues(maskH2,INDEXH(maskH2,i*fNumCols_ext+j),lowerPolyNum);
					goto startLoop;
				}
				if ((i==0 || i==fNumRows_ext-1 || j==0 || j==fNumCols_ext-1) && INDEXH(maskH2,i*fNumCols_ext+j)>3)
				{	// shouldn't get here
					ResetMaskValues(maskH2,INDEXH(maskH2,i*fNumCols_ext+j),3);
					goto startLoop;
				}
			}
		}
	}
	for (i=0;i<fNumRows_ext;i++)
	{
		for (j=0;j<fNumCols_ext;j++)
		{	// note, the numbers start at 3
			islandNum = INDEXH(maskH2,i*fNumCols_ext+j);
			if (islandNum < 3) continue;	// water point
			if (islandNum > maxIslandNum) maxIslandNum = islandNum;
		}
	}
	*islandNumberH = maskH2;
	*numIslands = maxIslandNum;
done:
	if (err) 
	{
		printError("Error numbering islands for map boundaries");
		if (maskH2) {DisposeHandle((Handle)maskH2); maskH2 = 0;}
	}
	if (maskH) {DisposeHandle((Handle)maskH); maskH = 0;}
	return err;
}

/*OSErr NetCDFWindMoverCurv::ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long currentIsland=0, islandNum, nBoundaryPts=0, nEndPts=0, waterStartPoint;
	long nSegs, segNum = 0, numIslands, rectIndex; 
	long iIndex, jIndex, index, currentIndex, startIndex; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
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

	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	

	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}

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
	if (this -> moverMap != model -> uMap) goto setFields;	// don't try to create a map
	/////////////////////////////////////////////////
	// go through topo look for -1, and list corresponding boundary sides
	// then reorder as contiguous boundary segments - need to group boundary rects by islands
	// will need a new field for list of boundary points since there can be duplicates, can't just order and list segment endpoints

	nSegs = 2*ntri; //number of -1's in topo
	boundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryPtsH));
	boundaryEndPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**boundaryEndPtsH));
	waterBoundaryPtsH = (LONGH)_NewHandleClear(nv_ext * sizeof(**waterBoundaryPtsH));
	segUsed = (Boolean**)_NewHandleClear(nSegs * sizeof(Boolean));
	segList = (SegInfoHdl)_NewHandleClear(nSegs * sizeof(**segList));
	// first go through rectangles and group by island
	// do this before making dagtree, 
	MySpinCursor(); // JLM 8/4/99
	err = NumberIslands(&maskH2, velocityH, landWaterInfo, &numIslands);	// numbers start at 3 (outer boundary)
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
		(*waterBoundaryPtsH)[nBoundaryPts] = (*segList)[i].isWater+1;
		(*boundaryPtsH)[nBoundaryPts++] = (*segList)[i].pt2;
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

/////////////////////////////////////////////////
	
	triGrid = new TTriGridVel;
	if (!triGrid)
	{		
		err = true;
		TechError("Error in NetCDFWindMoverCurv::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;

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

	// probably will assume wind goes on a map
	//if (waterBoundaryPtsH && this -> moverMap == model -> uMap)	// maybe assume rectangle grids will have map?
	//{
		//PtCurMap *map = CreateAndInitPtCurMap(fPathName,triBounds); // the map bounds are the same as the grid bounds
		//if (!map) {err=-1; goto done;}
		// maybe move up and have the map read in the boundary information
		//map->SetBoundarySegs(boundaryEndPtsH);	
		//map->SetWaterBoundaries(waterBoundaryPtsH);
		//map->SetBoundaryPoints(boundaryPtsH);

		// *newMap = map;
	//}
	//else
	//{
		if (waterBoundaryPtsH) {DisposeHandle((Handle)waterBoundaryPtsH); waterBoundaryPtsH=0;}
		if (boundaryEndPtsH) {DisposeHandle((Handle)boundaryEndPtsH); boundaryEndPtsH=0;}
		if (boundaryPtsH) {DisposeHandle((Handle)boundaryPtsH); boundaryPtsH=0;}
	//}

	/////////////////////////////////////////////////
done:
		if (landWaterInfo) {DisposeHandle((Handle)landWaterInfo); landWaterInfo=0;}
		if (ptIndexHdl) {DisposeHandle((Handle)ptIndexHdl); ptIndexHdl = 0;}
		if (gridCellInfo) {DisposeHandle((Handle)gridCellInfo); gridCellInfo = 0;}
		if (segUsed) {DisposeHandle((Handle)segUsed); segUsed = 0;}
		if (segList) {DisposeHandle((Handle)segList); segList = 0;}

	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in NetCDFWindMoverCurv::ReorderPoints");
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
}*/
// simplify for wind data - no map needed, no mask 
OSErr NetCDFWindMoverCurv::ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg) 
{
	long i, j, n, ntri, numVerdatPts=0;
	long fNumRows_ext = fNumRows+1, fNumCols_ext = fNumCols+1;
	long nv = fNumRows * fNumCols, nv_ext = fNumRows_ext*fNumCols_ext;
	long iIndex, jIndex, index; 
	long triIndex1, triIndex2, waterCellNum=0;
	long ptIndex = 0, cellNum = 0;
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

	TTriGridVel *triGrid = nil;
	tree.treeHdl = 0;
	TDagTree *dagTree = 0;
	

	if (!landWaterInfo || !ptIndexHdl || !gridCellInfo || !verdatPtsH || !maskH2) {err = memFullErr; goto done;}

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
		TechError("Error in NetCDFWindMoverCurv::ReorderPoints()","new TTriGridVel",err);
		goto done;
	}

	fGrid = (TTriGridVel*)triGrid;

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
			strcpy(errmsg,"An error occurred in NetCDFWindMoverCurv::ReorderPoints");
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
	}
	return err;
}

OSErr NetCDFWindMoverCurv::GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp)
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

			err = this -> SetInterval(errmsg);
			if(err) return;
			
			loaded = this -> CheckInterval(timeDataInterval);
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
				if (GetLatLonFromIndex(iIndex-1,jIndex+1,&wp2)!=-1)	// may want to get all four points and interpolate
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
	if (bShowGrid) fGrid->Draw(r,view,wayOffMapPt,fWindScale,fArrowScale,false,true);
		
	RGBForeColor(&colors[BLACK]);
}

/////////////////////////////////////////////////////////////////
OSErr NetCDFWindMoverCurv::ReadTopology(char* path, TMap **newMap)
{
	// import NetCDF curvilinear info so don't have to regenerate
	char s[1024], errmsg[256]/*, s[256], topPath[256]*/;
	long i, numPoints, numTopoPoints, line = 0, numPts;
	CHARH f = 0;
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
		

	if (!path || !path[0]) return 0;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) {
		TechError("NetCDFWindMover::ReadTopology()", "ReadFileContents()", err);
		goto done;
	}
	
	_HLock((Handle)f); // JLM 8/4/99
	
	// No header
	// start with transformation array and vertices
	MySpinCursor(); // JLM 8/4/99
	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	if(IsTransposeArrayHeaderLine(s,&numPts)) // 
	{
		if (err = ReadTransposeArray(f,&line,&fVerdatToNetCDFH,numPts,errmsg)) 
		{strcpy(errmsg,"Error in ReadTransposeArray"); goto done;}
	}
	else {err=-1; strcpy(errmsg,"Error in Transpose header line"); goto done;}

	if(err = ReadTVertices(f,&line,&pts,&depths,errmsg)) goto done;

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

	NthLineInTextOptimized(*f, (line)++, s, 1024); 
	//code goes here, boundary points
	if(IsBoundarySegmentHeaderLine(s,&numBoundarySegs)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundarySegs>0)
			err = ReadBoundarySegs(f,&line,&boundarySegs,numBoundarySegs,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not needed for 2D files, but we require for now
	}
	MySpinCursor(); // JLM 8/4/99

	if(IsWaterBoundaryHeaderLine(s,&numWaterBoundaries,&numBoundaryPts)) // Boundary types from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			err = ReadWaterBoundaries(f,&line,&waterBoundaries,numWaterBoundaries,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
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
	
	if(IsBoundaryPointsHeaderLine(s,&numBoundaryPts)) // Boundary data from CATs
	{
		MySpinCursor();
		if (numBoundaryPts>0)
			err = ReadBoundaryPts(f,&line,&boundaryPts,numBoundaryPts,errmsg);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
		//err = -1;
		//strcpy(errmsg,"Error in Boundary segment header line");
		//goto done;
		// not always needed ? probably always needed for curvilinear
	}
	MySpinCursor(); // JLM 8/4/99

	if(IsTTopologyHeaderLine(s,&numTopoPoints)) // Topology from CATs
	{
		MySpinCursor();
		err = ReadTTopologyBody(f,&line,&topo,&velH,errmsg,numTopoPoints,FALSE);
		if(err) goto done;
		NthLineInTextOptimized(*f, (line)++, s, 1024); 
	}
	else
	{
			err = -1; // for now we require TTopology
			strcpy(errmsg,"Error in topology header line");
			if(err) goto done;
	}
	MySpinCursor(); // JLM 8/4/99


	//NthLineInTextOptimized(*f, (line)++, s, 1024); 
	
	if(IsTIndexedDagTreeHeaderLine(s,&numPoints))  // DagTree from CATs
	{
		MySpinCursor();
		err = ReadTIndexedDagTreeBody(f,&line,&tree,errmsg,numPoints);
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
	if(f) 
	{
		_HUnlock((Handle)f); 
		DisposeHandle((Handle)f); 
		f = 0;
	}

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

enum {
	   I_GRIDWINDNAME = 0, I_GRIDWINDACTIVE, I_GRIDWINDSHOWGRID, I_GRIDWINDSHOWARROWS, I_GRIDWINDUNCERTAIN,
	   I_GRIDWINDSPEEDSCALE,I_GRIDWINDANGLESCALE, I_GRIDWINDSTARTTIME,I_GRIDWINDDURATION
		};


///////////////////////////////////////////////////////////////////////////

long GridWindMover::GetListLength()
{
	long count = 1; // wind name
	long mode = model->GetModelMode();
	
	if (bOpen) {
		if(mode == ADVANCEDMODE) count += 1; // active
		if(mode == ADVANCEDMODE) count += 1; // showgrid
		if(mode == ADVANCEDMODE) count += 1; // showarrows
		if(mode == ADVANCEDMODE && model->IsUncertain())count++;
		if(mode == ADVANCEDMODE && model->IsUncertain() && bUncertaintyPointOpen)count+=4;
	}
	
	return count;
}

ListItem GridWindMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { this, n, indent, 0 };
	long mode = model->GetModelMode();
	
	if (n == 0) {
		item.index = I_GRIDWINDNAME;
		if (mode == ADVANCEDMODE) item.bullet = bOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
		strcpy(text,"Wind File: ");
		strcat(text,fFileName);
		if(!bActive)*style = italic; // JLM 6/14/10
	
		return item;
	}
	
	if (bOpen) {
	
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_GRIDWINDACTIVE;
			item.bullet = bActive ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Active");
			
			return item;
		}
				
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_GRIDWINDSHOWGRID;
			item.bullet = bShowGrid ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Grid");
			
			return item;
		}
				
		if (mode == ADVANCEDMODE && --n == 0) {
			item.indent++;
			item.index = I_GRIDWINDSHOWARROWS;
			item.bullet = bShowArrows ? BULLET_FILLEDBOX : BULLET_EMPTYBOX;
			strcpy(text, "Show Velocity Vectors");
			
			return item;
		}
						
		if(mode == ADVANCEDMODE && model->IsUncertain())
		{
			if (--n == 0) 
			{
				item.indent++;
				item.index = I_GRIDWINDUNCERTAIN;
				item.bullet = bUncertaintyPointOpen ? BULLET_OPENTRIANGLE : BULLET_CLOSEDTRIANGLE;
				strcpy(text, "Uncertainty");
				
				return item;
			}
			
			if(bUncertaintyPointOpen)
			{
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_GRIDWINDSTARTTIME;
					sprintf(text, "Start Time: %.2f hours",fUncertainStartTime/3600);
					return item;
				}
			
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_GRIDWINDDURATION;
					sprintf(text, "Duration: %.2f hr", (float)(fDuration / 3600.0));
					return item;
				}
				
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_GRIDWINDSPEEDSCALE;
					sprintf(text, "Speed Scale: %.2f ", fSpeedScale);
					return item;
				}
				if (--n == 0) 
				{
					item.indent++;
					item.index = I_GRIDWINDANGLESCALE;
					sprintf(text, "Angle Scale: %.2f ", fAngleScale);
					return item;
				}
			}
		}
	}
	
	item.owner = 0;
	
	return item;
}

Boolean GridWindMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
{
	if (inBullet)
		switch (item.index) {
			case I_GRIDWINDNAME: bOpen = !bOpen; return TRUE;
			case I_GRIDWINDACTIVE: bActive = !bActive; 
					model->NewDirtNotification(); return TRUE;
		case I_GRIDWINDSHOWGRID: bShowGrid = !bShowGrid; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_GRIDWINDSHOWARROWS: bShowArrows = !bShowArrows; 
				model->NewDirtNotification(DIRTY_MAPDRAWINGRECT); return TRUE;
			case I_GRIDWINDUNCERTAIN:bUncertaintyPointOpen = !bUncertaintyPointOpen;return TRUE;
		}
	
	if (ShiftKeyDown() && item.index == I_GRIDWINDNAME) {
		fColor = MyPickColor(fColor,mapWindow);
		model->NewDirtNotification(DIRTY_LIST|DIRTY_MAPDRAWINGRECT);
	}

	if (doubleClick)
	{	
		switch(item.index)
		{
			case I_GRIDWINDACTIVE:
			case I_GRIDWINDUNCERTAIN:
			case I_GRIDWINDSPEEDSCALE:
			case I_GRIDWINDANGLESCALE:
			case I_GRIDWINDSTARTTIME:
			case I_GRIDWINDDURATION:
			case I_GRIDWINDNAME:
				//GridWindSettingsDialog(this, this -> moverMap,false,mapWindow);
				WindSettingsDialog(this, this -> moverMap,false,mapWindow,false);
				break;
			default:
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean GridWindMover::FunctionEnabled(ListItem item, short buttonID)
{
	long i;
	switch (item.index) {
		case I_GRIDWINDNAME:
			switch (buttonID) {
				case ADDBUTTON: return FALSE;
				case DELETEBUTTON:
					if(model->GetModelMode() < ADVANCEDMODE) return FALSE; // shouldn't happen
					else return TRUE;
				case UPBUTTON:
				case DOWNBUTTON:
					if (!moverMap->moverList->IsItemInList((Ptr)&item.owner, &i)) return FALSE;
					switch (buttonID) {
						case UPBUTTON: return i > 0;
						case DOWNBUTTON: return i < (moverMap->moverList->GetItemCount() - 1);
					}
			}
			break;
	}
	
	if (buttonID == SETTINGSBUTTON) return TRUE;
	
	return TWindMover::FunctionEnabled(item, buttonID);
}

OSErr GridWindMover::SettingsItem(ListItem item)
{
	//return GridWindSettingsDialog(this, this -> moverMap,false,mapWindow);
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = ListClick(item,inBullet,doubleClick);
	return 0;
}

OSErr GridWindMover::DeleteItem(ListItem item)
{
	if (item.index == I_GRIDWINDNAME)
		return moverMap -> DropMover(this);
	
	return 0;
}

GridWindMover::GridWindMover(TMap *owner,char* name) : TWindMover(owner, name)
{
	if(!name || !name[0]) this->SetClassName("Grid Wind");
	else 	SetClassName (name); // short file name
	
	// use wind defaults for uncertainty
	bShowGrid = false;
	bShowArrows = false;

	fGrid = 0;
	fTimeDataHdl = 0;
	fIsOptimizedForStep = false;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;
	
	//fUserUnits = kMetersPerSec;	
	fUserUnits = kUndefined;	
	fWindScale = 1.;		// not using this
	fArrowScale = 1.;		// not using this
	//fFillValue = -1e+34;

	memset(&fStartData,0,sizeof(fStartData));
	fStartData.timeIndex = UNASSIGNEDINDEX; 
	fStartData.dataHdl = 0; 
	memset(&fEndData,0,sizeof(fEndData));
	fEndData.timeIndex = UNASSIGNEDINDEX;
	fEndData.dataHdl = 0;
	
	fInputFilesHdl = 0;	// for multiple files case
}

long GridWindMover::GetVelocityIndex(WorldPoint p) 
{
	long rowNum, colNum;
	VelocityRec	velocity;
	
	LongRect gridLRect, geoRect;
	ScaleRec	thisScaleRec;

	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	// fNumRows, fNumCols members of GridWindMover

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

void GridWindMover::Dispose()
{
	if (fGrid)
	{
		fGrid -> Dispose();
		delete fGrid;
		fGrid = nil;
	}

	if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
	if(fStartData.dataHdl)DisposeLoadedData(&fStartData); 
	if(fEndData.dataHdl)DisposeLoadedData(&fEndData);

	TWindMover::Dispose ();
}

///////////////////////////////////////////////////////////////////////////
OSErr GridWindMover::CheckAndPassOnMessage(TModelMessage *message)
{	// JLM

/*	char ourName[kMaxNameLen];
	
	// see if the message is of concern to us
	
	this->GetClassName(ourName);
	if(message->IsMessage(M_SETFIELD,ourName))
	{
		double val;
		char str[256];
		OSErr err = 0;
		////////////////
		err = message->GetParameterAsDouble("uncertaintySpeedScale",&val);
		if(!err) this->fSpeedScale = val; 
		////////////////
		err = message->GetParameterAsDouble("uncertaintyAngleScale",&val);
		if(!err) this->fAngleScale = val; 
		////////////////
		model->NewDirtNotification();// tell model about dirt
	}*/
	
	/////////////////////////////////////////////////
	//  pass on this message to our base class
	/////////////////////////////////////////////////
	return TWindMover::CheckAndPassOnMessage(message);
}

OSErr GridWindMover::PrepareForModelStep()
{
	OSErr err = this->UpdateUncertainty();

	char errmsg[256];
	
	errmsg[0]=0;

	if (!bActive) return noErr;

	err = this -> SetInterval(errmsg); // SetInterval checks to see that the time interval is loaded
	if (err) goto done;	// might not want to have error if outside time interval

	fIsOptimizedForStep = true;	// is this needed?
	
done:
	
	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridWindMover::PrepareForModelStep");
		printError(errmsg); 
	}	

	return err;
}

void GridWindMover::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
}

Boolean GridWindMover::CheckInterval(long &timeDataInterval)
{
	Seconds time =  model->GetModelTime();
	long i,numTimes;

	numTimes = this -> GetNumTimesInFile(); 

	// check for constant wind
	if (numTimes==1) 
	{
		timeDataInterval = -1; // some flag here
		if(fStartData.timeIndex==0 && fStartData.dataHdl)
				return true;
		else
			return false;
	}
	
	if(fStartData.timeIndex!=UNASSIGNEDINDEX && fEndData.timeIndex!=UNASSIGNEDINDEX)
	{
		if (time>=(*fTimeDataHdl)[fStartData.timeIndex].time && time<=(*fTimeDataHdl)[fEndData.timeIndex].time)
				{	// we already have the right interval loaded
					timeDataInterval = fEndData.timeIndex;
					return true;
				}
	}

	if (GetNumFiles()>1 && fOverLap)
	{	
		if (time>=fOverLapStartTime && time<=(*fTimeDataHdl)[fEndData.timeIndex].time)
			return true;	// we already have the right interval loaded, time is in between two files
		else fOverLap = false;
	}
	
	for (i=0;i<numTimes;i++) 
	{	// find the time interval
		if (time>=(*fTimeDataHdl)[i].time && time<=(*fTimeDataHdl)[i+1].time)
		{
			timeDataInterval = i+1; // first interval is between 0 and 1, and so on
			return false;
		}
	}	
	// don't allow time before first or after last
	if (time<(*fTimeDataHdl)[0].time) 
		timeDataInterval = 0;
	if (time>(*fTimeDataHdl)[numTimes-1].time) 
		timeDataInterval = numTimes;

	return false;		
}

void GridWindMover::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void GridWindMover::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long GridWindMover::GetNumTimesInFile()
{
	long numTimes;

	numTimes = _GetHandleSize((Handle)fTimeDataHdl)/sizeof(**fTimeDataHdl);
	return numTimes;     
}

long GridWindMover::GetNumFiles()
{
	long numFiles = 0;

	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

OSErr GridWindMover::SetInterval(char *errmsg)
{
	long timeDataInterval;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval);
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
		
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
		
	// check for constant wind 
	if(numTimesInFile==1)	//or if(timeDataInterval==-1) 
	{
		indexOfStart = 0;
		indexOfEnd = UNASSIGNEDINDEX;	// should already be -1
	}
	
	if(timeDataInterval == 0)
	{	// before the first step in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}
	else if(timeDataInterval == numTimesInFile) 
	{	// past the last information in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}
	else // load the two intervals
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
		
		if(indexOfEnd < numTimesInFile && indexOfEnd != UNASSIGNEDINDEX)  // not past the last interval and not constant wind
		{
			err = this -> ReadTimeData(indexOfEnd,&fEndData.dataHdl,errmsg);
			if(err) goto done;
			fEndData.timeIndex = indexOfEnd;
		}
	}
	
done:	
	if(err)
	{
		if(!errmsg[0])strcpy(errmsg,"Error in GridWindMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}

	return err;
}

WorldPoint3D GridWindMover::GetMove(Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType)
{
	double dLong, dLat;
	WorldPoint3D deltaPoint ={0,0,0.};
	WorldPoint refPoint = (*theLE).p;	
	double timeAlpha;
	long index; 
	Seconds startTime,endTime;
	Seconds time = model->GetModelTime();
	VelocityRec windVelocity;
	OSErr err = noErr;
	char errmsg[256];
	
	// if ((*theLE).z > 0) return deltaPoint; // wind doesn't act below surface
	// or use some sort of exponential decay below the surface...
	
	if(!fIsOptimizedForStep) 
	{
		err = this -> SetInterval(errmsg);	// ok, but don't print error message here
		if (err) return deltaPoint;
	}
	index = GetVelocityIndex(refPoint);  // regular grid
							
	// Check for constant wind 
	if(GetNumTimesInFile()==1)
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
			startTime = fOverLapStartTime;
		else
			startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		//startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
		endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
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

//scale:

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

#define GridWindMoverREADWRITEVERSION 1 //JLM	7/10/01

OSErr GridWindMover::Write (BFPB *bfpb)
{
	char c;
	long i, version = GridWindMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	VelocityRec velocity;
	long amtTimeData = GetNumTimesInFile();
	long numPoints, numFiles;
	float val;
	PtCurTimeData timeData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;

	if (err = TWindMover::Write (bfpb)) return err;

	StartReadWriteSequence("GridWindMover::Write()");
	if (err = WriteMacValue(bfpb, id)) return err;
	if (err = WriteMacValue(bfpb, version)) return err;
	////
	id = fGrid -> GetClassID (); //JLM
	if (err = WriteMacValue(bfpb, id)) return err; //JLM
	if (err = fGrid -> Write (bfpb)) goto done;

	if (err = WriteMacValue(bfpb, bShowGrid)) goto done;
	if (err = WriteMacValue(bfpb, bShowArrows)) goto done;
	if (err = WriteMacValue(bfpb, fUserUnits)) goto done;
	//if (err = WriteMacValue(bfpb, fWindScale)) goto done; // not using this
	//if (err = WriteMacValue(bfpb, fArrowScale)) goto done; // not using this

	if (err = WriteMacValue(bfpb, fNumRows)) goto done;
	if (err = WriteMacValue(bfpb, fNumCols)) goto done;
	if (err = WriteMacValue(bfpb, fPathName, kMaxNameLen)) goto done;
	if (err = WriteMacValue(bfpb, fFileName, kPtCurUserNameLen)) goto done;

	if (err = WriteMacValue(bfpb, amtTimeData)) goto done;
	for (i=0;i<amtTimeData;i++)
	{
		timeData = INDEXH(fTimeDataHdl,i);
		if (err = WriteMacValue(bfpb, timeData.fileOffsetToStartOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.lengthOfData)) goto done;
		if (err = WriteMacValue(bfpb, timeData.time)) goto done;
	}

	numFiles = GetNumFiles();
	if (err = WriteMacValue(bfpb, numFiles)) goto done;
	if (numFiles > 0)
	{
		for (i = 0 ; i < numFiles ; i++) {
			fileInfo = INDEXH(fInputFilesHdl,i);
			if (err = WriteMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
			if (err = WriteMacValue(bfpb, fileInfo.startTime)) goto done;
			if (err = WriteMacValue(bfpb, fileInfo.endTime)) goto done;
		}
		if (err = WriteMacValue(bfpb, fOverLap)) return err;
		if (err = WriteMacValue(bfpb, fOverLapStartTime)) return err;
	}
	
done:
	if(err)
		TechError("GridWindMover::Write(char* path)", " ", 0); 

	return err;
}

OSErr GridWindMover::Read(BFPB *bfpb)
{
	char c;
	long i, version, amtTimeData, numPoint, numFiles;
	ClassID id;
	float val;
	PtCurTimeData timeData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TWindMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("GridWindMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("GridWindMover::Read()", "id != TYPE_GRIDWINDMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > GridWindMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the GridWind mover (should always be rectgrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
	 	case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
		//case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
		//case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in GridWindMover::Read()."); return -1;
	}
	if (err = fGrid -> Read (bfpb)) goto done;

	if (err = ReadMacValue(bfpb, &bShowGrid)) goto done;
	if (err = ReadMacValue(bfpb, &bShowArrows)) goto done;
	if (err = ReadMacValue(bfpb, &fUserUnits)) goto done;
	//if (err = ReadMacValue(bfpb, &fWindScale)) goto done; // not using this
	//if (err = ReadMacValue(bfpb, &fArrowScale)) goto done; // not using this

	if (err = ReadMacValue(bfpb, &fNumRows)) goto done;	
	if (err = ReadMacValue(bfpb, &fNumCols)) goto done;	
	if (err = ReadMacValue(bfpb, fPathName, kMaxNameLen)) goto done;	
	ResolvePath(fPathName); // JLM 6/3/10
	if (err = ReadMacValue(bfpb, fFileName, kPtCurUserNameLen)) goto done;	

	if (err = ReadMacValue(bfpb, &amtTimeData)) goto done;	
	fTimeDataHdl = (PtCurTimeDataHdl)_NewHandleClear(sizeof(PtCurTimeData)*amtTimeData);
	if(!fTimeDataHdl)
		{TechError("GridWindMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i = 0 ; i < amtTimeData ; i++) {
		if (err = ReadMacValue(bfpb, &timeData.fileOffsetToStartOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.lengthOfData)) goto done;
		if (err = ReadMacValue(bfpb, &timeData.time)) goto done;
		INDEXH(fTimeDataHdl, i) = timeData;
	}
	
	if (err = ReadMacValue(bfpb, &numFiles)) goto done;	
	if (numFiles > 0)
	{
		fInputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
		if(!fInputFilesHdl)
			{TechError("GridWindMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
		for (i = 0 ; i < numFiles ; i++) {
			if (err = ReadMacValue(bfpb, fileInfo.pathName, kMaxNameLen)) goto done;
			ResolvePath(fileInfo.pathName); // JLM 6/3/10
			if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
			if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
			INDEXH(fInputFilesHdl,i) = fileInfo;
		}
		if (err = ReadMacValue(bfpb, &fOverLap)) return err;
		if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
	}
	
done:
	if(err)
	{
		TechError("GridWindMover::Read(char* path)", " ", 0); 
		if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
		if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	}
	return err;
}

OSErr GridWindMover::ReadHeaderLines(char *path, WorldRect *bounds)
{
	char s[256], classicPath[256];
	long line = 0;
	CHARH f = 0;
	OSErr err = 0;
	long /*numLines,*/numScanned;
	double dLon,dLat,oLon,oLat;
	double lowLon,lowLat,highLon,highLat;
	Boolean velAtCenter = 0;
	Boolean velAtCorners = 0;
	//long numLinesInText, headerLines = 8;
	
	if (!path) return -1;
	
	if (err = ReadFileContents(TERMINATED,0, 0, path, 0, 0, &f)) goto done;
	//numLinesInText = NumLinesInText(*f);
	////
	// read the header
	///////////////////////
	/////////////////////////////////////////////////
	// have units in the file somewhere?
	NthLineInTextOptimized(*f, line++, s, 256); // gridcur header
	if(fUserUnits == kUndefined)
	{	
		// we have to ask the user for units...
		Boolean userCancel=false;
		short selectedUnits = kKnots; // knots will be default
		err = AskUserForUnits(&selectedUnits,&userCancel);
		if(err || userCancel) { err = -1; goto done;}
		fUserUnits = selectedUnits;
	}
	
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"NUMROWS")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("NUMROWS"),"%ld",&fNumRows);
	if(numScanned != 1 || fNumRows <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 
	if(!strstr(s,"NUMCOLS")) { err = -2; goto done; }
	numScanned = sscanf(s+strlen("NUMCOLS"),"%ld",&fNumCols);
	if(numScanned != 1 || fNumCols <= 0) { err = -2; goto done; }
	//
	NthLineInTextOptimized(*f, line++, s, 256); 

	if(s[0]=='S') // check if lat/long given as corner point and increment, and read in
	{
		if(!strstr(s,"STARTLAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("STARTLAT"),lfFix("%lf"),&oLat);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"STARTLONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("STARTLONG"),lfFix("%lf"),&oLon);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"DLAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("DLAT"),lfFix("%lf"),&dLat);
		if(numScanned != 1 || dLat <= 0) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"DLONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("DLONG"),lfFix("%lf"),&dLon);
		if(numScanned != 1 || dLon <= 0) { err = -2; goto done; }
	
		velAtCorners=true;
		//
	}
	else if(s[0]=='L') // check if lat/long bounds given, and read in
	{
		if(!strstr(s,"LOLAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("LOLAT"),lfFix("%lf"),&lowLat);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"HILAT")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("HILAT"),lfFix("%lf"),&highLat);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"LOLONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("LOLONG"),lfFix("%lf"),&lowLon);
		if(numScanned != 1 ) { err = -2; goto done; }
		//
		NthLineInTextOptimized(*f, line++, s, 256); 
		if(!strstr(s,"HILONG")) { err = -2; goto done; }
		numScanned = sscanf(s+strlen("HILONG"),lfFix("%lf"),&highLon);
		if(numScanned != 1 ) { err = -2; goto done; }
		
		velAtCenter=true;
	}
	else {err = -2; goto done; }
	//
	//NthLineInTextOptimized(*f, line++, s, 256); // row col u v header
	//
	
	// check hemisphere stuff here , code goes here
	if(velAtCenter)
	{
		(*bounds).loLat = lowLat*1000000;
		(*bounds).hiLat = highLat*1000000;
		(*bounds).loLong = lowLon*1000000;
		(*bounds).hiLong = highLon*1000000;
	}
	else if(velAtCorners)
	{
		(*bounds).loLat = round((oLat - dLat/2.0)*1000000);
		(*bounds).hiLat = round((oLat + (fNumRows-1)*dLat + dLat/2.0)*1000000);
		(*bounds).loLong = round((oLon - dLon/2.0)*1000000);
		(*bounds).hiLong = round((oLon + (fNumCols-1)*dLon + dLon/2.0)*1000000);
	}
	//numLines = numLinesInText - headerLines;	// allows user to leave out land points within grid (or standing water)

	NthLineInTextOptimized(*f, (line)++, s, 256);
	RemoveLeadingAndTrailingWhiteSpace(s);
	while ((s[0]=='[' && s[1]=='U') || s[0]==0)
	{	// [USERDATA] lines, and blank lines, generalize to anything but [FILE] ?
		NthLineInTextOptimized(*f, (line)++, s, 256);
		RemoveLeadingAndTrailingWhiteSpace(s);
	}
	if(!strstr(s,"[FILE]")) 
	{	// single file
		err = ScanFileForTimes(path,&fTimeDataHdl,true);
		if (err) goto done;
	}
	else
	{	// multiple files
		long numLinesInText = NumLinesInText(*f);
		long numFiles = (numLinesInText - (line - 1))/3;	// 3 lines for each file - filename, starttime, endtime
		//strcpy(fPathName,s+strlen("[FILE]\t"));
		strcpy(fPathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace(fPathName);
		ResolvePathFromInputFile(path,fPathName); // JLM 6/8/10
		if(fPathName[0] && FileExists(0,0,fPathName))
		{
			err = ScanFileForTimes(fPathName,&fTimeDataHdl,true);
			if (err) goto done;
			// code goes here, maybe do something different if constant wind
			line--;
			err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to GridWind data File does not exist.%s%s",NEWLINESTRING,fPathName);
			printError(msg);
			err = true;
		}

		/*err = ScanFileForTimes(fPathName,&fTimeDataHdl,true);
		if (err) goto done;
		// code goes here, maybe do something different if constant wind
		line--;
		err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);*/
	}

done:
	if(f) { DisposeHandle((Handle)f); f = 0;}
	if(err)
	{
		if(err==memFullErr)
			 TechError("TRectGridVel::ReadGridWindFile()", "_NewHandleClear()", 0); 
		else
			printError("Unable to read GridWind file.");
	}
	return err;
}
	

/////////////////////////////////////////////////////////////////

OSErr GridWindMover::TextRead(char *path) 
{
	WorldRect bounds;
	OSErr err = 0;
	char s[256], fileName[64];

	TRectGridVel *rectGrid = nil;

	if (!path || !path[0]) return 0;
	
	strcpy(fPathName,path);
	
	strcpy(s,path);
	SplitPathFile (s, fileName);
	strcpy(fFileName, fileName);	// maybe use a name from the file
	// code goes here, we need to worry about really big files

	// do the readgridcur file stuff, store numrows, numcols, return the bounds
	err = this -> ReadHeaderLines(path,&bounds);
	if(err)
		goto done;
	
	/////////////////////////////////////////////////

	rectGrid = new TRectGridVel;
	if (!rectGrid)
	{		
		err = true;
		TechError("Error in GridWindMover::TextRead()","new TRectGridVel" ,err);
		goto done;
	}

	fGrid = (TGridVel*)rectGrid;

	rectGrid -> SetBounds(bounds); 

	// scan through the file looking for "[TIME ", then read and record the time, filePosition, and length of data
	// consider the possibility of multiple files
	/*NthLineInTextOptimized(*f, (line)++, s, 256); 
	if(!strstr(s,"[FILE]")) 
	{	// single file
		err = ScanFileForTimes(path,&fTimeDataHdl,true);
		if (err) goto done;
	}
	else
	{	// multiple files
		long numLinesInText = NumLinesInText(*f);
		long numFiles = (numLinesInText - (line - 1))/3;	// 3 lines for each file - filename, starttime, endtime
		strcpy(fPathName,s+strlen("[FILE]\t"));
		err = ScanFileForTimes(fPathName,&fTimeDataHdl,true);
		if (err) goto done;
		// code goes here, maybe do something different if constant wind
		line--;
		err = ReadInputFileNames(f,&line,numFiles,&fInputFilesHdl,path);
	}*/

	//err = ScanFileForTimes(path,&fTimeDataHdl);
	//if (err) goto done;
		

done:

	if(err)
	{
		printError("An error occurred in GridWindMover::TextRead"); 
		if(fGrid)
		{
			fGrid ->Dispose();
			delete fGrid;
			fGrid = 0;
		}
	}
	return err;

	// rest of file (i.e. velocity data) is read as needed
}
	 	 
OSErr GridWindMover::ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile)
{
	long i,numScanned;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], classicPath[256];
	
	PtCurFileInfoH inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("GridWindMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
	for (i=0;i<numFiles;i++)	// should count files as go along, and check that they exist ?
	{
		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); 	// check it is a [FILE] line
		RemoveLeadingAndTrailingWhiteSpace(s);
		strcpy((*inputFilesHdl)[i].pathName,s+strlen("[FILE] "));
		RemoveLeadingAndTrailingWhiteSpace((*inputFilesHdl)[i].pathName);
		ResolvePathFromInputFile(pathOfInputfile,(*inputFilesHdl)[i].pathName); // JLM 6/8/10 , need to have path here to use this function

		if((*inputFilesHdl)[i].pathName[0] && FileExists(0,0,(*inputFilesHdl)[i].pathName))
		{
			//
		}	
		else 
		{
			char msg[256];
			sprintf(msg,"PATH to GridWind data File does not exist.%s%s",NEWLINESTRING,(*inputFilesHdl)[i].pathName);
			printError(msg);
			err = true;
			goto done;
		}

		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); // check it is a [STARTTIME] line
		RemoveLeadingAndTrailingWhiteSpace(s);

		numScanned=sscanf(s+strlen("[STARTTIME]"), "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			{ err = -1; TechError("GridWindMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		// not allowing constant wind in separate file
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
		//if (time.day == time.month == time.year == time.hour == time.minute == -1)
		{
			//timeSeconds = CONSTANTCURRENT;
			timeSeconds = CONSTANTWIND;
		}
		else // time varying wind
		{
			CheckYear(&time.year);
	
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}
		(*inputFilesHdl)[i].startTime = timeSeconds;

		NthLineInTextNonOptimized(*fileBufH, (*line)++, s, 1024); // check it is an [ENDTIME] line
		RemoveLeadingAndTrailingWhiteSpace(s);

		/*strToMatch = "[ENDTIME]";
		len = strlen(strToMatch);
		//NthLineInTextOptimized (sectionOfFile, line = 0, s, 1024);
		if(!strncmp(s,strToMatch,len)) 
		{
			numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
			if (numScanned!= 5)
				{ err = -1; TechError("GridWindMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		}
		*/
		numScanned=sscanf(s+strlen("[ENDTIME]"), "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			{ err = -1; TechError("GridWindMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		//if (time.day == time.month == time.year == time.hour == time.minute == -1)
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
		{
			//timeSeconds = CONSTANTCURRENT;
			timeSeconds = CONSTANTWIND;
		}
		else // time varying wind
		{
			CheckYear(&time.year);
		
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}
		(*inputFilesHdl)[i].endTime = timeSeconds;
	}
	*inputFilesH = inputFilesHdl;

done:
	if (err)
	{
		if(inputFilesHdl) {DisposeHandle((Handle)inputFilesHdl); inputFilesHdl=0;}
	}
	return err;
}

OSErr GridWindMover::ScanFileForTimes(char *path, PtCurTimeDataHdl *timeDataH,Boolean setStartTime)
{
	// scan through the file looking for times "[TIME "  (close file if necessary...)
	
	OSErr err = 0;
	CHARH h = 0;
	char *sectionOfFile = 0;

	long fileLength,lengthRemainingToScan,offset;
	long lengthToRead,lengthOfPartToScan,numTimeBlocks=0;
	long i, numScanned;
	DateTimeRec time;
	Seconds timeSeconds;	
	
	// allocate an empty handle
	PtCurTimeDataHdl timeDataHdl;
	timeDataHdl = (PtCurTimeDataHdl)_NewHandle(0);
	if(!timeDataHdl) {TechError("GridWindMover::ScanFileForTimes()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	// think in terms of 100K blocks, allocate 101K, read 101K, scan 100K

	#define kGridCurFileBufferSize  100000 // code goes here, increase to 100K or more
	#define kGridCurFileBufferExtraCharSize  256

	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) goto done;

	offset = 0;
	lengthRemainingToScan = fileLength - 5;

	// loop until whole file is read 
	
	h = (CHARH)_NewHandle(2* kGridCurFileBufferSize+1);
	if(!h){TechError("GridWindMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}

	_HLock((Handle)h);
	sectionOfFile = *h;
	
	while (lengthRemainingToScan>0)
	{
		if(lengthRemainingToScan > 2* kGridCurFileBufferSize)
		{
			lengthToRead = kGridCurFileBufferSize + kGridCurFileBufferExtraCharSize; 
			lengthOfPartToScan = kGridCurFileBufferSize; 		
		}
		else
		{
			// deal with it in one piece
			// just read the rest of the file
			lengthToRead = fileLength - offset;
			lengthOfPartToScan = lengthToRead - 5; 
		}
		
		err = ReadSectionOfFile(0,0,path,offset,lengthToRead,sectionOfFile,0);
		if(err || !h) goto done;
		sectionOfFile[lengthToRead] = 0; // make it a C string
		
		lengthRemainingToScan -= lengthOfPartToScan;
		
			
		// scan 100K chars of the buffer for '['
		for(i = 0; i < lengthOfPartToScan; i++)
		{
			if(	sectionOfFile[i] == '[' 
				&& sectionOfFile[i+1] == 'T'
				&& sectionOfFile[i+2] == 'I'
				&& sectionOfFile[i+3] == 'M'
				&& sectionOfFile[i+4] == 'E')
			{
				// read and record the time and filePosition
				PtCurTimeData timeData;
				memset(&timeData,0,sizeof(timeData));
				timeData.fileOffsetToStartOfData = i + offset;
				
				if (numTimeBlocks > 0) 
				{
					(*timeDataHdl)[numTimeBlocks-1].lengthOfData = i+offset - (*timeDataHdl)[numTimeBlocks-1].fileOffsetToStartOfData;					
				}
				// some sort of a scan
				numScanned=sscanf(sectionOfFile+i+6, "%hd %hd %hd %hd %hd",
							  &time.day, &time.month, &time.year,
							  &time.hour, &time.minute) ;
				if (numScanned != 5)
					{ err = -1; TechError("GridWindMover::TextRead()", "sscanf() == 5", 0); goto done; }
				// check for constant wind
				if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
				//if (time.day == time.month == time.year == time.hour == time.minute == -1)
				{
					//timeSeconds = CONSTANTCURRENT;
					timeSeconds = CONSTANTWIND;
					setStartTime = false;
				}
				else // time varying wind
				{
					if (time.year < 1900)					// two digit date, so fix it
					{
						if (time.year >= 40 && time.year <= 99)	
							time.year += 1900;
						else
							time.year += 2000;					// correct for year 2000 (00 to 40)
					}
	
					time.second = 0;
					DateToSeconds (&time, &timeSeconds);
				}

				timeData.time = timeSeconds;
				
				// if we don't know the number of times ahead of time
				_SetHandleSize((Handle) timeDataHdl, (numTimeBlocks+1)*sizeof(timeData));
				if (_MemError()) { TechError("GridWindMover::TextRead()", "_SetHandleSize()", 0); goto done; }
				if (numTimeBlocks==0 && setStartTime) 
				{	// set the default times to match the file
					model->SetModelTime(timeSeconds);
					model->SetStartTime(timeSeconds);
					model->NewDirtNotification(DIRTY_RUNBAR); // must reset the runbar
				}
				(*timeDataHdl)[numTimeBlocks++] = timeData;				
			}
		}
		offset += lengthOfPartToScan;
	}
	if (numTimeBlocks > 0)  // last block goes to end of file
	{
		(*timeDataHdl)[numTimeBlocks-1].lengthOfData = fileLength - (*timeDataHdl)[numTimeBlocks-1].fileOffsetToStartOfData;				
	}
	*timeDataH = timeDataHdl;


done:

	if(h) {
		_HUnlock((Handle)h); 
		DisposeHandle((Handle)h); 
		h = 0;
	}
	if (err)
	{
			if(timeDataHdl) {DisposeHandle((Handle)timeDataHdl); timeDataHdl=0;}
	}
	return err;
}
/////////////////////////////////////////////////

OSErr GridWindMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	char s[256], path[256]; 
	long i,line = 0;
	long offset,lengthToRead;
	CHARH h = 0;
	char *sectionOfFile = 0;
	char *strToMatch = 0;
	long len,numScanned;
	VelocityFH velH = 0;
	long totalNumberOfVels = fNumRows * fNumCols;
	long numLinesInBlock;

	OSErr err = 0;
	DateTimeRec time;
	Seconds timeSeconds;
	errmsg[0]=0;

	strcpy(path,fPathName);
	if (!path || !path[0]) return -1;

	lengthToRead = (*fTimeDataHdl)[index].lengthOfData;
	offset = (*fTimeDataHdl)[index].fileOffsetToStartOfData;

	h = (CHARH)_NewHandle(lengthToRead+1);
	if(!h){TechError("GridWindMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}

	_HLock((Handle)h);
	sectionOfFile = *h;			
	
	err = ReadSectionOfFile(0,0,path,offset,lengthToRead,sectionOfFile,0);
	if(err || !h) 
	{
		char firstPartOfLine[128];
		sprintf(errmsg,"Unable to open data file:%s",NEWLINESTRING);
		strncpy(firstPartOfLine,path,120);
		strcpy(firstPartOfLine+120,"...");
		strcat(errmsg,firstPartOfLine);
		goto done;
	}
	sectionOfFile[lengthToRead] = 0; // make it a C string
	numLinesInBlock = NumLinesInText(sectionOfFile);
	
	// some other way to calculate
	velH = (VelocityFH)_NewHandleClear(sizeof(**velH)*totalNumberOfVels);
	if(!velH){TechError("GridWindMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}

	strToMatch = "[TIME]";
	len = strlen(strToMatch);
	NthLineInTextOptimized (sectionOfFile, line = 0, s, 256);
	if(!strncmp(s,strToMatch,len)) 
	{
		numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
					  &time.day, &time.month, &time.year,
					  &time.hour, &time.minute) ;
		if (numScanned!= 5)
			{ err = -1; TechError("GridWindMover::ReadTimeData()", "sscanf() == 5", 0); goto done; }
		// check for constant wind
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
		//if (time.year == time.month == time.day == time.hour == time.minute == -1) 
		{
			//timeSeconds = CONSTANTCURRENT;
			timeSeconds = CONSTANTWIND;
		}
		else // time varying wind
		{
			if (time.year < 1900)					// two digit date, so fix it
			{
				if (time.year >= 40 && time.year <= 99)	
					time.year += 1900;
				else
					time.year += 2000;					// correct for year 2000 (00 to 40)
			}
	
			time.second = 0;
			DateToSeconds (&time, &timeSeconds);
		}

		// check time is correct
		if (timeSeconds!=(*fTimeDataHdl)[index].time)
			{ err = -1;  strcpy(errmsg,"Can't read data - times in the file have changed."); goto done; }
		line++;
	}

	// allow to omit areas of the grid with zero velocity, use length of data info
	//for(i=0;i<totalNumberOfVels;i++) // interior points
	for(i=0;i<numLinesInBlock-1;i++) // lines of data
	{
		VelocityRec vel;
		long rowNum,colNum;
		long index;

		NthLineInTextOptimized(sectionOfFile, line++, s, 256); 	// in theory should run out of lines eventually
		RemoveLeadingAndTrailingWhiteSpace(s);
		if(s[0] == 0) continue; // it's a blank line, allow this and skip the line
		numScanned = sscanf(s,lfFix("%ld %ld %lf %lf"),&rowNum,&colNum,&vel.u,&vel.v);
		if(numScanned != 4 
			|| rowNum <= 0 || rowNum > fNumRows
			|| colNum <= 0 || colNum > fNumCols
			)
		{ 
			err = -1;  
			char firstPartOfLine[128];
			sprintf(errmsg,"Unable to read velocity data from line %ld:%s",line,NEWLINESTRING);
			strncpy(firstPartOfLine,s,120);
			strcpy(firstPartOfLine+120,"...");
			strcat(errmsg,firstPartOfLine);
			goto done; 
		}
		index = (rowNum -1) * fNumCols + colNum-1;
		(*velH)[index].u = vel.u; // units ??? assumed m/s
		(*velH)[index].v = vel.v; 
	}
	*velocityH = velH;
	
done:

	if(h) {
		_HUnlock((Handle)h); 
		DisposeHandle((Handle)h); 
		h = 0;
	}


	if(err)
	{
		if(!errmsg[0])
			strcpy(errmsg,"An error occurred in GridWindMover::ReadTimeData");
		//printError(errmsg); // This alert causes a freeze up...
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	return err;

}

Boolean GridWindMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant wind, we can say "no"
	if(this->GetNumTimesInFile()==1) depends = false;
	return depends;
}

void GridWindMover::Draw(Rect r, WorldRect view) 
{	// Use this for regular grid
	short row, col, pixX, pixY;
	long dLong, dLat, index, timeDataInterval;
	float inchesX, inchesY;
	double timeAlpha;
	Seconds startTime, endTime, time = model->GetModelTime();
	Point p, p2;
	WorldPoint wp;
	WorldRect boundsRect, bounds;
	VelocityRec velocity;
	Rect c, newGridRect = {0, 0, fNumRows - 1, fNumCols - 1}; // fNumRows, fNumCols members of NetCDFWindMover
	Boolean offQuickDrawPlane = false, loaded;
	char errmsg[256];
	OSErr err = 0;
	
	TRectGridVel* rectGrid = (TRectGridVel*)fGrid;	

	if (!bShowArrows && !bShowGrid) return;
	
	bounds = rectGrid->GetBounds();
	
	// need to get the bounds from the grid
	dLong = (WRectWidth(bounds) / fNumCols) / 2;
	dLat = (WRectHeight(bounds) / fNumRows) / 2;
	//RGBForeColor(&colors[PURPLE]);
	RGBForeColor(&fColor);

	boundsRect = bounds;
	InsetWRect (&boundsRect, dLong, dLat);

	if (bShowArrows)
	{
		err = this -> SetInterval(errmsg);
		if(err && !bShowGrid) return;	// want to show grid even if there's no wind data
		
		loaded = this -> CheckInterval(timeDataInterval);
		if(!loaded && !bShowGrid) return;
	
		if(GetNumTimesInFile()>1 && loaded && !err)
		{
			// Calculate the time weight factor
			if (GetNumFiles()>1 && fOverLap)
				startTime = fOverLapStartTime;
			else
				startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
			//startTime = (*fTimeDataHdl)[fStartData.timeIndex].time;
			endTime = (*fTimeDataHdl)[fEndData.timeIndex].time;
			timeAlpha = (endTime - time)/(double)(endTime - startTime);
		}
	}	
	
	for (row = 0 ; row < fNumRows ; row++)
		for (col = 0 ; col < fNumCols ; col++) {

			SetPt(&p, col, row);
			wp = ScreenToWorldPoint(p, newGridRect, boundsRect);
			velocity.u = velocity.v = 0.;
			if (loaded && !err)
			{
				index = GetVelocityIndex(wp);	
	
				if (bShowArrows && index >= 0)
				{
					// Check for constant wind pattern 
					if(GetNumTimesInFile()==1)
					{
						velocity.u = INDEXH(fStartData.dataHdl,index).u;
						velocity.v = INDEXH(fStartData.dataHdl,index).v;
					}
					else // time varying wind
					{
						velocity.u = timeAlpha*INDEXH(fStartData.dataHdl,index).u + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).u;
						velocity.v = timeAlpha*INDEXH(fStartData.dataHdl,index).v + (1-timeAlpha)*INDEXH(fEndData.dataHdl,index).v;
					}
				}
			}
	
			p = GetQuickDrawPt(wp.pLong, wp.pLat, &r, &offQuickDrawPlane);
			MySetRect(&c, p.h - 1, p.v - 1, p.h + 1, p.v + 1);
			
			if (bShowGrid && bShowArrows && (velocity.u != 0 || velocity.v != 0)) 
				PaintRect(&c);	// should check fill_value
			if (bShowGrid && !bShowArrows) 
				PaintRect(&c);	// should check fill_value
				
			if (bShowArrows && (velocity.u != 0 || velocity.v != 0))
			{
				inchesX = (velocity.u * fWindScale) / fArrowScale;
				inchesY = (velocity.v * fWindScale) / fArrowScale;
				pixX = inchesX * PixelsPerInchCurrent();
				pixY = inchesY * PixelsPerInchCurrent();
				p2.h = p.h + pixX;
				p2.v = p.v - pixY;
				MyMoveTo(p.h, p.v);
				MyLineTo(p2.h, p2.v);
				MyDrawArrow(p.h,p.v,p2.h,p2.v);
			}
		}
		
	RGBForeColor(&colors[BLACK]);
}
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////
/*static GridWindMover *sharedGWMover;

short GridWindClick(DialogPtr dialog, short itemNum, long lParam, VOIDPTR data)
{	
	switch (itemNum) {
		case M18OK:
			
			sharedGWMover -> bActive = GetButton(dialog, M18ACTIVE);
			sharedGWMover -> fAngleScale = EditText2Float(dialog,M18ANGLESCALE);
			sharedGWMover -> fSpeedScale = EditText2Float(dialog,M18SPEEDSCALE);
			sharedGWMover -> fUncertainStartTime = (long) round(EditText2Float(dialog,M18UNCERTAINSTARTTIME)*3600);
						
			sharedGWMover -> fDuration = EditText2Float(dialog, M18DURATION) * 3600;
			
			return M18OK;
		
		case M18CANCEL: return M18CANCEL;
		
		case M18ACTIVE:
			ToggleButton(dialog, itemNum);
			break;
		
		case M18DURATION:
		case M18UNCERTAINSTARTTIME:
		case M18ANGLESCALE:
		case M18SPEEDSCALE:
			CheckNumberTextItem(dialog, itemNum, TRUE);
			break;
		
	}
	
	return noErr;
}

OSErr GridWindInit(DialogPtr dialog, VOIDPTR data)
{
	SetDialogItemHandle(dialog, M18HILITEDEFAULT, (Handle)FrameDefault);
	
	SetButton(dialog, M18ACTIVE, sharedGWMover->bActive);
	
	Float2EditText(dialog, M18SPEEDSCALE, sharedGWMover->fSpeedScale, 4);
	Float2EditText(dialog, M18ANGLESCALE, sharedGWMover->fAngleScale, 4);

	Float2EditText(dialog, M18DURATION, sharedGWMover->fDuration / 3600.0, 2);
	Float2EditText(dialog, M18UNCERTAINSTARTTIME, sharedGWMover->fUncertainStartTime / 3600.0, 2);
	
	MySelectDialogItemText(dialog, M18SPEEDSCALE, 0, 255);
	
	ShowHideDialogItem(dialog,M18ANGLEUNITSPOPUP,false);	// for now don't allow the units option here

	SetDialogItemHandle(dialog,M18SETTINGSFRAME,(Handle)FrameEmbossed);
	SetDialogItemHandle(dialog,M18UNCERTAINFRAME,(Handle)FrameEmbossed);

	return 0;
}

// maybe should revamp windmover and extend for netcdf...
OSErr GridWindSettingsDialog(GridWindMover *mover, TMap *owner,Boolean bAddMover,DialogPtr parentWindow)
{ // Note: returns USERCANCEL when user cancels
	OSErr err = noErr;
	short item;
	
	if(!owner && bAddMover) {printError("Programmer error"); return -1;}
		
	sharedGWMover = mover;			// existing mover is being edited
	
	if(parentWindow == 0) parentWindow = mapWindow; // JLM 6/2/99
	item = MyModalDialog(M18, parentWindow, 0, GridWindInit, GridWindClick);
	if (item == M18OK)
	{
		if (bAddMover)
		{
			err = owner -> AddMover (sharedGWMover, 0);
		}
		model->NewDirtNotification();
	}
	if(item == M18CANCEL)
	{
		err = USERCANCEL;
	}
	
	return err;
}*/

