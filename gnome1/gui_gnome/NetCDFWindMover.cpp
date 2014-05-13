// This is a cross between TWindMover and NetCDFMover, possibly reorganize to build off one or the other
// The uncertainty is from the wind, the reading, storing, accessing, displaying data is from NetCDFMover


#include "Cross.h"
#include "NetCDFMover.h"
#include "netcdf.h"
#include "TWindMover.h"
#include "GridCurMover.h"
#include "GridWndMover.h"
#include "Outils.h"
#include "DagTreeIO.h"
#include "PtCurMover.h"

#ifdef MAC
#ifdef MPW
//#include <QDOffscreen.h>
#pragma SEGMENT NETCDFWINDMOVER
#endif
#endif

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
	 	 

long NetCDFWindMover::GetNumFiles()
{
	long numFiles = 0;
	
	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

OSErr NetCDFWindMover::CheckAndScanFile(char *errmsg, const Seconds& model_time)
{
	Seconds time = model_time, startTime, endTime, lastEndTime, testTime, firstStartTime; // AH 07/17/2012
	
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
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);	// AH 07/17/2012
			
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
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeHdl,false);	// AH 07/17/2012
				
				DisposeLoadedData(&fEndData);
				strcpy(fPathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeHdl) {DisposeHandle((Handle)fTimeHdl); fTimeHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeHdl,false);	// AH 07/17/2012
			
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
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeHdl,false);	// AH 07/17/2012
			
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
		err = ScanFileForTimes((*fInputFilesHdl)[numFiles-1].pathName,&fTimeHdl,false); // AH 07/17/2012
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
}

Boolean NetCDFWindMover::CheckInterval(long &timeDataInterval, const Seconds& model_time)
{
	Seconds time = model_time, startTime, endTime;	
	
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


OSErr NetCDFWindMover::SetInterval(char *errmsg, const Seconds& model_time)
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
	
	if(timeDataInterval == 0 && fAllowExtrapolationOfWinds)
	{
		indexOfStart = 0;
		indexOfEnd = -1;
	}

	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile /*|| (timeDataInterval==1 && fAllowExtrapolationOfWinds)*/)
	{	// before the first step in the file
		
		if (GetNumFiles()>1)
		{
			if ((err = CheckAndScanFile(errmsg, model_time)) || fOverLap) goto done;	// AH 07/17/2012
				
			intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	// AH 07/17/2012
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
			status = nc_open(path, NC_NOWRITE, &ncid);
			if (status != NC_NOERR) 
			{
#if TARGET_API_MAC_CARBON
				err = ConvertTraditionalPathToUnixPath((const char *) path, outPath, kMaxNameLen) ;
				status = nc_open(outPath, NC_NOWRITE, &ncid);
#endif
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			
			status = nc_inq_dimid(ncid, "time", &recid); 
			if (status != NC_NOERR) 
			{
				status = nc_inq_unlimdim(ncid, &recid);	// maybe time is unlimited dimension
				if (status != NC_NOERR) {err = -2; goto done;}
			}
			
			status = nc_inq_varid(ncid, "time", &timeid); 
			if (status != NC_NOERR) {err = -2; goto done;} 
			
			/////////////////////////////////////////////////
			status = nc_inq_attlen(ncid, timeid, "units", &t_len);
			if (status != NC_NOERR) 
			{
				err = -2; goto done;
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
				else if (numScanned==7) // has two extra time entries ??	
					time.second = 0;
				else if (numScanned<8)	
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
	ListItem item = { dynamic_cast<NetCDFWindMover *>(this), n, indent, 0 };
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
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainStartTime)/3600.);
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
				NetCDFWindSettingsDialog(dynamic_cast<NetCDFWindMover *>(this), this -> moverMap,false,mapWindow);
				//WindSettingsDialog(this, this -> moverMap,false,mapWindow,false);
				break;
			default:	// why not call this for everything?
				NetCDFWindSettingsDialog(dynamic_cast<NetCDFWindMover *>(this), this -> moverMap,false,mapWindow);
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
		return moverMap -> DropMover(dynamic_cast<NetCDFWindMover *>(this));
	
	return 0;
}

OSErr NetCDFWindMover::CheckAndPassOnMessage(TModelMessage *message)
{	
	
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
		{bPathIsValid = false;}
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
		{bPathIsValid = false;}
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
				// code goes here, check the path (or get an error returned...) and ask user to find it, but not every time...
				if (!fileInfo.pathName[0] || !FileExists(0,0,fileInfo.pathName)) 
					bPathIsValid = false;	// if any one can not be found try to re-load the file list
				else bPathIsValid = true;
				if (err = ReadMacValue(bfpb, &fileInfo.startTime)) goto done;
				if (err = ReadMacValue(bfpb, &fileInfo.endTime)) goto done;
				INDEXH(fInputFilesHdl,i) = fileInfo;
			}
			if (err = ReadMacValue(bfpb, &fOverLap)) return err;
			if (err = ReadMacValue(bfpb, &fOverLapStartTime)) return err;
		//}
		// otherwise ask the user, trusting that user actually chooses the same data file (should insist name is the same?)
		if(!bPathIsValid)
		{
			if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
			Point where;
			OSType typeList[] = { 'NULL', 'NULL', 'NULL', 'NULL' };
			MySFReply reply;
			where = CenteredDialogUpLeft(M38c);
			char newPath[kMaxNameLen], s[kMaxNameLen];
			//sprintf(msg,"This save file references a wind file list which cannot be found.  Please find the file \"%s\".",fPathName);printNote(msg);
			sprintf(msg,"This save file references a wind file list which cannot be found.  Please find the file.");printNote(msg);
#if TARGET_API_MAC_CARBON
			mysfpgetfile(&where, "", -1, typeList,
						 (MyDlgHookUPP)0, &reply, M38c, MakeModalFilterUPP(STDFilter));
			if (!reply.good) return USERCANCEL;
			strcpy(newPath, reply.fullPath);
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
			}
#endif
			err = ReadInputFileNames(newPath);
		}
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
		err = this -> SetInterval(errmsg, model->GetModelTime()); // AH 07/17/2012
		
		if(err && !bShowGrid) return;	// want to show grid even if there's no wind data
		
		loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
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
				index = dynamic_cast<NetCDFWindMover *>(this)->GetVelocityIndex(wp);	
				
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


OSErr NetCDFWindMover::TextRead(char *path) 
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
	char errmsg[256] = "",className[256] = "";
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
		
		numScanned=sscanf(timeUnits, "%s %s %hd %hd %hd %hd %hd %hd",
						  unitStr, junk, &time.year, &time.month, &time.day,
						  &time.hour, &time.minute, &time.second) ;
		if (numScanned==5)	
		{time.hour = 0; time.minute = 0; time.second = 0; }
		else if (numScanned==7) // has two extra time entries ??	
			time.second = 0;
		else if (numScanned<8)	
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
	GetClassName(className);
	if (!strcmp("NetCDF Wind",className))
		SetClassName(fFileName); //first check that name is now the default and not set by command file ("NetCDF Wind")
	status = nc_inq_dimid(ncid, "lat", &latid); 
	if (status != NC_NOERR) 
	{	// add new check if error for LON, LAT with extensions based on subset from LAS 1/29/09
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
	// probabaly don't want to set based on wind ?
	if (model->GetStartTime() != startTime || model->GetModelTime()!=model->GetStartTime())
	{
		if (true)	// maybe use NOAA.ver here?
		{
			short buttonSelected;
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
