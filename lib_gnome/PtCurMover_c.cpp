/*
 *  PtCurMover_c.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 1/23/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "PtCurMover_c.h"
#include "MemUtils.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

PtCurMover_c::PtCurMover_c (TMap *owner, char *name) : CurrentMover_c(owner, name), Mover_c(owner, name)
{
	memset(&fVar,0,sizeof(fVar));
	fVar.arrowScale = 5;
	fVar.arrowDepth = 0;
	fVar.alongCurUncertainty = .5;
	fVar.crossCurUncertainty = .25;
	//fVar.uncertMinimumInMPS = .05;
	fVar.uncertMinimumInMPS = 0.0;
	fVar.curScale = 1.0;
	fVar.startTimeInHrs = 0.0;
	fVar.durationInHrs = 24.0;
	fVar.numLandPts = 0; // default that boundary velocities are given
	fVar.maxNumDepths = 1; // 2D default
	fVar.gridType = TWO_D; // 2D default
	fVar.bLayerThickness = 0.; // FREESLIP default
	//
	// Override TCurrentMover defaults
	fDownCurUncertainty = -fVar.alongCurUncertainty; 
	fUpCurUncertainty = fVar.alongCurUncertainty; 	
	fRightCurUncertainty = fVar.crossCurUncertainty;  
	fLeftCurUncertainty = -fVar.crossCurUncertainty; 
	fDuration=fVar.durationInHrs*3600.; //24 hrs as seconds 
	fUncertainStartTime = (long) (fVar.startTimeInHrs*3600.);
	//
	fGrid = 0;
	fTimeDataHdl = 0;
	fIsOptimizedForStep = false;
	fOverLap = false;		// for multiple files case
	fOverLapStartTime = 0;
	
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
}

OSErr PtCurMover_c::AddUncertainty(long setIndex, long leIndex,VelocityRec *velocity,double timeStep,Boolean useEddyUncertainty)
{
	LEUncertainRec unrec;
	double u,v,lengthS,alpha,beta,v0;
	OSErr err = 0;
	
	//err = this -> UpdateUncertainty();
	//if(err) return err;
	
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
		TechError("PtCurMover::AddUncertainty()", "fUncertaintyListH==nil", 0);
		err = -1;
		velocity->u=velocity->v=0;
	}
	return err;
}


OSErr PtCurMover_c::PrepareForModelRun()
{
	if (moverMap->IAm(TYPE_PTCURMAP))
	{
		(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
		(dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
		if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)	// I think this will always be 3D, but maybe old SAV files...
			((TTriGridVel3D*)fGrid)->ClearOutputHandles();
	}
	return CurrentMover_c::PrepareForModelRun();
}

OSErr PtCurMover_c::PrepareForModelStep(const Seconds& model_time,const Seconds& time_step, bool uncertain, int numLESets, int* LESetsSizesList)
{
	long timeDataInterval;
	//Boolean intervalLoaded;
	OSErr err=0;
	char errmsg[256];
	
	errmsg[0]=0;
	
	//this is done first thing in SetInterval
	//check to see that the time interval is loaded
	/*intervalLoaded = this -> CheckInterval(timeDataInterval);
	 
	 if(timeDataInterval<=0||timeDataInterval>=this->GetNumTimesInFile())
	 {
	 err = -1;
	 strcpy(errmsg,"Time outside of interval being modeled");
	 goto done;
	 }
	 
	 if(!intervalLoaded)*/
	
	/*if (model_time == start_time)	// first step, save depth range here?
	 {
	 if (moverMap->IAm(TYPE_PTCURMAP))
	 {
	 (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth1;	
	 (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2AtStartOfRun = (dynamic_cast<PtCurMap *>(moverMap))->fContourDepth2;	
	 if (fGrid->GetClassID()==TYPE_TRIGRIDVEL3D)	// I think this will always be 3D, but maybe old SAV files...
	 ((TTriGridVel3D*)fGrid)->ClearOutputHandles();
	 }
	 }*/
	if (!bActive) return 0; 
	err = this -> SetInterval(errmsg, model_time); // AH 07/17/2012
	
	if(err) goto done;
	
	if (bIsFirstStep)
		fModelStartTime = model_time;
	
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
			strcpy(errmsg,"An error occurred in PtCurMover::PrepareForModelStep");
		printError(errmsg); 
	}
	return err;
}

void PtCurMover_c::ModelStepIsDone()
{
	fIsOptimizedForStep = false;
	bIsFirstStep = false;
}

void PtCurMover_c::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void PtCurMover_c::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}


OSErr PtCurMover_c::CheckAndScanFile(char *errmsg, const Seconds& model_time)
{
	Seconds time = model_time, startTime, endTime, lastEndTime, testTime; // AH 07/17/2012
	
	long i,numFiles = GetNumFiles();
	OSErr err = 0;
	
	errmsg[0]=0;
	if (fEndData.timeIndex!=UNASSIGNEDINDEX)
		testTime = (*fTimeDataHdl)[fEndData.timeIndex].time;	// currently loaded end time
	
	for (i=0;i<numFiles;i++)
	{
		startTime = (*fInputFilesHdl)[i].startTime;
		endTime = (*fInputFilesHdl)[i].endTime;
		if (startTime<=time&&time<=endTime && !(startTime==endTime))
		{
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
			
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
				if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
				
				DisposeLoadedData(&fEndData);
				strcpy(fVar.pathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
			
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
				if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[i-1].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
				
				DisposeLoadedData(&fEndData);
				strcpy(fVar.pathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
			
			strcpy(fVar.pathName,(*fInputFilesHdl)[i].pathName);
			err = this -> ReadTimeData(0,&fEndData.dataHdl,errmsg);
			if(err) return err;
			fEndData.timeIndex = 0;
			fOverLap = true;
			return noErr;
		}
		lastEndTime = endTime;
	}
	strcpy(errmsg,"Time outside of interval being modeled");
	return -1;	
	//return err;
}

Boolean PtCurMover_c::CheckInterval(long &timeDataInterval, const Seconds& model_time)
{
	Seconds time = model_time; // AH 07/17/2012
	
	long i,numTimes;
	
	
	numTimes = this -> GetNumTimesInFile(); 
	
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


OSErr PtCurMover_c::SetInterval(char *errmsg, const Seconds& model_time)
{
	long timeDataInterval=0;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	// AH 07/17/2012
	
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
	
	if(timeDataInterval == 0 || timeDataInterval == numTimesInFile)
	{	// before the first step in the file
		if (GetNumFiles()>1)
		{
			if ((err = CheckAndScanFile(errmsg, model_time)) || fOverLap) goto done;	// AH 07/17/2012
			
			intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	// AH 07/17/2012
			
			indexOfStart = timeDataInterval-1;
			indexOfEnd = timeDataInterval;
			numTimesInFile = this -> GetNumTimesInFile();
		}
		else
		{
			err = -1;
			strcpy(errmsg,"Time outside of interval being modeled");
			goto done;
		}
	}
	// load the two intervals
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
		if(!errmsg[0])strcpy(errmsg,"Error in PtCurMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	return err;
	
}


long PtCurMover_c::GetNumTimesInFile()
{
	long numTimes = 0;
	
	if (fTimeDataHdl) numTimes = _GetHandleSize((Handle)fTimeDataHdl)/sizeof(**fTimeDataHdl);
	return numTimes;     
}

long PtCurMover_c::GetNumFiles()
{
	long numFiles = 0;
	
	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

long PtCurMover_c::GetNumDepths(void)
{
	long numDepths = 0;
	if (fDepthsH) numDepths = _GetHandleSize((Handle)fDepthsH)/sizeof(**fDepthsH);
	
	return numDepths;
}

LongPointHdl PtCurMover_c::GetPointsHdl()
{
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid); // don't think need 3D here
	return triGrid -> GetPointsHdl();
}

TopologyHdl PtCurMover_c::GetTopologyHdl()
{
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid); // don't think need 3D here
	return triGrid -> GetTopologyHdl();
}

long PtCurMover_c::WhatTriAmIIn(WorldPoint wp)
{
	LongPoint lp;
	TTriGridVel* triGrid = dynamic_cast<TTriGridVel*>(fGrid);	// don't think need 3D here
	TDagTree *dagTree = triGrid->GetDagTree();
	lp.h = wp.pLong;
	lp.v = wp.pLat;
	return dagTree -> WhatTriAmIIn(lp);
}



VelocityRec PtCurMover_c::GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty)
{
	VelocityRec v = {0,0};
	printError("PtCurMover::GetScaledPatValue is unimplemented");
	return v;
}


VelocityRec PtCurMover_c::GetPatValue(WorldPoint p)
{
	VelocityRec v = {0,0};
	printError("PtCurMover::GetPatValue is unimplemented");
	return v;
}


void PtCurMover_c::GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2)
{
	long indexToDepthData = (*fDepthDataInfo)[ptIndex].indexToDepthData;
	long numDepths = (*fDepthDataInfo)[ptIndex].numDepths;
	float totalDepth = (*fDepthDataInfo)[ptIndex].totalDepth;
	
	
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
			//break;
		case SIGMA: // 
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

OSErr PtCurMover_c::ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataH,Boolean setStartTime)
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
	if(!timeDataHdl) {TechError("PtCurMover::ScanFileForTimes()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	// think in terms of 100K blocks, allocate 101K, read 101K, scan 100K
	
#define kPtCurFileBufferSize  100000 // code goes here, increase to 100K or more
#define kPtCurFileBufferExtraCharSize  256
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) goto done;
	
	offset = 0;
	lengthRemainingToScan = fileLength - 5;
	
	// loop until whole file is read 
	
	h = (CHARH)_NewHandle(2* kPtCurFileBufferSize+1);
	if(!h){TechError("PtCurMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	_HLock((Handle)h);
	sectionOfFile = *h;
	
	while (lengthRemainingToScan>0)
	{
		if(lengthRemainingToScan > 2* kPtCurFileBufferSize)
		{
			lengthToRead = kPtCurFileBufferSize + kPtCurFileBufferExtraCharSize; 
			lengthOfPartToScan = kPtCurFileBufferSize; 		
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
				{ err = -1; TechError("PtCurMover::TextRead()", "sscanf() == 5", 0); goto done; }
				// check for constant current
				//if (time.day == time.month == time.year == time.hour == time.minute == -1)
				if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
				{
					timeSeconds = CONSTANTCURRENT;
					setStartTime = false;
				}
				else // time varying current
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
				if (_MemError()) { TechError("PtCurMover::TextRead()", "_SetHandleSize()", 0); goto done; }
				if (numTimeBlocks==0 && setStartTime) 
				{	// set the default times to match the file (only if this is the first time...)
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

OSErr PtCurMover_c::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
{
	char s[256], path[256]; 
	long i,j,line = 0;
	long offset,lengthToRead;
	CHARH h = 0;
	char *sectionOfFile = 0;
	char *strToMatch = 0;
	long len,numScanned;
	VelocityFH velH = 0;
	long totalNumberOfVels = 0;
	long numDepths = 1;
	
	LongPointHdl ptsHdl = 0;
	TTriGridVel* triGrid = (TTriGridVel*)fGrid; // don't think need 3D here
	
	OSErr err = 0;
	DateTimeRec time;
	Seconds timeSeconds;
	long numPoints; 
	errmsg[0]=0;
	
	strcpy(path,fVar.pathName);
	if (!path || !path[0]) return -1;
	
	lengthToRead = (*fTimeDataHdl)[index].lengthOfData;
	offset = (*fTimeDataHdl)[index].fileOffsetToStartOfData;
	
	ptsHdl = triGrid -> GetPointsHdl();
	if(ptsHdl)
		numPoints = _GetHandleSize((Handle)ptsHdl)/sizeof(**ptsHdl);
	else 
	{err=-1; goto done;} // no data
	
	
	h = (CHARH)_NewHandle(lengthToRead+1);
	if(!h){TechError("PtCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
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
	
	totalNumberOfVels = (*fDepthDataInfo)[numPoints-1].indexToDepthData+(*fDepthDataInfo)[numPoints-1].numDepths;
	if(totalNumberOfVels<numPoints) {err=-1; goto done;} // must have at least full set of 2D velocity data
	velH = (VelocityFH)_NewHandle(sizeof(**velH)*totalNumberOfVels);
	if(!velH){TechError("PtCurMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	strToMatch = "[TIME]";
	len = strlen(strToMatch);
	NthLineInTextOptimized (sectionOfFile, line = 0, s, 256);
	if(!strncmp(s,strToMatch,len)) 
	{
		numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("PtCurMover::ReadTimeData()", "sscanf() == 5", 0); goto done; }
		// check for constant current
		if (time.day == -1 && time.month == -1 && time.year == -1 && time.hour == -1 && time.minute == -1)
			//if (time.year == time.month == time.day == time.hour == time.minute == -1) 
		{
			timeSeconds = CONSTANTCURRENT;
		}
		else // time varying current
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
	
	
	for(i=0;i<fVar.numLandPts;i++)	// zero out boundary velocity
	{
		numDepths = (*fDepthDataInfo)[i].numDepths;
		for(j=0;j<numDepths;j++) 
		{
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u = 0.0;
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v = 0.0;
		}
	}
	
	for(i=fVar.numLandPts;i<numPoints;i++) // interior points
	{
		VelocityRec vel;
		char *startScan;
		long scanLength,stringIndex=0;
		numDepths = (*fDepthDataInfo)[i].numDepths;
		
		char *s1 = new char[numDepths*64];
		if(!s1) {TechError("PtCurMover::ReadTimeData()", "new[]", 0); err = memFullErr; goto done;}
		
		NthLineInTextOptimized (sectionOfFile, line, s1, numDepths*64);
		startScan = &s1[stringIndex];
		
		for(j=0;j<numDepths;j++) 
		{
			err = ScanVelocity(startScan,&vel,&scanLength); 
			// ScanVelocity is faster than scanf, but doesn't handle scientific notation. Try a scanf on error.
			if (err)
			{
				if(sscanf(&s1[stringIndex],lfFix("%lf%lf"),&vel.u,&vel.v) < 2)
				{
					char firstPartOfLine[128];
					sprintf(errmsg,"Unable to read velocity data from line %ld:%s",line,NEWLINESTRING);
					strncpy(firstPartOfLine,s1,120);
					strcpy(firstPartOfLine+120,"...");
					strcat(errmsg,firstPartOfLine);
					delete[] s1; s1=0;
					goto done;
				}
				err = 0;
			}
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].u = vel.u; 
			(*velH)[(*fDepthDataInfo)[i].indexToDepthData+j].v = vel.v; 
			stringIndex += scanLength;
			startScan = &s1[stringIndex];
		}
		line++;
		delete[] s1; s1=0;
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
			strcpy(errmsg,"An error occurred in PtCurMover::ReadTimeData");
		//printError(errmsg); // This alert causes a freeze up...
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	return err;
	
}

/**************************************************************************************************/
/*OSErr ScanVelocity (char *startChar, VelocityRec *VelocityPtr, long *scanLength)
{	// expects a number of the form 
	// <number><comma><number>
	//e.g.  "-120.2345,40.345"
	// JLM, 2/26/99 extended to handle
	// <number><whiteSpace><number>
	long	j, k, pairIndex;
	char	num [64];
	OSErr	err = 0;
	char delimiterChar = ',';
	Boolean scientificNotation = false;
	
	j = 0;	// index into supplied string //
	
	for (pairIndex = 1; pairIndex <= 2 && !err; ++pairIndex)
	{
		// scan u, then v //
		Boolean keepGoing = true;
		for (k = 0 ; keepGoing; j++)
		{	   			
			switch(startChar[j])
			{
				case ',': // delimiter
				case 0: // end of string
					keepGoing = false;
					break;
				case '.':
					num[k++] = startChar[j];
					break;
				case '+': // number
					if (!scientificNotation) 
					{
						err=-1; 
						return err;
					}
					// else number
				case '-': // number
				case '0': case '1': case '2': case '3': case '4':
				case '5': case '6': case '7': case '8': case '9':
					num[k++] = startChar[j];
					if(k>=32) // no space or comma found to signal end of number
					{
						err = -1;
						return err;
					}
					break;
				case 'e': 
					num[k++] = startChar[j];
					if(k>=32) // no space or comma found to signal end of number
					{
						err = -1;
						return err;
					}
					scientificNotation = true;
					break;
				case ' ':
				case '\t': // white space
					if(k == 0) continue; // ignore leading white space
					while(startChar[j+1] == ' ' || startChar[j+1] == '\t') j++; // move past any additional whitespace chars
					if(startChar[j+1] == ',') j++; // it was <whitespace><comma>, use the comma as the delimiter
					// we have either found a comma or will use the white space as a delimiter
					// in either case we stop this loop
					keepGoing = false;
					break;
				default:
					err = -1;
					return err;
					break;
			}
		}
		
		if(err) break; // so we break out of the main loop, shouldn't happen
		
		num[k++] = 0;									// terminate the number-string //
		
		if (pairIndex == 1)
		{
			if (!scientificNotation) VelocityPtr -> u = atof(num);
			
			if (startChar[j] == ',')					// increment j past the comma to next coordinate //
			{
				++j;
				delimiterChar = ','; // JLM reset the delimiter char
			}
		}
		else
		{
			if (!scientificNotation) VelocityPtr -> v = atof(num);
			*scanLength = j; // amount of input string that was read
		}
	}
	///////////////
	
	if (scientificNotation) return -2;
	return err;

}*/
