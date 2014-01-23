/*
 *  GridWndMover.cpp
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "GridWndMover.h"
#include "CROSS.H"

GridWndMover::GridWndMover(TMap *owner,char* name) : TWindMover(owner, name)
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


void GridWndMover::Dispose()
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



bool IsGridWindFile (vector<string> &linesInFile, short *selectedUnitsOut)
{
	long lineIdx = 0;
	string currentLine;
	
	short selectedUnits = kUndefined;
	string value1S, value2S;
	char errmsg[256];
	// First line, must start with '[GRIDCURTIME] <units>'
	// <units> == the designation of units for the file.
	currentLine = trim(linesInFile[lineIdx++]);
	
	istringstream lineStream(currentLine);
	lineStream >> value1S >> value2S;
	sprintf(errmsg,"values = %s,%s\n",value1S,value2S);
	printNote(errmsg);
	if (lineStream.fail())
		return false;
	
	if (value1S != "[GRIDWIND]" && value1S != "[GRIDWINDTIME]")	
		return false;
	
	selectedUnits = StrToSpeedUnits((char *)value2S.c_str());
	if (selectedUnits == kUndefined)
		return false;
	
	*selectedUnitsOut = selectedUnits;
	
	return true;
}


Boolean IsGridWindFile(char *path,short *selectedUnitsP)
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
}

long GridWndMover::GetListLength()
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

ListItem GridWndMover::GetNthListItem(long n, short indent, short *style, char *text)
{
	ListItem item = { dynamic_cast<GridWndMover *>(this), n, indent, 0 };
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
					sprintf(text, "Start Time: %.2f hours",((double)fUncertainStartTime)/3600.);
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

Boolean GridWndMover::ListClick(ListItem item, Boolean inBullet, Boolean doubleClick)
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
				WindSettingsDialog(dynamic_cast<GridWndMover *>(this), this -> moverMap,false,mapWindow,false);
				break;
			default:
				break;
		}
	}
	
	// do other click operations...
	
	return FALSE;
}

Boolean GridWndMover::FunctionEnabled(ListItem item, short buttonID)
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

OSErr GridWndMover::SettingsItem(ListItem item)
{
	//return GridWindSettingsDialog(this, this -> moverMap,false,mapWindow);
	// JLM we want this to behave like a double click
	Boolean inBullet = false;
	Boolean doubleClick = true;
	Boolean b = ListClick(item,inBullet,doubleClick);
	return 0;
}

OSErr GridWndMover::DeleteItem(ListItem item)
{
	if (item.index == I_GRIDWINDNAME)
		return moverMap -> DropMover(dynamic_cast<GridWndMover *>(this));
	
	return 0;
}

OSErr GridWndMover::CheckAndPassOnMessage(TModelMessage *message)
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


OSErr GridWndMover::CheckAndScanFile(char *errmsg, const Seconds& model_time)
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
				if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[fileNum-1].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
				DisposeLoadedData(&fEndData);
				strcpy(fPathName,(*fInputFilesHdl)[fileNum-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[fileNum].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
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
				if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
				err = ScanFileForTimes((*fInputFilesHdl)[i-1].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
				DisposeLoadedData(&fEndData);
				strcpy(fPathName,(*fInputFilesHdl)[i-1].pathName);
				if (err = this -> ReadTimeData(GetNumTimesInFile()-1,&fStartData.dataHdl,errmsg)) return err;	
			}
			fStartData.timeIndex = UNASSIGNEDINDEX;
			if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
			err = ScanFileForTimes((*fInputFilesHdl)[i].pathName,&fTimeDataHdl,false);	// AH 07/17/2012
			if (err) return err;
			strcpy(fPathName,(*fInputFilesHdl)[i].pathName);
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

Boolean GridWndMover::CheckInterval(long &timeDataInterval, const Seconds& model_time)
{
	Seconds time =  model_time;	// AH 07/17/2012
	long i,numTimes;
	
	numTimes = this -> GetNumTimesInFile(); 
	if (numTimes==0) {timeDataInterval = 0; return false;}	// really something is wrong, no data exists
	
	// check for constant wind
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

void GridWndMover::DisposeLoadedData(LoadedData *dataPtr)
{
	if(dataPtr -> dataHdl) DisposeHandle((Handle) dataPtr -> dataHdl);
	ClearLoadedData(dataPtr);
}

void GridWndMover::ClearLoadedData(LoadedData *dataPtr)
{
	dataPtr -> dataHdl = 0;
	dataPtr -> timeIndex = UNASSIGNEDINDEX;
}

long GridWndMover::GetNumTimesInFile()
{
	long numTimes;
	
	numTimes = _GetHandleSize((Handle)fTimeDataHdl)/sizeof(**fTimeDataHdl);
	return numTimes;     
}

long GridWndMover::GetNumFiles()
{
	long numFiles = 0;
	
	if (fInputFilesHdl) numFiles = _GetHandleSize((Handle)fInputFilesHdl)/sizeof(**fInputFilesHdl);
	return numFiles;     
}

OSErr GridWndMover::SetInterval(char *errmsg, const Seconds& model_time)
{
	long timeDataInterval;
	Boolean intervalLoaded = this -> CheckInterval(timeDataInterval, model_time);	// AH 07/17/2012
	
	long indexOfStart = timeDataInterval-1;
	long indexOfEnd = timeDataInterval;
	long numTimesInFile = this -> GetNumTimesInFile();
	OSErr err = 0;
	
	strcpy(errmsg,"");
	
	if(intervalLoaded) 
		return 0;
	
	// check for constant wind 
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
	/*else if(timeDataInterval == numTimesInFile) 
	{	// past the last information in the file
		err = -1;
		strcpy(errmsg,"Time outside of interval being modeled");
		goto done;
	}*/
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
		if(!errmsg[0])strcpy(errmsg,"Error in GridWndMover::SetInterval()");
		DisposeLoadedData(&fStartData);
		DisposeLoadedData(&fEndData);
	}
	
	return err;
}


#define GridWndMoverREADWRITEVERSION 1 //JLM	7/10/01

OSErr GridWndMover::Write (BFPB *bfpb)
{
	char c;
	long i, version = GridWndMoverREADWRITEVERSION; //JLM
	ClassID id = GetClassID ();
	VelocityRec velocity;
	long amtTimeData = GetNumTimesInFile();
	long numPoints, numFiles;
	float val;
	PtCurTimeData timeData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TWindMover::Write (bfpb)) return err;
	
	StartReadWriteSequence("GridWndMover::Write()");
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
		TechError("GridWndMover::Write(char* path)", " ", 0); 
	
	return err;
}

OSErr GridWndMover::Read(BFPB *bfpb)
{
	char c;
	long i, version, amtTimeData, numPoint, numFiles;
	ClassID id;
	float val;
	PtCurTimeData timeData;
	PtCurFileInfo fileInfo;
	OSErr err = 0;
	
	if (err = TWindMover::Read(bfpb)) return err;
	
	StartReadWriteSequence("GridWndMover::Read()");
	if (err = ReadMacValue(bfpb,&id)) return err;
	if (id != GetClassID ()) { TechError("GridWndMover::Read()", "id != TYPE_GRIDWNDMOVER", 0); return -1; }
	if (err = ReadMacValue(bfpb,&version)) return err;
	if (version > GridWndMoverREADWRITEVERSION) { printSaveFileVersionError(); return -1; }
	////
	// read the type of grid used for the GridWind mover (should always be rectgrid...)
	if (err = ReadMacValue(bfpb,&id)) return err;
	switch(id)
	{
	 	case TYPE_RECTGRIDVEL: fGrid = new TRectGridVel;break;
			//case TYPE_TRIGRIDVEL: fGrid = new TTriGridVel;break;
			//case TYPE_TRIGRIDVEL3D: fGrid = new TTriGridVel3D;break;
		default: printError("Unrecognized Grid type in GridWndMover::Read()."); return -1;
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
	{TechError("GridWndMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
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
		{TechError("GridWndMover::Read()", "_NewHandle()", 0); err = memFullErr; goto done;}
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
		TechError("GridWndMover::Read(char* path)", " ", 0); 
		if(fTimeDataHdl) {DisposeHandle((Handle)fTimeDataHdl); fTimeDataHdl=0;}
		if(fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl); fInputFilesHdl=0;}
	}
	return err;
}

OSErr GridWndMover::ReadHeaderLines(char *path, WorldRect *bounds)
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

OSErr GridWndMover::TextRead(char *path) 
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
		TechError("Error in GridWndMover::TextRead()","new TRectGridVel" ,err);
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
		printError("An error occurred in GridWndMover::TextRead"); 
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

OSErr GridWndMover::ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile)
{
	long i,numScanned;
	DateTimeRec time;
	Seconds timeSeconds;
	OSErr err = 0;
	char s[1024], classicPath[256];
	
	PtCurFileInfoH inputFilesHdl = (PtCurFileInfoH)_NewHandle(sizeof(PtCurFileInfo)*numFiles);
	if(!inputFilesHdl) {TechError("GridWndMover::ReadInputFileNames()", "_NewHandle()", 0); err = memFullErr; goto done;}
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
		{ err = -1; TechError("GridWndMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
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
		 { err = -1; TechError("GridWndMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
		 }
		 */
		numScanned=sscanf(s+strlen("[ENDTIME]"), "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("GridWndMover::ReadInputFileNames()", "sscanf() == 5", 0); goto done; }
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

OSErr GridWndMover::ScanFileForTimes(char *path, PtCurTimeDataHdl *timeDataH,Boolean setStartTime)
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
	if(!timeDataHdl) {TechError("GridWndMover::ScanFileForTimes()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	// think in terms of 100K blocks, allocate 101K, read 101K, scan 100K
	
#define kGridCurFileBufferSize  100000 // code goes here, increase to 100K or more
#define kGridCurFileBufferExtraCharSize  256
	
	err = MyGetFileSize(0,0,path,&fileLength);
	if(err) goto done;
	
	offset = 0;
	lengthRemainingToScan = fileLength - 5;
	
	// loop until whole file is read 
	
	h = (CHARH)_NewHandle(2* kGridCurFileBufferSize+1);
	if(!h){TechError("GridWndMover::TextRead()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
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
				{ err = -1; TechError("GridWndMover::TextRead()", "sscanf() == 5", 0); goto done; }
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
				if (_MemError()) { TechError("GridWndMover::TextRead()", "_SetHandleSize()", 0); goto done; }
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

OSErr GridWndMover::ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) 
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
	if(!h){TechError("GridWndMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
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
	if(!velH){TechError("GridWndMover::ReadTimeData()", "_NewHandle()", 0); err = memFullErr; goto done;}
	
	strToMatch = "[TIME]";
	len = strlen(strToMatch);
	NthLineInTextOptimized (sectionOfFile, line = 0, s, 256);
	if(!strncmp(s,strToMatch,len)) 
	{
		numScanned=sscanf(s+len, "%hd %hd %hd %hd %hd",
						  &time.day, &time.month, &time.year,
						  &time.hour, &time.minute) ;
		if (numScanned!= 5)
		{ err = -1; TechError("GridWndMover::ReadTimeData()", "sscanf() == 5", 0); goto done; }
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
			strcpy(errmsg,"An error occurred in GridWndMover::ReadTimeData");
		//printError(errmsg); // This alert causes a freeze up...
		// We don't want to put up an error message here because it can lead to an infinite loop of messages.
		if(velH) {DisposeHandle((Handle)velH); velH = 0;}
	}
	return err;
	
}

Boolean GridWndMover::DrawingDependsOnTime(void)
{
	Boolean depends = bShowArrows;
	// if this is a constant wind, we can say "no"
	if(this->GetNumTimesInFile()==1) depends = false;
	return depends;
}

void GridWndMover::Draw(Rect r, WorldRect view) 
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
		err = this -> SetInterval(errmsg, model->GetModelTime());	// AH 07/17/2012
		
		if(err && !bShowGrid) return;	// want to show grid even if there's no wind data
		
		loaded = this -> CheckInterval(timeDataInterval, model->GetModelTime());	// AH 07/17/2012
		
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
				index = dynamic_cast<GridWndMover *>(this)->GetVelocityIndex(wp);	
				
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
/*static GridWndMover *sharedGWMover;
 
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
 OSErr GridWindSettingsDialog(GridWndMover *mover, TMap *owner,Boolean bAddMover,DialogPtr parentWindow)
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

